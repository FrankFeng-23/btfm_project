#!/usr/bin/env python3
# Robust Sentinel-1 SAR Data Downloader - Seasonal version with separate VV/VH output files

import os
import sys
import numpy as np
import pystac
import pystac_client
import planetary_computer
import logging
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
import concurrent.futures
import datetime
import tqdm
import argparse
import stackstac
import dask.array as da
from distributed import Client, LocalCluster, TimeoutError, CancelledError
import distributed  # Import for distributed.comm.core.CommClosedError
import shapely.geometry
from collections import defaultdict
import tempfile
import uuid
import shutil
import json
import time
import functools
import traceback
import warnings
from contextlib import contextmanager
import socket
import psutil
from pathlib import Path
import threading

# Try to import dask_gateway - it might not be available in all environments
try:
    import dask_gateway
    DASK_GATEWAY_AVAILABLE = True
except ImportError:
    DASK_GATEWAY_AVAILABLE = False

# Suppress specific warnings to reduce log noise
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("sentinel1_download.log")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download Sentinel-1 data matching input TIFF geography for a specific season with separate VV/VH files')
parser.add_argument('--input_tiff', type=str, required=True, help='Input TIFF file to extract geography from')
parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
parser.add_argument('--workers', type=int, default=8, help='Number of parallel download workers (default: 8)')
parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory for intermediate files (default: system temp)')
parser.add_argument('--output', type=str, default='sentinel1_output', help='Output directory (default: sentinel1_output)')
parser.add_argument('--chunksize', type=int, default=1024, help='Chunk size for processing (default: 1024)')
parser.add_argument('--use_dask_gateway', action='store_true', help='Use Dask Gateway for distributed processing if available')
parser.add_argument('--dask_workers', type=int, default=32, help='Number of Dask workers to use (default: 32)')
parser.add_argument('--worker_memory', type=int, default=32, help='Memory per Dask worker in GB (default: 32)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files instead of skipping them')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for operations (default: 3)')
parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for individual operations (default: 600 seconds)')
parser.add_argument('--orbit_state', type=str, default='both', choices=['ascending', 'descending', 'both'], 
                   help='Orbit state to process: ascending, descending, or both (default: both)')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args = parser.parse_args()

# Set debug logging if requested
if args.debug:
    root_logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Create output directory
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

# Create checkpoint directory
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Create or use specified temporary directory
if args.temp_dir:
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
else:
    # Create a temp dir that will persist across the run
    temp_dir = tempfile.mkdtemp(prefix="sentinel1_")
logger.info(f"Using temporary directory: {temp_dir}")

# Extract season name for logs and checkpoints
start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
season_name = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

# Update log file name to include season
log_file_path = os.path.join(output_dir, f"sentinel1_{season_name}.log")
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        handler.close()
        root_logger.removeHandler(handler)

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Define checkpoint file paths for each polarization
vv_ascending_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vv_ascending_{season_name}_checkpoint.json")
vh_ascending_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vh_ascending_{season_name}_checkpoint.json")
vv_descending_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vv_descending_{season_name}_checkpoint.json")
vh_descending_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vh_descending_{season_name}_checkpoint.json")

# Define Sentinel-1 resolution (meters)
SAR_RESOLUTION = 10.0  # Sentinel-1 resolution is 10 meters

# Custom exceptions
class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

class DaskClusterError(Exception):
    """Exception raised when there's a problem with the Dask cluster that requires restart"""
    pass

def gracefully_shutdown_dask(client, cluster=None):
    """Gracefully shutdown Dask Client and Cluster"""
    logger.info("Shutting down Dask resources...")
    
    # Completely silence distributed logging during shutdown
    distributed_logger = logging.getLogger('distributed')
    original_level = distributed_logger.level
    distributed_logger.setLevel(logging.CRITICAL)
    
    try:
        # Cancel any remaining tasks
        client.cancel(list(client.futures))
    except:
        pass
    
    # Add a small delay to allow comms to settle
    time.sleep(1)
    
    # Close client first
    try:
        # Tell workers to stop heartbeating before shutting down
        if hasattr(client, 'scheduler_info'):
            try:
                workers = list(client.scheduler_info().get('workers', {}).keys())
                if workers:
                    logger.debug(f"Telling {len(workers)} workers to stop heartbeating")
                    client.run(lambda: setattr(distributed.worker.thread_state, 'stop', True))
            except:
                pass
        
        # Force close client - get address to avoid log messages
        try:
            scheduler_addr = client.scheduler.address
        except:
            scheduler_addr = None
        
        # Ensure all comms are closed
        client.close(timeout=1)
        logger.info("Closed Dask client")
        
        # Cleanup remaining connections directly if possible
        if scheduler_addr:
            try:
                from distributed.comm.addressing import parse_address
                from distributed.comm.core import connect
                from distributed.comm.utils import to_frames
                
                scheme, loc = parse_address(scheduler_addr)
                if scheme == 'tcp':
                    comm = connect(scheduler_addr)
                    comm.write({'op': 'close-stream'})
                    comm.close()
            except:
                pass
    except (distributed.comm.core.CommClosedError, 
            distributed.client.TimeoutError, 
            ConnectionResetError) as e:
        # These errors during shutdown are expected and can be ignored
        logger.debug(f"Dask client close: {type(e).__name__} (expected during shutdown)")
    except Exception as e:
        logger.debug(f"Error closing Dask client: {e}")
        
    # Then close cluster if we have access to it
    if cluster and client.cluster is not cluster:  # Avoid double-closing
        try:
            # Add a small delay to allow client to finish closing first
            time.sleep(1)
            cluster.close()
            logger.info("Closed Dask cluster")
        except (distributed.comm.core.CommClosedError, 
                distributed.client.TimeoutError, 
                ConnectionResetError) as e:
            # These errors during shutdown are expected and can be ignored
            logger.debug(f"Dask cluster close: {type(e).__name__} (expected during shutdown)")
        except Exception as e:
            logger.debug(f"Error closing Dask cluster: {e}")
   
    # One last attempt to clean up any leaked workers
    try:
        import signal
        import psutil
        
        # Try to kill any remaining dask-worker processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if any('dask-worker' in cmd for cmd in cmdline if cmd):
                    proc.send_signal(signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to dask-worker process {proc.info['pid']}")
            except:
                pass
    except:
        pass
        
    # Wait a moment for workers to finish shutting down
    time.sleep(2)
    
    # Restore original logging level
    distributed_logger.setLevel(original_level)

# Define a thread-safe timeout context manager
@contextmanager
def time_limit_context(seconds):
    """A context manager for timing out operations"""
    def timeout_handler():
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    if seconds is None or seconds <= 0:
        yield
        return
        
    timer = threading.Timer(seconds, timeout_handler)
    timer.daemon = True
    try:
        timer.start()
        yield
    finally:
        timer.cancel()

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=None, initial_backoff=1, max_backoff=60, backoff_factor=2, 
                      exceptions=(Exception,), timeout_seconds=None):
    """
    Retry decorator with exponential backoff and timeout
    """
    if max_retries is None:
        max_retries = args.max_retries
        
    if timeout_seconds is None:
        timeout_seconds = args.timeout
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    # Using our simpler timeout context manager
                    if timeout_seconds and timeout_seconds > 0:
                        with time_limit_context(timeout_seconds):
                            return func(*args, **kwargs)
                    else:
                        # No timeout, just run the function directly
                        return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Failed after {max_retries} retries: {func.__name__}, Error: {e}")
                        raise
                    
                    # Calculate backoff time
                    wait_time = min(backoff, max_backoff)
                    backoff *= backoff_factor
                    
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Function to monitor system resources
def get_system_status():
    """Get system status information."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    disk = psutil.disk_usage('/')
    
    return {
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024 ** 3),
        'cpu_percent': cpu_percent,
        'disk_percent': disk.percent,
        'disk_free_gb': disk.free / (1024 ** 3)
    }

def log_system_status():
    """Log system status."""
    status = get_system_status()
    logger.info(f"System status: "
               f"RAM {status['memory_percent']:.1f}% used ({status['memory_available_gb']:.1f} GB available), "
               f"CPU {status['cpu_percent']:.1f}% used, "
               f"Disk {status['disk_percent']:.1f}% used ({status['disk_free_gb']:.1f} GB free)")

def setup_dask_client():
    """
    Set up the appropriate Dask client based on availability and arguments
    Returns both client and cluster (to allow proper shutdown)
    """
    cluster = None
    client = None
    
    if args.use_dask_gateway and DASK_GATEWAY_AVAILABLE:
        logger.info("Setting up Dask Gateway cluster...")
        try:
            gateway = dask_gateway.Gateway()
            cluster_options = gateway.cluster_options()
            
            # Configure cluster options
            cluster_options["worker_memory"] = f"{args.worker_memory}GB"
            
            # Create the cluster
            cluster = gateway.new_cluster(cluster_options)
            client = cluster.get_client()
            
            # Scale the cluster based on the specified number of workers
            if args.dask_workers > 0:
                logger.info(f"Scaling Dask Gateway cluster to {args.dask_workers} workers...")
                cluster.scale(args.dask_workers)
                
                # Wait for workers to be ready
                logger.info("Waiting for workers to become available...")
                client.wait_for_workers(n_workers=min(args.dask_workers, 4), timeout=300)
                logger.info(f"Cluster scaled to {len(client.scheduler_info()['workers'])} workers")
            else:
                # Use adaptive scaling
                logger.info(f"Setting Dask Gateway cluster to adapt between 2 and {max(6, args.workers)} workers...")
                cluster.adapt(minimum=2, maximum=max(6, args.workers))
            
            logger.info(f"Dask Gateway dashboard available at: {cluster.dashboard_link}")
            return client, cluster
            
        except Exception as e:
            logger.warning(f"Failed to set up Dask Gateway cluster: {e}")
            logger.warning("Falling back to local Dask Client...")
    
    # If we get here, either Dask Gateway is not available or we don't want to use it
    # Fall back to local Dask client
    try:
        logger.info(f"Setting up local Dask cluster with {args.dask_workers} workers...")
        # Use the specified number of dask workers, but cap at CPU count for safety
        n_workers = min(args.dask_workers, max(1, os.cpu_count()))
        
        # Calculate memory per worker (leave some for system)
        memory_per_worker = f"{args.worker_memory}GB"
        
        # Create a local cluster with specified parameters
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # Better stability with 1 thread per worker
            memory_limit=memory_per_worker,
            processes=True,
            silence_logs=logging.WARNING,
            dashboard_address=':8787'
        )
        
        client = Client(cluster)
        
        # Safely wait for workers
        try:
            client.wait_for_workers(n_workers=n_workers, timeout=60)
            logger.info(f"Local Dask client created with {len(client.scheduler_info()['workers'])} workers")
        except Exception as e:
            logger.warning(f"Failed to wait for all workers, but continuing: {e}")
        
        logger.info(f"Memory per worker: {memory_per_worker}")
        try:
            logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        except:
            logger.info("Dask dashboard link not available")
        
        return client, cluster
        
    except Exception as e:
        logger.warning(f"Failed to set up local Dask client with {args.dask_workers} workers: {e}")
        logger.warning("Trying with default settings...")
        
        # Final fallback - use basic configuration
        try:
            client = Client(silence_logs=logging.WARNING)
            logger.info(f"Default Dask client created")
            try:
                logger.info(f"Dask dashboard available at: {client.dashboard_link}")
            except:
                logger.info("Dask dashboard link not available")
            return client, client.cluster
        except Exception as e2:
            logger.error(f"Failed to create any Dask client: {e2}")
            raise e2

def amplitude_to_db_numpy(amp, mask=None):
    """
    Convert amplitude to scaled dB values with better handling of invalid values
    NumPy version for processing smaller chunks directly
    
    Args:
        amp: Input amplitude array (float32)
        mask: Optional mask (0 = invalid, >0 = valid)
        
    Returns:
        Scaled dB values as int16:
        dB = 20 * log10(amp)
        shifted = dB + 50
        scaled = shifted * 200
        clipped = np.clip(scaled, 0, 30000)
    """
    # Make a copy to avoid modifying the input
    output = np.zeros_like(amp, dtype=np.int16)
    
    # Safely handle possible invalid values (explicitly filter first to avoid warnings)
    with np.errstate(invalid='ignore', divide='ignore'):
        # Ensure amp is finite (not NaN or inf) before comparison
        amp_finite = np.isfinite(amp)
        # Create validity mask for valid amplitude values (> 0)
        valid_mask = amp_finite & (amp > 0)
    
    # Process only valid pixels
    if np.any(valid_mask):
        # Calculate dB for valid pixels only
        with np.errstate(invalid='ignore', divide='ignore'):
            db = 20.0 * np.log10(amp[valid_mask])
            db_shift = db + 50.0
            scaled = db_shift * 200.0
            clipped = np.clip(scaled, 0, 32767)  # Clip to int16 range
        
        # Assign to output array
        output[valid_mask] = clipped.astype(np.int16)
    
    # Apply external mask if provided
    if mask is not None:
        output = np.where(mask > 0, output, 0)
        
    return output

def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """
    Validate that a TIFF file has the expected properties
    Returns True if valid, False if not
    """
    try:
        with rasterio.open(file_path) as src:
            # Check basic properties
            if src.shape != expected_shape:
                logger.warning(f"Validation failed: Shape mismatch in {file_path}. Expected {expected_shape}, got {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logger.warning(f"Validation failed: CRS mismatch in {file_path}. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check transform
            if not np.allclose(np.array(src.transform)[:6], np.array(expected_transform)[:6], rtol=1e-05, atol=1e-08):
                logger.warning(f"Validation failed: Transform mismatch in {file_path}. Expected {expected_transform}, got {src.transform}")
                return False
            
            # Check for data presence
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logger.warning(f"Validation failed: Band contains all zeros in {file_path}")
                return False
            
            # Check file size (should be reasonable for the given dimensions)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            expected_size_mb = (src.width * src.height * src.count * 2) / (1024 * 1024)  # int16 = 2 bytes
            if file_size_mb < expected_size_mb * 0.05:  # Allow for compression, but not too small
                logger.warning(f"Validation failed: File size too small in {file_path}. Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
                return False
            
            logger.debug(f"TIFF validation passed for {file_path}: shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logger.error(f"Error validating TIFF {file_path}: {e}")
        return False

def process_item(client, item, src_crs, src_transform, src_shape, src_bounds, src_mask, temp_dir=None):
    """
    Process a single item and save as separate TIFF files for VV and VH polarizations
    Returns a tuple of (vv_path, vh_path) with the paths to the created files, or (None, None) if processing failed
    """
    orbit_state = item.properties.get("sat:orbit_state")
    date_str = item.properties.get("datetime").split("T")[0]
    
    # Generate unique filenames for each polarization
    if temp_dir:
        # For temporary files in mosaicking stage, use UUID to avoid conflicts
        vv_temp_id = uuid.uuid4().hex[:8]
        vh_temp_id = uuid.uuid4().hex[:8]
        vv_output_path = os.path.join(temp_dir, f"{date_str}_vv_{orbit_state}_{vv_temp_id}.tiff")
        vh_output_path = os.path.join(temp_dir, f"{date_str}_vh_{orbit_state}_{vh_temp_id}.tiff")
    else:
        # For final output files, use standard naming convention
        vv_output_path = os.path.join(output_dir, f"{date_str}_vv_{orbit_state}.tiff")
        vh_output_path = os.path.join(output_dir, f"{date_str}_vh_{orbit_state}.tiff")
    
    try:
        logger.info(f"Processing item for {date_str}_{orbit_state}...")
        
        # Wrap STAC and data access operations in try-except to handle GDAL errors
        try:
            # Stack the data using stackstac, explicitly setting resolution to 10 meters
            ds = stackstac.stack(
                [item], 
                bounds_latlon=src_bounds,
                epsg=int(src_crs.to_epsg()),
                resolution=SAR_RESOLUTION,  # Fixed 10m resolution for Sentinel-1
                chunksize=args.chunksize
            )
            
            # Check if bands exist
            if 'vv' not in ds.band.values or 'vh' not in ds.band.values:
                logger.warning(f"Data for {date_str}_{orbit_state} is missing required bands. Skipping.")
                return None, None
        except Exception as e:
            logger.error(f"Error stacking data for {date_str}_{orbit_state}: {e}")
            raise
            
        # Extract VV and VH data - wrapped in try/except
        try:
            logger.debug(f"Computing VV data for {date_str}_{orbit_state}...")
            # Using synchronous compute to avoid potential hanging futures
            vv_data = client.compute(ds.sel(band="vv")[0], sync=True)
            
            logger.debug(f"Computing VH data for {date_str}_{orbit_state}...")
            vh_data = client.compute(ds.sel(band="vh")[0], sync=True)
        except (TimeoutError, CancelledError, distributed.comm.core.CommClosedError) as e:
            logger.error(f"Dask communication error for {date_str}_{orbit_state}: {e}")
            # Signal that we need to restart the Dask client
            raise DaskClusterError(f"Communication error in Dask cluster: {e}")
        except Exception as e:
            logger.error(f"Error computing data for {date_str}_{orbit_state}: {e}")
            # Bypass the Dask distributed computing for problematic cases
            try:
                logger.info(f"Attempting to process data locally for {date_str}_{orbit_state}...")
                # Extract data to numpy arrays directly
                vv_selection = ds.sel(band="vv")[0]
                vh_selection = ds.sel(band="vh")[0]
                
                # Compute the selections locally with a timeout
                with time_limit_context(300):  # 5-minute timeout
                    vv_data = vv_selection.compute()
                    vh_data = vh_selection.compute()
            except Exception as e2:
                logger.error(f"Local processing also failed for {date_str}_{orbit_state}: {e2}")
                raise
            
        # Check for valid data with robust error handling
        try:
            # Check for completely empty datasets
            vv_valid = not np.isnan(vv_data.values).all()
            vh_valid = not np.isnan(vh_data.values).all()
            
            if not (vv_valid and vh_valid):
                logger.warning(f"Data for {date_str}_{orbit_state} contains all NaN values. Skipping.")
                return None, None
                
            # Get coordinate information
            x_coords = vv_data.x.values
            y_coords = vv_data.y.values
            
            # Verify we have enough coordinates to work with
            if len(x_coords) < 2 or len(y_coords) < 2:
                logger.warning(f"Insufficient coordinate data for {date_str}_{orbit_state}. Skipping.")
                return None, None
                
            # Log data shapes for debugging
            logger.debug(f"SAR data shape: vv={vv_data.shape}, vh={vh_data.shape}")
            logger.debug(f"Target shape: {src_shape}")
            
        except Exception as e:
            logger.error(f"Error validating data for {date_str}_{orbit_state}: {e}")
            raise
        
        # Create output arrays with the EXACT dimensions of the source TIFF
        vv_final = np.zeros(src_shape, dtype=np.int16)
        vh_final = np.zeros(src_shape, dtype=np.int16)
        
        # Create temporary arrays for reprojection
        vv_reproject = np.zeros(src_shape, dtype=np.float32)
        vh_reproject = np.zeros(src_shape, dtype=np.float32)
        
        # Get the CRS and transform from the dataset
        src_data_crs = CRS.from_string(ds.attrs.get('crs', 'EPSG:4326'))
        
        # Calculate the transform from the coordinate arrays
        x_res = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
        y_res = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)
        src_data_transform = rasterio.transform.from_origin(
            x_coords[0] - x_res/2,
            y_coords[0] - y_res/2,
            x_res,
            -y_res
        )
        
        # Reproject the data with error handling
        try:
            logger.debug(f"Reprojecting data to match input TIFF for {date_str}_{orbit_state}...")
            
            # First try with bilinear resampling for better quality
            reproject(
                source=vv_data.values,
                destination=vv_reproject,
                src_transform=src_data_transform,
                src_crs=src_data_crs,
                dst_transform=src_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear
            )
            
            reproject(
                source=vh_data.values,
                destination=vh_reproject,
                src_transform=src_data_transform,
                src_crs=src_data_crs,
                dst_transform=src_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear
            )
            
            # Log min/max for debugging
            logger.debug(f"After reprojection: VV min={np.nanmin(vv_reproject)}, max={np.nanmax(vv_reproject)}")
            logger.debug(f"After reprojection: VH min={np.nanmin(vh_reproject)}, max={np.nanmax(vh_reproject)}")
            
        except Exception as e:
            logger.error(f"Error in bilinear reprojection for {date_str}_{orbit_state}: {e}")
            # Try with nearest neighbor if bilinear fails
            try:
                logger.info(f"Retrying reprojection with nearest neighbor for {date_str}_{orbit_state}...")
                reproject(
                    source=vv_data.values,
                    destination=vv_reproject,
                    src_transform=src_data_transform,
                    src_crs=src_data_crs,
                    dst_transform=src_transform,
                    dst_crs=src_crs,
                    resampling=Resampling.nearest
                )
                
                reproject(
                    source=vh_data.values,
                    destination=vh_reproject,
                    src_transform=src_data_transform,
                    src_crs=src_data_crs,
                    dst_transform=src_transform,
                    dst_crs=src_crs,
                    resampling=Resampling.nearest
                )
            except Exception as e2:
                logger.error(f"Nearest neighbor reprojection also failed for {date_str}_{orbit_state}: {e2}")
                raise
        
        # Process data: convert amplitude to dB and apply mask
        logger.debug(f"Converting to dB and applying mask for {date_str}_{orbit_state}...")
        
        # Use with np.errstate to suppress warnings
        with np.errstate(invalid='ignore', divide='ignore'):
            vv_final = amplitude_to_db_numpy(vv_reproject, mask=src_mask)
            vh_final = amplitude_to_db_numpy(vh_reproject, mask=src_mask)
        
        # Log the percentage of non-zero pixels for debugging
        vv_nonzero = np.count_nonzero(vv_final) / vv_final.size
        vh_nonzero = np.count_nonzero(vh_final) / vh_final.size
        logger.debug(f"Non-zero pixels: VV={vv_nonzero:.1%}, VH={vh_nonzero:.1%}")
        
        # Write VV polarization to TIFF
        try:
            logger.debug(f"Saving VV polarization for {date_str}_{orbit_state} to {vv_output_path}...")
            with rasterio.open(
                vv_output_path,
                'w',
                driver='GTiff',
                height=src_shape[0],
                width=src_shape[1],
                count=1,  # Single band for VV
                dtype='int16',
                crs=src_crs,
                transform=src_transform,
                compress='lzw',  # Adding compression for smaller files
                tiled=True,      # Make file tiled for better read performance
                blockxsize=256,  # Tile size
                blockysize=256,
            ) as dst:
                # Write VV band
                dst.write(vv_final, 1)
                
                # Set band description
                dst.set_band_description(1, "VV polarization, amplitude to dB, +50 offset, scale=200")
                
                # Add metadata
                dst.update_tags(
                    TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    ORBIT_STATE=orbit_state,
                    DATE_ACQUIRED=date_str,
                    POLARIZATION="VV",
                    SATELLITE=item.properties.get("platform", ""),
                    SOURCE_TIFF=os.path.basename(args.input_tiff),
                    DESCRIPTION="Sentinel-1 SAR data (VV). Values are amplitude converted to dB, shifted by +50, scaled by 200, and stored as int16. Areas masked in source TIFF are set to 0."
                )
        except Exception as e:
            logger.error(f"Error writing VV TIFF for {date_str}_{orbit_state}: {e}")
            # Try to remove failed file if it exists
            if os.path.exists(vv_output_path):
                try:
                    os.remove(vv_output_path)
                except:
                    pass
            raise
        
        # Write VH polarization to TIFF
        try:
            logger.debug(f"Saving VH polarization for {date_str}_{orbit_state} to {vh_output_path}...")
            with rasterio.open(
                vh_output_path,
                'w',
                driver='GTiff',
                height=src_shape[0],
                width=src_shape[1],
                count=1,  # Single band for VH
                dtype='int16',
                crs=src_crs,
                transform=src_transform,
                compress='lzw',  # Adding compression for smaller files
                tiled=True,      # Make file tiled for better read performance
                blockxsize=256,  # Tile size
                blockysize=256,
            ) as dst:
                # Write VH band
                dst.write(vh_final, 1)
                
                # Set band description
                dst.set_band_description(1, "VH polarization, amplitude to dB, +50 offset, scale=200")
                
                # Add metadata
                dst.update_tags(
                    TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    ORBIT_STATE=orbit_state,
                    DATE_ACQUIRED=date_str,
                    POLARIZATION="VH",
                    SATELLITE=item.properties.get("platform", ""),
                    SOURCE_TIFF=os.path.basename(args.input_tiff),
                    DESCRIPTION="Sentinel-1 SAR data (VH). Values are amplitude converted to dB, shifted by +50, scaled by 200, and stored as int16. Areas masked in source TIFF are set to 0."
                )
        except Exception as e:
            logger.error(f"Error writing VH TIFF for {date_str}_{orbit_state}: {e}")
            # Try to remove failed file if it exists
            if os.path.exists(vh_output_path):
                try:
                    os.remove(vh_output_path)
                except:
                    pass
            # Also remove VV file if VH fails
            if os.path.exists(vv_output_path):
                try:
                    os.remove(vv_output_path)
                except:
                    pass
            raise
        
        # Validate the output files
        vv_valid = validate_tiff(vv_output_path, src_shape, src_crs, src_transform)
        vh_valid = validate_tiff(vh_output_path, src_shape, src_crs, src_transform)
        
        if not vv_valid:
            logger.error(f"VV output TIFF validation failed for {date_str}_{orbit_state}. Removing file.")
            try:
                if os.path.exists(vv_output_path):
                    os.remove(vv_output_path)
                vv_output_path = None
            except:
                pass
            
        if not vh_valid:
            logger.error(f"VH output TIFF validation failed for {date_str}_{orbit_state}. Removing file.")
            try:
                if os.path.exists(vh_output_path):
                    os.remove(vh_output_path)
                vh_output_path = None
            except:
                pass
        
        # Log successful completion and file sizes
        if vv_output_path:
            vv_file_size_mb = os.path.getsize(vv_output_path) / (1024 * 1024)
            logger.info(f"Successfully created VV file: {vv_output_path} ({vv_file_size_mb:.2f} MB)")
        
        if vh_output_path:
            vh_file_size_mb = os.path.getsize(vh_output_path) / (1024 * 1024)
            logger.info(f"Successfully created VH file: {vh_output_path} ({vh_file_size_mb:.2f} MB)")
        
        return vv_output_path, vh_output_path
    
    except DaskClusterError:
        # Propagate DaskClusterError so it can be handled at a higher level
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {date_str}_{orbit_state}: {str(e)}", exc_info=True)
        return None, None

@retry_with_backoff(exceptions=(Exception,), max_retries=2, timeout_seconds=300)
def mosaic_tiffs(tiff_paths, output_path, src_crs, src_transform, src_shape, date_str, orbit_state, polarization):
    """
    Mosaic multiple single-band TIFFs into a single output TIFF
    """
    try:
        # Open all source TIFFs
        src_files = []
        for path in tiff_paths:
            if path and os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logger.warning(f"Failed to open {path} for mosaicking {polarization}: {e}")
        
        if not src_files:
            logger.warning(f"No valid files to mosaic for {date_str}_{polarization}_{orbit_state}")
            return None
        
        # Perform mosaic operation with error handling
        try:
            logger.info(f"Mosaicking {len(src_files)} {polarization} files for {date_str}_{orbit_state}...")
            mosaic_data, out_transform = merge(src_files, nodata=0)
        except Exception as e:
            logger.error(f"Error during merge operation for {date_str}_{polarization}_{orbit_state}: {e}")
            # Close files before returning
            for src in src_files:
                try:
                    src.close()
                except:
                    pass
            raise
        
        # Close all source files
        for src in src_files:
            try:
                src.close()
            except Exception as e:
                logger.warning(f"Error closing source file: {e}")
        
        # Check if mosaic_data has the expected structure
        if mosaic_data.shape[0] < 1:
            logger.error(f"Mosaic data has incorrect structure for {date_str}_{polarization}_{orbit_state}")
            return None
        
        # Write the mosaic to the output file with error handling
        try:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src_shape[0],
                width=src_shape[1],
                count=1,  # Single band
                dtype='int16',
                crs=src_crs,
                transform=src_transform,
                compress='lzw',  # Adding compression for smaller files
                tiled=True,      # Make file tiled for better read performance
                blockxsize=256,  # Tile size
                blockysize=256,
            ) as dst:
                # Write the data
                dst.write(mosaic_data[0], 1)
                
                # Set band description
                dst.set_band_description(1, f"{polarization} polarization, amplitude to dB, +50 offset, scale=200")
                
                # Add metadata
                dst.update_tags(
                    TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    ORBIT_STATE=orbit_state,
                    DATE_ACQUIRED=date_str,
                    POLARIZATION=polarization,
                    MOSAIC_SOURCE_COUNT=len(src_files),
                    SOURCE_TIFF=os.path.basename(args.input_tiff),
                    DESCRIPTION=f"Mosaicked Sentinel-1 SAR data ({polarization}). Values are amplitude converted to dB, shifted by +50, scaled by 200, and stored as int16. Areas masked in source TIFF are set to 0."
                )
        except Exception as e:
            logger.error(f"Error writing mosaic file for {date_str}_{polarization}_{orbit_state}: {e}")
            # Try to remove failed file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
        
        # Validate the output file
        if not validate_tiff(output_path, src_shape, src_crs, src_transform):
            logger.error(f"Mosaic TIFF validation failed for {date_str}_{polarization}_{orbit_state}. Removing file.")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    return None
            except:
                pass
            return None
        
        # Log successful completion and file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Successfully created mosaic {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Unexpected error creating mosaic for {date_str}_{polarization}_{orbit_state}: {str(e)}", exc_info=True)
        return None

def group_items_by_date(items):
    """
    Group items by date and orbit state
    Returns a dictionary: {(date_str, orbit_state): [items]}
    """
    groups = defaultdict(list)
    for item in items:
        date_str = item.properties.get("datetime").split("T")[0]
        orbit_state = item.properties.get("sat:orbit_state")
        groups[(date_str, orbit_state)].append(item)
    return groups

def print_date_statistics(grouped_items):
    """
    Print statistics about the grouped items
    """
    # Count dates with multiple observations
    multiple_obs_dates = {key: items for key, items in grouped_items.items() if len(items) > 1}
    
    # Count by orbit state
    asc_count = sum(1 for key, _ in grouped_items.items() if key[1] == "ascending")
    desc_count = sum(1 for key, _ in grouped_items.items() if key[1] == "descending")
    
    # Count total unique dates
    unique_dates = set(key[0] for key in grouped_items.keys())
    
    # Print statistics
    logger.info(f"Total unique dates: {len(unique_dates)}")
    logger.info(f"Total ascending orbit dates: {asc_count}")
    logger.info(f"Total descending orbit dates: {desc_count}")
    logger.info(f"Dates with multiple observations: {len(multiple_obs_dates)}")
    
    # Print details about dates with multiple observations
    if multiple_obs_dates:
        logger.info("\nDates with multiple observations:")
        for (date_str, orbit_state), items in multiple_obs_dates.items():
            logger.info(f"  {date_str} ({orbit_state}): {len(items)} observations")

def load_checkpoint(checkpoint_file):
    """
    Load checkpoint data from file
    """
    if os.path.exists(checkpoint_file) and args.resume:
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_file}: {e}")
    
    # Return empty checkpoint if file doesn't exist or resume is not enabled
    return {'completed': [], 'failed': []}

def save_checkpoint(checkpoint_file, completed, failed):
    """
    Save checkpoint data to file
    """
    checkpoint_data = {
        'completed': completed,
        'failed': failed,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_file}: {e}")

def process_date_group(client, date_group, src_crs, src_transform, src_shape, bbox, src_mask):
    """
    Process a group of items for the same date and orbit state
    Creates separate files for VV and VH polarizations
    If there are multiple items, mosaic them
    Returns a tuple of (vv_success, vh_success, key, result_message)
    """
    try:
        (date_str, orbit_state), items = date_group
        key = f"{date_str}_{orbit_state}"
        
        # Define output file paths for VV and VH
        vv_final_filename = f"{date_str}_vv_{orbit_state}.tiff"
        vh_final_filename = f"{date_str}_vh_{orbit_state}.tiff"
        vv_final_output_path = os.path.join(output_dir, vv_final_filename)
        vh_final_output_path = os.path.join(output_dir, vh_final_filename)
        
        # Check if output files already exist and handle according to overwrite flag
        vv_exists = os.path.exists(vv_final_output_path)
        vh_exists = os.path.exists(vh_final_output_path)
        
        if vv_exists and vh_exists and not args.overwrite:
            # Validate existing files
            vv_valid = validate_tiff(vv_final_output_path, src_shape, src_crs, src_transform)
            vh_valid = validate_tiff(vh_final_output_path, src_shape, src_crs, src_transform)
            
            if vv_valid and vh_valid:
                logger.info(f"Files for {key} already exist and are valid, skipping.")
                return (True, True, key, f"Skipped {key} (files already exist)")
            else:
                logger.warning(f"Files for {key} exist but at least one is invalid. Reprocessing.")
                # Remove invalid files
                if not vv_valid and vv_exists:
                    try:
                        os.remove(vv_final_output_path)
                    except Exception as e:
                        logger.error(f"Failed to remove invalid VV file for {key}: {e}")
                
                if not vh_valid and vh_exists:
                    try:
                        os.remove(vh_final_output_path)
                    except Exception as e:
                        logger.error(f"Failed to remove invalid VH file for {key}: {e}")
        
        # Create a subdirectory in temp_dir for this specific date-orbit group
        group_temp_dir = os.path.join(temp_dir, f"{date_str}_{orbit_state}")
        try:
            os.makedirs(group_temp_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create temp directory for {key}: {e}")
            # Continue using the parent temp directory if this fails
            group_temp_dir = temp_dir
        
        # If only one item, process directly
        if len(items) == 1:
            logger.info(f"Processing single observation for {key}...")
            
            # Skip processing if files already exist and are valid
            if vv_exists and vh_exists and not args.overwrite:
                vv_valid = validate_tiff(vv_final_output_path, src_shape, src_crs, src_transform)
                vh_valid = validate_tiff(vh_final_output_path, src_shape, src_crs, src_transform)
                
                if vv_valid and vh_valid:
                    return (True, True, key, f"Files for {key} already exist and are valid")
            
            # Process the single item directly to final output paths
            vv_result, vh_result = process_item(client, items[0], src_crs, src_transform, src_shape, bbox, src_mask)
            
            # If successful, check if we need to move files to final location
            if vv_result and vh_result:
                return (True, True, key, f"Successfully processed {key}")
            elif vv_result:
                return (True, False, key, f"Successfully processed VV only for {key}")
            elif vh_result:
                return (False, True, key, f"Successfully processed VH only for {key}")
            else:
                return (False, False, key, f"Failed to process {key}")
        
        # If multiple items, process each one to a temporary file then mosaic
        else:
            logger.info(f"Processing {len(items)} observations for {key}...")
            vv_temp_files = []
            vh_temp_files = []
            
            # Process each item to temporary files
            for i, item in enumerate(items):
                try:
                    vv_result, vh_result = process_item(
                        client, item, src_crs, src_transform, src_shape, bbox, src_mask, group_temp_dir
                    )
                    if vv_result:
                        vv_temp_files.append(vv_result)
                    if vh_result:
                        vh_temp_files.append(vh_result)
                except DaskClusterError:
                    # Propagate DaskClusterError so it triggers a client restart
                    raise
                except Exception as e:
                    logger.error(f"Error processing item {i} for {key}: {e}")
                    # Continue processing other items even if one fails
            
            # Now mosaic the temporary files for each polarization
            vv_success = False
            vh_success = False
            
            # Handle VV files
            if vv_temp_files:
                # If there's only one valid VV file, just rename it
                if len(vv_temp_files) == 1:
                    logger.info(f"Only one valid VV file for {key}, using it directly.")
                    try:
                        shutil.copy2(vv_temp_files[0], vv_final_output_path)
                        os.remove(vv_temp_files[0])
                        vv_success = True
                    except Exception as e:
                        logger.error(f"Error copying single VV file for {key}: {e}")
                
                # Multiple valid VV files need mosaicking
                else:
                    try:
                        vv_mosaic_result = mosaic_tiffs(
                            vv_temp_files, 
                            vv_final_output_path, 
                            src_crs, 
                            src_transform, 
                            src_shape, 
                            date_str, 
                            orbit_state,
                            "VV"
                        )
                        
                        # Clean up temporary VV files
                        for temp_file in vv_temp_files:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception as e:
                                logger.warning(f"Failed to remove temporary VV file {temp_file}: {e}")
                        
                        if vv_mosaic_result:
                            vv_success = True
                        
                    except Exception as e:
                        logger.error(f"Error mosaicking VV files for {key}: {e}")
            
            # Handle VH files
            if vh_temp_files:
                # If there's only one valid VH file, just rename it
                if len(vh_temp_files) == 1:
                    logger.info(f"Only one valid VH file for {key}, using it directly.")
                    try:
                        shutil.copy2(vh_temp_files[0], vh_final_output_path)
                        os.remove(vh_temp_files[0])
                        vh_success = True
                    except Exception as e:
                        logger.error(f"Error copying single VH file for {key}: {e}")
                
                # Multiple valid VH files need mosaicking
                else:
                    try:
                        vh_mosaic_result = mosaic_tiffs(
                            vh_temp_files, 
                            vh_final_output_path, 
                            src_crs, 
                            src_transform, 
                            src_shape, 
                            date_str, 
                            orbit_state,
                            "VH"
                        )
                        
                        # Clean up temporary VH files
                        for temp_file in vh_temp_files:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception as e:
                                logger.warning(f"Failed to remove temporary VH file {temp_file}: {e}")
                        
                        if vh_mosaic_result:
                            vh_success = True
                            
                    except Exception as e:
                        logger.error(f"Error mosaicking VH files for {key}: {e}")
            
            # Return status based on which polarizations were successfully processed
            if vv_success and vh_success:
                return (True, True, key, f"Successfully mosaicked VV and VH files for {key}")
            elif vv_success:
                return (True, False, key, f"Successfully mosaicked VV files only for {key}")
            elif vh_success:
                return (False, True, key, f"Successfully mosaicked VH files only for {key}")
            else:
                return (False, False, key, f"Failed to create mosaics for {key}")
    
    except DaskClusterError:
        # This is a special error that indicates we need to restart the Dask client
        # Propagate it up to be handled by the caller
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error processing group {date_str}_{orbit_state}: {e}", exc_info=True)
        return (False, False, key, f"Error processing {key}: {str(e)}")

def process_orbit_state(orbit_state, items, src_crs, src_transform, src_shape, bbox, src_mask):
    """
    Process items for a specific orbit state (ascending or descending)
    Creates separate files for VV and VH polarizations
    """
    # Filter items by orbit state
    orbit_items = [item for item in items if item.properties.get("sat:orbit_state") == orbit_state]
    if not orbit_items:
        logger.info(f"No {orbit_state} orbit items to process")
        return [], [], [], []
    
    logger.info(f"Processing {len(orbit_items)} {orbit_state} orbit items")
    
    # Group items by date
    grouped_items = group_items_by_date(orbit_items)
    logger.info(f"Grouped into {len(grouped_items)} unique date-orbit combinations")
    
    # Print statistics
    print_date_statistics(grouped_items)
    
    # Load checkpoints for VV and VH separately
    vv_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vv_{orbit_state}_{season_name}_checkpoint.json")
    vh_checkpoint_file = os.path.join(checkpoint_dir, f"sentinel1_vh_{orbit_state}_{season_name}_checkpoint.json")
    
    vv_checkpoint_data = load_checkpoint(vv_checkpoint_file)
    vh_checkpoint_data = load_checkpoint(vh_checkpoint_file)
    
    vv_completed_keys = vv_checkpoint_data.get('completed', [])
    vv_failed_keys = vv_checkpoint_data.get('failed', [])
    vh_completed_keys = vh_checkpoint_data.get('completed', [])
    vh_failed_keys = vh_checkpoint_data.get('failed', [])
    
    logger.info(f"VV checkpoint loaded: {len(vv_completed_keys)} completed, {len(vv_failed_keys)} failed")
    logger.info(f"VH checkpoint loaded: {len(vh_completed_keys)} completed, {len(vh_failed_keys)} failed")
    
    # Filter groups based on checkpoint if resume is enabled
    if args.resume:
        filtered_groups = {}
        for key, items in grouped_items.items():
            date_str, orbit_state = key
            group_key = f"{date_str}_{orbit_state}"
            
            # Check if both VV and VH are completed
            vv_completed = group_key in vv_completed_keys
            vh_completed = group_key in vh_completed_keys
            
            # If both are completed and we're not overwriting, check if files exist and are valid
            if vv_completed and vh_completed and not args.overwrite:
                vv_file_path = os.path.join(output_dir, f"{date_str}_vv_{orbit_state}.tiff")
                vh_file_path = os.path.join(output_dir, f"{date_str}_vh_{orbit_state}.tiff")
                
                vv_file_valid = os.path.exists(vv_file_path) and validate_tiff(vv_file_path, src_shape, src_crs, src_transform)
                vh_file_valid = os.path.exists(vh_file_path) and validate_tiff(vh_file_path, src_shape, src_crs, src_transform)
                
                if vv_file_valid and vh_file_valid:
                    logger.debug(f"Skipping {group_key} (both polarizations completed in previous run)")
                    continue
                else:
                    logger.info(f"{group_key} marked as completed but files are missing or invalid. Re-processing.")
            
            # Include if either VV or VH needs processing
            filtered_groups[key] = items
        
        logger.info(f"After filtering based on checkpoints: {len(filtered_groups)} groups to process")
        grouped_items = filtered_groups
    
    # Check if there's anything to process
    if not grouped_items:
        logger.info(f"No {orbit_state} groups to process after filtering")
        return vv_completed_keys, vv_failed_keys, vh_completed_keys, vh_failed_keys
    
    # Convert to list and sort by date for more predictable processing
    groups_list = sorted(grouped_items.items(), key=lambda x: x[0][0])
    
    # Set up a Dask client for processing
    client, cluster = setup_dask_client()
    
    try:
        # Log system status before starting
        log_system_status()
        
        # Process all date groups
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_group = {}
            for group in groups_list:
                future = executor.submit(
                    process_date_group, client, group, src_crs, src_transform, src_shape, bbox, src_mask
                )
                future_to_group[future] = group
            
            # Process as they complete
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_group), 
                                  total=len(future_to_group), desc=f"Processing {orbit_state} items"):
                group = future_to_group[future]
                (date_str, orbit_state), _ = group
                key = f"{date_str}_{orbit_state}"
                
                try:
                    vv_success, vh_success, result_key, message = future.result()
                    results.append((vv_success, vh_success, result_key, message))
                    
                    # Update VV completed/failed lists
                    if vv_success:
                        if key not in vv_completed_keys:
                            vv_completed_keys.append(key)
                        if key in vv_failed_keys:
                            vv_failed_keys.remove(key)
                    else:
                        if key not in vv_failed_keys:
                            vv_failed_keys.append(key)
                    
                    # Update VH completed/failed lists
                    if vh_success:
                        if key not in vh_completed_keys:
                            vh_completed_keys.append(key)
                        if key in vh_failed_keys:
                            vh_failed_keys.remove(key)
                    else:
                        if key not in vh_failed_keys:
                            vh_failed_keys.append(key)
                    
                    # Periodically save checkpoints
                    if len(results) % 5 == 0:
                        save_checkpoint(vv_checkpoint_file, vv_completed_keys, vv_failed_keys)
                        save_checkpoint(vh_checkpoint_file, vh_completed_keys, vh_failed_keys)
                        
                except DaskClusterError as e:
                    # Handle Dask cluster issues
                    error_msg = f"Group {date_str}_{orbit_state} encountered Dask cluster error: {e}"
                    logger.error(error_msg)
                    results.append((False, False, key, error_msg))
                    
                    # Add to failed lists
                    if key not in vv_failed_keys:
                        vv_failed_keys.append(key)
                    if key not in vh_failed_keys:
                        vh_failed_keys.append(key)
                    
                    # Try to restart the client
                    logger.warning("Dask client error detected. Attempting to restart...")
                    
                    # Gracefully shutdown the current client
                    try:
                        gracefully_shutdown_dask(client, cluster)
                    except:
                        pass
                    
                    # Create a new client
                    try:
                        client, cluster = setup_dask_client()
                        logger.info("Successfully restarted Dask client")
                    except Exception as e2:
                        logger.error(f"Failed to restart Dask client: {e2}")
                        # Continue with next item
                
                except Exception as exc:
                    error_msg = f"Group {date_str}_{orbit_state} generated an unhandled exception: {exc}"
                    logger.error(error_msg, exc_info=True)
                    results.append((False, False, key, error_msg))
                    
                    # Add to failed lists
                    if key not in vv_failed_keys:
                        vv_failed_keys.append(key)
                    if key not in vh_failed_keys:
                        vh_failed_keys.append(key)
        
        # Save final checkpoints
        save_checkpoint(vv_checkpoint_file, vv_completed_keys, vv_failed_keys)
        save_checkpoint(vh_checkpoint_file, vh_completed_keys, vh_failed_keys)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
    finally:
        # Always close the client to clean up resources
        gracefully_shutdown_dask(client, cluster)
    
    # Calculate and log statistics
    vv_successes = sum(1 for vv_success, _, _, _ in results if vv_success)
    vh_successes = sum(1 for _, vh_success, _, _ in results if vh_success)
    both_successes = sum(1 for vv_success, vh_success, _, _ in results if vv_success and vh_success)
    total_processed = len(results)
    
    logger.info(f"\n===== {orbit_state.upper()} ORBIT PROCESSING SUMMARY =====")
    logger.info(f"Groups processed: {total_processed}")
    logger.info(f"VV successful: {vv_successes}")
    logger.info(f"VH successful: {vh_successes}")
    logger.info(f"Both polarizations successful: {both_successes}")
    logger.info(f"VV success rate: {(vv_successes/total_processed)*100 if total_processed > 0 else 0:.1f}%")
    logger.info(f"VH success rate: {(vh_successes/total_processed)*100 if total_processed > 0 else 0:.1f}%")
    logger.info(f"Complete success rate: {(both_successes/total_processed)*100 if total_processed > 0 else 0:.1f}%")
    
    return vv_completed_keys, vv_failed_keys, vh_completed_keys, vh_failed_keys

def main():
    # Create a crash log file to catch unhandled exceptions
    sys.excepthook = lambda exctype, value, traceback: logger.critical(
        "Unhandled exception", exc_info=(exctype, value, traceback)
    )
    
    # Configure logging behavior for distributed module
    distributed_logger = logging.getLogger('distributed')
    # Only show warnings and errors from distributed module
    distributed_logger.setLevel(logging.WARNING)
    
    # Suppress verbose logging from other libraries
    logging.getLogger('planetary_computer').setLevel(logging.INFO)
    logging.getLogger('pystac_client').setLevel(logging.INFO)
    
    # Register signal handlers for graceful exit
    import signal
    
    # Variable to track if we're in shutdown mode
    is_shutting_down = [False]
    
    def graceful_exit_handler(signum, frame):
        if is_shutting_down[0]:
            # If we're already shutting down and get another signal, exit immediately
            logger.warning(f"Received second termination signal ({signum}). Exiting immediately.")
            sys.exit(1)
        
        is_shutting_down[0] = True
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name} signal. Gracefully shutting down...")
        
        # Set distributed logger to critical to suppress shutdown errors
        logging.getLogger('distributed').setLevel(logging.CRITICAL)
        
        # Just exit with a proper status code - resources will be cleaned up by atexit handlers
        sys.exit(0)
    
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGTERM, graceful_exit_handler)
    signal.signal(signal.SIGINT, graceful_exit_handler)
    
    # Print system info
    logger.info("=== System Information ===")
    system_info = get_system_status()
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {system_info['memory_available_gb']:.1f} GB")
    logger.info(f"Disk free: {system_info['disk_free_gb']:.1f} GB")
    
    # Log runtime parameters
    logger.info("=== Runtime Parameters ===")
    logger.info(f"Input TIFF: {args.input_tiff}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Orbit state: {args.orbit_state}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Dask workers: {args.dask_workers}")
    logger.info(f"Worker memory: {args.worker_memory} GB")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Timeout: {args.timeout} seconds")
    logger.info(f"Overwrite existing: {args.overwrite}")
    logger.info(f"Resume from checkpoint: {args.resume}")
    
    # Check if input TIFF exists
    if not os.path.exists(args.input_tiff):
        logger.error(f"Input TIFF file not found: {args.input_tiff}")
        sys.exit(1)
    
    logger.info(f"Reading input TIFF: {args.input_tiff}")
    
    # Read the input TIFF to extract geography and mask
    with rasterio.open(args.input_tiff) as src:
        # Get CRS, transform, and bounds
        src_crs = src.crs
        src_transform = src.transform
        src_bounds = src.bounds
        src_shape = src.shape
        
        # Read the data to create a mask (0 = invalid, >0 = valid)
        src_data = src.read(1)  # Read first band
        src_mask = (src_data > 0).astype(np.uint8)
        
        # Calculate pixel resolution
        pixel_width = src.transform.a  # Width of a pixel in coordinate units
        pixel_height = -src.transform.e  # Height of a pixel in coordinate units (usually negative)
        
        logger.info(f"Input TIFF CRS: {src_crs}")
        logger.info(f"Input TIFF bounds: {src_bounds}")
        logger.info(f"Input TIFF shape: {src_shape} pixels")
        logger.info(f"Input TIFF resolution: {pixel_width} x {pixel_height} meters")
        
        # Log the percentage of valid pixels
        valid_pixels = np.count_nonzero(src_mask) / src_mask.size
        logger.info(f"Input TIFF valid pixels: {valid_pixels:.1%}")
        
        # Convert bounds to WGS84 for API query
        wgs84_bounds = transform_bounds(src_crs, CRS.from_epsg(4326), 
                                        src_bounds.left, src_bounds.bottom, 
                                        src_bounds.right, src_bounds.top)
        
        # Create bbox for API query [minx, miny, maxx, maxy]
        bbox = [wgs84_bounds[0], wgs84_bounds[1], wgs84_bounds[2], wgs84_bounds[3]]
        
        # Create a WKT geometry for more precise querying if needed
        geom = shapely.geometry.box(*bbox)
        wkt_geom = geom.wkt
        
        logger.info(f"WGS84 bounding box for API query: {bbox}")
    
    # Format date range for API query
    date_range = f"{args.start_date}/{args.end_date}"
    logger.info(f"Searching for Sentinel-1 data for date range: {date_range}")
    
    # Set up the Planetary Computer STAC client
    # Each season execution gets a fresh token
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Planetary Computer STAC API: {e}")
        sys.exit(1)

    # Search for data using the bbox from the input TIFF
    try:
        search = catalog.search(
            collections=["sentinel-1-rtc"], 
            bbox=bbox,
            datetime=date_range
        )
        items = search.item_collection()
        logger.info(f"Found {len(items)} items")
        
        # Verify we have items before proceeding
        if len(items) == 0:
            logger.warning(f"No Sentinel-1 items found for the specified date range and location")
            logger.info("Processing complete - no data to download")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to search for Sentinel-1 data: {e}")
        sys.exit(1)

    # Process based on selected orbit state
    vv_ascending_completed = []
    vv_ascending_failed = []
    vh_ascending_completed = []
    vh_ascending_failed = []
    vv_descending_completed = []
    vv_descending_failed = []
    vh_descending_completed = []
    vh_descending_failed = []
    
    if args.orbit_state in ["ascending", "both"]:
        logger.info("\n=== PROCESSING ASCENDING ORBIT DATA ===")
        vv_ascending_completed, vv_ascending_failed, vh_ascending_completed, vh_ascending_failed = process_orbit_state(
            "ascending", items, src_crs, src_transform, src_shape, bbox, src_mask
        )
    else:
        logger.info("Skipping ascending orbit data processing")
    
    if args.orbit_state in ["descending", "both"]:
        logger.info("\n=== PROCESSING DESCENDING ORBIT DATA ===")
        vv_descending_completed, vv_descending_failed, vh_descending_completed, vh_descending_failed = process_orbit_state(
            "descending", items, src_crs, src_transform, src_shape, bbox, src_mask
        )
    else:
        logger.info("Skipping descending orbit data processing")
    
    # Print final summary
    total_vv_completed = len(vv_ascending_completed) + len(vv_descending_completed)
    total_vh_completed = len(vh_ascending_completed) + len(vh_descending_completed)
    total_vv_failed = len(vv_ascending_failed) + len(vv_descending_failed)
    total_vh_failed = len(vh_ascending_failed) + len(vh_descending_failed)
    
    logger.info(f"\n===== DOWNLOAD SUMMARY =====")
    logger.info(f"Total VV files processed successfully: {total_vv_completed}")
    logger.info(f"  VV Ascending orbits: {len(vv_ascending_completed)}")
    logger.info(f"  VV Descending orbits: {len(vv_descending_completed)}")
    logger.info(f"Total VH files processed successfully: {total_vh_completed}")
    logger.info(f"  VH Ascending orbits: {len(vh_ascending_completed)}")
    logger.info(f"  VH Descending orbits: {len(vh_descending_completed)}")
    logger.info(f"Total VV files failed: {total_vv_failed}")
    logger.info(f"  VV Ascending orbits: {len(vv_ascending_failed)}")
    logger.info(f"  VV Descending orbits: {len(vv_descending_failed)}")
    logger.info(f"Total VH files failed: {total_vh_failed}")
    logger.info(f"  VH Ascending orbits: {len(vh_ascending_failed)}")
    logger.info(f"  VH Descending orbits: {len(vh_descending_failed)}")
    logger.info(f"Season date range: {args.start_date} to {args.end_date}")
    logger.info(f"All files are saved in: {os.path.abspath(output_dir)}")
    logger.info(f"============================")
    
    # Generate a list of failed files for reference
    failed_items = set(vv_ascending_failed + vv_descending_failed + vh_ascending_failed + vh_descending_failed)
    
    if failed_items:
        failed_file = os.path.join(output_dir, f"failed_downloads_{season_name}.txt")
        try:
            with open(failed_file, 'w') as f:
                f.write(f"# Failed downloads for {args.start_date} to {args.end_date}\n")
                f.write(f"# Format: date_orbit - polarizations that failed\n\n")
                
                for key in sorted(failed_items):
                    # Determine which polarizations failed for this date/orbit
                    vv_failed = key in vv_ascending_failed or key in vv_descending_failed
                    vh_failed = key in vh_ascending_failed or key in vh_descending_failed
                    
                    # Extract the orbit state
                    orbit_state = "ascending" if key in vv_ascending_failed or key in vh_ascending_failed else "descending"
                    
                    # Write to file
                    failed_polarizations = []
                    if vv_failed:
                        failed_polarizations.append("VV")
                    if vh_failed:
                        failed_polarizations.append("VH")
                    
                    f.write(f"{key} ({orbit_state}) - {', '.join(failed_polarizations)}\n")
                    
            logger.info(f"List of failed downloads saved to: {failed_file}")
        except Exception as e:
            logger.error(f"Failed to save list of failed downloads: {e}")
    
    # Completely silence distributed logs before shutdown
    distributed_logger = logging.getLogger('distributed')
    distributed_logger.setLevel(logging.CRITICAL)  # Higher than ERROR - hide even critical errors
    
    # Clean up temporary directory if we created one
    if not args.temp_dir:
        try:
            # Add a delay to ensure no processes are still accessing the temp directory
            time.sleep(1)
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    # Force kill any remaining dask-worker processes that might belong to us
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if any('dask-worker' in cmd for cmd in cmdline if cmd):
                    proc.kill()
            except:
                pass
    except:
        pass
            
    # Successful exit
    logger.info("Script completed successfully")
    
    # Exit with a successful status code to ensure clean exit
    sys.exit(0)

if __name__ == "__main__":
    main()