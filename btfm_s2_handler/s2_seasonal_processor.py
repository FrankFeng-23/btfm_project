#!/usr/bin/env python3
# Optimized Sentinel-2 L2A Data Downloader

import os
import sys
import numpy as np
import pystac
import pystac_client
import planetary_computer
import logging
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.mask import mask
import concurrent.futures
import datetime
import tqdm
import argparse
import stackstac
import dask.array as da
import dask
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
import xarray as xr
from pystac.extensions.eo import EOExtension as eo

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
file_handler = logging.FileHandler("sentinel2_download.log")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Define band mappings - Sentinel-2 bands to output names
BAND_MAPPING = {
    "B04": "red",
    "B02": "blue",
    "B03": "green",
    "B08": "nir",
    "B8A": "nir08",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl"
}

# Define band resolutions
BAND_RESOLUTIONS = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20
}

# List of all bands we'll be using
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]

# Baseline change cutoff date
BASELINE_CUTOFF = datetime.datetime(2022, 1, 25)
BASELINE_OFFSET = 1000

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download Sentinel-2 data matching input TIFF geography for a specific season with separate band files')
parser.add_argument('--input_tiff', type=str, required=True, help='Input TIFF file to extract geography from')
parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
parser.add_argument('--workers', type=int, default=8, help='Number of parallel download workers (default: 8)')
parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory for intermediate files (default: system temp)')
parser.add_argument('--output', type=str, default='sentinel2_output', help='Output directory (default: sentinel2_output)')
parser.add_argument('--chunksize', type=int, default=4096, help='Chunk size for processing (default: 4096)')
parser.add_argument('--use_dask_gateway', action='store_true', help='Use Dask Gateway for distributed processing if available')
parser.add_argument('--dask_workers', type=int, default=32, help='Number of Dask workers to use (default: 32)')
parser.add_argument('--worker_memory', type=int, default=32, help='Memory per Dask worker in GB (default: 32)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files instead of skipping them')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for operations (default: 3)')
parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for individual operations (default: 600 seconds)')
parser.add_argument('--max_cloud_cover', type=float, default=90.0, help='Maximum cloud cover percentage to accept (default: 90)')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
parser.add_argument('--use_stackstac', action='store_true', help='Use stackstac for more efficient data processing')
parser.add_argument('--parallel_bands', action='store_true', default=True, help='Process bands in parallel')
parser.add_argument('--parallel_items', action='store_true', default=True, help='Process items in parallel')
parser.add_argument('--threads_per_worker', type=int, default=1, help='Threads per worker')
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

# Create directories for each band
for band_name in BAND_MAPPING.values():
    band_dir = os.path.join(output_dir, band_name)
    os.makedirs(band_dir, exist_ok=True)

# Create or use specified temporary directory
if args.temp_dir:
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
else:
    # Create a temp dir that will persist across the run
    temp_dir = tempfile.mkdtemp(prefix="sentinel2_")
logger.info(f"Using temporary directory: {temp_dir}")

# Extract season name for logs and checkpoints
start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
season_name = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

# Update log file name to include season
log_file_path = os.path.join(output_dir, f"sentinel2_{season_name}.log")
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        handler.close()
        root_logger.removeHandler(handler)

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Define checkpoint file path
checkpoint_file = os.path.join(checkpoint_dir, f"sentinel2_{season_name}_checkpoint.json")

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
        # Close client
        client.close(timeout=1)
        logger.info("Closed Dask client")
    except Exception as e:
        logger.debug(f"Error closing Dask client: {e}")
        
    # Then close cluster if we have access to it
    if cluster and client.cluster is not cluster:  # Avoid double-closing
        try:
            # Add a small delay to allow client to finish closing first
            time.sleep(1)
            cluster.close()
            logger.info("Closed Dask cluster")
        except Exception as e:
            logger.debug(f"Error closing Dask cluster: {e}")
   
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
    # OPTIMIZATION: Configure Dask for better performance
    dask.config.set({
        'distributed.worker.memory.target': 0.9,       # Target 90% memory use
        'distributed.worker.memory.spill': 0.95,       # Spill at 95% memory use
        'distributed.worker.memory.pause': 0.98,       # Pause at 98% memory use
        'distributed.scheduler.work-stealing': True,   # Enable work stealing for better load balancing
        'distributed.worker.profile.interval': '1000ms', # Reduce profiling overhead
        'distributed.comm.timeouts.connect': '20s',    # Increase connection timeout
        'distributed.comm.timeouts.tcp': '20s',        # Increase TCP timeout
        'distributed.admin.tick.limit': '3s',          # Increase tick limit
        'array.slicing.split_large_chunks': True,      # Better array slicing
    })
    
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
        
        # OPTIMIZATION: Improved worker configuration
        threads_per_worker = args.threads_per_worker
        memory_limit = f"{args.worker_memory}GB"
        
        # Create a local cluster with specified parameters
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,  # OPTIMIZATION: Configurable threads per worker
            memory_limit=memory_limit,
            processes=True,
            silence_logs=logging.WARNING
        )
        
        client = Client(cluster)
        
        # Safely wait for workers
        try:
            client.wait_for_workers(n_workers=n_workers, timeout=60)
            logger.info(f"Local Dask client created with {len(client.scheduler_info()['workers'])} workers")
        except Exception as e:
            logger.warning(f"Failed to wait for all workers, but continuing: {e}")
        
        logger.info(f"Memory per worker: {memory_limit}")
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

def validate_tiff(file_path, expected_shape=None, expected_crs=None, expected_transform=None):
    """
    Validate that a TIFF file has the expected properties
    Returns True if valid, False if not
    """
    try:
        with rasterio.open(file_path) as src:
            # If no expected values are provided, just check if the file can be opened
            if expected_shape is None and expected_crs is None and expected_transform is None:
                # Check for data presence
                stats = [src.statistics(i) for i in range(1, src.count + 1)]
                if any(s.max == 0 and s.min == 0 for s in stats):
                    logger.warning(f"Validation failed: Band contains all zeros in {file_path}")
                    return False
                return True
                
            # Check basic properties
            if expected_shape is not None and src.shape != expected_shape:
                logger.warning(f"Validation failed: Shape mismatch in {file_path}. Expected {expected_shape}, got {src.shape}")
                return False
            
            if expected_crs is not None and src.crs != expected_crs:
                logger.warning(f"Validation failed: CRS mismatch in {file_path}. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check transform if provided
            if expected_transform is not None and not np.allclose(np.array(src.transform)[:6], np.array(expected_transform)[:6], rtol=1e-05, atol=1e-08):
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
            if file_size_mb < expected_size_mb * 0.01:  # Allow for compression, but not too small
                logger.warning(f"Validation failed: File size too small in {file_path}. Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
                return False
            
            logger.debug(f"TIFF validation passed for {file_path}: shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logger.error(f"Error validating TIFF {file_path}: {e}")
        return False

def harmonize_to_old(data, valid_mask=None):
    """
    Harmonize new Sentinel-2 data (acquired after January 25, 2022) to the old baseline.

    Parameters
    ----------
    data: numpy.ndarray
        The data to harmonize
    valid_mask: numpy.ndarray, optional
        Mask of valid pixels (1 = valid, 0 = invalid)

    Returns
    -------
    harmonized: numpy.ndarray
        Harmonized data with values adjusted to match the old baseline.
    """
    # Make a copy to avoid modifying original data
    result = data.copy()
    
    # Apply the valid mask if provided
    if valid_mask is not None:
        # Only apply harmonization to valid pixels (non-zero in the mask)
        process_mask = valid_mask > 0
    else:
        # Process all non-zero pixels if no mask provided
        process_mask = data > 0
    
    # Apply the baseline adjustment to all values >= BASELINE_OFFSET
    baseline_mask = (result >= BASELINE_OFFSET) & process_mask
    
    if np.any(baseline_mask):
        result[baseline_mask] -= BASELINE_OFFSET
        logger.debug(f"Harmonized data: adjusted {np.sum(baseline_mask)} pixels")
    
    # Ensure invalid pixels remain at 0
    if valid_mask is not None:
        result[valid_mask == 0] = 0
    
    return result

def load_checkpoint(checkpoint_file):
    """Load checkpoint data from file"""
    if os.path.exists(checkpoint_file) and args.resume:
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_file}: {e}")
    
    # Return empty checkpoint if file doesn't exist or resume is not enabled
    return {'completed_dates': [], 'failed_dates': []}

def save_checkpoint(checkpoint_file, completed_dates, failed_dates):
    """Save checkpoint data to file"""
    checkpoint_data = {
        'completed_dates': completed_dates,
        'failed_dates': failed_dates,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_file}: {e}")

def group_items_by_date(items):
    """
    Group items by date
    Returns a dictionary: {date_str: [items]}
    """
    groups = defaultdict(list)
    for item in items:
        date_str = item.properties.get("datetime").split("T")[0]
        groups[date_str].append(item)
    return groups

def print_date_statistics(grouped_items):
    """
    Print statistics about the grouped items
    """
    # Count dates with multiple observations
    multiple_obs_dates = {key: items for key, items in grouped_items.items() if len(items) > 1}
    
    # Count total unique dates
    unique_dates = list(grouped_items.keys())
    
    # Sort dates
    unique_dates.sort()
    
    # Print statistics
    logger.info(f"Total unique dates: {len(unique_dates)}")
    logger.info(f"Dates with multiple observations: {len(multiple_obs_dates)}")
    
    # Print first and last date
    if unique_dates:
        logger.info(f"First date: {unique_dates[0]}")
        logger.info(f"Last date: {unique_dates[-1]}")
    
    # Print details about dates with multiple observations
    if multiple_obs_dates:
        logger.info("\nDates with multiple observations:")
        for date_str, items in multiple_obs_dates.items():
            cloud_cover_values = [eo.ext(item).cloud_cover for item in items if hasattr(eo.ext(item), 'cloud_cover')]
            if cloud_cover_values:
                avg_cloud_cover = sum(cloud_cover_values) / len(cloud_cover_values)
                logger.info(f"  {date_str}: {len(items)} observations, avg cloud cover: {avg_cloud_cover:.2f}%")
            else:
                logger.info(f"  {date_str}: {len(items)} observations, cloud cover: unknown")

# OPTIMIZATION: More efficient warping function
def warp_to_template(src_path, dst_path, template_profile, mask=None, resampling=Resampling.nearest):
    """
    Warp a source file to match a template profile
    """
    try:
        with rasterio.open(src_path) as src:
            # Create output profile using template and source
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': template_profile['crs'],
                'transform': template_profile['transform'],
                'width': template_profile['width'],
                'height': template_profile['height'],
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256
            })
            
            # Write to the destination file
            with rasterio.open(dst_path, 'w', **dst_profile) as dst:
                # Reproject data in one operation
                reproject(
                    source=rasterio.band(src, 1),
                    destination=np.zeros((template_profile['height'], template_profile['width']), dtype=dst_profile['dtype']),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=template_profile['transform'],
                    dst_crs=template_profile['crs'],
                    resampling=resampling,
                    dst_shape=(template_profile['height'], template_profile['width']),
                    dst_kwargs={'out': dst.read(1, masked=False)}
                )
                
                # Apply mask if provided
                if mask is not None:
                    data = dst.read(1)
                    data[mask == 0] = 0
                    dst.write(data, 1)
                    
                # Copy metadata from source
                dst.update_tags(**src.tags())
                
                # Copy band-specific metadata
                for tag, value in src.tags(1).items():
                    dst.update_tags(1, **{tag: value})
                    
        return True
    
    except Exception as e:
        logger.error(f"Error warping {src_path} to {dst_path}: {e}")
        return False

# OPTIMIZATION: More efficient mosaic function
@retry_with_backoff(exceptions=(Exception,), max_retries=2, timeout_seconds=300)
def mosaic_tiffs(tiff_paths, output_path, template_profile, date_str, band_name, mask=None):
    """
    Mosaic multiple single-band TIFFs into a single output TIFF
    that matches the template profile
    """
    if not tiff_paths:
        logger.warning(f"No files to mosaic for {date_str}_{band_name}")
        return None
        
    try:
        # Open all source TIFFs
        src_files = []
        for path in tiff_paths:
            if path and os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logger.warning(f"Failed to open {path} for mosaicking {band_name}: {e}")
        
        if not src_files:
            logger.warning(f"No valid files to mosaic for {date_str}_{band_name}")
            return None
        
        # Perform mosaic operation with error handling
        try:
            logger.info(f"Mosaicking {len(src_files)} {band_name} files for {date_str}...")
            mosaic_data, out_transform = merge(src_files)
        except Exception as e:
            logger.error(f"Error during merge operation for {date_str}_{band_name}: {e}")
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
            logger.error(f"Mosaic data has incorrect structure for {date_str}_{band_name}")
            return None
        
        # Determine datatype - keep SCL as uint8, others as uint16
        if band_name == 'scl':
            output_dtype = 'uint8'
        else:
            output_dtype = 'uint16'
        
        # Write directly to the output file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=template_profile['height'],
            width=template_profile['width'],
            count=1,
            dtype=output_dtype,
            crs=template_profile['crs'],
            transform=template_profile['transform'],
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            # Reproject the mosaic data to match the template
            dst_data = np.zeros((template_profile['height'], template_profile['width']), dtype=output_dtype)
            
            reproject(
                source=mosaic_data[0],
                destination=dst_data,
                src_transform=out_transform,
                src_crs=src_files[0].crs,
                dst_transform=template_profile['transform'],
                dst_crs=template_profile['crs'],
                resampling=Resampling.nearest
            )
            
            # Apply mask if provided
            if mask is not None:
                dst_data[mask == 0] = 0
            
            # Write the data
            dst.write(dst_data, 1)
            
            # Add metadata
            dst.update_tags(
                TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                DATE_ACQUIRED=date_str,
                BAND=band_name,
                MOSAIC_SOURCE_COUNT=len(src_files),
                SOURCE_TIFF=os.path.basename(args.input_tiff),
                DESCRIPTION=f"Mosaicked Sentinel-2 {band_name} band"
            )
        
        # Validate the output file
        if not validate_tiff(output_path):
            logger.error(f"Mosaic TIFF validation failed for {date_str}_{band_name}. Removing file.")
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
        logger.error(f"Unexpected error creating mosaic for {date_str}_{band_name}: {str(e)}", exc_info=True)
        return None

# OPTIMIZATION: Improved processing with direct STAC item handling
def process_band(client, item, s2_band, output_band, template_profile, temp_dir, mask=None, date_str=None, is_after_cutoff=False):
    """Process a single band from a STAC item"""
    try:
        if date_str is None:
            date_str = item.properties.get("datetime").split("T")[0]
            
        item_id = item.id
        resolution = BAND_RESOLUTIONS.get(s2_band, 10)  # Default to 10m if unknown
        
        # Create a unique filename for the band
        unique_id = uuid.uuid4().hex[:8]
        temp_file = os.path.join(temp_dir, f"{date_str}_{output_band}_{unique_id}.tiff")
        
        # Get the asset URL
        if s2_band not in item.assets:
            logger.warning(f"Band {s2_band} not found in item {item_id}")
            return None
            
        asset_href = item.assets[s2_band].href
        
        # Download to a temporary file first
        raw_temp = os.path.join(temp_dir, f"{date_str}_{s2_band}_{uuid.uuid4().hex}.tiff")
        
        # Use rasterio to download
        try:
            # OPTIMIZATION: More efficient download
            with rasterio.open(asset_href) as src:
                profile = src.profile.copy()
                with rasterio.open(raw_temp, 'w', **profile) as dst:
                    # Read and write in one operation
                    dst.write(src.read())
        except Exception as e:
            logger.error(f"Error downloading {s2_band} for {item_id}: {e}")
            return None
        
        # Apply harmonization if needed (to the downloaded file)
        if is_after_cutoff and s2_band != 'SCL':
            logger.debug(f"Harmonizing band {s2_band} to old baseline")
            try:
                with rasterio.open(raw_temp, 'r+') as src:
                    data = src.read(1)
                    # Harmonize the data
                    harmonized = harmonize_to_old(data)
                    # Write back to the same file
                    src.write(harmonized, 1)
            except Exception as e:
                logger.error(f"Error harmonizing {s2_band} for {item_id}: {e}")
                # Continue with unharmonized data
        
        # Warp the raw temp file to match the template
        warp_success = warp_to_template(
            raw_temp,
            temp_file,
            template_profile,
            mask=mask,
            resampling=Resampling.nearest
        )
        
        # Clean up raw temp file
        try:
            if os.path.exists(raw_temp):
                os.remove(raw_temp)
        except:
            pass
        
        if warp_success:
            # Add metadata to the warped file
            try:
                with rasterio.open(temp_file, 'r+') as dst:
                    # Add metadata
                    dst.update_tags(
                        TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        DATE_ACQUIRED=date_str,
                        BAND=s2_band,
                        OUTPUT_BAND=output_band,
                        RESOLUTION=f"{resolution}m",
                        ITEM_ID=item_id,
                        SOURCE_TIFF=os.path.basename(args.input_tiff),
                        BASELINE_HARMONIZED=str(is_after_cutoff),
                        DESCRIPTION=f"Sentinel-2 {s2_band} band ({output_band}) at {resolution}m resolution"
                    )
            except Exception as e:
                logger.warning(f"Error adding metadata to {temp_file}: {e}")
        
            # Validate the output file
            if validate_tiff(temp_file):
                file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                logger.info(f"Successfully processed {s2_band} ({output_band}) band: {file_size_mb:.2f} MB")
                return temp_file
            else:
                logger.error(f"Failed to validate {s2_band} band output")
                # Try to remove the invalid file
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        else:
            logger.error(f"Failed to warp {s2_band} band for {item_id}")
        
        return None
    except Exception as e:
        logger.error(f"Error processing band {s2_band} for {item.id}: {e}")
        return None

# OPTIMIZATION: Process_item function that efficiently processes multiple bands
def process_item(client, item, template_profile, temp_dir=None, mask=None):
    """
    Process a single item and save individual band files
    """
    try:
        date_str = item.properties.get("datetime").split("T")[0]
        item_id = item.id
        
        logger.info(f"Processing item {item_id} for {date_str}...")
        
        # Check if this item is after the baseline cutoff
        item_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        is_after_cutoff = item_date > BASELINE_CUTOFF
        if is_after_cutoff:
            logger.info(f"Item {item_id} is after baseline cutoff (Jan 25, 2022), will apply harmonization")
        
        # OPTIMIZATION: Process bands in parallel if enabled
        band_outputs = {}
        
        if args.parallel_bands:
            # Process bands in parallel
            futures = {}
            
            for s2_band, output_band in BAND_MAPPING.items():
                # Skip any bands not available in this item's assets
                if s2_band not in item.assets:
                    continue
                
                # Submit the band processing task
                future = client.submit(
                    process_band,
                    client, item, s2_band, output_band, template_profile, 
                    temp_dir, mask, date_str, is_after_cutoff,
                    pure=False  # Important for I/O tasks
                )
                futures[output_band] = future
            
            # Gather results
            for output_band, future in futures.items():
                try:
                    result = future.result()
                    if result:
                        band_outputs[output_band] = result
                except Exception as e:
                    logger.error(f"Error processing band {output_band} for {item_id}: {e}")
        else:
            # Process bands sequentially
            for s2_band, output_band in BAND_MAPPING.items():
                # Skip any bands not available in this item's assets
                if s2_band not in item.assets:
                    continue
                
                result = process_band(
                    client, item, s2_band, output_band, template_profile, 
                    temp_dir, mask, date_str, is_after_cutoff
                )
                
                if result:
                    band_outputs[output_band] = result
        
        return band_outputs
            
    except Exception as e:
        logger.error(f"Unexpected error processing {item_id}: {str(e)}", exc_info=True)
        return {}

# OPTIMIZATION: Using stackstac for more efficient data processing
def process_date_with_stackstac(client, date_str, items, template_profile, mask=None):
    """
    Process date using stackstac for more efficient data loading and processing
    """
    try:
        logger.info(f"Processing date {date_str} with {len(items)} items using stackstac")
        
        # Create a subdirectory in temp_dir for this specific date
        date_temp_dir = os.path.join(temp_dir, date_str)
        os.makedirs(date_temp_dir, exist_ok=True)
        
        # Extract geobox (bounds, crs) from template_profile
        with rasterio.open(args.input_tiff) as src:
            bounds = src.bounds
            dest_crs = src.crs
            
        # Get STAC items as a list
        items_list = list(items)
        
        # Check if any item is after baseline cutoff date
        any_after_cutoff = any(
            datetime.datetime.strptime(item.properties.get("datetime").split("T")[0], "%Y-%m-%d") 
            > BASELINE_CUTOFF 
            for item in items_list
        )
        
        # Define bands to load
        bands_to_load = list(BAND_MAPPING.keys())
        
        # Set up stackstac with bound data
        stack = stackstac.stack(
            items_list,
            bounds=bounds,
            epsg=dest_crs.to_epsg(),
            resolution=10,  # Use 10m resolution (can set per-band later if needed)
            bands=bands_to_load,
            chunksize=args.chunksize  # Larger chunks for better performance
        )
        
        final_outputs = {}
        
        # Process each band
        for s2_band, output_band in BAND_MAPPING.items():
            try:
                if s2_band not in stack.band.values:
                    logger.warning(f"Band {s2_band} not found in stackstac result")
                    continue
                
                # Extract the single band data
                band_data = stack.sel(band=s2_band)
                
                # Take the median if multiple observations (reduces cloud impact)
                if len(band_data.time) > 1:
                    result = band_data.median(dim="time")
                else:
                    result = band_data.isel(time=0)
                
                # Apply harmonization if needed
                if any_after_cutoff and s2_band != 'SCL':
                    logger.info(f"Harmonizing band {s2_band} data to old baseline")
                    # Convert to numpy for harmonization
                    data_array = result.compute().values
                    
                    # Apply harmonization
                    harmonized = harmonize_to_old(data_array, mask)
                    
                    # Convert back to xarray
                    result = xr.DataArray(
                        harmonized,
                        dims=result.dims,
                        coords=result.coords,
                        attrs=result.attrs
                    )
                
                # Define output path
                band_dir = os.path.join(output_dir, output_band)
                output_path = os.path.join(band_dir, f"{date_str}_mosaic.tiff")
                
                # Check if output already exists
                if os.path.exists(output_path) and not args.overwrite:
                    if validate_tiff(output_path):
                        logger.info(f"File for {date_str}_{output_band} already exists and is valid, skipping.")
                        final_outputs[output_band] = output_path
                        continue
                    else:
                        logger.warning(f"File for {date_str}_{output_band} exists but is invalid. Reprocessing.")
                        try:
                            os.remove(output_path)
                        except:
                            pass
                
                # Write to file
                if s2_band == 'SCL':
                    dtype = 'uint8'
                else:
                    dtype = 'uint16'
                
                # Convert to dask array for chunked processing
                data_values = da.from_array(result.values, chunks=args.chunksize)
                
                # Apply mask
                if mask is not None:
                    data_values = da.where(mask > 0, data_values, 0)
                
                # Write to file
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=template_profile['height'],
                    width=template_profile['width'],
                    count=1,
                    dtype=dtype,
                    crs=template_profile['crs'],
                    transform=template_profile['transform'],
                    compress='lzw',
                    tiled=True,
                    blockxsize=256,
                    blockysize=256
                ) as dst:
                    # Compute and write
                    dst.write(data_values.compute(), 1)
                    
                    # Add metadata
                    dst.update_tags(
                        TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        DATE_ACQUIRED=date_str,
                        BAND=s2_band,
                        OUTPUT_BAND=output_band,
                        SOURCE_TIFF=os.path.basename(args.input_tiff),
                        BASELINE_HARMONIZED=str(any_after_cutoff),
                        DESCRIPTION=f"Sentinel-2 {s2_band} band processed with stackstac"
                    )
                
                # Validate the output file
                if validate_tiff(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"Successfully processed {s2_band} ({output_band}) band with stackstac: {file_size_mb:.2f} MB")
                    final_outputs[output_band] = output_path
                else:
                    logger.error(f"Failed to validate {output_band} output from stackstac")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"Error processing band {s2_band} with stackstac: {e}")
        
        # Calculate success status
        num_expected_bands = len(BAND_MAPPING)
        num_processed_bands = len(final_outputs)
        
        if num_processed_bands == 0:
            logger.error(f"Failed to process any bands for {date_str} with stackstac")
            return False, f"Failed to process any bands for {date_str} with stackstac"
        elif num_processed_bands < num_expected_bands:
            logger.warning(f"Partially processed {date_str} with stackstac: {num_processed_bands}/{num_expected_bands} bands")
            return True, f"Partially processed {date_str} with stackstac: {num_processed_bands}/{num_expected_bands} bands"
        else:
            logger.info(f"Successfully processed all bands for {date_str} with stackstac")
            return True, f"Successfully processed all bands for {date_str} with stackstac"
            
    except Exception as e:
        logger.error(f"Error in stackstac processing for {date_str}: {e}")
        return False, f"Error in stackstac processing for {date_str}: {e}"

# OPTIMIZATION: Improved process_date function with parallel item processing
def process_date(client, date_str, items, template_profile, mask=None):
    """
    Process all items for a specific date, creating mosaics if necessary
    Returns a dictionary of successfully processed bands
    """
    # Try stackstac if enabled
    if args.use_stackstac:
        try:
            return process_date_with_stackstac(client, date_str, items, template_profile, mask)
        except Exception as e:
            logger.error(f"Stackstac processing failed for {date_str}: {e}. Falling back to traditional processing.")
            # Fall back to traditional processing
    
    try:
        logger.info(f"Processing date {date_str} with {len(items)} items")
        
        # Create a subdirectory in temp_dir for this specific date
        date_temp_dir = os.path.join(temp_dir, date_str)
        try:
            os.makedirs(date_temp_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create temp directory for {date_str}: {e}")
            # Continue using the parent temp directory if this fails
            date_temp_dir = temp_dir
        
        # Process each item separately - in parallel if enabled
        all_band_files = defaultdict(list)
        
        if args.parallel_items:
            # Process items in parallel with Dask
            futures = []
            
            for item in items:
                future = client.submit(
                    process_item,
                    client, item, template_profile, date_temp_dir, mask,
                    pure=False  # Important for I/O tasks
                )
                futures.append(future)
            
            # Gather results as they complete
            for future in distributed.as_completed(futures):
                try:
                    item_band_files = future.result()
                    
                    # Add the processed files to the collection
                    for band_name, file_path in item_band_files.items():
                        all_band_files[band_name].append(file_path)
                except Exception as e:
                    logger.error(f"Error processing item for {date_str}: {e}")
        else:
            # Process items sequentially
            for item in items:
                try:
                    item_band_files = process_item(client, item, template_profile, date_temp_dir, mask)
                    
                    # Add the processed files to the collection
                    for band_name, file_path in item_band_files.items():
                        all_band_files[band_name].append(file_path)
                
                except Exception as e:
                    logger.error(f"Error processing item {item.id} for {date_str}: {e}")
                    # Continue with other items
        
        # Create final mosaics for each band
        final_outputs = {}
        mosaic_futures = []
        
        for band_name, file_paths in all_band_files.items():
            # Define the final output path
            band_dir = os.path.join(output_dir, band_name)
            final_output_path = os.path.join(band_dir, f"{date_str}_mosaic.tiff")
            
            # Check if output already exists and is valid
            if os.path.exists(final_output_path) and not args.overwrite:
                if validate_tiff(final_output_path):
                    logger.info(f"File for {date_str}_{band_name} already exists and is valid, skipping.")
                    final_outputs[band_name] = final_output_path
                    continue
                else:
                    logger.warning(f"File for {date_str}_{band_name} exists but is invalid. Reprocessing.")
                    try:
                        os.remove(final_output_path)
                    except Exception as e:
                        logger.error(f"Failed to remove invalid file for {date_str}_{band_name}: {e}")
            
            # If we have only one file, just rename it
            if len(file_paths) == 1:
                logger.info(f"Only one valid {band_name} file for {date_str}, using it directly.")
                try:
                    shutil.copy2(file_paths[0], final_output_path)
                    os.remove(file_paths[0])
                    final_outputs[band_name] = final_output_path
                except Exception as e:
                    logger.error(f"Error copying single {band_name} file for {date_str}: {e}")
            
            # If we have multiple files, create a mosaic
            elif len(file_paths) > 1:
                # Submit mosaic task to Dask
                future = client.submit(
                    mosaic_tiffs,
                    file_paths, 
                    final_output_path, 
                    template_profile, 
                    date_str, 
                    band_name,
                    mask,
                    pure=False  # Important for I/O operations
                )
                mosaic_futures.append((band_name, future, file_paths))
            
            else:
                logger.warning(f"No valid {band_name} files for {date_str}")
        
        # Process mosaic results
        for band_name, future, file_paths in mosaic_futures:
            try:
                mosaic_result = future.result()
                
                # Clean up temporary files
                for temp_file in file_paths:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary {band_name} file {temp_file}: {e}")
                
                if mosaic_result:
                    final_outputs[band_name] = mosaic_result
                
            except Exception as e:
                logger.error(f"Error mosaicking {band_name} files for {date_str}: {e}")
        
        # Calculate success status
        num_expected_bands = len(BAND_MAPPING)
        num_processed_bands = len(final_outputs)
        
        if num_processed_bands == 0:
            logger.error(f"Failed to process any bands for {date_str}")
            return False, f"Failed to process any bands for {date_str}"
        elif num_processed_bands < num_expected_bands:
            logger.warning(f"Partially processed {date_str}: {num_processed_bands}/{num_expected_bands} bands")
            return True, f"Partially processed {date_str}: {num_processed_bands}/{num_expected_bands} bands"
        else:
            logger.info(f"Successfully processed all bands for {date_str}")
            return True, f"Successfully processed all bands for {date_str}"
    
    except Exception as e:
        logger.error(f"Unexpected error processing date {date_str}: {e}", exc_info=True)
        return False, f"Error processing {date_str}: {str(e)}"

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
    logger.info(f"Max cloud cover: {args.max_cloud_cover}%")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Dask workers: {args.dask_workers}")
    logger.info(f"Worker memory: {args.worker_memory} GB")
    logger.info(f"Using stackstac: {args.use_stackstac}")
    logger.info(f"Parallel bands: {args.parallel_bands}")
    logger.info(f"Parallel items: {args.parallel_items}")
    logger.info(f"Chunk size: {args.chunksize}")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Timeout: {args.timeout} seconds")
    logger.info(f"Overwrite existing: {args.overwrite}")
    logger.info(f"Resume from checkpoint: {args.resume}")
    
    # Check if input TIFF exists
    if not os.path.exists(args.input_tiff):
        logger.error(f"Input TIFF file not found: {args.input_tiff}")
        sys.exit(1)
    
    logger.info(f"Reading input TIFF: {args.input_tiff}")
    
    # Read the input TIFF to extract geography and create template profile
    template_profile = {}
    input_mask = None
    
    with rasterio.open(args.input_tiff) as src:
        # Get CRS, transform, and bounds
        template_profile['crs'] = src.crs
        template_profile['transform'] = src.transform
        template_profile['width'] = src.width
        template_profile['height'] = src.height
        src_bounds = src.bounds
        
        # Read the data to create a mask (0 = invalid, 1 = valid)
        mask_data = src.read(1)
        input_mask = (mask_data > 0).astype(np.uint8)
        
        # Calculate pixel resolution
        pixel_width = src.transform.a  # Width of a pixel in coordinate units
        pixel_height = -src.transform.e  # Height of a pixel in coordinate units (usually negative)
        
        logger.info(f"Input TIFF CRS: {template_profile['crs']}")
        logger.info(f"Input TIFF bounds: {src_bounds}")
        logger.info(f"Input TIFF dimensions: {template_profile['width']} x {template_profile['height']} pixels")
        logger.info(f"Input TIFF resolution: {pixel_width} x {pixel_height} meters")
        
        # Log the percentage of valid pixels
        valid_pixels = np.count_nonzero(input_mask) / input_mask.size
        logger.info(f"Input TIFF valid pixels: {valid_pixels:.1%}")
        
        # Convert bounds to WGS84 for API query
        wgs84_bounds = transform_bounds(template_profile['crs'], CRS.from_epsg(4326), 
                                      src_bounds.left, src_bounds.bottom, 
                                      src_bounds.right, src_bounds.top)
        
        # Create bbox for API query [minx, miny, maxx, maxy]
        bbox = [wgs84_bounds[0], wgs84_bounds[1], wgs84_bounds[2], wgs84_bounds[3]]
        
        logger.info(f"WGS84 bounding box for API query: {bbox}")
    
    # Format date range for API query
    date_range = f"{args.start_date}/{args.end_date}"
    logger.info(f"Searching for Sentinel-2 data for date range: {date_range}")
    
    # Set up the Planetary Computer STAC client
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
            collections=["sentinel-2-l2a"], 
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": args.max_cloud_cover}}
        )
        items = search.item_collection()
        logger.info(f"Found {len(items)} items with cloud cover < {args.max_cloud_cover}%")
        
        # Verify we have items before proceeding
        if len(items) == 0:
            logger.warning(f"No Sentinel-2 items found for the specified date range and location")
            logger.info("Processing complete - no data to download")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to search for Sentinel-2 data: {e}")
        sys.exit(1)

    # Group items by date
    grouped_items = group_items_by_date(items)
    logger.info(f"Grouped into {len(grouped_items)} unique dates")
    
    # Print statistics
    print_date_statistics(grouped_items)
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_file)
    completed_dates = checkpoint_data.get('completed_dates', [])
    failed_dates = checkpoint_data.get('failed_dates', [])
    
    logger.info(f"Checkpoint loaded: {len(completed_dates)} completed, {len(failed_dates)} failed")
    
    # Filter dates based on checkpoint if resume is enabled
    if args.resume:
        filtered_groups = {}
        for date_str, date_items in grouped_items.items():
            # Check if date is already completed
            if date_str in completed_dates and not args.overwrite:
                # Check if all band files exist and are valid
                all_bands_valid = True
                for band_name in BAND_MAPPING.values():
                    band_dir = os.path.join(output_dir, band_name)
                    band_file = os.path.join(band_dir, f"{date_str}_mosaic.tiff")
                    if not os.path.exists(band_file) or not validate_tiff(band_file):
                        all_bands_valid = False
                        break
                
                if all_bands_valid:
                    logger.debug(f"Skipping {date_str} (completed in previous run)")
                    continue
                else:
                    logger.info(f"{date_str} marked as completed but files are missing or invalid. Re-processing.")
            
            # Include date for processing
            filtered_groups[date_str] = date_items
        
        logger.info(f"After filtering based on checkpoints: {len(filtered_groups)} dates to process")
        grouped_items = filtered_groups
    
    # Check if there's anything to process
    if not grouped_items:
        logger.info(f"No dates to process after filtering")
        sys.exit(0)
    
    # Convert to list and sort by date for more predictable processing
    dates_list = sorted(grouped_items.items(), key=lambda x: x[0])
    
    # OPTIMIZATION: Configure Dask for better performance
    dask.config.set({'distributed.worker.resources.local_threads': args.threads_per_worker})
    
    # Set up a Dask client for processing
    client, cluster = setup_dask_client()
    
    try:
        # Log system status before starting
        log_system_status()
        
        # OPTIMIZATION: Process dates in batches to avoid overwhelming the scheduler
        results = []
        batch_size = min(args.workers * 2, len(dates_list))  # Process 2 dates per worker at once
        
        for i in range(0, len(dates_list), batch_size):
            batch = dates_list[i:i+batch_size]
            logger.info(f"Processing batch of {len(batch)} dates ({i+1}-{i+len(batch)} of {len(dates_list)})")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks for this batch
                future_to_date = {}
                for date_str, date_items in batch:
                    future = executor.submit(
                        process_date, client, date_str, date_items, template_profile, input_mask
                    )
                    future_to_date[future] = date_str
                
                # Process as they complete
                for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_date), 
                                      total=len(future_to_date), desc=f"Processing dates"):
                    date_str = future_to_date[future]
                    
                    try:
                        success, message = future.result()
                        results.append((success, date_str, message))
                        
                        # Update completed/failed lists
                        if success:
                            if date_str not in completed_dates:
                                completed_dates.append(date_str)
                            if date_str in failed_dates:
                                failed_dates.remove(date_str)
                        else:
                            if date_str not in failed_dates:
                                failed_dates.append(date_str)
                        
                    except Exception as exc:
                        error_msg = f"Date {date_str} generated an unhandled exception: {exc}"
                        logger.error(error_msg, exc_info=True)
                        results.append((False, date_str, error_msg))
                        
                        # Add to failed lists
                        if date_str not in failed_dates:
                            failed_dates.append(date_str)
            
            # Save checkpoint after each batch
            save_checkpoint(checkpoint_file, completed_dates, failed_dates)
            
            # OPTIMIZATION: Clear any cached data between batches
            client.run(lambda: gc.collect())
            time.sleep(2)  # Short pause between batches
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
    finally:
        # Always close the client to clean up resources
        gracefully_shutdown_dask(client, cluster)
    
    # Calculate and log statistics
    successes = sum(1 for success, _, _ in results if success)
    total_processed = len(results)
    
    logger.info(f"\n===== PROCESSING SUMMARY =====")
    logger.info(f"Dates processed: {total_processed}")
    logger.info(f"Successful: {successes}")
    logger.info(f"Failed: {total_processed - successes}")
    logger.info(f"Success rate: {(successes/total_processed)*100 if total_processed > 0 else 0:.1f}%")
    
    # Generate a list of failed dates for reference
    if failed_dates:
        failed_file = os.path.join(output_dir, f"failed_downloads_{season_name}.txt")
        try:
            with open(failed_file, 'w') as f:
                f.write(f"# Failed downloads for {args.start_date} to {args.end_date}\n")
                f.write(f"# Format: date - failure reason\n\n")
                
                for date_str in sorted(failed_dates):
                    # Find the failure message if available
                    message = next((msg for success, date, msg in results if not success and date == date_str), "Unknown error")
                    f.write(f"{date_str} - {message}\n")
                    
            logger.info(f"List of failed downloads saved to: {failed_file}")
        except Exception as e:
            logger.error(f"Failed to save list of failed downloads: {e}")
    
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
        import gc
        gc.collect()  # Attempt to clean up any remaining references
        
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