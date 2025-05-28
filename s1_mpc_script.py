#!/usr/bin/env python3
# Robust Sentinel-1 SAR Data Downloader with fixed timeout handling

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
from rasterio.warp import transform_bounds, reproject, Resampling
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
parser = argparse.ArgumentParser(description='Download Sentinel-1 data matching input TIFF geography with robust handling')
parser.add_argument('--input_tiff', type=str, required=True, help='Input TIFF file to extract geography from')
parser.add_argument('--workers', type=int, default=8, help='Number of parallel download workers (default: 4)')
parser.add_argument('--temp_dir', type=str, default='sentinel1_tmp', help='Temporary directory for intermediate files (default: system temp)')
parser.add_argument('--output', type=str, default='sentinel1_output', help='Output directory (default: sentinel1_output)')
parser.add_argument('--start_time', type=str, required=True, help='Start date in format YYYY-MM-DD (e.g., 2019-01-01)')
parser.add_argument('--end_time', type=str, required=True, help='End date in format YYYY-MM-DD (e.g., 2019-12-31)')
parser.add_argument('--chunksize', type=int, default=1024, help='Chunk size for processing (default: 256)')
parser.add_argument('--dask_workers', type=int, default=32, help='Number of Dask workers to use (default: 1)')
parser.add_argument('--worker_memory', type=int, default=32, help='Memory per Dask worker in GB (default: 32)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files instead of skipping them')
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

# Create or use specified temporary directory
if args.temp_dir:
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
else:
    # Create a temp dir that will persist across the run
    temp_dir = tempfile.mkdtemp(prefix="sentinel1_")
logger.info(f"Using temporary directory: {temp_dir}")

# Custom exceptions
class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

class DaskClusterError(Exception):
    """Exception raised when there's a problem with the Dask cluster that requires restart"""
    pass

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
    """
    # Set up local Dask client
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
        
        return client
        
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
            return client
        except Exception as e2:
            logger.error(f"Failed to create any Dask client: {e2}")
            raise e2

def amplitude_to_db_numpy(amp, mask=None):
    """
    Convert amplitude to scaled dB values with better handling of invalid values
    NumPy version for processing smaller chunks directly
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
            clipped = np.clip(scaled, 0, 30000)
        
        # Assign to output array
        output[valid_mask] = clipped.astype(np.int16)
    
    # Apply external mask if provided
    if mask is not None:
        output = np.where(mask > 0, output, 0)
        
    return output

def process_item(client, item, src_crs, src_transform, src_shape, src_res, bbox, src_mask, temp_output_path=None):
    """Process a single item and save as TIFF with VV and VH bands, matching input TIFF geography"""
    orbit_state = item.properties.get("sat:orbit_state")
    date_str = item.properties.get("datetime").split("T")[0]
    
    # Generate a unique filename for this specific acquisition
    # If this is temporary output (for later mosaicking), use a unique ID
    if temp_output_path:
        output_path = temp_output_path
    else:
        filename = f"{date_str}_{orbit_state}.tiff"
        output_path = os.path.join(output_dir, filename)
    
    try:
        logger.debug(f"Processing item for {date_str}_{orbit_state} to {output_path}...")
        
        # Wrap STAC and data access operations in try-except to handle GDAL errors
        try:
            # Stack the data using stackstac
            ds = stackstac.stack(
                [item], 
                bounds_latlon=bbox, 
                epsg=int(src_crs.to_epsg()),
                resolution=min(src_res),
                chunksize=args.chunksize
            )
            
            # Check if bands exist
            if 'vv' not in ds.band.values or 'vh' not in ds.band.values:
                logger.warning(f"Data for {date_str}_{orbit_state} is missing required bands. Skipping.")
                return None
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
                return None
                
            # Get coordinate information
            x_coords = vv_data.x.values
            y_coords = vv_data.y.values
            
            # Verify we have enough coordinates to work with
            if len(x_coords) < 2 or len(y_coords) < 2:
                logger.warning(f"Insufficient coordinate data for {date_str}_{orbit_state}. Skipping.")
                return None
                
            x_res = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
            y_res = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)
        except Exception as e:
            logger.error(f"Error validating data for {date_str}_{orbit_state}: {e}")
            raise
        
        # Create a transform from the coordinates
        try:
            src_data_transform = rasterio.transform.from_origin(
                x_coords[0] - x_res/2,
                y_coords[0] - y_res/2,
                x_res,
                -y_res
            )
        except Exception as e:
            logger.error(f"Error creating transform for {date_str}_{orbit_state}: {e}")
            raise
        
        # Get CRS from the dataset attributes or use default
        try:
            if 'crs' in ds.attrs:
                src_data_crs = CRS.from_string(ds.attrs['crs'])
            else:
                src_data_crs = src_crs
        except Exception as e:
            logger.error(f"Error getting CRS for {date_str}_{orbit_state}: {e}")
            src_data_crs = src_crs  # Fall back to source CRS
        
        # We'll use a batch approach to avoid large graph sizes
        logger.debug(f"Reprojecting and processing data for {date_str}_{orbit_state}...")
        
        # Create output arrays
        vv_final = np.zeros(src_shape, dtype=np.int16)
        vh_final = np.zeros(src_shape, dtype=np.int16)
        
        # Create temporary arrays for reprojection
        vv_reproject = np.zeros(src_shape, dtype=np.float32)
        vh_reproject = np.zeros(src_shape, dtype=np.float32)
        
        # Reproject the data with error handling
        try:
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
        except Exception as e:
            logger.error(f"Error reprojecting data for {date_str}_{orbit_state}: {e}")
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
        
        # Process in smaller blocks to avoid large graphs
        # We'll use direct NumPy processing instead of Dask for this step
        logger.debug(f"Converting to dB and applying mask for {date_str}_{orbit_state}...")
        
        # Use with np.errstate to suppress warnings
        with np.errstate(invalid='ignore', divide='ignore'):
            vv_final = amplitude_to_db_numpy(vv_reproject, mask=src_mask)
            vh_final = amplitude_to_db_numpy(vh_reproject, mask=src_mask)
        
        # Write to TIFF with the same geography as the input TIFF
        try:
            logger.debug(f"Saving {date_str}_{orbit_state} to {output_path}...")
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src_shape[0],
                width=src_shape[1],
                count=2,  # Two bands: VV and VH
                dtype='int16',
                crs=src_crs,
                transform=src_transform,
                compress='lzw',  # Adding compression for smaller files
            ) as dst:
                # Write each band
                dst.write(vv_final, 1)  # VV
                dst.write(vh_final, 2)  # VH
                
                # Add metadata
                dst.update_tags(
                    TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    ORBIT_STATE=orbit_state,
                    DATE_ACQUIRED=date_str,
                    SATELLITE=item.properties.get("platform", ""),
                    SOURCE_TIFF=os.path.basename(args.input_tiff),
                    DESCRIPTION="Sentinel-1 SAR data (VV/VH). Band 1: VV, Band 2: VH. Values are amplitude converted to dB, shifted by +50, scaled by 200, and stored as int16. Areas masked in source TIFF are set to 0."
                )
        except Exception as e:
            logger.error(f"Error writing TIFF for {date_str}_{orbit_state}: {e}")
            # Try to remove failed file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
            
        return output_path
    
    except DaskClusterError:
        # Propagate DaskClusterError so it can be handled at a higher level
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {date_str}_{orbit_state}: {str(e)}", exc_info=True)
        raise

@retry_with_backoff(exceptions=(Exception,), max_retries=2, timeout_seconds=300)
def mosaic_tiffs(tiff_paths, output_path, src_crs, src_transform, src_shape, date_str, orbit_state):
    """
    Mosaic multiple TIFFs into a single output TIFF
    """
    try:
        # Open all source TIFFs
        src_files = []
        for path in tiff_paths:
            if os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logger.warning(f"Failed to open {path} for mosaicking: {e}")
        
        if not src_files:
            logger.warning(f"No valid files to mosaic for {date_str}_{orbit_state}")
            return None
        
        # Perform mosaic operation with error handling
        try:
            logger.info(f"Mosaicking {len(src_files)} files for {date_str}_{orbit_state}...")
            mosaic_data, out_transform = merge(src_files, nodata=0)
        except Exception as e:
            logger.error(f"Error during merge operation for {date_str}_{orbit_state}: {e}")
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
        if mosaic_data.shape[0] < 2:
            logger.error(f"Mosaic data has incorrect number of bands ({mosaic_data.shape[0]}) for {date_str}_{orbit_state}")
            return None
        
        # Write the mosaic to the output file with error handling
        try:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src_shape[0],
                width=src_shape[1],
                count=2,  # Two bands: VV and VH
                dtype='int16',
                crs=src_crs,
                transform=src_transform,
                compress='lzw',  # Adding compression for smaller files
            ) as dst:
                # Write each band
                dst.write(mosaic_data[0], 1)  # VV
                dst.write(mosaic_data[1], 2)  # VH
                
                # Add metadata
                dst.update_tags(
                    TIFFTAG_DATETIME=datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    ORBIT_STATE=orbit_state,
                    DATE_ACQUIRED=date_str,
                    MOSAIC_SOURCE_COUNT=len(src_files),
                    SOURCE_TIFF=os.path.basename(args.input_tiff),
                    DESCRIPTION="Mosaicked Sentinel-1 SAR data (VV/VH). Band 1: VV, Band 2: VH. Values are amplitude converted to dB, shifted by +50, scaled by 200, and stored as int16. Areas masked in source TIFF are set to 0."
                )
        except Exception as e:
            logger.error(f"Error writing mosaic file for {date_str}_{orbit_state}: {e}")
            # Try to remove failed file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
        
        logger.info(f"Successfully created mosaic for {date_str}_{orbit_state}")
        return output_path
    
    except Exception as e:
        logger.error(f"Unexpected error creating mosaic for {date_str}_{orbit_state}: {str(e)}", exc_info=True)
        raise

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

def process_date_group(client, date_group, src_crs, src_transform, src_shape, src_res, bbox, src_mask):
    """
    Process a group of items for the same date and orbit state
    If there are multiple items, mosaic them
    Returns a tuple of (success, key, result_message)
    """
    try:
        (date_str, orbit_state), items = date_group
        key = f"{date_str}_{orbit_state}"
        final_filename = f"{key}.tiff"
        final_output_path = os.path.join(output_dir, final_filename)
        
        # Check if output already exists and handle according to overwrite flag
        if os.path.exists(final_output_path) and not args.overwrite:
            logger.info(f"File {final_filename} already exists, skipping.")
            return (True, key, f"Skipped {final_filename} (already exists)")
        
        # Create a subdirectory in temp_dir for this specific date-orbit group
        group_temp_dir = os.path.join(temp_dir, f"{date_str}_{orbit_state}")
        try:
            os.makedirs(group_temp_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create temp directory for {date_str}_{orbit_state}: {e}")
            # Continue using the parent temp directory if this fails
            group_temp_dir = temp_dir
        
        # If only one item, process directly
        if len(items) == 1:
            logger.info(f"Processing single observation for {date_str}_{orbit_state}...")
            result = process_item(client, items[0], src_crs, src_transform, src_shape, src_res, bbox, src_mask, final_output_path)
            if result:
                return (True, key, f"Successfully processed {final_filename}")
            else:
                return (False, key, f"Failed to process {final_filename}")
        
        # If multiple items, process each one to a temporary file then mosaic
        else:
            logger.info(f"Processing {len(items)} observations for {date_str}_{orbit_state}...")
            temp_files = []
            
            # Process each item to a temporary file
            for i, item in enumerate(items):
                temp_filename = f"{date_str}_{orbit_state}_{i}_{uuid.uuid4().hex[:8]}.tiff"
                temp_output_path = os.path.join(group_temp_dir, temp_filename)
                
                try:
                    result = process_item(client, item, src_crs, src_transform, src_shape, src_res, bbox, src_mask, temp_output_path)
                    if result:
                        temp_files.append(result)
                except DaskClusterError:
                    # Propagate DaskClusterError so it triggers a client restart
                    raise
                except Exception as e:
                    logger.error(f"Error processing item {i} for {date_str}_{orbit_state}: {e}")
                    # Continue processing other items even if one fails
            
            # Now mosaic the temporary files
            if temp_files:
                # If there's only one valid temp file, just rename it
                if len(temp_files) == 1:
                    logger.info(f"Only one valid file for {date_str}_{orbit_state}, using it directly.")
                    try:
                        shutil.copy2(temp_files[0], final_output_path)
                        os.remove(temp_files[0])
                        return (True, key, f"Successfully copied single file for {final_filename}")
                    except Exception as e:
                        logger.error(f"Error copying single file for {date_str}_{orbit_state}: {e}")
                        return (False, key, f"Failed to copy single file for {final_filename}")
                
                # Multiple valid files need mosaicking
                try:
                    mosaic_result = mosaic_tiffs(temp_files, final_output_path, src_crs, src_transform, src_shape, date_str, orbit_state)
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
                    
                    if mosaic_result:
                        return (True, key, f"Successfully mosaicked {len(temp_files)} files for {final_filename}")
                    else:
                        return (False, key, f"Failed to create mosaic for {final_filename}")
                except Exception as e:
                    logger.error(f"Error mosaicking files for {date_str}_{orbit_state}: {e}")
                    return (False, key, f"Failed to mosaic files for {final_filename}: {e}")
            else:
                return (False, key, f"No valid data to mosaic for {final_filename}")
    
    except DaskClusterError:
        # This is a special error that indicates we need to restart the Dask client
        # Propagate it up to be handled by the batch processor
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error processing group {date_str}_{orbit_state}: {e}", exc_info=True)
        return (False, key, f"Error processing {final_filename}: {str(e)}")

def process_orbit_state(orbit_state, items, src_crs, src_transform, src_shape, src_res, bbox, src_mask):
    """
    Process items for a specific orbit state (ascending or descending)
    """
    # Filter items by orbit state
    orbit_items = [item for item in items if item.properties.get("sat:orbit_state") == orbit_state]
    if not orbit_items:
        logger.info(f"No {orbit_state} orbit items to process")
        return [], []
    
    logger.info(f"Processing {len(orbit_items)} {orbit_state} orbit items")
    
    # Group items by date
    grouped_items = group_items_by_date(orbit_items)
    logger.info(f"Grouped into {len(grouped_items)} unique date-orbit combinations")
    
    # Print statistics
    print_date_statistics(grouped_items)
    
    # Check if there's anything to process
    if not grouped_items:
        logger.info(f"No {orbit_state} groups to process")
        return [], []
    
    # Convert to list and sort by date for more predictable processing
    groups_list = sorted(grouped_items.items(), key=lambda x: x[0][0])
    
    # Process groups
    results = []
    completed_keys = []
    failed_keys = []
    total_start_time = datetime.datetime.now()
    
    # Set up a Dask client
    client = setup_dask_client()
    
    try:
        # Log system status
        log_system_status()
        
        # Set up a ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all the tasks
            future_to_group = {
                executor.submit(
                    process_date_group, client, group, src_crs, src_transform, src_shape, src_res, bbox, src_mask
                ): group for group in groups_list
            }
            
            # Process as they complete
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_group), 
                                   total=len(future_to_group), desc=f"Processing {orbit_state} items"):
                group = future_to_group[future]
                (date_str, orbit_state), _ = group
                key = f"{date_str}_{orbit_state}"
                
                try:
                    success, result_key, message = future.result()
                    results.append((success, result_key, message))
                    
                    # Update completed/failed lists
                    if success:
                        if key not in completed_keys:
                            completed_keys.append(key)
                        if key in failed_keys:
                            failed_keys.remove(key)
                    else:
                        if key not in failed_keys:
                            failed_keys.append(key)
                    
                except DaskClusterError as e:
                    # Handle Dask cluster issues
                    error_msg = f"Group {date_str}_{orbit_state} encountered Dask cluster error: {e}"
                    logger.error(error_msg)
                    results.append((False, key, error_msg))
                    
                    # Add to failed list
                    if key not in failed_keys:
                        failed_keys.append(key)
                    
                    # Restart the client
                    try:
                        client.close()
                    except:
                        pass
                    
                    logger.info("Restarting Dask client due to errors...")
                    client = setup_dask_client()
                    
                except Exception as exc:
                    error_msg = f"Group {date_str}_{orbit_state} generated an unhandled exception: {exc}"
                    logger.error(error_msg, exc_info=True)
                    results.append((False, key, error_msg))
                    
                    # Add to failed list
                    if key not in failed_keys:
                        failed_keys.append(key)
                
                # Periodically log system status
                if len(results) % 10 == 0:
                    log_system_status()
    
    finally:
        # Close the Dask client
        try:
            client.close()
            logger.info("Closed Dask client")
        except:
            logger.warning("Error closing Dask client")
    
    # Calculate and log final statistics
    successes = sum(1 for success, _, _ in results if success)
    failures = len(results) - successes
    
    total_duration = datetime.datetime.now() - total_start_time
    logger.info(f"\n===== {orbit_state.upper()} ORBIT PROCESSING SUMMARY =====")
    logger.info(f"Total processing time: {total_duration}")
    logger.info(f"Groups processed: {len(results)}")
    logger.info(f"Successful: {successes}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Success rate: {(successes/len(results))*100 if len(results) > 0 else 0:.1f}%")
    
    return completed_keys, failed_keys

def main():
    # Create a crash log file to catch unhandled exceptions
    sys.excepthook = lambda exctype, value, traceback: logger.critical(
        "Unhandled exception", exc_info=(exctype, value, traceback)
    )
    
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
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Date range: {args.start_time} to {args.end_time}")
    logger.info(f"Orbit state: {args.orbit_state}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Dask workers: {args.dask_workers}")
    logger.info(f"Worker memory: {args.worker_memory} GB")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Timeout: {args.timeout} seconds")
    logger.info(f"Overwrite existing: {args.overwrite}")
    
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
        src_res = (src.transform.a, -src.transform.e)  # (width, height) resolution
        
        # Read the data to create a mask (0 = invalid, >0 = valid)
        src_data = src.read(1)  # Read first band
        src_mask = (src_data > 0).astype(np.uint8)
        
        logger.info(f"Input TIFF CRS: {src_crs}")
        logger.info(f"Input TIFF bounds: {src_bounds}")
        logger.info(f"Input TIFF shape: {src_shape}")
        logger.info(f"Input TIFF resolution: {src_res}")
        
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
    
    logger.info(f"Starting Sentinel-1 data download for date range {args.start_time} to {args.end_time}")
    
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
    date_range = f"{args.start_time}/{args.end_time}"
    logger.info(f"Searching for Sentinel-1 data for date range: {date_range}")
    
    try:
        search = catalog.search(
            collections=["sentinel-1-rtc"], 
            bbox=bbox,
            datetime=date_range
        )
        items = search.item_collection()
        logger.info(f"Found {len(items)} items")
    except Exception as e:
        logger.error(f"Failed to search for Sentinel-1 data: {e}")
        sys.exit(1)

    # Process based on selected orbit state
    if args.orbit_state in ["ascending", "both"]:
        logger.info("\n=== PROCESSING ASCENDING ORBIT DATA ===")
        ascending_completed, ascending_failed = process_orbit_state(
            "ascending", items, src_crs, src_transform, src_shape, src_res, bbox, src_mask
        )
    else:
        logger.info("Skipping ascending orbit data processing")
        ascending_completed, ascending_failed = [], []
    
    if args.orbit_state in ["descending", "both"]:
        logger.info("\n=== PROCESSING DESCENDING ORBIT DATA ===")
        descending_completed, descending_failed = process_orbit_state(
            "descending", items, src_crs, src_transform, src_shape, src_res, bbox, src_mask
        )
    else:
        logger.info("Skipping descending orbit data processing")
        descending_completed, descending_failed = [], []
    
    # Print final summary
    logger.info(f"\n===== DOWNLOAD SUMMARY =====")
    logger.info(f"Total files processed successfully: {len(ascending_completed) + len(descending_completed)}")
    logger.info(f"  Ascending orbits: {len(ascending_completed)}")
    logger.info(f"  Descending orbits: {len(descending_completed)}")
    logger.info(f"Total files failed: {len(ascending_failed) + len(descending_failed)}")
    logger.info(f"  Ascending orbits: {len(ascending_failed)}")
    logger.info(f"  Descending orbits: {len(descending_failed)}")
    logger.info(f"All files are saved in: {os.path.abspath(output_dir)}")
    logger.info(f"============================")
    
    # Generate a list of failed files for reference
    if ascending_failed or descending_failed:
        date_range_str = f"{args.start_time}_to_{args.end_time}"
        failed_file = os.path.join(output_dir, f"failed_downloads_{date_range_str}.txt")
        try:
            with open(failed_file, 'w') as f:
                f.write("# Failed downloads\n")
                for key in ascending_failed:
                    f.write(f"{key} (ascending)\n")
                for key in descending_failed:
                    f.write(f"{key} (descending)\n")
            logger.info(f"List of failed downloads saved to: {failed_file}")
        except Exception as e:
            logger.error(f"Failed to save list of failed downloads: {e}")
    
    # Clean up temporary directory if we created one
    if not args.temp_dir:
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    main()