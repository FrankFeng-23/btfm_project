#!/usr/bin/env python3
"""
s2_fast_processor.py — Sentinel-2 L2A Fast Download & ROI Mosaic (Dynamic Window Optimized Version)
Updated: 2025-07-18
Supports dynamic window processing with enhanced retry mechanisms, timeout control, and memory optimization
Features: Optimized for short processing windows, improved error handling, and robust retry logic
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
import concurrent.futures
import uuid
import tempfile
import shutil
import gc
import random

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
import pystac_client, planetary_computer, stackstac

import dask
from dask.distributed import Client, LocalCluster, performance_report, wait

# ▶ distributed version compatibility
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# ─── Constants ──────────────────────────────────────────────────────────────────────
BAND_MAPPING = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}
S2_BANDS        = list(BAND_MAPPING.keys())
BASELINE_CUTOFF = datetime.datetime(2022, 1, 25)
BASELINE_OFFSET = 1000

# SCL invalid value list (cloud-free/shadow-free/non-water etc. are valid)
SCL_INVALID = {0, 1, 2, 3, 8, 9, np.nan}

# SCL value descriptions for logging
SCL_DESCRIPTIONS = {
    0: "No data",
    1: "Saturated or defective",
    2: "Dark area pixels",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud medium probability",
    9: "Cloud high probability",
    10: "Thin cirrus",
    11: "Snow"
}

# Valid coverage threshold (skip processing below this value)
MIN_VALID_COVERAGE = 5.0  # percentage

# Temporary file directory setting (default uses system temp directory)
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Updated timeout settings for window processing
WINDOW_TIMEOUT = 480      # Overall window processing timeout
DAY_TIMEOUT = 200         # Single day processing timeout
ITEM_TIMEOUT = 180         # Single item processing timeout
BAND_TIMEOUT = 120         # Single band processing timeout
SCL_BAND_TIMEOUT = 60     # SCL band processing timeout
STAC_SEARCH_TIMEOUT = 10  # STAC search timeout

# Enhanced retry configuration
SCL_MAX_ATTEMPTS = 5         # SCL processing maximum attempts
MAX_RETRIES = 10            # Network request retry attempts (10 times for STAC search)
RETRY_BACKOFF_FACTOR = 1.2  # Exponential backoff factor
INITIAL_RETRY_DELAY = 3     # Initial retry delay

# Parallel processing configuration
DEFAULT_MAX_WORKERS = 1     # Default number of threads for processing items

# Memory optimization for window processing
MEMORY_CONSERVATIVE_MODE = True  # Enable conservative memory usage
MAX_CHUNK_SIZE = 512      # Smaller chunk size for windows
LIGHTWEIGHT_PROCESSING = True    # Enable lightweight processing mode

# ─── Processing result status enums ─────────────────────────────────────────────
class ProcessingStatus:
    """SCL processing status enumeration"""
    SUCCESS = "success"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    PROCESSING_ERROR = "processing_error"

# ─── Timeout Control ──────────────────────────────────────────────────────────────────
class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """Timeout context manager with thread safety"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Operation timeout ({seconds} seconds)")
    
    # Check if running in main thread (Unix signals can only be handled in main thread)
    import threading
    if threading.current_thread() is not threading.main_thread():
        # If not main thread, just yield without setting signal
        yield
        return
    
    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ─── Enhanced Retry Mechanism ─────────────────────────────────────────────────────────
def retry_with_backoff(func, max_attempts=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY, 
                      backoff_factor=RETRY_BACKOFF_FACTOR, timeout_per_attempt=None, 
                      partition_id="unknown"):
    """
    Enhanced retry mechanism with exponential backoff and timeout per attempt
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if timeout_per_attempt:
                with timeout_handler(timeout_per_attempt):
                    return func()
            else:
                return func()
                
        except Exception as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                # Last attempt failed
                logging.error(f"[{partition_id}] Operation failed after {max_attempts} attempts: {e}")
                break
            
            # Calculate delay with jitter
            delay = initial_delay * (backoff_factor ** attempt)
            jitter = random.uniform(0.8, 1.2)
            actual_delay = min(60, delay * jitter)  # Cap at 60 seconds
            
            logging.warning(f"[{partition_id}] Attempt {attempt + 1}/{max_attempts} failed: {e}, "
                          f"retrying in {actual_delay:.1f}s...")
            time.sleep(actual_delay)
    
    # All attempts failed
    raise last_exception

# ─── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    P = argparse.ArgumentParser("Fast Sentinel-2 L2A Processor (Dynamic Window Optimized Edition)")
    P.add_argument("--input_tiff",   required=True, help="ROI mask or template raster")
    P.add_argument("--start_date",   required=True, help="Start date (YYYY-MM-DD[THH:MM:SS]) - inclusive")
    P.add_argument("--end_date",     required=True, help="End date (YYYY-MM-DD[THH:MM:SS]) - inclusive")
    P.add_argument("--output",       default="sentinel2_output", help="Output directory")
    P.add_argument("--max_cloud",    type=float, default=90, help="Maximum cloud cover percentage")
    P.add_argument("--dask_workers", type=int,   default=1, help="Number of Dask workers for this window")
    P.add_argument("--worker_memory",type=int,   default=4, help="Memory per worker in GB")
    P.add_argument("--chunksize",    type=int,   default=MAX_CHUNK_SIZE, help="stackstac x/y chunk size")
    P.add_argument("--resolution",   type=float,   default=10.0, help="Output resolution (meters)")
    P.add_argument("--overwrite",    action="store_true", help="Overwrite existing files")
    P.add_argument("--debug",        action="store_true", help="Output debug logs")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="Minimum valid pixel coverage percentage")
    P.add_argument("--partition_id", default="unknown",
                   help="Window ID (for log identification)")
    P.add_argument("--temp_dir",     default=TEMP_DIR,
                   help="Temporary file storage directory, default uses system temp directory")
    return P.parse_args()

# ─── logging ──────────────────────────────────────────────────────────────────
def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """Setup logging with window ID identifier"""
    fmt = f"%(asctime)s [{partition_id}] [%(levelname)s] %(message)s"
    lvl = logging.DEBUG if debug else logging.INFO
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(lvl)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(fmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (window-specific log file)
    file_handler = logging.FileHandler(
        out_dir / f"s2_{partition_id}_detail.log", 
        "a", 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_sys(partition_id: str):
    m = psutil.virtual_memory()
    logging.info(f"[{partition_id}] System info - CPU {os.cpu_count()} | "
                 f"RAM {m.total/1e9:.1f} GB (free {m.available/1e9:.1f} GB)")

def fmt_bbox(b):
    return f"{b[0]:.5f},{b[1]:.5f} ⇢ {b[2]:.5f},{b[3]:.5f}"

# ─── Dask ─────────────────────────────────────────────────────────────────────
def make_client(req_workers:int, req_mem:int, partition_id: str):
    """Create optimized Dask client for window processing"""
    total_mem = psutil.virtual_memory().total / 1e9
    
    # For window processing, use conservative settings
    workers = min(1, req_workers)  # Use single worker for window processing
    threads = 2  # Fewer threads to reduce overhead
    req_mem = min(4, req_mem)  # Limit memory to 4GB per worker
    
    logging.info(f"[{partition_id}] Creating Dask client with {workers} workers, {threads} threads, {req_mem}GB memory")
    
    # Automatically assign dashboard port for different windows
    port_base = 8780
    port_range = 100
    dashboard_port = port_base + (abs(hash(partition_id)) % port_range)
    
    # Configure Dask for window processing optimization
    dask_config = {
        "distributed.worker.memory.target": 0.60,
        "distributed.worker.memory.spill": 0.70,
        "distributed.worker.memory.pause": 0.85,
        "distributed.worker.memory.terminate": 0.95,
        "array.slicing.split_large_chunks": True,
        "optimization.fuse.active": True,
        "optimization.fuse.ave-width": 2,
        "distributed.scheduler.work-stealing": False,  # Disable for single worker
        "distributed.worker.connections.incoming": 10,
        "distributed.worker.connections.outgoing": 10,
        "distributed.comm.retry.count": 3,
        "distributed.comm.retry.delay.min": "1s",
        "distributed.comm.retry.delay.max": "3s",
        "distributed.client.heartbeat": "10s",  # Faster heartbeat
        "distributed.comm.timeouts.connect": "30s",
        "distributed.comm.timeouts.tcp": "30s"
    }
    
    dask.config.set(dask_config)
    
    try:
        cluster = LocalCluster(
            n_workers         = workers,
            threads_per_worker= threads,
            processes         = True,
            memory_limit      = f"{req_mem}GB",
            dashboard_address = f":{dashboard_port}",
            silence_logs      = "ERROR",
            death_timeout     = "30s",
        )
        
        cli = Client(cluster, asynchronous=False, timeout="15s")
        logging.info(f"[{partition_id}] Dask dashboard → {cli.dashboard_link}")
        return cli
    except Exception as e:
        logging.error(f"[{partition_id}] Failed to create Dask client: {e}")
        return None

# ─── ROI & Mask ────────────────────────────────────────────────────────────────
def load_roi(tiff: Path, partition_id: str):
    """Load ROI data with memory optimization for window processing"""
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        
        # Read mask and convert to 1-bit data to save memory
        mask_np = (src.read(1) > 0).astype(np.uint8)
        
    # Check ROI size and log info
    roi_size_mb = (mask_np.size * mask_np.itemsize) / (1024 * 1024)
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']} ({roi_size_mb:.2f} MB)")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    
    return tpl, bbox_proj, bbox_ll, mask_np

def mask_to_xr(mask_np, tpl):
    """Convert mask to xarray object"""
    da = xr.DataArray(mask_np, dims=("y", "x"))
    return da.rio.write_crs(tpl["crs"]).rio.write_transform(tpl["transform"])

# ─── STAC ─────────────────────────────────────────────────────────────────────
def search_items(bbox_ll, date_range:str, max_cloud, partition_id: str):
    """
    Search STAC items with enhanced exception handling and retry logic for window processing
    Critical: If this fails after all retries, the entire window should be marked as failed
    """
    # Parse start and end times
    start_date, end_date = date_range.split("/")
    
    # Parse end time and add one second to ensure end time is included
    try:
        end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00').replace(' ', 'T'))
        end_dt_plus = end_dt + datetime.timedelta(seconds=1)
        search_date_range = f"{start_date}/{end_dt_plus.isoformat()}"
    except ValueError:
        # If time format parsing fails, use original date range
        logging.warning(f"[{partition_id}] Unable to parse end date format, using original range: {date_range}")
        search_date_range = date_range
    
    logging.info(f"[{partition_id}] STAC search date range: {search_date_range}")
    logging.info(f"[{partition_id}] STAC search will attempt {MAX_RETRIES} times with {STAC_SEARCH_TIMEOUT}s timeout each")
    
    def _search_stac():
        cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)
        q = cat.search(collections=["sentinel-2-l2a"],
                      bbox=bbox_ll, datetime=search_date_range,
                      query={"eo:cloud_cover": {"lt": max_cloud}})
        items = list(q.get_items())
        logging.info(f"[{partition_id}] STAC found {len(items)} items (cloud < {max_cloud}%)")
        if items:
            b = np.array([it.bbox for it in items])
            union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
            logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
        return items
    
    # Use retry mechanism for STAC search - this is critical, failure here means window failure
    try:
        items = retry_with_backoff(
            _search_stac,
            max_attempts=MAX_RETRIES,
            timeout_per_attempt=STAC_SEARCH_TIMEOUT,
            partition_id=partition_id
        )
        
        if not items:
            logging.warning(f"[{partition_id}] STAC search returned no items, but search was successful")
        
        return items
        
    except Exception as e:
        logging.error(f"[{partition_id}] CRITICAL: STAC search failed after all {MAX_RETRIES} attempts: {e}")
        logging.error(f"[{partition_id}] This window will be marked as FAILED due to STAC search failure")
        # Re-raise the exception to ensure the window is marked as failed
        raise Exception(f"STAC search failed after {MAX_RETRIES} attempts: {e}")

def group_by_date(items, partition_id: str):
    """Group items by date"""
    g = {}
    for it in items:
        d = it.properties["datetime"][:10]
        g.setdefault(d, []).append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} observation days")
    return dict(sorted(g.items()))

# ─── baseline correction ─────────────────────────────────────────────────────────────
def harmonize_arr(arr: np.ndarray, date_key:str):
    """Perform Baseline correction"""
    if datetime.datetime.strptime(date_key, "%Y-%m-%d") > BASELINE_CUTOFF:
        # Handle NaN values to avoid warnings
        valid_mask = ~np.isnan(arr) & (arr >= BASELINE_OFFSET)
        np.subtract(arr, BASELINE_OFFSET, out=arr, where=valid_mask)
    return arr

# ─── Memory Check ──────────────────────────────────────────────────────────────────
def check_memory_requirements(shape, dtype=np.uint16):
    """Check if array memory requirements are reasonable for window processing"""
    try:
        # Calculate required memory (GB)
        element_size = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        memory_gb = (total_elements * element_size) / (1024**3)
        
        logging.debug(f"Calculating memory requirements: shape{shape}, type{dtype}, size{memory_gb:.2f}GB")
        
        # Get current available memory
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        # Use more conservative threshold for window processing
        threshold_gb = min(available_gb * 0.3, 8)  # Not exceeding 8GB and 30% of available
        
        if memory_gb > threshold_gb:
            logging.warning(f"⚠️  Memory requirement {memory_gb:.2f}GB exceeds available threshold {threshold_gb:.2f}GB, skipping processing")
            return False
        return True
    except (OverflowError, ValueError) as e:
        logging.warning(f"⚠️  Memory calculation error: {e}, skipping processing")
        return False

# ─── GeoTIFF Write ──────────────────────────────────────────────────────────────
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    """Write GeoTIFF with optimized compression and configuration, adding metadata"""
    # Handle NaN values
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
        
    profile = dict(driver="GTiff", dtype=dtype, count=1,
                   width=tpl["width"], height=tpl["height"],
                   crs=tpl["crs"], transform=tpl["transform"],
                   tiled=True,
                   blockxsize=256, blockysize=256,
                   nodata=0)
    
    # Write file
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        # Add metadata
        if metadata:
            dst.update_tags(**metadata)

# ─── Validate TIFF ─────────────────────────────────────────────────────────────
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """Validate if TIFF file is valid"""
    try:
        with rasterio.open(file_path) as src:
            # Check basic attributes
            if src.shape != expected_shape:
                logging.warning(f"Validation failed: {file_path} shape mismatch. Expected {expected_shape}, got {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"Validation failed: {file_path} CRS mismatch. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check data existence (through statistics, avoid reading entire array)
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"Validation failed: {file_path} all bands are zero")
                return False
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            logging.debug(f"TIFF validation passed: {file_path}, shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"Error validating TIFF {file_path}: {e}")
        return False

# ─── SCL Quality Assessment ─────────────────────────────────────────────────────────
def is_valid_scl(scl_arr):
    """Determine if SCL value is valid observation (non-cloud/shadow/water etc.)"""
    # Vectorized operation: fill nan with 0, then check if in invalid values list
    return ~np.isin(np.nan_to_num(scl_arr, nan=0), list(SCL_INVALID - {np.nan}))

def process_scl(scl_arr, roi_mask, partition_id="unknown"):
    """
    Process SCL data, highly optimized vectorized version
    Returns:
        valid_mask: validity mask (boolean array)
        tile_selection: selected tile index array (for subsequent band processing)
        valid_pct: valid coverage percentage
    """
    # Get array shape and handle single tile and multi-tile cases
    if len(scl_arr.shape) == 3:
        n_tiles, scl_height, scl_width = scl_arr.shape
    else:
        scl_height, scl_width = scl_arr.shape
        n_tiles = 1
        scl_arr = scl_arr.reshape(1, scl_height, scl_width)
    
    # Ensure ROI mask is boolean type
    roi_mask = roi_mask.astype(bool)
    roi_height, roi_width = roi_mask.shape
    
    # Check shape matching, if not matching, adjust SCL data
    if scl_height != roi_height or scl_width != roi_width:
        logging.warning(f"[{partition_id}] Detected shape mismatch: SCL shape{(scl_height, scl_width)} differs from ROI shape{(roi_height, roi_width)}")
        
        # Determine final shape to use (take smaller values to avoid index overflow)
        use_height = min(scl_height, roi_height)
        use_width = min(scl_width, roi_width)
        
        logging.info(f"[{partition_id}] Using common area: {use_height}x{use_width}")
        
        # Crop data to common area
        scl_arr = scl_arr[:, :use_height, :use_width]
        roi_mask = roi_mask[:use_height, :use_width]
    
    # Calculate valid mask for each tile - vectorized operation
    valid_mask = is_valid_scl(scl_arr)
    
    # Calculate valid pixel positions and ROI pixel count
    roi_pixel_count = np.sum(roi_mask)
    
    # Create tile selection array, initialize to -1 (no valid tile)
    tile_selection = np.full(roi_mask.shape, -1, dtype=np.int8)
    
    # Optimized vectorized algorithm for selecting first valid tile
    # 1. Create a boolean mask indicating which pixels have been assigned a valid tile
    assigned = np.zeros(roi_mask.shape, dtype=bool)
    
    # 2. Loop through each tile (this loop cannot be completely avoided as we need to select first valid tile in order)
    for tile_idx in range(n_tiles):
        # 2.1 Find pixels that are valid in current tile and within ROI and not yet assigned
        # valid_mask[tile_idx] is the valid mask for current tile
        # roi_mask is the ROI area mask
        # ~assigned is mask for pixels not yet assigned a tile
        current_valid = valid_mask[tile_idx] & roi_mask & ~assigned
        
        # 2.2 Assign current tile_idx to current valid pixels in tile_selection
        tile_selection[current_valid] = tile_idx
        
        # 2.3 Update assigned flag
        assigned |= current_valid
    
    # Calculate valid coverage - number of pixels assigned a valid tile divided by total ROI pixels
    valid_pixel_count = np.sum(tile_selection >= 0)
    valid_pct = 100.0 * valid_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
    
    logging.info(f"[{partition_id}] SCL processing result: Valid pixels in ROI {valid_pixel_count}/{roi_pixel_count}, coverage {valid_pct:.2f}%")
    
    return valid_mask, tile_selection, valid_pct

def create_scl_mosaic(scl_arr, tile_selection, roi_mask, target_shape, date_key=None, partition_id="unknown"):
    """
    Create SCL mosaic based on tile selection results, preserving original SCL values - vectorized version
    """
    try:
        # Get shape information
        if len(scl_arr.shape) == 3:
            n_tiles, arr_height, arr_width = scl_arr.shape
        else:
            arr_height, arr_width = scl_arr.shape
            n_tiles = 1
            scl_arr = scl_arr.reshape(1, arr_height, arr_width)
        
        target_height, target_width = target_shape
        
        # Create target size result array (initial value 0, meaning no data)
        result = np.zeros(target_shape, dtype=np.uint8)
        
        # Determine common area size
        common_height = min(arr_height, roi_mask.shape[0], target_height, tile_selection.shape[0])
        common_width = min(arr_width, roi_mask.shape[1], target_width, tile_selection.shape[1])
        
        # Crop to common area
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # Vectorized implementation: use valid tile indices within ROI range to select SCL values
        # 1. Create a valid ROI mask indicating which pixels need processing
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. Get coordinates of valid ROI
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. Get tile indices corresponding to these coordinates
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. Create mapping to extract corresponding values from scl_arr
            result[y_coords, x_coords] = scl_arr[tile_indices, y_coords, x_coords]
        
        # Count SCL value distribution
        unique_values, unique_counts = np.unique(result, return_counts=True)
        value_counts = dict(zip(unique_values, unique_counts))
        
        # Log SCL value distribution
        if date_key:
            total_pixels = np.sum(roi_mask)
            logging.info(f"[{partition_id}] SCL value distribution statistics ({date_key}):")
            for val in sorted(value_counts.keys()):
                count = value_counts[val]
                desc = SCL_DESCRIPTIONS.get(val, "Unknown")
                percent = 100 * count / total_pixels if total_pixels > 0 else 0
                logging.info(f"[{partition_id}]   SCL value {val} ({desc}): {count} pixels ({percent:.2f}%)")
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] SCL mosaic creation failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── Smart Mosaic ──────────────────────────────────────────────────────────────────
def smart_mosaic(data_arr, tile_selection, roi_mask, partition_id="unknown"):
    """
    Smart mosaic based on tile_selection, using best tiles determined in SCL processing - vectorized version
    """
    try:
        # Single tile case return directly
        if len(data_arr.shape) < 3 or data_arr.shape[0] == 1:
            result = data_arr[0] if len(data_arr.shape) == 3 else data_arr
            
            # Check shape matching
            if result.shape != roi_mask.shape:
                logging.debug(f"[{partition_id}] Single tile mosaic shape mismatch: data{result.shape}, ROI mask{roi_mask.shape}")
                
                # Determine common size
                common_height = min(result.shape[0], roi_mask.shape[0])
                common_width = min(result.shape[1], roi_mask.shape[1])
                
                # Create target size result array
                final_result = np.zeros(roi_mask.shape, dtype=result.dtype)
                
                # Crop data and mask
                result_cropped = result[:common_height, :common_width] 
                roi_mask_cropped = roi_mask[:common_height, :common_width]
                
                # Apply ROI mask - vectorized operation
                final_result[:common_height, :common_width] = result_cropped * roi_mask_cropped
                return final_result
            
            # Apply ROI mask - vectorized operation
            return result * roi_mask
        
        # Get dimensions
        n_tiles, data_height, data_width = data_arr.shape
        
        # Ensure ROI mask is boolean type
        roi_mask = roi_mask.astype(bool)
        
        # Create output array (using original ROI size)
        result = np.zeros(roi_mask.shape, dtype=data_arr.dtype)
        
        # Determine common area size
        common_height = min(data_height, roi_mask.shape[0], tile_selection.shape[0])
        common_width = min(data_width, roi_mask.shape[1], tile_selection.shape[1])
        
        # Crop to common area
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # Vectorized implementation: use valid tile indices within ROI range to select data values
        # 1. Create a valid ROI mask indicating which pixels need processing
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. Get coordinates of valid ROI
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. Get tile indices corresponding to these coordinates
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. Extract corresponding values from data_arr
            result[y_coords, x_coords] = data_arr[tile_indices, y_coords, x_coords]
        
        # For ROI pixels without valid tiles, use random selection logic
        # 1. Create mask for pixels within ROI but without valid tiles
        invalid_roi = (roi_crop & (tile_sel_crop < 0))
        
        if np.any(invalid_roi):
            # 2. Get these coordinates
            y_coords_invalid, x_coords_invalid = np.where(invalid_roi)
            
            # 3. Randomly select a tile for each invalid pixel
            random_tiles = np.random.randint(0, n_tiles, size=len(y_coords_invalid))
            
            # 4. Fill these pixels using randomly selected tiles
            result[y_coords_invalid, x_coords_invalid] = data_arr[random_tiles, y_coords_invalid, x_coords_invalid]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] Smart mosaic failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── Process Single Band ───────────────────────────────────────────────────────────
def process_band(items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
               res, chunksize, out_path, partition_id="unknown", max_retries=3):
    """Process single band with enhanced retry and error handling for window processing"""
    
    # SCL band has been processed in quality assessment phase, skip directly
    if band_name == "SCL":
        logging.info(f"[{partition_id}]     Band {band_name} already processed in quality assessment phase, skipping")
        return True

    logging.info(f"[{partition_id}]     Processing band {band_name}")
   
    # Check if output path already exists
    if out_path.exists():
        if validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]     {band_name} valid file already exists, skipping")
            return True
        else:
            logging.warning(f"[{partition_id}]     {band_name} file exists but invalid, reprocessing")
            out_path.unlink()
   
    # Enhanced retry mechanism
    def _process_band_operation():
        t0 = time.time()
        
        # Use stackstac.stack to load single band
        assets = [band_name]
        
        da = stackstac.stack(
            items=items,
            assets=assets,
            resolution=res,
            epsg=tpl["crs"].to_epsg(),
            bounds=bbox_proj,
            chunksize=chunksize,
            rescale=False,
            resampling=Resampling.nearest
        )
        
        # Flatten useless dimensions but keep multi-item dimension
        item_dim = None
        for dim in da.dims:
            if dim not in ('band', 'x', 'y'):
                if da.sizes[dim] > 1:
                    item_dim = dim  # Find multi-item dimension
                elif da.sizes[dim] == 1:
                    da = da.squeeze(dim, drop=True)
        
        # Extract band data
        band_da = da.sel(band=band_name)
        
        # Convert to numpy array for processing
        if item_dim:
            # Multiple items case
            band_arr = band_da.values
            
            # Log and output shape info for debugging
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(f"[{partition_id}]     {band_name} array shape: {band_arr.shape}, ROI shape: {mask_np.shape}")
            
            # Check if array size is reasonable
            if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                logging.warning(f"[{partition_id}]     {band_name} array too large, skipping")
                return False
            
            # Apply SCL-based smart mosaic - vectorized version
            if tile_selection is not None:
                arr = smart_mosaic(band_arr, tile_selection, mask_np, partition_id)
            else:
                # If no valid tile_selection, use random selection
                logging.warning(f"[{partition_id}]     No valid tile_selection, using random tile selection")
                n_tiles = band_arr.shape[0]
                
                # Create random selection matrix - vectorized implementation
                random_selection = np.random.randint(0, n_tiles, size=mask_np.shape)
                arr = np.zeros(mask_np.shape, dtype=band_arr.dtype)
                
                # Calculate common area
                common_height = min(mask_np.shape[0], band_arr.shape[1])
                common_width = min(mask_np.shape[1], band_arr.shape[2])
                
                # Create index matrix
                for i in range(n_tiles):
                    # Find pixels that should use current tile
                    use_tile_i = (random_selection[:common_height, :common_width] == i) & (mask_np[:common_height, :common_width] > 0)
                    if np.any(use_tile_i):
                        # Direct assignment using boolean indexing - vectorized operation
                        arr[:common_height, :common_width][use_tile_i] = band_arr[i, :common_height, :common_width][use_tile_i]
            
            # Check if output shape matches target
            if arr.shape != (tpl["height"], tpl["width"]):
                logging.warning(f"[{partition_id}]     Mosaic result shape {arr.shape} does not match target shape {(tpl['height'], tpl['width'])}, adjusting")
                # Create target size array
                full_arr = np.zeros((tpl["height"], tpl["width"]), dtype=arr.dtype)
                # Copy common area
                h = min(arr.shape[0], tpl["height"])
                w = min(arr.shape[1], tpl["width"])
                full_arr[:h, :w] = arr[:h, :w]
                arr = full_arr
            
            # Apply baseline correction
            harmonize_arr(arr, date_key)
            
        else:
            # Single item case
            band_arr = band_da.values
            
            # Log and output shape info for debugging
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(f"[{partition_id}]     {band_name} array shape: {band_arr.shape}, ROI shape: {mask_np.shape}")
            
            # Check if array size is reasonable
            if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                logging.warning(f"[{partition_id}]     {band_name} array too large, skipping")
                return False
            
            # Check shape matching
            if band_arr.shape != mask_np.shape:
                logging.warning(f"[{partition_id}]     Data shape {band_arr.shape} does not match ROI shape {mask_np.shape}, adjusting")
                # Create target size array
                arr = np.zeros((tpl["height"], tpl["width"]), dtype=band_arr.dtype)
                # Copy common area
                h = min(band_arr.shape[0], tpl["height"])
                w = min(band_arr.shape[1], tpl["width"])
                # Apply mask to common area - vectorized operation
                mask_crop = mask_np[:h, :w]
                arr[:h, :w] = band_arr[:h, :w] * mask_crop
            else:
                # Apply mask - vectorized operation
                arr = band_arr * mask_np
            
            # Apply baseline correction
            harmonize_arr(arr, date_key)
        
        # Create metadata
        metadata = {
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "DATE_ACQUIRED": date_key,
            "BAND_NAME": band_name,
            "ITEMS_COUNT": len(items)
        }
        
        # Write GeoTIFF
        dtype = "uint16"  # SCL has been skipped, only process other bands here
        write_tiff(arr, out_path, tpl, dtype, metadata)
        
        # Validate output file
        if not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]     ✗ Band {band_name} validation failed")
            if out_path.exists():
                out_path.unlink()
            raise Exception(f"Band {band_name} validation failed")
        
        logging.info(f"[{partition_id}]     ✓ {band_name:9s}  "
                    f"{os.path.getsize(out_path)/1e6:.2f} MB, time {time.time()-t0:.1f}s")
        
        return True
    
    # Use retry mechanism for band processing
    try:
        return retry_with_backoff(
            _process_band_operation,
            max_attempts=max_retries,
            timeout_per_attempt=BAND_TIMEOUT,
            partition_id=partition_id
        )
    except Exception as e:
        logging.error(f"[{partition_id}]     ✗ Band {band_name} processing failed after all retries: {e}")
        return False

# ─── Process SCL Assessment and Band Generation ───────────────────────────────────────────────────────────
def process_scl_assessment_and_generation(items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                                      min_coverage, out_root, overwrite, partition_id="unknown"):
    """
    Process SCL band with enhanced retry mechanism for window processing
    Returns: (status, valid_pct, tile_selection)
        status: ProcessingStatus.SUCCESS, ProcessingStatus.INSUFFICIENT_COVERAGE, or ProcessingStatus.PROCESSING_ERROR
        valid_pct: valid coverage percentage
        tile_selection: tile selection array for other bands
    """
    logging.info(f"[{partition_id}]   Processing SCL band for quality assessment and generating SCL output file")

    # Build SCL output path
    scl_out_name = BAND_MAPPING["SCL"]
    scl_dir = out_root / scl_out_name
    scl_dir.mkdir(parents=True, exist_ok=True)
    scl_out_path = scl_dir / f"{date_key}_mosaic.tiff"

    # Check if valid SCL file already exists
    if not overwrite and scl_out_path.exists():
        if validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]   SCL file already exists and is valid, reading file to check validity...")
            
            # Read existing SCL file and analyze its content
            try:
                with rasterio.open(scl_out_path) as src:
                    scl_data = src.read(1)
                    # Check SCL value distribution - vectorized operation
                    unique_values, unique_counts = np.unique(scl_data, return_counts=True)
                    value_counts = dict(zip(unique_values, unique_counts))
                    
                    # Log SCL value distribution
                    total_pixels = np.sum(mask_np)
                    logging.info(f"[{partition_id}]   Existing SCL file value distribution statistics ({date_key}):")
                    for val in sorted(value_counts.keys()):
                        count = value_counts[val]
                        desc = SCL_DESCRIPTIONS.get(val, "Unknown")
                        percent = 100 * count / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]     SCL value {val} ({desc}): {count} pixels ({percent:.2f}%)")
                    
                    # Check if only contains 0 and 4 (indicating old processing method) - vectorized check
                    if set(unique_values) == {0, 4} and unique_values.size == 2:
                        logging.warning(f"[{partition_id}]   Detected old version SCL file format (only contains 0 and 4), will regenerate")
                        scl_out_path.unlink()
                    else:
                        # Calculate valid coverage (non-SCL_INVALID pixels) - vectorized calculation
                        invalid_mask = np.isin(scl_data, list(SCL_INVALID - {np.nan}))
                        valid_pixels = np.sum(~invalid_mask & (scl_data > 0))
                        
                        valid_pct = 100 * valid_pixels / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]   Existing SCL file valid coverage: {valid_pct:.2f}%")
                        
                        # If coverage is sufficient, return success
                        if valid_pct >= min_coverage:
                            # Generate generic tile_selection for other bands
                            # For existing SCL files, we assume all non-0 and non-invalid pixels have a valid tile
                            tile_selection = np.where(~np.isin(scl_data, list(SCL_INVALID)) & (scl_data > 0), 0, -1)
                            return ProcessingStatus.SUCCESS, valid_pct, tile_selection
                        else:
                            logging.warning(f"[{partition_id}]   Existing SCL file coverage insufficient ({valid_pct:.2f}% < {min_coverage}%), will regenerate")
                            scl_out_path.unlink()
            except Exception as e:
                logging.warning(f"[{partition_id}]   Error analyzing existing SCL file: {e}, will regenerate")
                scl_out_path.unlink()

    # Check if SCL assets are included in items
    if not all('SCL' in item.assets for item in items):
        logging.warning(f"[{partition_id}]   Some items missing SCL assets, trying to use only available SCL")
        # Filter items with SCL assets
        scl_items = [item for item in items if 'SCL' in item.assets]
        if not scl_items:
            logging.error(f"[{partition_id}]   All items missing SCL assets, cannot perform quality assessment!")
            # Return processing error result
            return ProcessingStatus.PROCESSING_ERROR, 0.0, None
        items = scl_items

    # Enhanced SCL processing with retry mechanism
    def _process_scl_operation():
        t0 = time.time()
        
        # Use stackstac.stack to load SCL band
        da = stackstac.stack(
            items=items,
            assets=['SCL'],
            resolution=res,
            epsg=tpl["crs"].to_epsg(),
            bounds=bbox_proj,
            chunksize=chunksize,
            rescale=False,
            resampling=Resampling.nearest
        )
        
        # Flatten useless dimensions but keep multi-item dimension
        item_dim = None
        for dim in da.dims:
            if dim not in ('band', 'x', 'y'):
                if da.sizes[dim] > 1:
                    item_dim = dim  # Find multi-item dimension
                elif da.sizes[dim] == 1:
                    da = da.squeeze(dim, drop=True)
        
        # Extract SCL data
        scl_da = da.sel(band='SCL')
        
        # Get numpy array
        scl_arr = scl_da.values
        
        # Check if array size is reasonable
        if not check_memory_requirements(scl_arr.shape, scl_arr.dtype):
            logging.warning(f"[{partition_id}]   SCL array too large, cannot perform quality assessment")
            raise Exception("SCL array too large")
        
        logging.info(f"[{partition_id}]   SCL band data extracted in {time.time() - t0:.1f}s")
        
        # Perform SCL processing - vectorized version
        valid_mask, tile_selection, valid_pct = process_scl(scl_arr, mask_np, partition_id)
        
        # Check if valid coverage meets threshold
        if valid_pct < min_coverage:
            logging.warning(f"[{partition_id}]   ⚠️ {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping SCL generation")
            return ProcessingStatus.INSUFFICIENT_COVERAGE, valid_pct, None  # Return insufficient coverage result but don't generate file
        
        # Create SCL mosaic output based on valid_mask and tile_selection, preserving original SCL values - vectorized version
        scl_output = create_scl_mosaic(scl_arr, tile_selection, mask_np, (tpl["height"], tpl["width"]), date_key, partition_id)
        
        # Create metadata
        metadata = {
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "DATE_ACQUIRED": date_key,
            "BAND_NAME": "SCL",
            "ITEMS_COUNT": len(items),
            "VALID_COVERAGE_PCT": f"{valid_pct:.2f}"
        }
        
        # Write SCL GeoTIFF
        write_tiff(scl_output, scl_out_path, tpl, "uint8", metadata)
        
        # Validate output file
        if not validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ SCL file validation failed")
            if scl_out_path.exists():
                scl_out_path.unlink()
            raise Exception("SCL file validation failed")
        
        logging.info(f"[{partition_id}]   ✓ SCL quality assessment and file generation complete, validity: {valid_pct:.2f}%, "
                    f"file size: {os.path.getsize(scl_out_path)/1e6:.2f} MB, time {time.time()-t0:.1f}s")
        
        return ProcessingStatus.SUCCESS, valid_pct, tile_selection
    
    # Use retry mechanism for SCL processing
    try:
        return retry_with_backoff(
            _process_scl_operation,
            max_attempts=SCL_MAX_ATTEMPTS,
            timeout_per_attempt=SCL_BAND_TIMEOUT,
            partition_id=partition_id
        )
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ SCL processing failed after all retries: {e}")
        # This is a critical processing error - return PROCESSING_ERROR status
        return ProcessingStatus.PROCESSING_ERROR, 0.0, None

# ─── Single Day Task ──────────────────────────────────────────────────────────────────
def process_day(date_key:str, items, tpl, bbox_proj, mask_np,
               out_root:Path, res:int, chunksize:int,
               overwrite:bool, min_coverage:float=5.0,
               partition_id:str="unknown") -> bool:
    """Process single day data with enhanced timeout and retry for window processing"""
    logging.info(f"[{partition_id}] → {date_key} (item={len(items)})")
    t0 = time.time()

    def _process_day_operation():
        # Create band output directories
        for outname in BAND_MAPPING.values():
            band_dir = out_root / outname
            band_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if all processing is already complete
        if not overwrite:
            all_exist = True
            for band_name in S2_BANDS:
                out_name = BAND_MAPPING[band_name]
                out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                if not out_path.exists() or not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                    all_exist = False
                    break
            
            if all_exist:
                logging.info(f"[{partition_id}]   All bands already exist with valid files, skipping")
                return True
        
        # Process SCL band, perform quality assessment and generate SCL file, while keeping tile_selection for other bands
        scl_status, valid_pct, tile_selection = process_scl_assessment_and_generation(
            items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
            min_coverage, out_root, overwrite, partition_id
        )
        
        # Check SCL processing status
        if scl_status == ProcessingStatus.PROCESSING_ERROR:
            logging.error(f"[{partition_id}]   {date_key} SCL processing error, marking as FAILED")
            return False  # Critical error - return False to mark day as failed
        elif scl_status == ProcessingStatus.INSUFFICIENT_COVERAGE:
            logging.warning(f"[{partition_id}]   {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping other band processing")
            return True  # Insufficient coverage is not failure, just skip
        elif scl_status != ProcessingStatus.SUCCESS:
            logging.error(f"[{partition_id}]   {date_key} Unknown SCL processing status: {scl_status}")
            return False  # Unknown status - treat as error
        
        # SCL processing successful, continue with other bands
        
        # Create temporary directory for processing
        day_temp_dir = tempfile.mkdtemp(prefix=f"s2_{date_key}_", dir=TEMP_DIR)
        logging.debug(f"[{partition_id}]   Temporary directory: {day_temp_dir}")
        
        try:
            # Use thread pool to process bands in parallel (excluding SCL)
            other_bands = [band for band in S2_BANDS if band != "SCL"]
            max_workers = min(DEFAULT_MAX_WORKERS, os.cpu_count())
            logging.info(f"[{partition_id}]   Using {max_workers} threads to process {len(other_bands)} bands in parallel (excluding SCL)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {}
                
                for band_name in other_bands:
                    out_name = BAND_MAPPING[band_name]
                    out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                    
                    # If file already exists and is valid, skip
                    if not overwrite and out_path.exists() and validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        logging.info(f"[{partition_id}]     Band {band_name} valid file already exists, skipping")
                        continue
                    
                    # Create temporary output path
                    temp_path = Path(day_temp_dir) / f"{band_name}_{date_key}.tiff"
                    
                    # Submit task
                    future = executor.submit(
                        process_band, 
                        items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                        res, chunksize, temp_path, partition_id
                    )
                    futures[future] = (band_name, out_path, temp_path)
                
                # Process results
                success_count = 0
                for future in concurrent.futures.as_completed(futures):
                    band_name, out_path, temp_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            # Check if temporary file is valid
                            if temp_path.exists() and validate_tiff(temp_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                                # Move temporary file to final location
                                shutil.copy2(temp_path, out_path)
                                success_count += 1
                                logging.info(f"[{partition_id}]     ✓ {band_name} processing complete")
                            else:
                                logging.error(f"[{partition_id}]     ✗ {band_name} temporary file invalid or does not exist")
                        else:
                            logging.warning(f"[{partition_id}]     ✗ {band_name} processing failed")
                    except Exception as e:
                        logging.error(f"[{partition_id}]     ✗ {band_name} processing exception: {e}")
        finally:
            # Clean up daily temporary directory
            try:
                shutil.rmtree(day_temp_dir)
                logging.debug(f"[{partition_id}]   Cleaned temporary directory: {day_temp_dir}")
            except Exception as e:
                logging.warning(f"[{partition_id}]   Failed to clean temporary directory: {e}")
        
        # Log processing results
        total_other_bands = len(other_bands)
        total_bands = len(S2_BANDS)  # Including SCL
        
        # Calculate total success: SCL success (1) + other bands success count
        total_success = 1 + success_count  # SCL was successful if we reach here
        
        if total_success == total_bands:
            logging.info(f"[{partition_id}] ← {date_key} all bands processed successfully ({total_success}/{total_bands})")
            return True
        elif total_success > 0:
            logging.warning(f"[{partition_id}] ← {date_key} partial bands processed successfully ({total_success}/{total_bands})")
            return True  # Partial success is also success
        else:
            logging.error(f"[{partition_id}] ← {date_key} all bands processing failed")
            return False
    
    # Use retry mechanism for day processing
    try:
        result = retry_with_backoff(
            _process_day_operation,
            max_attempts=3,  # Fewer retries for day processing
            timeout_per_attempt=DAY_TIMEOUT,
            partition_id=partition_id
        )
        
        proc_time = time.time() - t0
        logging.info(f"[{partition_id}] ← {date_key} processing completed in {proc_time:.1f}s")
        return result
        
    except Exception as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} processing failed after all retries ({proc_time:.1f}s): {e}")
        return False

# ─── Main Program ───────────────────────────────────────────────────────────────────
def main():
    a = get_args()
    out_dir = Path(a.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # Use command line specified temporary directory
    global TEMP_DIR
    TEMP_DIR = a.temp_dir

    setup_logging(a.debug, out_dir, a.partition_id)
    logging.info(f"[{a.partition_id}] ⚡ S2 Fast Processor startup (Dynamic Window Optimized Edition)"); 
    log_sys(a.partition_id)
    logging.info(f"[{a.partition_id}] Processing timeout settings: Window {WINDOW_TIMEOUT//60} minutes, Single day {DAY_TIMEOUT//60} minutes, Single band {BAND_TIMEOUT//60} minutes, SCL {SCL_BAND_TIMEOUT//60} minutes")
    logging.info(f"[{a.partition_id}] Enhanced retry settings: STAC attempts {MAX_RETRIES} (each {STAC_SEARCH_TIMEOUT}s), SCL attempts {SCL_MAX_ATTEMPTS}")
    logging.info(f"[{a.partition_id}] Memory optimization: Conservative mode {MEMORY_CONSERVATIVE_MODE}, Max chunk size {MAX_CHUNK_SIZE}")
    logging.info(f"[{a.partition_id}] Temporary directory: {TEMP_DIR}")
    logging.info(f"[{a.partition_id}] Processing time period: {a.start_date} → {a.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(a.input_tiff), a.partition_id)

    # Search STAC items - CRITICAL: failure here means window failure
    search_date_range = f"{a.start_date}/{a.end_date}"

    try:
        items = search_items(bbox_ll, search_date_range, a.max_cloud, a.partition_id)
    except Exception as e:
        logging.error(f"[{a.partition_id}] CRITICAL: STAC search failed, marking window as FAILED: {e}")
        sys.exit(1)  # Exit with error to mark window as failed
    
    if not items:
        logging.warning(f"[{a.partition_id}] No images meeting criteria, exiting successfully (empty window)")
        return

    # Group by date
    groups = group_by_date(items, a.partition_id)

    # Create temporary directory for processing
    base_temp_dir = tempfile.mkdtemp(prefix=f"s2_proc_{a.partition_id}_", dir=TEMP_DIR)
    logging.info(f"[{a.partition_id}] Main temporary directory: {base_temp_dir}")

    try:
        # Create optimized client for window processing
        dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
        
        if dask_client is None:
            logging.error(f"[{a.partition_id}] Failed to create Dask client, exiting")
            sys.exit(1)
        
        report_path = out_dir / f"dask-report-{a.partition_id}.html"
        with performance_report(filename=report_path):
            # Process each day's data with overall window timeout
            results = []
            window_start_time = time.time()
            has_critical_error = False  # Track if any SCL processing error occurred
            
            for i, (d, its) in enumerate(groups.items()):
                # Check overall window timeout
                elapsed = time.time() - window_start_time
                if elapsed > WINDOW_TIMEOUT:
                    logging.warning(f"[{a.partition_id}] Window processing timeout exceeded ({elapsed:.1f}s > {WINDOW_TIMEOUT}s), stopping processing")
                    break
                
                # Perform garbage collection between days
                if i > 0:
                    gc.collect()
                
                # Process current day data
                try:
                    success = process_day(
                        d, its, tpl, bbox_proj, mask_np,
                        out_dir, a.resolution, a.chunksize,
                        a.overwrite, a.min_coverage, a.partition_id
                    )
                    results.append(success)
                    
                    # If this day failed, it could be due to SCL processing error
                    # Check the log to see if SCL processing error occurred
                    if not success:
                        # We need to check if this was a critical SCL processing error
                        # Since we can't easily determine this from the return value alone,
                        # we'll treat any day failure as potentially critical
                        logging.error(f"[{a.partition_id}] Day {d} processing failed - this could indicate SCL processing error")
                        has_critical_error = True
                        
                except Exception as day_error:
                    logging.error(f"[{a.partition_id}] Exception occurred while processing date {d}: {day_error}")
                    # Add a failure result and mark as critical error
                    results.append(False)
                    has_critical_error = True
        
        # Try to close client
        try:
            dask_client.close(timeout=15)
        except:
            pass
        
        # Summary statistics
        success_count = sum(results)
        total_count = len(results)
        total_elapsed = time.time() - window_start_time
        
        logging.info(f"[{a.partition_id}] ✅ Window processing complete: successful {success_count}/{total_count} days in {total_elapsed:.1f}s")
        logging.info(f"[{a.partition_id}] 📊 Dask performance report saved: {report_path}")
        
        # NEW LOGIC: If any SCL processing error occurred, fail the entire window
        if has_critical_error:
            logging.error(f"[{a.partition_id}] CRITICAL: SCL processing errors detected during window processing")
            logging.error(f"[{a.partition_id}] Marking entire window as FAILED to ensure data quality")
            sys.exit(1)  # Fail the entire window due to SCL processing errors
        
        # Return appropriate exit code based on success rate
        if success_count == 0 and total_count > 0:
            logging.error(f"[{a.partition_id}] All days failed, marking window as FAILED")
            sys.exit(1)  # All failed - window fails
        elif success_count < total_count:
            logging.warning(f"[{a.partition_id}] ⚠️  Some dates processing failed ({total_count - success_count}/{total_count})")
            # For partial success, we still consider it successful but log warnings
            sys.exit(0)  # Partial success is still success for window processing
        else:
            sys.exit(0)  # All successful

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(base_temp_dir)
            logging.info(f"[{a.partition_id}] Cleaned main temporary directory")
        except Exception as e:
            logging.warning(f"[{a.partition_id}] Failed to clean main temporary directory: {e}")

if __name__ == "__main__":
    main()