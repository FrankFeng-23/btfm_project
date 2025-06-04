#!/usr/bin/env python3
"""
s1_fast_processor.py - Sentinel-1 RTC Fast Download & ROI Mosaic (Robust Network Edition)
Updated: 2025-05-30
Supports robust network handling with comprehensive retry mechanisms and timeout control
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from rasterio.merge import merge
import pystac_client, planetary_computer, stackstac
import shapely.geometry
import concurrent.futures
import uuid
import tempfile
import shutil
import random

import dask
from dask.distributed import Client, LocalCluster, performance_report, wait

# distributed version compatibility
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

# Import for robust network handling
from pystac_client.exceptions import APIError
from urllib3.exceptions import ReadTimeoutError, ConnectTimeoutError
from requests.exceptions import RequestException, Timeout, ConnectionError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# Valid coverage threshold (skip processing if below this value)
MIN_VALID_COVERAGE = 10.0  # percentage

# Timeout settings (seconds)
PROCESS_TIMEOUT = 25 * 60  # overall timeout
DAY_TIMEOUT = 10 * 60      # single day processing timeout
ITEM_TIMEOUT = 6 * 60      # single item processing timeout
SEARCH_TIMEOUT = 40    # STAC search timeout

# Network retry settings
MAX_SEARCH_RETRIES = 8     # Maximum retries for STAC search
MAX_DOWNLOAD_RETRIES = 5   # Maximum retries for data download
BASE_RETRY_DELAY = 2       # Base delay for exponential backoff

# Timeout Control
class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """Timeout context manager"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Operation timeout ({seconds}seconds)")
    
    # Check if in main thread (Unix signals can only be handled in main thread)
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

def robust_sleep(base_delay, attempt, max_jitter=1.0):
    """Exponential backoff with jitter"""
    delay = base_delay * (2 ** attempt) + random.uniform(0, max_jitter)
    time.sleep(min(delay, 60))  # Cap at 60 seconds
    return delay

def is_network_error(error):
    """Check if error is network-related"""
    error_str = str(error).lower()
    network_keywords = [
        'timeout', 'connection', 'network', 'exceeded the maximum allowed time',
        'read timeout', 'connect timeout', 'unreachable', 'refused', 'reset',
        'temporary failure', 'service unavailable'
    ]
    return any(keyword in error_str for keyword in network_keywords)

# CLI
def get_args():
    P = argparse.ArgumentParser("Fast Sentinel-1 RTC Processor (Robust Network Edition)")
    P.add_argument("--input_tiff",   required=True)
    P.add_argument("--start_date",   required=True)
    P.add_argument("--end_date",     required=True)
    P.add_argument("--output",       default="sentinel1_output")
    P.add_argument("--orbit_state",  default="both", choices=["ascending", "descending", "both"])
    P.add_argument("--dask_workers", type=int,   default=8)
    P.add_argument("--worker_memory",type=int,   default=16)
    P.add_argument("--chunksize",    type=int,   default=1024)
    P.add_argument("--resolution",   type=float, default=10.0)
    P.add_argument("--workers",      type=int,   default=8)
    P.add_argument("--overwrite",    action="store_true")
    P.add_argument("--debug",        action="store_true")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="Minimum valid pixel coverage ratio (percentage)")
    P.add_argument("--partition_id", default="unknown",
                   help="Partition ID (for log identification)")
    P.add_argument("--max_search_retries", type=int, default=MAX_SEARCH_RETRIES,
                   help="Maximum retries for STAC search")
    P.add_argument("--search_chunk_days", type=int, default=15,
                   help="Days per search chunk when splitting large date ranges")
    return P.parse_args()

# Logging
def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """Setup logging with partition ID identification"""
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
    
    # File handler (partition-specific log file)
    file_handler = logging.FileHandler(
        out_dir / f"s1_{partition_id}_detail.log", 
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

# Dask
def make_client(req_workers:int, req_mem:int, partition_id: str):
    """Create Dask client with partition-specific dashboard port"""
    total_mem = psutil.virtual_memory().total / 1e9
    workers = min(req_workers, os.cpu_count(),
                  max(1, int(total_mem // (req_mem*1.2))))
    if workers < req_workers:
        logging.warning(f"⚠️  worker count {req_workers}→{workers} (resource limit)")
    
    # Auto-assign dashboard port for different partitions
    # Use hash of partition ID to determine port, ensure same ID always uses same port
    # Limit ports to 8700-8779 range
    port_base = 8700
    port_range = 80
    dashboard_port = port_base + (hash(partition_id) % port_range)
    
    cluster = LocalCluster(
        n_workers         = workers,
        threads_per_worker= 4,
        processes         = True,
        memory_limit      = f"{req_mem}GB",
        dashboard_address = f":{dashboard_port}",
        silence_logs      = "ERROR",
    )
    dask.config.set({
        "distributed.worker.memory.target": 0.80,
        "distributed.worker.memory.spill":  0.90,
        "distributed.worker.memory.pause":  0.95,
    })
    cli = Client(cluster, asynchronous=False)
    logging.info(f"[{partition_id}] Dask dashboard → {cli.dashboard_link}")
    return cli

# ROI & Mask
def load_roi(tiff: Path, partition_id: str):
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        mask_np   = (src.read(1) > 0).astype(np.uint8)
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']}")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    return tpl, bbox_proj, bbox_ll, mask_np

def mask_to_xr(mask_np, tpl):
    da = xr.DataArray(mask_np, dims=("y", "x"))
    return da.rio.write_crs(tpl["crs"]).rio.write_transform(tpl["transform"])

# Robust STAC Search
def search_items_single(bbox_ll, date_range: str, orbit_state="both", partition_id="unknown"):
    """Single STAC search attempt"""
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )
    
    query = {
        "collections": ["sentinel-1-rtc"], 
        "bbox": bbox_ll, 
        "datetime": date_range
    }
    
    # Add filter condition if orbit state is specified
    if orbit_state != "both":
        query["query"] = {"sat:orbit_state": {"eq": orbit_state}}
    
    q = cat.search(**query)
    
    # Use items() instead of get_items() to avoid deprecation warning
    items = list(q.items())
    
    return items

def search_items_chunked(bbox_ll, start_date: str, end_date: str, chunk_days: int, orbit_state="both", partition_id="unknown"):
    """Search items by splitting date range into smaller chunks"""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_items = []
    current_dt = start_dt
    chunk_num = 0
    
    while current_dt <= end_dt:
        chunk_end = min(current_dt + timedelta(days=chunk_days-1), end_dt)
        chunk_range = f"{current_dt.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
        chunk_num += 1
        
        logging.info(f"[{partition_id}] Searching chunk {chunk_num}: {chunk_range}")
        
        for attempt in range(MAX_SEARCH_RETRIES):
            try:
                with timeout_handler(SEARCH_TIMEOUT):
                    chunk_items = search_items_single(bbox_ll, chunk_range, orbit_state, partition_id)
                    
                logging.info(f"[{partition_id}] Chunk {chunk_num} found {len(chunk_items)} items")
                all_items.extend(chunk_items)
                break
                
            except (APIError, TimeoutException, RequestException, ReadTimeoutError, ConnectTimeoutError) as e:
                if attempt < MAX_SEARCH_RETRIES - 1:
                    delay = robust_sleep(BASE_RETRY_DELAY, attempt)
                    logging.warning(f"[{partition_id}] Chunk {chunk_num} search failed (attempt {attempt+1}/{MAX_SEARCH_RETRIES}): {type(e).__name__} - {e}")
                    logging.info(f"[{partition_id}] Retrying chunk {chunk_num} in {delay:.1f}s...")
                else:
                    logging.error(f"[{partition_id}] Chunk {chunk_num} search failed after all retries: {type(e).__name__} - {e}")
            except Exception as e:
                if attempt < MAX_SEARCH_RETRIES - 1:
                    delay = robust_sleep(BASE_RETRY_DELAY, attempt)
                    logging.warning(f"[{partition_id}] Chunk {chunk_num} unexpected error (attempt {attempt+1}/{MAX_SEARCH_RETRIES}): {type(e).__name__} - {e}")
                    logging.info(f"[{partition_id}] Retrying chunk {chunk_num} in {delay:.1f}s...")
                else:
                    logging.error(f"[{partition_id}] Chunk {chunk_num} failed after all retries: {type(e).__name__} - {e}")
        
        current_dt = chunk_end + timedelta(days=1)
    
    return all_items

def search_items(bbox_ll, date_range: str, orbit_state="both", partition_id="unknown", max_retries=None, chunk_days=15):
    """Search STAC items with robust retry mechanism and chunking strategy"""
    if max_retries is None:
        max_retries = MAX_SEARCH_RETRIES
    
    start_date, end_date = date_range.split("/")
    
    # Calculate date range span
    from datetime import datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days_span = (end_dt - start_dt).days + 1
    
    logging.info(f"[{partition_id}] Searching {days_span} days of data ({date_range})")
    
    # Strategy 1: Try full range first if span is reasonable
    if days_span <= 31:  # Less than a month, try full range
        logging.info(f"[{partition_id}] Attempting full range search...")
        
        for attempt in range(max_retries):
            try:
                with timeout_handler(SEARCH_TIMEOUT):
                    items = search_items_single(bbox_ll, date_range, orbit_state, partition_id)
                    
                logging.info(f"[{partition_id}] Full range search found {len(items)} items")
                if items:
                    b = np.array([it.bbox for it in items])
                    union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
                    logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
                return items
                
            except (APIError, TimeoutException, RequestException, ReadTimeoutError, ConnectTimeoutError) as e:
                if is_network_error(e) and attempt < max_retries - 1:
                    delay = robust_sleep(BASE_RETRY_DELAY, attempt)
                    logging.warning(f"[{partition_id}] Full range search failed (attempt {attempt+1}/{max_retries}): {type(e).__name__} - {e}")
                    logging.info(f"[{partition_id}] Retrying full range search in {delay:.1f}s...")
                else:
                    logging.error(f"[{partition_id}] Full range search failed: {type(e).__name__} - {e}")
                    break
            except Exception as e:
                logging.error(f"[{partition_id}] Full range search unexpected error: {type(e).__name__} - {e}")
                break
    
    # Strategy 2: Chunked search
    logging.info(f"[{partition_id}] Falling back to chunked search (chunk size: {chunk_days} days)")
    
    try:
        items = search_items_chunked(bbox_ll, start_date, end_date, chunk_days, orbit_state, partition_id)
        
        # Remove duplicates based on item ID
        unique_items = []
        seen_ids = set()
        for item in items:
            if item.id not in seen_ids:
                unique_items.append(item)
                seen_ids.add(item.id)
        
        logging.info(f"[{partition_id}] Chunked search found {len(unique_items)} unique items")
        
        if unique_items:
            b = np.array([it.bbox for it in unique_items])
            union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
            logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
        
        return unique_items
        
    except Exception as e:
        logging.error(f"[{partition_id}] Chunked search failed: {type(e).__name__} - {e}")
        return []

def group_by_date_orbit(items, partition_id: str):
    g = defaultdict(list)
    for it in items:
        d = it.properties["datetime"][:10]
        orbit = it.properties.get("sat:orbit_state", "unknown")
        key = f"{d}_{orbit}"
        g[key].append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} observation date-orbit combinations")
    return dict(sorted(g.items()))

# Amplitude to dB conversion
def amplitude_to_db(amp, mask=None):
    """
    Convert amplitude to dB values (with offset and scaling for int16 storage)
    
    Conversion formula:
    dB = 20 * log10(amp)
    shifted = dB + 50     # offset to avoid negative values
    scaled = shifted * 200 # scale to preserve precision
    clipped = np.clip(scaled, 0, 30000) # clip to int16 range
    """
    # Ensure amp is numpy array
    if hasattr(amp, 'values'):
        amp_array = amp.values
    elif hasattr(amp, 'compute'):
        amp_array = amp.compute()
    else:
        amp_array = np.asarray(amp)
    
    # Create output array
    output = np.zeros_like(amp_array, dtype=np.int16)
    
    # Safely handle possible invalid values
    with np.errstate(invalid='ignore', divide='ignore'):
        # Ensure amp is finite (not NaN or inf)
        amp_finite = np.isfinite(amp_array)
        # Create valid value mask (> 0)
        valid_mask = amp_finite & (amp_array > 0)
    
    # Process only valid pixels
    if np.any(valid_mask):
        # Calculate dB values only for valid pixels
        with np.errstate(invalid='ignore', divide='ignore'):
            # Direct calculation on valid positions to avoid boolean indexing issues
            valid_indices = np.where(valid_mask)
            valid_amp = amp_array[valid_indices]
            
            db = 20.0 * np.log10(valid_amp)
            db_shift = db + 50.0
            scaled = db_shift * 200.0
            clipped = np.clip(scaled, 0, 32767)  # clip to int16 range
        
        # Assign to output array
        output[valid_indices] = clipped.astype(np.int16)
    
    # Apply external mask
    if mask is not None:
        # Ensure mask is numpy array
        if hasattr(mask, 'values'):
            mask_array = mask.values
        else:
            mask_array = np.asarray(mask)
        
        # Handle shape mismatch
        if output.shape != mask_array.shape:
            # Calculate common area
            common_shape = tuple(min(output.shape[i], mask_array.shape[i]) for i in range(len(output.shape)))
            
            # Create cropped arrays
            if len(common_shape) == 2:
                output_cropped = output[:common_shape[0], :common_shape[1]]
                mask_cropped = mask_array[:common_shape[0], :common_shape[1]]
                output[:common_shape[0], :common_shape[1]] = np.where(mask_cropped > 0, output_cropped, 0)
                # Zero out areas beyond mask range
                if output.shape[0] > common_shape[0]:
                    output[common_shape[0]:, :] = 0
                if output.shape[1] > common_shape[1]:
                    output[:, common_shape[1]:] = 0
            else:
                # If not 2D, use simple cropping
                output = np.where(mask_array > 0, output, 0)
        else:
            output = np.where(mask_array > 0, output, 0)
        
    return output

# GeoTIFF output
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    # Handle NaN values
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
        
    profile = dict(driver="GTiff", dtype=dtype, count=1,
                   width=tpl["width"], height=tpl["height"],
                   crs=tpl["crs"], transform=tpl["transform"],
                   tiled=True,
                   blockxsize=256, blockysize=256,
                   nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        # Set band description and metadata
        if metadata:
            dst.set_band_description(1, metadata.get("band_desc", ""))
            dst.update_tags(**metadata)

# TIFF validation
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """Validate if TIFF file has expected attributes"""
    try:
        with rasterio.open(file_path) as src:
            # Check basic attributes
            if src.shape != expected_shape:
                logging.warning(f"Validation failed: {file_path} shape mismatch. Expected {expected_shape}, got {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"Validation failed: {file_path} CRS mismatch. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check transform matrix
            if not np.allclose(np.array(src.transform)[:6], np.array(expected_transform)[:6], rtol=1e-05, atol=1e-08):
                logging.warning(f"Validation failed: {file_path} transform matrix mismatch.")
                return False
            
            # Check data existence
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"Validation failed: {file_path} all bands are zero")
                return False
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            expected_size_mb = (src.width * src.height * src.count * 2) / (1024 * 1024)  # int16 = 2 bytes
            if file_size_mb < expected_size_mb * 0.05:  # Consider compression ratio, but can't be too small
                logging.warning(f"Validation failed: {file_path} file too small. Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
                return False
            
            logging.debug(f"TIFF validation passed: {file_path}, shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"Error validating TIFF {file_path}: {e}")
        return False

# Coverage analysis
def analyze_coverage(data_arr, roi_mask, partition_id: str):
    """Analyze data coverage over ROI, handle shape mismatch"""
    # Ensure data_arr is numpy array
    if hasattr(data_arr, 'values'):
        data_values = data_arr.values
    elif hasattr(data_arr, 'compute'):
        data_values = data_arr.compute()
    else:
        data_values = np.asarray(data_arr)
    
    # Create valid value mask (non-zero and finite), suppress invalid value warnings
    with np.errstate(invalid='ignore'):
        valid_mask = (data_values > 0) & np.isfinite(data_values)
    
    # Check if shapes match, adjust if not
    if len(valid_mask.shape) == 2 and valid_mask.shape != roi_mask.shape:
        logging.info(f"[{partition_id}]     Data shape {valid_mask.shape} doesn't match ROI shape {roi_mask.shape}, cropping to common area")
        # Calculate common height and width
        common_height = min(valid_mask.shape[0], roi_mask.shape[0])
        common_width = min(valid_mask.shape[1], roi_mask.shape[1])
        
        # Crop both arrays to common area
        valid_mask_cropped = valid_mask[:common_height, :common_width]
        roi_mask_cropped = roi_mask[:common_height, :common_width]
        
        # Continue analysis with cropped masks, ensure returning numeric values
        valid_count = int(np.sum(valid_mask_cropped & roi_mask_cropped))
        roi_count = int(np.sum(roi_mask_cropped))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     Single Tile: Valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct
    
    # Handle 3D arrays
    elif len(valid_mask.shape) == 3:
        # Multiple tiles case
        n_tiles = valid_mask.shape[0]
        tile_stats = []
        
        for i in range(n_tiles):
            # Check if current tile shape matches ROI
            if valid_mask[i].shape != roi_mask.shape:
                logging.info(f"[{partition_id}]     Tile {i} shape {valid_mask[i].shape} doesn't match ROI shape {roi_mask.shape}, cropping to common area")
                # Calculate common area
                common_height = min(valid_mask[i].shape[0], roi_mask.shape[0])
                common_width = min(valid_mask[i].shape[1], roi_mask.shape[1])
                
                # Crop current tile and ROI mask
                tile_valid = valid_mask[i][:common_height, :common_width]
                roi_cropped = roi_mask[:common_height, :common_width]
            else:
                tile_valid = valid_mask[i]
                roi_cropped = roi_mask
            
            # Calculate statistics, ensure returning numeric values
            valid_count = int(np.sum(tile_valid & roi_cropped))
            roi_count = int(np.sum(roi_cropped))
            valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
            logging.info(f"[{partition_id}]     Tile {i}: Valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
            tile_stats.append(valid_pct)
        
        # Take maximum coverage as total coverage (simplified handling)
        if tile_stats:
            total_valid_pct = max(tile_stats)
            logging.info(f"[{partition_id}]     Merged: Valid pixel coverage in ROI {total_valid_pct:.2f}%")
        else:
            total_valid_pct = 0
            logging.info(f"[{partition_id}]     No valid coverage")
        
        return valid_mask, total_valid_pct
    
    else:
        # Single tile case, shapes match
        tile_valid = valid_mask & roi_mask
        valid_count = int(np.sum(tile_valid))
        roi_count = int(np.sum(roi_mask))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     Single Tile: Valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct

# Robust data loading

def robust_stackstac_load(item, bounds, epsg, resolution, chunksize, partition_id, max_retries=None):
    """Robust data loading with retry mechanism"""
    if max_retries is None:
        max_retries = MAX_DOWNLOAD_RETRIES
    
    for attempt in range(max_retries):
        try:
            with timeout_handler(ITEM_TIMEOUT):
                ds = stackstac.stack(
                    [item], 
                    bounds=bounds,
                    epsg=epsg,
                    resolution=resolution,
                    chunksize=chunksize
                )
                
                # Check if bands exist
                if 'vv' not in ds.band.values or 'vh' not in ds.band.values:
                    # Return special status for missing bands instead of error
                    return None, "NO_BANDS"
                
                # Extract VV and VH data
                vv_data = ds.sel(band="vv").squeeze()
                vh_data = ds.sel(band="vh").squeeze()
                
                # Try to compute data (this triggers actual download)
                vv_values = vv_data.compute()
                vh_values = vh_data.compute()
                
                return (vv_values, vh_values), None
                
        except (TimeoutException, RuntimeError, RequestException, ReadTimeoutError, ConnectTimeoutError) as e:
            if is_network_error(e) and attempt < max_retries - 1:
                delay = robust_sleep(BASE_RETRY_DELAY, attempt, max_jitter=2.0)
                logging.warning(f"[{partition_id}]   Data loading failed (attempt {attempt+1}/{max_retries}): {type(e).__name__} - {e}")
                logging.info(f"[{partition_id}]   Retrying data loading in {delay:.1f}s...")
            else:
                error_msg = f"Data loading failed after {attempt+1} attempts: {type(e).__name__} - {e}"
                logging.error(f"[{partition_id}]   {error_msg}")
                return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during data loading: {type(e).__name__} - {e}"
            logging.error(f"[{partition_id}]   {error_msg}")
            return None, error_msg
    
    return None, f"Data loading failed after all {max_retries} retries"

# Process single item
def process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id):
    """Process single Sentinel-1 item with robust network handling"""
    orbit_state = item.properties.get("sat:orbit_state", "unknown")
    date_str = item.properties.get("datetime").split("T")[0]
    item_id = item.id
    
    # Ensure temp_dir is Path object
    temp_dir = Path(temp_dir)
    
    # Generate unique temporary filenames
    uid = uuid.uuid4().hex[:8]
    vv_temp = temp_dir / f"{date_str}_vv_{orbit_state}_{uid}.tiff"
    vh_temp = temp_dir / f"{date_str}_vh_{orbit_state}_{uid}.tiff"
    
    logging.info(f"[{partition_id}]   Processing item {item_id} ({date_str}_{orbit_state})")
    
    # Robust data loading
    data_result, error_msg = robust_stackstac_load(
        item, bbox_proj, tpl["crs"].to_epsg(), resolution, chunksize, partition_id
    )
    
    if data_result is None:
        if error_msg == "NO_BANDS":
            logging.info(f"[{partition_id}]   ℹ️ {date_str}_{orbit_state}: No bands available, skipping")
            return None, None, "no_data"
        else:
            logging.error(f"[{partition_id}]   ✗ {date_str}_{orbit_state}: {error_msg}")
            return None, None, "failed"
    
    vv_values, vh_values = data_result
    
    # Get actual dimensions of data
    vv_shape = vv_values.shape
    logging.debug(f"[{partition_id}]   VV data shape: {vv_shape}, ROI shape: {mask_np.shape}")
    
    # Analyze coverage
    logging.info(f"[{partition_id}]   Analyzing VV band coverage")
    vv_valid_mask, vv_valid_pct = analyze_coverage(vv_values, mask_np, partition_id)
    
    logging.info(f"[{partition_id}]   Analyzing VH band coverage")
    vh_valid_mask, vh_valid_pct = analyze_coverage(vh_values, mask_np, partition_id)
    
    # Check if coverage meets threshold
    if vv_valid_pct < min_coverage and vh_valid_pct < min_coverage:
        logging.warning(f"[{partition_id}]   ⚠️ {date_str}_{orbit_state} valid coverage VV={vv_valid_pct:.2f}%, VH={vh_valid_pct:.2f}% both below {min_coverage}%, skipping")
        return None, None, "skipped"
    
    # Process VV data: amplitude to dB and apply ROI mask
    if vv_valid_pct >= min_coverage:
        logging.info(f"[{partition_id}]   Processing VV band")
        
        # Ensure mask shapes match
        common_height = min(vv_values.shape[0], mask_np.shape[0])
        common_width = min(vv_values.shape[1], mask_np.shape[1])
        
        # Crop data and mask
        vv_cropped = vv_values[:common_height, :common_width]
        mask_cropped = mask_np[:common_height, :common_width]
        
        # Apply amplitude to dB processing
        vv_db = amplitude_to_db(vv_cropped, mask=mask_cropped)
        
        # Prepare full-size output array
        vv_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
        vv_final[:common_height, :common_width] = vv_db
        
        # Write VV TIFF
        vv_metadata = {
            "band_desc": "VV polarization, amplitude to dB, +50 offset, scale=200",
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "ORBIT_STATE": orbit_state,
            "DATE_ACQUIRED": date_str,
            "POLARIZATION": "VV",
            "DESCRIPTION": "Sentinel-1 SAR data (VV). Values are amplitude converted to dB, shifted by +50, scaled by 200."
        }
        write_tiff(vv_final, vv_temp, tpl, "int16", vv_metadata)
        
        # Validate VV output
        if not validate_tiff(vv_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ VV output validation failed")
            if vv_temp.exists():
                vv_temp.unlink()
            vv_temp = None
        else:
            logging.info(f"[{partition_id}]   ✓ VV: {os.path.getsize(vv_temp)/1e6:.2f} MB")
    else:
        logging.warning(f"[{partition_id}]   ⚠️ VV coverage insufficient, skipping")
        vv_temp = None
    
    # Process VH data: amplitude to dB and apply ROI mask
    if vh_valid_pct >= min_coverage:
        logging.info(f"[{partition_id}]   Processing VH band")
        
        # Ensure mask shapes match
        common_height = min(vh_values.shape[0], mask_np.shape[0])
        common_width = min(vh_values.shape[1], mask_np.shape[1])
        
        # Crop data and mask
        vh_cropped = vh_values[:common_height, :common_width]
        mask_cropped = mask_np[:common_height, :common_width]
        
        # Apply amplitude to dB processing
        vh_db = amplitude_to_db(vh_cropped, mask=mask_cropped)
        
        # Prepare full-size output array
        vh_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
        vh_final[:common_height, :common_width] = vh_db
        
        # Write VH TIFF
        vh_metadata = {
            "band_desc": "VH polarization, amplitude to dB, +50 offset, scale=200",
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "ORBIT_STATE": orbit_state,
            "DATE_ACQUIRED": date_str,
            "POLARIZATION": "VH",
            "DESCRIPTION": "Sentinel-1 SAR data (VH). Values are amplitude converted to dB, shifted by +50, scaled by 200."
        }
        write_tiff(vh_final, vh_temp, tpl, "int16", vh_metadata)
        
        # Validate VH output
        if not validate_tiff(vh_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ VH output validation failed")
            if vh_temp.exists():
                vh_temp.unlink()
            vh_temp = None
        else:
            logging.info(f"[{partition_id}]   ✓ VH: {os.path.getsize(vh_temp)/1e6:.2f} MB")
    else:
        logging.warning(f"[{partition_id}]   ⚠️ VH coverage insufficient, skipping")
        vh_temp = None
    
    # Check final status
    if vv_temp or vh_temp:
        return vv_temp, vh_temp, "success"
    else:
        return None, None, "skipped"

# Mosaic multiple TIFFs
def mosaic_tiffs(tiff_paths, output_path, tpl, date_str, orbit_state, polarization, partition_id):
    """Mosaic multiple TIFFs into one output TIFF"""
    try:
        # Ensure output_path is Path object
        output_path = Path(output_path)
        
        # Open all source TIFFs
        src_files = []
        for path in tiff_paths:
            if path and os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logging.warning(f"[{partition_id}]   Failed to open {path} for mosaicking: {e}")
        
        if not src_files:
            logging.warning(f"[{partition_id}]   No valid files for mosaicking {date_str}_{polarization}_{orbit_state}")
            return None
        
        # Perform mosaic operation
        logging.info(f"[{partition_id}]   Mosaicking {len(src_files)} {polarization} files ({date_str}_{orbit_state})")
        mosaic_data, out_transform = merge(src_files, nodata=0)
        
        # Close all source files
        for src in src_files:
            src.close()
        
        # Check mosaic data structure
        if mosaic_data.shape[0] < 1:
            logging.error(f"[{partition_id}]   Mosaic data structure incorrect ({date_str}_{polarization}_{orbit_state})")
            return None
        
        # Create metadata
        metadata = {
            "band_desc": f"{polarization} polarization, amplitude to dB, +50 offset, scale=200",
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "ORBIT_STATE": orbit_state,
            "DATE_ACQUIRED": date_str,
            "POLARIZATION": polarization,
            "MOSAIC_SOURCE_COUNT": len(src_files),
            "DESCRIPTION": f"Mosaicked Sentinel-1 SAR data ({polarization}). Values are amplitude converted to dB, shifted by +50, scaled by 200."
        }
        
        # Write mosaic TIFF
        write_tiff(mosaic_data[0], output_path, tpl, "int16", metadata)
        
        # Validate output file
        if not validate_tiff(output_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ Mosaic TIFF validation failed ({date_str}_{polarization}_{orbit_state})")
            if Path(output_path).exists():
                Path(output_path).unlink()
            return None
        
        # Log successful completion and file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"[{partition_id}]   ✓ Successfully created mosaic {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ Error creating mosaic {date_str}_{polarization}_{orbit_state}: {e}")
        return None

# Process single day observation
def process_day_orbit(key, items, tpl, bbox_proj, mask_np, out_dir, resolution, chunksize, min_coverage, partition_id, overwrite=False):
    """Process all items for same date and orbit state, with timeout control"""
    date_str, orbit_state = key.split("_")
    logging.info(f"[{partition_id}] → {key} (item={len(items)})")
    t0 = time.time()
    
    try:
        # Use timeout control for single day processing
        with timeout_handler(DAY_TIMEOUT):
            # Ensure out_dir is Path object
            out_dir = Path(out_dir)
            
            # Create output directory
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if output files already exist (output directly to target directory, no subdirectories)
            vv_out = out_dir / f"{date_str}_vv_{orbit_state}.tiff"
            vh_out = out_dir / f"{date_str}_vh_{orbit_state}.tiff"
            
            # Skip if files exist and not overwriting
            if not overwrite and vv_out.exists() and vh_out.exists():
                # Validate existing files
                vv_valid = validate_tiff(vv_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                vh_valid = validate_tiff(vh_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                
                if vv_valid and vh_valid:
                    logging.info(f"[{partition_id}]   Valid files exist, skipping")
                    return True
                else:
                    logging.warning(f"[{partition_id}]   Files exist but validation failed, reprocessing")
                    # Delete invalid files
                    if not vv_valid and vv_out.exists():
                        vv_out.unlink()
                    if not vh_valid and vh_out.exists():
                        vh_out.unlink()
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix=f"s1_{date_str}_{orbit_state}_")
            logging.debug(f"[{partition_id}]   Temp directory: {temp_dir}")
            
            try:
                # Process each item, allow partial failures
                vv_temp_files = []
                vh_temp_files = []
                processed_count = 0  # successfully processed count
                failed_count = 0     # truly failed count
                skipped_count = 0    # skipped count
                no_data_count = 0    # no data count (NEW)
                
                for i, item in enumerate(items):
                    item_start_time = time.time()
                    logging.info(f"[{partition_id}]   Processing item {i+1}/{len(items)}")
                    
                    vv_path, vh_path, status = process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id)
                    
                    if status == "success":
                        processed_count += 1
                        if vv_path:
                            vv_temp_files.append(str(vv_path))
                        if vh_path:
                            vh_temp_files.append(str(vh_path))
                        
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} processed successfully, took {item_duration:.1f}s")
                    elif status == "skipped":
                        skipped_count += 1
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} skipped (insufficient coverage), took {item_duration:.1f}s")
                    elif status == "no_data":
                        no_data_count += 1
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} no data available, took {item_duration:.1f}s")
                    else:  # status == "failed"
                        failed_count += 1
                        item_duration = time.time() - item_start_time
                        logging.warning(f"[{partition_id}]   item {i+1} processing failed, took {item_duration:.1f}s")
                
                # Log processing statistics
                logging.info(f"[{partition_id}]   items processing stats: success {processed_count}, skipped {skipped_count}, no_data {no_data_count}, failed {failed_count} (total {len(items)})")
                
                # If no valid files, give different messages based on reason
                if not vv_temp_files and not vh_temp_files:
                    if processed_count == 0 and (skipped_count > 0 or no_data_count > 0):
                        logging.info(f"[{partition_id}]   {key} all items skipped due to insufficient coverage or no data")
                        return True  # skipped/no_data is not failure
                    else:
                        logging.warning(f"[{partition_id}]   No valid files generated for {key}")
                        return False
                
                # Process VV files
                vv_success = False
                if vv_temp_files:
                    # If only one VV file, use directly
                    if len(vv_temp_files) == 1:
                        logging.info(f"[{partition_id}]   Only one valid VV file, using directly")
                        shutil.copy2(vv_temp_files[0], vv_out)
                        vv_success = True
                    # Multiple VV files need mosaicking
                    else:
                        vv_mosaic = mosaic_tiffs(vv_temp_files, vv_out, tpl, date_str, orbit_state, "VV", partition_id)
                        vv_success = vv_mosaic is not None
                
                # Process VH files
                vh_success = False
                if vh_temp_files:
                    # If only one VH file, use directly
                    if len(vh_temp_files) == 1:
                        logging.info(f"[{partition_id}]   Only one valid VH file, using directly")
                        shutil.copy2(vh_temp_files[0], vh_out)
                        vh_success = True
                    # Multiple VH files need mosaicking
                    else:
                        vh_mosaic = mosaic_tiffs(vh_temp_files, vh_out, tpl, date_str, orbit_state, "VH", partition_id)
                        vh_success = vh_mosaic is not None
                
                # Output processing results
                total_duration = time.time() - t0
                if vv_success and vh_success:
                    logging.info(f"[{partition_id}] ← {key} successfully processed VV and VH, took {total_duration:.1f}s")
                    return True
                elif vv_success:
                    logging.info(f"[{partition_id}] ← {key} only successfully processed VV, took {total_duration:.1f}s")
                    return True
                elif vh_success:
                    logging.info(f"[{partition_id}] ← {key} only successfully processed VH, took {total_duration:.1f}s")
                    return True
                else:
                    logging.error(f"[{partition_id}] ← {key} processing failed, took {total_duration:.1f}s")
                    return False
                    
            finally:
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                    logging.debug(f"[{partition_id}]   Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logging.warning(f"[{partition_id}]   Failed to clean up temp directory: {e}")
                    
    except TimeoutException as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {key} processing timeout ({total_duration:.1f}s > {DAY_TIMEOUT}s): {e}")
        return False
    except Exception as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ✗ Error processing {key} (took {total_duration:.1f}s): {type(e).__name__} - {e}")
        return False

# Main program
def main():
    args = get_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    no_data_days = 0

    setup_logging(args.debug, out_dir, args.partition_id)
    logging.info(f"[{args.partition_id}] ⚡ S1 Fast Processor starting (Robust Network Edition)"); 
    log_sys(args.partition_id)
    logging.info(f"[{args.partition_id}] Network settings: search retries {args.max_search_retries}, chunk size {args.search_chunk_days} days")
    logging.info(f"[{args.partition_id}] Processing timeout settings: overall {PROCESS_TIMEOUT//60}min, single day {DAY_TIMEOUT//60}min, single item {ITEM_TIMEOUT//60}min, search {SEARCH_TIMEOUT//60}min")
    logging.info(f"[{args.partition_id}] Processing time period: {args.start_date} → {args.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(args.input_tiff), args.partition_id)
    
    # Search items based on orbit state with robust retry
    if args.orbit_state == "both":
        logging.info(f"[{args.partition_id}] Searching ascending and descending orbit data")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", 
                           partition_id=args.partition_id, 
                           max_retries=args.max_search_retries,
                           chunk_days=args.search_chunk_days)
    else:
        logging.info(f"[{args.partition_id}] Searching {args.orbit_state} orbit data")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", 
                           args.orbit_state, args.partition_id,
                           max_retries=args.max_search_retries,
                           chunk_days=args.search_chunk_days)
    
    if not items:
        logging.warning(f"[{args.partition_id}] No qualifying images found, exiting")
        return

    # Group by date and orbit state
    groups = group_by_date_orbit(items, args.partition_id)

    with make_client(args.dask_workers, args.worker_memory, args.partition_id):
        report_path = out_dir / f"dask-report-{args.partition_id}.html"
        with performance_report(filename=report_path):
            # Create thread pool to process multiple dates
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks
                future_to_group = {}
                for key, group_items in groups.items():
                    future = executor.submit(
                        process_day_orbit, key, group_items, tpl, bbox_proj, mask_np,
                        str(out_dir), args.resolution, args.chunksize, args.min_coverage,
                        args.partition_id, args.overwrite
                    )
                    future_to_group[future] = key
                
                # Process results
                results = []
                for future in concurrent.futures.as_completed(future_to_group):
                    key = future_to_group[future]
                    try:
                        success = future.result()
                        results.append(success)
                    except Exception as e:
                        logging.error(f"[{args.partition_id}] Exception occurred while processing {key}: {e}")
                        results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    logging.info(f"[{args.partition_id}] ✅ Partition processing complete: success {success_count}/{total_count} days")
    logging.info(f"[{args.partition_id}] 📊 Dask performance report saved: {report_path}")
    
    # Return appropriate exit code
    if success_count == 0:
        logging.warning(f"[{args.partition_id}] ⚠️  No successful processing (may be due to no data available)")
        sys.exit(1)  # all failed
    elif success_count < total_count:
        failed_with_data = total_count - success_count - no_data_days
        if no_data_days > 0 and failed_with_data == 0:
            logging.info(f"[{args.partition_id}] ℹ️  Some dates had no data ({no_data_days}/{total_count})")
            sys.exit(0) # success with no data
        if failed_with_data > 0:
            logging.warning(f"[{args.partition_id}] ⚠️  Some dates failed processing ({failed_with_data}/{total_count})")
            sys.exit(2)  # partial failure
    else:
        sys.exit(0)  # all success

if __name__ == "__main__":
    main()
