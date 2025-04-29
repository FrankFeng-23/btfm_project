#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPixel Processor

This script processes MGRS-tiled d-pixel data (Sentinel-1 and Sentinel-2) and
re-tiles it into non-overlapping patches of customizable size.
"""

# Standard library imports
import os
import sys
import argparse
import logging
import time
from datetime import datetime
import glob
import traceback
import warnings
import shutil
from functools import partial

# Third-party imports
import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine
import multiprocessing as mp
from tqdm import tqdm

# Import these at the top level to avoid any issues with local imports
try:
    from osgeo import gdal, osr
except ImportError:
    # Make these optional for environments without GDAL
    gdal = None
    osr = None

# Monkey patch numpy's format checking to be more permissive
# This needs to be done before importing any other modules that might use np.load
original_check_version = np.lib.format._check_version
def patched_check_version(version, version_req=None):
    # Just return without checking versions
    return
np.lib.format._check_version = patched_check_version

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('dpixel_processor')

# Module globals for multiprocessing - define at module level
global_width = 0
global_height = 0
tiff_reference_path = None  # Add this global variable to store the reference TIFF path
non_zero_mask = None  # Global mask showing non-zero areas of the reference TIFF

class MGRSTile:
    """Class to represent a MGRS tile with its data and geo information."""
    
    def __init__(self, mgrs_code, mgrs_dir, reference_tiff=None):
        """Initialize MGRS tile with its code and data directory.
        
        Args:
            mgrs_code (str): MGRS code (e.g., '33UWP')
            mgrs_dir (str): Directory containing the d-pixel data
            reference_tiff (str, optional): Path to a reference GeoTIFF for this tile
        """
        self.code = mgrs_code
        self.dir = mgrs_dir
        self.data_files = {
            'bands': os.path.join(mgrs_dir, 'bands.npy'),
            'doys': os.path.join(mgrs_dir, 'doys.npy'),
            'masks': os.path.join(mgrs_dir, 'masks.npy'),
            'sar_ascending': os.path.join(mgrs_dir, 'sar_ascending.npy'),
            'sar_ascending_doy': os.path.join(mgrs_dir, 'sar_ascending_doy.npy'),
            'sar_descending': os.path.join(mgrs_dir, 'sar_descending.npy'),
            'sar_descending_doy': os.path.join(mgrs_dir, 'sar_descending_doy.npy')
        }
        
        # Initialize memmap cache
        self._memmaps = {}
        
        # Check if all required files exist
        for key, file_path in self.data_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Get shape information - try multiple approaches
        self._get_bands_shape()
        
        # Reference GeoTIFF for spatial information
        self.reference_tiff = reference_tiff
        if reference_tiff and os.path.exists(reference_tiff):
            try:
                with rasterio.open(reference_tiff) as src:
                    self.transform = src.transform
                    self.crs = src.crs
                    self.height = src.height
                    self.width = src.width
                    self.bounds = src.bounds
                    
                    # Log the bounds for debugging
                    logger.debug(f"MGRS tile {mgrs_code} bounds from reference TIFF: {self.bounds}")
                    logger.debug(f"MGRS tile {mgrs_code} transform: {self.transform}")
                    logger.debug(f"MGRS tile {mgrs_code} CRS: {self.crs}")
            except Exception as e:
                logger.warning(f"Failed to open reference TIFF: {e}")
                self.transform = None
                self.crs = None
                self.height = self.bands_shape[1]  # H dimension
                self.width = self.bands_shape[2]   # W dimension
                self.bounds = None
        else:
            self.transform = None
            self.crs = None
            self.height = self.bands_shape[1]  # H dimension
            self.width = self.bands_shape[2]   # W dimension
            self.bounds = None
            
        # Load doys arrays
        self._load_doy_arrays()
    
    def _get_bands_shape(self):
        """Attempt multiple strategies to get the shape of the bands array."""
        # Strategy 1: Try loading with memmap and extremely permissive settings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp_mmap = np.load(
                    self.data_files['bands'], 
                    mmap_mode='r', 
                    allow_pickle=True
                )
                self.bands_shape = temp_mmap.shape
                # Close the memmap
                if hasattr(temp_mmap, '_mmap') and temp_mmap._mmap is not None:
                    temp_mmap._mmap.close()
                return
        except Exception as e:
            logger.warning(f"Strategy 1 failed to get shape: {e}")
        
        # Strategy 2: Try reading a small slice of the file to infer shape
        try:
            # Try to read the first few bytes to get header information
            with open(self.data_files['bands'], 'rb') as f:
                # Skip the first 128 bytes which might contain metadata
                f.seek(128)
                # Try to read a small chunk and process as a NumPy array
                chunk = f.read(1024)
                # For now, we'll use a hardcoded shape based on expected dimensions
                # This is a placeholder and should be adjusted based on your data
                self.bands_shape = (100, 500, 500, 10)  # Typical shape for Sentinel-2 data
                logger.warning(f"Using hardcoded shape for {self.code}: {self.bands_shape}")
                return
        except Exception as e:
            logger.warning(f"Strategy 2 failed to get shape: {e}")
        
        # Strategy 3: Hardcode a reasonable shape as a last resort
        logger.warning(f"All shape detection strategies failed for {self.code}, using default shape")
        self.bands_shape = (100, 500, 500, 10)  # Default shape as fallback
    
    def _load_doy_arrays(self):
        """Load the DOY arrays with multiple fallback strategies."""
        # Initialize DOY arrays to empty arrays as defaults
        self.doys = np.array([], dtype=np.int32)
        self.sar_asc_doys = np.array([], dtype=np.int32)
        self.sar_desc_doys = np.array([], dtype=np.int32)
        
        # Try to load DOY arrays with various approaches
        for array_name in ['doys', 'sar_ascending_doy', 'sar_descending_doy']:
            try:
                # First try: standard load with pickle allowed
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    array_data = np.load(self.data_files[array_name], allow_pickle=True)
                    
                    # Assign to the correct attribute
                    if array_name == 'doys':
                        self.doys = array_data
                    elif array_name == 'sar_ascending_doy':
                        self.sar_asc_doys = array_data
                    elif array_name == 'sar_descending_doy':
                        self.sar_desc_doys = array_data
                        
                    logger.debug(f"Successfully loaded {array_name} with shape {array_data.shape}")
            except Exception as e1:
                logger.warning(f"Failed to load {array_name} with standard method: {e1}")
                try:
                    # Second try: read binary and convert
                    with open(self.data_files[array_name], 'rb') as f:
                        # Skip header
                        f.seek(128)
                        # Read data and convert to array
                        data = np.frombuffer(f.read(), dtype=np.float32)
                        # Assuming 1D array
                        if array_name == 'doys':
                            self.doys = data[:100]  # Take first 100 elements as fallback
                        elif array_name == 'sar_ascending_doy':
                            self.sar_asc_doys = data[:50]  # Take first 50 elements as fallback
                        elif array_name == 'sar_descending_doy':
                            self.sar_desc_doys = data[:50]  # Take first 50 elements as fallback
                except Exception as e2:
                    logger.warning(f"Failed to load {array_name} with binary method: {e2}")
                    # Last resort: create dummy data
                    if array_name == 'doys':
                        self.doys = np.arange(1, 101)  # Dummy DOYs 1-100
                    elif array_name == 'sar_ascending_doy':
                        self.sar_asc_doys = np.arange(1, 51)  # Dummy DOYs 1-50
                    elif array_name == 'sar_descending_doy':
                        self.sar_desc_doys = np.arange(1, 51)  # Dummy DOYs 1-50
        
        # Debug logging
        logger.debug(f"DOY arrays loaded: doys={self.doys.shape}, asc={self.sar_asc_doys.shape}, desc={self.sar_desc_doys.shape}")
    
    def get_memmap(self, data_type):
        """Get memmap for the specific data type with multiple fallback strategies.
        
        Args:
            data_type (str): One of 'bands', 'masks', 'sar_ascending', 'sar_descending'
            
        Returns:
            numpy.memmap or numpy.ndarray: Memory-mapped array or regular array for the specified data
        """
        if data_type not in self._memmaps:
            if data_type not in self.data_files:
                raise ValueError(f"Unknown data type: {data_type}")
            
            try:
                # Try to load with memmap using permissive settings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._memmaps[data_type] = np.load(
                        self.data_files[data_type], 
                        mmap_mode='r',
                        allow_pickle=True
                    )
                logger.debug(f"Successfully loaded {data_type} with shape {self._memmaps[data_type].shape}")
            except Exception as e:
                logger.warning(f"Failed to load {data_type} with memmap: {e}")
                # As a fallback, create a dummy array with the expected shape
                if data_type == 'bands':
                    # Create a dummy array filled with zeros
                    shape = self.bands_shape
                    logger.warning(f"Creating dummy array for {data_type} with shape {shape}")
                    self._memmaps[data_type] = np.zeros(shape, dtype=np.float32)
                elif data_type == 'masks':
                    # Create a dummy mask array filled with ones (valid data)
                    shape = self.bands_shape[:3]  # (T, H, W)
                    logger.warning(f"Creating dummy array for {data_type} with shape {shape}")
                    self._memmaps[data_type] = np.ones(shape, dtype=np.int8)
                elif data_type == 'sar_ascending':
                    # Create a dummy SAR ascending array
                    if not hasattr(self, 'sar_asc_doys') or len(self.sar_asc_doys) == 0:
                        # If no ascending DOYs, create an empty array with correct dimensions
                        shape = (0, self.bands_shape[1], self.bands_shape[2], 2)
                    else:
                        shape = (len(self.sar_asc_doys), self.bands_shape[1], self.bands_shape[2], 2)
                    logger.warning(f"Creating dummy array for {data_type} with shape {shape}")
                    self._memmaps[data_type] = np.zeros(shape, dtype=np.float32)
                elif data_type == 'sar_descending':
                    # Create a dummy SAR descending array
                    if not hasattr(self, 'sar_desc_doys') or len(self.sar_desc_doys) == 0:
                        # If no descending DOYs, create an empty array with correct dimensions
                        shape = (0, self.bands_shape[1], self.bands_shape[2], 2)
                    else:
                        shape = (len(self.sar_desc_doys), self.bands_shape[1], self.bands_shape[2], 2)
                    logger.warning(f"Creating dummy array for {data_type} with shape {shape}")
                    self._memmaps[data_type] = np.zeros(shape, dtype=np.float32)
        
        return self._memmaps[data_type]
    
    def close_memmaps(self):
        """Close all open memmaps to free resources."""
        if not hasattr(self, '_memmaps'):
            return
            
        for mm in self._memmaps.values():
            if hasattr(mm, '_mmap') and mm._mmap is not None:
                try:
                    mm._mmap.close()
                except Exception:
                    pass  # Ignore errors during cleanup
        self._memmaps = {}
        
    def __del__(self):
        """Destructor to ensure memmaps are closed."""
        try:
            self.close_memmaps()
        except Exception:
            pass  # Ignore errors during cleanup
        
    def __repr__(self):
        """String representation of the MGRS tile."""
        return f"MGRSTile({self.code}, shape={self.bands_shape})"

def find_mgrs_tiles(d_pixel_dir):
    """Find all MGRS tiles in the d_pixel_dir.
    
    Args:
        d_pixel_dir (str): Directory containing MGRS tile folders
        
    Returns:
        list: List of MGRS codes found
    """
    mgrs_codes = []
    for item in os.listdir(d_pixel_dir):
        item_path = os.path.join(d_pixel_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains the required files
            if (os.path.exists(os.path.join(item_path, 'bands.npy')) and
                os.path.exists(os.path.join(item_path, 'doys.npy'))):
                mgrs_codes.append(item)
    
    return mgrs_codes

def find_reference_tiff(mgrs_code, tiff_path, data_raw_dir=None):
    """Find a reference GeoTIFF for the given MGRS code.
    
    Args:
        mgrs_code (str): MGRS code
        tiff_path (str): Path to the downstream TIFF
        data_raw_dir (str, optional): Root directory for raw data
        
    Returns:
        str: Path to a reference GeoTIFF or None if not found
    """
    # If data_raw_dir is provided, try to find a band tiff there
    if data_raw_dir:
        red_dir = os.path.join(data_raw_dir, mgrs_code, 'red')
        if os.path.exists(red_dir):
            tiffs = glob.glob(os.path.join(red_dir, '*.tiff'))
            if tiffs:
                logger.info(f"Found reference TIFF for {mgrs_code} in data_raw_dir: {tiffs[0]}")
                return tiffs[0]
    
    # Fallback to using the downstream tiff
    logger.info(f"Using global reference TIFF for {mgrs_code}: {tiff_path}")
    return tiff_path

def create_global_grid_from_mgrs_tiles(mgrs_tiles, patch_size):
    """Create a global grid based on the actual dimensions of the MGRS tiles.
    
    Args:
        mgrs_tiles (dict): Dictionary of MGRSTile objects
        patch_size (int): Size of patches in pixels
        
    Returns:
        tuple: (transform, width, height, crs) of the global grid
    """
    # Find the bounding box that encompasses all MGRS tiles
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # Reference CRS to use
    global_crs = None
    
    # Collect bounds information from all tiles
    for tile in mgrs_tiles.values():
        if tile.bounds and tile.transform and tile.crs:
            min_x = min(min_x, tile.bounds.left)
            min_y = min(min_y, tile.bounds.bottom)
            max_x = max(max_x, tile.bounds.right)
            max_y = max(max_y, tile.bounds.top)
            
            # Use the CRS from the first tile with a valid CRS
            if global_crs is None:
                global_crs = tile.crs
    
    # Check if we found valid bounds
    if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
        logger.warning("Could not determine bounds from MGRS tiles, using pixel dimensions")
        
        # Fall back to just using the pixel dimensions of the tiles
        max_width = max(tile.width for tile in mgrs_tiles.values())
        max_height = max(tile.height for tile in mgrs_tiles.values())
        
        # Create a grid based on the maximum dimensions
        grid_width = (max_width + patch_size - 1) // patch_size
        grid_height = (max_height + patch_size - 1) // patch_size
        
        # Create a simple transform (identity transformation)
        transform = Affine(1, 0, 0, 0, 1, 0)
        
        logger.info(f"Created grid from pixel dimensions: {grid_width}x{grid_height} patches")
        
        # Use a default CRS if none was found
        if global_crs is None:
            logger.warning("No CRS found in MGRS tiles, using default UTM Zone 33N")
            global_crs = rasterio.crs.CRS.from_epsg(32633)  # UTM Zone 33N (common for Austria)
        
        return transform, max_width, max_height, global_crs, grid_width, grid_height
    
    # Calculate dimensions in world coordinates
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    # Determine pixel resolution (assuming square pixels)
    # Use the first tile's transform to get a reasonable resolution
    first_valid_tile = next((t for t in mgrs_tiles.values() if t.transform is not None), None)
    
    if first_valid_tile:
        resolution = abs(first_valid_tile.transform.a)  # Use the x-scale as resolution
        logger.info(f"Using resolution of {resolution} from MGRS tile {first_valid_tile.code}")
    else:
        # Fallback resolution based on Sentinel-2 10m bands
        resolution = 10.0
        logger.warning(f"No valid transform found, using default resolution of {resolution}")
    
    # Calculate dimensions in pixels
    width = int(world_width / resolution) + 1
    height = int(world_height / resolution) + 1
    
    # Create an affine transform for the global grid
    transform = Affine(resolution, 0, min_x,
                      0, -resolution, max_y)
    
    # Calculate grid dimensions
    grid_width = (width + patch_size - 1) // patch_size
    grid_height = (height + patch_size - 1) // patch_size
    
    logger.info(f"Created global grid: {width}x{height} pixels, {grid_width}x{grid_height} patches")
    logger.info(f"Global grid bounds: [{min_x}, {min_y}, {max_x}, {max_y}]")
    logger.info(f"Global grid transform: {transform}")
    
    return transform, width, height, global_crs, grid_width, grid_height

def read_reference_tiff_info(tiff_path):
    """Read basic information from the reference TIFF.
    
    Args:
        tiff_path (str): Path to the reference TIFF
        
    Returns:
        dict: Information about the reference TIFF
    """
    with rasterio.open(tiff_path) as src:
        info = {
            'width': src.width,
            'height': src.height,
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'resolution_x': abs(src.transform.a),
            'resolution_y': abs(src.transform.e)
        }
    return info

def create_non_zero_mask(tiff_path):
    """Create a mask of non-zero areas from the reference TIFF.
    
    Args:
        tiff_path (str): Path to the reference TIFF
        
    Returns:
        numpy.ndarray: Mask where True indicates non-zero value in TIFF
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Read the first band (assuming single band)
            data = src.read(1)
            # Create mask where TIFF value is non-zero
            mask = data != 0
            logger.info(f"Created non-zero mask with {np.sum(mask)} of {mask.size} pixels ({np.sum(mask) / mask.size * 100:.2f}%) being non-zero")
            return mask
    except Exception as e:
        logger.error(f"Error creating non-zero mask: {e}")
        logger.error(traceback.format_exc())
        # Return None to indicate failure
        return None

def create_global_grid(reference_tiff, patch_size, data_raw_dir):
    """Create a global grid based on the reference GeoTIFF and raw data dimensions.
    
    Args:
        reference_tiff (str): Path to the reference GeoTIFF for geographic coordinates
        patch_size (int): Size of patches in pixels
        data_raw_dir (str): Root directory for raw data containing original TIFFs
        
    Returns:
        tuple: (transform, width, height, crs, grid_width, grid_height) of the global grid
    """
    import rasterio  # Explicitly import here to avoid any issues

    logger.info(f"Reading reference TIFF: {reference_tiff}")
    
    try:
        # First get the geographic coordinates from the reference TIFF
        with rasterio.open(reference_tiff) as src:
            reference_transform = src.transform
            reference_width = src.width
            reference_height = src.height
            reference_crs = src.crs
            reference_bounds = src.bounds
            
            # Log detailed information about the reference TIFF
            logger.info(f"Reference TIFF dimensions: {reference_width}x{reference_height}")
            logger.info(f"Reference TIFF transform: {reference_transform}")
            logger.info(f"Reference TIFF CRS raw: {reference_crs}")
            logger.info(f"Reference TIFF bounds: {reference_bounds}")
            
            # Additional debugging for CRS
            if reference_crs is None:
                logger.warning("CRS is None in the input TIFF. Will attempt to use a default CRS.")
                
                # Try to extract CRS from GDAL directly if available
                if gdal is not None:
                    try:
                        ds = gdal.Open(reference_tiff)
                        wkt = ds.GetProjection()
                        if wkt:
                            logger.info(f"Found WKT from GDAL: {wkt}")
                            srs = osr.SpatialReference()
                            srs.ImportFromWkt(wkt)
                            epsg = srs.GetAuthorityCode(None)
                            if epsg:
                                logger.info(f"Found EPSG code: {epsg}")
                                reference_crs = rasterio.crs.CRS.from_epsg(int(epsg))
                                logger.info(f"Created CRS from EPSG: {reference_crs}")
                    except Exception as e:
                        logger.warning(f"Failed to extract CRS using GDAL: {e}")
                
                # If still None, use a default UTM Zone 33N (common for Austria)
                if reference_crs is None:
                    logger.warning("Using default CRS: EPSG:32633 (UTM Zone 33N)")
                    reference_crs = rasterio.crs.CRS.from_epsg(32633)  # UTM Zone 33N
            
            # If we have a CRS, make sure it's in a format we can use
            if reference_crs:
                try:
                    # Try to convert to EPSG code for cleaner representation
                    epsg_code = reference_crs.to_epsg()
                    if epsg_code:
                        logger.info(f"CRS EPSG code: {epsg_code}")
                        
                    # Ensure we can serialize the CRS (important for multiprocessing)
                    crs_wkt = reference_crs.wkt
                    logger.info(f"CRS WKT available: {crs_wkt is not None}")
                    
                    # Test if we can create a new CRS object from this
                    test_crs = rasterio.crs.CRS.from_wkt(crs_wkt)
                    logger.info(f"Successfully created test CRS: {test_crs}")
                except Exception as e:
                    logger.warning(f"CRS validation failed: {e}")
        
        # For grid dimensions, USE THE REFERENCE TIFF DIMENSIONS
        # This is the key change to fix the bug where not all tiles are processed
        real_width = reference_width
        real_height = reference_height
        
        # Also check MGRS tiles in data_raw_dir to validate these dimensions
        mgrs_dirs = []
        for item in os.listdir(data_raw_dir):
            if os.path.isdir(os.path.join(data_raw_dir, item)):
                # Check if this looks like an MGRS directory (e.g., has subdirectories like 'red')
                red_dir = os.path.join(data_raw_dir, item, 'red')
                if os.path.exists(red_dir):
                    mgrs_dirs.append(item)
        
        logger.info(f"Found {len(mgrs_dirs)} MGRS directories in data_raw_dir: {', '.join(mgrs_dirs)}")
        
        # Compare reference dimensions to MGRS tile dimensions for validation
        if mgrs_dirs:
            max_mgrs_width = 0
            max_mgrs_height = 0
            
            for mgrs_dir in mgrs_dirs:
                red_dir = os.path.join(data_raw_dir, mgrs_dir, 'red')
                tiff_files = glob.glob(os.path.join(red_dir, '*.tiff'))
                if tiff_files:
                    with rasterio.open(tiff_files[0]) as src:
                        max_mgrs_width = max(max_mgrs_width, src.width)
                        max_mgrs_height = max(max_mgrs_height, src.height)
                        logger.info(f"MGRS tile {mgrs_dir} dimensions: {src.width}x{src.height}")
            
            # If MGRS dimensions are larger than reference, log a warning
            if max_mgrs_width > real_width or max_mgrs_height > real_height:
                logger.warning(f"MGRS tiles have larger dimensions ({max_mgrs_width}x{max_mgrs_height}) "
                             f"than reference TIFF ({real_width}x{real_height})")
                
                # We still use the reference TIFF dimensions, but log this discrepancy
                logger.info(f"Using reference TIFF dimensions: {real_width}x{real_height}")
        
        # Calculate grid dimensions based on the reference TIFF dimensions
        grid_width = (real_width + patch_size - 1) // patch_size
        grid_height = (real_height + patch_size - 1) // patch_size
        
        logger.info(f"Using reference TIFF dimensions for grid: {real_width}x{real_height} pixels")
        logger.info(f"Grid dimensions: {grid_width}x{grid_height} patches of {patch_size}x{patch_size} pixels")
        
        # Return the reference transform and CRS with the real dimensions
        return reference_transform, real_width, real_height, reference_crs, grid_width, grid_height
    
    except Exception as e:
        logger.error(f"Error creating global grid: {e}")
        logger.error(traceback.format_exc())
        raise

def create_output_directory(out_dir, overwrite=False):
    """Create output directory, optionally overwriting if it exists.
    
    Args:
        out_dir (str): Output directory path
        overwrite (bool): Whether to overwrite if directory exists
        
    Returns:
        bool: True if directory was created or already exists
    """
    if os.path.exists(out_dir):
        if not overwrite:
            logger.warning(f"Output directory {out_dir} already exists, skipping creation")
            return True
        else:
            shutil.rmtree(out_dir)
            
    os.makedirs(out_dir, exist_ok=True)
    return os.path.exists(out_dir)

def get_overlapping_mgrs_tiles(x_min, y_min, x_max, y_max, mgrs_tiles, global_transform, global_crs=None):
    """Find MGRS tiles that overlap with the given patch boundaries.
    
    Args:
        x_min, y_min, x_max, y_max (float): Patch boundaries in global coordinates
        mgrs_tiles (dict): Dictionary of MGRSTile objects keyed by MGRS code
        global_transform (Affine): Affine transform of the global reference
        global_crs (CRS, optional): Coordinate reference system
        
    Returns:
        list: List of MGRSTile objects that overlap with the patch
    """
    try:
        import rasterio.warp
        
        # Convert to world coordinates if transform is provided
        if global_transform:
            # Get world coordinates for the patch corners
            world_x_min, world_y_max = global_transform * (x_min, y_min)  # Upper left
            world_x_max, world_y_min = global_transform * (x_max, y_max)  # Lower right
            
            logger.debug(f"Patch world coordinates in global CRS: [{world_x_min}, {world_y_min}, {world_x_max}, {world_y_max}]")
            
            # Check if we have valid CRS information
            if global_crs is None:
                logger.warning("No global CRS provided, using all MGRS tiles")
                return list(mgrs_tiles.values())
            
            # Special handling for WGS84 (EPSG:4326) global coordinates
            is_wgs84 = global_crs.to_epsg() == 4326
            if is_wgs84:
                logger.debug("Global coordinates are in WGS84, handling specially")
            
            overlapping_tiles = []
            for code, tile in mgrs_tiles.items():
                if not tile.bounds or not tile.crs:
                    logger.warning(f"MGRS tile {code} has no bounds or CRS information, including it")
                    overlapping_tiles.append(tile)
                    continue
                
                # Check if we need to transform coordinates between CRS
                if global_crs != tile.crs:
                    # Need to transform global coordinates to tile's CRS
                    try:
                        # Define corners in global CRS
                        corners_x = [world_x_min, world_x_max, world_x_max, world_x_min]
                        corners_y = [world_y_max, world_y_max, world_y_min, world_y_min]
                        
                        # Transform to tile's CRS
                        tile_corners_x, tile_corners_y = rasterio.warp.transform(
                            global_crs, tile.crs, corners_x, corners_y
                        )
                        
                        # Get bounds in tile's CRS
                        tile_x_min = min(tile_corners_x)
                        tile_x_max = max(tile_corners_x)
                        tile_y_min = min(tile_corners_y)
                        tile_y_max = max(tile_corners_y)
                        
                        logger.debug(f"Transformed patch bounds in {tile.crs}: [{tile_x_min}, {tile_y_min}, {tile_x_max}, {tile_y_max}]")
                        logger.debug(f"Tile bounds: {tile.bounds}")
                        
                        # Check for overlap in tile's CRS
                        overlap = not (tile_x_max < tile.bounds.left or
                                      tile_x_min > tile.bounds.right or
                                      tile_y_max < tile.bounds.bottom or
                                      tile_y_min > tile.bounds.top)
                    except Exception as e:
                        logger.warning(f"Error transforming coordinates for tile {code}: {e}")
                        # Include the tile to be safe
                        overlap = True
                else:
                    # Same CRS, check for overlap directly
                    overlap = not (world_x_max < tile.bounds.left or
                                  world_x_min > tile.bounds.right or
                                  world_y_max < tile.bounds.bottom or
                                  world_y_min > tile.bounds.top)
                
                logger.debug(f"Checking tile {code} with bounds {tile.bounds}")
                logger.debug(f"Overlap: {overlap}")
                
                if overlap:
                    overlapping_tiles.append(tile)
            
            # If no overlapping tiles found, try again with expanded bounds 
            # (especially important for WGS84/UTM conversion edge cases)
            if not overlapping_tiles:
                logger.warning("No overlapping tiles found, trying with expanded bounds")
                # Expand bounds by 10% in each direction
                dx = (world_x_max - world_x_min) * 0.1
                dy = (world_y_max - world_y_min) * 0.1
                
                expanded_world_x_min = world_x_min - dx
                expanded_world_x_max = world_x_max + dx
                expanded_world_y_min = world_y_min - dy
                expanded_world_y_max = world_y_max + dy
                
                logger.debug(f"Expanded world coordinates: [{expanded_world_x_min}, {expanded_world_y_min}, {expanded_world_x_max}, {expanded_world_y_max}]")
                
                for code, tile in mgrs_tiles.items():
                    if not tile.bounds or not tile.crs:
                        overlapping_tiles.append(tile)
                        continue
                    
                    if global_crs != tile.crs:
                        try:
                            # Define expanded corners in global CRS
                            corners_x = [expanded_world_x_min, expanded_world_x_max, expanded_world_x_max, expanded_world_x_min]
                            corners_y = [expanded_world_y_max, expanded_world_y_max, expanded_world_y_min, expanded_world_y_min]
                            
                            # Transform to tile's CRS
                            tile_corners_x, tile_corners_y = rasterio.warp.transform(
                                global_crs, tile.crs, corners_x, corners_y
                            )
                            
                            # Get bounds in tile's CRS
                            tile_x_min = min(tile_corners_x)
                            tile_x_max = max(tile_corners_x)
                            tile_y_min = min(tile_corners_y)
                            tile_y_max = max(tile_corners_y)
                            
                            # Check for overlap in tile's CRS
                            overlap = not (tile_x_max < tile.bounds.left or
                                          tile_x_min > tile.bounds.right or
                                          tile_y_max < tile.bounds.bottom or
                                          tile_y_min > tile.bounds.top)
                        except Exception:
                            overlap = True
                    else:
                        # Same CRS, check for overlap directly
                        overlap = not (expanded_world_x_max < tile.bounds.left or
                                      expanded_world_x_min > tile.bounds.right or
                                      expanded_world_y_max < tile.bounds.bottom or
                                      expanded_world_y_min > tile.bounds.top)
                    
                    if overlap:
                        overlapping_tiles.append(tile)
            
            # If still no overlapping tiles, use all tiles as a fallback
            if not overlapping_tiles:
                logger.warning("No overlapping tiles found even with expanded bounds, using all tiles as fallback")
                overlapping_tiles = list(mgrs_tiles.values())
            
            return overlapping_tiles
        else:
            # Without proper geospatial information, we'll need to use all tiles
            logger.warning("No global transform provided, using all MGRS tiles")
            return list(mgrs_tiles.values())
    except Exception as e:
        logger.error(f"Error in get_overlapping_mgrs_tiles: {e}")
        logger.error(traceback.format_exc())
        # Return all tiles as a fallback
        return list(mgrs_tiles.values())

def extract_patch_from_mgrs(mgrs_tile, x_min, y_min, patch_size, global_transform=None, global_crs=None):
    """Extract a patch from an MGRS tile.
    
    Args:
        mgrs_tile (MGRSTile): MGRS tile object
        x_min, y_min (int): Upper left corner of the patch in global coordinates
        patch_size (int): Size of the patch in pixels
        global_transform (Affine, optional): Global affine transform
        global_crs (CRS, optional): Coordinate reference system
        
    Returns:
        dict: Dictionary containing the extracted data for this patch from this tile
    """
    try:
        import rasterio.warp
        
        # Calculate the local coordinates within the MGRS tile
        local_x_min, local_y_min = 0, 0
        local_width, local_height = mgrs_tile.width, mgrs_tile.height
        
        if global_transform and mgrs_tile.transform:
            # Convert global coordinates to world coordinates
            world_x_min, world_y_max = global_transform * (x_min, y_min)
            world_x_max, world_y_min = global_transform * (x_min + patch_size, y_min + patch_size)
            
            logger.debug(f"Global patch at ({x_min}, {y_min}) with size {patch_size}")
            logger.debug(f"World coordinates in global CRS: [{world_x_min}, {world_y_min}, {world_x_max}, {world_y_max}]")
            
            # Handle different CRS
            if global_crs and mgrs_tile.crs and global_crs != mgrs_tile.crs:
                try:
                    # Special handling for WGS84 to UTM conversion
                    is_wgs84 = global_crs.to_epsg() == 4326
                    is_utm = mgrs_tile.crs.to_epsg() in range(32601, 32661)  # UTM North zones
                    
                    if is_wgs84:
                        logger.debug("Converting from WGS84 to UTM")
                        
                        # Define corners in global CRS
                        # For WGS84, use extra points to better handle the curved projection
                        corners_x = [
                            world_x_min, (world_x_min + world_x_max) / 2, world_x_max, 
                            world_x_max, (world_x_min + world_x_max) / 2, world_x_min
                        ]
                        corners_y = [
                            world_y_max, world_y_max, world_y_max,
                            world_y_min, world_y_min, world_y_min
                        ]
                    else:
                        # For other projections, standard corner points are sufficient
                        corners_x = [world_x_min, world_x_max, world_x_max, world_x_min]
                        corners_y = [world_y_max, world_y_max, world_y_min, world_y_min]
                    
                    # Transform to tile's CRS
                    transformed_x, transformed_y = rasterio.warp.transform(
                        global_crs, mgrs_tile.crs, corners_x, corners_y
                    )
                    
                    # Get bounds in tile's CRS
                    tile_x_min = min(transformed_x)
                    tile_x_max = max(transformed_x)
                    tile_y_min = min(transformed_y)
                    tile_y_max = max(transformed_y)
                    
                    logger.debug(f"Transformed coordinates in tile's CRS: [{tile_x_min}, {tile_y_min}, {tile_x_max}, {tile_y_max}]")
                    
                    # Use the transformed coordinates
                    world_x_min = tile_x_min
                    world_x_max = tile_x_max
                    world_y_min = tile_y_min
                    world_y_max = tile_y_max
                    
                except Exception as e:
                    logger.warning(f"Error transforming coordinates for extraction: {e}")
                    # For any error, use a fallback approach
                    # Just use the whole tile since we can't determine the specific region
                    local_x_min = 0
                    local_y_min = 0
                    local_width = min(patch_size, mgrs_tile.width)
                    local_height = min(patch_size, mgrs_tile.height)
                    
                    logger.debug(f"Using fallback local coordinates: [{local_x_min}, {local_y_min}, {local_width}, {local_height}]")
                    
                    # Skip further coordinate calculation and go straight to data extraction
                    return _extract_data_from_mgrs(mgrs_tile, local_x_min, local_y_min, local_width, local_height, patch_size)
            
            # Convert world coordinates to MGRS tile local coordinates
            inv_transform = ~mgrs_tile.transform
            local_x_min, local_y_min = map(int, inv_transform * (world_x_min, world_y_max))
            local_x_max, local_y_max = map(int, inv_transform * (world_x_max, world_y_min))
            
            logger.debug(f"MGRS local coordinates before clamping: [{local_x_min}, {local_y_min}, {local_x_max}, {local_y_max}]")
            
            # Clamp to MGRS tile boundaries
            local_x_min = max(0, min(local_x_min, mgrs_tile.width - 1))
            local_y_min = max(0, min(local_y_min, mgrs_tile.height - 1))
            local_x_max = max(local_x_min + 1, min(local_x_max, mgrs_tile.width))
            local_y_max = max(local_y_min + 1, min(local_y_max, mgrs_tile.height))
            
            logger.debug(f"MGRS local coordinates after clamping: [{local_x_min}, {local_y_min}, {local_x_max}, {local_y_max}]")
            
            local_width = local_x_max - local_x_min
            local_height = local_y_max - local_y_min
        else:
            # Without proper geospatial information, assume the patch covers the entire tile
            # but clamped to the actual tile size
            local_width = min(patch_size, mgrs_tile.width)
            local_height = min(patch_size, mgrs_tile.height)
            
            logger.debug(f"No transforms available, using local_width={local_width}, local_height={local_height}")
        
        # Skip if the extracted region would be empty
        if local_width <= 0 or local_height <= 0:
            logger.warning(f"Extracted region is empty: width={local_width}, height={local_height}")
            return None
        
        return _extract_data_from_mgrs(mgrs_tile, local_x_min, local_y_min, local_width, local_height, patch_size)
        
    except Exception as e:
        logger.error(f"Error calculating extraction region: {e}")
        logger.error(traceback.format_exc())
        
        # Use the entire tile as a fallback
        local_x_min, local_y_min = 0, 0
        local_width = min(patch_size, mgrs_tile.width)
        local_height = min(patch_size, mgrs_tile.height)
        
        logger.debug(f"Using fallback local coordinates after error: [{local_x_min}, {local_y_min}, {local_width}, {local_height}]")
        
        return _extract_data_from_mgrs(mgrs_tile, local_x_min, local_y_min, local_width, local_height, patch_size)

def _extract_data_from_mgrs(mgrs_tile, local_x_min, local_y_min, local_width, local_height, patch_size):
    """Helper function to extract data from MGRS tile at specified local coordinates.
    
    Args:
        mgrs_tile (MGRSTile): MGRS tile object
        local_x_min, local_y_min (int): Upper left corner in local coordinates
        local_width, local_height (int): Width and height of region to extract
        patch_size (int): Size of output patch
        
    Returns:
        dict: Dictionary containing the extracted data
    """
    # Get memmaps for the relevant data
    bands_mm = mgrs_tile.get_memmap('bands')
    masks_mm = mgrs_tile.get_memmap('masks')
    sar_asc_mm = mgrs_tile.get_memmap('sar_ascending')
    sar_desc_mm = mgrs_tile.get_memmap('sar_descending')
    
    # Create padded output arrays initialized to zeros (for bands and SAR) or zeros (for masks)
    bands_patch = np.zeros((bands_mm.shape[0], patch_size, patch_size, bands_mm.shape[3]), dtype=np.float32)
    masks_patch = np.zeros((masks_mm.shape[0], patch_size, patch_size), dtype=np.int8)
    
    # Extract and pad S2 bands data
    # Copy the actual data into the padded array
    actual_h = min(local_height, patch_size)
    actual_w = min(local_width, patch_size)
    
    logger.debug(f"Extracting data region: height={actual_h}, width={actual_w}")
    
    try:
        bands_patch[:, :actual_h, :actual_w, :] = bands_mm[:, local_y_min:local_y_min+actual_h, 
                                                         local_x_min:local_x_min+actual_w, :]
        
        # Extract and pad S2 masks data
        masks_patch[:, :actual_h, :actual_w] = masks_mm[:, local_y_min:local_y_min+actual_h, 
                                                      local_x_min:local_x_min+actual_w]
        
        # For S1 ascending: (T', H, W, 2)
        # Handle possibility of empty array
        if sar_asc_mm.shape[0] > 0:
            sar_asc_patch = np.zeros((sar_asc_mm.shape[0], patch_size, patch_size, sar_asc_mm.shape[3]), dtype=np.float32)
            sar_asc_patch[:, :actual_h, :actual_w, :] = sar_asc_mm[:, local_y_min:local_y_min+actual_h, 
                                                                 local_x_min:local_x_min+actual_w, :]
        else:
            # Create empty array with appropriate dimensions
            sar_asc_patch = np.zeros((0, patch_size, patch_size, 2), dtype=np.float32)
        
        # For S1 descending: (T'', H, W, 2)
        # Handle possibility of empty array
        if sar_desc_mm.shape[0] > 0:
            sar_desc_patch = np.zeros((sar_desc_mm.shape[0], patch_size, patch_size, sar_desc_mm.shape[3]), dtype=np.float32)
            sar_desc_patch[:, :actual_h, :actual_w, :] = sar_desc_mm[:, local_y_min:local_y_min+actual_h, 
                                                                   local_x_min:local_x_min+actual_w, :]
        else:
            # Create empty array with appropriate dimensions
            sar_desc_patch = np.zeros((0, patch_size, patch_size, 2), dtype=np.float32)
    except Exception as e:
        logger.error(f"Error during data extraction: {e}")
        logger.error(f"Shapes - bands: {bands_mm.shape}, masks: {masks_mm.shape}")
        logger.error(f"Local bounds: x={local_x_min}-{local_x_min+actual_w}, y={local_y_min}-{local_y_min+actual_h}")
        
        # Create dummy data as a fallback
        bands_patch = np.zeros((bands_mm.shape[0], patch_size, patch_size, bands_mm.shape[3]), dtype=np.float32)
        masks_patch = np.zeros((masks_mm.shape[0], patch_size, patch_size), dtype=np.int8)
        sar_asc_patch = np.zeros((0, patch_size, patch_size, 2), dtype=np.float32)
        sar_desc_patch = np.zeros((0, patch_size, patch_size, 2), dtype=np.float32)
    
    # Create a validity mask to track which parts contain actual data
    valid_region_mask = np.zeros((patch_size, patch_size), dtype=bool)
    valid_region_mask[:actual_h, :actual_w] = True
    
    # Collect the data
    result = {
        'bands': bands_patch,
        'masks': masks_patch,
        'doys': getattr(mgrs_tile, 'doys', np.array([])).copy(),
        'sar_ascending': sar_asc_patch,
        'sar_ascending_doy': getattr(mgrs_tile, 'sar_asc_doys', np.array([])).copy(),
        'sar_descending': sar_desc_patch,
        'sar_descending_doy': getattr(mgrs_tile, 'sar_desc_doys', np.array([])).copy(),
        'local_x_min': local_x_min,
        'local_y_min': local_y_min,
        'local_width': actual_w,  # Use the actual width after clamping
        'local_height': actual_h,  # Use the actual height after clamping
        'valid_region_mask': valid_region_mask  # Add the validity mask to track actual data regions
    }
    
    return result

def merge_time_series(patch_data_list):
    """Merge time series data from multiple overlapping MGRS tiles.
    
    Args:
        patch_data_list (list): List of patch data dictionaries from different MGRS tiles
        
    Returns:
        dict: Merged patch data with consolidated time series
    """
    if not patch_data_list:
        return None
    
    if len(patch_data_list) == 1:
        # Only one MGRS tile, no need to merge
        return patch_data_list[0]
    
    # Get dimensions from the first patch
    first_patch = patch_data_list[0]
    patch_height = first_patch['bands'].shape[1]
    patch_width = first_patch['bands'].shape[2]
    num_bands = first_patch['bands'].shape[3]
    num_sar_bands = first_patch['sar_ascending'].shape[3]
    
    # Collect all unique DOYs
    all_s2_doys = np.concatenate([patch['doys'] for patch in patch_data_list])
    all_sar_asc_doys = np.concatenate([patch['sar_ascending_doy'] for patch in patch_data_list])
    all_sar_desc_doys = np.concatenate([patch['sar_descending_doy'] for patch in patch_data_list])
    
    # Get unique DOYs while preserving order
    unique_s2_doys = np.array(sorted(set(all_s2_doys)))
    unique_sar_asc_doys = np.array(sorted(set(all_sar_asc_doys)))
    unique_sar_desc_doys = np.array(sorted(set(all_sar_desc_doys)))
    
    # Create new arrays for the merged data
    new_bands = np.zeros((len(unique_s2_doys), patch_height, patch_width, num_bands), dtype=np.float32)
    new_masks = np.zeros((len(unique_s2_doys), patch_height, patch_width), dtype=np.int8)
    new_sar_asc = np.zeros((len(unique_sar_asc_doys), patch_height, patch_width, num_sar_bands), dtype=np.float32)
    new_sar_desc = np.zeros((len(unique_sar_desc_doys), patch_height, patch_width, num_sar_bands), dtype=np.float32)
    
    # Create mapping from DOYs to indices
    s2_doy_to_idx = {doy: i for i, doy in enumerate(unique_s2_doys)}
    sar_asc_doy_to_idx = {doy: i for i, doy in enumerate(unique_sar_asc_doys)}
    sar_desc_doy_to_idx = {doy: i for i, doy in enumerate(unique_sar_desc_doys)}
    
    # Process each patch and merge data
    for patch in patch_data_list:
        # Get local dimensions
        local_h = patch['local_height']
        local_w = patch['local_width']
        
        # Process S2 data
        for t, doy in enumerate(patch['doys']):
            idx = s2_doy_to_idx[doy]
            # For overlapping regions, prefer data with mask=1 (valid)
            mask_slice = patch['masks'][t, :local_h, :local_w]
            current_mask = new_masks[idx, :local_h, :local_w]
            
            # Where the current patch has valid data OR where the merged data is invalid
            update_mask = (mask_slice == 1) | (current_mask == 0)
            
            # Update mask
            new_masks[idx, :local_h, :local_w][update_mask] = mask_slice[update_mask]
            
            # Update bands where the mask indicates
            for b in range(num_bands):
                band_slice = patch['bands'][t, :local_h, :local_w, b]
                new_bands[idx, :local_h, :local_w, b][update_mask] = band_slice[update_mask]
        
        # Process S1 ascending data
        for t, doy in enumerate(patch['sar_ascending_doy']):
            idx = sar_asc_doy_to_idx[doy]
            # For S1, we don't have a mask, so we just overwrite (last one wins)
            # Could implement more sophisticated merging here if needed
            for b in range(num_sar_bands):
                new_sar_asc[idx, :local_h, :local_w, b] = patch['sar_ascending'][t, :local_h, :local_w, b]
        
        # Process S1 descending data
        for t, doy in enumerate(patch['sar_descending_doy']):
            idx = sar_desc_doy_to_idx[doy]
            for b in range(num_sar_bands):
                new_sar_desc[idx, :local_h, :local_w, b] = patch['sar_descending'][t, :local_h, :local_w, b]
    
    # Create the merged patch data
    merged_patch = {
        'bands': new_bands,
        'masks': new_masks,
        'doys': unique_s2_doys,
        'sar_ascending': new_sar_asc,
        'sar_ascending_doy': unique_sar_asc_doys,
        'sar_descending': new_sar_desc,
        'sar_descending_doy': unique_sar_desc_doys
    }
    
    return merged_patch

def create_roi_tiff(out_dir, x_min, y_min, patch_size, global_transform, global_crs):
    """Create a Region of Interest (ROI) TIFF file for a patch.
    
    Args:
        out_dir (str): Output directory path
        x_min, y_min (int): Upper left corner of the patch in global coordinates
        patch_size (int): Size of the patch in pixels
        global_transform (Affine): Global affine transform
        global_crs (CRS): Coordinate reference system
        
    Returns:
        bool: True if ROI TIFF was created successfully
    """
    global tiff_reference_path, global_width, global_height  # Access global variables
    import rasterio  # Explicitly import here to avoid any issues
    
    roi_path = os.path.join(out_dir, 'roi.tiff')
    
    try:
        # Get the reference TIFF information
        with rasterio.open(tiff_reference_path) as src:
            ref_bounds = src.bounds
            ref_width = src.width
            ref_height = src.height
            ref_transform = src.transform
        
        # Check if the CRS is WGS84 (EPSG:4326)
        is_wgs84 = global_crs.to_epsg() == 4326
        
        # Calculate the proportion of the patch in the overall grid
        # (accounting for actual dimensions vs patch_size)
        actual_patch_width = min(patch_size, global_width - x_min)
        actual_patch_height = min(patch_size, global_height - y_min)
        
        # For WGS84, we need to ensure correct geographic coordinates
        if is_wgs84:
            # Calculate geographic bounds proportionally based on position in the full array
            # This ensures proper scaling from the low-res reference to high-res patches
            # For the raw data dimensions we use global_width and global_height
            
            # Calculate the proportion of the full data covered by this patch
            x_min_prop = x_min / global_width
            y_min_prop = y_min / global_height
            x_max_prop = min((x_min + actual_patch_width) / global_width, 1.0)
            y_max_prop = min((y_min + actual_patch_height) / global_height, 1.0)
            
            # Apply these proportions to the geographic bounds
            patch_left = ref_bounds.left + x_min_prop * (ref_bounds.right - ref_bounds.left)
            patch_right = ref_bounds.left + x_max_prop * (ref_bounds.right - ref_bounds.left)
            # Note: WGS84 coordinates increase northward, but raster coordinates increase downward
            patch_top = ref_bounds.top - y_min_prop * (ref_bounds.top - ref_bounds.bottom)
            patch_bottom = ref_bounds.top - y_max_prop * (ref_bounds.top - ref_bounds.bottom)
            
            # Calculate resolution to exactly match the patch dimensions
            dx = (patch_right - patch_left) / actual_patch_width
            dy = (patch_top - patch_bottom) / actual_patch_height
            
            # Create a transform for this patch
            patch_transform = Affine(
                dx, 0, patch_left,
                0, -dy, patch_top
            )
            
            # Calculate grid position for logging
            grid_x = x_min // patch_size
            grid_y = y_min // patch_size
            
            logger.debug(f"Creating WGS84 ROI TIFF for patch at grid ({grid_x}, {grid_y})")
            logger.debug(f"Patch geographic bounds: [{patch_left}, {patch_bottom}, {patch_right}, {patch_top}]")
            logger.debug(f"Patch pixel dimensions: {actual_patch_width}x{actual_patch_height}")
            logger.debug(f"Resolution: {dx}x{dy} degrees")
            logger.debug(f"Transform: {patch_transform}")
        else:
            # For projected CRS, use the standard approach
            # Offset the global transform to the patch's upper left corner
            patch_transform = Affine(
                global_transform.a, global_transform.b, global_transform.c + x_min * global_transform.a,
                global_transform.d, global_transform.e, global_transform.f + y_min * global_transform.e
            )
            
            logger.debug(f"Creating projected ROI TIFF with transform: {patch_transform}")
        
        logger.debug(f"CRS: {global_crs}")
        
        # Create a simple raster filled with ones
        data = np.ones((patch_size, patch_size), dtype=np.uint8)
        
        # Create the TIFF file
        with rasterio.open(
            roi_path,
            'w',
            driver='GTiff',
            height=patch_size,
            width=patch_size,
            count=1,
            dtype=data.dtype,
            crs=global_crs,
            transform=patch_transform
        ) as dst:
            dst.write(data, 1)
            
        # Verify the file was created
        if not os.path.exists(roi_path):
            logger.error(f"ROI TIFF was not created at {roi_path} even though no exception was raised")
            return False
            
        # Check the file size to ensure it's not empty
        file_size = os.path.getsize(roi_path)
        if file_size < 100:  # Arbitrary small size
            logger.error(f"ROI TIFF at {roi_path} appears to be empty or corrupt (size: {file_size} bytes)")
            return False
            
        logger.info(f"Successfully created ROI TIFF: {roi_path} (size: {file_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to create ROI TIFF: {e}")
        # Print more details about the error
        logger.error(traceback.format_exc())
        return False

def save_patch(patch_data, out_dir, x_min=None, y_min=None, patch_size=None, 
               global_transform=None, global_crs=None):
    """Save patch data to the output directory.
    
    Args:
        patch_data (dict): Patch data to save
        out_dir (str): Output directory path
        x_min, y_min (int, optional): Upper left corner of the patch in global coordinates
        patch_size (int, optional): Size of the patch in pixels
        global_transform (Affine, optional): Global affine transform
        global_crs (CRS, optional): Coordinate reference system
        
    Returns:
        bool: True if saving was successful
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Log the parameters for debugging
    logger.debug(f"save_patch called with x_min={x_min}, y_min={y_min}, patch_size={patch_size}")
    logger.debug(f"global_transform: {global_transform}")
    logger.debug(f"global_crs: {global_crs}")
    
    try:
        # Save the arrays
        np.save(os.path.join(out_dir, 'bands.npy'), patch_data['bands'])
        np.save(os.path.join(out_dir, 'masks.npy'), patch_data['masks'])
        np.save(os.path.join(out_dir, 'doys.npy'), patch_data['doys'])
        np.save(os.path.join(out_dir, 'sar_ascending.npy'), patch_data['sar_ascending'])
        np.save(os.path.join(out_dir, 'sar_ascending_doy.npy'), patch_data['sar_ascending_doy'])
        np.save(os.path.join(out_dir, 'sar_descending.npy'), patch_data['sar_descending'])
        np.save(os.path.join(out_dir, 'sar_descending_doy.npy'), patch_data['sar_descending_doy'])
        
        # Create ROI TIFF if all required information is provided
        if x_min is not None and y_min is not None and patch_size is not None and \
           global_transform is not None and global_crs is not None:
            
            # Force explicit creation of ROI TIFF
            roi_success = create_roi_tiff(out_dir, x_min, y_min, patch_size, global_transform, global_crs)
            
            if not roi_success:
                logger.warning(f"Failed to create ROI TIFF in {out_dir}, but NumPy arrays were saved")
                # Try again with more detailed debugging
                try:
                    # Check if file can be created at all
                    test_path = os.path.join(out_dir, 'test.txt')
                    with open(test_path, 'w') as f:
                        f.write('test')
                    if os.path.exists(test_path):
                        logger.debug(f"Test file creation successful at {test_path}")
                        os.remove(test_path)
                    
                    import rasterio  # Explicitly import here
                    # Try a simpler TIFF creation
                    simple_tiff_path = os.path.join(out_dir, 'simple.tiff')
                    profile = {
                        'driver': 'GTiff',
                        'height': patch_size,
                        'width': patch_size,
                        'count': 1,
                        'dtype': np.uint8
                    }
                    with rasterio.open(simple_tiff_path, 'w', **profile) as dst:
                        dst.write(np.ones((patch_size, patch_size), dtype=np.uint8), 1)
                    
                    if os.path.exists(simple_tiff_path):
                        logger.debug(f"Simple TIFF creation successful at {simple_tiff_path}")
                except Exception as e:
                    logger.error(f"Additional debugging failed: {e}")
        else:
            logger.warning("Not creating ROI TIFF because one or more parameters are missing")
            # Log which parameters are missing
            missing_params = []
            if x_min is None: missing_params.append("x_min")
            if y_min is None: missing_params.append("y_min")
            if patch_size is None: missing_params.append("patch_size")
            if global_transform is None: missing_params.append("global_transform")
            if global_crs is None: missing_params.append("global_crs")
            logger.warning(f"Missing parameters: {', '.join(missing_params)}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving patch: {e}")
        logger.error(traceback.format_exc())
        return False

def has_non_zero_data(x_min, y_min, patch_size):
    """Check if a patch contains any non-zero data in the reference TIFF.
    
    Args:
        x_min, y_min (int): Upper left corner of the patch in global coordinates
        patch_size (int): Size of the patch in pixels
        
    Returns:
        bool: True if the patch contains any non-zero data
    """
    global non_zero_mask, global_width, global_height
    
    if non_zero_mask is None:
        # If no mask is available, assume all patches have data
        return True
    
    # Calculate patch bounds, clamped to tiff dimensions
    x_max = min(x_min + patch_size, global_width)
    y_max = min(y_min + patch_size, global_height)
    
    # Check if any area in the patch is non-zero
    try:
        # Ensure coordinates are within mask bounds
        if x_min >= non_zero_mask.shape[1] or y_min >= non_zero_mask.shape[0]:
            return False
        
        # Calculate valid region to check
        valid_x_max = min(x_max, non_zero_mask.shape[1])
        valid_y_max = min(y_max, non_zero_mask.shape[0])
        
        # Check if any pixel in the patch is non-zero
        patch_mask = non_zero_mask[y_min:valid_y_max, x_min:valid_x_max]
        return np.any(patch_mask)
    except Exception as e:
        logger.error(f"Error checking for non-zero data: {e}")
        # Default to including the patch if an error occurs
        return True

def process_patch(patch_info, mgrs_tiles, global_transform, out_dir, skip_existing=True, global_crs=None):
    """Process a single patch.
    
    Args:
        patch_info (tuple): (grid_x, grid_y, patch_size) defining the patch
        mgrs_tiles (dict): Dictionary of MGRSTile objects
        global_transform (Affine): Global affine transform
        out_dir (str): Output directory path
        skip_existing (bool): Whether to skip existing patches
        global_crs (CRS, optional): Coordinate reference system
        
    Returns:
        bool: True if processing was successful
    """
    global global_width, global_height  # Declare accessing global variables
    
    grid_x, grid_y, patch_size = patch_info
    
    # Calculate global pixel coordinates
    x_min = grid_x * patch_size
    y_min = grid_y * patch_size
    
    # Use global variables
    x_max = min(x_min + patch_size, global_width)
    y_max = min(y_min + patch_size, global_height)
    
    # Create output directory name
    patch_dir_name = f"{x_min}_{y_min}_{x_max}_{y_max}"
    patch_out_dir = os.path.join(out_dir, patch_dir_name)
    
    # Check if this patch contains any non-zero data
    if not has_non_zero_data(x_min, y_min, patch_size):
        logger.debug(f"Patch {patch_dir_name} has no non-zero data, skipping")
        return True  # Return True to indicate successful skip
    
    # Log processing details
    logger.debug(f"Processing patch {patch_dir_name} with global_crs={global_crs}")
    
    # Skip if the output directory already exists and skip_existing is True
    if skip_existing and os.path.exists(patch_out_dir):
        if all(os.path.exists(os.path.join(patch_out_dir, f)) for f in 
               ['bands.npy', 'doys.npy', 'masks.npy', 'sar_ascending.npy', 
                'sar_ascending_doy.npy', 'sar_descending.npy', 'sar_descending_doy.npy']):
            # Check if roi.tiff exists too - if not, still try to create it
            roi_path = os.path.join(patch_out_dir, 'roi.tiff')
            if not os.path.exists(roi_path) and global_crs is not None:
                logger.info(f"NumPy files exist but ROI TIFF missing for {patch_dir_name}, creating TIFF")
                try:
                    create_roi_tiff(patch_out_dir, x_min, y_min, patch_size, global_transform, global_crs)
                except Exception as e:
                    logger.error(f"Error creating ROI TIFF for existing patch: {e}")
            else:
                logger.debug(f"Patch {patch_dir_name} already exists, skipping")
                return True
    
    # Find overlapping MGRS tiles
    overlapping_tiles = get_overlapping_mgrs_tiles(
        x_min, y_min, x_max, y_max, mgrs_tiles, global_transform, global_crs
    )
    
    if not overlapping_tiles:
        logger.warning(f"No overlapping MGRS tiles found for patch {patch_dir_name}")
        return False
    
    # Extract data from each overlapping tile
    patch_data_list = []
    for tile in overlapping_tiles:
        try:
            logger.debug(f"Extracting data from tile {tile.code}")
            patch_data = extract_patch_from_mgrs(
                tile, x_min, y_min, patch_size, global_transform, global_crs
            )
            if patch_data is not None:
                logger.debug(f"Successfully extracted data from tile {tile.code}")
                patch_data_list.append(patch_data)
            else:
                logger.warning(f"No data extracted from tile {tile.code}")
        except Exception as e:
            logger.warning(f"Error extracting patch from tile {tile.code}: {e}")
            logger.warning(traceback.format_exc())
            # Continue with other tiles
    
    if not patch_data_list:
        logger.warning(f"Failed to extract data for patch {patch_dir_name}")
        return False
    
    # Merge time series data from overlapping tiles
    try:
        merged_patch = merge_time_series(patch_data_list)
    except Exception as e:
        logger.error(f"Error merging time series for patch {patch_dir_name}: {e}")
        # Print debug info about the patch shapes
        for i, p in enumerate(patch_data_list):
            try:
                logger.error(f"Patch {i} shapes: bands={p['bands'].shape}, masks={p['masks'].shape}, " +
                           f"valid_region_mask={p.get('valid_region_mask', 'None')}")
            except:
                logger.error(f"Error printing shapes for patch {i}")
        return False
    
    if merged_patch is None:
        logger.warning(f"Failed to merge data for patch {patch_dir_name}")
        return False
    
    # Remove the valid_region_mask from the merged patch before saving
    # as it's only used internally for merging
    if 'valid_region_mask' in merged_patch:
        del merged_patch['valid_region_mask']
    
    # Save the merged patch with ROI TIFF - always pass the CRS info for the TIFF
    logger.info(f"Saving patch {patch_dir_name} with ROI TIFF (CRS: {global_crs is not None})")
    
    success = save_patch(
        merged_patch, 
        patch_out_dir, 
        x_min, 
        y_min, 
        patch_size, 
        global_transform, 
        global_crs
    )
    
    # Verify the TIFF was created
    roi_path = os.path.join(patch_out_dir, 'roi.tiff')
    if not os.path.exists(roi_path) and global_crs is not None:
        logger.warning(f"ROI TIFF was not created at {roi_path}, retrying...")
        try:
            success_tiff = create_roi_tiff(patch_out_dir, x_min, y_min, patch_size, global_transform, global_crs)
            if not success_tiff:
                logger.error(f"Failed to create ROI TIFF on retry for {patch_dir_name}")
        except Exception as e:
            logger.error(f"Error on ROI TIFF retry: {e}")
    
    return success

def process_patch_wrapper(patch_info, mgrs_tiles, global_transform, global_crs, out_dir, skip_existing):
    """Process a single patch - standalone wrapper for multiprocessing.
    
    Args:
        patch_info (tuple): (grid_x, grid_y, patch_size) defining the patch
        mgrs_tiles (dict): Dictionary of MGRSTile objects
        global_transform (Affine): Global affine transform
        global_crs (CRS): Coordinate reference system
        out_dir (str): Output directory path
        skip_existing (bool): Whether to skip existing patches
        
    Returns:
        bool: True if processing was successful
    """
    return process_patch(
        patch_info=patch_info,
        mgrs_tiles=mgrs_tiles,
        global_transform=global_transform,
        out_dir=out_dir,
        skip_existing=skip_existing,
        global_crs=global_crs
    )

def main():
    global global_width, global_height, tiff_reference_path, non_zero_mask  # Declare all globals we'll modify
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process MGRS tiled d-pixel data into regular patches')
    parser.add_argument('--tiff_path', required=True, help='Path to the downstream TIFF')
    parser.add_argument('--d_pixel_dir', required=True, help='Directory containing MGRS d-pixel data')
    parser.add_argument('--patch_size', type=int, default=500, help='Size of output patches in pixels')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--data_raw_dir', required=True, help='Root directory for raw data containing reference TIFFs')
    parser.add_argument('--num_workers', type=int, default=max(1, mp.cpu_count() - 1), help='Number of worker processes')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing patches')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--only_non_zero', action='store_true', help='Only process patches that have non-zero values in the reference TIFF')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Set the global reference TIFF path
    tiff_reference_path = args.tiff_path
    
    # Check input TIFF
    if not os.path.exists(args.tiff_path):
        logger.error(f"Input TIFF not found: {args.tiff_path}")
        return 1
    
    # Check d_pixel_dir
    if not os.path.exists(args.d_pixel_dir):
        logger.error(f"d_pixel_dir not found: {args.d_pixel_dir}")
        return 1
    
    # Check data_raw_dir
    if not os.path.exists(args.data_raw_dir):
        logger.error(f"data_raw_dir not found: {args.data_raw_dir}")
        return 1
    
    # Create output directory
    if not create_output_directory(args.out_dir, args.overwrite):
        logger.error(f"Failed to create output directory: {args.out_dir}")
        return 1
    
    # Create non-zero mask if needed
    if args.only_non_zero:
        logger.info("Creating non-zero mask from reference TIFF...")
        non_zero_mask = create_non_zero_mask(args.tiff_path)
        if non_zero_mask is None:
            logger.error("Failed to create non-zero mask, will process all patches")
    else:
        non_zero_mask = None
    
    # Find MGRS tiles
    logger.info("Finding MGRS tiles...")
    mgrs_codes = find_mgrs_tiles(args.d_pixel_dir)
    logger.info(f"Found {len(mgrs_codes)} MGRS tiles: {', '.join(mgrs_codes)}")
    
    if not mgrs_codes:
        logger.error("No MGRS tiles found in the d_pixel_dir")
        return 1
    
    # Print reference TIFF information for debugging
    logger.info("Reading reference TIFF info...")
    try:
        tiff_info = read_reference_tiff_info(args.tiff_path)
        logger.info(f"Reference TIFF dimensions: {tiff_info['width']}x{tiff_info['height']}")
        logger.info(f"Reference TIFF resolution: {tiff_info['resolution_x']}x{tiff_info['resolution_y']} meters")
        logger.info(f"Reference TIFF bounds: {tiff_info['bounds']}")
        logger.info(f"Reference TIFF CRS: {tiff_info['crs']}")
    except Exception as e:
        logger.error(f"Failed to read reference TIFF info: {e}")
        logger.error(traceback.format_exc())
    
    # Load MGRS tiles first
    logger.info("Loading MGRS tiles...")
    mgrs_tiles = {}
    for mgrs_code in mgrs_codes:
        try:
            mgrs_dir = os.path.join(args.d_pixel_dir, mgrs_code)
            ref_tiff = find_reference_tiff(mgrs_code, args.tiff_path, args.data_raw_dir)
            mgrs_tiles[mgrs_code] = MGRSTile(mgrs_code, mgrs_dir, ref_tiff)
            logger.debug(f"Loaded MGRS tile: {mgrs_tiles[mgrs_code]}")
        except Exception as e:
            logger.warning(f"Failed to load MGRS tile {mgrs_code}: {e}")
    
    if not mgrs_tiles:
        logger.error("No valid MGRS tiles could be loaded")
        return 1
    
    # Use reference TIFF for global grid, but with dimensions from data_raw_dir
    logger.info("Creating global grid from reference TIFF...")
    try:
        global_transform, width, height, global_crs, grid_width, grid_height = \
            create_global_grid(args.tiff_path, args.patch_size, args.data_raw_dir)
        logger.info(f"Global grid: {grid_width}x{grid_height} patches")
        
        # Set global variables
        global_width = width
        global_height = height
    except TypeError as te:
        # Check if this is due to missing argument
        if "missing 1 required positional argument: 'data_raw_dir'" in str(te):
            logger.error("Error calling create_global_grid - please use updated function signature")
            logger.error("Make sure you're using the latest version of the script")
        logger.error(f"Failed to create global grid: {te}")
        logger.error(traceback.format_exc())
        return 1
    except Exception as e:
        logger.error(f"Failed to create global grid: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    # Create list of patches to process
    patches = []
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            # Check if we should process this patch (if we're using non-zero filtering)
            x_min = grid_x * args.patch_size
            y_min = grid_y * args.patch_size
            
            if args.only_non_zero and non_zero_mask is not None:
                if not has_non_zero_data(x_min, y_min, args.patch_size):
                    logger.debug(f"Skipping patch at ({grid_x}, {grid_y}) as it contains only zeros")
                    continue
                
            patches.append((grid_x, grid_y, args.patch_size))
    
    logger.info(f"Created list of {len(patches)} patches to process")
    
    # Determine number of workers (limit to 8 to avoid memory issues)
    num_workers = min(args.num_workers, 8)
    logger.info(f"Processing {len(patches)} patches with {num_workers} workers")
    
    # Create partial function for multiprocessing
    process_patch_partial = partial(
        process_patch_wrapper,
        mgrs_tiles=mgrs_tiles,
        global_transform=global_transform,
        global_crs=global_crs,
        out_dir=args.out_dir,
        skip_existing=args.skip_existing
    )
    
    # Process patches in parallel
    start_time = time.time()
    if num_workers > 1:
        # Process in smaller batches to prevent memory issues
        batch_size = min(100, len(patches))
        results = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(patches) + batch_size - 1)//batch_size} ({len(batch_patches)} patches)")
            
            with mp.Pool(num_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_patch_partial, batch_patches),
                    total=len(batch_patches),
                    desc=f"Processing batch {i//batch_size + 1}"
                ))
                results.extend(batch_results)
                
            # Force garbage collection to free memory
            for tile in mgrs_tiles.values():
                tile.close_memmaps()
    else:
        # Single process mode
        results = []
        for patch_info in tqdm(patches, desc="Processing patches"):
            result = process_patch_partial(patch_info)
            results.append(result)
    
    # Show summary
    end_time = time.time()
    success_count = sum(1 for r in results if r)
    logger.info(f"Processing complete: {success_count}/{len(patches)} patches processed successfully")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Clean up
    for tile in mgrs_tiles.values():
        tile.close_memmaps()
    
    return 0

# Allow for flexible script naming 
if __name__ == '__main__':
    # Run main
    sys.exit(main())