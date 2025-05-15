# btfm_tools/representation_loader.py
import numpy as np
import os
import logging
import rasterio
from pyproj import Transformer as PyprojTransformer, CRS as PyprojCRS

logger = logging.getLogger(__name__)

def get_latlon_for_pixel(row: int, col: int, ref_tiff_path: str):
    """
    Calculates the WGS84 latitude and longitude for the center of a given pixel.

    Args:
        row (int): The row index of the pixel (0-based).
        col (int): The column index of the pixel (0-based).
        ref_tiff_path (str): Path to the reference GeoTIFF.

    Returns:
        tuple: (latitude, longitude) or None if an error occurs.
    """
    geo_info = get_representation_geoinfo(ref_tiff_path)
    if not geo_info:
        logger.error(f"Could not get geoinfo from {ref_tiff_path}")
        return None

    if not (0 <= row < geo_info['height'] and 0 <= col < geo_info['width']):
        logger.error(f"Pixel ({row}, {col}) is out of bounds for TIFF dimensions ({geo_info['height']}, {geo_info['width']}).")
        return None

    # Get map coordinates (e.g., UTM) for the center of the pixel
    # Rasterio's transform maps (col, row) to (x, y) of the *top-left corner*
    # To get the center, add 0.5 to col and row
    map_x, map_y = geo_info['transform'] * (col + 0.5, row + 0.5)

    # Transform map coordinates to WGS84 (lat/lon)
    try:
        # Source CRS from the TIFF
        src_crs_pyproj = PyprojCRS.from_string(geo_info['crs'].to_wkt())
        # Target CRS is WGS84
        tgt_crs_pyproj = PyprojCRS.from_epsg(4326)

        transformer = PyprojTransformer.from_crs(src_crs_pyproj, tgt_crs_pyproj, always_xy=True)
        lon, lat = transformer.transform(map_x, map_y)
        return lat, lon
    except Exception as e:
        logger.error(f"Error transforming coordinates for pixel ({row}, {col}): {e}")
        return None

def get_all_latlon_coordinates(ref_tiff_path: str):
    """
    Generates a 2D array of (latitude, longitude) for the center of each pixel
    in the reference TIFF.

    Args:
        ref_tiff_path (str): Path to the reference GeoTIFF.

    Returns:
        np.ndarray: An array of shape (H, W, 2) where the last dimension
                    contains [latitude, longitude], or None if an error occurs.
    """
    geo_info = get_representation_geoinfo(ref_tiff_path)
    if not geo_info:
        logger.error(f"Could not get geoinfo from {ref_tiff_path}")
        return None

    H, W = geo_info['height'], geo_info['width']
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    
    # Get map coordinates for all pixel centers
    # Add 0.5 for pixel centers
    xs, ys = geo_info['transform'] * (cols.flatten() + 0.5, rows.flatten() + 0.5)

    try:
        src_crs_pyproj = PyprojCRS.from_string(geo_info['crs'].to_wkt())
        tgt_crs_pyproj = PyprojCRS.from_epsg(4326)
        transformer = PyprojTransformer.from_crs(src_crs_pyproj, tgt_crs_pyproj, always_xy=True)
        
        lons, lats = transformer.transform(xs, ys)
        
        # Reshape to (H, W, 2) with [lat, lon]
        coordinates = np.dstack((lats.reshape(H, W), lons.reshape(H, W)))
        return coordinates
    except Exception as e:
        logger.error(f"Error transforming all coordinates: {e}")
        return None
        
def load_representation(npy_file_path: str) -> np.ndarray:
    """
    Loads a BTFM representation from a .npy file.

    Args:
        npy_file_path (str): Absolute path to the .npy representation file.

    Returns:
        np.ndarray: The loaded representation.
    
    Raises:
        FileNotFoundError: If the .npy file does not exist.
        ValueError: If the file is not a valid NumPy file.
    """
    if not os.path.exists(npy_file_path):
        logger.error(f"Representation file not found: {npy_file_path}")
        raise FileNotFoundError(f"Representation file not found: {npy_file_path}")
    try:
        representation = np.load(npy_file_path)
        logger.info(f"Loaded representation from {npy_file_path} with shape {representation.shape}")
        return representation
    except Exception as e:
        logger.error(f"Failed to load NumPy file {npy_file_path}: {e}")
        raise ValueError(f"Failed to load NumPy file {npy_file_path}: {e}")

def get_representation_geoinfo(ref_tiff_path: str):
    """
    Gets georeferencing information (transform, CRS, width, height) from a reference TIFF.
    This can be used to correctly geolocate pixels in the NumPy representation.

    Args:
        ref_tiff_path (str): Path to the reference GeoTIFF (e.g., the roi.tiff used for processing).

    Returns:
        dict: A dictionary containing 'transform', 'crs', 'width', 'height'.
              Returns None if the TIFF cannot be read.
    """
    try:
        import rasterio
        with rasterio.open(ref_tiff_path) as src:
            info = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds
            }
            logger.info(f"Loaded geoinfo from {ref_tiff_path}: CRS={info['crs']}, Size=({info['width']}x{info['height']})")
            return info
    except ImportError:
        logger.error("rasterio library is not installed. Cannot get geoinfo.")
        return None
    except Exception as e:
        logger.error(f"Failed to read reference TIFF {ref_tiff_path}: {e}")
        return None