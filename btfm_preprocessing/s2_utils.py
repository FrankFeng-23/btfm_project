#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_utils.py

Shared S2 helpers: tiff_to_polygon, subset_tiff_with_polygon, download_tile
No local environment variables here. The caller sets them.
"""

import os
CLOUD_COVERAGE_THRESHOLD = os.environ.get("CLOUD_COVERAGE_THRESHOLD", 90)
import json
import logging
import asyncio
from pathlib import Path

import aiohttp
import pyproj
import mgrs
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import transform
from osgeo import gdal, osr
from pystac_client import Client
import shutil
# Import the datetime module
from datetime import datetime

logging.basicConfig(level=logging.INFO)

band_assets = [
    "red", "blue", "green", "nir", "nir08",
    "rededge1", "rededge2", "rededge3", "swir16", "swir22", "scl"
]

# def tiff_to_polygon(tiff_path, densify_pts=21):
#     """
#     Convert a GeoTIFF file to a polygon representing its boundary.
#     Uses pyproj and shapely for coordinate transformation instead of GDAL OSR.
#     Includes extensive error checking.
#     """
#     import os
#     import pyproj
#     from shapely.geometry import Polygon
#     from shapely.ops import transform
#     import logging

#     # Check if file exists
#     if not os.path.exists(tiff_path):
#         raise FileNotFoundError(f"TIFF file does not exist: {tiff_path}")

#     # Open the TIFF file with GDAL
#     ds = gdal.Open(tiff_path, gdal.GA_ReadOnly)
#     if ds is None:
#         raise IOError(f"Cannot open {tiff_path} with GDAL. File might be corrupt or not a GeoTIFF.")

#     # Check if the file has georeferencing information
#     gt = ds.GetGeoTransform()
#     if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
#         raise ValueError(f"TIFF file {tiff_path} does not have valid georeferencing information.")

#     # Get projection as WKT
#     proj_wkt = ds.GetProjection()
#     if not proj_wkt:
#         raise ValueError(f"TIFF file {tiff_path} does not have valid projection information.")

#     # Get dimensions
#     xsize = ds.RasterXSize
#     ysize = ds.RasterYSize
#     if xsize <= 0 or ysize <= 0:
#         raise ValueError(f"TIFF file {tiff_path} has invalid dimensions: {xsize}x{ysize}")

#     # Calculate corners
#     left = gt[0]
#     top = gt[3]
#     right = left + gt[1]*xsize
#     bottom = top + gt[5]*ysize

#     # Close dataset to free resources
#     ds = None

#     # Log diagnostic info
#     logging.info(f"TIFF info: size={xsize}x{ysize}, bbox=({left},{top}) - ({right},{bottom})")
#     logging.info(f"Projection: {proj_wkt[:100]}...")  # Print first 100 chars of projection

#     try:
#         # Try to create CRS from WKT for source projection
#         src_crs = pyproj.CRS.from_wkt(proj_wkt)
#         # Target is WGS84 (lon/lat)
#         tgt_crs = pyproj.CRS.from_epsg(4326)

#         # Create transformer
#         transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

#         # Function to transform coordinates
#         def transform_fn(x, y):
#             return transformer.transform(x, y)

#         # Create boundary points (densified)
#         points = []

#         # Top edge (left to right)
#         for i in range(densify_pts):
#             t = i / (densify_pts - 1)
#             x = left + t * (right - left)
#             y = top
#             points.append((x, y))

#         # Right edge (top to bottom)
#         for i in range(1, densify_pts):
#             t = i / (densify_pts - 1)
#             x = right
#             y = top + t * (bottom - top)
#             points.append((x, y))

#         # Bottom edge (right to left)
#         for i in range(1, densify_pts):
#             t = i / (densify_pts - 1)
#             x = right - t * (right - left)
#             y = bottom
#             points.append((x, y))

#         # Left edge (bottom to top)
#         for i in range(1, densify_pts - 1):
#             t = i / (densify_pts - 1)
#             x = left
#             y = bottom - t * (bottom - top)
#             points.append((x, y))

#         # Create polygon in source CRS
#         poly_src = Polygon(points)
#         # Swap longitude and latitude
#         # poly_src = Polygon([(lon, lat) for lat, lon in poly_src.exterior.coords])

#         # Transform to target CRS using shapely's transform
#         poly_tgt = transform(transform_fn, poly_src)

#         return poly_tgt

#     except Exception as e:
#         # If any step fails, log detailed error and raise
#         logging.error(f"Error creating polygon: {e}")
#         raise RuntimeError(f"Failed to create polygon from TIFF: {e}")

def tiff_to_polygon(tiff_path, densify_pts=21):
    """
    Convert a GeoTIFF file to polygons representing the boundaries of non-zero values.
    Ignores areas where pixel values are zero.
    Uses pyproj and shapely for coordinate transformation instead of GDAL OSR.
    Includes polygon simplification to avoid API size limits.
    Includes extensive error checking.
    Returns a simplified polygon of non-zero areas.
    """
    import os
    import numpy as np
    import pyproj
    from osgeo import gdal, ogr
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform, unary_union
    import logging
 
    # Check if file exists
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF file does not exist: {tiff_path}")
 
    # Open the TIFF file with GDAL
    ds = gdal.Open(tiff_path, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"Cannot open {tiff_path} with GDAL. File might be corrupt or not a GeoTIFF.")
 
    # Check if the file has georeferencing information
    gt = ds.GetGeoTransform()
    if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"TIFF file {tiff_path} does not have valid georeferencing information.")
 
    # Get projection as WKT
    proj_wkt = ds.GetProjection()
    if not proj_wkt:
        raise ValueError(f"TIFF file {tiff_path} does not have valid projection information.")
 
    # Get dimensions
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    if xsize <= 0 or ysize <= 0:
        raise ValueError(f"TIFF file {tiff_path} has invalid dimensions: {xsize}x{ysize}")
 
    # Calculate appropriate downsample factor for large images
    # This helps reduce processing time and memory usage
    max_dimension = 1000  # Maximum dimension for processing
    downsample_factor = max(1, max(xsize, ysize) // max_dimension)
    
    # Read downsampled raster data
    band = ds.GetRasterBand(1)
    if downsample_factor > 1:
        data = band.ReadAsArray(
            0, 0, xsize, ysize, 
            buf_xsize=xsize // downsample_factor, 
            buf_ysize=ysize // downsample_factor
        )
        # Adjust geotransform for downsampled data
        downsampled_gt = list(gt)
        downsampled_gt[1] *= downsample_factor  # Pixel width
        downsampled_gt[5] *= downsample_factor  # Pixel height
        working_gt = tuple(downsampled_gt)
        working_xsize = xsize // downsample_factor
        working_ysize = ysize // downsample_factor
    else:
        data = band.ReadAsArray()
        working_gt = gt
        working_xsize = xsize
        working_ysize = ysize
    
    # Create binary mask of non-zero pixels
    mask = (data != 0).astype(np.uint8)
    
    # Log diagnostic info
    logging.info(f"TIFF info: size={xsize}x{ysize}, downsampled={working_xsize}x{working_ysize}, non-zero pixels: {np.sum(mask)}")
    logging.info(f"Projection: {proj_wkt[:100]}...")  # Print first 100 chars of projection
 
    try:
        # Try to create CRS from WKT for source projection
        src_crs = pyproj.CRS.from_wkt(proj_wkt)
        # Target is WGS84 (lon/lat)
        tgt_crs = pyproj.CRS.from_epsg(4326)
 
        # Create transformer
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
 
        # Function to transform coordinates
        def transform_fn(x, y):
            return transformer.transform(x, y)
 
        # If no non-zero pixels found, return a default small polygon
        if np.sum(mask) == 0:
            logging.warning("No non-zero pixels found in the raster.")
            # Create a small polygon at the origin of the image
            x = working_gt[0]
            y = working_gt[3]
            small_poly = Polygon([(x, y), (x+1, y), (x+1, y-1), (x, y-1)])
            return transform(transform_fn, small_poly)
 
        # Create a temporary memory raster to hold the mask
        driver = gdal.GetDriverByName('MEM')
        mask_ds = driver.Create('', working_xsize, working_ysize, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(working_gt)
        mask_ds.SetProjection(proj_wkt)
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.WriteArray(mask)
        
        # Create a vector layer in memory to hold polygons
        drv = ogr.GetDriverByName('MEMORY')
        outds = drv.CreateDataSource('memdata')
        outlyr = outds.CreateLayer('polygons', None, ogr.wkbPolygon)
        
        # Add an ID field
        field_defn = ogr.FieldDefn('ID', ogr.OFTInteger)
        outlyr.CreateField(field_defn)
        
        # Polygonize the raster band
        gdal.Polygonize(mask_band, None, outlyr, 0, [], callback=None)
        
        # Extract polygons from the layer
        polygons = []
        for feature in outlyr:
            # Get the geometry
            geom = feature.GetGeometryRef()
            if geom is not None:
                # Convert OGR geometry to WKT
                wkt_str = geom.ExportToWkt()
                try:
                    # Try to import shapely.wkt for newer versions of Shapely
                    try:
                        from shapely.wkt import loads
                        shapely_poly = loads(wkt_str)
                    except ImportError:
                        # Fall back to older style import for earlier versions of Shapely
                        from shapely import wkt
                        shapely_poly = wkt.loads(wkt_str)
                    
                    # Check if this is a valid polygon with significant area
                    if shapely_poly.is_valid and not shapely_poly.is_empty and shapely_poly.area > 0:
                        polygons.append(shapely_poly)
                except Exception as e:
                    logging.warning(f"Error converting polygon: {e}")
        
        # Clean up
        mask_ds = None
        outds = None
        
        if not polygons:
            logging.warning("No valid polygons extracted from non-zero areas.")
            # Create a polygon from the entire image extent as fallback
            x_min = working_gt[0]
            y_max = working_gt[3]
            x_max = x_min + working_gt[1] * working_xsize
            y_min = y_max + working_gt[5] * working_ysize
            poly = Polygon([(x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)])
            poly_wgs84 = transform(transform_fn, poly)
            ds = None
            return poly_wgs84
        
        # Merge all polygons into one
        merged_poly = unary_union(polygons)
        
        # Calculate appropriate simplification tolerance
        # This is a critical step to ensure the polygon isn't too complex
        # Higher values = more simplification
        
        # Calculate the bounding box size in projected units
        bounds = merged_poly.bounds
        bbox_width = bounds[2] - bounds[0]
        bbox_height = bounds[3] - bounds[1]
        
        # Set simplification tolerance based on the size of the bounding box
        # This is typically a fraction (like 1/1000) of the largest dimension
        simplify_tolerance = max(bbox_width, bbox_height) / 500
        
        # Apply simplification with preservation of topology
        simplified_poly = merged_poly.simplify(simplify_tolerance, preserve_topology=True)
        
        # If the polygon is still very complex, apply more aggressive simplification
        if simplified_poly.wkt.count(',') > 1000:  # Crude measure of complexity
            # More aggressive simplification
            simplify_tolerance = max(bbox_width, bbox_height) / 200
            simplified_poly = merged_poly.simplify(simplify_tolerance, preserve_topology=True)
            
            # If still too complex, apply even more aggressive simplification
            if simplified_poly.wkt.count(',') > 500:
                simplify_tolerance = max(bbox_width, bbox_height) / 100
                simplified_poly = merged_poly.simplify(simplify_tolerance, preserve_topology=True)
        
        # Transform to target CRS using shapely's transform
        poly_wgs84 = transform(transform_fn, simplified_poly)
        
        # Clean up
        ds = None
        
        return poly_wgs84
 
    except Exception as e:
        # If any step fails, log detailed error and raise
        logging.error(f"Error creating polygon: {e}")
        ds = None
        raise RuntimeError(f"Failed to create polygon from TIFF: {e}")


def subset_tiff_with_polygon(input_tiff: str, output_tiff: str, polygon: Polygon):
    """
    Use GDAL Warp to perform vector clipping on input_tiff, keeping only the intersection of the valid extent of input_tiff and the polygon extent.
    The resolution and projection of the output image remain consistent with the original image.

    :param input_tiff: Path to the original image
    :param output_tiff: Output path for the clipped image
    :param polygon: Polygon (lon/lat order under EPSG:4326), shapely Polygon object
    :return: Boolean value indicating whether clipping was necessary (True means clipping was performed, False means the original and clipped image sizes are the same)
    """
    import os
    import logging
    import json
    import shutil
    import uuid
    from osgeo import gdal, osr
    from shapely.geometry import Polygon
    
    # Handle different shapely versions (1.x vs 2.x)
    try:
        # Shapely 2.x
        from shapely import mapping
    except ImportError:
        try:
            # Shapely 1.x
            from shapely.geometry.mapping import mapping
        except ImportError:
            # Last resort - implement a simple version for basic geometries
            def mapping(geom):
                if isinstance(geom, Polygon):
                    coords = [list(p) for p in geom.exterior.coords]
                    return {"type": "Polygon", "coordinates": [coords]}

    logging.info(f"[subset_tiff_with_polygon] Start subsetting {input_tiff}")

    # Generate unique IDs for temporary files to avoid conflicts in parallel processing
    process_id = os.getpid()
    unique_id = str(uuid.uuid4())[:8]
    temp_prefix = f"temp_{process_id}_{unique_id}"

    # Reverse the coordinate order of the polygon
    polygon = Polygon([(lon, lat) for lat, lon in polygon.exterior.coords])

    # 1. Open the original image, get projection, resolution, and image extent
    ds_in = gdal.Open(input_tiff, gdal.GA_ReadOnly)
    if ds_in is None:
        raise IOError(f"Cannot open input TIFF: {input_tiff}")
    tif_proj = ds_in.GetProjection()
    gt = ds_in.GetGeoTransform()
    xsize = ds_in.RasterXSize
    ysize = ds_in.RasterYSize

    # Original resolution (usually gt[1] is +resolution, gt[5] is -resolution)
    x_res = abs(gt[1])
    y_res = abs(gt[5])

    # Calculate the four corners of the image in projected coordinates using geotransform
    # Top-left corner (x0, y0)
    x0 = gt[0]
    y0 = gt[3]
    # Bottom-right corner (x1, y1)
    x1 = x0 + gt[1] * xsize
    y1 = y0 + gt[5] * ysize
    # Construct the image quadrilateral (tile_polygon)
    # Note: gt[1] is usually positive, gt[5] is usually negative
    tile_polygon = Polygon([
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0, y0)
    ])

    # Close the dataset, release resources
    ds_in = None

    logging.info(f"[subset_tiff_with_polygon] Input TIFF proj: {tif_proj}")
    logging.info(f"[subset_tiff_with_polygon] Input TIFF size: {xsize} x {ysize}, res=({x_res}, {y_res})")

    # 2. Transform the input polygon (longitude/latitude coordinates) to the same projected coordinate system as the input image
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)  # polygon's current coordinate system is EPSG:4326 (longitude/latitude)

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromWkt(tif_proj)

    coord_trans = osr.CoordinateTransformation(src_srs, tgt_srs)
    transformed_coords = [coord_trans.TransformPoint(lon, lat)[:2] for (lon, lat) in polygon.exterior.coords]
    transformed_polygon = Polygon(transformed_coords)

    if not transformed_polygon.is_valid:
        transformed_polygon = transformed_polygon.buffer(0)
        logging.info("[subset_tiff_with_polygon] Polygon was invalid; applied buffer(0) to fix")

    # 3. Find the intersection of the image coverage extent tile_polygon and the polygon transformed_polygon
    intersection_polygon = tile_polygon.intersection(transformed_polygon)
    if intersection_polygon.is_empty:
        logging.warning("[subset_tiff_with_polygon] Intersection is empty. Producing an empty raster.")
        # Here you can choose to directly generate an empty file, or skip processing. Demonstrating generating a minimal image.
        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(output_tiff, 1, 1, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(tif_proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(0)
        out_band.Fill(0)
        out_ds.FlushCache()
        out_ds = None
        return True  # Clipping needed (empty file generated)

    logging.info(f"[subset_tiff_with_polygon] Intersection polygon bounds: {intersection_polygon.bounds}")

    # New: Check if the intersection equals the original image extent
    if intersection_polygon.equals(tile_polygon):
        logging.info("[subset_tiff_with_polygon] Intersection equals original tile. No subsetting needed.")
        return False  # No clipping needed

    # 4. Generate a temporary GeoJSON as the clipping boundary (using the intersection polygon)
    # Use process-specific temporary files to avoid conflicts in parallel processing
    tmp_geojson = os.path.join(os.path.dirname(output_tiff), f"{temp_prefix}_cutline.geojson")

    # Extract EPSG code from WKT projection string
    def extract_epsg_from_wkt(wkt):
        """Extract EPSG code from WKT projection string"""
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        # Try to get the EPSG code
        epsg = srs.GetAuthorityCode(None)
        if epsg:
            return int(epsg)
        return None

    # Then extract the EPSG code before generating GeoJSON
    epsg_code = extract_epsg_from_wkt(tif_proj)

    # Create GeoJSON dictionary, add CRS information
    geojson_dict = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": mapping(intersection_polygon)
        }]
    }

    # Add CRS information
    if epsg_code:
        geojson_dict["crs"] = {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{epsg_code}"
            }
        }
    else:
        # If the EPSG code cannot be extracted, use the full WKT
        logging.warning("Could not extract EPSG code from WKT, using default CRS")
        # Use default WGS84
        geojson_dict["crs"] = {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        }

    # Write GeoJSON file
    with open(tmp_geojson, "w") as f:
        json.dump(geojson_dict, f)

    # Bounding box of the intersection
    inter_minx, inter_miny, inter_maxx, inter_maxy = intersection_polygon.bounds

    # 5. Call gdal.Warp to clip to a temporary file
    #   - Use cropToCutline=True to keep only the area inside the cutline
    #   - Keep original resolution (xRes, yRes)
    #   - Set outputBounds to the bounding box of the intersection polygon
    #   - Specify format='GTiff' to explicitly output GeoTIFF
    temp_subset = os.path.join(os.path.dirname(output_tiff), f"{temp_prefix}_subset.tiff")

    warp_options = gdal.WarpOptions(
        format='GTiff',
        cutlineDSName=tmp_geojson,
        # cutlineSRS=tif_proj,
        cropToCutline=True,
        dstNodata=0,
        xRes=x_res,
        yRes=y_res,
        outputBounds=(inter_minx, inter_miny, inter_maxx, inter_maxy),
        outputBoundsSRS=tif_proj
    )

    logging.info(f"[subset_tiff_with_polygon] GDAL Warp with options: {warp_options}")
    ds_out = gdal.Warp(
        temp_subset,    # Target file
        input_tiff,     # Source file
        options=warp_options
    )

    # 6. Check the result after clipping
    if ds_out is None:
        # Warp failed
        if os.path.exists(tmp_geojson):
            os.remove(tmp_geojson)
        raise RuntimeError("[subset_tiff_with_polygon] gdal.Warp failed to produce output.")

    w = ds_out.RasterXSize
    h = ds_out.RasterYSize
    ds_out.FlushCache()
    ds_out = None
    logging.info(f"[subset_tiff_with_polygon] Subset temp file size: {w} x {h}")

    # 7. Try reopening to test
    check_ds = gdal.Open(temp_subset, gdal.GA_ReadOnly)
    if check_ds is None:
        if os.path.exists(tmp_geojson):
            os.remove(tmp_geojson)
        if os.path.exists(temp_subset):
            os.remove(temp_subset)
        raise RuntimeError("[subset_tiff_with_polygon] The warped output TIFF is unreadable by GDAL.")
    check_ds = None

    # 8. Delete the temporary cutline
    if os.path.exists(tmp_geojson):
        os.remove(tmp_geojson)

    # 9. If you need to "overwrite" the original file, first delete the original file, then rename the temporary file to the original filename
    # Use file locking or atomic operations for thread safety
    try:
        # Move the temporary file to replace the input file (original behavior)
        if os.path.exists(temp_subset):
            # Create a backup of the original file for safety (optional)
            # backup_file = input_tiff + ".bak"
            # if os.path.exists(input_tiff):
            #    shutil.copy2(input_tiff, backup_file)
            
            # Replace the original file with the subset version
            if os.path.exists(input_tiff):
                os.remove(input_tiff)
            shutil.move(temp_subset, input_tiff)
            
            # If output_tiff path differs from input_tiff and it exists, remove it
            if output_tiff != input_tiff and os.path.exists(output_tiff):
                os.remove(output_tiff)
    except Exception as e:
        logging.error(f"Error moving temporary file to input: {e}")
        if os.path.exists(temp_subset):
            os.remove(temp_subset)
        raise

    # Clean up any remaining temp files
    if os.path.exists(temp_subset):
        os.remove(temp_subset)

    logging.info(f"[subset_tiff_with_polygon] Done. Output saved to: {output_tiff}")
    return True  # Clipping needed (clipping completed)


def remove_raw_s2_except_one(tile_id: str, band_name="red"):
    DATA_DIR = os.environ.get("S2_DATA_RAW")
    tile_path = Path(DATA_DIR) / tile_id
    if not tile_path.exists():
        return
    band_dir = tile_path / band_name
    keep_tiff = None
    if band_dir.exists() and band_dir.is_dir():
        tiff_list = sorted(band_dir.glob("*.tiff"))
        if tiff_list:
            keep_tiff = tiff_list[0]
    for subdir in tile_path.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.tiff"):
                if keep_tiff and f.resolve() == keep_tiff.resolve():
                    continue
                f.unlink()
            try:
                subdir.rmdir()
            except OSError:
                pass
    for f in tile_path.glob("*.json"):
        f.unlink()
    logging.info(f"[Clean S2] Tile {tile_path.name}: Kept {keep_tiff}, removed other files.")

async def download_asset(session, asset_href, output_path, semaphore):
    if os.path.exists(output_path):
        logging.info(f"Skipping download of {asset_href} because file exists.")
        return
    async with semaphore:
        tries = 0
        while tries < 10:
            tries += 1
            try:
                async with session.get(asset_href) as response:
                    if response.status == 200:
                        content = await response.read()
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(content)
                        break
                    else:
                        logging.error(f"Failed to download {asset_href}: Status {response.status}. Retrying.")
            except asyncio.exceptions.TimeoutError:
                logging.error(f"Timeout downloading {asset_href}")
            except aiohttp.ClientPayloadError as e:
                logging.error(f"Client payload error {asset_href} -> {e}")
            wait_time = 2 ** tries
            logging.info(f"Waiting for {wait_time} seconds before retrying...")
            await asyncio.sleep(wait_time)
        else:
            logging.error(f"Failed to download {asset_href} after multiple attempts.")

async def process_item(session, item, loc, tile_id, semaphore):
    item_dict = item.to_dict(include_self_link=False)
    os.makedirs(loc, exist_ok=True)
    with open(f"{loc}/{item.id}.json", "w") as f:
        json.dump(item_dict, f)
    tasks = []
    for asset in band_assets:
        try:
            asset_href = item.assets[asset].href
            asset_dir = os.path.join(loc, asset)
            os.makedirs(asset_dir, exist_ok=True)
            output_path = os.path.join(asset_dir, f"{item.id}.tiff")
            tasks.append(download_asset(session, asset_href, output_path, semaphore))
        except KeyError:
            logging.error(f"Asset '{asset}' not found for item {item.id}")
    if tasks:
        await asyncio.gather(*tasks)

async def _async_download_tile(tile_id: str, start_date: str, end_date: str, data_dir="data_s2_raw"):
    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.strptime(end_date, "%Y-%m-%d")
    try:
        m = mgrs.MGRS()
        mgrs_center = tile_id + '5000050000'
        lat, lon = m.toLatLon(mgrs_center)
        point = Point(lon, lat)
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")
        project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
        project_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform
        point_utm = transform(project_to_utm, point)
        buffered_utm = point_utm.buffer(500)
        tile_poly = transform(project_to_wgs84, buffered_utm)
    except Exception as e:
        logging.error(f"Error converting MGRS to polygon for tile {tile_id}: {e}")
        return
    api_url = "https://earth-search.aws.element84.com/v1"
    collection = "sentinel-2-l2a"
    client = Client.open(api_url)
    search = client.search(
        collections=[collection],
        intersects=mapping(tile_poly),
        datetime=f"{start_date}T00:00:00Z/{end_date}T00:00:00Z",
        limit=20,
        query={"s2:processing_baseline": "05.00"}
    )
    num_matched = search.matched()
    logging.info(f"[Download] Tile {tile_id} matched {num_matched} items between {start_date} and {end_date}.")
    loc = Path(data_dir) / tile_id
    loc.mkdir(parents=True, exist_ok=True)
    for asset in band_assets:
        (loc / asset).mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(20)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1000), connector=aiohttp.TCPConnector(ssl=False)) as session:
        for batch in search.pages():
            items_in_batch = len(batch)
            logging.info(f"Processing batch of {items_in_batch} items for tile {tile_id}")
            tasks = []
            # Process tile_id starting with "0", i.e. "03VXG" -> "3VXG"
            if tile_id.startswith('0'):
                tile_id = tile_id[1:]
            for item in batch:
                if item.id.startswith(f"S2A_{tile_id}_") or item.id.startswith(f"S2B_{tile_id}_"):
                    cloud_coverage = item.properties.get("eo:cloud_cover", 0)
                    if cloud_coverage > int(CLOUD_COVERAGE_THRESHOLD):
                        logging.info(f"Skipping item {item.id} with cloud coverage {cloud_coverage}")
                        continue
                    tasks.append(process_item(session, item, str(loc), tile_id, semaphore))
                else:
                    logging.info(f"Skipping item {item.id} not matching tile {tile_id}")
            if tasks:
                await asyncio.gather(*tasks)
            logging.info(f"Processed batch; downloaded so far for tile {tile_id}.")
    logging.info(f"[Download] Tile {tile_id} finished downloading items.")

def download_tile(tile_id: str, start_date: str, end_date: str, data_dir="data_s2_raw"):
    asyncio.run(_async_download_tile(tile_id, start_date, end_date, data_dir))
