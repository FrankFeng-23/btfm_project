# btfm_tools/roi_processor.py
import os
import fiona
import rasterio
import rasterio.features
from rasterio.windows import Window
import logging
import numpy as np
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer, CRS as PyprojCRS
import argparse
import json
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS (determine_utm_zone, determine_best_utm_crs from original) ---
def determine_utm_zone(lon, lat):
    zone_number = int((lon + 180) / 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12: zone_number = 32
    if 72 <= lat < 84:
        if 0 <= lon < 9: zone_number = 31
        elif 9 <= lon < 21: zone_number = 33
        elif 21 <= lon < 33: zone_number = 35
        elif 33 <= lon < 42: zone_number = 37
    is_northern = lat >= 0
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
    return epsg_code, zone_number, is_northern

def determine_best_utm_crs(geometries_or_bbox, src_crs_str="EPSG:4326"):
    logger.info("Determining the best UTM zone...")
    src_crs_obj = PyprojCRS.from_string(src_crs_str)
    
    shapely_geoms = []
    if isinstance(geometries_or_bbox, list) and geometries_or_bbox: # List of GeoJSON-like dicts
        shapely_geoms = [shape(g) for g in geometries_or_bbox]
    elif isinstance(geometries_or_bbox, tuple) and len(geometries_or_bbox) == 4: # BBox tuple
        min_lon, min_lat, max_lon, max_lat = geometries_or_bbox
        shapely_geoms = [box(min_lon, min_lat, max_lon, max_lat)]
    else:
        raise ValueError("Input for UTM determination must be a list of geometries or a bbox tuple.")

    if src_crs_obj.to_epsg() != 4326:
        logger.info(f"Source CRS is {src_crs_str}, converting to WGS84 for centroid calculation...")
        wgs84_crs_obj = PyprojCRS.from_epsg(4326)
        transformer_to_wgs84 = Transformer.from_crs(src_crs_obj, wgs84_crs_obj, always_xy=True).transform
        wgs84_geoms = [shp_transform(transformer_to_wgs84, sg) for sg in shapely_geoms]
    else:
        wgs84_geoms = shapely_geoms

    union_geom = unary_union(wgs84_geoms)
    if union_geom.is_empty:
        raise ValueError("Cannot determine UTM zone from empty or invalid geometries.")
    centroid = union_geom.centroid
    cent_lon, cent_lat = centroid.x, centroid.y
    
    epsg_code, zone_number, is_northern = determine_utm_zone(cent_lon, cent_lat)
    hemisphere = "Northern Hemisphere" if is_northern else "Southern Hemisphere"
    logger.info(f"Data centroid location: {cent_lon:.6f} E, {cent_lat:.6f} N")
    logger.info(f"Selected UTM zone: {zone_number}{hemisphere[0]} (EPSG:{epsg_code})")
    return rasterio.crs.CRS.from_epsg(epsg_code)


def _create_roi_tiff_from_geometries(
    geometries, src_crs_str, output_tiff_path, pixel_size=10, target_crs_obj=None
):
    """
    Internal function to rasterize a list of geometries to a ROI TIFF.
    Geometries should be GeoJSON-like dicts.
    src_crs_str is the CRS of the input geometries (e.g., "EPSG:4326").
    target_crs_obj is a rasterio.crs.CRS object for the output.
    """
    os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
    src_crs_obj_pyproj = PyprojCRS.from_string(src_crs_str) # For pyproj transformations

    if not target_crs_obj:
        target_crs_obj = determine_best_utm_crs(geometries, src_crs_str)
    
    logger.info(f"Source CRS for geometries: {src_crs_str}")
    logger.info(f"Target CRS for TIFF: {target_crs_obj.to_string()}")

    transformer_to_target = Transformer.from_crs(src_crs_obj_pyproj, PyprojCRS.from_string(target_crs_obj.to_string()), always_xy=True).transform
    
    reprojected_shapely_geoms = [shp_transform(transformer_to_target, shape(geom)) for geom in geometries]
    reprojected_geoms_for_rasterize = [mapping(sg) for sg in reprojected_shapely_geoms]

    if not reprojected_shapely_geoms:
        raise ValueError("No valid geometries to rasterize after reprojection.")

    # Determine bounds for the raster from the union of reprojected geometries
    # Using convex hull of the union for raster extent is generally safer.
    union_for_bounds = unary_union(reprojected_shapely_geoms)
    if union_for_bounds.is_empty:
        raise ValueError("Union of reprojected geometries is empty.")
    
    bounds_geom = union_for_bounds.convex_hull
    minx, miny, maxx, maxy = bounds_geom.bounds

    # Calculate raster dimensions
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions: W={width}, H={height} from bounds {bounds_geom.bounds} and pixel_size {pixel_size}")

    logger.info(f"Output raster dimensions: Width={width}, Height={height} pixels")
    transform_affine = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    # Rasterize the original reprojected geometries (not their union for content)
    image = rasterio.features.rasterize(
        reprojected_geoms_for_rasterize,
        out_shape=(height, width),
        transform=transform_affine,
        fill=0,  # Background
        default_value=1,  # ROI pixels
        dtype=rasterio.uint8
    )

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.uint8,
        'crs': target_crs_obj,
        'transform': transform_affine,
        'nodata': 0 # Explicitly set nodata if fill is 0
    }

    with rasterio.open(output_tiff_path, 'w', **profile) as dst:
        dst.write(image.astype(rasterio.uint8), 1)
    
    logger.info(f"Successfully created ROI TIFF: {output_tiff_path}")
    return output_tiff_path

def process_shapefile_or_geojson(input_path, output_tiff_path, pixel_size=10):
    logger.info(f"Processing vector file: {input_path}")
    with fiona.open(input_path, 'r') as src:
        src_crs_str = rasterio.crs.CRS(src.crs).to_string() if src.crs else "EPSG:4326"
        geometries = [feature['geometry'] for feature in src if feature['geometry']]
    if not geometries:
        raise ValueError(f"No geometries found in {input_path}")
    return _create_roi_tiff_from_geometries(geometries, src_crs_str, output_tiff_path, pixel_size)

def process_bbox(min_lon, min_lat, max_lon, max_lat, output_tiff_path, pixel_size=10):
    logger.info(f"Processing BBox: lon({min_lon}-{max_lon}), lat({min_lat}-{max_lat})")
    # Shapely box: box(minx, miny, maxx, maxy)
    bbox_shapely = box(min_lon, min_lat, max_lon, max_lat)
    geometries = [mapping(bbox_shapely)] # Convert to GeoJSON-like dict
    src_crs_str = "EPSG:4326" # Assume BBox is WGS84
    return _create_roi_tiff_from_geometries(geometries, src_crs_str, output_tiff_path, pixel_size)

def process_input_tiff(input_tiff_path, output_roi_tiff_path, pixel_size=10):
    logger.info(f"Processing input TIFF: {input_tiff_path} to output ROI: {output_roi_tiff_path}")
    os.makedirs(os.path.dirname(output_roi_tiff_path), exist_ok=True)

    with rasterio.open(input_tiff_path) as src:
        src_profile = src.profile
        src_crs = src.crs
        src_transform = src.transform

        # If the input TIFF is already in the desired format (10m, UTM-like, binary 1/0)
        # This check is simplified; a more robust check would involve checking values.
        is_10m_approx = np.isclose(abs(src_transform.a), pixel_size, atol=0.1) and \
                        np.isclose(abs(src_transform.e), pixel_size, atol=0.1)
        
        # For simplicity, we'll attempt to reprocess it to ensure consistency.
        # A more advanced version could have a flag to "trust" the input TIFF if it meets criteria.
        logger.info(f"Input TIFF CRS: {src_crs}, Resolution: ({abs(src_transform.a):.2f}x{abs(src_transform.e):.2f})m. Will reprocess to standard 10m ROI format.")

        # Determine target CRS (UTM) based on the TIFF's bounds
        bounds = src.bounds
        # Convert bounds to WGS84 for UTM determination if not already WGS84
        if src_crs and src_crs.to_epsg() != 4326:
            transformer_to_wgs84 = Transformer.from_crs(PyprojCRS.from_string(src_crs.to_string()), PyprojCRS.from_epsg(4326), always_xy=True).transform
            wgs84_bounds_geom = shp_transform(transformer_to_wgs84, box(bounds.left, bounds.bottom, bounds.right, bounds.top))
            wgs84_bounds_tuple = wgs84_bounds_geom.bounds # (min_lon, min_lat, max_lon, max_lat)
            target_crs_obj = determine_best_utm_crs(wgs84_bounds_tuple, "EPSG:4326")

        elif src_crs and src_crs.to_epsg() == 4326:
            target_crs_obj = determine_best_utm_crs( (bounds.left, bounds.bottom, bounds.right, bounds.top) , "EPSG:4326")
        else: # No CRS or unknown
            logger.warning(f"Input TIFF {input_tiff_path} has no/unclear CRS. Attempting WGS84 assumption for bounds.")
            try: # Assume bounds are lon/lat if no CRS
                target_crs_obj = determine_best_utm_crs((bounds.left, bounds.bottom, bounds.right, bounds.top), "EPSG:4326")
            except Exception as e:
                logger.error(f"Cannot determine target CRS for {input_tiff_path} due to missing source CRS and bounds interpretation: {e}. Defaulting to its own CRS if possible, or failing.")
                if src_crs:
                    target_crs_obj = src_crs # Fallback to original CRS if UTM determination fails
                else:
                    raise ValueError(f"Cannot proceed without a target CRS for input TIFF {input_tiff_path}")


        # Calculate new transform and dimensions in the target_crs_obj at 10m resolution
        # Use gdal.Warp for reprojection, resampling, and binarization
        temp_reprojected_path = output_roi_tiff_path + ".temp.tif"

        # Binarize: non-zero pixels become 1, zero (nodata) pixels remain 0.
        # We create a VRT to do this calculation on the fly.
        vrt_template = f"""
<VRTDataset rasterXSize="{src.width}" rasterYSize="{src.height}">
  <SRS>{src_crs.to_wkt()}</SRS>
  <GeoTransform>{', '.join(map(str, src.transform.to_gdal()))}</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <ColorInterp>Gray</ColorInterp>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{os.path.abspath(input_tiff_path)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>
      <DstRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>
    </SimpleSource>
    <LUT> <!-- Map non-zero to 1, zero to 0 -->
      <Value N="0" V="0"/>
      <Value N="1" V="1" default="1"/> <!-- Any value >= 1 maps to 1 -->
    </LUT>
  </VRTRasterBand>
</VRTDataset>
"""
        vrt_path = output_roi_tiff_path + ".vrt"
        with open(vrt_path, 'w') as f:
            f.write(vrt_template)

        gdal.Warp(
            output_roi_tiff_path,
            vrt_path, # Use VRT for on-the-fly binarization
            dstSRS=target_crs_obj.to_wkt(),
            xRes=pixel_size,
            yRes=pixel_size,
            resampleAlg='near', # Nearest neighbor for binary data
            outputType=gdal.GDT_Byte,
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            dstNodata=0, # Ensure output nodata is 0
            srcNodata=src.nodata if src.nodata is not None else None # Respect source nodata if defined
        )
        
        if os.path.exists(vrt_path): os.remove(vrt_path)
        if os.path.exists(temp_reprojected_path): os.remove(temp_reprojected_path) # Should not exist with direct warp

    logger.info(f"Successfully processed input TIFF to ROI TIFF: {output_roi_tiff_path}")
    return output_roi_tiff_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert various ROI inputs to a standardized ROI TIFF.")
    parser.add_argument("--input_path", required=True, help="Path to input file (Shapefile, GeoJSON, existing TIFF, or BBox text file).")
    parser.add_argument("--output_tiff_path", required=True, help="Path for the output ROI TIFF.")
    parser.add_argument("--input_type", choices=['shp', 'geojson', 'tiff', 'bbox'], required=True,
                        help="Type of the input file. For 'bbox', input_path should be a text file with 'min_lon,min_lat,max_lon,max_lat'.")
    parser.add_argument("--pixel_size", type=float, default=10.0, help="Pixel size in meters for output ROI TIFF.")

    args = parser.parse_args()

    try:
        if args.input_type == 'shp' or args.input_type == 'geojson':
            process_shapefile_or_geojson(args.input_path, args.output_tiff_path, args.pixel_size)
        elif args.input_type == 'bbox':
            with open(args.input_path, 'r') as f:
                coords_str = ''.join(f.readlines()).replace('\n', '').replace('\r', '').strip()
            coords = [float(c.strip()) for c in coords_str.split(',') if c.strip()]
            if len(coords) != 4:
                raise ValueError("BBox file must contain 4 comma-separated values: min_lon,min_lat,max_lon,max_lat")
            process_bbox(coords[0], coords[1], coords[2], coords[3], args.output_tiff_path, args.pixel_size)
        elif args.input_type == 'tiff':
            process_input_tiff(args.input_path, args.output_tiff_path, args.pixel_size)
        
        logger.info(f"ROI TIFF generation successful: {args.output_tiff_path}")
    except Exception as e:
        logger.error(f"ROI TIFF generation failed for {args.input_path}: {e}", exc_info=True)
        sys.exit(1)