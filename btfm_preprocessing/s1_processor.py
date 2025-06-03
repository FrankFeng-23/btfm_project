#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_processor.py

Processes a SINGLE S1 .zip -> .tif using SNAP.
Env var:
    S1_DATA_PROCESSED = output .tif location (we pass as --out_dir, but that might be the same path)

Usage:
    python3 s1_processor.py --zip_file /some/S1.zip \
                                                    --reference_tiff /path/to/red_subset.tiff \
                                                    --out_dir /some/where
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("org.esa.snap.core.util.ServiceLoader").setLevel(logging.ERROR)
logging.getLogger("javax.media.jai").setLevel(logging.ERROR)
import sys
import os

from pathlib import Path
import time
import shutil

from osgeo import gdal, osr
from shapely.geometry import Polygon

try:
        import esa_snappy as snappy
        from esa_snappy import ProductIO, GPF, HashMap
except ImportError:
        logging.error("SNAP (esa_snappy) not found.")
        sys.exit(1)

def clear_snap_tmp(directory_path = '/local/user/snap_tmp', max_age = 7200):
        """
        Clears files/directories in SNAP temporary directory that are older than 1 hour.
        """
        if not os.path.exists(directory_path):
                logging.info(f"The directory {directory_path} does not exist.")
                return

        current_time = time.time()
        max_age_seconds = max_age  # seconds

        for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                try:
                        # Get the last modification time
                        mtime = os.path.getmtime(item_path)
                        if current_time - mtime > max_age_seconds:
                                if os.path.isfile(item_path):
                                        os.unlink(item_path)
                                elif os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                                logging.info(f"Removed old item: {item_path}")
                except OSError as e:
                        logging.error(f"Error removing {item_path}: {e}")

        logging.info(f"Finished cleaning old files in: {directory_path}")

def read_reference_tiff_as_polygon(reference_tiff: str):
        ds = gdal.Open(reference_tiff, gdal.GA_ReadOnly)
        if not ds:
                raise IOError(f"Cannot open {reference_tiff}")
        proj = ds.GetProjection()
        gt   = ds.GetGeoTransform()
        xsize= ds.RasterXSize
        ysize= ds.RasterYSize
        ds = None

        src = osr.SpatialReference()
        src.ImportFromWkt(proj)
        tgt = osr.SpatialReference()
        tgt.ImportFromEPSG(4326)
        ct = osr.CoordinateTransformation(src, tgt)
        corners = [
                (gt[0], gt[3]),
                (gt[0]+gt[1]*xsize, gt[3]),
                (gt[0]+gt[1]*xsize, gt[3]+gt[5]*ysize),
                (gt[0], gt[3]+gt[5]*ysize)
        ]
        ll_corners=[]
        for (x,y) in corners:
                lonlat = ct.TransformPoint(x,y)[:2]
                ll_corners.append(lonlat)
        return Polygon(ll_corners), proj

def process_zip(zip_file, reference_tiff, out_dir):
        zf = Path(zip_file)
        if not zf.exists():
                logging.error(f"No zip: {zf}")
                return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        product = ProductIO.readProduct(str(zf))
        if not product:
                logging.error(f"Error reading product from {zf}")
                return

        # Check if required bands exist before proceeding
        band_names = product.getBandNames()
        if "Intensity_VV" not in band_names or "Intensity_VH" not in band_names:
                logging.warning(f"[S1 chain] Missing required VV or VH bands in {zf}. Skipping processing.")
                return

        # Extract PASS information
        try:
                pass_str = product.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeString('PASS')
                pass_str = pass_str.upper()
        except Exception as e:
                logging.warning(f"[S1 chain] cannot get PASS => {e}, set to ASCENDING")
                pass_str = "ASCENDING"

        logging.info(f"[S1 chain] {zip_file} => PASS={pass_str}")

        # Add PASS information to the output filename
        out_tif = out_dir / (zf.stem + f"_{pass_str}.tif")
        if out_tif.exists():
                logging.info(f"{out_tif} exists, skip.")
                return

        poly, ref_proj = read_reference_tiff_as_polygon(reference_tiff)
        # Swap longitude and latitude of the polygon
        poly = Polygon([(p[1], p[0]) for p in poly.exterior.coords])
        geoRegion = poly.wkt
        logging.info(f"Reference Tiff GeoRegion: {geoRegion}")

        logging.info(f"Processing {zf} => {out_tif}")

        # 1) Orbit
        from esa_snappy import GPF, HashMap
        orbit_params = HashMap()
        orbit_params.put("ApplyOrbitFile", True)
        try:
                product_orb = GPF.createProduct("Apply-Orbit-File", orbit_params, product)
        except Exception as e:
                logging.error(f"Orbit fail: {e}")
                product_orb = product

        # 2) Calibration
        cal_params = HashMap()
        cal_params.put("outputBetaBand", False)
        cal_params.put("outputSigmaBand", True)
        cal_params.put("outputGammaBand", False)
        cal_params.put("sourceBands","Intensity_VV,Intensity_VH")
        try:
                product_cal = GPF.createProduct("Calibration", cal_params, product_orb)
                if not product_cal:
                        logging.error("Calibration None.")
                        return
                w_cal = product_cal.getSceneRasterWidth()
                h_cal = product_cal.getSceneRasterHeight()
                if w_cal < 1 or h_cal < 1:
                        logging.warning(f"[S1 chain] Calibration produced zero size for {zf}")
                        return
                logging.info(f"[S1 chain] Calibration size=({w_cal},{h_cal}) for {zf}")
        except Exception as e:
                logging.error(f"Calibration error: {e}")
                return

        # 3) Subset
        sub_params = HashMap()
        sub_params.put("geoRegion", geoRegion)
        sub_params.put("copyMetadata", True)
        try:
                product_sub = GPF.createProduct("Subset", sub_params, product_cal)
                if not product_sub:
                        logging.error("Subset None.")
                        return
                sw = product_sub.getSceneRasterWidth()
                sh = product_sub.getSceneRasterHeight()
                logging.info(f"[S1 chain] Subset size=({sw},{sh}) for {zf}")
                if sw < 1 or sh < 1:
                        logging.warning(f"[S1 chain] Subset produced zero size for {zf}")
                        return
        except Exception as e:
                logging.error(f"Subset error: {e}")
                return

        # 4) Terrain Correction
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ref_proj)
        epsg_code = srs.GetAttrValue("AUTHORITY", 1)
        if not epsg_code:
                epsg_code = "4326"
        tc_params = HashMap()
        # tc_params.put("demName","SRTM 1Sec HGT")
        tc_params.put("pixelSpacingInMeter",10.0)
        tc_params.put("mapProjection", f"EPSG:{epsg_code}")
        tc_params.put("nodataValueAtSea",False)
        tc_params.put("saveDEM",False)
        tc_params.put("saveLatLon",False)

        dem_options = ["SRTM 1Sec HGT", "ACE30", "GETASSE30"]
        product_tc = None
        for dem in dem_options:
                tc_params.put("demName", dem)
                logging.info(f"Attempting terrain correction with DEM: {dem}")
                try:
                        product_tc = GPF.createProduct("Terrain-Correction", tc_params, product_sub)
                        if product_tc:
                                logging.info(f"Successfully used DEM: {dem}")
                                break
                except Exception as e:
                        logging.warning(f"Failed with DEM {dem}: {e}")
                        continue

        if not product_tc:
                logging.error("All DEM options failed for terrain correction.")
                # If all DEMs failed, we might still want to try writing the last attempted product_tc
                # or handle this case differently, e.g., return or raise an error.
                # For now, let's proceed cautiously, assuming the loop might have exited without a successful product_tc.
                # If product_tc is None after the loop, the next block will fail gracefully.
                # However, it's better to explicitly handle the failure.
                logging.error(f"Terrain Correction failed for {zf} after trying all DEMs.")
                # return # Exit if TC failed completely

        # This block assumes product_tc is valid if the loop succeeded.
        try:
                # Check again if product_tc is valid (it should be if the loop broke successfully)
                if not product_tc:
                         logging.error("TC product is unexpectedly None after DEM loop.")
                         return
                w_tc = product_tc.getSceneRasterWidth()
                h_tc = product_tc.getSceneRasterHeight()
                logging.info(f"[S1 chain] Terrain Correction size=({w_tc},{h_tc}) for {zf}")
                if w_tc < 1 or h_tc < 1:
                        logging.warning(f"[S1 chain] Terrain Correction produced zero size for {zf}")
                        return
                ProductIO.writeProduct(product_tc, str(out_tif), "GeoTIFF")
                logging.info(f"Wrote {out_tif}")
        except Exception as e:
                # This error might occur during writing or getting dimensions if product_tc is somehow invalid
                logging.error(f"Final TC/Write error: {e}")


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--zip_file", required=True)
        parser.add_argument("--reference_tiff", required=True)
        parser.add_argument("--out_dir", required=True)
        args = parser.parse_args()

        clear_snap_tmp(directory_path=os.environ["TMPDIR"], max_age=7200)
        clear_snap_tmp(directory_path=os.environ["HOME_DEM_DIR"], max_age=7200)

        process_zip(args.zip_file, args.reference_tiff, args.out_dir)

if __name__=="__main__":
        main()
