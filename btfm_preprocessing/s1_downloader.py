#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_downloader.py

Downloads Sentinel-1 .zip for a given tile + date range.
Before downloading, checks if files already exist in other MGRS tile folders.

Reads from env:
  S1_DATA_RAW

Usage:
  python3 s1_downloader.py --tile_id XXX --s1_start_date YYYY-MM-DD --s1_end_date YYYY-MM-DD
"""

import argparse
import logging
import asyncio
import os
import shutil
from datetime import date
from pathlib import Path

import mgrs
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform

logging.basicConfig(level=logging.INFO)

try:
    import asf_search as asf
    from asf_search import ASFSession, PRODUCT_TYPE, PLATFORM
except ImportError:
    asf = None
    logging.error("asf_search not installed")

def from_mgrs_to_polygon(tile_id: str) -> Polygon:
    m = mgrs.MGRS()
    lat, lon = m.toLatLon(tile_id + "5000050000")
    half_deg = 0.2
    return Polygon([
        (lon-half_deg, lat-half_deg),
        (lon+half_deg, lat-half_deg),
        (lon+half_deg, lat+half_deg),
        (lon-half_deg, lat+half_deg),
        (lon-half_deg, lat-half_deg),
    ])

async def _async_download_s1(tile_id, s1_start_date, s1_end_date, wkt_str, out_dir):
    if asf is None:
        logging.error("asf_search not available, skip.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def parse_ymd(s):
        y,m,d = s.split("-")
        return date(int(y), int(m), int(d))
    dt1 = parse_ymd(s1_start_date)
    dt2 = parse_ymd(s1_end_date)
    
    USERNAME = os.environ.get("S1_DOWNLOAD_USERNAME")
    PASSWORD = os.environ.get("S1_DOWNLOAD_PASSWORD")
    session = asf.ASFSession().auth_with_creds(USERNAME, PASSWORD)

    results = asf.search(
        platform=[PLATFORM.SENTINEL1A, PLATFORM.SENTINEL1B],
        processingLevel=[PRODUCT_TYPE.GRD_HD],
        start=dt1, end=dt2,
        intersectsWith=wkt_str,
        polarization=[asf.POLARIZATION.VV_VH],
    )
    logging.info(f"[s1_downloader] Tile={tile_id}, found {len(results)} scenes.")
    if not results:
        return

    # Before downloading, check if files already exist in other MGRS tile folders
    s1_data_raw = Path(os.environ.get("S1_DATA_RAW", ""))
    existing_tiles = [d for d in s1_data_raw.iterdir() if d.is_dir() and d.name != tile_id]
    
    moved_count = 0
    # Get filenames from results and check for existing files
    for result in results:
        # Get the filename - try different attributes since API might vary
        filename = None
        
        # Try to get filename
        if hasattr(result, 'properties') and 'fileName' in result.properties:
            filename = result.properties['fileName']
        
        # Sometimes the filename might not have the .zip extension
        if filename and not filename.endswith('.zip'):
            filename += '.zip'
        
        if not filename:
            logging.warning(f"Could not determine filename for result: {result}")
            continue
        
        # First check if file already exists in target directory
        target_file = out_dir / filename
        if target_file.exists():
            logging.info(f"File already exists in target directory: {filename}")
            continue
        
        # Check if this file already exists in other tile folders
        for tile_dir in existing_tiles:
            existing_file = tile_dir / filename
            if existing_file.exists():
                try:
                    logging.info(f"Moving existing file from {existing_file} to {target_file}")
                    existing_file.rename(target_file)
                    moved_count += 1
                except Exception as e:
                    logging.error(f"Failed to move file {existing_file}: {e}")
                    # If move fails (e.g., cross-filesystem), try to copy instead
                    try:
                        shutil.copy2(existing_file, target_file)
                        moved_count += 1
                        logging.info(f"Copied file instead of moving: {filename}")
                    except Exception as e2:
                        logging.error(f"Failed to copy file {existing_file}: {e2}")
                break
    
    if moved_count > 0:
        logging.info(f"Moved/copied {moved_count} existing files from other tiles")
    
    # Now download remaining files (asf_search usually skips existing files)
    try:
        results.download(path=str(out_dir), session=session, processes=12)
        
        # Check for incomplete downloads
        downloaded_files = 0
        for zip_path in out_dir.glob("*.zip"):
            if zip_path.stat().st_size < 10_000_000:  
                logging.warning(f"Zip looks too small (likely incomplete): {zip_path}")
                zip_path.unlink()  # remove it
            else:
                downloaded_files += 1
        
        logging.info(f"Total files in {out_dir}: {downloaded_files} (moved: {moved_count}, downloaded: {downloaded_files - moved_count})")
    except Exception as e:
        logging.error(f"Download error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_id", required=True)
    parser.add_argument("--s1_start_date", default="2021-01-01")
    parser.add_argument("--s1_end_date",   default="2021-12-31")
    args = parser.parse_args()

    S1_DATA_RAW = os.environ.get("S1_DATA_RAW")
    poly = from_mgrs_to_polygon(args.tile_id)
    wkt_str = poly.wkt

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_async_download_s1(
            tile_id=args.tile_id,
            s1_start_date=args.s1_start_date,
            s1_end_date=args.s1_end_date,
            wkt_str=wkt_str,
            out_dir=S1_DATA_RAW+"/"+args.tile_id
        ))
    finally:
        loop.close()

if __name__ == "__main__":
    main()