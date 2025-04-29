#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_downloader.py

Downloads Sentinel-1 .zip for a given tile + date range.
No skipping logic here; we rely on main_pipeline.

Reads from env:
  S1_DATA_RAW

Usage:
  python3 s1_downloader.py --tile_id XXX --s1_start_date YYYY-MM-DD --s1_end_date YYYY-MM-DD
"""

import argparse
import logging
import asyncio
import os
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

    try:
        results.download(path=str(out_dir), session=session, processes=12)
        for zip_path in out_dir.glob("*.zip"):
            # Simple size check – for GRD ~ 100 MB or more:
            if zip_path.stat().st_size < 10_000_000:  
                # This is <10MB, almost certainly incomplete
                logging.warning(f"Zip looks too small (likely incomplete): {zip_path}")
                zip_path.unlink()  # remove it
                # optionally trigger a retry here
        logging.info(f"Downloaded S1 zips to {out_dir}")
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
