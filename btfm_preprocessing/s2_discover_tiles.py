#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_discover_tiles.py

Reads a downstream TIFF, uses STAC to discover intersecting
Sentinel-2 L2A items, extracts MGRS tile codes, writes them to a file.

Usage:
  python3 s2_discover_tiles.py \
    --tiff_path /path/to/tiff \
    --out_tiles tile_list.txt
    [--shortlist some_tiles.txt]
"""
import argparse
import logging
import sys
import json

from shapely.geometry import mapping
from pystac_client import Client
from s2_utils import tiff_to_polygon
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", required=True)
    parser.add_argument("--out_tiles", required=True)
    # parser.add_argument("--shortlist", help="optional tile code list to filter")
    args = parser.parse_args()

    poly = tiff_to_polygon(args.tiff_path, densify_pts=21)
    # poly = Polygon([(lon, lat) for lat, lon in poly.exterior.coords])

    start_datetime = "2020-01-01T00:00:00Z"
    end_datetime   = "2020-06-01T00:00:00Z"
    collection     = "sentinel-2-l2a"
    api_url        = "https://earth-search.aws.element84.com/v1"
    client         = Client.open(api_url)

    search = client.search(
        collections=[collection],
        intersects=json.loads(json.dumps(mapping(poly))),
        datetime=f"{start_datetime}/{end_datetime}",
        limit=100,
        query={"s2:processing_baseline": "05.00"}
    )
    items = list(search.items())
    logging.info(f"Found {len(items)} items for tile discovery.")

    tile_codes = set()
    for it in items:
        parts = it.id.split("_")
        if len(parts) >= 2:
            tile_codes.add(parts[1])
    tile_list = sorted(tile_codes)

    # optional shortlist
    # if args.shortlist:
    #     try:
    #         with open(args.shortlist,"r") as f:
    #             valid = set([x.strip() for x in f if x.strip()])
    #         tile_list = [t for t in tile_list if t in valid]
    #     except Exception as e:
    #         logging.warning(f"Cannot read shortlist {args.shortlist}: {e}")

    with open(args.out_tiles, "w") as f:
        for t in tile_list:
            f.write(t + "\n")

    logging.info(f"Wrote {len(tile_list)} tiles to {args.out_tiles}.")

if __name__ == "__main__":
    main()
