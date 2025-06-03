#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_processor.py

Processes S2 for one tile: Download -> Subset -> Rust stack
(We do NOT skip here; skipping logic is done in main_pipeline.sh.)

Environment variables for paths:
    S2_DATA_RAW        : where raw S2 files are stored
    S2_DATA_PROCESSED  : where final npy outputs go
    RUST_S2_EXE        : path to Rust binary

Usage:
    python3 s2_processor.py \
        --tile_id 34VEH \
        --tiff_path /path/to/downstream.tiff \
        --s2_start_date 2021-01-01 \
        --s2_end_date   2021-12-31 \
        --sample_rate 1
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from s2_utils import (
    tiff_to_polygon,
    subset_tiff_with_polygon,
    download_tile,
    remove_raw_s2_except_one
)

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_id", required=True)
    parser.add_argument("--tiff_path", required=True)
    parser.add_argument("--s2_start_date", default="2021-01-01")
    parser.add_argument("--s2_end_date",   default="2021-12-31")
    parser.add_argument("--sample_rate", type=int, default=1)
    args = parser.parse_args()

    # read env vars
    DATA_DIR        = os.environ.get("S2_DATA_RAW")
    PROCESSED_S2_DIR= os.environ.get("S2_DATA_PROCESSED")
    RUST_EXE        = os.environ.get("RUST_S2_EXE")

    logging.info(f"[s2_processor] Using S2_DATA_RAW={DATA_DIR}")
    logging.info(f"[s2_processor] Using S2_DATA_PROCESSED={PROCESSED_S2_DIR}")
    logging.info(f"[s2_processor] Using RUST_EXE={RUST_EXE}")

    # 1) polygon
    downstream_poly = tiff_to_polygon(args.tiff_path, densify_pts=21)

    # 2) download
    download_tile(
            tile_id=args.tile_id,
            start_date=args.s2_start_date,
            end_date=args.s2_end_date,
            data_dir=DATA_DIR
    )

    # 3) subset all .tiff
    tile_dir = Path(DATA_DIR) / args.tile_id
    tiff_list = list(tile_dir.rglob("*.tiff"))
    if not tiff_list:
            logging.warning(f"No raw S2 .tiff found for tile {args.tile_id}, cannot subset.")
            return

    # Check the first tiff file to determine if the tile needs subsetting
    needs_subsetting = True  # Assume subsetting is needed by default
    if tiff_list:
            first_tiff = tiff_list[0]
            first_out_tiff = str(first_tiff).replace(".tiff", "_subset.tiff")
            if not os.path.exists(first_out_tiff):
                    try:
                            # Check if the first file needs subsetting
                            needs_subsetting = subset_tiff_with_polygon(str(first_tiff), first_out_tiff, downstream_poly)
                            if not needs_subsetting:
                                    logging.info(f"First file {first_tiff} doesn't need subsetting - skipping all files in tile {args.tile_id}")
                    except Exception as e:
                            logging.error(f"Error checking first file {first_tiff}: {e}")

    # If the first file needs subsetting, process the remaining files
    if needs_subsetting:
            # Start processing from the second file (the first one has already been processed)
            for tp in tiff_list[1:]:
                    out_tiff = str(tp).replace(".tiff", "_subset.tiff")
                    if not os.path.exists(out_tiff):
                            try:
                                    subset_tiff_with_polygon(str(tp), out_tiff, downstream_poly)
                            except Exception as e:
                                    logging.error(f"Error subsetting {tp}: {e}")

    # Log the processing result
    if not needs_subsetting and tiff_list:
            logging.info(f"Tile {args.tile_id} doesn't require subsetting - skip the rest of the files.")

    # 4) run rust
    out_dir = Path(PROCESSED_S2_DIR) / args.tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
            RUST_EXE,
            "--input", str(tile_dir),
            "--output", str(out_dir),
            "--sample-rate", str(args.sample_rate)
    ]
    logging.info(f"Running Rust: {' '.join(cmd)}")
    try:
            subprocess.run(cmd, check=True)
            remove_raw_s2_except_one(tile_id=args.tile_id)
    except subprocess.CalledProcessError as e:
            logging.error(f"Rust S2 stack failed: {e}")

if __name__ == "__main__":
        main()
