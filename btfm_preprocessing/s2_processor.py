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
        --sample_rate 1 \
        --num_processes 16
"""

import argparse
import logging
import os
import subprocess
import sys
import multiprocessing
from pathlib import Path
from functools import partial

from s2_utils import (
    tiff_to_polygon,
    subset_tiff_with_polygon,
    download_tile,
    remove_raw_s2_except_one
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def process_tiff(tiff_path, output_suffix="_subset.tiff", polygon=None):
    """
    Process a single TIFF file with subsetting
    
    Args:
        tiff_path: Path to the TIFF file
        output_suffix: Suffix to append to create output filename (for temporary storage)
        polygon: Polygon for subsetting
        
    Returns:
        Bool: Whether subsetting was needed
    """
    # Create temporary output path (will be moved back to original in subset_tiff_with_polygon)
    temp_out_tiff = str(tiff_path).replace(".tiff", output_suffix)
    
    # Process file only if not already processed
    # Note: We can't check for temp_out_tiff since that gets moved back to input_tiff
    try:
        return subset_tiff_with_polygon(str(tiff_path), temp_out_tiff, polygon)
    except Exception as e:
        logging.error(f"Error subsetting {tiff_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_id", required=True)
    parser.add_argument("--tiff_path", required=True)
    parser.add_argument("--s2_start_date", default="2021-01-01")
    parser.add_argument("--s2_end_date",   default="2021-12-31")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=8,
                        help="Number of parallel processes for subsetting TIFFs")
    args = parser.parse_args()

    # read env vars
    DATA_DIR        = os.environ.get("S2_DATA_RAW")
    PROCESSED_S2_DIR= os.environ.get("S2_DATA_PROCESSED")
    RUST_EXE        = os.environ.get("RUST_S2_EXE")

    logging.info(f"[s2_processor] Using S2_DATA_RAW={DATA_DIR}")
    logging.info(f"[s2_processor] Using S2_DATA_PROCESSED={PROCESSED_S2_DIR}")
    logging.info(f"[s2_processor] Using RUST_EXE={RUST_EXE}")
    logging.info(f"[s2_processor] Using {args.num_processes} parallel processes")

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

    # If the first file needs subsetting, process the remaining files in parallel
    if needs_subsetting and len(tiff_list) > 1:
        # Prepare for parallel processing (starting from second file)
        process_func = partial(process_tiff, output_suffix="_subset.tiff", polygon=downstream_poly)
        
        # Use a pool of worker processes
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            logging.info(f"Starting parallel subsetting with {args.num_processes} processes for {len(tiff_list)-1} files")
            # Map the processing function to all TIFFs except the first one (already processed)
            results = pool.map(process_func, tiff_list[1:])
            
            # Count processed files
            processed_count = sum(1 for result in results if result)
            logging.info(f"Parallel subsetting completed. {processed_count} files were processed.")

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
    # Ensure proper multiprocessing behavior in Windows environments
    multiprocessing.set_start_method('spawn', force=True)
    main()