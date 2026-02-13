#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized DPixel Processor with Block-based Processing

This script processes consolidated d-pixel data and re-tiles it into 
non-overlapping patches using a memory-efficient block-based approach.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
import traceback
import shutil
from functools import partial
import concurrent.futures
import threading
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine
import multiprocessing as mp
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('dpixel_processor')

def read_reference_tiff(tiff_path):
    """Read the reference TIFF file and create a binary mask."""
    try:
        with rasterio.open(tiff_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            bounds = src.bounds
            
            logger.info(f"Reference TIFF dimensions: {width}x{height}")
            logger.info(f"Reference TIFF CRS: {crs}")
            logger.info(f"Reference TIFF bounds: {bounds}")
            
            # Create binary mask (1=valid, 0=invalid)
            binary_mask = mask > 0
            valid_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            logger.info(f"Valid pixels: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.2f}%)")
            
            return binary_mask, transform, crs, width, height
            
    except Exception as e:
        logger.error(f"Error reading reference TIFF: {e}")
        raise

def get_file_info(d_pixel_dir):
    """Get information about the numpy files without loading them."""
    data_files = {
        'bands': os.path.join(d_pixel_dir, 'bands.npy'),
        'doys': os.path.join(d_pixel_dir, 'doys.npy'),
        'masks': os.path.join(d_pixel_dir, 'masks.npy'),
        'sar_ascending': os.path.join(d_pixel_dir, 'sar_ascending.npy'),
        'sar_ascending_doy': os.path.join(d_pixel_dir, 'sar_ascending_doy.npy'),
        'sar_descending': os.path.join(d_pixel_dir, 'sar_descending.npy'),
        'sar_descending_doy': os.path.join(d_pixel_dir, 'sar_descending_doy.npy')
    }
    
    # Check if all required files exist
    for key, file_path in data_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Get array shapes and dtypes without loading data
    file_info = {}
    for key, file_path in data_files.items():
        # Use numpy's load with mmap_mode='r' to peek at metadata
        temp_array = np.load(file_path, mmap_mode='r')
        file_info[key] = {
            'path': file_path,
            'shape': temp_array.shape,
            'dtype': temp_array.dtype,
            'size_mb': os.path.getsize(file_path) / 1024 / 1024
        }
        logger.info(f"{key}: shape={temp_array.shape}, dtype={temp_array.dtype}, size={file_info[key]['size_mb']:.1f}MB")
        del temp_array  # Important: release the memory map
    
    return file_info

def load_block_data(file_info, y_start, y_end, x_start, x_end):
    """Load a block of data from all numpy files."""
    block_data = {}
    
    # Load small arrays completely (they're tiny)
    small_arrays = ['doys', 'sar_ascending_doy', 'sar_descending_doy']
    for key in small_arrays:
        block_data[key] = np.load(file_info[key]['path'])
    
    # Load spatial blocks from large arrays
    large_arrays = ['bands', 'masks', 'sar_ascending', 'sar_descending']
    for key in large_arrays:
        # Use memory mapping to load only the required block
        full_array = np.load(file_info[key]['path'], mmap_mode='r')
        
        if len(full_array.shape) == 4:  # bands, sar_ascending, sar_descending
            # Shape: (T, H, W, C)
            block_data[key] = np.array(full_array[:, y_start:y_end, x_start:x_end, :])
        else:  # masks
            # Shape: (T, H, W)
            block_data[key] = np.array(full_array[:, y_start:y_end, x_start:x_end])
        
        del full_array  # Release memory map
    
    return block_data

def has_valid_data(x_min, y_min, patch_size, ref_mask):
    """Check if a patch contains any valid data."""
    if x_min >= ref_mask.shape[1] or y_min >= ref_mask.shape[0]:
        return False
    
    x_max = min(x_min + patch_size, ref_mask.shape[1])
    y_max = min(y_min + patch_size, ref_mask.shape[0])
    
    return np.any(ref_mask[y_min:y_max, x_min:x_max])

def extract_patch_from_block(block_data, block_x_start, block_y_start, 
                           patch_x_min, patch_y_min, patch_size):
    """Extract a patch from block data."""
    # Convert global coordinates to block-relative coordinates
    local_x_min = patch_x_min - block_x_start
    local_y_min = patch_y_min - block_y_start
    local_x_max = min(local_x_min + patch_size, block_data['bands'].shape[2])
    local_y_max = min(local_y_min + patch_size, block_data['bands'].shape[1])
    
    # Calculate actual dimensions
    actual_h = local_y_max - local_y_min
    actual_w = local_x_max - local_x_min
    
    # Create patch data
    patch_data = {}
    
    # Copy small arrays directly
    patch_data['doys'] = block_data['doys'].copy()
    patch_data['sar_ascending_doy'] = block_data['sar_ascending_doy'].copy()
    patch_data['sar_descending_doy'] = block_data['sar_descending_doy'].copy()
    
    # Extract spatial data - only allocate the exact size needed
    if actual_h > 0 and actual_w > 0:
        # Extract from block data directly to the right size
        patch_data['bands'] = np.zeros((block_data['bands'].shape[0], patch_size, patch_size, 
                                      block_data['bands'].shape[3]), dtype=block_data['bands'].dtype)
        patch_data['masks'] = np.zeros((block_data['masks'].shape[0], patch_size, patch_size), 
                                     dtype=block_data['masks'].dtype)
        patch_data['sar_ascending'] = np.zeros((block_data['sar_ascending'].shape[0], patch_size, patch_size,
                                              block_data['sar_ascending'].shape[3]), dtype=block_data['sar_ascending'].dtype)
        patch_data['sar_descending'] = np.zeros((block_data['sar_descending'].shape[0], patch_size, patch_size,
                                               block_data['sar_descending'].shape[3]), dtype=block_data['sar_descending'].dtype)
        
        # Copy actual data
        patch_data['bands'][:, :actual_h, :actual_w, :] = block_data['bands'][:, local_y_min:local_y_max, local_x_min:local_x_max, :]
        patch_data['masks'][:, :actual_h, :actual_w] = block_data['masks'][:, local_y_min:local_y_max, local_x_min:local_x_max]
        patch_data['sar_ascending'][:, :actual_h, :actual_w, :] = block_data['sar_ascending'][:, local_y_min:local_y_max, local_x_min:local_x_max, :]
        patch_data['sar_descending'][:, :actual_h, :actual_w, :] = block_data['sar_descending'][:, local_y_min:local_y_max, local_x_min:local_x_max, :]
    else:
        # Create empty arrays if no valid area
        patch_data['bands'] = np.zeros((block_data['bands'].shape[0], patch_size, patch_size, 
                                      block_data['bands'].shape[3]), dtype=block_data['bands'].dtype)
        patch_data['masks'] = np.zeros((block_data['masks'].shape[0], patch_size, patch_size), 
                                     dtype=block_data['masks'].dtype)
        patch_data['sar_ascending'] = np.zeros((block_data['sar_ascending'].shape[0], patch_size, patch_size,
                                              block_data['sar_ascending'].shape[3]), dtype=block_data['sar_ascending'].dtype)
        patch_data['sar_descending'] = np.zeros((block_data['sar_descending'].shape[0], patch_size, patch_size,
                                               block_data['sar_descending'].shape[3]), dtype=block_data['sar_descending'].dtype)
    
    return patch_data

def create_roi_tiff(out_dir, x_min, y_min, patch_size, ref_mask, transform, crs):
    """Create a ROI TIFF file for a patch."""
    roi_path = os.path.join(out_dir, 'roi.tiff')
    
    try:
        x_max = min(x_min + patch_size, ref_mask.shape[1])
        y_max = min(y_min + patch_size, ref_mask.shape[0])
        actual_h = y_max - y_min
        actual_w = x_max - x_min
        
        roi_data = np.zeros((patch_size, patch_size), dtype=np.uint8)
        
        if x_min < ref_mask.shape[1] and y_min < ref_mask.shape[0]:
            roi_data[:actual_h, :actual_w] = ref_mask[y_min:y_max, x_min:x_max].astype(np.uint8)
        
        patch_transform = Affine(
            transform.a, transform.b, transform.c + x_min * transform.a,
            transform.d, transform.e, transform.f + y_min * transform.e
        )
        
        profile = {
            'driver': 'GTiff', 'height': patch_size, 'width': patch_size, 'count': 1,
            'dtype': roi_data.dtype, 'crs': crs, 'transform': patch_transform,
            'compress': 'lzw', 'tiled': True, 'blockxsize': 256, 'blockysize': 256
        }
        
        with rasterio.open(roi_path, 'w', **profile) as dst:
            dst.write(roi_data, 1)
            
        return True
    except Exception:
        return False

def save_patch(patch_data, out_dir, x_min, y_min, patch_size, ref_mask, transform, crs):
    """Save patch data to the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # Save numpy files
        files_to_save = [
            ('bands.npy', patch_data['bands']),
            ('masks.npy', patch_data['masks']),
            ('doys.npy', patch_data['doys']),
            ('sar_ascending.npy', patch_data['sar_ascending']),
            ('sar_ascending_doy.npy', patch_data['sar_ascending_doy']),
            ('sar_descending.npy', patch_data['sar_descending']),
            ('sar_descending_doy.npy', patch_data['sar_descending_doy'])
        ]
        
        for filename, array in files_to_save:
            np.save(os.path.join(out_dir, filename), array)
        
        # Create ROI TIFF
        roi_success = create_roi_tiff(out_dir, x_min, y_min, patch_size, ref_mask, transform, crs)
        
        return True
    except Exception as e:
        logger.error(f"Error saving patch: {e}")
        return False

def process_patch_in_block(args):
    """Process a single patch within a block."""
    (patch_x_min, patch_y_min, patch_size, block_data, block_x_start, block_y_start,
     ref_mask, transform, crs, out_dir, skip_existing) = args
    
    x_max = min(patch_x_min + patch_size, ref_mask.shape[1])
    y_max = min(patch_y_min + patch_size, ref_mask.shape[0])
    
    patch_dir_name = f"{patch_x_min}_{patch_y_min}_{x_max}_{y_max}"
    patch_out_dir = os.path.join(out_dir, patch_dir_name)
    
    # Check if patch has valid data
    if not has_valid_data(patch_x_min, patch_y_min, patch_size, ref_mask):
        return True
    
    # Skip if already exists
    if skip_existing and os.path.exists(patch_out_dir):
        if os.path.exists(os.path.join(patch_out_dir, 'bands.npy')):
            return True
    
    try:
        # Extract patch from block
        patch_data = extract_patch_from_block(
            block_data, block_x_start, block_y_start,
            patch_x_min, patch_y_min, patch_size
        )
        
        # Save patch
        success = save_patch(
            patch_data, patch_out_dir, patch_x_min, patch_y_min, 
            patch_size, ref_mask, transform, crs
        )
        
        return success
    except Exception as e:
        logger.error(f"Error processing patch {patch_dir_name}: {e}")
        return False

def process_super_block(file_info, ref_mask, transform, crs, out_dir, patch_size,
                       block_x_start, block_y_start, block_width, block_height,
                       skip_existing, num_workers):
    """Process a super block of data."""
    
    logger.info(f"Processing super block: x={block_x_start}-{block_x_start+block_width}, "
                f"y={block_y_start}-{block_y_start+block_height}")
    
    # Load block data
    block_x_end = min(block_x_start + block_width, ref_mask.shape[1])
    block_y_end = min(block_y_start + block_height, ref_mask.shape[0])
    
    try:
        block_data = load_block_data(file_info, block_y_start, block_y_end, 
                                   block_x_start, block_x_end)
        logger.info(f"Loaded block data into memory")
    except Exception as e:
        logger.error(f"Failed to load block data: {e}")
        return []
    
    # Generate patches within this block
    patches_in_block = []
    for y in range(block_y_start, block_y_end, patch_size):
        for x in range(block_x_start, block_x_end, patch_size):
            # Check if patch overlaps with block
            if (x < block_x_end and y < block_y_end and
                x + patch_size > block_x_start and y + patch_size > block_y_start):
                patches_in_block.append((x, y, patch_size, block_data, block_x_start, block_y_start,
                                       ref_mask, transform, crs, out_dir, skip_existing))
    
    logger.info(f"Processing {len(patches_in_block)} patches in this block")
    
    # Process patches in parallel within the block
    results = []
    if num_workers > 1 and len(patches_in_block) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(patches_in_block))) as executor:
            futures = [executor.submit(process_patch_in_block, args) for args in patches_in_block]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="Block patches", leave=False):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in patch processing: {e}")
                    results.append(False)
    else:
        # Single-threaded processing within block
        for args in tqdm(patches_in_block, desc="Block patches", leave=False):
            result = process_patch_in_block(args)
            results.append(result)
    
    # Clean up block data
    del block_data
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process d-pixel data into regular patches with block-based processing')
    parser.add_argument('--tiff_path', required=True, help='Path to the reference TIFF')
    parser.add_argument('--d_pixel_dir', required=True, help='Directory containing d-pixel data')
    parser.add_argument('--patch_size', type=int, default=500, help='Size of output patches in pixels')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads per block')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing patches')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    parser.add_argument('--block_size', type=int, default=2000, help='Size of super blocks for processing')
    
    """Example usage:
    python dpixel_retiler.py \
        --tiff_path /your_data_path/roi.tif \
        --d_pixel_dir /your_data_path/data_processed \
        --patch_size 500 \
        --out_dir /your_data_path/retiled_d_pixel \
        --num_workers 16 \
        --overwrite \
        --block_size 2000
    """
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tiff_path):
        logger.error(f"Input TIFF not found: {args.tiff_path}")
        return 1
    
    if not os.path.exists(args.d_pixel_dir):
        logger.error(f"d_pixel_dir not found: {args.d_pixel_dir}")
        return 1
    
    # Create output directory
    if os.path.exists(args.out_dir):
        if not args.overwrite:
            logger.warning(f"Output directory {args.out_dir} already exists")
        else:
            shutil.rmtree(args.out_dir)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read reference TIFF
    logger.info(f"Reading reference TIFF: {args.tiff_path}")
    ref_mask, transform, crs, width, height = read_reference_tiff(args.tiff_path)
    
    # Get file information
    logger.info(f"Analyzing d-pixel files: {args.d_pixel_dir}")
    file_info = get_file_info(args.d_pixel_dir)
    
    # Calculate super blocks
    block_size = args.block_size
    num_blocks_x = (width + block_size - 1) // block_size
    num_blocks_y = (height + block_size - 1) // block_size
    
    logger.info(f"Processing in {num_blocks_x}x{num_blocks_y} super blocks of size {block_size}")
    logger.info(f"Using {args.num_workers} workers per block")
    
    # Process each super block
    start_time = time.time()
    all_results = []
    total_patches = 0
    
    with tqdm(total=num_blocks_x * num_blocks_y, desc="Super blocks") as pbar:
        for block_y in range(num_blocks_y):
            for block_x in range(num_blocks_x):
                block_x_start = block_x * block_size
                block_y_start = block_y * block_size
                block_width = min(block_size, width - block_x_start)
                block_height = min(block_size, height - block_y_start)
                
                results = process_super_block(
                    file_info, ref_mask, transform, crs, args.out_dir, args.patch_size,
                    block_x_start, block_y_start, block_width, block_height,
                    args.skip_existing, args.num_workers
                )
                
                all_results.extend(results)
                total_patches += len(results)
                pbar.update(1)
    
    # Summary
    end_time = time.time()
    success_count = sum(1 for r in all_results if r)
    logger.info(f"Processing complete: {success_count}/{total_patches} patches processed successfully")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())