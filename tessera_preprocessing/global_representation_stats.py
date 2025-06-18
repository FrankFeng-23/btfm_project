#!/usr/bin/env python3
"""
global_representation_stats.py

Manages a CSV file for tracking global 0.1-degree grid tile
processing status.
Handles both d-pixel generation status and inference status
across multiple years.
Designed for concurrent access from multiple VMs with file
locking.

Main functions:
- create_csv: Creates initial CSV with all grid IDs
- init_csv: Initializes status columns by checking remote
servers
- update_grid: Updates specific grid status
- get_one_grid_code_for_d_pixel_generation: Gets unprocessed
grid for d-pixel
- get_one_grid_code_for_inference: Gets unprocessed grid for
inference
"""

import argparse
import os
import sys
import csv
import time
import random
import socket
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# =================== Configuration (MODIFIED) ===================
HOSTNAME = socket.gethostname()
REMOTE_CSV_PATH = "/tank/zf281/global_representation_stats.csv"
# 使用一个原子锁目录，而不是一个包含多个文件的目录
REMOTE_LOCK_PATH = "/tank/zf281/global_stats.csv.lockdir"
LOCAL_TEMP_DIR = "/tmp"
OTRERA_HOST = "zf281@otrera.caelum.ci.dev"
ANTIOPE_HOST = "zf281@antiope.caelum.ci.dev"
OTRERA_TIFF_PATH = "/tank/zf281/global_0.1_degree_tiff"
OTRERA_D_PIXEL_PATH = "/tank/zf281/global_0.1_degree_tiff_d_pixel"
ANTIOPE_REPRESENTATION_PATH = "/home/zf281/zf281/global_0.1_degree_representation"
YEARS = list(range(2017, 2025))
MAX_RETRIES = 30  # 稍微增加重试次数以应对高并发
RETRY_DELAY = 0.5 # 增加基础延迟

# Required .npy files for d-pixel (9 files)
REQUIRED_D_PIXEL_FILES = 9
# Required .npy files for representation (2 files)
REQUIRED_REPRESENTATION_FILES = 2

def get_csv_headers():
    """
    Generate CSV headers based on years.
    Returns: grid_id, 2024_status, 2024_agent, 2024_infer_status, 2024_note, 2023_status, ...
    """
    headers = ['grid_id']
    for year in sorted(YEARS, reverse=True): # 2024, 2023, ..., 2017
        headers.extend([
            f'{year}_status',
            f'{year}_agent',
            f'{year}_infer_status',
            f'{year}_note'
        ])
    return headers


def remote_file_operation(func):
    """
    Decorator to handle remote CSV file operations with a robust, atomic lock.
    Uses 'mkdir' as an atomic locking mechanism.
    Downloads CSV from otrera, performs operation, uploads back.
    """
    def wrapper(*args, **kwargs):
        # Generate a unique identifier for local temp file
        unique_id = f"{HOSTNAME}_{os.getpid()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        local_csv = os.path.join(LOCAL_TEMP_DIR, f"grid_stats_{unique_id}.csv")
        lock_acquired = False

        for attempt in range(MAX_RETRIES):
            try:
                # 1. Acquire lock using atomic mkdir
                acquire_cmd = f"ssh {OTRERA_HOST} 'mkdir {REMOTE_LOCK_PATH}'"
                # If mkdir fails because the dir exists, it returns a non-zero exit code
                subprocess.run(acquire_cmd, shell=True, check=True, capture_output=True, text=True)
                lock_acquired = True

                # --- LOCK ACQUIRED ---

                # 2. Download CSV if it exists
                check_cmd = f"ssh {OTRERA_HOST} 'test -f {REMOTE_CSV_PATH} && echo EXISTS || echo NOTFOUND' 2>/dev/null"
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                csv_exists = "EXISTS" in check_result.stdout

                if csv_exists:
                    download_cmd = f"scp -q {OTRERA_HOST}:{REMOTE_CSV_PATH} {local_csv}"
                    subprocess.run(download_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)

                # 3. Execute the wrapped function
                kwargs['local_csv_path'] = local_csv
                kwargs['csv_exists'] = csv_exists
                func_result = func(*args, **kwargs)

                # 4. Upload the modified CSV back
                if os.path.exists(local_csv):
                    upload_cmd = f"scp -q {local_csv} {OTRERA_HOST}:{REMOTE_CSV_PATH}"
                    subprocess.run(upload_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
                
                return func_result

            except subprocess.CalledProcessError:
                # This likely means 'mkdir' failed because the lock is held.
                # print(f"DEBUG: Attempt {attempt + 1}/{MAX_RETRIES} failed to acquire lock. Waiting...")
                time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
                continue # Go to next attempt in the loop

            except Exception as e:
                print(f"An unexpected error occurred during remote operation: {e}", file=sys.stderr)
                # If a critical error occurs, we still try to release the lock in 'finally'
                # and then re-raise the exception.
                raise e

            finally:
                # 5. Release lock if it was acquired
                if lock_acquired:
                    release_cmd = f"ssh {OTRERA_HOST} 'rmdir {REMOTE_LOCK_PATH}'"
                    subprocess.run(release_cmd, shell=True, capture_output=True) # Don't check, just try
                
                # Clean up local temp file
                if os.path.exists(local_csv):
                    try:
                        os.remove(local_csv)
                    except OSError:
                        pass
        
        # If the loop finishes without returning, it means we failed to acquire the lock
        raise Exception(f"Failed to acquire remote lock on {REMOTE_LOCK_PATH} after {MAX_RETRIES} attempts.")

    return wrapper

def create_csv():
    """
    Creates the initial CSV file with all grid IDs from the TIFF directory.
    Only populates the grid_id column, leaving all status columns empty.

    Usage example:
        python global_representation_stats.py --option create_csv
    """
    print("Creating CSV file...")

    # Ensure remote directory exists
    mkdir_cmd = f"ssh {OTRERA_HOST} 'mkdir -p /tank/zf281'"
    subprocess.run(mkdir_cmd, shell=True, stderr=subprocess.DEVNULL)

    # Get list of grid files from otrera
    try:
        cmd = f"ssh {OTRERA_HOST} 'ls {OTRERA_TIFF_PATH}/*.tiff 2>/dev/null'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        tiff_files = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error accessing remote server: {e}", file=sys.stderr)
        print(f"Make sure you have SSH access to {OTRERA_HOST}", file=sys.stderr)
        sys.exit(1)

    # Extract grid IDs (remove path and .tiff extension)
    grid_ids = []
    for tiff_path in tiff_files:
        if tiff_path:
            grid_id = os.path.basename(tiff_path).replace('.tiff', '')
            grid_ids.append(grid_id)

    grid_ids.sort()  # Sort for consistency

    # Create CSV with headers
    headers = get_csv_headers()

    @remote_file_operation
    def write_csv(**kwargs):
        local_csv_path = kwargs['local_csv_path']

        with open(local_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            # Write grid IDs with empty status columns
            for grid_id in grid_ids:
                row = {'grid_id': grid_id}
                # All other columns remain empty
                writer.writerow(row)

    write_csv()
    print(f"CSV created with {len(grid_ids)} grid IDs at {OTRERA_HOST}:{REMOTE_CSV_PATH}")

def clean_row(row, headers):
    """
    Clean a row dictionary to only contain valid headers.
    Removes None keys and any keys not in the headers list.
    """
    cleaned = {}
    for key in headers:
        if key in row:
            cleaned[key] = row[key]
    return cleaned

def init_csv():
    """
    Initializes the CSV by checking d-pixel and inference status on remote servers.
    - Checks otrera for d-pixel files (9 npy files required)
    - Checks antiope for representation files (2 npy files required)
    - Updates status columns: 'processed' or 'broken' (leaves empty if not found)

    Usage example:
        python global_representation_stats.py --option init
    """
    print("Initializing CSV with current status...")

    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.", file=sys.stderr)
        tqdm = None

    # Get expected headers
    headers = get_csv_headers()

    # Read existing CSV
    @remote_file_operation
    def read_csv(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.", file=sys.stderr)
            sys.exit(1)
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                # Clean each row to remove None keys and extra fields
                cleaned_row = clean_row(row, headers)
                rows.append(cleaned_row)
        return rows

    rows = read_csv()
    total_grids = len(rows)
    print(f"Found {total_grids} grids to check")

    # Check d-pixel status on otrera
    print("\n=== Checking d-pixel status on otrera ===")
    for year in YEARS:
        print(f"\nProcessing year {year}...")
        try:
            # Use a single command to get all grid folders and their npy counts
            cmd = f"""ssh {OTRERA_HOST} '
                year_path="{OTRERA_D_PIXEL_PATH}/{year}"
                if [ -d "$year_path" ]; then
                    for grid_dir in "$year_path"/grid_*; do
                        if [ -d "$grid_dir" ]; then
                            grid_name=$(basename "$grid_dir")
                            npy_count=$(ls "$grid_dir"/*.npy 2>/dev/null | wc -l)
                            echo "$grid_name:$npy_count"
                        fi
                    done
                fi
            ' 2>/dev/null"""
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            grid_data = {}
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        grid_id, count = line.strip().split(':')
                        grid_data[grid_id] = int(count)
            
            print(f"  Found {len(grid_data)} grids with d-pixel data")
            
            # Update status in rows with progress bar
            iterator = tqdm(rows, desc=f"  Updating {year} d-pixel status", leave=False) if tqdm else rows
            for row in iterator:
                grid_id = row['grid_id']
                if grid_id in grid_data:
                    npy_count = grid_data[grid_id]
                    if npy_count == REQUIRED_D_PIXEL_FILES:
                        row[f'{year}_status'] = 'processed'
                    elif npy_count > 0:
                        row[f'{year}_status'] = 'broken'
                if tqdm:
                    iterator.set_postfix_str(f"Grid: {grid_id[:20]}... Status: {row.get(f'{year}_status', 'not found')}")

            # Log summary
            processed_count = sum(1 for row in rows if row.get(f'{year}_status') == 'processed')
            broken_count = sum(1 for row in rows if row.get(f'{year}_status') == 'broken')
            print(f"  Summary: {processed_count} processed, {broken_count} broken, {total_grids - processed_count - broken_count} not found")
                
        except Exception as e:
            print(f"  Error checking year {year}: {e}", file=sys.stderr)

    # Check inference status on antiope
    print("\n=== Checking inference status on antiope ===")
    for year in YEARS:
        print(f"\nProcessing year {year}...")
        try:
            # Use a single command to get all grid folders and their npy counts
            cmd = f"""ssh {ANTIOPE_HOST} '
                year_path="{ANTIOPE_REPRESENTATION_PATH}/{year}"
                if [ -d "$year_path" ]; then
                    for grid_dir in "$year_path"/grid_*; do
                        if [ -d "$grid_dir" ]; then
                            grid_name=$(basename "$grid_dir")
                            npy_count=$(ls "$grid_dir"/*.npy 2>/dev/null | wc -l)
                            echo "$grid_name:$npy_count"
                        fi
                    done
                fi
            ' 2>/dev/null"""
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            grid_data = {}
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        grid_id, count = line.strip().split(':')
                        grid_data[grid_id] = int(count)
            
            print(f"  Found {len(grid_data)} grids with inference data")
            
            # Update status in rows with progress bar
            iterator = tqdm(rows, desc=f"  Updating {year} inference status", leave=False) if tqdm else rows
            for row in iterator:
                grid_id = row['grid_id']
                if grid_id in grid_data:
                    npy_count = grid_data[grid_id]
                    if npy_count == REQUIRED_REPRESENTATION_FILES:
                        row[f'{year}_infer_status'] = 'processed'
                if tqdm:
                    iterator.set_postfix_str(f"Grid: {grid_id[:20]}... Status: {row.get(f'{year}_infer_status', 'not found')}")
            
            # Log summary
            processed_count = sum(1 for row in rows if row.get(f'{year}_infer_status') == 'processed')
            print(f"  Summary: {processed_count} processed, {total_grids - processed_count} not processed")
                
        except Exception as e:
            print(f"  Error checking year {year}: {e}", file=sys.stderr)

    # Write updated CSV
    @remote_file_operation
    def write_csv(rows_to_write, **kwargs):
        local_csv_path = kwargs['local_csv_path']
        headers = get_csv_headers()
        
        with open(local_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            # Clean each row before writing
            for row in rows_to_write:
                cleaned_row = clean_row(row, headers)
                writer.writerow(cleaned_row)

    print("\nWriting updated CSV...")
    write_csv(rows)

    # Print final summary
    print("\n=== Final Summary ===")
    for year in YEARS:
        d_pixel_processed = sum(1 for row in rows if row.get(f'{year}_status') == 'processed')
        d_pixel_broken = sum(1 for row in rows if row.get(f'{year}_status') == 'broken')
        inference_processed = sum(1 for row in rows if row.get(f'{year}_infer_status') == 'processed')
        print(f"Year {year}:")
        print(f"  D-pixel: {d_pixel_processed} processed, {d_pixel_broken} broken")
        print(f"  Inference: {inference_processed} processed")

    print("\nCSV initialization completed.")

def update_grid(grid_id, column_name, content):
    """
    Updates a specific grid's column with new content.

    Args:
        grid_id: The grid identifier (e.g., 'grid_-0.05_50.75')
        column_name: The column to update (e.g., '2024_status', '2023_agent')
        content: The new value for the column

    Usage example:
        python global_representation_stats.py --option update grid_-0.05_50.75 2024_status processing
    """
    @remote_file_operation
    def update(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.", file=sys.stderr)
            sys.exit(1)
            
        # Get expected headers
        headers = get_csv_headers()
            
        # Read CSV
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                # Clean each row to remove None keys and extra fields
                cleaned_row = clean_row(row, headers)
                rows.append(cleaned_row)
        
        # Check if column exists
        if column_name not in headers:
            print(f"Error: Column '{column_name}' not found in CSV", file=sys.stderr)
            sys.exit(1)
        
        # Find and update the grid
        found = False
        for row in rows:
            if row['grid_id'] == grid_id:
                row[column_name] = content
                found = True
                break
        
        if not found:
            # Do not exit, just print an error. The process might continue.
            print(f"Error: Grid '{grid_id}' not found in CSV", file=sys.stderr)
            return
        
        # Write back
        with open(local_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                cleaned_row = clean_row(row, headers)
                writer.writerow(cleaned_row)

    update()
    # This print goes to stdout, which is captured by the calling script
    print(f"Updated grid '{grid_id}': '{column_name}' = '{content}'")

def get_one_grid_code_for_d_pixel_generation(year, random_mode=False):
    """
    Returns one grid ID that needs d-pixel generation for the specified year.
    
    Selection logic:
    1. Skip if {year}_infer_status is 'processed' (already has inference results)
    2. Select if {year}_status is empty (not processed)
    3. If all grids have {year}_status filled, select 'broken' ones for reprocessing
    4. Return empty string if nothing needs processing
    """
    @remote_file_operation
    def get_grid(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.", file=sys.stderr)
            return ""
        
        # Get expected headers
        headers = get_csv_headers()
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                # Clean each row to remove None keys and extra fields
                cleaned_row = clean_row(row, headers)
                rows.append(cleaned_row)
        
        status_column = f'{year}_status'
        infer_column = f'{year}_infer_status'
        
        # Categorize grids
        unprocessed_grids = []
        broken_grids = []
        
        for row in rows:
            infer_status = row.get(infer_column, '').strip()
            d_pixel_status = row.get(status_column, '').strip()
            
            # Skip if inference is already processed
            if infer_status == 'processed':
                continue
            
            # If d-pixel status is empty, it needs processing
            if not d_pixel_status:
                unprocessed_grids.append(row['grid_id'])
            # If d-pixel status is 'broken', it needs reprocessing
            elif d_pixel_status == 'broken':
                broken_grids.append(row['grid_id'])
        
        # Priority: unprocessed first, then broken
        candidates = unprocessed_grids if unprocessed_grids else broken_grids
        
        if not candidates:
            return ""
        
        if random_mode:
            return random.choice(candidates)
        else:
            return candidates[0]

    return get_grid()

def get_one_grid_code_for_inference(year, random_mode=False):
    """
    Returns one grid ID that is ready for inference for the specified year.
    Condition: {year}_status is 'processed' AND {year}_infer_status is empty.
    """
    @remote_file_operation
    def get_grid(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.", file=sys.stderr)
            return None
        
        # Get expected headers
        headers = get_csv_headers()
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                # Clean each row to remove None keys and extra fields
                cleaned_row = clean_row(row, headers)
                rows.append(cleaned_row)
        
        status_column = f'{year}_status'
        infer_column = f'{year}_infer_status'
        unprocessed_grids = []
        
        # Debug: count total rows and processed rows
        total_rows = len(rows)
        processed_count = 0
        
        for row in rows:
            d_pixel_status = row.get(status_column, '').strip()
            infer_status = row.get(infer_column, '').strip()
            
            # Debug counting
            if d_pixel_status == 'processed':
                processed_count += 1
            
            # Only consider grids where d-pixel is processed but inference is not done
            if d_pixel_status == 'processed' and not infer_status:
                unprocessed_grids.append(row['grid_id'])
        
        # Debug output to stderr (won't interfere with grid_id output)
        print(f"DEBUG: Year {year} - Total rows: {total_rows}, Processed d-pixel: {processed_count}, Ready for inference: {len(unprocessed_grids)}", file=sys.stderr)
        
        if not unprocessed_grids:
            return None
        
        if random_mode:
            selected = random.choice(unprocessed_grids)
        else:
            selected = unprocessed_grids[0]
            
        print(f"DEBUG: Selected grid for year {year}: {selected}", file=sys.stderr)
        return selected

    result = get_grid()
    return result if result else ""

def main():
    parser = argparse.ArgumentParser(description="Manage global grid processing status in CSV.")
    parser.add_argument("--option", required=True,
                        choices=["create_csv", "init", "update", "get_d_pixel", "get_inference"],
                        help="Operation to perform")
    parser.add_argument("args", nargs="*", help="Additional arguments for the selected option")
    args = parser.parse_args()

    try:
        if args.option == "create_csv":
            if len(args.args) != 0:
                print("Usage: python global_representation_stats.py --option create_csv", file=sys.stderr)
                sys.exit(1)
            create_csv()

        elif args.option == "init":
            if len(args.args) != 0:
                print("Usage: python global_representation_stats.py --option init", file=sys.stderr)
                sys.exit(1)
            init_csv()

        elif args.option == "update":
            if len(args.args) != 3:
                print("Usage: python global_representation_stats.py --option update <grid_id> <column_name> <content>", file=sys.stderr)
                sys.exit(1)
            grid_id, column_name, content = args.args
            update_grid(grid_id, column_name, content)

        elif args.option == "get_d_pixel":
            if len(args.args) < 1 or len(args.args) > 2:
                print("Usage: python global_representation_stats.py --option get_d_pixel <year> [random]", file=sys.stderr)
                sys.exit(1)
            year = int(args.args[0])
            random_mode = len(args.args) > 1 and args.args[1].lower() == "random"
            grid_id = get_one_grid_code_for_d_pixel_generation(year, random_mode)
            if grid_id:
                print(grid_id)

        elif args.option == "get_inference":
            if len(args.args) < 1 or len(args.args) > 2:
                print("Usage: python global_representation_stats.py --option get_inference <year> [random]", file=sys.stderr)
                sys.exit(1)
            year = int(args.args[0])
            random_mode = len(args.args) > 1 and args.args[1].lower() == "random"
            grid_id = get_one_grid_code_for_inference(year, random_mode)
            if grid_id:
                print(grid_id)
            else:
                # Print nothing to stdout, but log to stderr for debugging
                print(f"DEBUG: No grids available for inference in year {year}", file=sys.stderr)

    except Exception as e:
        print(f"An error occurred in global_representation_stats.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()