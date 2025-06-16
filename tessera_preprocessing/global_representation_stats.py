#!/usr/bin/env python3
"""
global_representation_stats.py

Manages a CSV file for tracking global 0.1-degree grid tile processing status.
Handles both d-pixel generation status and inference status across multiple years.
Designed for concurrent access from multiple VMs with file locking.

Main functions:
- create_csv: Creates initial CSV with all grid IDs
- init_csv: Initializes status columns by checking remote servers
- update_grid: Updates specific grid status
- get_one_grid_code_for_d_pixel_generation: Gets unprocessed grid for d-pixel
- get_one_grid_code_for_inference: Gets unprocessed grid for inference
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

# =================== Configuration ===================
HOSTNAME = socket.gethostname()
REMOTE_CSV_PATH = "/tank/zf281/global_representation_stats.csv"  # Path to the CSV file on otrera
LOCAL_TEMP_DIR = "/tmp"  # Local temporary directory for CSV operations
OTRERA_HOST = "zf281@otrera.caelum.ci.dev"
ANTIOPE_HOST = "zf281@antiope.caelum.ci.dev"
OTRERA_TIFF_PATH = "/tank/zf281/global_0.1_degree_tiff"
OTRERA_D_PIXEL_PATH = "/tank/zf281/global_0.1_degree_tiff_d_pixel"
ANTIOPE_REPRESENTATION_PATH = "/home/zf281/zf281/global_0.1_degree_representation"
YEARS = list(range(2017, 2025))  # 2017-2024
MAX_RETRIES = 10
RETRY_DELAY = 0.1  # seconds

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
    for year in sorted(YEARS, reverse=True):  # 2024, 2023, ..., 2017
        headers.extend([
            f'{year}_status',
            f'{year}_agent',
            f'{year}_infer_status',
            f'{year}_note'
        ])
    return headers

def remote_file_operation(func):
    """
    Decorator to handle remote CSV file operations with locking.
    Downloads CSV from otrera, performs operation, uploads back.
    Uses atomic lock file creation for concurrent access control.
    """
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                # Generate unique temp file names
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                pid = os.getpid()
                local_csv = os.path.join(LOCAL_TEMP_DIR, f"grid_stats_{timestamp}_{HOSTNAME}_{pid}.csv")
                lock_id = f"{HOSTNAME}_{pid}_{timestamp}"
                
                # Try to acquire remote lock using atomic file creation
                lock_acquired = False
                lock_cmd = f"""ssh {OTRERA_HOST} '
                    mkdir -p /tank/zf281 &&
                    if ln -s "{lock_id}" {REMOTE_CSV_PATH}.lock 2>/dev/null; then
                        echo "LOCKED"
                    else
                        echo "FAILED"
                    fi
                ' 2>/dev/null"""
                
                result = subprocess.run(lock_cmd, shell=True, capture_output=True, text=True)
                
                if "LOCKED" in result.stdout:
                    lock_acquired = True
                else:
                    # Check if lock is stale (older than 5 minutes)
                    stale_check_cmd = f"""ssh {OTRERA_HOST} '
                        if [ -L {REMOTE_CSV_PATH}.lock ]; then
                            lock_time=$(stat -c %Y {REMOTE_CSV_PATH}.lock 2>/dev/null || echo 0)
                            current_time=$(date +%s)
                            age=$((current_time - lock_time))
                            if [ $age -gt 300 ]; then
                                rm -f {REMOTE_CSV_PATH}.lock
                                echo "STALE_REMOVED"
                            else
                                echo "ACTIVE"
                            fi
                        else
                            echo "NO_LOCK"
                        fi
                    ' 2>/dev/null"""
                    
                    stale_result = subprocess.run(stale_check_cmd, shell=True, capture_output=True, text=True)
                    
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY + random.uniform(0, 0.1))
                        continue
                    else:
                        raise Exception(f"Failed to acquire remote lock after {MAX_RETRIES} attempts")
                
                try:
                    # Check if CSV exists on remote
                    check_cmd = f"ssh {OTRERA_HOST} 'test -f {REMOTE_CSV_PATH} && echo EXISTS || echo NOTFOUND' 2>/dev/null"
                    check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                    csv_exists = "EXISTS" in check_result.stdout
                    
                    # Download CSV if it exists
                    if csv_exists:
                        download_cmd = f"scp -q {OTRERA_HOST}:{REMOTE_CSV_PATH} {local_csv}"
                        subprocess.run(download_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
                    
                    # Execute the function with local CSV path
                    kwargs['local_csv_path'] = local_csv
                    kwargs['csv_exists'] = csv_exists
                    result = func(*args, **kwargs)
                    
                    # Upload the modified CSV back
                    if os.path.exists(local_csv):
                        upload_cmd = f"scp -q {local_csv} {OTRERA_HOST}:{REMOTE_CSV_PATH}"
                        subprocess.run(upload_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
                    
                    return result
                    
                finally:
                    # Release remote lock
                    if lock_acquired:
                        unlock_cmd = f"ssh {OTRERA_HOST} 'rm -f {REMOTE_CSV_PATH}.lock' 2>/dev/null"
                        subprocess.run(unlock_cmd, shell=True)
                    
                    # Clean up local files
                    if os.path.exists(local_csv):
                        try:
                            os.remove(local_csv)
                        except:
                            pass
                                
            except subprocess.CalledProcessError as e:
                print(f"SSH/SCP error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY + random.uniform(0, 0.1))
                else:
                    raise Exception(f"Failed to complete remote operation after {MAX_RETRIES} attempts")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY + random.uniform(0, 0.1))
                else:
                    raise e
                    
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
        print(f"Error accessing remote server: {e}")
        print(f"Make sure you have SSH access to {OTRERA_HOST}")
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
        print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
        tqdm = None
    
    # Read existing CSV
    @remote_file_operation
    def read_csv(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.")
            sys.exit(1)
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
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
            if tqdm:
                pbar = tqdm(rows, desc=f"  Updating {year} d-pixel status", leave=False)
                for row in pbar:
                    grid_id = row['grid_id']
                    if grid_id in grid_data:
                        npy_count = grid_data[grid_id]
                        if npy_count == REQUIRED_D_PIXEL_FILES:
                            row[f'{year}_status'] = 'processed'
                        elif npy_count > 0:
                            row[f'{year}_status'] = 'broken'
                        pbar.set_postfix_str(f"Grid: {grid_id[:20]}... Status: {row[f'{year}_status']}")
            else:
                for i, row in enumerate(rows):
                    grid_id = row['grid_id']
                    if grid_id in grid_data:
                        npy_count = grid_data[grid_id]
                        if npy_count == REQUIRED_D_PIXEL_FILES:
                            row[f'{year}_status'] = 'processed'
                        elif npy_count > 0:
                            row[f'{year}_status'] = 'broken'
                    if i % 100 == 0:
                        print(f"    Processed {i}/{total_grids} grids...", end='\r')
                print(f"    Processed {total_grids}/{total_grids} grids")
            
            # Log summary
            processed_count = sum(1 for row in rows if row.get(f'{year}_status') == 'processed')
            broken_count = sum(1 for row in rows if row.get(f'{year}_status') == 'broken')
            print(f"  Summary: {processed_count} processed, {broken_count} broken, {total_grids - processed_count - broken_count} not found")
                        
        except Exception as e:
            print(f"  Error checking year {year}: {e}")
    
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
            if tqdm:
                pbar = tqdm(rows, desc=f"  Updating {year} inference status", leave=False)
                for row in pbar:
                    grid_id = row['grid_id']
                    if grid_id in grid_data:
                        npy_count = grid_data[grid_id]
                        if npy_count == REQUIRED_REPRESENTATION_FILES:
                            row[f'{year}_infer_status'] = 'processed'
                        pbar.set_postfix_str(f"Grid: {grid_id[:20]}... Status: {row.get(f'{year}_infer_status', 'not found')}")
            else:
                for i, row in enumerate(rows):
                    grid_id = row['grid_id']
                    if grid_id in grid_data:
                        npy_count = grid_data[grid_id]
                        if npy_count == REQUIRED_REPRESENTATION_FILES:
                            row[f'{year}_infer_status'] = 'processed'
                    if i % 100 == 0:
                        print(f"    Processed {i}/{total_grids} grids...", end='\r')
                print(f"    Processed {total_grids}/{total_grids} grids")
            
            # Log summary
            processed_count = sum(1 for row in rows if row.get(f'{year}_infer_status') == 'processed')
            print(f"  Summary: {processed_count} processed, {total_grids - processed_count} not processed")
                        
        except Exception as e:
            print(f"  Error checking year {year}: {e}")
    
    # Write updated CSV
    @remote_file_operation
    def write_csv(rows_to_write, **kwargs):
        local_csv_path = kwargs['local_csv_path']
        headers = get_csv_headers()
        
        with open(local_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_to_write)
    
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
        python global_representation_stats.py --option update grid_-0.05_50.75 2024_agent node123
    """
    @remote_file_operation
    def update(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.")
            sys.exit(1)
            
        # Read CSV
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            headers = reader.fieldnames
        
        # Check if column exists
        if column_name not in headers:
            print(f"Error: Column '{column_name}' not found in CSV")
            sys.exit(1)
        
        # Find and update the grid
        found = False
        for row in rows:
            if row['grid_id'] == grid_id:
                row[column_name] = content
                found = True
                break
        
        if not found:
            print(f"Error: Grid '{grid_id}' not found in CSV")
            sys.exit(1)
        
        # Write back
        with open(local_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    
    update()
    print(f"Updated grid '{grid_id}': '{column_name}' = '{content}'")

def get_one_grid_code_for_d_pixel_generation(year, random_mode=False):
    """
    Returns one grid ID that hasn't been processed for d-pixel generation in the specified year.
    Checks the {year}_status column - returns grids where this column is empty.
    
    Args:
        year: The year to check (e.g., 2024)
        random_mode: If True, returns a random unprocessed grid; if False, returns the first one
    
    Returns:
        A grid ID string if found, empty string if none available
    
    Usage example:
        python global_representation_stats.py --option get_d_pixel 2024
        python global_representation_stats.py --option get_d_pixel 2024 random
    """
    @remote_file_operation
    def get_grid(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.")
            return ""
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        status_column = f'{year}_status'
        unprocessed_grids = []
        
        for row in rows:
            status = row.get(status_column, '').strip()
            if not status:  # Empty means not processed
                unprocessed_grids.append(row['grid_id'])
        
        if not unprocessed_grids:
            return ""
        
        if random_mode:
            return random.choice(unprocessed_grids)
        else:
            return unprocessed_grids[0]
    
    return get_grid()

def get_one_grid_code_for_inference(year, random_mode=False):
    """
    Returns one grid ID that hasn't been processed for inference in the specified year.
    Checks the {year}_infer_status column - returns grids where this column is empty
    AND where {year}_status is 'processed' (d-pixel is ready).
    
    Args:
        year: The year to check (e.g., 2024)
        random_mode: If True, returns a random unprocessed grid; if False, returns the first one
    
    Returns:
        A grid ID string if found, empty string if none available
    
    Usage example:
        python global_representation_stats.py --option get_inference 2024
        python global_representation_stats.py --option get_inference 2024 random
    """
    @remote_file_operation
    def get_grid(**kwargs):
        local_csv_path = kwargs['local_csv_path']
        if not kwargs['csv_exists']:
            print("Error: CSV file does not exist. Run create_csv first.")
            return ""
            
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        status_column = f'{year}_status'
        infer_column = f'{year}_infer_status'
        unprocessed_grids = []
        
        for row in rows:
            d_pixel_status = row.get(status_column, '').strip()
            infer_status = row.get(infer_column, '').strip()
            
            # Only consider grids where d-pixel is processed but inference is not done
            if d_pixel_status == 'processed' and not infer_status:
                unprocessed_grids.append(row['grid_id'])
        
        if not unprocessed_grids:
            return ""
        
        if random_mode:
            return random.choice(unprocessed_grids)
        else:
            return unprocessed_grids[0]
    
    return get_grid()

def main():
    parser = argparse.ArgumentParser(description="Manage global grid processing status in CSV.")
    parser.add_argument("--option", required=True, 
                        choices=["create_csv", "init", "update", "get_d_pixel", "get_inference"],
                        help="Operation to perform")
    parser.add_argument("args", nargs="*", help="Additional arguments for the selected option")
    args = parser.parse_args()

    if args.option == "create_csv":
        create_csv()
    elif args.option == "init":
        init_csv()
    elif args.option == "update":
        if len(args.args) != 3:
            print("Usage: python global_representation_stats.py --option update <grid_id> <column_name> <content>")
            sys.exit(1)
        grid_id, column_name, content = args.args
        update_grid(grid_id, column_name, content)
    elif args.option == "get_d_pixel":
        if len(args.args) < 1:
            print("Usage: python global_representation_stats.py --option get_d_pixel <year> [random]")
            sys.exit(1)
        year = int(args.args[0])
        random_mode = len(args.args) > 1 and args.args[1].lower() == "random"
        grid_id = get_one_grid_code_for_d_pixel_generation(year, random_mode)
        if grid_id:
            print(grid_id)
    elif args.option == "get_inference":
        if len(args.args) < 1:
            print("Usage: python global_representation_stats.py --option get_inference <year> [random]")
            sys.exit(1)
        year = int(args.args[0])
        random_mode = len(args.args) > 1 and args.args[1].lower() == "random"
        grid_id = get_one_grid_code_for_inference(year, random_mode)
        if grid_id:
            print(grid_id)
    else:
        print("Invalid option.")
        sys.exit(1)

if __name__ == "__main__":
    main()