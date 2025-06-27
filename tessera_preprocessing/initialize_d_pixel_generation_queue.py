#!/usr/bin/env python3
"""
initialize_d_pixel_generation_queue.py

Enhanced version that:
1. Gets all grid TIFFs from both directories:
   - /tank/zf281/global_0.1_degree_tiff
   - /tank/zf281/global_0.1_degree_tiff_express
2. Handles different year requirements for each source
3. Prioritizes express TIFFs when duplicates exist
4. FIRST checks if representation exists in antiope:/tank/zf281/global_0.1_degree_representation
5. THEN checks completion status in otrera:/tank/zf281/global_0.1_degree_tiff_d_pixel
6. Creates tasks only if both representation and d-pixel are missing/incomplete
7. Optimized for speed with connection pooling and batch operations
"""
import subprocess
import os
from pathlib import Path
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
import sys
from tqdm import tqdm
import json

# --- Configuration ---
OTRERA_HOST = "zf281@otrera.cl.cam.ac.uk"
ANTIOPE_HOST = "zf281@antiope.cl.cam.ac.uk"

# Paths
REMOTE_TIFF_BASE_PATH = "/tank/zf281/global_0.1_degree_tiff"
REMOTE_TIFF_EXPRESS_PATH = "/tank/zf281/global_0.1_degree_tiff_express"
# REMOTE_TIFF_EXPRESS_PATH = "/tank/zf281/global_0.1_degree_tiff_express_maddy"
REMOTE_D_PIXEL_BASE_PATH = "/tank/zf281/global_0.1_degree_tiff_d_pixel"
REMOTE_REPRESENTATION_BASE_PATH = "/tank/zf281/global_0.1_degree_representation"
REMOTE_TASK_QUEUE_BASE = "/tank/zf281/task_queue/d_pixel_generation"

# Years to process for each source
# Empty list means don't process that source
YEARS_TO_PROCESS_NORMAL = list(range(2024, 2016, -1))  # [2024, 2023, ..., 2017]
YEARS_TO_PROCESS_EXPRESS = [2024]  # Only 2024 for express
# YEARS_TO_PROCESS_EXPRESS = [2018,2019,2021]

# Performance settings
MAX_WORKERS = 20  # Number of parallel threads
BATCH_SIZE = 100  # Number of tasks to create in a single SSH command

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('initialize_d_pixel_queue.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SSHConnectionPool:
    """Manages persistent SSH connections with ControlMaster"""
    def __init__(self, host):
        self.host = host
        self.control_path = f"/tmp/ssh_mux_{host.replace('@', '_')}"
        self._setup_control_master()
    
    def _setup_control_master(self):
        """Establish a persistent SSH connection"""
        cmd = [
            'ssh', '-fN', '-M', 
            '-o', 'ControlMaster=yes',
            '-o', f'ControlPath={self.control_path}',
            '-o', 'ControlPersist=600',  # Keep alive for 10 minutes
            self.host
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Established SSH control master for {self.host}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup control master: {e}")
            raise
    
    def run_command(self, cmd):
        """Execute command using the persistent connection"""
        ssh_cmd = [
            'ssh',
            '-o', f'ControlPath={self.control_path}',
            self.host,
            cmd
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}\nError: {result.stderr}")
            return None
        return result.stdout.strip()
    
    def close(self):
        """Close the control master connection"""
        cmd = ['ssh', '-O', 'exit', '-o', f'ControlPath={self.control_path}', self.host]
        subprocess.run(cmd, capture_output=True)
        logger.info(f"Closed SSH control master for {self.host}")

def batch_create_files(ssh_pool, remote_dir, filenames):
    """Create multiple files in a single SSH command"""
    if not filenames:
        return True
    
    # Create files in batches
    for i in range(0, len(filenames), BATCH_SIZE):
        batch = filenames[i:i+BATCH_SIZE]
        # Use touch to create multiple files at once
        file_list = ' '.join([f'"{f}"' for f in batch])
        cmd = f'cd {remote_dir} && touch {file_list}'
        result = ssh_pool.run_command(cmd)
        if result is None:
            return False
    return True

def get_grid_tiffs_from_path(ssh_pool, path):
    """Get grid TIFF files from a specific path"""
    logger.info(f"Fetching grid TIFFs from {path}...")
    
    cmd = f'find {path} -maxdepth 1 -name "grid_*.tiff" -type f -printf "%f\n" | sort'
    result = ssh_pool.run_command(cmd)
    
    if not result:
        logger.warning(f"No grid TIFFs found in {path} or path doesn't exist")
        return []
    
    # Extract grid names (remove .tiff extension)
    grid_names = []
    for line in result.split('\n'):
        if line.strip() and line.endswith('.tiff'):
            grid_name = line.strip()[:-5]  # Remove .tiff
            grid_names.append(grid_name)
    
    logger.info(f"Found {len(grid_names)} grid TIFFs in {path}")
    return grid_names

def get_all_grid_tiffs_with_source(ssh_pool):
    """Get all grid TIFF files from both directories and track their source"""
    grid_to_source = {}  # Maps grid_name to source path
    grid_to_years = {}   # Maps grid_name to years to process
    
    # Get normal TIFFs if years list is not empty
    if YEARS_TO_PROCESS_NORMAL:
        normal_grids = get_grid_tiffs_from_path(ssh_pool, REMOTE_TIFF_BASE_PATH)
        for grid in normal_grids:
            grid_to_source[grid] = REMOTE_TIFF_BASE_PATH
            grid_to_years[grid] = YEARS_TO_PROCESS_NORMAL
    
    # Get express TIFFs if years list is not empty
    if YEARS_TO_PROCESS_EXPRESS:
        express_grids = get_grid_tiffs_from_path(ssh_pool, REMOTE_TIFF_EXPRESS_PATH)
        for grid in express_grids:
            # Express overrides normal if duplicate
            if grid in grid_to_source:
                logger.info(f"Grid {grid} found in both directories, using express version")
            grid_to_source[grid] = REMOTE_TIFF_EXPRESS_PATH
            grid_to_years[grid] = YEARS_TO_PROCESS_EXPRESS
    
    logger.info(f"Total unique grids: {len(grid_to_source)}")
    return grid_to_source, grid_to_years

def check_representations_batch(ssh_pool, year, grid_names):
    """Check multiple grids for completed representations in one command"""
    if not grid_names:
        return {}
    
    # Build a command that checks all grids at once
    grid_paths = [f'{REMOTE_REPRESENTATION_BASE_PATH}/{year}/{grid_name}' for grid_name in grid_names]
    
    # Use find to count .npy files in each directory
    cmd = 'for dir in ' + ' '.join([f'"{p}"' for p in grid_paths]) + '; do '
    cmd += 'if [ -d "$dir" ]; then '
    cmd += 'count=$(find "$dir" -maxdepth 1 -name "*.npy" -type f 2>/dev/null | wc -l); '
    cmd += 'echo "$(basename "$dir"):$count"; '
    cmd += 'fi; done'
    
    result = ssh_pool.run_command(cmd)
    
    completed = {}
    if result:
        for line in result.split('\n'):
            if ':' in line:
                grid_name, count = line.strip().split(':', 1)
                completed[grid_name] = int(count) >= 2  # 2 npy files means complete representation
    
    return completed

def check_d_pixels_batch(ssh_pool, year, grid_names):
    """Check multiple grids for completed d-pixels in one command"""
    if not grid_names:
        return {}
    
    # Build a command that checks all grids at once
    grid_paths = [f'{REMOTE_D_PIXEL_BASE_PATH}/{year}/{grid_name}' for grid_name in grid_names]
    
    # Use find to count .npy files in each directory
    cmd = 'for dir in ' + ' '.join([f'"{p}"' for p in grid_paths]) + '; do '
    cmd += 'if [ -d "$dir" ]; then '
    cmd += 'count=$(find "$dir" -maxdepth 1 -name "*.npy" -type f 2>/dev/null | wc -l); '
    cmd += 'echo "$(basename "$dir"):$count"; '
    cmd += 'fi; done'
    
    result = ssh_pool.run_command(cmd)
    
    completed = {}
    if result:
        for line in result.split('\n'):
            if ':' in line:
                grid_name, count = line.strip().split(':', 1)
                completed[grid_name] = int(count) == 9  # 9 npy files means complete d-pixel
    
    return completed

def clear_task_queue(ssh_pool):
    """Clear existing task queue contents"""
    logger.info("Clearing existing task queue...")
    
    # Create directories if they don't exist
    for subdir in ['pending', 'processing', 'done', 'failed']:
        path = f"{REMOTE_TASK_QUEUE_BASE}/{subdir}"
        ssh_pool.run_command(f'mkdir -p {path}')
    
    # Clear all subdirectories
    for subdir in ['pending', 'processing', 'done', 'failed']:
        path = f"{REMOTE_TASK_QUEUE_BASE}/{subdir}"
        
        if subdir == 'processing':
            # For processing, handle potential subdirectories from worker nodes
            cmd = f'find {path} -type f -name "*.task" -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
            cmd = f'find {path} -type d -empty -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
            ssh_pool.run_command(f'mkdir -p {path}')
        else:
            # For other directories, just remove .task files
            cmd = f'find {path} -maxdepth 1 -type f -name "*.task" -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
    
    logger.info("Task queue cleared")

def process_year_with_grid_years(year, grid_names, grid_to_years, otrera_pool, antiope_pool):
    """Process a single year but only for grids that need this year"""
    logger.info(f"Processing year {year}...")
    
    # Filter grids that need this year
    relevant_grids = [grid for grid in grid_names if year in grid_to_years.get(grid, [])]
    
    if not relevant_grids:
        logger.info(f"No grids need processing for year {year}")
        return [], [], 0, 0
    
    # First check which grids have completed representations on antiope
    representations_completed = {}
    batch_size = 50  # Check 50 grids at a time
    
    with tqdm(total=len(relevant_grids), desc=f"Checking representations for year {year}", unit="grids") as pbar:
        for i in range(0, len(relevant_grids), batch_size):
            batch = relevant_grids[i:i+batch_size]
            batch_completed = check_representations_batch(antiope_pool, year, batch)
            representations_completed.update(batch_completed)
            pbar.update(len(batch))
    
    # Filter out grids that already have representations
    grids_needing_d_pixel = [grid for grid in relevant_grids if not representations_completed.get(grid, False)]
    representation_done_count = len(relevant_grids) - len(grids_needing_d_pixel)
    
    if not grids_needing_d_pixel:
        logger.info(f"Year {year}: All {len(relevant_grids)} grids already have representations")
        return [], [], representation_done_count, 0
    
    # Then check which of the remaining grids have completed d-pixels
    d_pixels_completed = {}
    
    with tqdm(total=len(grids_needing_d_pixel), desc=f"Checking d-pixels for year {year}", unit="grids") as pbar:
        for i in range(0, len(grids_needing_d_pixel), batch_size):
            batch = grids_needing_d_pixel[i:i+batch_size]
            batch_completed = check_d_pixels_batch(otrera_pool, year, batch)
            d_pixels_completed.update(batch_completed)
            pbar.update(len(batch))
    
    # Categorize tasks
    pending_tasks = []
    done_tasks = []
    
    for grid_name in grids_needing_d_pixel:
        task_name = f"{grid_name}_{year}.task"
        if d_pixels_completed.get(grid_name, False):
            done_tasks.append(task_name)
        else:
            pending_tasks.append(task_name)
    
    logger.info(f"Year {year}: {representation_done_count} have representations, "
                f"{len(done_tasks)} have d-pixels, {len(pending_tasks)} need processing")
    
    return pending_tasks, done_tasks, representation_done_count, len(done_tasks)

def check_incomplete_d_pixels(ssh_pool, years, grid_names):
    """Check for incomplete d-pixels (between 1-8 npy files)"""
    logger.info("Checking for incomplete d-pixels...")
    
    incomplete = []
    
    for year in years:
        # Use batch checking but track exact counts
        cmd = f"""
        cd {REMOTE_D_PIXEL_BASE_PATH}/{year} 2>/dev/null || exit 0
        for grid_dir in grid_*; do
            if [ -d "$grid_dir" ]; then
                count=$(find "$grid_dir" -maxdepth 1 -name "*.npy" -type f | wc -l)
                if [ $count -gt 0 ] && [ $count -lt 9 ]; then
                    echo "$year|$grid_dir|$count"
                fi
            fi
        done
        """
        
        result = ssh_pool.run_command(cmd)
        if result:
            for line in result.split('\n'):
                if '|' in line:
                    year_str, grid_name, count = line.strip().split('|')
                    incomplete.append((year_str, grid_name, int(count)))
    
    return incomplete

def handle_incomplete_grids(ssh_pool, incomplete_grids):
    """Create failed tasks for incomplete grids"""
    if not incomplete_grids:
        return
    
    logger.info(f"Found {len(incomplete_grids)} incomplete d-pixels")
    
    failed_tasks = []
    for year, grid_name, npy_count in incomplete_grids:
        logger.info(f"  - {grid_name} ({year}): {npy_count}/9 files")
        task_name = f"{grid_name}_{year}.task"
        failed_tasks.append(task_name)
    
    # Create failed tasks
    if failed_tasks:
        logger.info(f"Creating {len(failed_tasks)} failed tasks...")
        batch_create_files(ssh_pool, f"{REMOTE_TASK_QUEUE_BASE}/failed", failed_tasks)

def main():
    parser = argparse.ArgumentParser(description="Initialize the d-pixel generation task queue.")
    parser.add_argument('--dry-run', action='store_true', help="Print tasks without creating them")
    parser.add_argument('--years-normal', nargs='+', type=int, help="Years for normal TIFFs")
    parser.add_argument('--years-express', nargs='+', type=int, help="Years for express TIFFs")
    parser.add_argument('--clean-incomplete', action='store_true', 
                       help="Move incomplete d-pixels to failed queue")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, 
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    # Override year configurations if specified
    global YEARS_TO_PROCESS_NORMAL, YEARS_TO_PROCESS_EXPRESS
    if args.years_normal is not None:
        YEARS_TO_PROCESS_NORMAL = sorted(args.years_normal, reverse=True)
    if args.years_express is not None:
        YEARS_TO_PROCESS_EXPRESS = sorted(args.years_express, reverse=True)
    
    # Collect all unique years
    all_years = sorted(set(YEARS_TO_PROCESS_NORMAL + YEARS_TO_PROCESS_EXPRESS), reverse=True)
    
    start_time = time.time()
    logger.info(f"Starting d-pixel generation task queue initialization")
    logger.info(f"Years for normal TIFFs: {YEARS_TO_PROCESS_NORMAL}")
    logger.info(f"Years for express TIFFs: {YEARS_TO_PROCESS_EXPRESS}")
    
    # Setup SSH connection pools
    try:
        otrera_pool = SSHConnectionPool(OTRERA_HOST)
        antiope_pool = SSHConnectionPool(ANTIOPE_HOST)
    except Exception as e:
        logger.error(f"Failed to establish SSH connections: {e}")
        return
    
    try:
        # Get all grid TIFFs from both sources
        grid_to_source, grid_to_years = get_all_grid_tiffs_with_source(otrera_pool)
        if not grid_to_source:
            logger.error("No grid TIFFs found in any source!")
            return
        
        grid_names = list(grid_to_source.keys())
        
        # Clear existing queue
        if not args.dry_run:
            clear_task_queue(otrera_pool)
        
        # Check for incomplete d-pixels if requested
        if args.clean_incomplete and not args.dry_run:
            incomplete = check_incomplete_d_pixels(otrera_pool, all_years, grid_names)
            handle_incomplete_grids(otrera_pool, incomplete)
        
        # Process each year in parallel
        all_pending = []
        all_done = []
        total_representation_done = 0
        total_d_pixel_done = 0
        
        with ThreadPoolExecutor(max_workers=min(args.workers, len(all_years))) as executor:
            futures = {
                executor.submit(process_year_with_grid_years, year, grid_names, grid_to_years, 
                              otrera_pool, antiope_pool): year 
                for year in all_years
            }
            
            for future in as_completed(futures):
                year = futures[future]
                try:
                    pending_tasks, done_tasks, repr_done, d_pixel_done = future.result()
                    all_pending.extend(pending_tasks)
                    all_done.extend(done_tasks)
                    total_representation_done += repr_done
                    total_d_pixel_done += d_pixel_done
                except Exception as e:
                    logger.error(f"Error processing year {year}: {e}")
        
        # Calculate expected total (sum of years for each grid)
        expected_total = sum(len(years) for years in grid_to_years.values())
        actual_total = len(all_pending) + len(all_done) + total_representation_done
        
        if expected_total != actual_total:
            logger.warning(f"Task count mismatch! Expected: {expected_total}, Actual: {actual_total}")
        
        # Create task files
        if not args.dry_run:
            logger.info("Creating task files...")
            
            # Create pending tasks
            if all_pending:
                logger.info(f"Creating {len(all_pending)} pending tasks...")
                with tqdm(total=len(all_pending), desc="Creating pending tasks") as pbar:
                    for i in range(0, len(all_pending), BATCH_SIZE):
                        batch = all_pending[i:i+BATCH_SIZE]
                        success = batch_create_files(
                            otrera_pool, 
                            f"{REMOTE_TASK_QUEUE_BASE}/pending", 
                            batch
                        )
                        if not success:
                            logger.error("Failed to create some pending tasks")
                        pbar.update(len(batch))
            
            # Create done tasks
            if all_done:
                logger.info(f"Creating {len(all_done)} done tasks...")
                with tqdm(total=len(all_done), desc="Creating done tasks") as pbar:
                    for i in range(0, len(all_done), BATCH_SIZE):
                        batch = all_done[i:i+BATCH_SIZE]
                        success = batch_create_files(
                            otrera_pool, 
                            f"{REMOTE_TASK_QUEUE_BASE}/done", 
                            batch
                        )
                        if not success:
                            logger.error("Failed to create some done tasks")
                        pbar.update(len(batch))
        
        # Summary statistics
        normal_grids = [g for g, src in grid_to_source.items() if src == REMOTE_TIFF_BASE_PATH]
        express_grids = [g for g, src in grid_to_source.items() if src == REMOTE_TIFF_EXPRESS_PATH]
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Normal TIFFs found: {len(normal_grids)}")
        logger.info(f"Express TIFFs found: {len(express_grids)}")
        logger.info(f"Total unique grids: {len(grid_names)}")
        logger.info(f"Years processed: {all_years}")
        logger.info(f"Grids with completed representations: {total_representation_done}")
        logger.info(f"Grids with completed d-pixels: {total_d_pixel_done}")
        logger.info(f"Total pending tasks: {len(all_pending)}")
        logger.info(f"Total tasks created: {len(all_pending) + len(all_done)}")
        logger.info(f"Total grid-year combinations: {expected_total}")
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        
        if args.dry_run:
            logger.info("\nThis was a DRY RUN - no files were created")
        
        # Save summary to JSON for later reference
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "normal_grid_count": len(normal_grids),
            "express_grid_count": len(express_grids),
            "total_unique_grids": len(grid_names),
            "years_normal": YEARS_TO_PROCESS_NORMAL,
            "years_express": YEARS_TO_PROCESS_EXPRESS,
            "representation_done_count": total_representation_done,
            "d_pixel_done_count": total_d_pixel_done,
            "pending_count": len(all_pending),
            "expected_total": expected_total,
            "actual_total": actual_total,
            "execution_time": elapsed_time,
            "dry_run": args.dry_run
        }
        
        with open('d_pixel_initialization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
    finally:
        # Clean up SSH connections
        otrera_pool.close()
        antiope_pool.close()

if __name__ == "__main__":
    main()