#!/usr/bin/env python3
"""
initialize_d_pixel_generation_queue.py

Enhanced version that:
1. Gets all grid TIFFs from Otrera's /tank/zf281/global_0.1_degree_tiff
2. Checks completion status in /tank/zf281/global_0.1_degree_tiff_d_pixel
3. Creates tasks in pending or done folders based on completion status
4. Optimized for speed with connection pooling and batch operations
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
OTRERA_HOST = "zf281@otrera.caelum.ci.dev"

# Paths
REMOTE_TIFF_BASE_PATH = "/tank/zf281/global_0.1_degree_tiff"
REMOTE_D_PIXEL_BASE_PATH = "/tank/zf281/global_0.1_degree_tiff_d_pixel"
REMOTE_TASK_QUEUE_BASE = "/tank/zf281/task_queue/d_pixel_generation"

# Years to process (in priority order - most recent first)
YEARS_TO_PROCESS = list(range(2024, 2016, -1))  # [2024, 2023, ..., 2017]
# YEARS_TO_PROCESS = [2024]

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


def get_all_grid_tiffs(ssh_pool):
    """Get all grid TIFF files from Otrera"""
    logger.info(f"Fetching grid TIFFs from {REMOTE_TIFF_BASE_PATH}...")
    
    cmd = f'find {REMOTE_TIFF_BASE_PATH} -maxdepth 1 -name "grid_*.tiff" -type f -printf "%f\n" | sort'
    result = ssh_pool.run_command(cmd)
    
    if not result:
        logger.error("Failed to fetch grid TIFFs")
        return []
    
    # Extract grid names (remove .tiff extension)
    grid_names = []
    for line in result.split('\n'):
        if line.strip() and line.endswith('.tiff'):
            grid_name = line.strip()[:-5]  # Remove .tiff
            grid_names.append(grid_name)
    
    logger.info(f"Found {len(grid_names)} grid TIFFs")
    return grid_names


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
                completed[grid_name] = int(count) == 9  # 9 npy files means complete
    
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


def process_year(year, grid_names, ssh_pool):
    """Process a single year - returns (pending_tasks, done_tasks)"""
    logger.info(f"Processing year {year}...")
    
    # Check which grids are completed (in batches)
    completed = {}
    batch_size = 50  # Check 50 grids at a time
    
    with tqdm(total=len(grid_names), desc=f"Checking year {year}", unit="grids") as pbar:
        for i in range(0, len(grid_names), batch_size):
            batch = grid_names[i:i+batch_size]
            batch_completed = check_d_pixels_batch(ssh_pool, year, batch)
            completed.update(batch_completed)
            pbar.update(len(batch))
    
    # Categorize tasks
    pending_tasks = []
    done_tasks = []
    
    for grid_name in grid_names:
        task_name = f"{grid_name}_{year}.task"
        if completed.get(grid_name, False):
            done_tasks.append(task_name)
        else:
            pending_tasks.append(task_name)
    
    logger.info(f"Year {year}: {len(pending_tasks)} pending, {len(done_tasks)} done")
    return pending_tasks, done_tasks


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
    parser.add_argument('--years', nargs='+', type=int, help="Specific years to process")
    parser.add_argument('--clean-incomplete', action='store_true', 
                       help="Move incomplete d-pixels to failed queue")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, 
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    if args.years:
        years_to_process = sorted(args.years, reverse=True)  # Most recent first
    else:
        years_to_process = YEARS_TO_PROCESS
    
    start_time = time.time()
    logger.info(f"Starting d-pixel generation task queue initialization")
    logger.info(f"Years to process (in priority order): {years_to_process}")
    
    # Setup SSH connection pool
    try:
        ssh_pool = SSHConnectionPool(OTRERA_HOST)
    except Exception as e:
        logger.error(f"Failed to establish SSH connection: {e}")
        return
    
    try:
        # Get all grid TIFFs from Otrera
        grid_names = get_all_grid_tiffs(ssh_pool)
        if not grid_names:
            logger.error("No grid TIFFs found!")
            return
        
        # Clear existing queue
        if not args.dry_run:
            clear_task_queue(ssh_pool)
        
        # Check for incomplete d-pixels if requested
        if args.clean_incomplete and not args.dry_run:
            incomplete = check_incomplete_d_pixels(ssh_pool, years_to_process, grid_names)
            handle_incomplete_grids(ssh_pool, incomplete)
        
        # Process each year in parallel
        all_pending = []
        all_done = []
        
        with ThreadPoolExecutor(max_workers=min(args.workers, len(years_to_process))) as executor:
            futures = {
                executor.submit(process_year, year, grid_names, ssh_pool): year 
                for year in years_to_process
            }
            
            for future in as_completed(futures):
                year = futures[future]
                try:
                    pending_tasks, done_tasks = future.result()
                    all_pending.extend(pending_tasks)
                    all_done.extend(done_tasks)
                except Exception as e:
                    logger.error(f"Error processing year {year}: {e}")
        
        # Verify total task count
        expected_total = len(grid_names) * len(years_to_process)
        actual_total = len(all_pending) + len(all_done)
        
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
                            ssh_pool, 
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
                            ssh_pool, 
                            f"{REMOTE_TASK_QUEUE_BASE}/done", 
                            batch
                        )
                        if not success:
                            logger.error("Failed to create some done tasks")
                        pbar.update(len(batch))
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Grid TIFFs found: {len(grid_names)}")
        logger.info(f"Years processed: {years_to_process}")
        logger.info(f"Total pending tasks: {len(all_pending)}")
        logger.info(f"Total completed tasks: {len(all_done)}")
        logger.info(f"Total tasks: {len(all_pending) + len(all_done)}")
        logger.info(f"Expected tasks: {expected_total}")
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        
        if args.dry_run:
            logger.info("\nThis was a DRY RUN - no files were created")
        
        # Save summary to JSON for later reference
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grid_count": len(grid_names),
            "years": years_to_process,
            "pending_count": len(all_pending),
            "done_count": len(all_done),
            "expected_total": expected_total,
            "actual_total": actual_total,
            "execution_time": elapsed_time,
            "dry_run": args.dry_run
        }
        
        with open('d_pixel_initialization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
    finally:
        # Clean up SSH connection
        ssh_pool.close()


if __name__ == "__main__":
    main()