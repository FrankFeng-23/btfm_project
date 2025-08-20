#!/usr/bin/env python3
"""
initialize_inference_queue_enhanced.py

Enhanced version that:
1. Clears existing queue content
2. Checks both Otrera (d-pixel) and Antiope (completed representations)
3. Creates tasks in pending or done folders based on completion status
4. Optimized for speed with connection pooling and batch operations
5. Can remove d-pixel folders for completed grids to save storage
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
REMOTE_D_PIXEL_BASE_PATH = "/tank/zf281/global_0.1_degree_tiff_d_pixel"
REMOTE_REPRESENTATION_BASE_PATH = "/tank/zf281/global_0.1_degree_representation"
REMOTE_TASK_QUEUE_BASE = "/tank/zf281/task_queue/representation_inference"

YEARS_TO_PROCESS = list(range(2017, 2025))

# Performance settings
MAX_WORKERS = 20  # Number of parallel threads
BATCH_SIZE = 100  # Number of tasks to create in a single SSH command

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('initialize_queue.log'),
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
        # Use printf to create multiple files at once
        file_list = ' '.join([f'"{remote_dir}/{f}"' for f in batch])
        cmd = f'cd {remote_dir} && touch {" ".join(batch)}'
        result = ssh_pool.run_command(cmd)
        if result is None:
            return False
    return True

def get_all_grids_for_year(ssh_pool, year, base_path):
    """Get all grid directories for a specific year"""
    cmd = f'find {base_path}/{year} -maxdepth 1 -type d -name "grid_*" -printf "%f\n" 2>/dev/null | sort'
    result = ssh_pool.run_command(cmd)
    if result:
        return [g for g in result.split('\n') if g]
    return []

def check_representations_batch(ssh_pool, year, grid_ids, base_path):
    """Check multiple grids for completed representations in one command"""
    if not grid_ids:
        return {}
    
    # Build a find command that checks all grids at once
    grid_paths = [f'{base_path}/{year}/{grid_id}' for grid_id in grid_ids]
    
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
                grid_id, count = line.strip().split(':', 1)
                completed[grid_id] = int(count) >= 2
    
    return completed

def clear_task_queue(ssh_pool):
    """Clear existing task queue contents"""
    logger.info("Clearing existing task queue...")
    
    # Clear all subdirectories recursively
    for subdir in ['pending', 'processing', 'done', 'failed']:
        path = f"{REMOTE_TASK_QUEUE_BASE}/{subdir}"
        
        # Use find to recursively delete all .task files
        # Also remove empty subdirectories in processing folder
        if subdir == 'processing':
            # First remove all .task files recursively
            cmd = f'find {path} -type f -name "*.task" -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
            
            # Then remove empty directories (bottom-up)
            cmd = f'find {path} -type d -empty -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
            # 以上操作会删除processing这个目录，需要重建
            ssh_pool.run_command(f'mkdir -p {path}')
        else:
            # For other directories, just remove .task files in the main directory
            cmd = f'find {path} -maxdepth 1 -type f -name "*.task" -delete 2>/dev/null || true'
            ssh_pool.run_command(cmd)
    
    logger.info("Task queue cleared")

def remove_d_pixel_folders(otrera_pool, year, grid_ids, dry_run=True):
    """Remove d-pixel folders for completed grids"""
    if not grid_ids:
        return 0
    
    removed_count = 0
    
    if dry_run:
        logger.info(f"[DRY RUN] Would remove {len(grid_ids)} d-pixel folders for year {year}:")
        for grid_id in grid_ids:
            path = f"{REMOTE_D_PIXEL_BASE_PATH}/{year}/{grid_id}"
            logger.info(f"[DRY RUN] Would remove: {path}")
        removed_count = len(grid_ids)
    else:
        # Remove folders in batches for efficiency
        for i in range(0, len(grid_ids), 10):  # Process 10 at a time
            batch = grid_ids[i:i+10]
            paths = [f"{REMOTE_D_PIXEL_BASE_PATH}/{year}/{grid_id}" for grid_id in batch]
            
            # Use rm -rf to remove directories
            cmd = 'for dir in ' + ' '.join([f'"{p}"' for p in paths]) + '; do '
            cmd += 'if [ -d "$dir" ]; then rm -rf "$dir" && echo "Removed: $dir"; fi; done'
            
            result = otrera_pool.run_command(cmd)
            if result:
                removed_count += len([line for line in result.split('\n') if line.startswith('Removed:')])
    
    return removed_count

def process_year(year, otrera_pool, antiope_pool, dry_remove=False):
    """Process a single year - returns (pending_tasks, done_tasks, grids_to_remove)"""
    logger.info(f"Processing year {year}...")
    
    # Get all grids from Otrera (d-pixel source)
    grids = get_all_grids_for_year(otrera_pool, year, REMOTE_D_PIXEL_BASE_PATH)
    
    if not grids:
        logger.warning(f"No grids found for year {year}")
        return [], [], []
    
    logger.info(f"Found {len(grids)} grids for year {year}")
    
    # Check which grids are completed on Antiope (in batches)
    completed = {}
    batch_size = 50  # Check 50 grids at a time
    
    with tqdm(total=len(grids), desc=f"Checking year {year}", unit="grids") as pbar:
        for i in range(0, len(grids), batch_size):
            batch = grids[i:i+batch_size]
            batch_completed = check_representations_batch(antiope_pool, year, batch, REMOTE_REPRESENTATION_BASE_PATH)
            completed.update(batch_completed)
            pbar.update(len(batch))
    
    # Categorize tasks
    pending_tasks = []
    done_tasks = []
    grids_to_remove = []
    
    for grid_id in grids:
        task_name = f"{grid_id}_{year}.task"
        if completed.get(grid_id, False):
            done_tasks.append(task_name)
            grids_to_remove.append(grid_id)
        else:
            pending_tasks.append(task_name)
    
    logger.info(f"Year {year}: {len(pending_tasks)} pending, {len(done_tasks)} done, {len(grids_to_remove)} d-pixel folders can be removed")
    return pending_tasks, done_tasks, grids_to_remove

def main():
    parser = argparse.ArgumentParser(description="Initialize the inference task queue with optimization.")
    parser.add_argument('--dry-run', action='store_true', help="Print tasks without creating them")
    parser.add_argument('--dry-remove', action='store_true', help="Print d-pixel folders to remove without actually removing them")
    parser.add_argument('--remove-d-pixel', default=True, help="Actually remove d-pixel folders for completed grids")
    parser.add_argument('--years', nargs='+', type=int, help="Specific years to process")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()
    
    if args.years:
        years_to_process = args.years
    else:
        years_to_process = YEARS_TO_PROCESS
    
    start_time = time.time()
    logger.info(f"Starting enhanced task queue initialization for years: {years_to_process}")
    
    if args.dry_remove:
        logger.info("DRY REMOVE MODE: Will only print d-pixel folders that would be removed")
    elif args.remove_d_pixel:
        logger.info("REMOVE MODE: Will actually remove d-pixel folders for completed grids")
    
    # Setup SSH connection pools
    try:
        otrera_pool = SSHConnectionPool(OTRERA_HOST)
        antiope_pool = SSHConnectionPool(ANTIOPE_HOST)
    except Exception as e:
        logger.error(f"Failed to establish SSH connections: {e}")
        return
    
    try:
        # Clear existing queue
        if not args.dry_run:
            clear_task_queue(otrera_pool)
        
        # Process each year in parallel
        all_pending = []
        all_done = []
        removal_stats = {}  # year -> number of grids to remove
        
        with ThreadPoolExecutor(max_workers=min(args.workers, len(years_to_process))) as executor:
            futures = {
                executor.submit(process_year, year, otrera_pool, antiope_pool, args.dry_remove): year 
                for year in years_to_process
            }
            
            for future in as_completed(futures):
                year = futures[future]
                try:
                    pending_tasks, done_tasks, grids_to_remove = future.result()
                    all_pending.extend(pending_tasks)
                    all_done.extend(done_tasks)
                    
                    # Handle d-pixel removal
                    if grids_to_remove and (args.dry_remove or args.remove_d_pixel):
                        removed_count = remove_d_pixel_folders(
                            otrera_pool, 
                            year, 
                            grids_to_remove, 
                            dry_run=args.dry_remove or not args.remove_d_pixel
                        )
                        removal_stats[year] = len(grids_to_remove)
                    else:
                        removal_stats[year] = len(grids_to_remove)
                        
                except Exception as e:
                    logger.error(f"Error processing year {year}: {e}")
                    removal_stats[year] = 0
        
        # Create task files
        if not args.dry_run:
            logger.info("Creating task files...")
            
            # Create pending tasks
            if all_pending:
                logger.info(f"Creating {len(all_pending)} pending tasks...")
                success = batch_create_files(
                    otrera_pool, 
                    f"{REMOTE_TASK_QUEUE_BASE}/pending", 
                    all_pending
                )
                if not success:
                    logger.error("Failed to create some pending tasks")
            
            # Create done tasks
            if all_done:
                logger.info(f"Creating {len(all_done)} done tasks...")
                success = batch_create_files(
                    otrera_pool, 
                    f"{REMOTE_TASK_QUEUE_BASE}/done", 
                    all_done
                )
                if not success:
                    logger.error("Failed to create some done tasks")
        
        # Calculate total removals
        total_removals = sum(removal_stats.values())
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Years processed: {years_to_process}")
        logger.info(f"Total pending tasks: {len(all_pending)}")
        logger.info(f"Total completed tasks: {len(all_done)}")
        logger.info(f"Total tasks: {len(all_pending) + len(all_done)}")
        
        # D-pixel removal summary
        logger.info("\nD-PIXEL REMOVAL SUMMARY:")
        logger.info("-" * 40)
        for year in sorted(removal_stats.keys()):
            logger.info(f"Year {year}: {removal_stats[year]} grids can be removed")
        logger.info(f"Total d-pixel folders that can be removed: {total_removals}")
        
        if args.dry_remove:
            logger.info("\n[DRY REMOVE] No d-pixel folders were actually removed")
        elif args.remove_d_pixel:
            logger.info(f"\n[REMOVED] {total_removals} d-pixel folders were removed")
        else:
            logger.info("\nTo remove d-pixel folders, use --dry-remove to preview or --remove-d-pixel to actually remove")
        
        logger.info(f"\nExecution time: {elapsed_time:.2f} seconds")
        
        if args.dry_run:
            logger.info("\nThis was a DRY RUN - no files were created")
        
        # Save summary to JSON for later reference
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "years": years_to_process,
            "pending_count": len(all_pending),
            "done_count": len(all_done),
            "d_pixel_removal_stats": removal_stats,
            "total_removals": total_removals,
            "execution_time": elapsed_time,
            "dry_run": args.dry_run,
            "dry_remove": args.dry_remove,
            "remove_d_pixel": args.remove_d_pixel
        }
        
        with open('initialization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
    finally:
        # Clean up SSH connections
        otrera_pool.close()
        antiope_pool.close()

if __name__ == "__main__":
    main()