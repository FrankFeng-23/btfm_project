#!/bin/bash -l

#SBATCH --job-name=global-dpixel-gen
#SBATCH --partition=sapphire
#SBATCH --account=AIRR-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/dpixel_gen_%A_%a_%N.out
#SBATCH --error=slurm_logs/dpixel_gen_%A_%a_%N.err

set -u

# Python environment
# source /home/azureuser/miniconda3/bin/activate d-pixel-generation
export PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"

# --- Task Queue Directories ---
TASK_QUEUE_BASE="/tank/zf281/task_queue"
DPIXEL_PENDING_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/pending"
DPIXEL_PROCESSING_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/processing"
DPIXEL_DONE_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/done"
DPIXEL_FAILED_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/failed"

# Base directories
LOCAL_TIFF_DIR="/home/azureuser/data/uk_tiff"
LOCAL_BASE_OUTPUT_DIR="/home/azureuser/data/uk_d_pixel"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remote hosts and paths
OTRERA_HOST="zf281@otrera.caelum.ci.dev"
OTRERA_D_PIXEL_PATH="/tank/zf281/global_0.1_degree_tiff_d_pixel"

# Worker configuration
PROCESSES_PER_NODE=1
LOG_INTERVAL=60

# Get node-specific information
NODE_ID="${SLURM_NODEID:-0}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-$$}"
UNIQUE_JOB_ID="${NODE_NAME}_${JOB_ID}"

# Create local directories for this job run
NODE_LOG_DIR="logs/dpixel_job_${JOB_ID}"
mkdir -p "$NODE_LOG_DIR"
mkdir -p "$LOCAL_BASE_OUTPUT_DIR"
mkdir -p slurm_logs

# Record script start time
SCRIPT_START=$(date +%s)

# Terminal colors
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
CYAN="\033[36m"
RESET="\033[0m"

# Logging functions
log_header() { echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2; }
log_info() { echo -e "${CYAN}[INFO][${NODE_NAME}]${RESET} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS][${NODE_NAME}]${RESET} $1" >&2; }
log_warning() { echo -e "${YELLOW}[WARNING][${NODE_NAME}]${RESET} $1" >&2; }
log_error() { echo -e "${RED}[ERROR][${NODE_NAME}]${RESET} $1" >&2; }

format_time() {
    local seconds=$1; local hours=$((seconds / 3600)); local minutes=$(((seconds % 3600) / 60)); local secs=$((seconds % 60))
    if [[ $hours -gt 0 ]]; then printf "%02dh:%02dm:%02ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then printf "%02dm:%02ds" $minutes $secs
    else printf "%02ds" $secs; fi
}

# Function to kill process and all its children
kill_process_tree() {
    local pid=$1
    local children=$(pgrep -P $pid 2>/dev/null || true)
    for child in $children; do kill_process_tree $child; done
    if ps -p $pid > /dev/null 2>&1; then
        kill -9 $pid 2>/dev/null || true
    fi
}

# Arrays to store process PIDs
declare -a WORKER_PIDS
declare -a WORKER_LOG_FILES

# Function to clean up processes on exit
cleanup() {
    log_warning "Received interrupt signal. Cleaning up..."

    # Return stranded tasks to pending queue
    log_warning "Returning in-progress tasks to pending queue..."
    ssh "$OTRERA_HOST" "
        cd ${DPIXEL_PROCESSING_DIR}
        for f in ${UNIQUE_JOB_ID}_*.task; do
            if [ -f \"\$f\" ]; then
                base_name=\$(echo \"\$f\" | sed 's/^${UNIQUE_JOB_ID}_worker[0-9]*_//')
                mv \"\$f\" \"${DPIXEL_PENDING_DIR}/\$base_name\"
            fi
        done
    " 2>/dev/null || true

    # Kill all child worker processes
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            kill_process_tree $pid
        fi
    done

    log_warning "All processes terminated. Exiting."
    exit 1
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

log_header "GLOBAL D-PIXEL GENERATION (File Queue Pipeline)"
log_info "Starting at $(date)"
log_info "Node: ${NODE_NAME}, Job ID: ${JOB_ID}"

log_header "SETUP & PRE-FLIGHT CHECKS"

# Verify required files exist
log_info "Checking required files..."
missing_files=()
[[ ! -f "${SCRIPT_SOURCE_DIR}/s1_s2_downloader.sh" ]] && missing_files+=("s1_s2_downloader.sh")
[[ ! -f "${SCRIPT_SOURCE_DIR}/s1_s2_stacker.sh" ]] && missing_files+=("s1_s2_stacker.sh")
[[ ! -f "${SCRIPT_SOURCE_DIR}/s1_fast_processor.py" ]] && missing_files+=("s1_fast_processor.py")
[[ ! -f "${SCRIPT_SOURCE_DIR}/s2_fast_processor.py" ]] && missing_files+=("s2_fast_processor.py")
[[ ! -f "${SCRIPT_SOURCE_DIR}/s1_stack" ]] && missing_files+=("s1_stack")
[[ ! -f "${SCRIPT_SOURCE_DIR}/s2_stack" ]] && missing_files+=("s2_stack")

if [[ ${#missing_files[@]} -gt 0 ]]; then
    log_error "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi
log_success "All required files present"

# Check Python environment
log_info "Checking Python environment..."
if [[ ! -x "$PYTHON_ENV" ]]; then
    log_error "Python environment not found"
    exit 1
fi
log_success "Python environment ready"

# Test SSH connection
log_info "Testing SSH connection to Otrera..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${OTRERA_HOST}" "echo 'OK'" &>/dev/null; then
    log_error "Cannot connect to Otrera"
    exit 1
fi
log_success "SSH connection established"

# Clean up stale tasks (older than 1 hour)
log_header "CHECKING FOR STALE TASKS"
log_info "Moving stale tasks (>1 hour) back to pending..."
STALE_COUNT=$(ssh "$OTRERA_HOST" "
    cd ${DPIXEL_PROCESSING_DIR} 2>/dev/null || exit
    current_time=\$(date +%s)
    stale_count=0
    for task_file in *.task; do
        if [ -f \"\$task_file\" ]; then
            file_time=\$(stat -c %Y \"\$task_file\" 2>/dev/null || echo 0)
            age=\$((current_time - file_time))
            if [ \$age -gt 3600 ]; then
                # Extract original task name by removing worker prefix
                base_name=\$(echo \"\$task_file\" | sed -E 's/^[^_]+_[^_]+_worker[0-9]+_//')
                mv \"\$task_file\" \"${DPIXEL_PENDING_DIR}/\$base_name\" 2>/dev/null && stale_count=\$((stale_count + 1))
            fi
        fi
    done
    echo \$stale_count
" 2>/dev/null || echo "0")
log_info "Moved $STALE_COUNT stale tasks back to pending"

# Function to start a worker process
start_worker() {
    local worker_id=$1
    local worker_name="${UNIQUE_JOB_ID}_worker${worker_id}"
    
    log_info "Starting worker: ${worker_name}"
    
    local log_file="${NODE_LOG_DIR}/worker_${worker_id}.log"
    
    # The Python worker script
    $PYTHON_ENV -u - > "$log_file" 2>&1 <<'EOF' &
import os
import sys
import time
import random
import subprocess
import shutil
import traceback
import logging
import json
from pathlib import Path
from datetime import datetime

# Configuration from environment
WORKER_NAME = os.environ['WORKER_NAME']
WORKER_ID = os.environ['WORKER_ID']
NODE_NAME = os.environ['NODE_NAME']

OTRERA_HOST = os.environ['OTRERA_HOST']
DPIXEL_PENDING_DIR = os.environ['DPIXEL_PENDING_DIR']
DPIXEL_PROCESSING_DIR = os.environ['DPIXEL_PROCESSING_DIR']
DPIXEL_DONE_DIR = os.environ['DPIXEL_DONE_DIR']
DPIXEL_FAILED_DIR = os.environ['DPIXEL_FAILED_DIR']

LOCAL_TIFF_DIR = Path(os.environ['LOCAL_TIFF_DIR'])
LOCAL_BASE_OUTPUT_DIR = Path(os.environ['LOCAL_BASE_OUTPUT_DIR'])
SCRIPT_SOURCE_DIR = Path(os.environ['SCRIPT_SOURCE_DIR'])

OTRERA_D_PIXEL_PATH = os.environ['OTRERA_D_PIXEL_PATH']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - [{WORKER_NAME}] - %(message)s',
    stream=sys.stdout,
    force=True
)

def run_command(cmd, check=True):
    logging.debug(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(f"Command failed: {cmd}\nSTDERR: {result.stderr}")
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def run_ssh_command(host, cmd, check=True):
    full_cmd = f"ssh -o ConnectTimeout=10 {host} '{cmd}'"
    return run_command(full_cmd, check)

def get_task():
    """Atomically get a task from the pending queue."""
    try:
        # List pending tasks
        stdout, _, _ = run_ssh_command(OTRERA_HOST, f"ls -1 {DPIXEL_PENDING_DIR} | grep '\\.task$'", check=False)
        if not stdout:
            return None
        
        pending_files = [f for f in stdout.split('\n') if f.strip()]
        if not pending_files:
            return None
        
        # Prioritize by year (newer first) - tasks are named like grid_x_y_YEAR.task
        def get_year(filename):
            try:
                return int(filename.replace('.task', '').split('_')[-1])
            except:
                return 0
        
        pending_files.sort(key=get_year, reverse=True)
        
        # Try to acquire a task
        for task_basename in pending_files[:5]:  # Try up to 5 tasks
            task_processing_name = f"{WORKER_NAME}_{task_basename}"
            
            # Atomic move
            mv_cmd = f"mv {DPIXEL_PENDING_DIR}/{task_basename} {DPIXEL_PROCESSING_DIR}/{task_processing_name} 2>/dev/null"
            _, _, returncode = run_ssh_command(OTRERA_HOST, mv_cmd, check=False)
            
            if returncode == 0:
                logging.info(f"Acquired task: {task_basename}")
                return task_processing_name
        
        return None
        
    except Exception as e:
        logging.error(f"Error acquiring task: {e}")
        return None

def async_transfer(tile_name, year, local_dir, remote_dir):
    """Start asynchronous transfer to Otrera."""
    transfer_script = local_dir / "transfer_to_otrera.sh"
    
    script_content = f"""#!/bin/bash
# Create remote directory
ssh -o ConnectTimeout=30 "{OTRERA_HOST}" "mkdir -p '{remote_dir}'" 2>/dev/null

# Transfer NPY files
if rsync -azP "{local_dir}/data_processed/"*.npy "{OTRERA_HOST}:{remote_dir}/" 2>/dev/null; then
    # Clean up local files after successful transfer
    rm -rf "{local_dir}/data_raw"
    rm -rf "{local_dir}/data_sar_raw"
    rm -rf "{local_dir}/data_processed"
    rm -f "{local_dir}/downloader_temp_s1.sh"
    rm -f "{local_dir}/downloader_temp_s2.sh"
    rm -f "{local_dir}/stacker_temp.sh"
    echo "Transfer completed for {tile_name} ({year})"
else
    echo "Transfer failed for {tile_name} ({year})"
fi

# Self-delete
rm -f "{transfer_script}"
"""
    
    with open(transfer_script, 'w') as f:
        f.write(script_content)
    
    os.chmod(transfer_script, 0o755)
    
    # Start transfer in background
    subprocess.Popen(['nohup', str(transfer_script)], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp)
    
    logging.info(f"Started background transfer for {tile_name} ({year})")

def process_task(task_name):
    """Process a single d-pixel generation task."""
    # Extract grid_id and year from task name
    task_basename = task_name.replace(f"{WORKER_NAME}_", "")
    parts = task_basename.replace('.task', '').split('_')
    year = parts[-1]
    grid_id = '_'.join(parts[:-1])
    
    logging.info(f"Processing task: grid={grid_id}, year={year}")
    
    # Setup paths
    tiff_file = LOCAL_TIFF_DIR / f"{grid_id}.tiff"
    local_dir = LOCAL_BASE_OUTPUT_DIR / year / grid_id
    remote_dir = f"{OTRERA_D_PIXEL_PATH}/{year}/{grid_id}"
    
    # Check if TIFF exists
    if not tiff_file.exists():
        raise Exception(f"TIFF file not found: {tiff_file}")
    
    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processing script
    process_script = local_dir / "process_dpixel.sh"
    with open(process_script, 'w') as f:
        f.write(f"""#!/bin/bash
set -e

export TILE_NAME="{grid_id}"
export TIFF_FILE="{tiff_file}"
export LOCAL_DIR="{local_dir}"
export YEAR="{year}"
export SCRIPT_SOURCE_DIR="{SCRIPT_SOURCE_DIR}"
export PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"

cd "$SCRIPT_SOURCE_DIR"

# Step 1: Download Sentinel-1
cp s1_s2_downloader.sh "$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|INPUT_TIFF=.*|INPUT_TIFF=\\\"$TIFF_FILE\\\"|" "$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|OUT_DIR=.*|OUT_DIR=\\\"$LOCAL_DIR\\\"|" "$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|S1_ENABLED=false|S1_ENABLED=true|" "$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|S2_ENABLED=true|S2_ENABLED=false|" "$LOCAL_DIR/downloader_temp_s1.sh"

bash "$LOCAL_DIR/downloader_temp_s1.sh"

# Step 2: Download Sentinel-2
cp s1_s2_downloader.sh "$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|INPUT_TIFF=.*|INPUT_TIFF=\\\"$TIFF_FILE\\\"|" "$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|OUT_DIR=.*|OUT_DIR=\\\"$LOCAL_DIR\\\"|" "$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|S1_ENABLED=true|S1_ENABLED=false|" "$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|S2_ENABLED=false|S2_ENABLED=true|" "$LOCAL_DIR/downloader_temp_s2.sh"

bash "$LOCAL_DIR/downloader_temp_s2.sh"

# Step 3: Stack the data
cp "$SCRIPT_SOURCE_DIR/s1_s2_stacker.sh" "$LOCAL_DIR/stacker_temp.sh"
sed -i "s|BASE_DIR=.*|BASE_DIR=\\\"$LOCAL_DIR\\\"|" "$LOCAL_DIR/stacker_temp.sh"

cp "$SCRIPT_SOURCE_DIR/s1_stack" "$LOCAL_DIR/"
cp "$SCRIPT_SOURCE_DIR/s2_stack" "$LOCAL_DIR/"
chmod +x "$LOCAL_DIR/s1_stack" "$LOCAL_DIR/s2_stack"

cd "$LOCAL_DIR"
bash "./stacker_temp.sh"

# Verify output
npy_count=$(find "$LOCAL_DIR/data_processed" -name "*.npy" -type f 2>/dev/null | wc -l)
if [ "$npy_count" -ne 9 ]; then
    echo "ERROR: Expected 9 NPY files but found $npy_count"
    exit 1
fi

echo "Processing completed successfully"
""")
    
    os.chmod(process_script, 0o755)
    
    # Run the processing script
    stdout, stderr, returncode = run_command(str(process_script), check=False)
    
    if returncode != 0:
        logging.error(f"Processing failed: {stderr}")
        raise Exception(f"Processing script failed with code {returncode}")
    
    # Start async transfer
    async_transfer(grid_id, year, local_dir, remote_dir)
    
    return task_basename

def main():
    logging.info("Worker started")
    processed_count = 0
    consecutive_empty_checks = 0
    
    while True:
        task_name = get_task()
        
        if task_name:
            consecutive_empty_checks = 0
            try:
                start_time = time.time()
                task_basename = process_task(task_name)
                processing_time = time.time() - start_time
                
                # Move to done queue
                mv_cmd = f"mv {DPIXEL_PROCESSING_DIR}/{task_name} {DPIXEL_DONE_DIR}/{task_basename}"
                run_ssh_command(OTRERA_HOST, mv_cmd)
                
                logging.info(f"Successfully completed {task_basename} in {processing_time:.1f}s")
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Failed to process {task_name}: {str(e)}")
                logging.error(traceback.format_exc())
                
                # Move to failed queue
                try:
                    task_basename = task_name.replace(f"{WORKER_NAME}_", "")
                    mv_cmd = f"mv {DPIXEL_PROCESSING_DIR}/{task_name} {DPIXEL_FAILED_DIR}/{task_basename}"
                    run_ssh_command(OTRERA_HOST, mv_cmd)
                except:
                    logging.error("Failed to move task to failed queue")
        else:
            consecutive_empty_checks += 1
            if consecutive_empty_checks > 10:
                logging.info(f"No tasks for 10 minutes. Exiting after {processed_count} tasks.")
                break
            else:
                logging.debug("No tasks available. Sleeping...")
                time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Worker interrupted")
    except Exception as e:
        logging.error(f"Worker crashed: {e}")
        logging.error(traceback.format_exc())
EOF
    
    # Set environment variables for the worker
    export WORKER_NAME="$worker_name"
    export WORKER_ID="$worker_id"
    export NODE_NAME="$NODE_NAME"
    export OTRERA_HOST="$OTRERA_HOST"
    export DPIXEL_PENDING_DIR="$DPIXEL_PENDING_DIR"
    export DPIXEL_PROCESSING_DIR="$DPIXEL_PROCESSING_DIR"
    export DPIXEL_DONE_DIR="$DPIXEL_DONE_DIR"
    export DPIXEL_FAILED_DIR="$DPIXEL_FAILED_DIR"
    export LOCAL_TIFF_DIR="$LOCAL_TIFF_DIR"
    export LOCAL_BASE_OUTPUT_DIR="$LOCAL_BASE_OUTPUT_DIR"
    export SCRIPT_SOURCE_DIR="$SCRIPT_SOURCE_DIR"
    export OTRERA_D_PIXEL_PATH="$OTRERA_D_PIXEL_PATH"
    
    local pid=$!
    WORKER_PIDS+=($pid)
    WORKER_LOG_FILES+=($log_file)
    
    log_success "Started worker ${worker_id} with PID $pid"
}

log_header "STARTING D-PIXEL GENERATION WORKERS"
log_info "Starting $PROCESSES_PER_NODE worker processes"

# Start worker processes
for ((worker_id=0; worker_id<PROCESSES_PER_NODE; worker_id++)); do
    start_worker $worker_id
    sleep 2
done

log_header "MONITORING PROGRESS"
log_info "All workers launched. Logs in ${NODE_LOG_DIR}/"

# Monitor workers
while true; do
    RUNNING_WORKERS=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    if [ $RUNNING_WORKERS -eq 0 ]; then
        log_warning "All workers have stopped."
        break
    fi
    
    # Get queue status
    PENDING_COUNT=$(ssh "$OTRERA_HOST" "ls -1 ${DPIXEL_PENDING_DIR} 2>/dev/null | grep '\\.task$' | wc -l" 2>/dev/null || echo "0")
    PROCESSING_COUNT=$(ssh "$OTRERA_HOST" "ls -1 ${DPIXEL_PROCESSING_DIR} 2>/dev/null | grep '\\.task$' | wc -l" 2>/dev/null || echo "0")
    DONE_COUNT=$(ssh "$OTRERA_HOST" "ls -1 ${DPIXEL_DONE_DIR} 2>/dev/null | grep '\\.task$' | wc -l" 2>/dev/null || echo "0")
    FAILED_COUNT=$(ssh "$OTRERA_HOST" "ls -1 ${DPIXEL_FAILED_DIR} 2>/dev/null | grep '\\.task$' | wc -l" 2>/dev/null || echo "0")
    
    ELAPSED=$(($(date +%s) - SCRIPT_START))
    
    log_info "Workers: $RUNNING_WORKERS/$PROCESSES_PER_NODE | Queue - Pending: $PENDING_COUNT, Processing: $PROCESSING_COUNT, Done: $DONE_COUNT, Failed: $FAILED_COUNT | Time: $(format_time $ELAPSED)"
    
    sleep $LOG_INTERVAL
done

log_header "D-PIXEL GENERATION COMPLETE"
TOTAL_TIME=$(($(date +%s) - SCRIPT_START))
log_info "Total execution time: $(format_time $TOTAL_TIME)"
log_success "Finished at $(date)"
log_info "Note: Background transfers to Otrera may still be running"