#!/bin/bash -l

#SBATCH --job-name=global-infer-xpu
#SBATCH --partition=pvc9
#SBATCH --account=AIRR-P3-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/global_infer_%A_%a_%N.out
#SBATCH --error=slurm_logs/global_infer_%A_%a_%N.err

######################## DAWN ENVIRONMENT SETUP #########################
module purge
module load default-dawn
module load dawn-env/2024-12-29
module load intel-oneapi-mpi
module load oneapi-level-zero
source /usr/local/dawn/software/external/intel-oneapi/2024.0/setvars.sh

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
#########################################################################

###############This needs to be modified to your environment###############
# Python environment that has the required packages installed
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
export PYTHON_ENV="python"

# Base directories
LOCAL_D_PIXEL_DIR="/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/global_d_pixel"
LOCAL_REPRESENTATION_DIR="/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/global_representation"
REMOTE_CSV_PATH="/tank/zf281/global_representation_stats.csv"
OTRERA_HOST="zf281@otrera.caelum.ci.dev"
ANTIOPE_HOST="zf281@antiope.caelum.ci.dev"
OTRERA_D_PIXEL_PATH="/tank/zf281/global_0.1_degree_tiff_d_pixel"
ANTIOPE_REPRESENTATION_PATH="/home/zf281/global_0.1_degree_representation"

# Path to the CSV management script
CSV_SCRIPT="src/global_representation_stats.py"

# Path to the inference script
INFERENCE_SCRIPT="src/multi_tile_infer_QAT.py"

# Path to the checkpoint file for the model
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250608_220648_QAT.pt"

# Config file
CONFIG_FILE="configs/multi_tile_infer_config.py"

# XPU Configuration
PROCESSES_PER_XPU=4  # Number of concurrent processes per XPU
XPU_BATCH_SIZE=4096  # Batch size for XPU inference
XPU_NUM_WORKERS=2    # Workers for data loading
VERBOSE_XPU=true     # Enable detailed XPU logging
LOG_INTERVAL=5       # Log every N batches

# Years to process
YEARS=(2024 2023 2022 2021 2020 2019 2018 2017)
###########################################################################

# Get node-specific information
NODE_ID="${SLURM_NODEID:-0}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-$$}"

# Create node-specific directories
NODE_LOG_DIR="logs/node_${NODE_NAME}_job_${JOB_ID}"
NODE_UPLOAD_QUEUE="upload_queue/node_${NODE_NAME}_job_${JOB_ID}"

# Record script start time
SCRIPT_START=$(date +%s)

# Terminal colors for beautiful logging
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
RESET="\033[0m"

# Logging functions
log_header() {
    echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2
}

log_info() {
    echo -e "${CYAN}[INFO][${NODE_NAME}]${RESET} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS][${NODE_NAME}]${RESET} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING][${NODE_NAME}]${RESET} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR][${NODE_NAME}]${RESET} $1" >&2
}

log_debug() {
    echo -e "${MAGENTA}[DEBUG][${NODE_NAME}]${RESET} $1" >&2
}

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [[ $hours -gt 0 ]]; then
        printf "%02dh:%02dm:%02ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%02dm:%02ds" $minutes $secs
    else
        printf "%02ds" $secs
    fi
}

# Function to check XPU availability
check_xpu_availability() {
    # Try to get XPU count using PyTorch with Intel Extension
   XPU_COUNT=$($PYTHON_ENV -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0)" 2>/dev/null)
    
    # If there's an error or return is not a number, assume 0
    if ! [[ "$XPU_COUNT" =~ ^[0-9]+$ ]]; then
        XPU_COUNT=0
    fi
    
    echo $XPU_COUNT
}

# Arrays to store process PIDs and their log files
declare -a XPU_PIDS
declare -a XPU_LOG_FILES
declare -a UPLOAD_PIDS

# Function to kill process and all its children
kill_process_tree() {
    local pid=$1
    log_debug "Killing process tree for PID: $pid"
    
    # Get all child processes
    local children=$(pgrep -P $pid 2>/dev/null || true)
    
    # Kill children first (recursively)
    for child in $children; do
        kill_process_tree $child
    done
    
    # Kill the process
    if ps -p $pid > /dev/null 2>&1; then
        log_debug "Terminating process $pid"
        kill -9 $pid 2>/dev/null || true
    fi
}

# Function to clean up processes on exit (MODIFIED)
cleanup() {
    log_warning "Received interrupt signal. Cleaning up processes..."
    
    # Kill XPU worker processes
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            kill_process_tree $pid
        fi
    done
    
    # Kill upload processes
    for pid in "${UPLOAD_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            kill_process_tree $pid
        fi
    done
    
    # Attempt to clean the remote lock directory on exit, just in case.
    # This is a critical best-effort attempt to prevent deadlocks on the next run.
    log_warning "Attempting to release remote lock as a precaution..."
    local remote_lock_dir="/tank/zf281/global_stats.csv.lockdir"
    ssh "$OTRERA_HOST" "rm -rf ${remote_lock_dir}" || log_warning "Could not connect to Otrera to release lock directory."

    log_warning "All processes terminated. Exiting."
    exit 1
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

log_header "GLOBAL REPRESENTATION INFERENCE (Intel XPU)"
log_info "Starting at $(date)"
log_info "Node: ${NODE_NAME}, Node ID: ${NODE_ID}, Job ID: ${JOB_ID}"

# Check environment
if [ -n "$SLURM_JOB_ID" ]; then
    log_info "Running as SLURM job $SLURM_JOB_ID on node $HOSTNAME"
else
    log_info "Running in interactive mode on $HOSTNAME"
fi

log_header "ENVIRONMENT CHECK"

# Create necessary directories with node-specific paths
mkdir -p "$LOCAL_D_PIXEL_DIR"
mkdir -p "$LOCAL_REPRESENTATION_DIR"
mkdir -p "$NODE_LOG_DIR"
mkdir -p "$NODE_UPLOAD_QUEUE"
log_success "Created necessary directories"

# Check XPU availability
XPU_COUNT=$(check_xpu_availability)
if [ $XPU_COUNT -eq 0 ]; then
    log_error "No Intel XPUs detected! This script requires Intel GPUs."
    exit 1
fi
log_success "Detected $XPU_COUNT Intel XPUs"

# Print XPU information
log_info "XPU information:"
$PYTHON_ENV -c "
import torch
import intel_extension_for_pytorch as ipex

if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'PyTorch version: {torch.__version__}')
    print(f'IPEX version: {ipex.__version__}')
    print(f'XPU count: {torch.xpu.device_count()}')
    for i in range(torch.xpu.device_count()):
        if hasattr(torch.xpu, 'get_device_properties'):
            props = torch.xpu.get_device_properties(i)
            print(f'XPU {i}: {props.name if hasattr(props, \"name\") else \"Intel XPU\"}')
        else:
            print(f'XPU {i}: Intel XPU')
else:
    print('No XPUs detected through PyTorch')
" || log_warning "Could not get detailed XPU information"

# =================== FIX START (MODIFIED) ===================
# Pre-flight check to remove any stale lock directory from previous failed runs.
# This makes the script more robust.
log_header "PRE-FLIGHT CHECK: CLEARING STALE LOCK"
log_info "Attempting to remove any stale remote lock directory..."

# The new lock is a directory. Use rm -rf for robustness.
REMOTE_LOCK_DIR_PATH="/tank/zf281/global_stats.csv.lockdir"
ssh "$OTRERA_HOST" "rm -rf ${REMOTE_LOCK_DIR_PATH}"

if [ $? -eq 0 ]; then
    log_success "Stale lock check complete. Any existing lock on ${OTRERA_HOST} was removed."
else
    log_error "Could not connect to ${OTRERA_HOST} to clear stale lock. This could cause issues. Check SSH connectivity."
    # Depending on requirements, you might want to exit here
    # exit 1
fi
# =================== FIX END (MODIFIED) =====================

log_header "CSV INITIALIZATION"
log_info "Initializing CSV with current processing status..."

# Initialize the CSV by checking remote servers
$PYTHON_ENV "$CSV_SCRIPT" --option init
if [ $? -eq 0 ]; then
    log_success "CSV initialization completed"
else
    log_error "Failed to initialize CSV. Check logs and SSH connectivity."
    exit 1
fi

# Function to start an XPU worker process
start_xpu_worker() {
    local xpu_id=$1
    local process_id=$2
    local worker_id="${NODE_NAME}_${xpu_id}_${process_id}"
    
    # Calculate a unique numeric process ID for the inference script
    local numeric_process_id=$((xpu_id * 100 + process_id))
    
    log_info "Starting XPU $xpu_id worker $process_id (ID: $worker_id, numeric ID: $numeric_process_id)"
    
    # Create a Python worker script that will handle the continuous processing
    local worker_script="${NODE_LOG_DIR}/xpu_worker_${worker_id}.py"
    cat > "$worker_script" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import shutil
import traceback
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xpu-id', type=int, required=True)
    parser.add_argument('--worker-id', type=str, required=True)
    parser.add_argument('--numeric-process-id', type=int, required=True)
    parser.add_argument('--csv-script', type=str, required=True)
    parser.add_argument('--inference-script', type=str, required=True)
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--local-d-pixel-dir', type=str, required=True)
    parser.add_argument('--local-representation-dir', type=str, required=True)
    parser.add_argument('--otrera-host', type=str, required=True)
    parser.add_argument('--otrera-d-pixel-path', type=str, required=True)
    parser.add_argument('--antiope-host', type=str, required=True)
    parser.add_argument('--antiope-representation-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--verbose-xpu', action='store_true')
    parser.add_argument('--years', type=str, required=True)
    parser.add_argument('--node-name', type=str, required=True)
    parser.add_argument('--upload-queue-dir', type=str, required=True)
    return parser.parse_args()

def run_command(cmd, check=True):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        # Include stderr in the exception for better debugging
        error_message = f"Command failed: {cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        raise Exception(error_message)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def download_d_pixel(grid_name, year, otrera_host, otrera_path, local_path):
    """Download d-pixel data from otrera"""
    remote_dir = f"{otrera_host}:{otrera_path}/{year}/{grid_name}"
    local_dir = f"{local_path}/{year}/{grid_name}"
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Download using rsync for efficiency
    cmd = f"rsync -avz --progress {remote_dir}/ {local_dir}/"
    logging.info(f"Downloading {grid_name} for year {year}")
    stdout, stderr, returncode = run_command(cmd, check=False)
    if returncode != 0:
        raise Exception(f"rsync failed: {stderr}")
    
    # Verify download
    npy_files = list(Path(local_dir).glob("*.npy"))
    if len(npy_files) != 9:
        raise Exception(f"Expected 9 npy files, found {len(npy_files)}")
    
    return local_dir

def create_upload_task(grid_name, year, local_repr_dir, antiope_host, antiope_path, upload_queue_dir, node_name):
    """Create an upload task file for async upload"""
    task_file = f"{upload_queue_dir}/{node_name}_{grid_name}_{year}.json"
    task_data = {
        "grid_name": grid_name,
        "year": year,
        "local_dir": f"{local_repr_dir}/{year}/{grid_name}",
        "remote_dir": f"{antiope_host}:{antiope_path}/{year}/{grid_name}",
        "timestamp": time.time(),
        "node": node_name
    }
    
    with open(task_file, 'w') as f:
        json.dump(task_data, f)
    
    logging.info(f"Created upload task for {grid_name} year {year}")

def main():
    args = parse_args()
    years = json.loads(args.years)
    
    logging.info(f"XPU worker {args.worker_id} started on XPU {args.xpu_id} (node: {args.node_name})")
    logging.info(f"Processing years: {years}")
    
    processed_count = 0
    failed_count = 0
    
    while True:
        try:
            # Try to get a task for each year
            task_found = False
            grid_name = None
            year_of_task = None
            
            for year in years:
                # Get an unprocessed grid for this year
                cmd = f"{sys.executable} {args.csv_script} --option get_inference {year} random"
                stdout, stderr, returncode = run_command(cmd, check=False)
                
                # Log debug information from stderr
                if stderr:
                    logging.debug(f"get_inference stderr for year {year}: {stderr}")
                
                # Check if we got a valid grid name
                if returncode == 0 and stdout and stdout.strip():
                    grid_name = stdout.strip()
                    # Make sure it's not an error message
                    if not grid_name.startswith("Error") and not grid_name.startswith("DEBUG"):
                        year_of_task = year
                        logging.info(f"Got task: {grid_name} for year {year_of_task}")
                        task_found = True
                        break
                else:
                    logging.debug(f"No task available for year {year} (returncode: {returncode})")
            
            if not task_found:
                logging.info("No tasks available across all years. Checking again in 30 seconds.")
                time.sleep(30)
                continue # Restart the main while loop

            # --- Process the found task ---
            tile_list_path = f"{os.path.dirname(args.upload_queue_dir)}/tile_list_{args.worker_id}.json"
            try:
                # Update status to processing
                cmd = f"{sys.executable} {args.csv_script} --option update {grid_name} {year_of_task}_infer_status processing"
                stdout, stderr, returncode = run_command(cmd, check=False)
                if returncode != 0:
                    logging.error(f"Failed to update status to processing: {stderr}")
                
                # Update agent info
                agent_info = f"{args.node_name}_xpu{args.xpu_id}_worker{args.worker_id}"
                cmd = f"{sys.executable} {args.csv_script} --option update {grid_name} {year_of_task}_agent {agent_info}"
                stdout, stderr, returncode = run_command(cmd, check=False)
                if returncode != 0:
                    logging.error(f"Failed to update agent info: {stderr}")
                
                # Download d-pixel data
                download_start = time.time()
                local_d_pixel_path = download_d_pixel(
                    grid_name, year_of_task, 
                    args.otrera_host, args.otrera_d_pixel_path,
                    args.local_d_pixel_dir
                )
                download_time = time.time() - download_start
                logging.info(f"Downloaded {grid_name} in {download_time:.1f}s")
                
                # Prepare output directory
                output_dir = f"{args.local_representation_dir}/{year_of_task}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a temporary JSON file containing the path of the single tile to process.
                with open(tile_list_path, 'w') as f:
                    json.dump([local_d_pixel_path], f)
                logging.info(f"Created temporary tile list for {grid_name} at {tile_list_path}")

                # Run inference with Intel XPU
                inference_start = time.time()
                # Modified the command to use --tile_list instead of --tile_path and use numeric process ID
                inference_cmd = f"{sys.executable} {args.inference_script} " \
                                f"--config {args.config_file} " \
                                f"--mode xpu " \
                                f"--xpu_id {args.xpu_id} " \
                                f"--tile_list {tile_list_path} " \
                                f"--process_id {args.numeric_process_id} " \
                                f"--checkpoint_path {args.checkpoint_path} " \
                                f"--output_dir {output_dir} " \
                                f"--batch_size {args.batch_size} " \
                                f"--num_workers {args.num_workers} " \
                                f"--log_interval {args.log_interval} " \
                                f"--simplified_logging"
                
                if args.verbose_xpu:
                    inference_cmd += " --verbose_xpu"
                
                # Run inference
                stdout, stderr, returncode = run_command(inference_cmd, check=False)
                if returncode != 0:
                    raise Exception(f"Inference failed: {stderr}")
                
                inference_time = time.time() - inference_start
                logging.info(f"Inference completed for {grid_name} in {inference_time:.1f}s")
                
                # The output file from the script is named after the tile directory, not grid_name.
                tile_dir_name = os.path.basename(local_d_pixel_path)
                output_file = f"{output_dir}/{tile_dir_name}.npy"
                scale_file = f"{output_dir}/{tile_dir_name}_scales.npy"

                grid_output_dir = f"{output_dir}/{grid_name}"
                os.makedirs(grid_output_dir, exist_ok=True)
                
                if os.path.exists(output_file):
                    shutil.move(output_file, f"{grid_output_dir}/representation.npy")
                if os.path.exists(scale_file):
                    shutil.move(scale_file, f"{grid_output_dir}/scales.npy")
                
                # Create upload task
                create_upload_task(grid_name, year_of_task, args.local_representation_dir,
                                   args.antiope_host, args.antiope_representation_path,
                                   args.upload_queue_dir, args.node_name)
                
                # Update status to processed
                cmd = f"{sys.executable} {args.csv_script} --option update {grid_name} {year_of_task}_infer_status processed"
                stdout, stderr, returncode = run_command(cmd, check=False)
                if returncode != 0:
                    logging.error(f"Failed to update status to processed: {stderr}")
                
                # Clean up downloaded d-pixel data to save space
                shutil.rmtree(local_d_pixel_path)
                logging.info(f"Cleaned up d-pixel data for {grid_name}")
                
                processed_count += 1
                logging.info(f"Successfully processed {grid_name} for year {year_of_task} "
                             f"(Total processed: {processed_count}, Failed: {failed_count})")
            
            except Exception as e:
                failed_count += 1
                logging.error(f"Failed to process {grid_name} for year {year_of_task}: {str(e)}")
                logging.error(traceback.format_exc())
                
                # Update status to indicate failure
                try:
                    # Clean the error message for CSV to avoid format issues
                    clean_error = str(e).replace('"', "'").replace(',', ';').splitlines()[0]
                    note = f"Failed: {clean_error[:120]}"
                    cmd = f'{sys.executable} {args.csv_script} --option update "{grid_name}" "{year_of_task}_note" "{note}"'
                    stdout, stderr, returncode = run_command(cmd, check=False)
                    
                    # Reset status to empty so it can be retried
                    cmd = f"{sys.executable} {args.csv_script} --option update {grid_name} {year_of_task}_infer_status ''"
                    stdout, stderr, returncode = run_command(cmd, check=False)
                except Exception as e_update:
                    logging.error(f"Could not even update failure status for {grid_name}: {e_update}")
                    pass
            finally:
                # Clean up the temporary tile list file
                if os.path.exists(tile_list_path):
                    try:
                        os.remove(tile_list_path)
                    except OSError as e_clean:
                        logging.warning(f"Could not remove temporary tile list {tile_list_path}: {e_clean}")

        except KeyboardInterrupt:
            logging.info("Worker interrupted by user")
            break
        except Exception as e:
            logging.error(f"Critical worker error in main loop: {str(e)}")
            logging.error(traceback.format_exc())
            time.sleep(10) # Wait before retrying the loop
    
    logging.info(f"Worker {args.worker_id} exiting. Processed: {processed_count}, Failed: {failed_count}")
    # Create a success marker file
    Path(f"{os.path.dirname(args.upload_queue_dir)}/worker_{args.worker_id}.SUCCESS").touch()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If main fails catastrophically, log it and create a failure marker
        worker_id_arg = [arg for arg in sys.argv if arg.startswith('--worker-id')]
        worker_id = worker_id_arg[0].split('=')[1] if worker_id_arg else "unknown"
        logging.error(f"Catastrophic failure in worker {worker_id}: {e}")
        logging.error(traceback.format_exc())
        Path(f"logs/worker_{worker_id}.FAIL").touch()
        sys.exit(1)

EOF
    
    # Make the worker script executable
    chmod +x "$worker_script"
    
    # Build years JSON string
    YEARS_JSON=$(printf '%s\n' "${YEARS[@]}" | jq -R . | jq -s . | tr -d '\n')
    
    # Start the worker process
    local log_file="${NODE_LOG_DIR}/xpu_worker_${worker_id}.log"
    
    $PYTHON_ENV "$worker_script" \
        --xpu-id $xpu_id \
        --worker-id "$worker_id" \
        --numeric-process-id $numeric_process_id \
        --csv-script "$CSV_SCRIPT" \
        --inference-script "$INFERENCE_SCRIPT" \
        --config-file "$CONFIG_FILE" \
        --checkpoint-path "$CHECKPOINT_PATH" \
        --local-d-pixel-dir "$LOCAL_D_PIXEL_DIR" \
        --local-representation-dir "$LOCAL_REPRESENTATION_DIR" \
        --otrera-host "$OTRERA_HOST" \
        --otrera-d-pixel-path "$OTRERA_D_PIXEL_PATH" \
        --antiope-host "$ANTIOPE_HOST" \
        --antiope-representation-path "$ANTIOPE_REPRESENTATION_PATH" \
        --batch-size $XPU_BATCH_SIZE \
        --num-workers $XPU_NUM_WORKERS \
        --log-interval $LOG_INTERVAL \
        --years "$YEARS_JSON" \
        --node-name "$NODE_NAME" \
        --upload-queue-dir "$NODE_UPLOAD_QUEUE" \
        $([ "$VERBOSE_XPU" = "true" ] && echo "--verbose-xpu") \
        > "$log_file" 2>&1 &
    
    local pid=$!
    XPU_PIDS+=($pid)
    XPU_LOG_FILES+=($log_file) # Store log file path
    
    log_success "Started XPU $xpu_id worker $process_id with PID $pid, logging to $log_file"
    
    # Clean up the temporary worker script after a delay
    (sleep 5 && rm -f "$worker_script") &
    
    return 0
}

# Function to start the async upload daemon
start_upload_daemon() {
    log_info "Starting async upload daemon"
    
    # Create upload daemon script
    local upload_script="${NODE_LOG_DIR}/upload_daemon.py"
    cat > "$upload_script" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_command(cmd):
    """Run a shell command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def process_upload_queue(queue_dir):
    """Process pending upload tasks"""
    queue_dir = Path(queue_dir)
    
    while True:
        try:
            # Get all pending upload tasks
            task_files = sorted(list(queue_dir.glob("*.json")))
            
            if task_files:
                logging.info(f"Found {len(task_files)} upload tasks")
                
                for task_file in task_files:
                    try:
                        # Read task data
                        with open(task_file, 'r') as f:
                            task = json.load(f)
                        
                        grid_name = task["grid_name"]
                        year = task["year"]
                        local_dir = task["local_dir"]
                        remote_dir = task["remote_dir"]
                        node = task.get("node", "unknown")
                        
                        # Check if local files exist
                        if not os.path.exists(local_dir):
                            logging.warning(f"Local dir not found: {local_dir}, removing task")
                            task_file.unlink()
                            continue
                        
                        # Upload using rsync
                        logging.info(f"Uploading {grid_name} for year {year} from node {node}")
                        cmd = f"rsync -avz --progress {local_dir}/ {remote_dir}/"
                        success, stdout, stderr = run_command(cmd)
                        
                        if success:
                            logging.info(f"Successfully uploaded {grid_name} for year {year}")
                            
                            # Remove task file
                            task_file.unlink()
                            
                            # Now it's safe to remove local files
                            shutil.rmtree(local_dir)
                            logging.info(f"Cleaned up local representation files for {grid_name}")
                        else:
                            logging.error(f"Failed to upload {grid_name}: {stderr}")
                            # Don't delete the task, it will be retried
                            
                    except Exception as e:
                        logging.error(f"Error processing task {task_file}: {str(e)}")

            # Wait before checking again
            time.sleep(10)
            
        except KeyboardInterrupt:
            logging.info("Upload daemon interrupted")
            break
        except Exception as e:
            logging.error(f"Upload daemon error: {str(e)}")
            time.sleep(30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir', type=str, required=True)
    args = parser.parse_args()
    
    logging.info(f"Upload daemon started for queue directory: {args.queue_dir}")
    process_upload_queue(args.queue_dir)
    logging.info("Upload daemon exiting")
EOF
    
    # Make executable
    chmod +x "$upload_script"
    
    # Start the daemon
    local log_file="${NODE_LOG_DIR}/upload_daemon.log"
    $PYTHON_ENV "$upload_script" --queue-dir "$NODE_UPLOAD_QUEUE" > "$log_file" 2>&1 &
    
    local pid=$!
    UPLOAD_PIDS+=($pid)
    
    log_success "Started upload daemon with PID $pid"
    
    # Clean up script after delay
    (sleep 5 && rm -f "$upload_script") &
    
    return 0
}

# Function to check for failed workers and log their errors
check_failed_workers() {
    for i in "${!XPU_PIDS[@]}"; do
        local pid=${XPU_PIDS[$i]}
        local log_file=${XPU_LOG_FILES[$i]}
        local worker_id_str=$(basename "$log_file" .log)

        # Check if a failure marker exists for this worker
        if [ -f "${NODE_LOG_DIR}/${worker_id_str}.FAIL" ]; then
            log_error "Worker with PID $pid ($worker_id_str) has FAILED."
            log_error "--- Start of Error Log from $log_file ---"
            # Print the last 20 lines of the log for context
            tail -n 20 "$log_file" >&2
            log_error "--- End of Error Log from $log_file ---"
            # Remove the PID to avoid re-logging the error
            unset XPU_PIDS[$i]
            # Consider exiting the whole job if one worker fails
            # cleanup
            # exit 1
        fi

        # Also check if the process died without a success marker
        if ! ps -p $pid > /dev/null 2>&1 && [ ! -f "${NODE_LOG_DIR}/${worker_id_str}.SUCCESS" ]; then
            log_warning "Worker with PID $pid ($worker_id_str) appears to have died unexpectedly."
            log_warning "--- Start of Log from $log_file ---"
            tail -n 20 "$log_file" >&2
            log_warning "--- End of Log from $log_file ---"
            unset XPU_PIDS[$i]
        fi
    done
}

log_header "STARTING INFERENCE WORKERS"

# Calculate total number of worker processes
TOTAL_XPU_PROCESSES=$((XPU_COUNT * PROCESSES_PER_XPU))
log_info "Starting $TOTAL_XPU_PROCESSES XPU worker processes ($PROCESSES_PER_XPU per XPU)"

# Start XPU worker processes
for ((xpu_id=0; xpu_id<XPU_COUNT; xpu_id++)); do
    for ((proc_id=0; proc_id<PROCESSES_PER_XPU; proc_id++)); do
        start_xpu_worker $xpu_id $proc_id
        sleep 2  # Brief delay between starting processes
    done
done

# Start the upload daemon
start_upload_daemon

log_header "MONITORING PROGRESS"
log_info "All workers started. Monitoring progress..."
log_info "Check individual worker logs in ${NODE_LOG_DIR}/xpu_worker_*.log"
log_info "Check upload daemon log in ${NODE_LOG_DIR}/upload_daemon.log"

# Monitor workers
MONITORING=true
while $MONITORING; do
    # Check for any failed workers and log their errors
    check_failed_workers
    
    # Check how many workers are still running
    RUNNING_WORKERS=0
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    # Get upload queue status
    UPLOAD_QUEUE_SIZE=$(ls ${NODE_UPLOAD_QUEUE}/*.json 2>/dev/null | wc -l)
    
    # Calculate elapsed time
    ELAPSED=$(($(date +%s) - SCRIPT_START))
    
    # Log status
    log_info "Status: $RUNNING_WORKERS/$TOTAL_XPU_PROCESSES workers running, $UPLOAD_QUEUE_SIZE uploads pending - Elapsed: $(format_time $ELAPSED)"
    
    # Check if all workers have stopped
    if [ $RUNNING_WORKERS -eq 0 ]; then
        log_warning "All worker processes have stopped."
        MONITORING=false
    fi
    
    # Sleep before next check
    sleep 30
done

# Wait for uploads to complete
if [ ${#UPLOAD_PIDS[@]} -gt 0 ]; then
    log_info "Waiting for final uploads to complete..."
    
    # Give uploads up to 5 minutes to finish
    MAX_WAIT=300
    WAITED=0
    
    while [ $WAITED -lt $MAX_WAIT ]; do
        UPLOAD_QUEUE_SIZE=$(ls ${NODE_UPLOAD_QUEUE}/*.json 2>/dev/null | wc -l)
        
        if [ $UPLOAD_QUEUE_SIZE -eq 0 ]; then
            log_success "All uploads completed"
            break
        fi
        
        log_info "Waiting for $UPLOAD_QUEUE_SIZE uploads to complete..."
        sleep 10
        WAITED=$((WAITED + 10))
    done
    
    if [ $(ls ${NODE_UPLOAD_QUEUE}/*.json 2>/dev/null | wc -l) -ne 0 ]; then
        log_warning "Uploads did not complete within the wait time. There may be pending items in '${NODE_UPLOAD_QUEUE}/'."
    fi

    # Stop upload daemon
    for pid in "${UPLOAD_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            kill_process_tree $pid
        fi
    done
fi

log_header "INFERENCE COMPLETE"

# Calculate total time
TOTAL_TIME=$(($(date +%s) - SCRIPT_START))
log_info "Total execution time: $(format_time $TOTAL_TIME)"

log_success "All processing finished at $(date)"