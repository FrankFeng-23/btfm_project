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

############### This needs to be modified to your environment ###############
# Python environment that has the required packages installed
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
export PYTHON_ENV="python"

# --- Task Queue Directories on Otrera ---
TASK_QUEUE_BASE="/tank/zf281/task_queue"
INFER_PENDING_DIR="${TASK_QUEUE_BASE}/representation_inference/pending"
INFER_PROCESSING_DIR="${TASK_QUEUE_BASE}/representation_inference/processing"
INFER_DONE_DIR="${TASK_QUEUE_BASE}/representation_inference/done"
INFER_FAILED_DIR="${TASK_QUEUE_BASE}/representation_inference/failed"

# --- FIXED: Base directories on the compute node ---
# Using /local as per your environment
LOCAL_SCRATCH_DIR="/local/${USER}_${SLURM_JOB_ID}"
LOCAL_D_PIXEL_DIR="${LOCAL_SCRATCH_DIR}/global_d_pixel"
LOCAL_REPRESENTATION_DIR="${LOCAL_SCRATCH_DIR}/global_representation"

# Remote hosts and paths
OTRERA_HOST="zf281@otrera.cl.cam.ac.uk"
ANTIOPE_HOST="zf281@antiope.cl.cam.ac.uk"
OTRERA_D_PIXEL_PATH="/tank/zf281/global_0.1_degree_tiff_d_pixel"
ANTIOPE_REPRESENTATION_PATH="/tank/zf281/global_0.1_degree_representation"

# Paths to script and models
INFERENCE_SCRIPT="src/multi_tile_infer_QAT.py"
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250608_220648_QAT.pt"
CONFIG_FILE="configs/multi_tile_infer_config.py"

# XPU Configuration
PROCESSES_PER_XPU=4
XPU_BATCH_SIZE=4096
XPU_NUM_WORKERS=2
VERBOSE_XPU=true

# --- Node-level batching configuration ---
NODE_TASK_BATCH_SIZE=64
MAX_RETRIES=3  # Maximum retries for claiming tasks
RETRY_DELAY=5   # Delay between retries in seconds
###########################################################################

# Get node-specific information
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-$$}"
UNIQUE_JOB_ID="${NODE_NAME}_${JOB_ID}"

# Create local directories for this job run
NODE_LOG_DIR="logs/job_${JOB_ID}"
mkdir -p "$NODE_LOG_DIR"
mkdir -p "$LOCAL_D_PIXEL_DIR"
mkdir -p "$LOCAL_REPRESENTATION_DIR"

# Local task queue directories for this node
LOCAL_TASK_BASE_DIR="${LOCAL_SCRATCH_DIR}/local_task_queue"
LOCAL_PENDING_DIR="${LOCAL_TASK_BASE_DIR}/pending"
LOCAL_PROCESSING_DIR="${LOCAL_TASK_BASE_DIR}/processing"
LOCAL_DONE_DIR="${LOCAL_TASK_BASE_DIR}/done"
LOCAL_FAILED_DIR="${LOCAL_TASK_BASE_DIR}/failed"
mkdir -p "$LOCAL_PENDING_DIR" "$LOCAL_PROCESSING_DIR" "$LOCAL_DONE_DIR" "$LOCAL_FAILED_DIR"

# Script state variables
SCRIPT_START=$(date +%s)
declare -a XPU_PIDS
declare -A CURRENT_BATCH_DIRS  # Changed to associative array for better tracking

# Terminal colors for logging
BOLD="\033[1m"; RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; CYAN="\033[36m"; RESET="\033[0m"

# Logging functions
log_header() { echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2; }
log_info() { echo -e "${CYAN}[INFO][${NODE_NAME}]${RESET} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS][${NODE_NAME}]${RESET} $1" >&2; }
log_warning() { echo -e "${YELLOW}[WARNING][${NODE_NAME}]${RESET} $1" >&2; }
log_error() { echo -e "${RED}[ERROR][${NODE_NAME}]${RESET} $1" >&2; }
log_debug() { echo -e "${MAGENTA}[DEBUG][${NODE_NAME}]${RESET} $1" >&2; }

# Utility functions
format_time() {
    local s=$1; printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

kill_process_tree() {
    local pid=$1
    local children=$(pgrep -P "$pid" 2>/dev/null)
    for child in $children; do kill_process_tree "$child"; done
    kill -9 "$pid" 2>/dev/null || true
}

# Cleanup function on exit
cleanup() {
    log_warning "Received interrupt signal. Cleaning up..."

    # Return all stranded batches to pending queue
    for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
        if [[ -n "$batch_dir" ]]; then
            log_warning "Attempting to return stranded task batch from ${batch_dir} to pending queue..."
            ssh "$OTRERA_HOST" "find ${batch_dir} -name '*.task' -exec mv {} ${INFER_PENDING_DIR}/ \\; 2>/dev/null"
            ssh "$OTRERA_HOST" "rmdir ${batch_dir} 2>/dev/null"
        fi
    done

    for pid in "${XPU_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then kill_process_tree "$pid"; fi
    done

    log_warning "Cleaning up local scratch directory: ${LOCAL_SCRATCH_DIR}"
    rm -rf "${LOCAL_SCRATCH_DIR}"

    log_warning "All processes terminated. Exiting."
    exit 1
}

trap cleanup SIGINT SIGTERM

# Atomic task claiming function using file-by-file approach with year prioritization
claim_tasks_atomically() {
    local batch_size=$1
    local batch_dir=$2
    local claimed_count=0
    
    log_info "Attempting to atomically claim up to ${batch_size} tasks (prioritizing recent years)..."
    
    # Create remote batch directory
    ssh "$OTRERA_HOST" "mkdir -p ${batch_dir}" || return 1
    
    # Use a single SSH connection with optimized year-based claiming
    ssh "$OTRERA_HOST" "
        cd ${INFER_PENDING_DIR} 2>/dev/null || exit 1
        claimed=0
        
        # Get all years and sort them in descending order (newest first)
        years=\$(ls *.task 2>/dev/null | awk -F'_' '{print \$NF}' | sed 's/.task//' | sort -nur | uniq)
        
        # Process each year until we have enough tasks
        for year in \$years; do
            # Get tasks for this year and randomly sample
            # Using awk with random sampling for better performance
            year_tasks=\$(ls *_\${year}.task 2>/dev/null | awk 'BEGIN{srand()} {print rand() \"\t\" \$0}' | sort -n | cut -f2-)
            
            # Try to claim tasks from this randomized list
            for task in \$year_tasks; do
                [ -f \"\$task\" ] || continue
                if mv \"\$task\" ${batch_dir}/ 2>/dev/null; then
                    claimed=\$((claimed + 1))
                    [ \$claimed -ge ${batch_size} ] && break 2
                fi
            done
        done
        
        echo \$claimed
    " 2>/dev/null || echo "0"
}

# Improved function to finalize a batch with better error handling
finalize_batch() {
    local batch_dir=$1
    local batch_id="${CURRENT_BATCH_DIRS[$batch_dir]}"
    
    log_info "Finalizing batch ${batch_id} (${batch_dir})..."
    
    # Process completed tasks
    local done_count=0
    local done_errors=0
    
    # Check if there are any .task files in LOCAL_DONE_DIR
    if ls "${LOCAL_DONE_DIR}"/*.task >/dev/null 2>&1; then
        for task_file in "${LOCAL_DONE_DIR}"/*.task; do
            [[ -f "$task_file" ]] || continue
            
            local task_basename=$(basename "$task_file")
            
            # Try to move from batch dir to done dir on remote
            if ssh "$OTRERA_HOST" "
                if [ -f ${batch_dir}/${task_basename} ]; then
                    mv ${batch_dir}/${task_basename} ${INFER_DONE_DIR}/ 2>/dev/null
                elif [ -f ${INFER_DONE_DIR}/${task_basename} ]; then
                    exit 0  # Already moved, consider it success
                else
                    exit 1  # File not found anywhere
                fi
            "; then
                rm -f "$task_file"
                ((done_count++))
            else
                ((done_errors++))
                log_warning "Could not finalize ${task_basename} - may have been processed by another worker"
                # Still remove from local done dir to avoid repeated attempts
                rm -f "$task_file"
            fi
        done
    fi
    
    if [ $done_count -gt 0 ]; then
        log_info "Successfully finalized ${done_count} completed tasks"
    fi
    if [ $done_errors -gt 0 ]; then
        log_warning "${done_errors} tasks could not be finalized (may be normal if processed by other workers)"
    fi

    # Process failed tasks
    local failed_count=0
    
    # Check if there are any .task files in LOCAL_FAILED_DIR
    if ls "${LOCAL_FAILED_DIR}"/*.task >/dev/null 2>&1; then
        for task_file in "${LOCAL_FAILED_DIR}"/*.task; do
            [[ -f "$task_file" ]] || continue
            
            local task_basename=$(basename "$task_file")
            
            # Try to move from batch dir to failed dir on remote
            if ssh "$OTRERA_HOST" "
                if [ -f ${batch_dir}/${task_basename} ]; then
                    mv ${batch_dir}/${task_basename} ${INFER_FAILED_DIR}/ 2>/dev/null
                elif [ -f ${INFER_FAILED_DIR}/${task_basename} ]; then
                    exit 0  # Already moved, consider it success
                else
                    exit 1  # File not found anywhere
                fi
            "; then
                rm -f "$task_file"
                ((failed_count++))
            else
                log_warning "Could not move failed task ${task_basename}"
                rm -f "$task_file"
            fi
        done
    fi
    
    if [ $failed_count -gt 0 ]; then
        log_info "Moved ${failed_count} failed tasks to remote 'failed' folder"
    fi
    
    # Check if batch directory is empty and remove it
    remaining_tasks=$(ssh "$OTRERA_HOST" "ls -1 ${batch_dir}/*.task 2>/dev/null | wc -l" || echo "0")
    if [ "$remaining_tasks" -eq 0 ]; then
        ssh "$OTRERA_HOST" "rmdir ${batch_dir} 2>/dev/null"
        log_info "Removed empty batch directory ${batch_dir}"
        unset CURRENT_BATCH_DIRS["$batch_dir"]
    else
        log_info "Batch directory ${batch_dir} still contains ${remaining_tasks} tasks (being processed)"
    fi
}

log_header "GLOBAL REPRESENTATION INFERENCE (Node-Staging Pipeline)"
log_info "Starting at $(date)"
log_info "Node: ${NODE_NAME}, Job ID: ${JOB_ID}"
log_info "Local scratch space: ${LOCAL_SCRATCH_DIR}"

log_header "SETUP & PRE-FLIGHT CHECKS"
log_info "Ensuring remote task queue directories exist on ${OTRERA_HOST}..."
ssh "$OTRERA_HOST" "mkdir -p ${INFER_PENDING_DIR} ${INFER_PROCESSING_DIR} ${INFER_DONE_DIR} ${INFER_FAILED_DIR}"
if [ $? -ne 0 ]; then
    log_error "Could not create/verify remote directories on ${OTRERA_HOST}. Check SSH connectivity."
    exit 1
fi
log_success "Remote directories are ready."

XPU_COUNT=$($PYTHON_ENV -c "import torch, intel_extension_for_pytorch as ipex; print(torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0)" 2>/dev/null)
if ! [[ "$XPU_COUNT" =~ ^[0-9]+$ ]] || [ "$XPU_COUNT" -eq 0 ]; then
    log_error "No Intel XPUs detected! This script requires Intel GPUs."
    exit 1
fi
log_success "Detected $XPU_COUNT Intel XPUs"

# ==============================================================================
# ==== XPU WORKER PROCESS (modified to support continuous operation) ===========
# ==============================================================================

start_xpu_worker() {
    local xpu_id=$1
    local process_id=$2
    local worker_id="${UNIQUE_JOB_ID}_xpu${xpu_id}_proc${process_id}"
    local numeric_process_id=$((xpu_id * 100 + process_id))

    log_info "Starting XPU worker: ${worker_id}"
    local log_file="${NODE_LOG_DIR}/worker_${worker_id}.log"

    # Create a control file for this worker
    local control_file="${LOCAL_SCRATCH_DIR}/worker_${worker_id}.control"
    echo "running" > "$control_file"

    $PYTHON_ENV -u - > "$log_file" 2>&1 <<EOF &
import os, sys, time, random, subprocess, shutil, traceback, logging
from pathlib import Path

# Config passed from bash
WORKER_ID = "$worker_id"
XPU_ID = $xpu_id
NUMERIC_PROCESS_ID = $numeric_process_id
CONTROL_FILE = "$control_file"

# Local paths
LOCAL_PENDING_DIR = Path("$LOCAL_PENDING_DIR")
LOCAL_PROCESSING_DIR = Path("$LOCAL_PROCESSING_DIR")
LOCAL_DONE_DIR = Path("$LOCAL_DONE_DIR")
LOCAL_FAILED_DIR = Path("$LOCAL_FAILED_DIR")
LOCAL_D_PIXEL_DIR = Path("$LOCAL_D_PIXEL_DIR")
LOCAL_REPRESENTATION_DIR = Path("$LOCAL_REPRESENTATION_DIR")

# Remote upload destination
ANTIOPE_HOST = "$ANTIOPE_HOST"
ANTIOPE_REPRESENTATION_PATH = "$ANTIOPE_REPRESENTATION_PATH"

# Inference parameters
INFERENCE_SCRIPT = "$INFERENCE_SCRIPT"
CHECKPOINT_PATH = "$CHECKPOINT_PATH"
CONFIG_FILE = "$CONFIG_FILE"
XPU_BATCH_SIZE = $XPU_BATCH_SIZE
XPU_NUM_WORKERS = $XPU_NUM_WORKERS
VERBOSE_XPU_FLAG = "--verbose_xpu" if "$VERBOSE_XPU" == "true" else ""

# Setup logging
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(levelname)s - [{WORKER_ID}] - %(message)s', stream=sys.stdout)

def should_continue():
    try:
        with open(CONTROL_FILE, 'r') as f:
            return f.read().strip() == "running"
    except: return False

def run_command(cmd, check=True):
    logging.debug(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(f"Command failed with exit code {result.returncode}: {cmd}\\nSTDOUT: {result.stdout}\\nSTDERR: {result.stderr}")
    return result

def get_local_task():
    try:
        pending_files = [f for f in LOCAL_PENDING_DIR.iterdir() if f.is_file() and f.name.endswith('.task')]
        if not pending_files: return None
        task_file = random.choice(pending_files)
        # Keep the original task name but store it in a worker-specific location
        processing_path = LOCAL_PROCESSING_DIR / f"{WORKER_ID}_{task_file.name}"
        shutil.move(str(task_file), str(processing_path))
        logging.info(f"Acquired local task: {task_file.name}")
        return processing_path
    except FileNotFoundError: return None
    except Exception as e:
        logging.error(f"Error acquiring local task: {e}"); return None

def process_task(task_path):
    # Extract original task name without worker prefix
    original_task_name = task_path.name
    if original_task_name.startswith(f"{WORKER_ID}_"):
        original_task_name = original_task_name[len(f"{WORKER_ID}_"):]
    
    grid_id, year = original_task_name.replace('.task', '').rsplit('_', 1)
    logging.info(f"Processing task: grid={grid_id}, year={year}")
    
    local_d_pixel_path = LOCAL_D_PIXEL_DIR / year / grid_id
    logging.info(f"Using pre-downloaded data from: {local_d_pixel_path}")
    
    local_representation_path = LOCAL_REPRESENTATION_DIR / year
    local_representation_path.mkdir(parents=True, exist_ok=True)
    tile_list_path = Path(f"/tmp/tile_list_{WORKER_ID}.json")
    with open(tile_list_path, 'w') as f: import json; json.dump([str(local_d_pixel_path)], f)

    logging.info("Starting inference on XPU...")
    inference_cmd = (f"{sys.executable} {INFERENCE_SCRIPT} --config {CONFIG_FILE} --mode xpu --xpu_id {XPU_ID} "
                     f"--tile_list {tile_list_path} --process_id {NUMERIC_PROCESS_ID} --checkpoint_path {CHECKPOINT_PATH} "
                     f"--output_dir {local_representation_path} --batch_size {XPU_BATCH_SIZE} --num_workers {XPU_NUM_WORKERS} "
                     f"--simplified_logging {VERBOSE_XPU_FLAG}")
    run_command(inference_cmd)
    
    source_dir_name = local_d_pixel_path.name
    generated_repr_file = local_representation_path / f"{source_dir_name}.npy"
    generated_scale_file = local_representation_path / f"{source_dir_name}_scales.npy"
    if not generated_repr_file.exists() or not generated_scale_file.exists():
        raise FileNotFoundError("Inference output files not found")

    remote_upload_path = f"{ANTIOPE_REPRESENTATION_PATH}/{year}/{grid_id}"
    run_command(f"ssh {ANTIOPE_HOST} 'mkdir -p {remote_upload_path}'")
    logging.info(f"Uploading representation to {ANTIOPE_HOST}:{remote_upload_path}")
    # Use rsync instead of scp
    run_command(f"rsync -azP {generated_repr_file} {generated_scale_file} {ANTIOPE_HOST}:{remote_upload_path}/")
    
    # Clean up local files
    os.remove(generated_repr_file)
    os.remove(generated_scale_file)
    os.remove(tile_list_path)
    
    # Delete the downloaded d-pixel data after successful upload
    logging.info(f"Deleting local d-pixel data: {local_d_pixel_path}")
    try:
        shutil.rmtree(local_d_pixel_path)
        logging.info(f"Successfully deleted {local_d_pixel_path}")
        
        # Also try to remove the year directory if it's empty
        year_dir = local_d_pixel_path.parent
        if year_dir.exists() and not any(year_dir.iterdir()):
            year_dir.rmdir()
            logging.info(f"Removed empty year directory: {year_dir}")
    except Exception as e:
        logging.warning(f"Failed to delete d-pixel data: {e}")
    
    return original_task_name  # Return the original task name without worker prefix

def main():
    logging.info("Worker started. Waiting for tasks...")
    idle_count = 0
    while should_continue():
        task_path = get_local_task()
        if task_path:
            idle_count = 0
            try:
                original_task_name = process_task(task_path)
                # Move to done with original task name (no worker prefix)
                shutil.move(str(task_path), str(LOCAL_DONE_DIR / original_task_name))
                logging.info(f"Successfully completed task {original_task_name}")
            except Exception:
                logging.error(f"FATAL error processing task {task_path.name}\\n{traceback.format_exc()}")
                # Extract original task name for failed tasks too
                original_task_name = task_path.name
                if original_task_name.startswith(f"{WORKER_ID}_"):
                    original_task_name = original_task_name[len(f"{WORKER_ID}_"):]
                shutil.move(str(task_path), str(LOCAL_FAILED_DIR / original_task_name))
        else:
            idle_count += 1
            if idle_count > 1800:  # Exit after 30 minutes of no tasks
                logging.info("No tasks for 30 minutes. Exiting.")
                break
            time.sleep(1)
    logging.info("Worker shutting down.")
    
if __name__ == "__main__": main()
EOF

    local pid=$!
    XPU_PIDS+=($pid)
    log_success "Started worker ${worker_id} with PID $pid, logging to $log_file"
}

log_header "STARTING INFERENCE WORKERS"
TOTAL_XPU_PROCESSES=$((XPU_COUNT * PROCESSES_PER_XPU))
log_info "Starting $TOTAL_XPU_PROCESSES XPU worker processes"

for ((xpu=0; xpu<XPU_COUNT; xpu++)); do
    for ((proc=0; proc<PROCESSES_PER_XPU; proc++)); do
        start_xpu_worker "$xpu" "$proc"
        sleep 0.5
    done
done

# ==============================================================================
# ==== MAIN CONTROL LOOP WITH CONTINUOUS BATCH FETCHING =======================
# ==============================================================================

log_header "ENTERING MAIN CONTROL LOOP"
BATCH_NUMBER=0
NO_TASKS_COUNT=0
MAX_NO_TASKS_COUNT=10  # Exit after this many consecutive checks with no tasks
LAST_FINALIZE_TIME=$(date +%s)
FINALIZE_INTERVAL=900  # Finalize batches every 15 minutes

while true; do
    # Check if there are pending tasks remotely
    REMOTE_PENDING_COUNT=$(ssh "$OTRERA_HOST" "ls -1q ${INFER_PENDING_DIR}/*.task 2>/dev/null | wc -l")
    LOCAL_PENDING_COUNT=$(ls -1q "${LOCAL_PENDING_DIR}"/*.task 2>/dev/null | wc -l)
    
    # If local queue is low and remote has tasks, fetch a new batch
    if [ "$LOCAL_PENDING_COUNT" -lt 20 ] && [ "$REMOTE_PENDING_COUNT" -gt 0 ]; then
        BATCH_NUMBER=$((BATCH_NUMBER + 1))
        BATCH_ID="${UNIQUE_JOB_ID}_batch${BATCH_NUMBER}"
        REMOTE_BATCH_DIR="${INFER_PROCESSING_DIR}/${BATCH_ID}"
        CURRENT_BATCH_DIRS["$REMOTE_BATCH_DIR"]="$BATCH_ID"
        
        log_header "FETCHING BATCH #${BATCH_NUMBER}"
        
        # Atomically claim tasks
        CLAIMED_COUNT=$(claim_tasks_atomically "$NODE_TASK_BATCH_SIZE" "$REMOTE_BATCH_DIR")
        
        if [ "$CLAIMED_COUNT" -gt 0 ]; then
            NO_TASKS_COUNT=0
            log_success "Successfully claimed ${CLAIMED_COUNT} tasks in batch #${BATCH_NUMBER}"
            
            # Get list of claimed tasks
            CLAIMED_TASKS=$(ssh "$OTRERA_HOST" "ls ${REMOTE_BATCH_DIR}/*.task 2>/dev/null")
            
            # Download data for claimed tasks
            log_info "Downloading data for ${CLAIMED_COUNT} tasks..."
            DOWNLOAD_SUCCESS_COUNT=0
            
            for task_path in $CLAIMED_TASKS; do
                task_basename=$(basename "$task_path")
                grid_id=$(echo "$task_basename" | sed -E 's/^(.*)_([^_]+)\.task$/\1/')
                year=$(echo "$task_basename" | sed -E 's/^(.*)_([^_]+)\.task$/\2/')
                
                remote_d_pixel_dir="${OTRERA_D_PIXEL_PATH}/${year}/${grid_id}"
                local_grid_dir="${LOCAL_D_PIXEL_DIR}/${year}/${grid_id}"
                
                # Skip if already downloaded
                if [ -d "$local_grid_dir" ] && [ "$(ls -A "$local_grid_dir")" ]; then
                    log_info "  Data already exists for ${task_basename}, skipping download"
                    touch "${LOCAL_PENDING_DIR}/${task_basename}"
                    DOWNLOAD_SUCCESS_COUNT=$((DOWNLOAD_SUCCESS_COUNT + 1))
                    continue
                fi
                
                mkdir -p "$local_grid_dir"
                rsync -azP "${OTRERA_HOST}:${remote_d_pixel_dir}/" "${local_grid_dir}/"
                if [ $? -eq 0 ]; then
                    touch "${LOCAL_PENDING_DIR}/${task_basename}"
                    DOWNLOAD_SUCCESS_COUNT=$((DOWNLOAD_SUCCESS_COUNT + 1))
                else
                    log_error "  Download FAILED for ${task_basename}"
                    ssh "$OTRERA_HOST" "mv ${REMOTE_BATCH_DIR}/${task_basename} ${INFER_FAILED_DIR}/"
                fi
            done
            
            log_success "Downloaded data for ${DOWNLOAD_SUCCESS_COUNT}/${CLAIMED_COUNT} tasks"
            
        else
            log_warning "Failed to claim any tasks from batch #${BATCH_NUMBER}"
        fi
    elif [ "$REMOTE_PENDING_COUNT" -eq 0 ]; then
        NO_TASKS_COUNT=$((NO_TASKS_COUNT + 1))
        if [ "$NO_TASKS_COUNT" -ge "$MAX_NO_TASKS_COUNT" ] && [ "$LOCAL_PENDING_COUNT" -eq 0 ]; then
            log_info "No more tasks available remotely or locally. Preparing to shut down..."
            break
        fi
    fi
    
    # Monitor worker status
    RUNNING_WORKERS=0
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    if [ "$RUNNING_WORKERS" -eq 0 ]; then
        log_warning "All worker processes have exited unexpectedly!"
        break
    fi
    
    # Periodically finalize completed batches
    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - LAST_FINALIZE_TIME)) -ge "$FINALIZE_INTERVAL" ]; then
        log_info "Running periodic batch finalization..."
        for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
            finalize_batch "$batch_dir"
        done
        LAST_FINALIZE_TIME=$CURRENT_TIME
    fi
    
    # Status update
    LOCAL_DONE_COUNT=$(ls -1q "${LOCAL_DONE_DIR}"/*.task 2>/dev/null | wc -l)
    LOCAL_FAILED_COUNT=$(ls -1q "${LOCAL_FAILED_DIR}"/*.task 2>/dev/null | wc -l)
    ELAPSED=$(($(date +%s) - SCRIPT_START))
    
    log_info "STATUS | Batch: #${BATCH_NUMBER} | Workers: ${RUNNING_WORKERS}/${TOTAL_XPU_PROCESSES} | Local (Pend/Done/Failed): ${LOCAL_PENDING_COUNT}/${LOCAL_DONE_COUNT}/${LOCAL_FAILED_COUNT} | Remote Pending: ${REMOTE_PENDING_COUNT} | Elapsed: $(format_time "$ELAPSED")"
    
    sleep 30
done

# ==============================================================================
# ==== SHUTDOWN SEQUENCE =======================================================
# ==============================================================================

log_header "INITIATING SHUTDOWN SEQUENCE"

# Signal all workers to stop
log_info "Signaling all workers to stop..."
for ((xpu=0; xpu<XPU_COUNT; xpu++)); do
    for ((proc=0; proc<PROCESSES_PER_XPU; proc++)); do
        control_file="${LOCAL_SCRATCH_DIR}/worker_${UNIQUE_JOB_ID}_xpu${xpu}_proc${proc}.control"
        echo "stop" > "$control_file"
    done
done

# Wait for workers to finish current tasks
log_info "Waiting for workers to finish current tasks..."
WAIT_COUNT=0
while [ "$WAIT_COUNT" -lt 60 ]; do  # Wait up to 30 minutes
    RUNNING_WORKERS=0
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    if [ "$RUNNING_WORKERS" -eq 0 ]; then
        break
    fi
    
    sleep 30
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# Force kill any remaining workers
for pid in "${XPU_PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        kill_process_tree "$pid"
    fi
done

# Finalize all remaining batches
log_info "Finalizing all remaining batches..."
for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
    finalize_batch "$batch_dir"
done

# Final cleanup
log_info "Cleaning up local scratch directory: ${LOCAL_SCRATCH_DIR}"
rm -rf "${LOCAL_SCRATCH_DIR}"

TOTAL_TIME=$(($(date +%s) - SCRIPT_START))
log_info "Total execution time: $(format_time $TOTAL_TIME)"
log_success "All processing finished at $(date)"