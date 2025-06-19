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
export PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"

# --- Task Queue Directories on Otrera ---
TASK_QUEUE_BASE="/tank/zf281/task_queue"
DPIXEL_PENDING_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/pending"
DPIXEL_PROCESSING_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/processing"
DPIXEL_DONE_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/done"
DPIXEL_FAILED_DIR="${TASK_QUEUE_BASE}/d_pixel_generation/failed"

# --- Base directories on the compute node ---
LOCAL_SCRATCH_DIR="/home/azureuser/data/${USER}_${SLURM_JOB_ID:-$$}"
LOCAL_TIFF_DIR="${LOCAL_SCRATCH_DIR}/grid_tiffs"
LOCAL_D_PIXEL_DIR="${LOCAL_SCRATCH_DIR}/d_pixels"
LOCAL_UPLOAD_QUEUE_DIR="${LOCAL_SCRATCH_DIR}/upload_queue"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remote hosts and paths
OTRERA_HOST="zf281@otrera.caelum.ci.dev"
OTRERA_TIFF_PATH="/tank/zf281/global_0.1_degree_tiff"
OTRERA_D_PIXEL_PATH="/tank/zf281/global_0.1_degree_tiff_d_pixel"

# Worker configuration
PROCESSES_PER_NODE=1  # Multiple workers per node for CPU tasks
UPLOAD_WORKERS=2  # Number of concurrent upload workers
NODE_TASK_BATCH_SIZE=16  # Number of tasks to claim at once
MAX_RETRIES=3
RETRY_DELAY=5

# Get node-specific information
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-$$}"
UNIQUE_JOB_ID="${NODE_NAME}_${JOB_ID}"

# Create local directories for this job run
NODE_LOG_DIR="logs/dpixel_job_${JOB_ID}"
mkdir -p "$NODE_LOG_DIR"
mkdir -p "$LOCAL_TIFF_DIR"
mkdir -p "$LOCAL_D_PIXEL_DIR"
mkdir -p "$LOCAL_UPLOAD_QUEUE_DIR"
mkdir -p slurm_logs

# Local task queue directories for this node
LOCAL_TASK_BASE_DIR="${LOCAL_SCRATCH_DIR}/local_task_queue"
LOCAL_PENDING_DIR="${LOCAL_TASK_BASE_DIR}/pending"
LOCAL_PROCESSING_DIR="${LOCAL_TASK_BASE_DIR}/processing"
LOCAL_DONE_DIR="${LOCAL_TASK_BASE_DIR}/done"
LOCAL_FAILED_DIR="${LOCAL_TASK_BASE_DIR}/failed"
mkdir -p "$LOCAL_PENDING_DIR" "$LOCAL_PROCESSING_DIR" "$LOCAL_DONE_DIR" "$LOCAL_FAILED_DIR"

# Script state variables
SCRIPT_START=$(date +%s)
declare -a WORKER_PIDS
declare -a UPLOAD_WORKER_PIDS
declare -A CURRENT_BATCH_DIRS

# Terminal colors for logging
BOLD="\033[1m"; RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; CYAN="\033[36m"; RESET="\033[0m"

# Logging functions
log_header() { echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2; }
log_info() { echo -e "${CYAN}[INFO][${NODE_NAME}]${RESET} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS][${NODE_NAME}]${RESET} $1" >&2; }
log_warning() { echo -e "${YELLOW}[WARNING][${NODE_NAME}]${RESET} $1" >&2; }
log_error() { echo -e "${RED}[ERROR][${NODE_NAME}]${RESET} $1" >&2; }

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

# SSH connection management
setup_ssh_control_master() {
    local host=$1
    local control_path="/tmp/ssh_mux_${host//[@.]/_}_$$"
    
    # Close any existing connection
    ssh -O exit -o ControlPath="$control_path" "$host" 2>/dev/null || true
    
    # Setup new control master with error checking
    if ! ssh -fN -M -o ControlMaster=yes -o ControlPath="$control_path" -o ControlPersist=600 "$host"; then
        log_error "Failed to establish SSH control master to $host"
        exit 1
    fi
    
    # Verify the connection works
    if ! ssh -o ControlPath="$control_path" "$host" "echo 'SSH connection test successful'" >/dev/null 2>&1; then
        log_error "SSH control master established but connection test failed"
        exit 1
    fi
    
    echo "$control_path"
}

run_ssh_command() {
    local host=$1
    local control_path=$2
    local cmd=$3
    ssh -o ControlPath="$control_path" "$host" "$cmd"
}

# Cleanup function on exit
cleanup() {
    log_warning "Received interrupt signal. Cleaning up..."

    # Return all stranded batches to pending queue
    for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
        if [[ -n "$batch_dir" ]]; then
            log_warning "Returning stranded task batch from ${batch_dir} to pending queue..."
            ssh "$OTRERA_HOST" "find ${batch_dir} -name '*.task' -exec mv {} ${DPIXEL_PENDING_DIR}/ \\; 2>/dev/null"
            ssh "$OTRERA_HOST" "rmdir ${batch_dir} 2>/dev/null"
        fi
    done

    # Kill all worker processes
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then kill_process_tree "$pid"; fi
    done
    
    # Kill all upload worker processes
    for pid in "${UPLOAD_WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then kill_process_tree "$pid"; fi
    done

    # Clean up local scratch
    log_warning "Cleaning up local scratch directory: ${LOCAL_SCRATCH_DIR}"
    rm -rf "${LOCAL_SCRATCH_DIR}"

    # Close SSH control master
    if [[ -n "${SSH_CONTROL_PATH:-}" ]]; then
        ssh -O exit -o ControlPath="$SSH_CONTROL_PATH" "$OTRERA_HOST" 2>/dev/null || true
    fi

    log_warning "All processes terminated. Exiting."
    exit 1
}

trap cleanup SIGINT SIGTERM

# Function to scan for existing processed data and create upload jobs
initialize_upload_queue() {
    log_info "Scanning for existing processed d-pixels that need uploading..."
    
    # Look for any existing processed data in the d_pixels directory
    if [ -d "$LOCAL_D_PIXEL_DIR" ]; then
        local upload_count=0
        
        # Find all data_processed directories
        find "$LOCAL_D_PIXEL_DIR" -name "data_processed" -type d | while read -r data_dir; do
            # Extract year and grid_id from path
            # Path structure: LOCAL_D_PIXEL_DIR/year/grid_id/data_processed
            local parent_dir=$(dirname "$data_dir")
            local grid_id=$(basename "$parent_dir")
            local year=$(basename "$(dirname "$parent_dir")")
            
            # Check if this directory has the expected 9 NPY files
            local npy_count=$(find "$data_dir" -name "*.npy" -type f | wc -l)
            if [ "$npy_count" -eq 9 ]; then
                # Create upload job
                local upload_metadata="{
                    \"grid_id\": \"$grid_id\",
                    \"year\": \"$year\",
                    \"local_data_dir\": \"$data_dir\",
                    \"task_name\": \"${grid_id}_${year}.task\",
                    \"timestamp\": $(date +%s),
                    \"recovered\": true
                }"
                
                local upload_job_file="${LOCAL_UPLOAD_QUEUE_DIR}/${grid_id}_${year}_recovered_$(date +%s%N).upload"
                echo "$upload_metadata" > "$upload_job_file"
                ((upload_count++))
                
                log_info "Created upload job for recovered data: ${grid_id}/${year}"
            else
                log_warning "Skipping incomplete data directory: $data_dir (found $npy_count NPY files, expected 9)"
            fi
        done
        
        if [ $upload_count -gt 0 ]; then
            log_success "Created $upload_count upload jobs for existing processed data"
        else
            log_info "No existing processed data found that needs uploading"
        fi
    fi
}

# Atomic task claiming function
claim_tasks_atomically() {
    local batch_size=$1
    local batch_dir=$2
    local claimed_count=0
    
    log_info "Attempting to atomically claim up to ${batch_size} tasks..."
    
    # Create remote batch directory
    run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "mkdir -p ${batch_dir}" || return 1
    
    # Use a single SSH connection to minimize overhead
    claimed_count=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "
        cd ${DPIXEL_PENDING_DIR} 2>/dev/null || exit 1
        claimed=0
        # Sort by year (newer first) for priority
        for task in \$(find . -maxdepth 1 -name '*.task' -type f -printf '%f\n' 2>/dev/null | sort -t_ -k4 -nr); do
            [ -f \"\$task\" ] || continue
            if mv \"\$task\" ${batch_dir}/ 2>/dev/null; then
                claimed=\$((claimed + 1))
                [ \$claimed -ge ${batch_size} ] && break
            fi
        done
        echo \$claimed
    " 2>/dev/null || echo "0")
    
    echo "$claimed_count"
}

# Function to finalize a batch
finalize_batch() {
    local batch_dir=$1
    local batch_id="${CURRENT_BATCH_DIRS[$batch_dir]}"
    
    log_info "Finalizing batch ${batch_id} (${batch_dir})..."
    
    # Process completed tasks
    local done_count=0
    local done_errors=0
    # Use find to safely handle the case where no files exist
    while IFS= read -r -d '' task_file; do
        local task_basename=$(basename "$task_file")
        
        # Try to move from batch dir to done dir on remote
        if run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "
            if [ -f ${batch_dir}/${task_basename} ]; then
                mv ${batch_dir}/${task_basename} ${DPIXEL_DONE_DIR}/ 2>/dev/null
            elif [ -f ${DPIXEL_DONE_DIR}/${task_basename} ]; then
                exit 0  # Already moved
            else
                exit 1  # File not found
            fi
        "; then
            rm -f "$task_file"
            ((done_count++))
        else
            ((done_errors++))
            log_warning "Could not finalize ${task_basename}"
            rm -f "$task_file"
        fi
    done < <(find "${LOCAL_DONE_DIR}" -name "*.task" -type f -print0 2>/dev/null)
    
    if [ $done_count -gt 0 ]; then
        log_info "Successfully finalized ${done_count} completed tasks"
    fi

    # Process failed tasks
    local failed_count=0
    while IFS= read -r -d '' task_file; do
        local task_basename=$(basename "$task_file")
        
        if run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "
            if [ -f ${batch_dir}/${task_basename} ]; then
                mv ${batch_dir}/${task_basename} ${DPIXEL_FAILED_DIR}/ 2>/dev/null
            else
                exit 0
            fi
        "; then
            rm -f "$task_file"
            ((failed_count++))
        fi
    done < <(find "${LOCAL_FAILED_DIR}" -name "*.task" -type f -print0 2>/dev/null)
    
    if [ $failed_count -gt 0 ]; then
        log_info "Moved ${failed_count} failed tasks to remote 'failed' folder"
    fi
    
    # Clean up empty batch directory
    remaining_tasks=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "ls ${batch_dir}/*.task 2>/dev/null | wc -l" || echo "0")
    if [ "$remaining_tasks" -eq 0 ]; then
        run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "rmdir ${batch_dir} 2>/dev/null"
        log_info "Removed empty batch directory ${batch_dir}"
        unset CURRENT_BATCH_DIRS["$batch_dir"]
    fi
}

log_header "GLOBAL D-PIXEL GENERATION (Multi-Cluster Pipeline)"
log_info "Starting at $(date)"
log_info "Node: ${NODE_NAME}, Job ID: ${JOB_ID}"
log_info "Local scratch space: ${LOCAL_SCRATCH_DIR}"

log_header "SETUP & PRE-FLIGHT CHECKS"

# Verify required files
log_info "Checking required files..."
required_files=(
    "${SCRIPT_SOURCE_DIR}/s1_s2_downloader.sh"
    "${SCRIPT_SOURCE_DIR}/s1_s2_stacker.sh"
    "${SCRIPT_SOURCE_DIR}/s1_fast_processor.py"
    "${SCRIPT_SOURCE_DIR}/s2_fast_processor.py"
    "${SCRIPT_SOURCE_DIR}/s1_stack"
    "${SCRIPT_SOURCE_DIR}/s2_stack"
)

missing_files=()
for file in "${required_files[@]}"; do
    [[ ! -f "$file" ]] && missing_files+=("$(basename "$file")")
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    log_error "Missing required files: ${missing_files[*]}"
    exit 1
fi
log_success "All required files present"

# Setup SSH control master
log_info "Setting up SSH connection to Otrera..."
SSH_CONTROL_PATH=$(setup_ssh_control_master "$OTRERA_HOST")
log_success "SSH control master established"

# Debug: Check if we can access the remote directory
log_info "Checking remote pending directory contents..."
PENDING_CHECK=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "ls -la ${DPIXEL_PENDING_DIR}/ 2>&1 | head -10")
log_info "Remote pending directory sample: $PENDING_CHECK"
PENDING_COUNT_DEBUG=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "find ${DPIXEL_PENDING_DIR} -name '*.task' -type f | wc -l 2>&1")
log_info "Debug pending count: $PENDING_COUNT_DEBUG"

# Ensure remote directories exist
log_info "Ensuring remote task queue directories exist..."
run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "mkdir -p ${DPIXEL_PENDING_DIR} ${DPIXEL_PROCESSING_DIR} ${DPIXEL_DONE_DIR} ${DPIXEL_FAILED_DIR}"
log_success "Remote directories ready"

# Initialize upload queue with any existing processed data
initialize_upload_queue

# Upload worker process function
start_upload_worker() {
    local worker_id=$1
    local worker_name="${UNIQUE_JOB_ID}_upload_worker${worker_id}"
    
    log_info "Starting upload worker: ${worker_name}"
    local log_file="${NODE_LOG_DIR}/upload_worker_${worker_id}.log"
    
    # Create control file for this worker
    local control_file="${LOCAL_SCRATCH_DIR}/upload_worker_${worker_name}.control"
    echo "running" > "$control_file"
    
    $PYTHON_ENV -u - > "$log_file" 2>&1 <<EOF &
import os, sys, time, json, subprocess, shutil, traceback, logging
from pathlib import Path

# Configuration
WORKER_NAME = "${worker_name}"
WORKER_ID = ${worker_id}
CONTROL_FILE = "${control_file}"

# Paths
LOCAL_UPLOAD_QUEUE_DIR = Path("${LOCAL_UPLOAD_QUEUE_DIR}")
LOCAL_DONE_DIR = Path("${LOCAL_DONE_DIR}")
LOCAL_FAILED_DIR = Path("${LOCAL_FAILED_DIR}")

# Remote paths
OTRERA_HOST = "${OTRERA_HOST}"
OTRERA_D_PIXEL_PATH = "${OTRERA_D_PIXEL_PATH}"
SSH_CONTROL_PATH = "${SSH_CONTROL_PATH}"

# Setup logging
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(levelname)s - [{WORKER_NAME}] - %(message)s', stream=sys.stdout)

def should_continue():
    try:
        with open(CONTROL_FILE, 'r') as f:
            return f.read().strip() == "running"
    except: return False

def run_command(cmd, check=True):
    logging.debug(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(f"Command failed: {cmd}\\nSTDERR: {result.stderr}")
    return result

def run_ssh_command(cmd, check=True):
    full_cmd = f"ssh -o ControlPath={SSH_CONTROL_PATH} {OTRERA_HOST} '{cmd}'"
    return run_command(full_cmd, check)

def get_upload_job():
    try:
        # Look for .upload files in the upload queue
        upload_files = [f for f in LOCAL_UPLOAD_QUEUE_DIR.iterdir() if f.is_file() and f.name.endswith('.upload')]
        if not upload_files:
            return None
        
        # Get the oldest upload job
        upload_file = min(upload_files, key=lambda f: f.stat().st_mtime)
        
        # Load upload metadata
        with open(upload_file, 'r') as f:
            metadata = json.load(f)
        
        # Rename to indicate processing
        processing_file = upload_file.with_suffix('.uploading')
        upload_file.rename(processing_file)
        
        return processing_file, metadata
    except:
        return None

def process_upload(upload_file, metadata):
    grid_id = metadata['grid_id']
    year = metadata['year']
    local_data_dir = Path(metadata['local_data_dir'])
    task_name = metadata['task_name']
    
    logging.info(f"Starting upload for {grid_id}/{year}")
    
    try:
        # Create remote directory
        remote_dir = f"{OTRERA_D_PIXEL_PATH}/{year}/{grid_id}"
        run_ssh_command(f"mkdir -p {remote_dir}")
        
        # Transfer files
        rsync_cmd = f"rsync -azP {local_data_dir}/*.npy {OTRERA_HOST}:{remote_dir}/"
        run_command(rsync_cmd)
        
        # Verify transfer
        remote_count = run_ssh_command(f"ls {remote_dir}/*.npy 2>/dev/null | wc -l").stdout.strip()
        if int(remote_count) != 9:
            raise Exception(f"Transfer verification failed: {remote_count} files on remote")
        
        # Clean up local data
        shutil.rmtree(local_data_dir.parent)  # Remove the year/grid_id directory
        
        # Mark task as done
        (LOCAL_DONE_DIR / task_name).touch()
        upload_file.unlink()
        
        logging.info(f"Successfully uploaded {grid_id}/{year}")
        return True
        
    except Exception as e:
        logging.error(f"Upload failed for {grid_id}/{year}: {str(e)}")
        # Mark task as failed
        (LOCAL_FAILED_DIR / task_name).touch()
        upload_file.unlink()
        return False

def main():
    logging.info("Upload worker started. Will run indefinitely until stopped.")
    
    while should_continue():
        job = get_upload_job()
        if job:
            upload_file, metadata = job
            try:
                process_upload(upload_file, metadata)
            except Exception:
                logging.error(f"Failed to process upload\\n{traceback.format_exc()}")
        else:
            # No timeout - just wait and try again
            time.sleep(1)
    
    logging.info("Upload worker shutting down.")

if __name__ == "__main__": main()
EOF
    
    local pid=$!
    UPLOAD_WORKER_PIDS+=($pid)
    log_success "Started upload worker ${worker_id} with PID $pid, logging to $log_file"
}

# Worker process function
start_worker() {
    local worker_id=$1
    local worker_name="${UNIQUE_JOB_ID}_worker${worker_id}"
    
    log_info "Starting worker: ${worker_name}"
    local log_file="${NODE_LOG_DIR}/worker_${worker_id}.log"
    
    # Create control file for this worker
    local control_file="${LOCAL_SCRATCH_DIR}/worker_${worker_name}.control"
    echo "running" > "$control_file"
    
    # Note: Using EOF without quotes to allow variable expansion
    $PYTHON_ENV -u - > "$log_file" 2>&1 <<EOF &
import os, sys, time, random, json, subprocess, shutil, traceback, logging
from pathlib import Path

# Configuration
WORKER_NAME = "${worker_name}"
WORKER_ID = ${worker_id}
CONTROL_FILE = "${control_file}"

# Paths
LOCAL_PENDING_DIR = Path("${LOCAL_PENDING_DIR}")
LOCAL_PROCESSING_DIR = Path("${LOCAL_PROCESSING_DIR}")
LOCAL_DONE_DIR = Path("${LOCAL_DONE_DIR}")
LOCAL_FAILED_DIR = Path("${LOCAL_FAILED_DIR}")
LOCAL_TIFF_DIR = Path("${LOCAL_TIFF_DIR}")
LOCAL_D_PIXEL_DIR = Path("${LOCAL_D_PIXEL_DIR}")
LOCAL_UPLOAD_QUEUE_DIR = Path("${LOCAL_UPLOAD_QUEUE_DIR}")
SCRIPT_SOURCE_DIR = Path("${SCRIPT_SOURCE_DIR}")

# Remote paths
OTRERA_HOST = "${OTRERA_HOST}"
OTRERA_TIFF_PATH = "${OTRERA_TIFF_PATH}"
SSH_CONTROL_PATH = "${SSH_CONTROL_PATH}"

# Setup logging
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(levelname)s - [{WORKER_NAME}] - %(message)s', stream=sys.stdout)

def should_continue():
    try:
        with open(CONTROL_FILE, 'r') as f:
            return f.read().strip() == "running"
    except: return False

def run_command(cmd, check=True):
    logging.debug(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(f"Command failed: {cmd}\\nSTDERR: {result.stderr}")
    return result

def run_ssh_command(cmd, check=True):
    full_cmd = f"ssh -o ControlPath={SSH_CONTROL_PATH} {OTRERA_HOST} '{cmd}'"
    return run_command(full_cmd, check)

def get_local_task():
    try:
        pending_files = [f for f in LOCAL_PENDING_DIR.iterdir() if f.is_file() and f.name.endswith('.task')]
        if not pending_files: return None
        task_file = random.choice(pending_files)
        processing_path = LOCAL_PROCESSING_DIR / f"{WORKER_NAME}_{task_file.name}"
        shutil.move(str(task_file), str(processing_path))
        logging.info(f"Acquired local task: {task_file.name}")
        return processing_path
    except: return None

def process_task(task_path):
    # Extract task info
    original_task_name = task_path.name
    if original_task_name.startswith(f"{WORKER_NAME}_"):
        original_task_name = original_task_name[len(f"{WORKER_NAME}_"):]
    
    parts = original_task_name.replace('.task', '').split('_')
    year = parts[-1]
    grid_id = '_'.join(parts[:-1])
    
    logging.info(f"Processing task: grid={grid_id}, year={year}")
    
    # Check if grid TIFF is already downloaded
    local_tiff_file = LOCAL_TIFF_DIR / f"{grid_id}.tiff"
    if not local_tiff_file.exists():
        logging.info(f"Downloading grid TIFF: {grid_id}.tiff")
        remote_tiff = f"{OTRERA_TIFF_PATH}/{grid_id}.tiff"
        rsync_cmd = f"rsync -azP {OTRERA_HOST}:{remote_tiff} {local_tiff_file}"
        run_command(rsync_cmd)
    
    # Setup processing directories
    local_work_dir = LOCAL_D_PIXEL_DIR / year / grid_id
    local_work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processing script
    process_script = local_work_dir / "process_dpixel.sh"
    with open(process_script, 'w') as f:
        f.write(f'''#!/bin/bash
set -e

export TILE_NAME="{grid_id}"
export TIFF_FILE="{local_tiff_file}"
export LOCAL_DIR="{local_work_dir}"
export YEAR="{year}"
export SCRIPT_SOURCE_DIR="{SCRIPT_SOURCE_DIR}"
export PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"

cd "\$SCRIPT_SOURCE_DIR"

# Step 1: Download Sentinel-1
cp s1_s2_downloader.sh "\$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|INPUT_TIFF=.*|INPUT_TIFF=\\"\$TIFF_FILE\\"|" "\$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|OUT_DIR=.*|OUT_DIR=\\"\$LOCAL_DIR\\"|" "\$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|S1_ENABLED=false|S1_ENABLED=true|" "\$LOCAL_DIR/downloader_temp_s1.sh"
sed -i "s|S2_ENABLED=true|S2_ENABLED=false|" "\$LOCAL_DIR/downloader_temp_s1.sh"

bash "\$LOCAL_DIR/downloader_temp_s1.sh"

# Step 2: Download Sentinel-2
cp s1_s2_downloader.sh "\$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|INPUT_TIFF=.*|INPUT_TIFF=\\"\$TIFF_FILE\\"|" "\$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|OUT_DIR=.*|OUT_DIR=\\"\$LOCAL_DIR\\"|" "\$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|S1_ENABLED=true|S1_ENABLED=false|" "\$LOCAL_DIR/downloader_temp_s2.sh"
sed -i "s|S2_ENABLED=false|S2_ENABLED=true|" "\$LOCAL_DIR/downloader_temp_s2.sh"

bash "\$LOCAL_DIR/downloader_temp_s2.sh"

# Step 3: Stack the data
cp "\$SCRIPT_SOURCE_DIR/s1_s2_stacker.sh" "\$LOCAL_DIR/stacker_temp.sh"
sed -i "s|BASE_DIR=.*|BASE_DIR=\\"\$LOCAL_DIR\\"|" "\$LOCAL_DIR/stacker_temp.sh"

cp "\$SCRIPT_SOURCE_DIR/s1_stack" "\$LOCAL_DIR/"
cp "\$SCRIPT_SOURCE_DIR/s2_stack" "\$LOCAL_DIR/"
chmod +x "\$LOCAL_DIR/s1_stack" "\$LOCAL_DIR/s2_stack"

cd "\$LOCAL_DIR"
bash "./stacker_temp.sh"

# Verify output
npy_count=\$(find "\$LOCAL_DIR/data_processed" -name "*.npy" -type f 2>/dev/null | wc -l)
if [ "\$npy_count" -ne 9 ]; then
    echo "ERROR: Expected 9 NPY files but found \$npy_count"
    exit 1
fi

echo "Processing completed successfully"
''')
    
    os.chmod(process_script, 0o755)
    
    # Run processing
    result = run_command(str(process_script), check=False)
    if result.returncode != 0:
        raise Exception(f"Processing failed with code {result.returncode}")
    
    # Verify we have the expected output
    npy_files = list((local_work_dir / "data_processed").glob("*.npy"))
    if len(npy_files) != 9:
        raise Exception(f"Expected 9 NPY files, found {len(npy_files)}")
    
    # Create upload job instead of doing it synchronously
    upload_metadata = {
        'grid_id': grid_id,
        'year': year,
        'local_data_dir': str(local_work_dir / "data_processed"),
        'task_name': original_task_name,
        'timestamp': time.time()
    }
    
    upload_job_file = LOCAL_UPLOAD_QUEUE_DIR / f"{grid_id}_{year}_{int(time.time()*1000)}.upload"
    with open(upload_job_file, 'w') as f:
        json.dump(upload_metadata, f)
    
    logging.info(f"Created upload job for {grid_id}/{year}")
    
    # Remove the task from processing (don't mark as done until upload completes)
    task_path.unlink()
    
    return original_task_name

def main():
    logging.info("Worker started. Waiting for tasks...")
    idle_count = 0
    
    while should_continue():
        task_path = get_local_task()
        if task_path:
            idle_count = 0
            try:
                process_task(task_path)
                logging.info(f"Successfully processed task {task_path.name}")
            except Exception:
                logging.error(f"Failed to process task {task_path.name}\\n{traceback.format_exc()}")
                original_task_name = task_path.name
                if original_task_name.startswith(f"{WORKER_NAME}_"):
                    original_task_name = original_task_name[len(f"{WORKER_NAME}_"):]
                shutil.move(str(task_path), str(LOCAL_FAILED_DIR / original_task_name))
        else:
            idle_count += 1
            if idle_count > 1800:  # 30 minutes
                logging.info("No tasks for 30 minutes. Exiting.")
                break
            time.sleep(1)
    
    logging.info("Worker shutting down.")

if __name__ == "__main__": main()
EOF
    
    local pid=$!
    WORKER_PIDS+=($pid)
    log_success "Started worker ${worker_id} with PID $pid, logging to $log_file"
}

log_header "STARTING D-PIXEL GENERATION WORKERS"
log_info "Starting $PROCESSES_PER_NODE worker processes"

for ((i=0; i<PROCESSES_PER_NODE; i++)); do
    start_worker $i
    sleep 0.5
done

log_header "STARTING UPLOAD WORKERS"
log_info "Starting $UPLOAD_WORKERS upload worker processes"

for ((i=0; i<UPLOAD_WORKERS; i++)); do
    start_upload_worker $i
    sleep 0.5
done

# Main control loop
log_header "ENTERING MAIN CONTROL LOOP"
BATCH_NUMBER=0
NO_TASKS_COUNT=0
MAX_NO_TASKS_COUNT=10
LAST_FINALIZE_TIME=$(date +%s)
FINALIZE_INTERVAL=900  # 15 minutes

while true; do
    # Check pending tasks with better error handling
    REMOTE_PENDING_COUNT=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "find ${DPIXEL_PENDING_DIR} -name '*.task' -type f 2>/dev/null | wc -l" 2>&1)
    if [[ ! "$REMOTE_PENDING_COUNT" =~ ^[0-9]+$ ]]; then
        log_warning "Failed to get remote pending count. Error: $REMOTE_PENDING_COUNT"
        REMOTE_PENDING_COUNT=0
        # Try alternative method
        REMOTE_PENDING_COUNT=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "ls ${DPIXEL_PENDING_DIR}/*.task 2>/dev/null | wc -l" 2>&1 || echo "0")
        log_info "Alternative count result: $REMOTE_PENDING_COUNT"
    fi
    
    LOCAL_PENDING_COUNT=$(find "${LOCAL_PENDING_DIR}" -name "*.task" -type f 2>/dev/null | wc -l)
    
    # Fetch new batch if needed
    if [ "$LOCAL_PENDING_COUNT" -lt 4 ] && [ "$REMOTE_PENDING_COUNT" -gt 0 ]; then
        BATCH_NUMBER=$((BATCH_NUMBER + 1))
        BATCH_ID="${UNIQUE_JOB_ID}_batch${BATCH_NUMBER}"
        REMOTE_BATCH_DIR="${DPIXEL_PROCESSING_DIR}/${BATCH_ID}"
        CURRENT_BATCH_DIRS["$REMOTE_BATCH_DIR"]="$BATCH_ID"
        
        log_header "FETCHING BATCH #${BATCH_NUMBER}"
        
        # Atomically claim tasks
        CLAIMED_COUNT=$(claim_tasks_atomically "$NODE_TASK_BATCH_SIZE" "$REMOTE_BATCH_DIR")
        
        if [ "$CLAIMED_COUNT" -gt 0 ]; then
            NO_TASKS_COUNT=0
            log_success "Successfully claimed ${CLAIMED_COUNT} tasks in batch #${BATCH_NUMBER}"
            
            # Get list of claimed tasks and move to local pending
            CLAIMED_TASKS=$(run_ssh_command "$OTRERA_HOST" "$SSH_CONTROL_PATH" "ls ${REMOTE_BATCH_DIR}/*.task 2>/dev/null")
            
            for task_path in $CLAIMED_TASKS; do
                task_basename=$(basename "$task_path")
                touch "${LOCAL_PENDING_DIR}/${task_basename}"
            done
            
            log_success "Added ${CLAIMED_COUNT} tasks to local queue"
        fi
    elif [ "$REMOTE_PENDING_COUNT" -eq 0 ]; then
        NO_TASKS_COUNT=$((NO_TASKS_COUNT + 1))
        if [ "$NO_TASKS_COUNT" -ge "$MAX_NO_TASKS_COUNT" ] && [ "$LOCAL_PENDING_COUNT" -eq 0 ]; then
            log_info "No more tasks available. Preparing to shut down..."
            break
        fi
    fi
    
    # Monitor worker status
    RUNNING_WORKERS=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    RUNNING_UPLOAD_WORKERS=0
    for pid in "${UPLOAD_WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_UPLOAD_WORKERS=$((RUNNING_UPLOAD_WORKERS + 1))
        fi
    done
    
    if [ "$RUNNING_WORKERS" -eq 0 ] && [ "$RUNNING_UPLOAD_WORKERS" -eq 0 ]; then
        log_warning "All worker processes have exited!"
        break
    fi
    
    # Periodically finalize batches
    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - LAST_FINALIZE_TIME)) -ge "$FINALIZE_INTERVAL" ]; then
        log_info "Running periodic batch finalization..."
        for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
            finalize_batch "$batch_dir"
        done
        LAST_FINALIZE_TIME=$CURRENT_TIME
    fi
    
    # Status update
    LOCAL_DONE_COUNT=$(find "${LOCAL_DONE_DIR}" -name "*.task" -type f 2>/dev/null | wc -l)
    LOCAL_FAILED_COUNT=$(find "${LOCAL_FAILED_DIR}" -name "*.task" -type f 2>/dev/null | wc -l)
    UPLOAD_QUEUE_COUNT=$(find "${LOCAL_UPLOAD_QUEUE_DIR}" -name "*.upload" -type f 2>/dev/null | wc -l)
    UPLOADING_COUNT=$(find "${LOCAL_UPLOAD_QUEUE_DIR}" -name "*.uploading" -type f 2>/dev/null | wc -l)
    ELAPSED=$(($(date +%s) - SCRIPT_START))
    
    log_info "STATUS | Batch: #${BATCH_NUMBER} | Workers: ${RUNNING_WORKERS}/${PROCESSES_PER_NODE} | Upload Workers: ${RUNNING_UPLOAD_WORKERS}/${UPLOAD_WORKERS} | Local (Pend/Done/Failed): ${LOCAL_PENDING_COUNT}/${LOCAL_DONE_COUNT}/${LOCAL_FAILED_COUNT} | Upload Queue/Active: ${UPLOAD_QUEUE_COUNT}/${UPLOADING_COUNT} | Remote Pending: ${REMOTE_PENDING_COUNT} | Elapsed: $(format_time "$ELAPSED")"
    
    sleep 30
done

# Shutdown sequence
log_header "INITIATING SHUTDOWN SEQUENCE"

# Signal workers to stop
log_info "Signaling all workers to stop..."
for ((i=0; i<PROCESSES_PER_NODE; i++)); do
    control_file="${LOCAL_SCRATCH_DIR}/worker_${UNIQUE_JOB_ID}_worker${i}.control"
    echo "stop" > "$control_file"
done

for ((i=0; i<UPLOAD_WORKERS; i++)); do
    control_file="${LOCAL_SCRATCH_DIR}/upload_worker_${UNIQUE_JOB_ID}_upload_worker${i}.control"
    echo "stop" > "$control_file"
done

# Wait for workers to finish
log_info "Waiting for all workers to finish current tasks..."
WAIT_COUNT=0
while [ "$WAIT_COUNT" -lt 60 ]; do  # 30 minutes max
    RUNNING_WORKERS=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_WORKERS=$((RUNNING_WORKERS + 1))
        fi
    done
    
    RUNNING_UPLOAD_WORKERS=0
    for pid in "${UPLOAD_WORKER_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_UPLOAD_WORKERS=$((RUNNING_UPLOAD_WORKERS + 1))
        fi
    done
    
    if [ "$RUNNING_WORKERS" -eq 0 ] && [ "$RUNNING_UPLOAD_WORKERS" -eq 0 ]; then break; fi
    
    UPLOAD_QUEUE_COUNT=$(find "${LOCAL_UPLOAD_QUEUE_DIR}" -name "*.upload" -type f 2>/dev/null | wc -l)
    UPLOADING_COUNT=$(find "${LOCAL_UPLOAD_QUEUE_DIR}" -name "*.uploading" -type f 2>/dev/null | wc -l)
    
    log_info "Waiting for shutdown - Workers: ${RUNNING_WORKERS}, Upload Workers: ${RUNNING_UPLOAD_WORKERS}, Upload Queue: ${UPLOAD_QUEUE_COUNT}, Uploading: ${UPLOADING_COUNT}"
    
    sleep 30
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# Force kill remaining workers
for pid in "${WORKER_PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        kill_process_tree "$pid"
    fi
done

for pid in "${UPLOAD_WORKER_PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        kill_process_tree "$pid"
    fi
done

# Final batch finalization
log_info "Finalizing all remaining batches..."
for batch_dir in "${!CURRENT_BATCH_DIRS[@]}"; do
    finalize_batch "$batch_dir"
done

# Cleanup
log_info "Cleaning up local scratch directory: ${LOCAL_SCRATCH_DIR}"
rm -rf "${LOCAL_SCRATCH_DIR}"

# Close SSH control master
if [[ -n "${SSH_CONTROL_PATH:-}" ]]; then
    ssh -O exit -o ControlPath="$SSH_CONTROL_PATH" "$OTRERA_HOST" 2>/dev/null || true
fi

TOTAL_TIME=$(($(date +%s) - SCRIPT_START))
log_info "Total execution time: $(format_time $TOTAL_TIME)"
log_success "All processing finished at $(date)"