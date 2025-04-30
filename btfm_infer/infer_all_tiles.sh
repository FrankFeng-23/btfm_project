#!/bin/bash
#
# infer_all_tiles.sh - Hybrid CPU/GPU inference script for multi-tile processing
#
# Processes a set of tiles using both CPU and GPU resources according to a specified ratio.
# Automatically skips already processed tiles and distributes workload efficiently.
#
# Usage:
#   bash infer_all_tiles.sh

###############This needs to be modified to your environment###############
# Set default values
# This is the main directory where all processed and raw data will be stored
BASE_DATA_DIR="/scratch/zf281/jovana"

# Python environment that has the required packages installed
export PYTHON_ENV="/maps/zf281/btfm-training-frank/venv/bin/python"

# CPU:GPU split ratio
# The script supports simultaneous inference using both CPU and GPU. This ratio specifies the proportion of retiled_patches each device will handle. 
# Default is 1:1 (even split). For GPU-only inference, set to 0:1.
CPU_GPU_SPLIT="1:1"  # Format: CPU:GPU ratio

# Maximum number of concurrent processes
MAX_CONCURRENT_PROCESSES_CPU=20

# CPU cores to use
# Calculate available CPU cores
TOTAL_CPU_CORES=$(nproc)  # Get total number of CPU cores
AVAILABLE_CORES=$((TOTAL_CPU_CORES / 2)) # Use 50% of the cores

# Define CPU batch size and workers
CPU_BATCH_SIZE=256  # Smaller for CPU
CPU_NUM_WORKERS=0   # Let main thread do the loading

# Maximum number of concurrent GPU processes; this value usually equals the number of GPUs on the device.
MAX_CONCURRENT_PROCESSES_GPU=1  # Number of GPUs to use

# Define GPU batch size and workers
GPU_BATCH_SIZE=1024  # Larger for GPU, if this takes too much memory, reduce it
GPU_NUM_WORKERS=4    # More workers for GPU

# Other settings
VERBOSE_GPU=true    # Enable detailed GPU logging
LOG_INTERVAL=5      # Log every N batches
###########################################################################

TILES_DIR="${BASE_DATA_DIR}/retiled_d_pixel"
OUTPUT_DIR="${BASE_DATA_DIR}/representation_retiled"
CONFIG_FILE="configs/multi_tile_infer_config.py"
PYTHON_SCRIPT="src/multi_tile_infer.py"

# Path to the checkpoint file for the model
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250408_101211.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cpu-gpu-split)
            CPU_GPU_SPLIT="$2"
            shift 2
            ;;
        --max-cpu)
            MAX_CONCURRENT_PROCESSES_CPU="$2"
            shift 2
            ;;
        --max-gpu)
            MAX_CONCURRENT_PROCESSES_GPU="$2"
            shift 2
            ;;
        --tiles-dir)
            TILES_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --verbose-gpu)
            VERBOSE_GPU="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

# Make sure all logging functions output to stderr
log_header() {
    echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}" >&2
}

log_info() {
    echo -e "${CYAN}[INFO]${RESET} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1" >&2
}

log_debug() {
    echo -e "${MAGENTA}[DEBUG]${RESET} $1" >&2
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

calculate_time() {
    local start_time=$1
    local end_time=$2
    local duration=$((end_time - start_time))
    format_time $duration
}

# Function to monitor GPU usage
monitor_gpu_usage() {
    local gpu_id=$1
    local monitor_interval=60  # seconds
    local monitor_pid_file="logs/gpu_monitor_${gpu_id}.pid"
    
    log_info "Starting GPU $gpu_id monitoring (every ${monitor_interval}s)"
    
    # Run GPU monitor in background
    (
        while true; do
            timestamp=$(date +"%Y-%m-%d %H:%M:%S")
            if command -v nvidia-smi &> /dev/null; then
                echo "[$timestamp] GPU $gpu_id monitoring:" >> "logs/gpu_monitor_${gpu_id}.log"
                nvidia-smi -i $gpu_id --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv >> "logs/gpu_monitor_${gpu_id}.log"
                echo "" >> "logs/gpu_monitor_${gpu_id}.log"
            else
                echo "[$timestamp] nvidia-smi not available" >> "logs/gpu_monitor_${gpu_id}.log"
            fi
            sleep $monitor_interval
        done
    ) &
    
    monitor_pid=$!
    echo $monitor_pid > "$monitor_pid_file"
    log_debug "GPU $gpu_id monitor started with PID $monitor_pid"
    
    return $monitor_pid
}

# Function to stop GPU monitoring
stop_gpu_monitoring() {
    local gpu_id=$1
    local monitor_pid_file="logs/gpu_monitor_${gpu_id}.pid"
    
    if [ -f "$monitor_pid_file" ]; then
        local monitor_pid=$(cat "$monitor_pid_file")
        if ps -p $monitor_pid > /dev/null; then
            kill $monitor_pid 2>/dev/null
            log_debug "Stopped GPU $gpu_id monitoring (PID $monitor_pid)"
        fi
        rm -f "$monitor_pid_file"
    fi
}

log_info "Total CPU cores: $TOTAL_CPU_CORES, Using: $AVAILABLE_CORES"

# Parse CPU:GPU split ratio
IFS=':' read -r CPU_RATIO GPU_RATIO <<< "$CPU_GPU_SPLIT"
CPU_RATIO=${CPU_RATIO:-1}
GPU_RATIO=${GPU_RATIO:-1}
TOTAL_RATIO=$((CPU_RATIO + GPU_RATIO))
log_info "CPU:GPU split ratio = $CPU_RATIO:$GPU_RATIO (total: $TOTAL_RATIO)"

# Arrays to store process PIDs
declare -a CPU_PIDS
declare -a GPU_PIDS
declare -a GPU_MONITOR_PIDS

# Function to kill process and all its children
kill_process_tree() {
    local pid=$1
    log_debug "Killing process tree for PID: $pid"
    
    # Get all child processes
    local children=$(pgrep -P $pid)
    
    # Kill children first (recursively)
    for child in $children; do
        kill_process_tree $child
    done
    
    # Kill the process
    if ps -p $pid > /dev/null; then
        log_debug "Terminating process $pid"
        kill -9 $pid 2>/dev/null || true
    fi
}

# Function to kill all GPU processes
kill_gpu_processes() {
    log_debug "Checking for remaining GPU processes..."
    
    # Only if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        # Get all Python processes using GPU
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || echo ""); do
            # Check if this is one of our processes or a child of our processes
            for our_pid in "${GPU_PIDS[@]}"; do
                if [ "$pid" = "$our_pid" ] || pgrep -P $our_pid | grep -q $pid; then
                    log_debug "Killing GPU process with PID: $pid"
                    kill -9 $pid 2>/dev/null || true
                fi
            done
        done
    else
        log_debug "nvidia-smi not found, skipping GPU process cleanup"
    fi
}

# Function to clean up processes on exit
cleanup() {
    log_warning "Received interrupt signal. Cleaning up processes..."
    
    # Kill CPU processes
    for pid in "${CPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill_process_tree $pid
        fi
    done
    
    # Kill GPU processes
    for pid in "${GPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill_process_tree $pid
        fi
    done
    
    # Then make sure no GPU processes are left
    kill_gpu_processes
    
    # Stop GPU monitoring
    for ((i=0; i<MAX_CONCURRENT_PROCESSES_GPU; i++)); do
        if [[ ${GPU_MONITOR_PIDS[$i]} -gt 0 ]]; then
            stop_gpu_monitoring $i
        fi
    done
    
    log_warning "All processes terminated. Exiting."
    exit 1
}

# Set up trap for Ctrl+C (SIGINT)
trap cleanup SIGINT SIGTERM

log_header "SETUP DIRECTORIES"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "src/tile_lists"
mkdir -p "logs"  # Make sure logs directory exists
log_success "Created necessary directories"

log_header "SCANNING TILES"
log_info "Tile directory: $TILES_DIR"
log_info "Output directory: $OUTPUT_DIR"

# Find all tile directories
TILES=($(find "$TILES_DIR" -maxdepth 1 -mindepth 1 -type d | sort))
NUM_TILES=${#TILES[@]}

if [[ $NUM_TILES -eq 0 ]]; then
    log_error "No tiles found in $TILES_DIR"
    exit 1
fi

log_success "Found $NUM_TILES tiles total"

# Show a few example tiles
log_info "Sample tiles:"
for ((i=0; i<3 && i<NUM_TILES; i++)); do
    echo "  - $(basename "${TILES[$i]}")" >&2
done
[[ $NUM_TILES -gt 3 ]] && echo "  - ..." >&2

log_header "FILTERING TILES"

# Filter out already processed tiles
TILES_TO_PROCESS=()
ALREADY_PROCESSED=0

for tile_path in "${TILES[@]}"; do
    tile_name=$(basename "$tile_path")
    output_path="$OUTPUT_DIR/${tile_name}.npy"
    
    if [ -f "$output_path" ]; then
        ALREADY_PROCESSED=$((ALREADY_PROCESSED + 1))
    else
        TILES_TO_PROCESS+=("$tile_path")
    fi
done

NUM_TILES_TO_PROCESS=${#TILES_TO_PROCESS[@]}
log_success "Found $NUM_TILES_TO_PROCESS tiles to process (skipping $ALREADY_PROCESSED already done)"

if [ $NUM_TILES_TO_PROCESS -eq 0 ]; then
    log_success "No tiles need processing. All done!"
    exit 0
fi

log_header "RESOURCE ALLOCATION"

# Calculate split based on ratio, handling special case of 0:x ratio
if [ "$CPU_RATIO" -eq 0 ]; then
    # All tiles go to GPU
    CPU_TILES_COUNT=0
    GPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
elif [ "$GPU_RATIO" -eq 0 ]; then
    # All tiles go to CPU
    CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
    GPU_TILES_COUNT=0
else
    # Normal split based on ratio
    CPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS * CPU_RATIO / TOTAL_RATIO ))
    GPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS - CPU_TILES_COUNT ))
fi

log_info "CPU will process $CPU_TILES_COUNT tiles, GPU will process $GPU_TILES_COUNT tiles"

# Prepare tile lists based on allocation
if [ $CPU_TILES_COUNT -gt 0 ]; then
    # Prepare CPU tiles list
    CPU_TILES=("${TILES_TO_PROCESS[@]:0:$CPU_TILES_COUNT}")
else
    # Empty CPU tiles array
    CPU_TILES=()
fi

if [ $GPU_TILES_COUNT -gt 0 ]; then
    # Prepare GPU tiles list
    GPU_TILES=("${TILES_TO_PROCESS[@]:$CPU_TILES_COUNT}")
else
    # Empty GPU tiles array
    GPU_TILES=()
fi

# Calculate CPU batch settings
if [ $CPU_TILES_COUNT -gt 0 ]; then
    NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
    THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
    if [ $THREADS_PER_CPU_PROCESS -lt 1 ]; then
        THREADS_PER_CPU_PROCESS=1
    fi
    
    log_info "Using $NUM_CPU_PROCESSES CPU processes with $THREADS_PER_CPU_PROCESS threads each"
    
else
    log_info "No tiles assigned to CPU processing"
    NUM_CPU_PROCESSES=0
fi

# Prepare GPU tiles lists
if [ $GPU_TILES_COUNT -gt 0 ]; then
    # Check if GPUs are available
    if command -v nvidia-smi &> /dev/null; then
        log_info "Checking available GPUs:"
        nvidia-smi -L
        
        # More detailed GPU info at start
        log_info "Current GPU status:"
        nvidia-smi
        
        AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
        if [ $AVAILABLE_GPUS -eq 0 ]; then
            log_warning "No GPUs found, moving all GPU tiles to CPU"
            CPU_TILES=("${TILES_TO_PROCESS[@]}")
            CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
            GPU_TILES_COUNT=0
            GPU_TILES=()
            
            # Recalculate CPU settings
            NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
            THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
            if [ $THREADS_PER_CPU_PROCESS -lt 1 ]; then
                THREADS_PER_CPU_PROCESS=1
            fi
            
            log_info "Updated to use $NUM_CPU_PROCESSES CPU processes with $THREADS_PER_CPU_PROCESS threads each"
        else
            NUM_GPU_PROCESSES=$(( MAX_CONCURRENT_PROCESSES_GPU < AVAILABLE_GPUS ? MAX_CONCURRENT_PROCESSES_GPU : AVAILABLE_GPUS ))
            # Get the GPU name without using a complex pipe
            GPU_NAME="GPU"
            if command -v nvidia-smi &> /dev/null; then
                GPU_NAME=$(nvidia-smi -L | head -n1 | cut -d'(' -f1 | cut -d: -f2 | xargs) 
                
                # Get detailed information for each GPU that will be used
                for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
                    log_info "GPU $i details:"
                    nvidia-smi -i $i --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
                done
            fi
            log_info "Using $NUM_GPU_PROCESSES GPU processes on $GPU_NAME"
            
            # Divide GPU tiles among GPU processes
            if [ $GPU_TILES_COUNT -gt 0 ]; then
                TILES_PER_GPU=$((GPU_TILES_COUNT / NUM_GPU_PROCESSES))
                REMAINDER=$((GPU_TILES_COUNT % NUM_GPU_PROCESSES))
                
                log_info "Each GPU will process approximately $TILES_PER_GPU tiles"
                
                # Create tile lists for each GPU
                for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
                    # Calculate start and end indices for this GPU
                    START=$((i * TILES_PER_GPU))
                    END=$(((i + 1) * TILES_PER_GPU))
                    
                    # Add one more tile to earlier GPUs if there's a remainder
                    if [ $i -lt $REMAINDER ]; then
                        END=$((END + 1))
                    fi
                    
                    # Calculate start index accounting for remainder distribution
                    if [ $i -ge $REMAINDER ]; then
                        START=$((START + REMAINDER))
                    else
                        START=$((START + i))
                    fi
                    
                    # Create tile list for this GPU
                    TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
                    
                    # Build JSON array of tiles
                    echo -n "[" > "$TILE_LIST"
                    for ((j=START; j<END && j<GPU_TILES_COUNT; j++)); do
                        # Add comma if not the first item
                        if [ $j -gt $START ]; then
                            echo -n "," >> "$TILE_LIST"
                        fi
                        echo -n "\"${GPU_TILES[$j]}\"" >> "$TILE_LIST"
                    done
                    echo "]" >> "$TILE_LIST"
                    
                    # Count tiles in the file rather than using jq
                    TILE_COUNT=$(grep -o "\"" "$TILE_LIST" | wc -l)
                    TILE_COUNT=$((TILE_COUNT / 2))  # Each tile has two quotes
                    log_info "Created tile list for GPU $i: $TILE_COUNT tiles"
                    
                    # Enhanced logging - show actual tiles
                    if [ "$VERBOSE_GPU" = "true" ]; then
                        log_debug "GPU $i tiles list content:"
                        cat "$TILE_LIST" | tr ',' '\n' | head -n 5
                        if [ $TILE_COUNT -gt 5 ]; then
                            echo "  ... and $((TILE_COUNT - 5)) more tiles" >&2
                        fi
                    else
                        log_debug "Tile list content sample: $(head -c 100 "$TILE_LIST")"
                    fi
                done
            else
                log_warning "No tiles to process on GPU!"
                NUM_GPU_PROCESSES=0
            fi

        fi
    else
        log_warning "nvidia-smi not found, moving all GPU tiles to CPU"
        CPU_TILES=("${TILES_TO_PROCESS[@]}")
        CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
        GPU_TILES_COUNT=0
        GPU_TILES=()
        
        # Recalculate CPU settings
        NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
        THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
        if [ $THREADS_PER_CPU_PROCESS -lt 1 ]; then
            THREADS_PER_CPU_PROCESS=1
        fi
        
        log_info "Updated to use $NUM_CPU_PROCESSES CPU processes with $THREADS_PER_CPU_PROCESS threads each"
    fi
else
    log_info "No tiles assigned to GPU processing"
    NUM_GPU_PROCESSES=0
fi

# Function to launch a single CPU process safely
start_cpu_process() {
    local tile_path="$1"
    local process_id="$2"
    local threads_per_process="$3"
    
    local tile_name=$(basename "$tile_path")
    
    # Clear any previous log file
    local log_file="logs/infer_cpu_${process_id}.log"
    > "$log_file"
    
    # Launch the process in the background directly
    $PYTHON_ENV "$PYTHON_SCRIPT" \
        --config "$CONFIG_FILE" \
        --mode "cpu" \
        --tile_path "$tile_path" \
        --process_id $process_id \
        --num_threads $threads_per_process \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $CPU_BATCH_SIZE \
        --num_workers $CPU_NUM_WORKERS \
        --log_interval $LOG_INTERVAL \
        --log_level "INFO" \
        --simplified_logging > "$log_file" 2>&1 &
    
    local pid=$!
    
    # Log the PID
    log_debug "Started CPU process $process_id for ${tile_name} with PID $pid"
    
    # Return the PID directly
    echo $pid
}

# Function to show progress for all running CPU processes
show_cpu_progress() {
    local running_pids=("$@")
    local num_running=${#running_pids[@]}
    
    if [ $num_running -eq 0 ]; then
        return
    fi
    
    log_info "CPU process status for $num_running running processes:"
    
    for pid in "${running_pids[@]}"; do
        local log_file="logs/infer_cpu_*.log"
        log_file=$(ls logs/infer_cpu_*.log 2>/dev/null | grep -l "PID $pid" | head -n1 || echo "")
        
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            # Get the process ID from log file name
            local proc_id=$(echo "$log_file" | sed 's/.*infer_cpu_\([0-9]*\)\.log/\1/')
            
            # Get tile name from the log
            local tile_name=$(grep "Processing tile:" "$log_file" | tail -n1 | awk -F': ' '{print $NF}' | head -n1)
            
            # Get latest batch progress
            local latest_progress=$(grep "Batch [0-9]*/[0-9]*" "$log_file" | tail -n1)
            
            if [ -n "$latest_progress" ]; then
                log_info "Process $proc_id ($tile_name): $latest_progress"
            else
                log_info "Process $proc_id ($tile_name): Starting..."
            fi
        fi
    done
}

# Function to launch GPU processes
launch_gpu_processes() {
    if [ $GPU_TILES_COUNT -gt 0 ]; then
        log_header "GPU PROCESSING"
        log_info "Starting GPU processing with $NUM_GPU_PROCESSES GPUs"
        
        # Check GPU status before starting
        if command -v nvidia-smi &> /dev/null; then
            log_info "GPU status before starting processes:"
            nvidia-smi
        else
            log_error "nvidia-smi not available - cannot verify GPU status"
        fi
        
        # Start GPU monitoring for each GPU that will be used
        for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
            monitor_gpu_usage $i
            GPU_MONITOR_PIDS[$i]=$?
        done
        
        # Launch python processes for each GPU
        for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
            TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
            LOG_FILE="logs/infer_gpu_${i}.log"
            
            # Verify tile list exists and has content
            if [ ! -f "$TILE_LIST" ]; then
                log_error "Tile list file not found: $TILE_LIST"
                continue
            fi
            
            # Count tiles in the file without using jq
            TILE_COUNT=$(grep -o "\"" "$TILE_LIST" | wc -l)
            TILE_COUNT=$((TILE_COUNT / 2))  # Each tile has two quotes
            log_info "Starting process for GPU $i with $TILE_COUNT tiles"
            
            # Print tile list content for debugging
            if [ "$VERBOSE_GPU" = "true" ]; then
                log_debug "First few tiles for GPU $i:"
                cat "$TILE_LIST" | tr ',' '\n' | head -n 3
                if [ $TILE_COUNT -gt 3 ]; then
                    echo "  ... and $((TILE_COUNT - 3)) more tiles" >&2
                fi
            else
                log_debug "Tile list content (first 200 chars): $(head -c 200 "$TILE_LIST")"
            fi
            
            # Clear any existing log file
            > "$LOG_FILE"
            
            # Launch the process in the foreground with nohup to avoid HUP signal
            log_info "Launching GPU process $i with command:"
            
            # Add verbose GPU flag if enabled
            VERBOSE_GPU_FLAG=""
            if [ "$VERBOSE_GPU" = "true" ]; then
                VERBOSE_GPU_FLAG="--verbose_gpu"
            fi
            
            CMD="$PYTHON_ENV $PYTHON_SCRIPT --config $CONFIG_FILE --mode gpu --gpu_id $i --tile_list $TILE_LIST --process_id $i --checkpoint_path $CHECKPOINT_PATH --output_dir $OUTPUT_DIR --batch_size $GPU_BATCH_SIZE --num_workers $GPU_NUM_WORKERS --log_interval $LOG_INTERVAL $VERBOSE_GPU_FLAG --simplified_logging"
            log_info "$CMD"
            
            # Start the GPU process
            nohup $CMD > "$LOG_FILE" 2>&1 &
            
            PID=$!
            log_info "Process for GPU $i started with PID $PID"
            GPU_PIDS[$i]=$PID
            
            # Verify process started
            if ps -p $PID > /dev/null; then
                log_success "GPU process $i successfully started with PID $PID"
            else
                log_error "Failed to start GPU process $i"
            fi
            
            # Sleep briefly to allow process to initialize
            sleep 5
        done
    fi
}

# Function to process CPU tiles with proper concurrency control
process_cpu_tiles() {
    if [ $CPU_TILES_COUNT -gt 0 ]; then
        CPU_START_TIME=$(date +%s)
        log_header "CPU PROCESSING"
        log_info "Starting CPU processing with maximum $MAX_CONCURRENT_PROCESSES_CPU concurrent processes"
        
        # Variables for process management
        local started=0
        local completed=0
        local active_pids=()
        
        # Variables to track success/failure
        local cpu_successful=0
        local cpu_failed=0
        
        # Map from PID to tile index
        declare -A pid_to_tile
        
        # First start the maximum number of processes
        local to_start=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
        log_info "Initially starting $to_start CPU processes"
        
        for ((i=0; i<to_start; i++)); do
            # Start a new process and get the PID
            local pid=$(start_cpu_process "${CPU_TILES[$i]}" "$i" "$THREADS_PER_CPU_PROCESS")
            
            if [ -n "$pid" ]; then
                active_pids+=("$pid")
                CPU_PIDS+=("$pid")
                pid_to_tile[$pid]=$i
                started=$((started + 1))
            fi
        done
        
        log_info "Started initial batch of $started CPU processes"
        
        # Loop until all tasks are completed
        while [ $completed -lt $CPU_TILES_COUNT ]; do
            # Show progress for running processes
            if [ ${#active_pids[@]} -gt 0 ]; then
                show_cpu_progress "${active_pids[@]}"
            fi
            
            # Check which processes have finished
            local new_active_pids=()
            
            for pid in "${active_pids[@]}"; do
                if ps -p $pid > /dev/null 2>&1; then
                    # Process still running
                    new_active_pids+=("$pid")
                else
                    # Process completed
                    wait $pid 2>/dev/null || true
                    local exit_code=$?
                    completed=$((completed + 1))
                    
                    local tile_index=${pid_to_tile[$pid]}
                    local tile_name=$(basename "${CPU_TILES[$tile_index]}")
                    
                    if [ $exit_code -eq 0 ]; then
                        cpu_successful=$((cpu_successful + 1))
                        log_success "CPU process (PID $pid) completed successfully for tile $tile_name"
                    else
                        cpu_failed=$((cpu_failed + 1))
                        log_error "CPU process (PID $pid) failed with exit code $exit_code for tile $tile_name"
                    fi
                    
                    # Start a new process if there are more tasks
                    if [ $started -lt $CPU_TILES_COUNT ]; then
                        local new_pid=$(start_cpu_process "${CPU_TILES[$started]}" "$started" "$THREADS_PER_CPU_PROCESS")
                        
                        if [ -n "$new_pid" ]; then
                            new_active_pids+=("$new_pid")
                            CPU_PIDS+=("$new_pid")
                            pid_to_tile[$new_pid]=$started
                            started=$((started + 1))
                        fi
                    fi
                fi
            done
            
            # Update active PIDs
            active_pids=("${new_active_pids[@]}")
            
            # Print progress periodically
            local elapsed=$(calculate_time $CPU_START_TIME $(date +%s))
            log_info "CPU progress: $completed/$CPU_TILES_COUNT complete, ${#active_pids[@]} running - Elapsed: $elapsed"
            
            # Sleep to avoid CPU spinning
            sleep 5
        done
        
        CPU_END_TIME=$(date +%s)
        CPU_DURATION=$(calculate_time $CPU_START_TIME $CPU_END_TIME)
        log_success "All CPU processing completed in $CPU_DURATION (success: $cpu_successful, failed: $cpu_failed)"
    fi
}

# Function to monitor GPU processes and report progress
monitor_gpu_processes() {
    if [ $GPU_TILES_COUNT -gt 0 ] && [ ${#GPU_PIDS[@]} -gt 0 ]; then
        log_info "Monitoring GPU processes progress..."
        
        # Check and report progress periodically
        check_interval=30  # seconds
        detailed_interval=300  # Detailed GPU info every 5 minutes
        last_check_time=$(date +%s)
        last_detailed_time=$(date +%s)
        gpu_completed=0
        
        while [ $gpu_completed -lt $NUM_GPU_PROCESSES ]; do
            gpu_completed=0
            for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
                if ! ps -p ${GPU_PIDS[$i]} > /dev/null; then
                    gpu_completed=$((gpu_completed + 1))
                fi
            done
            
            current_time=$(date +%s)
            if [ $((current_time - last_check_time)) -ge $check_interval ]; then
                elapsed=$(calculate_time $GPU_START_TIME $current_time)
                log_info "GPU progress: $gpu_completed/$NUM_GPU_PROCESSES complete - Running for $elapsed"
                last_check_time=$current_time
                
                # Show tail of all GPU process logs
                for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
                    if ps -p ${GPU_PIDS[$i]} > /dev/null; then
                        LOG_FILE="logs/infer_gpu_${i}.log"
                        
                        if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                            echo "Latest progress from GPU $i:" >&2
                            # Find the latest batch progress lines
                            grep "Batch" "$LOG_FILE" | tail -n 3 || grep "Processing tile" "$LOG_FILE" | tail -n 1 || log_warning "No progress lines found in log for GPU $i"
                            echo "..." >&2
                        else
                            log_warning "No logs available for GPU $i (file empty or missing)"
                        fi
                    fi
                done
                
                # Show detailed GPU info periodically
                if [ $((current_time - last_detailed_time)) -ge $detailed_interval ]; then
                    log_info "Detailed GPU status update:"
                    if command -v nvidia-smi &> /dev/null; then
                        nvidia-smi >&2
                    else
                        log_warning "nvidia-smi not available for detailed GPU info"
                    fi
                    last_detailed_time=$current_time
                fi
            fi
            
            # Exit loop if all processes are done
            if [ $gpu_completed -eq $NUM_GPU_PROCESSES ]; then
                break
            fi
            
            sleep 10
        done
        
        # Stop GPU monitoring
        for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
            stop_gpu_monitoring $i
        done
        
        # Check exit status for each process
        GPU_SUCCESSFUL=0
        GPU_FAILED=0
        for ((i=0; i<NUM_GPU_PROCESSES; i++)); do
            wait ${GPU_PIDS[$i]} 2>/dev/null || true
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                GPU_SUCCESSFUL=$((GPU_SUCCESSFUL + 1))
            else
                GPU_FAILED=$((GPU_FAILED + 1))
                log_error "Process on GPU $i failed with exit code $EXIT_CODE"
                # Show tail of log for failed process
                log_error "Last 20 lines of failed GPU $i log:"
                tail -n 20 "logs/infer_gpu_${i}.log" 2>/dev/null || echo "No log available" >&2
            fi
        done
        
        GPU_END_TIME=$(date +%s)
        GPU_DURATION=$(calculate_time $GPU_START_TIME $GPU_END_TIME)
        log_success "All GPU processing completed in $GPU_DURATION (success: $GPU_SUCCESSFUL, failed: $GPU_FAILED)"
    fi
}

# Now we will start both CPU and GPU processes in parallel

# Set start times for both CPU and GPU
CPU_START_TIME=$(date +%s)
GPU_START_TIME=$(date +%s)

# Start GPU processes first (usually take longer to initialize)
launch_gpu_processes

# Start CPU processes in parallel
if [ $CPU_TILES_COUNT -gt 0 ]; then
    # Start the CPU processing in the foreground
    process_cpu_tiles &
    CPU_CONTROLLER_PID=$!
    log_debug "CPU controller started with PID $CPU_CONTROLLER_PID"
fi

# Monitor GPU processes
if [ $GPU_TILES_COUNT -gt 0 ]; then
    monitor_gpu_processes &
    GPU_MONITOR_PID=$!
    log_debug "GPU monitor started with PID $GPU_MONITOR_PID" 
fi

# Wait for controller processes to complete
if [ $CPU_TILES_COUNT -gt 0 ]; then
    log_info "Waiting for CPU processes to complete..."
    wait $CPU_CONTROLLER_PID 2>/dev/null || true
    log_info "CPU processes completed"
fi

if [ $GPU_TILES_COUNT -gt 0 ]; then
    log_info "Waiting for GPU processes to complete..."
    wait $GPU_MONITOR_PID 2>/dev/null || true
    log_info "GPU processes completed"
fi

log_header "RESULTS SUMMARY"

# Check if all output files were created
EXPECTED_FILES=$NUM_TILES_TO_PROCESS
ACTUAL_NEW_FILES=$(find "$OUTPUT_DIR" -newer "$0" -name "*.npy" | wc -l)
ACTUAL_FILES=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)

log_info "Expected $EXPECTED_FILES new output files"
log_info "Found $ACTUAL_NEW_FILES new output files"
log_info "Total output files: $ACTUAL_FILES out of $NUM_TILES total tiles"

if [ $ACTUAL_FILES -eq $NUM_TILES ]; then
    log_success "All tiles have been processed successfully!"
else
    log_warning "Some tiles may not have been processed correctly"
    log_warning "Missing $(($NUM_TILES - $ACTUAL_FILES)) output files"
    
    # List the missing files
    log_info "Listing missing files:"
    for tile_path in "${TILES[@]}"; do
        tile_name=$(basename "$tile_path")
        output_path="$OUTPUT_DIR/${tile_name}.npy"
        
        if [ ! -f "$output_path" ]; then
            echo "  - Missing: ${tile_name}.npy" >&2
        fi
    done
fi

# Calculate and display total execution time
SCRIPT_END=$(date +%s)
TOTAL_DURATION=$(calculate_time $SCRIPT_START $SCRIPT_END)

log_info "CPU time: $([ -n "$CPU_DURATION" ] && echo "$CPU_DURATION" || echo "N/A")"
log_info "GPU time: $([ -n "$GPU_DURATION" ] && echo "$GPU_DURATION" || echo "N/A")"
log_header "COMPLETED IN $TOTAL_DURATION"

# Log where to find detailed process logs
log_info "Detailed logs can be found in:"
log_info "  - CPU logs: logs/infer_cpu_*.log"
log_info "  - GPU logs: logs/infer_gpu_*.log"
log_info "  - GPU monitoring: logs/gpu_monitor_*.log"