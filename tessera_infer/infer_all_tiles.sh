#!/bin/bash
#
# infer_all_tiles.sh - Hybrid CPU/XPU inference script for multi-tile processing
#
# Processes a set of tiles using both CPU and XPU resources according to a specified ratio.
# Automatically skips already processed tiles and distributes workload efficiently.
#
# Usage:
#   bash infer_all_tiles.sh

######################## DAWN EnvIRONMENT SETUP #########################
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
# Set default values
# This is the main directory where all processed and raw data will be stored
BASE_DATA_DIR="/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/jovana_gb/GA2"

# Python environment that has the required packages installed
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
export PYTHON_ENV="python"

# CPU:XPU split ratio
# The script supports simultaneous inference using both CPU and XPU. This ratio specifies the proportion of retiled_patches each device will handle. 
# Default is 1:1 (even split). For XPU-only inference, set to 0:1.
# CPU_XPU_SPLIT="20:39"  # Format: CPU:XPU ratio
CPU_XPU_SPLIT="0:1"  # Format: CPU:XPU ratio

# Maximum number of concurrent processes
MAX_CONCURRENT_PROCESSES_CPU=40

# Clean logs/* if there are any
rm -f logs/*

# CPU cores to use
# Check if running in a SLURM environment and get available CPU cores
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    TOTAL_CPU_CORES=$SLURM_CPUS_PER_TASK
    log_info "Running in SLURM environment with $SLURM_CPUS_PER_TASK CPUs per task"
elif [ -n "$SLURM_JOB_CPUS_PER_NODE" ]; then
    # Parse SLURM_JOB_CPUS_PER_NODE (format can be like "94(x1)" or just "94")
    TOTAL_CPU_CORES=$(echo $SLURM_JOB_CPUS_PER_NODE | sed 's/(.*//')
    log_info "Running in SLURM environment with $TOTAL_CPU_CORES CPUs on this node"
else
    TOTAL_CPU_CORES=$(nproc)  # Fallback to nproc if not in SLURM
    log_info "Detected $TOTAL_CPU_CORES CPU cores using nproc"
fi

# Calculate how many cores to use for inference
# Leaving some for system operations
# AVAILABLE_CORES=$((TOTAL_CPU_CORES * 80 / 100))
AVAILABLE_CORES=80

# Define CPU batch size and workers
CPU_BATCH_SIZE=1024  # Small batch size for CPU
CPU_NUM_WORKERS=0   # Let main thread do the loading

# Maximum number of concurrent XPU processes; this value usually equals the number of XPUs on the device.
MAX_CONCURRENT_PROCESSES_XPU=8  # Number of XPUs to use. Adjust based on your hardware.

# Define XPU batch size and workers
XPU_BATCH_SIZE=8192  # Larger for XPU, if this takes too much memory, reduce it
XPU_NUM_WORKERS=2    # More workers for XPU

# Other settings
VERBOSE_XPU=true    # Enable detailed XPU logging
LOG_INTERVAL=5      # Log every N batches
###########################################################################

TILES_DIR="${BASE_DATA_DIR}/retiled_d_pixel"
OUTPUT_DIR="${BASE_DATA_DIR}/representation_retiled"
CONFIG_FILE="configs/multi_tile_infer_config.py"
PYTHON_SCRIPT="src/multi_tile_infer.py"

# Path to the checkpoint file for the model
# CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250427_084307.pt"
# CHECKPOINT_PATH="checkpoints/checkpoint_20250603_100440_copy.pt"
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250605_221257_copy.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cpu-xpu-split)
            CPU_XPU_SPLIT="$2"
            shift 2
            ;;
        --max-cpu)
            MAX_CONCURRENT_PROCESSES_CPU="$2"
            shift 2
            ;;
        --max-xpu)
            MAX_CONCURRENT_PROCESSES_XPU="$2"
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
        --verbose-xpu)
            VERBOSE_XPU="$2"
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

# Function to check XPU availability
check_xpu_availability() {
    # First check if the Python environment can detect XPUs
    XPU_COUNT=$($PYTHON_ENV -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0)" 2>/dev/null)
    
    # If there's an error or return is not a number, assume 0
    if ! [[ "$XPU_COUNT" =~ ^[0-9]+$ ]]; then
        XPU_COUNT=0
    fi
    
    echo $XPU_COUNT
}

# Function to monitor XPU usage through logs
monitor_xpu_usage() {
    local xpu_id=$1
    local monitor_interval=60  # seconds
    local monitor_pid_file="logs/xpu_monitor_${xpu_id}.pid"
    
    log_info "Starting XPU $xpu_id monitoring (every ${monitor_interval}s)"
    
    # Run XPU monitor in background
    (
        while true; do
            timestamp=$(date +"%Y-%m-%d %H:%M:%S")
            echo "[$timestamp] XPU $xpu_id monitoring:" >> "logs/xpu_monitor_${xpu_id}.log"
            
            # Use the Python environment to get XPU stats
            $PYTHON_ENV -c "
import torch
import intel_extension_for_pytorch as ipex
import sys

try:
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print('XPU not available')
        sys.exit(0)
    
    dev = torch.xpu.device(${xpu_id})
    mem_allocated = torch.xpu.memory_allocated(dev) / (1024**2)
    mem_reserved = torch.xpu.memory_reserved(dev) / (1024**2)
    max_mem = torch.xpu.max_memory_allocated(dev) / (1024**2)
    
    print(f'XPU {${xpu_id}} Memory: {mem_allocated:.2f}MB allocated, {mem_reserved:.2f}MB reserved, {max_mem:.2f}MB peak')
except Exception as e:
    print(f'Error getting XPU stats: {e}')
" >> "logs/xpu_monitor_${xpu_id}.log" 2>&1
            
            echo "" >> "logs/xpu_monitor_${xpu_id}.log"
            sleep $monitor_interval
        done
    ) &
    
    monitor_pid=$!
    echo $monitor_pid > "$monitor_pid_file"
    log_debug "XPU $xpu_id monitor started with PID $monitor_pid"
    
    return $monitor_pid
}

# Function to stop XPU monitoring
stop_xpu_monitoring() {
    local xpu_id=$1
    local monitor_pid_file="logs/xpu_monitor_${xpu_id}.pid"
    
    if [ -f "$monitor_pid_file" ]; then
        local monitor_pid=$(cat "$monitor_pid_file")
        if ps -p $monitor_pid > /dev/null; then
            kill $monitor_pid 2>/dev/null
            log_debug "Stopped XPU $xpu_id monitoring (PID $monitor_pid)"
        fi
        rm -f "$monitor_pid_file"
    fi
}

# Print CPU and XPU detection information
log_info "CPU allocation: Total cores: $TOTAL_CPU_CORES, Using for inference: $AVAILABLE_CORES"

# Print SLURM job information if available
if [ -n "$SLURM_JOB_ID" ]; then
    log_info "Running as SLURM job $SLURM_JOB_ID on node $HOSTNAME"
    if [ -n "$SLURM_JOB_GPUS" ]; then
        log_info "SLURM allocated GPUs: $SLURM_JOB_GPUS"
    fi
fi

# Parse CPU:XPU split ratio
IFS=':' read -r CPU_RATIO XPU_RATIO <<< "$CPU_XPU_SPLIT"
CPU_RATIO=${CPU_RATIO:-1}
XPU_RATIO=${XPU_RATIO:-1}
TOTAL_RATIO=$((CPU_RATIO + XPU_RATIO))
log_info "Original CPU:XPU split ratio = $CPU_RATIO:$XPU_RATIO (approximately $(( CPU_RATIO * 100 / TOTAL_RATIO ))%:$(( XPU_RATIO * 100 / TOTAL_RATIO ))%)"

# Arrays to store process PIDs
declare -a CPU_PIDS
declare -a XPU_PIDS
declare -a XPU_MONITOR_PIDS

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

# Function to kill all XPU processes
kill_xpu_processes() {
    log_debug "Checking for remaining XPU processes..."
    
    # Kill each tracked XPU process
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            log_debug "Killing XPU process with PID: $pid"
            kill_process_tree $pid
        fi
    done
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
    
    # Kill XPU processes
    for pid in "${XPU_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill_process_tree $pid
        fi
    done
    
    # Then make sure no XPU processes are left
    kill_xpu_processes
    
    # Stop XPU monitoring
    for ((i=0; i<MAX_CONCURRENT_PROCESSES_XPU; i++)); do
        if [[ ${XPU_MONITOR_PIDS[$i]} -gt 0 ]]; then
            stop_xpu_monitoring $i
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

# Check for XPU availability
XPU_COUNT=$(check_xpu_availability)
log_info "Detected $XPU_COUNT Intel XPUs"

if [ $XPU_COUNT -lt $MAX_CONCURRENT_PROCESSES_XPU ]; then
    log_warning "Adjusting MAX_CONCURRENT_PROCESSES_XPU from $MAX_CONCURRENT_PROCESSES_XPU to $XPU_COUNT (available XPUs)"
    MAX_CONCURRENT_PROCESSES_XPU=$XPU_COUNT
fi

# Calculate split based on ratio and available resources
if [ "$CPU_RATIO" -eq 0 ]; then
    # All tiles go to XPU
    CPU_TILES_COUNT=0
    XPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
elif [ "$XPU_RATIO" -eq 0 ] || [ $XPU_COUNT -eq 0 ]; then
    # All tiles go to CPU
    CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
    XPU_TILES_COUNT=0
else
    # Normal split based on ratio - calculate exact distribution
    initial_cpu_tiles=$(( NUM_TILES_TO_PROCESS * CPU_RATIO / TOTAL_RATIO ))
    XPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS - initial_cpu_tiles ))
    
    # Adjust CPU_TILES_COUNT to ensure it's evenly divisible by MAX_CONCURRENT_PROCESSES_CPU
    # Only if there are enough tiles to process
    if [ $initial_cpu_tiles -gt $MAX_CONCURRENT_PROCESSES_CPU ]; then
        # Round to the nearest multiple of MAX_CONCURRENT_PROCESSES_CPU
        CPU_TILES_COUNT=$(( (initial_cpu_tiles + MAX_CONCURRENT_PROCESSES_CPU/2) / MAX_CONCURRENT_PROCESSES_CPU * MAX_CONCURRENT_PROCESSES_CPU ))
        
        # Ensure we don't exceed the total tiles
        if [ $CPU_TILES_COUNT -gt $NUM_TILES_TO_PROCESS ]; then
            # Instead of floor division, use exact number
            CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
            XPU_TILES_COUNT=0
        else
            # Recalculate XPU allocation
            XPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS - CPU_TILES_COUNT ))
        fi
    else
        # If not many CPU tiles, just use the initial calculation
        CPU_TILES_COUNT=$initial_cpu_tiles
    fi
fi

# Verify that all tiles are accounted for
if [ $(( CPU_TILES_COUNT + XPU_TILES_COUNT )) -ne $NUM_TILES_TO_PROCESS ]; then
    log_warning "Tile allocation mismatch: CPU ($CPU_TILES_COUNT) + XPU ($XPU_TILES_COUNT) != Total ($NUM_TILES_TO_PROCESS)"
    
    # Fix by adjusting XPU allocation to cover all remaining tiles
    XPU_TILES_COUNT=$(( NUM_TILES_TO_PROCESS - CPU_TILES_COUNT ))
    log_info "Adjusted allocation: CPU: $CPU_TILES_COUNT, XPU: $XPU_TILES_COUNT, Total: $(( CPU_TILES_COUNT + XPU_TILES_COUNT ))"
fi

# Calculate and display the actual percentages
if [ $NUM_TILES_TO_PROCESS -gt 0 ]; then
    CPU_PERCENT=$(( CPU_TILES_COUNT * 100 / NUM_TILES_TO_PROCESS ))
    XPU_PERCENT=$(( XPU_TILES_COUNT * 100 / NUM_TILES_TO_PROCESS ))
    log_info "Final allocation: CPU will process $CPU_TILES_COUNT tiles ($CPU_PERCENT%), XPU will process $XPU_TILES_COUNT tiles ($XPU_PERCENT%)"
else
    log_warning "No tiles to process!"
fi

# Prepare tile lists based on allocation with bounds checking
if [ $CPU_TILES_COUNT -gt 0 ]; then
    # Ensure CPU_TILES_COUNT doesn't exceed available tiles
    if [ $CPU_TILES_COUNT -gt ${#TILES_TO_PROCESS[@]} ]; then
        log_warning "CPU_TILES_COUNT ($CPU_TILES_COUNT) exceeds available tiles (${#TILES_TO_PROCESS[@]}). Adjusting."
        CPU_TILES_COUNT=${#TILES_TO_PROCESS[@]}
        XPU_TILES_COUNT=0
    fi
    
    # Prepare CPU tiles list
    CPU_TILES=("${TILES_TO_PROCESS[@]:0:$CPU_TILES_COUNT}")
    log_info "Assigned ${#CPU_TILES[@]} tiles to CPU processing"
else
    # Empty CPU tiles array
    CPU_TILES=()
fi

if [ $XPU_TILES_COUNT -gt 0 ]; then
    # Check if there are tiles left for XPU after CPU allocation
    if [ $CPU_TILES_COUNT -ge ${#TILES_TO_PROCESS[@]} ]; then
        log_warning "No tiles left for XPU after CPU allocation. Adjusting."
        XPU_TILES=()
        XPU_TILES_COUNT=0
    else
        # Prepare XPU tiles list
        XPU_TILES=("${TILES_TO_PROCESS[@]:$CPU_TILES_COUNT}")
        XPU_TILES_COUNT=${#XPU_TILES[@]}
        log_info "Assigned $XPU_TILES_COUNT tiles to XPU processing"
    fi
else
    # Empty XPU tiles array
    XPU_TILES=()
    XPU_TILES_COUNT=0
fi

# Double-check all tiles are accounted for
TOTAL_ASSIGNED_TILES=$(( ${#CPU_TILES[@]} + ${#XPU_TILES[@]} ))
if [ $TOTAL_ASSIGNED_TILES -ne $NUM_TILES_TO_PROCESS ]; then
    log_error "Tile assignment mismatch: CPU (${#CPU_TILES[@]}) + XPU (${#XPU_TILES[@]}) != Total ($NUM_TILES_TO_PROCESS)"
    
    # Find unassigned tiles and assign them
    declare -A assigned_tiles
    for tile in "${CPU_TILES[@]}" "${XPU_TILES[@]}"; do
        assigned_tiles["$tile"]=1
    done
    
    # Create array of unassigned tiles
    UNASSIGNED_TILES=()
    for tile in "${TILES_TO_PROCESS[@]}"; do
        if [ -z "${assigned_tiles[$tile]}" ]; then
            UNASSIGNED_TILES+=("$tile")
            log_warning "Found unassigned tile: $(basename "$tile")"
        fi
    done
    
    # Assign unassigned tiles
    if [ ${#UNASSIGNED_TILES[@]} -gt 0 ]; then
        log_info "Assigning ${#UNASSIGNED_TILES[@]} unassigned tiles"
        
        # Decide where to assign based on current allocation
        if [ $XPU_COUNT -gt 0 ] && [ $XPU_TILES_COUNT -gt 0 ]; then
            # Add to XPU if available and already processing
            XPU_TILES+=("${UNASSIGNED_TILES[@]}")
            XPU_TILES_COUNT=${#XPU_TILES[@]}
            log_info "Added unassigned tiles to XPU processing (now $XPU_TILES_COUNT tiles)"
        else
            # Add to CPU otherwise
            CPU_TILES+=("${UNASSIGNED_TILES[@]}")
            CPU_TILES_COUNT=${#CPU_TILES[@]}
            log_info "Added unassigned tiles to CPU processing (now $CPU_TILES_COUNT tiles)"
        fi
        
        # Verify again
        TOTAL_ASSIGNED_TILES=$(( ${#CPU_TILES[@]} + ${#XPU_TILES[@]} ))
        if [ $TOTAL_ASSIGNED_TILES -ne $NUM_TILES_TO_PROCESS ]; then
            log_error "Failed to fix tile assignment. Aborting."
            exit 1
        else
            log_success "Successfully assigned all tiles. CPU: ${#CPU_TILES[@]}, XPU: ${#XPU_TILES[@]}, Total: $TOTAL_ASSIGNED_TILES"
        fi
    fi
fi

# Calculate CPU batch settings
if [ $CPU_TILES_COUNT -gt 0 ]; then
    # Limit max concurrent processes based on available cores
    # Each process should have at least 4 cores for good performance
    RECOMMENDED_MAX_PROCESSES=$((AVAILABLE_CORES / 4))
    
    # If we have more cores than default MAX_CONCURRENT_PROCESSES_CPU, adjust upward
    if [ $RECOMMENDED_MAX_PROCESSES -gt $MAX_CONCURRENT_PROCESSES_CPU ]; then
        log_info "Increasing MAX_CONCURRENT_PROCESSES_CPU from $MAX_CONCURRENT_PROCESSES_CPU to $RECOMMENDED_MAX_PROCESSES based on available cores"
        MAX_CONCURRENT_PROCESSES_CPU=$RECOMMENDED_MAX_PROCESSES
    fi
    
    # Calculate how many processes to start
    NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
    
    # Calculate threads per process
    THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
    if [ $THREADS_PER_CPU_PROCESS -lt 1 ]; then
        THREADS_PER_CPU_PROCESS=1
    fi
    
    # No need to calculate TILES_PER_CPU_PROCESS as each CPU process handles one tile
    log_info "Using $NUM_CPU_PROCESSES CPU processes with $THREADS_PER_CPU_PROCESS threads each (from $AVAILABLE_CORES available cores)"
else
    log_info "No tiles assigned to CPU processing"
    NUM_CPU_PROCESSES=0
fi

# Prepare XPU tiles lists
if [ $XPU_TILES_COUNT -gt 0 ]; then
    # Check if XPUs are available
    if [ $XPU_COUNT -gt 0 ]; then
        log_info "Using $MAX_CONCURRENT_PROCESSES_XPU Intel XPUs for processing"
        
        # Get basic XPU information
        log_info "XPU information:"
        $PYTHON_ENV -c "
import torch
import intel_extension_for_pytorch as ipex

if hasattr(torch, 'xpu') and torch.xpu.is_available():
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
" || log_warning "Error getting XPU information"
        
        NUM_XPU_PROCESSES=$MAX_CONCURRENT_PROCESSES_XPU
        
        # Divide XPU tiles among XPU processes
        if [ $XPU_TILES_COUNT -gt 0 ]; then
            TILES_PER_XPU=$((XPU_TILES_COUNT / NUM_XPU_PROCESSES))
            REMAINDER=$((XPU_TILES_COUNT % NUM_XPU_PROCESSES))
            
            log_info "Each XPU will process approximately $TILES_PER_XPU tiles"
            
            # Create tile lists for each XPU with improved distribution logic
            start_idx=0
            for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
                # Calculate how many tiles this XPU should process
                tiles_for_this_xpu=$TILES_PER_XPU
                if [ $i -lt $REMAINDER ]; then
                    tiles_for_this_xpu=$((tiles_for_this_xpu + 1))
                fi
                
                # Calculate end index with bounds checking
                end_idx=$((start_idx + tiles_for_this_xpu))
                if [ $end_idx -gt $XPU_TILES_COUNT ]; then
                    log_warning "XPU $i: End index $end_idx exceeds XPU tiles count $XPU_TILES_COUNT. Adjusting."
                    end_idx=$XPU_TILES_COUNT
                fi
                
                # Create tile list for this XPU
                TILE_LIST="src/tile_lists/tiles_xpu_${i}.json"
                
                # Build JSON array of tiles with bounds checking
                echo -n "[" > "$TILE_LIST"
                for ((j=start_idx; j<end_idx; j++)); do
                    if [ $j -ge $XPU_TILES_COUNT ]; then
                        log_error "Index $j out of bounds for XPU tiles array (size: $XPU_TILES_COUNT)"
                        break
                    fi
                    
                    # Add comma if not the first item
                    if [ $j -gt $start_idx ]; then
                        echo -n "," >> "$TILE_LIST"
                    fi
                    echo -n "\"${XPU_TILES[$j]}\"" >> "$TILE_LIST"
                done
                echo "]" >> "$TILE_LIST"
                
                # Count tiles in the file
                TILE_COUNT=$(grep -o "\"" "$TILE_LIST" | wc -l)
                TILE_COUNT=$((TILE_COUNT / 2))  # Each tile has two quotes
                log_info "Created tile list for XPU $i: $TILE_COUNT tiles"
                
                # Enhanced logging - show actual tiles
                if [ "$VERBOSE_XPU" = "true" ]; then
                    log_debug "XPU $i tiles list content:"
                    cat "$TILE_LIST" | tr ',' '\n' | head -n 5
                    if [ $TILE_COUNT -gt 5 ]; then
                        echo "  ... and $((TILE_COUNT - 5)) more tiles" >&2
                    fi
                else
                    log_debug "Tile list content sample: $(head -c 100 "$TILE_LIST")"
                fi
                
                # Update start index for next iteration
                start_idx=$end_idx
            done
            
            # Check if all XPU tiles were assigned
            if [ $start_idx -ne $XPU_TILES_COUNT ]; then
                log_warning "Not all XPU tiles were assigned: $start_idx out of $XPU_TILES_COUNT"
                
                # Assign remaining tiles to the last XPU process
                if [ $start_idx -lt $XPU_TILES_COUNT ] && [ $NUM_XPU_PROCESSES -gt 0 ]; then
                    last_xpu=$((NUM_XPU_PROCESSES - 1))
                    TILE_LIST="src/tile_lists/tiles_xpu_${last_xpu}.json"
                    
                    # Read existing list
                    content=$(cat "$TILE_LIST")
                    # Remove the closing bracket
                    content="${content%]}"
                    
                    # Add remaining tiles
                    for ((j=start_idx; j<XPU_TILES_COUNT; j++)); do
                        # Add comma if content doesn't end with '['
                        if [[ ! "$content" =~ \[$ ]]; then
                            content="${content},"
                        fi
                        content="${content}\"${XPU_TILES[$j]}\""
                    done
                    
                    # Close the JSON array
                    content="${content}]"
                    echo "$content" > "$TILE_LIST"
                    
                    # Count tiles in the updated file
                    TILE_COUNT=$(grep -o "\"" "$TILE_LIST" | wc -l)
                    TILE_COUNT=$((TILE_COUNT / 2))
                    log_info "Updated tile list for XPU $last_xpu: now contains $TILE_COUNT tiles"
                    
                    # Verify all tiles are now assigned
                    log_success "All XPU tiles are now assigned"
                fi
            else
                log_success "All XPU tiles successfully assigned"
            fi
        else
            log_warning "No tiles to process on XPU!"
            NUM_XPU_PROCESSES=0
        fi
    else
        log_warning "No XPUs detected, moving all XPU tiles to CPU"
        CPU_TILES=("${TILES_TO_PROCESS[@]}")
        CPU_TILES_COUNT=$NUM_TILES_TO_PROCESS
        XPU_TILES_COUNT=0
        XPU_TILES=()
        
        # Recalculate CPU settings
        NUM_CPU_PROCESSES=$(( CPU_TILES_COUNT < MAX_CONCURRENT_PROCESSES_CPU ? CPU_TILES_COUNT : MAX_CONCURRENT_PROCESSES_CPU ))
        THREADS_PER_CPU_PROCESS=$(( AVAILABLE_CORES / NUM_CPU_PROCESSES ))
        if [ $THREADS_PER_CPU_PROCESS -lt 1 ]; then
            THREADS_PER_CPU_PROCESS=1
        fi
        
        log_info "Updated to use $NUM_CPU_PROCESSES CPU processes with $THREADS_PER_CPU_PROCESS threads each"
    fi
else
    log_info "No tiles assigned to XPU processing"
    NUM_XPU_PROCESSES=0
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

# Function to launch XPU processes
launch_xpu_processes() {
    if [ $XPU_TILES_COUNT -gt 0 ]; then
        log_header "XPU PROCESSING"
        log_info "Starting XPU processing with $NUM_XPU_PROCESSES XPUs"
                
        # Start XPU monitoring for each XPU that will be used
        for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
            monitor_xpu_usage $i
            XPU_MONITOR_PIDS[$i]=$?
        done
        
        # Launch python processes for each XPU
        for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
            TILE_LIST="src/tile_lists/tiles_xpu_${i}.json"
            LOG_FILE="logs/infer_xpu_${i}.log"
            
            # Verify tile list exists and has content
            if [ ! -f "$TILE_LIST" ]; then
                log_error "Tile list file not found: $TILE_LIST"
                continue
            fi
            
            # Count tiles in the file without using jq
            TILE_COUNT=$(grep -o "\"" "$TILE_LIST" | wc -l)
            TILE_COUNT=$((TILE_COUNT / 2))  # Each tile has two quotes
            
            # Skip empty tile lists
            if [ $TILE_COUNT -eq 0 ]; then
                log_warning "Skipping XPU $i as it has no tiles assigned"
                continue
            fi
            
            log_info "Starting process for XPU $i with $TILE_COUNT tiles"
            
            # Print tile list content for debugging
            if [ "$VERBOSE_XPU" = "true" ]; then
                log_debug "First few tiles for XPU $i:"
                cat "$TILE_LIST" | tr ',' '\n' | head -n 3
                if [ $TILE_COUNT -gt 3 ]; then
                    echo "  ... and $((TILE_COUNT - 3)) more tiles" >&2
                fi
            else
                log_debug "Tile list content sample: $(head -c 200 "$TILE_LIST")"
            fi
            
            # Clear any existing log file
            > "$LOG_FILE"
            
            # Launch the process in the foreground with nohup to avoid HUP signal
            log_info "Launching XPU process $i with command:"
            
            # Add verbose XPU flag if enabled
            VERBOSE_XPU_FLAG=""
            if [ "$VERBOSE_XPU" = "true" ]; then
                VERBOSE_XPU_FLAG="--verbose_xpu"
            fi
            
            CMD="$PYTHON_ENV $PYTHON_SCRIPT --config $CONFIG_FILE --mode xpu --xpu_id $i --tile_list $TILE_LIST --process_id $i --checkpoint_path $CHECKPOINT_PATH --output_dir $OUTPUT_DIR --batch_size $XPU_BATCH_SIZE --num_workers $XPU_NUM_WORKERS --log_interval $LOG_INTERVAL $VERBOSE_XPU_FLAG --simplified_logging"
            log_info "$CMD"
            
            # Start the XPU process
            nohup $CMD > "$LOG_FILE" 2>&1 &
            
            PID=$!
            log_info "Process for XPU $i started with PID $PID"
            XPU_PIDS[$i]=$PID
            
            # Verify process started
            if ps -p $PID > /dev/null; then
                log_success "XPU process $i successfully started with PID $PID"
            else
                log_error "Failed to start XPU process $i"
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

# Function to monitor XPU processes and report progress
monitor_xpu_processes() {
    if [ $XPU_TILES_COUNT -gt 0 ] && [ ${#XPU_PIDS[@]} -gt 0 ]; then
        log_info "Monitoring XPU processes progress..."
        
        # Check and report progress periodically
        check_interval=30  # seconds
        detailed_interval=300  # Detailed XPU info every 5 minutes
        last_check_time=$(date +%s)
        last_detailed_time=$(date +%s)
        xpu_completed=0
        
        while [ $xpu_completed -lt $NUM_XPU_PROCESSES ]; do
            xpu_completed=0
            for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
                if ! ps -p ${XPU_PIDS[$i]} > /dev/null 2>&1; then
                    xpu_completed=$((xpu_completed + 1))
                fi
            done
            
            current_time=$(date +%s)
            if [ $((current_time - last_check_time)) -ge $check_interval ]; then
                elapsed=$(calculate_time $XPU_START_TIME $current_time)
                log_info "XPU progress: $xpu_completed/$NUM_XPU_PROCESSES complete - Running for $elapsed"
                last_check_time=$current_time
                
                # Show tail of all XPU process logs
                for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
                    if ps -p ${XPU_PIDS[$i]} > /dev/null 2>&1; then
                        LOG_FILE="logs/infer_xpu_${i}.log"
                        
                        if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                            echo "Latest progress from XPU $i:" >&2
                            # Find the latest batch progress lines
                            grep "Batch" "$LOG_FILE" | tail -n 3 || grep "Processing tile" "$LOG_FILE" | tail -n 1 || log_warning "No progress lines found in log for XPU $i"
                            echo "..." >&2
                        else
                            log_warning "No logs available for XPU $i (file empty or missing)"
                        fi
                    fi
                done
                
                # Show detailed XPU info periodically
                if [ $((current_time - last_detailed_time)) -ge $detailed_interval ]; then
                    log_info "Detailed XPU status update (through logs):"
                    for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
                        if ps -p ${XPU_PIDS[$i]} > /dev/null 2>&1; then
                            LOG_FILE="logs/xpu_monitor_${i}.log"
                            if [ -f "$LOG_FILE" ]; then
                                tail -n 3 "$LOG_FILE" >&2
                            else
                                log_warning "No monitoring log for XPU $i"
                            fi
                        fi
                    done
                    last_detailed_time=$current_time
                fi
            fi
            
            # Exit loop if all processes are done
            if [ $xpu_completed -eq $NUM_XPU_PROCESSES ]; then
                break
            fi
            
            sleep 10
        done
        
        # Stop XPU monitoring
        for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
            stop_xpu_monitoring $i
        done
        
        # Check exit status for each process
        XPU_SUCCESSFUL=0
        XPU_FAILED=0
        for ((i=0; i<NUM_XPU_PROCESSES; i++)); do
            wait ${XPU_PIDS[$i]} 2>/dev/null || true
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                XPU_SUCCESSFUL=$((XPU_SUCCESSFUL + 1))
            else
                XPU_FAILED=$((XPU_FAILED + 1))
                log_error "Process on XPU $i failed with exit code $EXIT_CODE"
                # Show tail of log for failed process
                log_error "Last 20 lines of failed XPU $i log:"
                tail -n 20 "logs/infer_xpu_${i}.log" 2>/dev/null || echo "No log available" >&2
            fi
        done
        
        XPU_END_TIME=$(date +%s)
        XPU_DURATION=$(calculate_time $XPU_START_TIME $XPU_END_TIME)
        log_success "All XPU processing completed in $XPU_DURATION (success: $XPU_SUCCESSFUL, failed: $XPU_FAILED)"
    fi
}

# Now we will start both CPU and XPU processes in parallel

# Set start times for both CPU and XPU
CPU_START_TIME=$(date +%s)
XPU_START_TIME=$(date +%s)

# Start XPU processes first (usually take longer to initialize)
launch_xpu_processes

# Start CPU processes in parallel
if [ $CPU_TILES_COUNT -gt 0 ]; then
    # Start the CPU processing in the foreground
    process_cpu_tiles &
    CPU_CONTROLLER_PID=$!
    log_debug "CPU controller started with PID $CPU_CONTROLLER_PID"
fi

# Monitor XPU processes
if [ $XPU_TILES_COUNT -gt 0 ]; then
    monitor_xpu_processes &
    XPU_MONITOR_PID=$!
    log_debug "XPU monitor started with PID $XPU_MONITOR_PID" 
fi

# Wait for controller processes to complete
if [ $CPU_TILES_COUNT -gt 0 ]; then
    log_info "Waiting for CPU processes to complete..."
    wait $CPU_CONTROLLER_PID 2>/dev/null || true
    log_info "CPU processes completed"
fi

if [ $XPU_TILES_COUNT -gt 0 ]; then
    log_info "Waiting for XPU processes to complete..."
    wait $XPU_MONITOR_PID 2>/dev/null || true
    log_info "XPU processes completed"
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
log_info "XPU time: $([ -n "$XPU_DURATION" ] && echo "$XPU_DURATION" || echo "N/A")"
log_header "COMPLETED IN $TOTAL_DURATION"

# Log where to find detailed process logs
log_info "Detailed logs can be found in:"
log_info "  - CPU logs: logs/infer_cpu_*.log"
log_info "  - XPU logs: logs/infer_xpu_*.log"
log_info "  - XPU monitoring: logs/xpu_monitor_*.log"