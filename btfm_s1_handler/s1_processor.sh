#!/bin/bash

# s1_processor.sh - Seasonal Sentinel-1 SAR data processor
# Usage: bash s1_processor.sh --input_tiff path/to/tiff --start_time YYYY-MM-DD --end_time YYYY-MM-DD [additional options]

# Function to display usage
usage() {
    echo "Usage: $0 --input_tiff <path> --start_time <YYYY-MM-DD> --end_time <YYYY-MM-DD> [options]"
    echo "Required:"
    echo "  --input_tiff    Path to input TIFF file for geography extraction"
    echo "  --start_time    Start date in YYYY-MM-DD format"
    echo "  --end_time      End date in YYYY-MM-DD format"
    echo "Options:"
    echo "  --output        Output directory (default: sentinel1_output)"
    echo "  --workers       Number of parallel download workers (default: 8)"
    echo "  --chunksize     Chunk size for processing (default: 1024)"
    echo "  --dask_workers  Number of Dask workers (default: 32)"
    echo "  --worker_memory Memory per Dask worker in GB (default: 32)"
    echo "  --temp_dir      Temporary directory for intermediate files"
    echo "  --orbit_state   Orbit state to process: ascending, descending, or both (default: both)"
    echo "  --overwrite     Overwrite existing files instead of skipping them"
    echo "  --resume        Resume from checkpoint if available"
    echo "  --max_retries   Maximum number of retries for operations (default: 3)"
    echo "  --timeout       Timeout in seconds for individual operations (default: 600)"
    echo "  --debug         Enable debug logging"
    exit 1
}

# Parse command line arguments
INPUT_TIFF=""
START_TIME=""
END_TIME=""
OUTPUT="sentinel1_output"
WORKERS=8
CHUNKSIZE=1024
DASK_WORKERS=20
WORKER_MEMORY=24
TEMP_DIR=""
ORBIT_STATE="both"
OVERWRITE=""
RESUME=""
MAX_RETRIES=3
TIMEOUT=600
DEBUG=""

while [ "$1" != "" ]; do
    case $1 in
        --input_tiff)  shift; INPUT_TIFF=$1 ;;
        --start_time)  shift; START_TIME=$1 ;;
        --end_time)    shift; END_TIME=$1 ;;
        --output)      shift; OUTPUT=$1 ;;
        --workers)     shift; WORKERS=$1 ;;
        --chunksize)   shift; CHUNKSIZE=$1 ;;
        --dask_workers) shift; DASK_WORKERS=$1 ;;
        --worker_memory) shift; WORKER_MEMORY=$1 ;;
        --temp_dir)    shift; TEMP_DIR=$1 ;;
        --orbit_state) shift; ORBIT_STATE=$1 ;;
        --overwrite)   OVERWRITE="--overwrite" ;;
        --resume)      RESUME="--resume" ;;
        --max_retries) shift; MAX_RETRIES=$1 ;;
        --timeout)     shift; TIMEOUT=$1 ;;
        --debug)       DEBUG="--debug" ;;
        -h|--help)     usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Check required parameters
if [ -z "$INPUT_TIFF" ] || [ -z "$START_TIME" ] || [ -z "$END_TIME" ]; then
    echo "Error: Missing required parameters"
    usage
fi

# Validate date format
validate_date() {
    if ! date -d "$1" >/dev/null 2>&1; then
        echo "Error: Invalid date format: $1. Use YYYY-MM-DD."
        exit 1
    fi
}

validate_date "$START_TIME"
validate_date "$END_TIME"

# Check if start date is before end date
if [ "$(date -d "$START_TIME" +%s)" -gt "$(date -d "$END_TIME" +%s)" ]; then
    echo "Error: Start date must be before end date"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT"

# Create log directory
LOG_DIR="${OUTPUT}/logs"
mkdir -p "$LOG_DIR"

echo "==== Sentinel-1 Seasonal Processor ===="
echo "Input TIFF: $INPUT_TIFF"
echo "Date range: $START_TIME to $END_TIME"
echo "Output directory: $OUTPUT"
echo "======================================="

# Function to determine seasons based on date
get_seasons() {
    local start_year=$(date -d "$START_TIME" +%Y)
    local end_year=$(date -d "$END_TIME" +%Y)

    for ((year=start_year; year<=end_year; year++)); do
        # Define seasons
        echo "${year}-01-01,${year}-03-31" # Winter
        echo "${year}-04-01,${year}-06-30" # Spring
        echo "${year}-07-01,${year}-09-30" # Summer
        echo "${year}-10-01,${year}-12-31" # Fall
    done
}

# Generate all seasons between start and end dates
SEASONS=()

for season in $(get_seasons); do
    season_start=$(echo "$season" | cut -d',' -f1)
    season_end=$(echo "$season" | cut -d',' -f2)
    
    # Skip seasons before start date or after end date
    if [ "$(date -d "$season_end" +%s)" -lt "$(date -d "$START_TIME" +%s)" ] || 
       [ "$(date -d "$season_start" +%s)" -gt "$(date -d "$END_TIME" +%s)" ]; then
        continue
    fi
    
    # Adjust season start if it's before the overall start date
    if [ "$(date -d "$season_start" +%s)" -lt "$(date -d "$START_TIME" +%s)" ]; then
        season_start="$START_TIME"
    fi
    
    # Adjust season end if it's after the overall end date
    if [ "$(date -d "$season_end" +%s)" -gt "$(date -d "$END_TIME" +%s)" ]; then
        season_end="$END_TIME"
    fi
    
    # Add to seasons array
    SEASONS+=("$season_start,$season_end")
done

# Process each season
echo "Identified ${#SEASONS[@]} seasons to process"
for ((i=0; i<${#SEASONS[@]}; i++)); do
    season_range="${SEASONS[$i]}"
    season_start=$(echo "$season_range" | cut -d',' -f1)
    season_end=$(echo "$season_range" | cut -d',' -f2)
    
    season_name=$(date -d "$season_start" +"%Y-%m")
    
    echo "Processing season $((i+1))/${#SEASONS[@]}: $season_start to $season_end"
    
    # Build Python command with all parameters
    PYTHON_CMD="python3 s1_seasonal_processor.py"
    PYTHON_CMD+=" --input_tiff $INPUT_TIFF"
    PYTHON_CMD+=" --start_date $season_start"
    PYTHON_CMD+=" --end_date $season_end"
    PYTHON_CMD+=" --output $OUTPUT"
    PYTHON_CMD+=" --workers $WORKERS"
    PYTHON_CMD+=" --chunksize $CHUNKSIZE"
    PYTHON_CMD+=" --dask_workers $DASK_WORKERS"
    PYTHON_CMD+=" --worker_memory $WORKER_MEMORY"
    PYTHON_CMD+=" --orbit_state $ORBIT_STATE"
    PYTHON_CMD+=" --max_retries $MAX_RETRIES"
    PYTHON_CMD+=" --timeout $TIMEOUT"
    
    # Add optional flags
    if [ ! -z "$TEMP_DIR" ]; then
        PYTHON_CMD+=" --temp_dir $TEMP_DIR"
    fi
    
    if [ ! -z "$OVERWRITE" ]; then
        PYTHON_CMD+=" $OVERWRITE"
    fi
    
    if [ ! -z "$RESUME" ]; then
        PYTHON_CMD+=" $RESUME"
    fi
    
    if [ ! -z "$DEBUG" ]; then
        PYTHON_CMD+=" $DEBUG"
    fi
    
    # Log file for this season
    LOG_FILE="${LOG_DIR}/sentinel1_${season_name}.log"
    
    echo "Starting Python process. Log: $LOG_FILE"
    echo "$PYTHON_CMD"
    
    # Execute Python command and log output
    $PYTHON_CMD 2>&1 | tee "$LOG_FILE"
    
    # Check if Python process exited successfully
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: Python process failed for season $season_start to $season_end"
        echo "Check log file: $LOG_FILE"
        # Continue with next season despite error
    fi
    
    echo "Completed season $((i+1))/${#SEASONS[@]}"
    echo "-------------------------------------------"
done

echo "All seasons processing complete!"