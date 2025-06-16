#!/usr/bin/env bash
#
# uk_tiff_pipeline.sh - Check and process UK TIFF pipeline with automatic remote transfer
#
# Usage:
#   bash uk_tiff_pipeline.sh

set -u

# Handle interruptions gracefully
trap 'echo -e "\n\nInterrupted! Showing partial results...\n"; show_summary; exit 130' INT TERM

# Function to show summary
show_summary() {
    echo -ne "\r\033[K"
    
    echo ""
    echo "Processing Status Summary:"
    echo "=========================="
    echo "  - Fully processed (on remote): ${processed_remote:-0}"
    echo "  - Processing in progress: ${processing_count:-0}"
    echo "  - Not processed: ${#not_processed[@]}"
    echo "  - Total checked: ${processed_count:-0} / ${total_tiffs:-0}"
}

# Configuration
YEAR=2024
TIFF_DIR="/home/azureuser/data/uk_tiff"
BASE_OUTPUT_DIR="/home/azureuser/data/uk_d_pixel"
REMOTE_HOST="zf281@otrera.caelum.ci.dev"
REMOTE_BASE_DIR="/tank/zf281/global_0.1_degree_tiff_d_pixel"
EXPECTED_NPY_FILES=9

echo "UK TIFF Processing Pipeline"
echo "=========================="
echo "Checking pipeline status for year ${YEAR}..."
echo ""

# Check required files exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -n "Checking required files... "
missing_files=()
[[ ! -f "${SCRIPT_DIR}/s1_s2_downloader.sh" ]] && missing_files+=("s1_s2_downloader.sh")
[[ ! -f "${SCRIPT_DIR}/s1_s2_stacker.sh" ]] && missing_files+=("s1_s2_stacker.sh")
[[ ! -f "${SCRIPT_DIR}/s1_fast_processor.py" ]] && missing_files+=("s1_fast_processor.py")
[[ ! -f "${SCRIPT_DIR}/s2_fast_processor.py" ]] && missing_files+=("s2_fast_processor.py")
[[ ! -f "${SCRIPT_DIR}/s1_stack" ]] && missing_files+=("s1_stack")
[[ ! -f "${SCRIPT_DIR}/s2_stack" ]] && missing_files+=("s2_stack")

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "MISSING FILES!"
    echo "ERROR: The following required files are missing from ${SCRIPT_DIR}:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
else
    echo "OK"
fi

# Check Python environment
echo -n "Checking Python environment... "
PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"
if [[ ! -x "$PYTHON_ENV" ]]; then
    echo "NOT FOUND!"
    echo "ERROR: Python environment not found: $PYTHON_ENV"
    echo "Please ensure the d-pixel-generation conda environment is installed"
    exit 1
else
    echo "OK"
fi

# Test SSH connection
echo -n "Testing SSH connection... "
if ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" "echo 'OK'" &>/dev/null; then
    echo "OK"
else
    echo "FAILED"
    echo "ERROR: Cannot connect to remote server. Please check SSH connection."
    exit 1
fi

# Quick check of remote directory structure
echo -n "Checking remote directory structure... "
remote_base_exists=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" \
    "test -d '${REMOTE_BASE_DIR}' && echo 'yes' || echo 'no'" 2>/dev/null || echo "no")

if [[ "$remote_base_exists" == "yes" ]]; then
    echo "OK (${REMOTE_BASE_DIR} exists)"
else
    echo "NOT FOUND"
    echo "Note: Remote base directory ${REMOTE_BASE_DIR} doesn't exist yet, will be created during processing"
fi

# Count total TIFF files
echo -n "Counting TIFF files... "
total_tiffs=$(find "$TIFF_DIR" -name "*.tiff" -type f 2>/dev/null | wc -l || echo "0")
echo "${total_tiffs} found"

if [[ "$total_tiffs" -eq 0 ]]; then
    echo "ERROR: No TIFF files found in $TIFF_DIR"
    exit 1
fi

# Initialize counters
processed_remote=0
processing_count=0
not_processed=()
processed_count=0

# Build list of TIFF files
echo "Building list of TIFF files..."
mapfile -t tiff_files < <(find "$TIFF_DIR" -name "*.tiff" -type f | sort)

# Get all processed tiles from remote server in one go
echo -n "Fetching list of processed tiles from remote server... "
remote_year_dir="${REMOTE_BASE_DIR}/${YEAR}"

# First check if remote directory exists, if not, no tiles are processed
remote_dir_exists=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" \
    "test -d '${remote_year_dir}' && echo 'yes' || echo 'no'" 2>/dev/null || echo "no")

declare -A remote_processed_set

if [[ "$remote_dir_exists" == "yes" ]]; then
    # Get list of directories on remote server that contain exactly 9 .npy files
    processed_on_remote=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" "
        cd '${remote_year_dir}' 2>/dev/null || exit
        for dir in */; do
            if [[ -d \"\$dir\" ]]; then
                dir_name=\${dir%/}
                npy_count=\$(find \"\$dir\" -maxdepth 1 -name '*.npy' -type f 2>/dev/null | wc -l)
                if [[ \"\$npy_count\" -eq ${EXPECTED_NPY_FILES} ]]; then
                    echo \"\$dir_name\"
                fi
            fi
        done
    " 2>/dev/null || echo "")
    
    # Convert to associative array for fast lookup
    if [[ -n "$processed_on_remote" ]]; then
        while IFS= read -r tile; do
            if [[ -n "$tile" ]]; then
                remote_processed_set["$tile"]=1
            fi
        done <<< "$processed_on_remote"
    fi
    echo "Found ${#remote_processed_set[@]} processed tiles"
else
    echo "Remote directory not found (normal for first run)"
fi

echo "Checking local TIFF files against remote status..."

# Now check each local TIFF file
last_percent=0
for tiff_file in "${tiff_files[@]}"; do
    tiff_name=$(basename "$tiff_file" .tiff)
    
    ((processed_count++))
    
    # Show progress based on percentage (every 5%)
    current_percent=$((processed_count * 100 / total_tiffs))
    if [[ $((current_percent / 5)) -gt $((last_percent / 5)) ]] || [[ $processed_count -eq $total_tiffs ]]; then
        echo -ne "\rProgress: ${current_percent}% (${processed_count}/${total_tiffs})                    "
        last_percent=$current_percent
    fi
    
    # Check if already processed on remote using our set
    if [[ -n "${remote_processed_set[$tiff_name]:-}" ]]; then
        ((processed_remote++))
    else
        not_processed+=("$tiff_name")
    fi
done

echo -ne "\r\033[K"  # Clear the progress line

# Show summary
show_summary

# Calculate percentage
if [[ "$total_tiffs" -gt 0 ]]; then
    percent_complete=$(( (processed_remote * 100) / total_tiffs ))
    echo ""
    echo "Remote completion: ${percent_complete}%"
fi

# Show unprocessed files
if [[ ${#not_processed[@]} -gt 0 ]]; then
    echo ""
    if [[ ${#not_processed[@]} -le 20 ]]; then
        echo "Unprocessed files:"
        for item in "${not_processed[@]}"; do
            echo "  - $item"
        done
    else
        echo "Sample of unprocessed files (showing first 20 of ${#not_processed[@]}):"
        for i in {0..19}; do
            echo "  - ${not_processed[$i]}"
        done
        echo "  ... and $((${#not_processed[@]} - 20)) more"
    fi
fi

echo ""
echo "============================================="
echo "Summary:"
echo "  - Total files: ${total_tiffs}"
echo "  - Processed on remote: ${processed_remote}"
echo "  - Need processing: ${#not_processed[@]}"
echo "============================================="

# Async transfer function
async_transfer() {
    local tile_name=$1
    local local_dir=$2
    local remote_dir=$3
    local remote_host=$4
    
    # Create a simple transfer script
    local transfer_script="${local_dir}/transfer.sh"
    cat > "$transfer_script" << EOF
#!/bin/bash
# Create remote directory
ssh "${remote_host}" "mkdir -p '${remote_dir}'" 2>/dev/null

# Transfer NPY files
rsync -aq "${local_dir}/data_processed/"*.npy "${remote_host}:${remote_dir}/" 2>/dev/null

# Clean up local files after successful transfer
if [[ \$? -eq 0 ]]; then
    rm -rf "${local_dir}/data_raw"
    rm -rf "${local_dir}/data_sar_raw"
    rm -rf "${local_dir}/data_processed"
    rm -f "${local_dir}/downloader_temp_s1.sh"
    rm -f "${local_dir}/downloader_temp_s2.sh"
    rm -f "${local_dir}/stacker_temp.sh"
fi

# Self-delete
rm -f "$transfer_script"
EOF
    
    chmod +x "$transfer_script"
    
    # Run transfer in background, detached from parent process
    nohup "$transfer_script" > /dev/null 2>&1 &
}

# Processing function with automatic transfer
process_tile_with_transfer() {
    local tile_name=$1
    local tiff_file="${TIFF_DIR}/${tile_name}.tiff"
    local local_dir="${BASE_OUTPUT_DIR}/${YEAR}/${tile_name}"
    local remote_dir="${REMOTE_BASE_DIR}/${YEAR}/${tile_name}"
    local log_file="${local_dir}/processing.log"
    local script_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting processing for tile: $tile_name"
    
    # Create directories
    mkdir -p "$local_dir"
    
    # Create processing script
    local process_script="${local_dir}/process.sh"
    cat > "$process_script" << 'EOF'
#!/bin/bash
set -u

TILE_NAME="$1"
TIFF_FILE="$2"
LOCAL_DIR="$3"
REMOTE_DIR="$4"
REMOTE_HOST="$5"
YEAR="$6"
LOG_FILE="$7"
SCRIPT_SOURCE_DIR="$8"

# Initialize exit codes
S1_DOWNLOAD_EXIT_CODE=1
S2_DOWNLOAD_EXIT_CODE=1
STACKER_EXIT_CODE=1

# Redirect all output to log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Processing tile: $TILE_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Step 1: Download Sentinel-1 data
echo ""
echo "Step 1: Downloading Sentinel-1 data..."

# Use the correct script directory
cd "${SCRIPT_SOURCE_DIR}"
if [[ ! -f "s1_s2_downloader.sh" ]]; then
    echo "ERROR: s1_s2_downloader.sh not found in ${SCRIPT_SOURCE_DIR}"
    exit 1
fi

# Check for required Python scripts
if [[ ! -f "s1_fast_processor.py" ]] || [[ ! -f "s2_fast_processor.py" ]]; then
    echo "ERROR: Required Python scripts (s1_fast_processor.py, s2_fast_processor.py) not found in ${SCRIPT_SOURCE_DIR}"
    exit 1
fi

# Check Python environment exists
PYTHON_ENV="/home/azureuser/miniconda3/envs/d-pixel-generation/bin/python"
if [[ ! -x "$PYTHON_ENV" ]]; then
    echo "ERROR: Python environment not found or not executable: $PYTHON_ENV"
    echo "Please ensure the d-pixel-generation conda environment is installed"
    exit 1
fi

# First download Sentinel-1
cp s1_s2_downloader.sh "${LOCAL_DIR}/downloader_temp_s1.sh"

# Update the script parameters for S1 only
sed -i "s|INPUT_TIFF=\".*\"|INPUT_TIFF=\"${TIFF_FILE}\"|" "${LOCAL_DIR}/downloader_temp_s1.sh"
sed -i "s|OUT_DIR=\".*\"|OUT_DIR=\"${LOCAL_DIR}\"|" "${LOCAL_DIR}/downloader_temp_s1.sh"
sed -i "s|S1_ENABLED=false|S1_ENABLED=true|" "${LOCAL_DIR}/downloader_temp_s1.sh"
sed -i "s|S2_ENABLED=true|S2_ENABLED=false|" "${LOCAL_DIR}/downloader_temp_s1.sh"

echo "Running downloader with S1_ENABLED=true and S2_ENABLED=false"
echo "--------------------------------------------------------"

# Run the S1 downloader from the source directory
cd "${SCRIPT_SOURCE_DIR}"
bash "${LOCAL_DIR}/downloader_temp_s1.sh"
S1_DOWNLOAD_EXIT_CODE=$?

if [[ $S1_DOWNLOAD_EXIT_CODE -eq 0 ]]; then
    echo "Sentinel-1 download completed successfully"
    
    # Check what was downloaded
    if [[ -d "${LOCAL_DIR}/data_sar_raw" ]]; then
        s1_count=$(find "${LOCAL_DIR}/data_sar_raw" -name "*.tiff" -type f | wc -l)
        echo "  - Sentinel-1 files: $s1_count"
    else
        echo "  - Sentinel-1: No data directory found"
    fi
else
    echo "ERROR: Sentinel-1 download failed"
    exit 1
fi

# Step 2: Download Sentinel-2 data
echo ""
echo "Step 2: Downloading Sentinel-2 data..."

cp s1_s2_downloader.sh "${LOCAL_DIR}/downloader_temp_s2.sh"

# Update the script parameters for S2 only
sed -i "s|INPUT_TIFF=\".*\"|INPUT_TIFF=\"${TIFF_FILE}\"|" "${LOCAL_DIR}/downloader_temp_s2.sh"
sed -i "s|OUT_DIR=\".*\"|OUT_DIR=\"${LOCAL_DIR}\"|" "${LOCAL_DIR}/downloader_temp_s2.sh"
sed -i "s|S1_ENABLED=false|S1_ENABLED=false|" "${LOCAL_DIR}/downloader_temp_s2.sh"
sed -i "s|S2_ENABLED=true|S2_ENABLED=true|" "${LOCAL_DIR}/downloader_temp_s2.sh"

echo "Running downloader with S1_ENABLED=false and S2_ENABLED=true"
echo "--------------------------------------------------------"

# Run the S2 downloader from the source directory
cd "${SCRIPT_SOURCE_DIR}"
bash "${LOCAL_DIR}/downloader_temp_s2.sh"
S2_DOWNLOAD_EXIT_CODE=$?

if [[ $S2_DOWNLOAD_EXIT_CODE -eq 0 ]]; then
    echo "Sentinel-2 download completed successfully"
    
    # Check what was downloaded
    if [[ -d "${LOCAL_DIR}/data_raw" ]]; then
        s2_count=$(find "${LOCAL_DIR}/data_raw" -type f | wc -l)
        echo "  - Sentinel-2 files: $s2_count"
    else
        echo "  - Sentinel-2: No data directory found"
    fi
else
    echo "ERROR: Sentinel-2 download failed"
    exit 1
fi

# Step 3: Stack the data
echo ""
echo "Step 3: Stacking Sentinel-1 and Sentinel-2 data..."

cp "${SCRIPT_SOURCE_DIR}/s1_s2_stacker.sh" "${LOCAL_DIR}/stacker_temp.sh"
sed -i "s|BASE_DIR=\".*\"|BASE_DIR=\"${LOCAL_DIR}\"|" "${LOCAL_DIR}/stacker_temp.sh"

# Make sure s1_stack and s2_stack binaries are accessible
if [[ -f "${SCRIPT_SOURCE_DIR}/s1_stack" ]] && [[ -f "${SCRIPT_SOURCE_DIR}/s2_stack" ]]; then
    # Copy the binaries to local dir temporarily
    cp "${SCRIPT_SOURCE_DIR}/s1_stack" "${LOCAL_DIR}/"
    cp "${SCRIPT_SOURCE_DIR}/s2_stack" "${LOCAL_DIR}/"
    chmod +x "${LOCAL_DIR}/s1_stack" "${LOCAL_DIR}/s2_stack"
    
    # Run stacker from the local directory
    cd "${LOCAL_DIR}"
    bash "./stacker_temp.sh"
    STACKER_EXIT_CODE=$?
    
    # Clean up binaries
    rm -f "${LOCAL_DIR}/s1_stack" "${LOCAL_DIR}/s2_stack"
else
    echo "ERROR: s1_stack or s2_stack binary not found in ${SCRIPT_SOURCE_DIR}"
    exit 2
fi

if [[ $STACKER_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Stacking completed successfully"
    
    # Verify NPY files were created
    local_count=$(find "${LOCAL_DIR}/data_processed" -name "*.npy" -type f 2>/dev/null | wc -l)
    
    if [[ "$local_count" -eq 9 ]]; then
        echo "Verified: $local_count NPY files created"
        
        echo ""
        echo "=========================================="
        echo "SUCCESS: Tile processing completed!"
        echo "End time: $(date)"
        echo "=========================================="
        exit 0
    else
        echo "ERROR: Expected 9 NPY files but found $local_count"
        exit 3
    fi
else
    echo "ERROR: Stacking failed"
    exit 2
fi
EOF
    
    chmod +x "$process_script"
    
    # Execute the processing script with script source directory
    "$process_script" "$tile_name" "$tiff_file" "$local_dir" "$remote_dir" "$REMOTE_HOST" "$YEAR" "$log_file" "$script_source_dir"
    local exit_code=$?
    
    # If processing succeeded, start async transfer
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: Processing completed for $tile_name, starting background transfer"
        async_transfer "$tile_name" "$local_dir" "$remote_dir" "$REMOTE_HOST"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: Processing failed for $tile_name"
    fi
    
    return $exit_code
}

# Ask if user wants to process unprocessed files
if [[ ${#not_processed[@]} -gt 0 ]]; then
    echo ""
    echo "============================================="
    echo "PROCESSING OPTIONS"
    echo "============================================="
    echo ""
    echo "Found ${#not_processed[@]} tiles that need processing"
    echo ""
    echo "Options:"
    echo "  1) Process all unprocessed tiles"
    echo "  2) Process a specific tile"
    echo "  3) Exit"
    echo ""
    echo -n "Select option (1-3): "
    read -r option
    
    case $option in
        1)
            echo ""
            echo -n "How many tiles to process in parallel? (1-4, default=2): "
            read -r max_parallel
            max_parallel=${max_parallel:-2}
            
            echo ""
            echo "Processing ${#not_processed[@]} tiles with $max_parallel parallel jobs..."
            echo "Note: Transfers will happen in the background automatically"
            echo ""
            
            # Function to monitor active jobs
            monitor_jobs() {
                local max_jobs=$1
                while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
                    sleep 2
                done
            }
            
            # Process tiles
            active_jobs=0
            completed=0
            failed=0
            TAIL_PID=""
            
            for tile_name in "${not_processed[@]}"; do
                monitor_jobs $max_parallel
                
                echo "[$(date '+%H:%M:%S')] Starting: $tile_name ($(($completed + $failed + 1))/${#not_processed[@]})"
                
                # Start processing in background
                (
                    if process_tile_with_transfer "$tile_name"; then
                        exit 0
                    else
                        exit 1
                    fi
                ) &
                
                # Show log output for the first tile to give user feedback
                if [[ $completed -eq 0 ]] && [[ $failed -eq 0 ]] && [[ -z "$TAIL_PID" ]]; then
                    sleep 3
                    log_file="${BASE_OUTPUT_DIR}/${YEAR}/${tile_name}/processing.log"
                    if [[ -f "$log_file" ]]; then
                        echo ""
                        echo ""
                        echo "Showing processing output for first tile (${tile_name}):"
                        echo "--------------------------------------------------------"
                        tail -f "$log_file" 2>/dev/null &
                        TAIL_PID=$!
                        
                        # Stop tailing after 30 seconds
                        (sleep 30 && kill $TAIL_PID 2>/dev/null) &
                    fi
                fi
            done
            
            echo ""
            echo "All jobs launched. Waiting for completion..."
            echo "You can monitor individual tile logs in: ${BASE_OUTPUT_DIR}/${YEAR}/*/processing.log"
            wait
            
            # Kill tail process if still running
            [[ ! -z ${TAIL_PID:-} ]] && kill $TAIL_PID 2>/dev/null
            
            echo ""
            echo "============================================="
            echo "Processing complete!"
            echo "Transfers are happening in the background."
            echo "Check individual logs for details."
            echo "============================================="
            ;;
            
        2)
            echo ""
            echo "Enter the tile name (e.g., grid_-3.45_51.65): "
            read -r tile_name
            
            if [[ " ${not_processed[@]} " =~ " ${tile_name} " ]]; then
                echo "Processing single tile: $tile_name"
                
                log_file="${BASE_OUTPUT_DIR}/${YEAR}/${tile_name}/processing.log"
                mkdir -p "$(dirname "$log_file")"
                
                # Show live output
                process_tile_with_transfer "$tile_name" 2>&1 | tee "$log_file"
                
                if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                    echo "SUCCESS: Tile processed successfully. Transfer will happen in background."
                else
                    echo "FAILED: Check log at $log_file"
                fi
            else
                echo "Tile not found in unprocessed list"
            fi
            ;;
            
        3)
            echo "Exiting..."
            ;;
            
        *)
            echo "Invalid option"
            ;;
    esac
fi

echo ""
echo "Script completed at: $(date)"
echo "Note: Background transfers may still be running. Check remote server later."