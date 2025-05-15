#!/bin/bash

# Individual steps will check for errors and 'return 1' from the function,
# and the main script will track overall success/failure.

set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that failed

# --- Configuration & Logging ---
SCRIPT_DIR_TOOLKIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CONFIG_FILE="${SCRIPT_DIR_TOOLKIT}/config.sh"
LOG_DIR_TOOLKIT="${SCRIPT_DIR_TOOLKIT}/toolkit_logs"
MAIN_LOG_FILE="${LOG_DIR_TOOLKIT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
PYTHON_ESTIMATE_TIME="${SCRIPT_DIR_TOOLKIT}/btfm_tools/estimate_time.py"

mkdir -p "$LOG_DIR_TOOLKIT"

# Redirect all stdout and stderr to the main log file and also to console
exec > >(tee -a "${MAIN_LOG_FILE}") 2>&1

echo "Starting BTFM Input Processing Pipeline: $(date)"
echo "Toolkit Script Directory: ${SCRIPT_DIR_TOOLKIT}"
echo "Main Log File: ${MAIN_LOG_FILE}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file config.sh not found in ${SCRIPT_DIR_TOOLKIT}."
    echo "Please copy config.sh.template to config.sh and fill in your details."
    exit 1
fi
source "$CONFIG_FILE"
echo "Configuration loaded from $CONFIG_FILE"

# --- Validate Configuration ---
if [ -z "$BTFM_PROJECT_DIR" ] || [ ! -d "$BTFM_PROJECT_DIR" ]; then
    echo "ERROR: BTFM_PROJECT_DIR ('${BTFM_PROJECT_DIR}') is not set or is not a valid directory in config.sh."
    exit 1
fi
if [ -z "$INFERENCE_PYTHON_ENV" ] || [ ! -x "$INFERENCE_PYTHON_ENV" ]; then # Check if executable
    echo "ERROR: INFERENCE_PYTHON_ENV ('${INFERENCE_PYTHON_ENV}') is not set or is not a valid Python executable in config.sh."
    exit 1
fi
if [ -z "$ASF_USERNAME" ] || [ "$ASF_USERNAME" == "your_asf_username" ] || [ "$ASF_USERNAME" == "xxxxxx" ]; then
    echo "ERROR: ASF_USERNAME is not set or is still the default placeholder in config.sh."
    exit 1
fi
if [ -z "$ASF_PASSWORD" ] || [ "$ASF_PASSWORD" == "your_asf_password" ] || [ "$ASF_PASSWORD" == "xxxxxx" ]; then
    echo "ERROR: ASF_PASSWORD is not set or is still the default placeholder in config.sh."
    exit 1
fi
CHECKPOINT_FILE_NAME=${CHECKPOINT_FILE_NAME:-"best_model_fsdp_20250427_084307.pt"}
MODEL_CHECKPOINT="${BTFM_PROJECT_DIR}/btfm_infer/checkpoints/${CHECKPOINT_FILE_NAME}"
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint ${MODEL_CHECKPOINT} not found."
    echo "Please download it and place it in ${BTFM_PROJECT_DIR}/btfm_infer/checkpoints/"
    exit 1
fi

PYTHON_ROI_PROCESSOR="${SCRIPT_DIR_TOOLKIT}/btfm_tools/roi_processor.py"
if [ ! -f "$PYTHON_ROI_PROCESSOR" ]; then
    echo "ERROR: ROI processor script not found at ${PYTHON_ROI_PROCESSOR}"
    exit 1
fi

# --- Argument Parsing ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir_or_file> <output_base_dir>"
    echo "  <input_dir_or_file>: Directory containing ROI files (e.g., *.shp, *.geojson, *.tif, area_bbox.txt)"
    echo "                         OR a single ROI file path."
    echo "  <output_base_dir>: Base directory where results will be stored."
    exit 1
fi

INPUT_SPECIFIER=$(realpath "$1" 2>/dev/null || echo "$1") # Handle relative paths, allow non-existent for now
OUTPUT_BASE_DIR=$(realpath "$2" 2>/dev/null || echo "$2")

if ! mkdir -p "$OUTPUT_BASE_DIR"; then
    echo "ERROR: Could not create or access output base directory: $OUTPUT_BASE_DIR"
    exit 1
fi
OUTPUT_BASE_DIR=$(realpath "$OUTPUT_BASE_DIR") # Get absolute path after creation

REPRESENTATIONS_DIR="${OUTPUT_BASE_DIR}/representations"
ROI_TIFFS_GENERATED_DIR="${OUTPUT_BASE_DIR}/roi_tiffs_generated" # Generated ROI tiffs
WORKING_DATA_BASE_DIR="${OUTPUT_BASE_DIR}/working_data" # For main_pipeline.sh BASE_DATA_DIR

mkdir -p "$REPRESENTATIONS_DIR"
mkdir -p "$ROI_TIFFS_GENERATED_DIR"
mkdir -p "$WORKING_DATA_BASE_DIR"

echo "Input Specifier: $INPUT_SPECIFIER"
echo "Output Base Directory: $OUTPUT_BASE_DIR"
echo "Representations will be saved to: $REPRESENTATIONS_DIR"
echo "Generated ROI TIFFs (for reference) will be saved to: $ROI_TIFFS_GENERATED_DIR"
echo "Working data will be stored under: $WORKING_DATA_BASE_DIR"

# --- Determine Data Processing Time Window ---
S2_START_TIME=""
S2_END_TIME=""
S1_START_TIME=""
S1_END_TIME=""
DOWNSAMPLE_RATE=1 # For 10m resolution

use_custom_window_lower=$(echo "$USE_CUSTOM_TIME_WINDOW" | tr '[:upper:]' '[:lower:]')

if [ "$use_custom_window_lower" == "true" ]; then
    echo "INFO: Using custom time window."
    # Validate custom date components
    if ! [[ "$CUSTOM_START_MONTH" =~ ^(0[1-9]|1[0-2])$ && \
            "$CUSTOM_START_DAY" =~ ^(0[1-9]|[12][0-9]|3[01])$ && \
            "$CUSTOM_END_MONTH" =~ ^(0[1-9]|1[0-2])$ && \
            "$CUSTOM_END_DAY" =~ ^(0[1-9]|[12][0-9]|3[01])$ && \
            "$CUSTOM_END_YEAR_OFFSET" =~ ^[0-1]$ ]]; then
        echo "ERROR: Invalid custom month, day, or year offset in config.sh. Please use MM, DD, and 0 or 1."
        exit 1
    fi

    S2_START_TIME="${PROCESSING_YEAR}-${CUSTOM_START_MONTH}-${CUSTOM_START_DAY}"
    
    # Calculate end year
    end_year=$((PROCESSING_YEAR + CUSTOM_END_YEAR_OFFSET))
    S2_END_TIME="${end_year}-${CUSTOM_END_MONTH}-${CUSTOM_END_DAY}"

    # Basic validation: Check if start date is valid
    if ! date -d "$S2_START_TIME" >/dev/null 2>&1; then
        echo "ERROR: Invalid custom start date generated: $S2_START_TIME. Check PROCESSING_YEAR, CUSTOM_START_MONTH, CUSTOM_START_DAY."
        exit 1
    fi
    # Basic validation: Check if end date is valid
    if ! date -d "$S2_END_TIME" >/dev/null 2>&1; then
        echo "ERROR: Invalid custom end date generated: $S2_END_TIME. Check PROCESSING_YEAR, CUSTOM_END_MONTH, CUSTOM_END_DAY, CUSTOM_END_YEAR_OFFSET."
        exit 1
    fi
    # Basic validation: Check if end date is after start date
    # Convert dates to seconds since epoch for comparison
    start_seconds=$(date -d "$S2_START_TIME" +%s)
    end_seconds=$(date -d "$S2_END_TIME" +%s)
    if [ "$end_seconds" -le "$start_seconds" ]; then
        echo "ERROR: Custom end date ($S2_END_TIME) must be after custom start date ($S2_START_TIME)."
        exit 1
    fi

    S1_START_TIME="$S2_START_TIME"
    S1_END_TIME="$S2_END_TIME"
else
    echo "INFO: Using full year for processing based on PROCESSING_YEAR."
    S2_START_TIME="${PROCESSING_YEAR}-01-01"
    # Calculate next year for the end date (exclusive)
    next_year=$((PROCESSING_YEAR + 1))
    S2_END_TIME="${next_year}-01-01"
    
    S1_START_TIME="$S2_START_TIME"
    S1_END_TIME="$S2_END_TIME"
fi

echo "Data processing time window set to: ${S2_START_TIME} to ${S2_END_TIME}"
echo "Sentinel-1 time window set to: ${S1_START_TIME} to ${S1_END_TIME}"

# --- Function to process a single ROI file ---
process_single_roi() {
    local roi_input_file="$1"
    local roi_basename_full=$(basename "$roi_input_file")
    local roi_name_sanitized=$(echo "$roi_basename_full" | sed 's/[^a-zA-Z0-9._-]/_/g')
    local roi_name_no_ext="${roi_name_sanitized%.*}"
    
    local roi_ext_lower=$(echo "${roi_basename_full##*.}" | tr '[:upper:]' '[:lower:]')

    echo ""
    echo "======================================================================"
    echo "Processing Input: $roi_basename_full (Sanitized name: $roi_name_no_ext)"
    echo "Input Path: $roi_input_file"
    echo "Timestamp: $(date)"
    echo "======================================================================"

    # Define paths for this specific ROI
    DATA_DIR="${WORKING_DATA_BASE_DIR}/${roi_name_no_ext}"
    # Paths INSIDE THE DOCKER CONTAINER for ROI processing
    local roi_input_file_container_path="/mnt/input_roi_file/${roi_basename_full}"
    local roi_tiff_for_pipeline_container_path="/mnt/my_data_dir_container/roi.tiff"
    # This is the standardized ROI TIFF that the rest of the pipeline uses
    ROI_TIFF_FOR_PIPELINE="${DATA_DIR}/roi.tiff" 
    # This is a copy of the generated/processed ROI TIFF for user reference
    GENERATED_ROI_TIFF_FOR_USER="${ROI_TIFFS_GENERATED_DIR}/${roi_name_no_ext}_roi_10m.tiff"
    
    FINAL_REPRESENTATION_PATH="${REPRESENTATIONS_DIR}/${roi_name_no_ext}_representation.npy"
    LOG_SUBDIR_TOOLKIT="${LOG_DIR_TOOLKIT}/${roi_name_no_ext}"
    
    # Clean up previous attempt's working data if it exists and failed partway
    if [ -d "$DATA_DIR" ] && [ ! -f "$FINAL_REPRESENTATION_PATH" ]; then
        echo "INFO: Previous incomplete working data found for '$roi_name_no_ext' at '$DATA_DIR'. Clearing it up."
        rm -rf "$DATA_DIR"/* # Clear contents, but keep dir for logs if it was created already.
    fi
    mkdir -p "$DATA_DIR" # Ensure it exists
    mkdir -p "$LOG_SUBDIR_TOOLKIT" # Ensure log dir exists

    if [ -f "$FINAL_REPRESENTATION_PATH" ]; then
        echo "INFO: Final representation ${FINAL_REPRESENTATION_PATH} already exists. Skipping."
        return 0 # Success (already done)
    fi

	
    # 1. Convert/Prepare ROI_TIFF_FOR_PIPELINE
    echo "[Step 1/4] Preparing 10m ROI TIFF for $roi_basename_full..."
    local input_type_arg=""

    if [[ "$roi_ext_lower" == "shp" ]]; then
        input_type_arg="shp"
    elif [[ "$roi_ext_lower" == "geojson" || "$roi_ext_lower" == "json" ]]; then
        input_type_arg="geojson"
    elif [[ "$roi_ext_lower" == "tif" || "$roi_ext_lower" == "tiff" ]]; then
        input_type_arg="tiff"
    elif [[ "$roi_ext_lower" == "txt" ]]; then
        if grep -qE '^-?[0-9. ]+[, ]+-?[0-9. ]+[, ]+-?[0-9. ]+[, ]+-?[0-9. ]+$' "$roi_input_file"; then
             input_type_arg="bbox"
        else
            echo "WARNING: File $roi_input_file is .txt but does not appear to be a BBox coordinate file (expected: min_lon,min_lat,max_lon,max_lat). Skipping."
            return 1 # Failure
        fi
    else
        echo "WARNING: Unknown or unsupported file type for $roi_input_file (extension: .$roi_ext_lower). Skipping."
        return 1 # Failure
    fi

    (
      set -e # Enable strict error checking for this subshell
      "$INFERENCE_PYTHON_ENV" "${PYTHON_ROI_PROCESSOR}" \
        --input_path "$roi_input_file" \
        --output_tiff_path "$ROI_TIFF_FOR_PIPELINE" \
        --input_type "$input_type_arg" \
        --pixel_size 10
    ) > "${LOG_SUBDIR_TOOLKIT}/1_roi_preparation.log" 2>&1

    if [ ! -f "$ROI_TIFF_FOR_PIPELINE" ]; then
        echo "ERROR: Failed to generate standardized ROI TIFF for $roi_name_no_ext. Check log: ${LOG_SUBDIR_TOOLKIT}/1_roi_preparation.log"
        return 1 # Failure
    fi
    cp "$ROI_TIFF_FOR_PIPELINE" "$GENERATED_ROI_TIFF_FOR_USER"
    echo "INFO: Standardized 10m ROI TIFF prepared: $ROI_TIFF_FOR_PIPELINE (and copied to $GENERATED_ROI_TIFF_FOR_USER)"

    # Define Docker command for ROI processing
	# This should work, but we need a docker image that actually has all the python packages; snap-gdal-python:v3 doesn't have fiona
	
    # (
      # set -e # Strict error checking for this subshell
      # docker run --rm \
        # -v "$(dirname "$roi_input_file"):/mnt/input_roi_file:ro" \
        # -v "${DATA_DIR}:/mnt/my_data_dir_container:rw" \
        # -v "${SCRIPT_DIR_TOOLKIT}/btfm_tools:/mnt/btfm_tools:ro" \
        # frankfeng1223/snap-gdal-python:v3 \
        # /mnt/btfm_tools/roi_processor.py \
          # --input_path "$roi_input_file_container_path" \
          # --output_tiff_path "$roi_tiff_for_pipeline_container_path" \
          # --input_type "$input_type_arg" \
          # --pixel_size 10
    # ) > "${LOG_SUBDIR_TOOLKIT}/1_roi_preparation.log" 2>&1
    
    # local roi_prep_exit_code=$?
    # if [ $roi_prep_exit_code -ne 0 ] || [ ! -f "$ROI_TIFF_FOR_PIPELINE_HOST_PATH" ]; then
        # echo "ERROR: Failed to generate standardized ROI TIFF for $roi_name_no_ext (Exit Code: $roi_prep_exit_code). Check log: ${LOG_SUBDIR_TOOLKIT}/1_roi_preparation.log"
        # return 1
    # fi
    # cp "$ROI_TIFF_FOR_PIPELINE_HOST_PATH" "$GENERATED_ROI_TIFF_FOR_USER_HOST_PATH"
    # echo "INFO: Standardized 10m ROI TIFF prepared: $ROI_TIFF_FOR_PIPELINE_HOST_PATH (and copied to $GENERATED_ROI_TIFF_FOR_USER_HOST_PATH)"

    # --- Time Estimate ---
	
	
	# Calculate window months for estimation
    # This is a rough approximation of months in the interval
    local s_year=$(echo "$S2_START_TIME" | cut -d'-' -f1)
    local s_month=$(echo "$S2_START_TIME" | cut -d'-' -f2 | sed 's/^0*//') # remove leading zeros
    local e_year=$(echo "$S2_END_TIME" | cut -d'-' -f1)
    local e_month=$(echo "$S2_END_TIME" | cut -d'-' -f2 | sed 's/^0*//') # remove leading zeros
    local e_day=$(echo "$S2_END_TIME" | cut -d'-' -f3 | sed 's/^0*//')

    # If end day is 1, it means exclusive of that month, so count up to previous month
    if [ "$e_day" -eq 1 ]; then
        e_month=$((e_month - 1))
        if [ "$e_month" -eq 0 ]; then
            e_month=12
            e_year=$((e_year - 1))
        fi
    fi
    local window_total_months=$(((e_year - s_year) * 12 + e_month - s_month + 1))
    if [ $window_total_months -le 0 ]; then window_total_months=1; fi # Min 1 month

	echo "INFO: Discovering MGRS tiles for refined estimate..."
	TEMP_TILE_LIST_FILE_HOST="${DATA_DIR}/temp_tile_list.txt"

	# Host paths
	HOST_PREPROCESSING_DIR="${BTFM_PROJECT_DIR}/btfm_preprocessing"
	HOST_ROI_TIFF_DIR=$(dirname "$ROI_TIFF_FOR_PIPELINE")
	ROI_TIFF_BASENAME=$(basename "$ROI_TIFF_FOR_PIPELINE")
	HOST_OUTPUT_DIR=$(dirname "$TEMP_TILE_LIST_FILE_HOST")
	TILE_LIST_BASENAME=$(basename "$TEMP_TILE_LIST_FILE_HOST")

	# Container mount points
	# Let's make the script mount point /data to align with main_pipeline.sh
	CONTAINER_SCRIPTS_MNT_IS_DATA="/data" 
	CONTAINER_ROI_INPUT_MNT="/mnt_roi_input"
	CONTAINER_TILE_LIST_OUTPUT_MNT="/mnt_tile_list_output"

	# Paths inside the container
	CONTAINER_S2_DISCOVER_PY_PATH="${CONTAINER_SCRIPTS_MNT_IS_DATA}/s2_discover_tiles.py" # Now /data/s2_discover_tiles.py
	CONTAINER_ROI_TIFF_PATH="${CONTAINER_ROI_INPUT_MNT}/${ROI_TIFF_BASENAME}"
	CONTAINER_TILE_LIST_PATH="${CONTAINER_TILE_LIST_OUTPUT_MNT}/${TILE_LIST_BASENAME}"

	DOCKER_PYTHON_EXEC="python"

	#echo "DEBUG: Docker s2_discover_tiles call details (using /data for scripts):"
	#echo "  Host Preprocessing Dir: $HOST_PREPROCESSING_DIR  -> Container: $CONTAINER_SCRIPTS_MNT_IS_DATA"
	#echo "  Container s2_discover_tiles.py path: $CONTAINER_S2_DISCOVER_PY_PATH"

	docker_s2_discover_cmd=(
	  docker run --rm \
	  -v "${HOST_PREPROCESSING_DIR}:${CONTAINER_SCRIPTS_MNT_IS_DATA}:ro" \
	  -v "${HOST_ROI_TIFF_DIR}:${CONTAINER_ROI_INPUT_MNT}:ro" \
	  -v "${HOST_OUTPUT_DIR}:${CONTAINER_TILE_LIST_OUTPUT_MNT}:rw" \
	  frankfeng1223/snap-gdal-python:v3 \
	  /data/s2_discover_tiles.py \
	  --tiff_path "$CONTAINER_ROI_TIFF_PATH" \
	  --out_tiles "$CONTAINER_TILE_LIST_PATH"
	)

	# echo "DEBUG: Executing Docker command (for estimation): ${docker_s2_discover_cmd[*]}"

	(
	  "${docker_s2_discover_cmd[@]}"
	) > "${LOG_SUBDIR_TOOLKIT}/estimate_s2_discover_docker.log" 2>&1

	num_mgrs_tiles=0

	# Check HOST path for the output file
	if [ -f "$TEMP_TILE_LIST_FILE_HOST" ]; then
		if [ -s "$TEMP_TILE_LIST_FILE_HOST" ]; then # Check if not empty
			num_mgrs_tiles=$(wc -l < "$TEMP_TILE_LIST_FILE_HOST" | awk '{print $1}')
		else
			num_mgrs_tiles=0 
			echo "INFO: s2_discover_tiles.py produced an empty tile list file: '$TEMP_TILE_LIST_FILE_HOST'."
		fi
	else
		echo "WARNING: s2_discover_tiles.py (for estimation) failed (exit code $s2_discover_exit_code) or did not produce '$TEMP_TILE_LIST_FILE_HOST'."
		# echo "DEBUG: Docker execution log available at ${LOG_SUBDIR_TOOLKIT}/estimate_s2_discover_docker.log. Check this log for Python errors inside Docker."
		echo "WARNING: Could not determine number of MGRS tiles. Preprocessing estimate may be inaccurate."
	fi
	echo "INFO: Found $num_mgrs_tiles MGRS tiles intersecting the ROI (via Docker estimation)."

    # echo "INFO: Found $num_mgrs_tiles MGRS tiles intersecting the ROI." # Dimensions will be printed by Python script
    echo "INFO: Calculating refined time estimate..."
    (
      "$INFERENCE_PYTHON_ENV" "$PYTHON_ESTIMATE_TIME" \
        --config_dir_path "$SCRIPT_DIR_TOOLKIT" \
        --roi_tiff_path "$ROI_TIFF_FOR_PIPELINE" \
        --num_mgrs_tiles "$num_mgrs_tiles" \
        --window_months "$window_total_months" \
        --inf_cpu_gpu_split "${INFERENCE_CPU_GPU_SPLIT:-1:1}" \
        --inf_max_cpu "${INFERENCE_MAX_CPU_PROCESSES:-20}" \
        --inf_max_gpu "${INFERENCE_MAX_GPU_PROCESSES:-1}"
    )

    # 2. Run Data Preprocessing (main_pipeline.sh)
    echo "[Step 2/4] Running data preprocessing (Sentinel-1 & Sentinel-2) for $roi_name_no_ext..."
    PREPROCESSING_SCRIPT="${BTFM_PROJECT_DIR}/btfm_preprocessing/main_pipeline.sh"
    TEMP_PREPROCESSING_SCRIPT="${DATA_DIR}/tmp_main_pipeline_$(date +%s).sh" # Unique temp script name
    
    cp "$PREPROCESSING_SCRIPT" "$TEMP_PREPROCESSING_SCRIPT"
    sed -i.bak "s|^BASE_DATA_DIR=.*|BASE_DATA_DIR=\"${DATA_DIR}\"|" "$TEMP_PREPROCESSING_SCRIPT"
    sed -i.bak "s|^S1_DOWNLOAD_USERNAME=.*|S1_DOWNLOAD_USERNAME=\"${ASF_USERNAME}\"|" "$TEMP_PREPROCESSING_SCRIPT"
    sed -i.bak "s|^S1_DOWNLOAD_PASSWORD=.*|S1_DOWNLOAD_PASSWORD=\"${ASF_PASSWORD}\"|" "$TEMP_PREPROCESSING_SCRIPT"
    # Ensure the Docker volume mount for /data points to the btfm_preprocessing directory for RUST_S2_EXE
    sed -i.bak "s|-v \${BASE_SCRIPT_DIR}:/data|-v ${BTFM_PROJECT_DIR}/btfm_preprocessing:/data|" "$TEMP_PREPROCESSING_SCRIPT"
    rm -f "${TEMP_PREPROCESSING_SCRIPT}.bak" # Clean up sed backup

    chmod +x "$TEMP_PREPROCESSING_SCRIPT"
    (
      cd "${BTFM_PROJECT_DIR}/btfm_preprocessing" && \
      bash "$TEMP_PREPROCESSING_SCRIPT" \
        "$ROI_TIFF_FOR_PIPELINE" \
        "$S2_START_TIME" \
        "$S2_END_TIME" \
        "$S1_START_TIME" \
        "$S1_END_TIME" \
        "$DOWNSAMPLE_RATE"
    ) > "${LOG_SUBDIR_TOOLKIT}/2_preprocessing.log" 2>&1
    
    local prep_exit_code=$?
    rm "$TEMP_PREPROCESSING_SCRIPT"
    if [ $prep_exit_code -ne 0 ]; then
        echo "ERROR: Data preprocessing failed for $roi_name_no_ext with exit code $prep_exit_code. Check log: ${LOG_SUBDIR_TOOLKIT}/2_preprocessing.log"
        return 1
    fi
    echo "INFO: Data preprocessing finished for $roi_name_no_ext."
    if [ ! -d "${DATA_DIR}/retiled_d_pixel" ] || [ -z "$(ls -A "${DATA_DIR}/retiled_d_pixel" 2>/dev/null)" ]; then
        echo "ERROR: Preprocessing output (retiled_d_pixel) missing or empty for $roi_name_no_ext. Check log: ${LOG_SUBDIR_TOOLKIT}/2_preprocessing.log"
        return 1
    fi

    # 3. Run Inference (infer_all_tiles.sh)
    echo "[Step 3/4] Running inference for $roi_name_no_ext..."
    INFERENCE_SCRIPT="${BTFM_PROJECT_DIR}/btfm_infer/infer_all_tiles.sh"
    TEMP_INFERENCE_SCRIPT="${DATA_DIR}/tmp_infer_all_tiles_$(date +%s).sh" # Unique temp script name
    
    cp "$INFERENCE_SCRIPT" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^BASE_DATA_DIR=.*|BASE_DATA_DIR=\"${DATA_DIR}\"|" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^export PYTHON_ENV=.*|export PYTHON_ENV=\"${INFERENCE_PYTHON_ENV}\"|" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^CPU_GPU_SPLIT=.*|CPU_GPU_SPLIT=\"${INFERENCE_CPU_GPU_SPLIT:-1:1}\"|" "$TEMP_INFERENCE_SCRIPT" # Provide default
    sed -i.bak "s|^MAX_CONCURRENT_PROCESSES_CPU=.*|MAX_CONCURRENT_PROCESSES_CPU=${INFERENCE_MAX_CPU_PROCESSES:-20}|" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^MAX_CONCURRENT_PROCESSES_GPU=.*|MAX_CONCURRENT_PROCESSES_GPU=${INFERENCE_MAX_GPU_PROCESSES:-1}|" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^GPU_BATCH_SIZE=.*|GPU_BATCH_SIZE=${INFERENCE_GPU_BATCH_SIZE:-1024}|" "$TEMP_INFERENCE_SCRIPT"
    sed -i.bak "s|^CHECKPOINT_PATH=.*|CHECKPOINT_PATH=\"${MODEL_CHECKPOINT}\"|" "$TEMP_INFERENCE_SCRIPT"
    rm -f "${TEMP_INFERENCE_SCRIPT}.bak"

    chmod +x "$TEMP_INFERENCE_SCRIPT"
    (
      cd "${BTFM_PROJECT_DIR}/btfm_infer" && \
      bash "$TEMP_INFERENCE_SCRIPT"
    ) > "${LOG_SUBDIR_TOOLKIT}/3_inference.log" 2>&1
    
    local infer_exit_code=$?
    rm "$TEMP_INFERENCE_SCRIPT"
    if [ $infer_exit_code -ne 0 ]; then
        echo "ERROR: Inference failed for $roi_name_no_ext with exit code $infer_exit_code. Check log: ${LOG_SUBDIR_TOOLKIT}/3_inference.log"
        return 1
    fi
    echo "INFO: Inference finished for $roi_name_no_ext."
    if [ ! -d "${DATA_DIR}/representation_retiled" ] || [ -z "$(ls -A "${DATA_DIR}/representation_retiled" 2>/dev/null)" ]; then
        echo "ERROR: Inference output (representation_retiled) missing or empty for $roi_name_no_ext. Check log: ${LOG_SUBDIR_TOOLKIT}/3_inference.log"
        return 1
    fi

    # 4. Stitch Representations
    echo "[Step 4/4] Stitching representations for $roi_name_no_ext..."
    STITCH_SCRIPT="${BTFM_PROJECT_DIR}/btfm_infer/stitch_tiled_representation.py"
	# We could also use a docker here, since stitch_tiled_representation.py needs rasterio
    (
      set -e # Enable strict error checking for this subshell
      cd "${BTFM_PROJECT_DIR}/btfm_infer" && \
      "$INFERENCE_PYTHON_ENV" "$STITCH_SCRIPT" \
        --d_pixel_retiled_path "${DATA_DIR}/retiled_d_pixel" \
        --representation_retiled_path "${DATA_DIR}/representation_retiled" \
        --downstream_tiff "$ROI_TIFF_FOR_PIPELINE" \
        --out_dir "$DATA_DIR"
    ) > "${LOG_SUBDIR_TOOLKIT}/4_stitching.log" 2>&1
    
    local stitch_exit_code=$?
    if [ $stitch_exit_code -ne 0 ]; then
         echo "ERROR: Stitching failed for $roi_name_no_ext with exit code $stitch_exit_code. Check log: ${LOG_SUBDIR_TOOLKIT}/4_stitching.log"
        return 1
    fi
    
    if [ -f "${DATA_DIR}/stitched_representation.npy" ]; then
        mv "${DATA_DIR}/stitched_representation.npy" "$FINAL_REPRESENTATION_PATH"
        echo "INFO: Stitching finished. Final representation: $FINAL_REPRESENTATION_PATH"
        echo "SUCCESS: Finished processing $roi_name_no_ext."

        # Optional: Clean up DATA_DIR for this ROI to save space
        # Be careful with this. For now commented out.
        # echo "INFO: Cleaning up working directory ${DATA_DIR}..."
        # rm -rf "${DATA_DIR}/data_sar" "${DATA_DIR}/data_sar_processed" \
        #        "${DATA_DIR}/data_raw" "${DATA_DIR}/data_processed" \
        #        "${DATA_DIR}/retiled_d_pixel" "${DATA_DIR}/representation_retiled" \
        #        "${DATA_DIR}/snap_tmp" "${DATA_DIR}/snap_auxdata"
        # Note: roi.tiff for pipeline is still in DATA_DIR. Could remove it too if GENERATED_ROI_TIFF_FOR_USER is sufficient.
        # rm -f "${DATA_DIR}/roi.tiff"
        return 0 # Success
    else
        echo "ERROR: Stitched representation not found for $roi_name_no_ext. Check log: ${LOG_SUBDIR_TOOLKIT}/4_stitching.log"
        return 1 # Failure
    fi
}
export -f process_single_roi
# Export all necessary variables that process_single_roi uses
export SCRIPT_DIR_TOOLKIT BTFM_PROJECT_DIR INFERENCE_PYTHON_ENV ASF_USERNAME ASF_PASSWORD PROCESSING_YEAR \
       INFERENCE_CPU_GPU_SPLIT INFERENCE_MAX_CPU_PROCESSES INFERENCE_MAX_GPU_PROCESSES INFERENCE_GPU_BATCH_SIZE MODEL_CHECKPOINT \
       PYTHON_ROI_PROCESSOR \
       WORKING_DATA_BASE_DIR ROI_TIFFS_GENERATED_DIR REPRESENTATIONS_DIR LOG_DIR_TOOLKIT \
       S2_START_TIME S2_END_TIME S1_START_TIME S1_END_TIME DOWNSAMPLE_RATE

# --- Main Loop for ROI Files ---
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_FILES_ATTEMPTED=0

if [ ! -e "$INPUT_SPECIFIER" ]; then
    echo "ERROR: Input specifier '$INPUT_SPECIFIER' does not exist."
    exit 1
fi

if [ -d "$INPUT_SPECIFIER" ]; then
    echo "INFO: Processing all compatible ROI files in directory: $INPUT_SPECIFIER"
    
    # Create an array of files to process
    # Using find with -print0 and mapfile for robustness with special characters in filenames
    mapfile -d $'\0' files_to_process < <(find "$INPUT_SPECIFIER" -maxdepth 1 -type f \( \
        -iname "*.shp" -o \
        -iname "*.geojson" -o \
        -iname "*.json" -o \
        -iname "*.tif" -o \
        -iname "*.tiff" -o \
        -iname "*.txt" \
    \) -print0)

    if [ ${#files_to_process[@]} -eq 0 ]; then
        echo "INFO: No compatible files found in $INPUT_SPECIFIER."
        # Exit successfully if no files to process
        echo "======================================================================"
        echo "Toolkit finished: $(date)"
        echo "Summary: No files to process."
        echo "======================================================================"
        exit 0
    fi

    echo "INFO: Found ${#files_to_process[@]} potential ROI files to process."

    for roi_file_path_loop_var in "${files_to_process[@]}"; do
        # Ensure the variable is treated as a path, especially if it contains spaces
        if [ -z "$roi_file_path_loop_var" ]; then continue; fi # Skip empty entries if any

        TOTAL_FILES_ATTEMPTED=$((TOTAL_FILES_ATTEMPTED + 1))
        process_single_roi "$roi_file_path_loop_var" # Pass the path quoted
        if [ $? -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "WARNING: Processing failed for $roi_file_path_loop_var. Continuing with next file."
        fi
    done
elif [ -f "$INPUT_SPECIFIER" ]; then
    echo "INFO: Processing single ROI file: $INPUT_SPECIFIER"
    TOTAL_FILES_ATTEMPTED=1
    process_single_roi "$INPUT_SPECIFIER"
    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=1
    else
        FAIL_COUNT=1
    fi
else
    echo "ERROR: Input specifier '$INPUT_SPECIFIER' is not a valid file or directory."
    exit 1
fi

echo ""
echo "======================================================================"
echo "All ROI inputs processed. Toolkit finished: $(date)"
echo "Summary: Attempted to process $TOTAL_FILES_ATTEMPTED inputs."
echo "         Successfully processed: $SUCCESS_COUNT"
echo "         Failed to process: $FAIL_COUNT"
echo "======================================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1 # Indicate failure if any ROI processing failed
fi
exit 0