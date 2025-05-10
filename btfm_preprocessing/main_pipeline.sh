#!/usr/bin/env bash
#
# main_pipeline.sh
#
# Processes each tile fully (S2 -> S1 -> S1 stack) in order. For each tile:
#   1) Run S2 if needed
#   2) Download S1 zips if needed
#   3) Convert S1 zips -> GeoTIFF in parallel (xargs -P4)
#   4) Run s1_stack once all S1 commands are done
#
# Example usage:
#   bash main_pipeline.sh /path/to/downstream.tiff 2021-01-01 2021-12-31 2021-01-01 2021-12-31 1

########################################
# CUSTOMIZABLE SETTINGS
########################################

# Base directory for all data storage
# This is the main directory where all processed and raw data will be stored
BASE_DATA_DIR="absolute/path/to/your/data_dir"

# Sentinel-1 (SAR) download credentials
# Required for downloading Sentinel-1 data from ASF
# If you don't have an account, create one at https://urs.earthdata.nasa.gov/
# Please make sure that you have enbaled the "Alaska Satellite Facility Data Access" in your account
S1_DOWNLOAD_USERNAME="xxxxxx"
S1_DOWNLOAD_PASSWORD="xxxxxx"

# Maximum cloud coverage percentage allowed for Sentinel-2 imagery (0-100)
# Higher values allow more cloudy images
CLOUD_COVERAGE_THRESHOLD=90

########################################
# DO NOT MODIFY BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
########################################

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

# Helper functions for timing and logging
log_header() {
    echo -e "\n${BOLD}${BLUE}==== $1 ====${RESET}"
}

log_info() {
    echo -e "${CYAN}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1" >&2
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

########################################
# 1) Environment variables
########################################
log_header "SETTING UP ENVIRONMENT"

# Set up base directory
# Get current script directory
BASE_SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "Base script directory: $BASE_SCRIPT_DIR"
# Get parent directory of BASE_SCRIPT_DIR
PARENT_DIR=$(dirname "$BASE_SCRIPT_DIR")
echo "Parent directory: $PARENT_DIR"

mkdir -p ${BASE_DATA_DIR}/{data_sar,data_sar_processed,data_raw,snap_tmp,snap_auxdata,retiled_d_pixel,representation_retiled}
log_info "Created necessary directories"

export S2_DATA_RAW="${BASE_DATA_DIR}/data_raw"
export S2_DATA_PROCESSED="${BASE_DATA_DIR}/data_processed"
export S1_DATA_RAW="${BASE_DATA_DIR}/data_sar"
export S1_DATA_PROCESSED="${BASE_DATA_DIR}/data_sar_processed"
export RETILED_DATA="${BASE_DATA_DIR}/retiled_d_pixel"
export RUST_S2_EXE="/data/s2_process_tile_downstream"

export CLOUD_COVERAGE_THRESHOLD=$CLOUD_COVERAGE_THRESHOLD

# S1 user/pwd if needed:
export S1_DOWNLOAD_USERNAME="$S1_DOWNLOAD_USERNAME"
export S1_DOWNLOAD_PASSWORD="$S1_DOWNLOAD_PASSWORD"

# Snap environment variables
export SNAP_USER_DIR="${BASE_DATA_DIR}/snap_auxdata"
export TMPDIR="${BASE_DATA_DIR}/snap_tmp"
export _JAVA_OPTIONS="-Djava.io.tmpdir=${BASE_DATA_DIR}/snap_tmp"
export HOME_DEM_DIR="~/.snap/auxdata/dem/SRTM 1Sec HGT"

export PYTHON="docker run --rm \
  -v ${PARENT_DIR}:${PARENT_DIR} \
  -v ${BASE_SCRIPT_DIR}:/data \
  -v ${BASE_DATA_DIR}/data_sar:${BASE_DATA_DIR}/data_sar \
  -v ${BASE_DATA_DIR}/data_raw:${BASE_DATA_DIR}/data_raw \
  -v ${BASE_DATA_DIR}/data_sar_processed:${BASE_DATA_DIR}/data_sar_processed \
  -v ${BASE_DATA_DIR}/snap_tmp:${BASE_DATA_DIR}/snap_tmp \
  -v ${BASE_DATA_DIR}/snap_auxdata:${BASE_DATA_DIR}/snap_auxdata \
  -e S2_DATA_RAW \
  -e S2_DATA_PROCESSED \
  -e S1_DATA_RAW \
  -e S1_DATA_PROCESSED \
  -e RUST_S2_EXE \
  -e CLOUD_COVERAGE_THRESHOLD \
  -e S1_DOWNLOAD_USERNAME \
  -e S1_DOWNLOAD_PASSWORD \
  -e SNAP_USER_DIR \
  -e TMPDIR \
  -e _JAVA_OPTIONS \
  -e HOME_DEM_DIR \
  frankfeng1223/snap-gdal-python:v3"

# We keep pipefail but not -e so it won't exit on first error
#set -e
set -o pipefail

log_info "Environment variables set"

########################################
# 2) Parse arguments
########################################
log_header "PARSING ARGUMENTS"

if [ "$#" -lt 6 ]; then
  log_error "Usage: $0 <downstream_tiff> <s2_start> <s2_end> <s1_start> <s1_end> <sample_rate>"
  exit 1
fi

DOWNSTREAM_TIFF="$1"
S2_START_DATE="$2"
S2_END_DATE="$3"
S1_START_DATE="$4"
S1_END_DATE="$5"
SAMPLE_RATE="$6"

log_info "Arguments parsed successfully:"
log_info "  DOWNSTREAM_TIFF: ${DOWNSTREAM_TIFF}"
log_info "  S2_START_DATE: ${S2_START_DATE}"
log_info "  S2_END_DATE: ${S2_END_DATE}"
log_info "  S1_START_DATE: ${S1_START_DATE}"
log_info "  S1_END_DATE: ${S1_END_DATE}"
log_info "  SAMPLE_RATE: ${SAMPLE_RATE}"

########################################
# Step A) Discover tiles -> tile_list.txt
########################################
log_header "STEP A: DISCOVER MGRS TILES"
STEP_A_START=$(date +%s)

$PYTHON s2_discover_tiles.py \
  --tiff_path "$DOWNSTREAM_TIFF" \
  --out_tiles tile_list.txt \
  || { log_warning "s2_discover_tiles.py failed but continuing..."; }

if [ ! -s tile_list.txt ]; then
  log_error "No tiles found. Exiting."
  exit 0
fi

log_info "Found tiles:"
cat tile_list.txt | while read -r TILE; do echo "  - ${TILE}"; done

STEP_A_END=$(date +%s)
STEP_A_DURATION=$(calculate_time $STEP_A_START $STEP_A_END)
log_success "Step A completed in ${BOLD}${STEP_A_DURATION}${RESET}"

########################################
# A function to see if S2 or S1 is done
########################################
function check_tile_status() {
  local TID="$1"
  local DIR="${S2_DATA_PROCESSED}/${TID}"

  # Check 5 S2 npy
  local S2_DONE=1
  for f in band_mean.npy bands.npy band_std.npy doys.npy masks.npy; do
    if [ ! -f "$DIR/$f" ]; then
      S2_DONE=0
      break
    fi
  done

  # Check 4 S1 npy
  local S1_DONE=1
  for f in sar_ascending.npy sar_ascending_doy.npy sar_descending.npy sar_descending_doy.npy; do
    if [ ! -f "$DIR/$f" ]; then
      S1_DONE=0
      break
    fi
  done

  if [ $S2_DONE -eq 1 ] && [ $S1_DONE -eq 1 ]; then
    echo "ALL_DONE"
  elif [ $S2_DONE -eq 1 ]; then
    echo "S2_DONE_ONLY"
  else
    echo "NOT_DONE"
  fi
}

########################################
# A helper to build the s1_processor cmd
########################################
function build_s1_cmd() {
  local ZF="$1"
  # tile_id = parent folder name
  local TID
  TID=$(basename "$(dirname "$ZF")")

  local RED_DIR="${S2_DATA_RAW}/${TID}/red"
  local REF_TIF
  REF_TIF=$(ls -1 "${RED_DIR}"/*.tiff 2>/dev/null | head -n 1)

  if [ -z "$REF_TIF" ]; then
    # Return an echo command complaining
    echo "echo 'No reference TIF for tile ${TID} (cannot process ${ZF})' >&2"
    return
  fi

  local OUT_DIR="${S1_DATA_PROCESSED}/${TID}"
  # Return the real command
  echo "${PYTHON} s1_processor.py --zip_file \"${ZF}\" --reference_tiff \"${REF_TIF}\" --out_dir \"${OUT_DIR}\""
}

########################################
# Step B) For each tile -> S2 -> S1 -> stack
########################################
log_header "STEP B: PROCESS EACH TILE"
STEP_B_START=$(date +%s)

# Summary statistics
TOTAL_TILES=0
SKIPPED_TILES=0
PROCESSED_TILES=0
S2_TOTAL_TIME=0
S1_TOTAL_TIME=0
TILE_TIMES=()
TILE_NAMES=()

while read -r TILE_ID; do
  TOTAL_TILES=$((TOTAL_TILES + 1))
  TILE_START=$(date +%s)
  
  log_header "PROCESSING TILE ${MAGENTA}${TILE_ID}${BLUE}"
  STATUS=$(check_tile_status "$TILE_ID")
  log_info "Tile status: ${STATUS}"

  if [ "$STATUS" = "ALL_DONE" ]; then
    log_info "Already fully processed. Skipping."
    SKIPPED_TILES=$((SKIPPED_TILES + 1))
    continue
  fi
  
  PROCESSED_TILES=$((PROCESSED_TILES + 1))

  # -- If S2 not done => run s2_processor
  if [ "$STATUS" = "NOT_DONE" ]; then
    log_info "S2 not done for ${TILE_ID}, running s2_processor..."
    S2_START=$(date +%s)
    
    $PYTHON s2_processor.py \
      --tile_id "$TILE_ID" \
      --tiff_path "$DOWNSTREAM_TIFF" \
      --s2_start_date "$S2_START_DATE" \
      --s2_end_date "$S2_END_DATE" \
      --sample_rate "$SAMPLE_RATE" \
      || { log_warning "s2_processor.py failed for ${TILE_ID} but continuing..."; }
    
    S2_END=$(date +%s)
    S2_DURATION=$((S2_END - S2_START))
    S2_TOTAL_TIME=$((S2_TOTAL_TIME + S2_DURATION))
    S2_TIME_FORMATTED=$(format_time $S2_DURATION)
    log_success "S2 processing completed in ${BOLD}${S2_TIME_FORMATTED}${RESET}"
  else
    log_info "S2 already done, skipping s2_processor for ${TILE_ID}."
  fi

  # re-check
  STATUS2=$(check_tile_status "$TILE_ID")
  if [ "$STATUS2" = "ALL_DONE" ]; then
    log_info "After S2, tile ${TILE_ID} is fully done, skipping S1."
    
    TILE_END=$(date +%s)
    TILE_DURATION=$((TILE_END - TILE_START))
    TILE_TIME_FORMATTED=$(format_time $TILE_DURATION)
    TILE_TIMES+=($TILE_DURATION)
    TILE_NAMES+=("$TILE_ID")
    
    log_success "Tile ${MAGENTA}${TILE_ID}${RESET} completed in ${BOLD}${TILE_TIME_FORMATTED}${RESET}"
    echo ""
    continue
  fi

  # If S1 not done => Download S1 & build commands
  if [ "$STATUS2" = "S2_DONE_ONLY" ] || [ "$STATUS2" = "NOT_DONE" ]; then
    log_info "S1 not done for ${TILE_ID}, downloading zips..."
    S1_START=$(date +%s)
    
    $PYTHON s1_downloader.py \
      --tile_id "$TILE_ID" \
      --s1_start_date "$S1_START_DATE" \
      --s1_end_date "$S1_END_DATE" \
      || { log_warning "s1_downloader.py failed for ${TILE_ID} but continuing..."; }

    ZIP_DIR="${S1_DATA_RAW}/${TILE_ID}"
    if [ ! -d "$ZIP_DIR" ]; then
      log_warning "No S1 zip dir found for ${TILE_ID}, skipping."
      
      TILE_END=$(date +%s)
      TILE_DURATION=$((TILE_END - TILE_START))
      TILE_TIME_FORMATTED=$(format_time $TILE_DURATION)
      TILE_TIMES+=($TILE_DURATION)
      TILE_NAMES+=("$TILE_ID")
      
      log_success "Tile ${MAGENTA}${TILE_ID}${RESET} completed in ${BOLD}${TILE_TIME_FORMATTED}${RESET}"
      echo ""
      continue
    fi

    log_info "Building s1 commands for ${TILE_ID}"
    rm -f s1_commands_tile.txt
    touch s1_commands_tile.txt

    find "$ZIP_DIR" -type f -name "*.zip" | while read -r ZF; do
      CMD=$(build_s1_cmd "$ZF")
      if [ -n "$CMD" ]; then
        echo "$CMD" >> s1_commands_tile.txt
      fi
    done

    NCMDS=$(wc -l < s1_commands_tile.txt)
    log_info "Found ${NCMDS} S1 commands for tile ${TILE_ID}"
    if [ "$NCMDS" -gt 0 ]; then
      log_info "Running S1 commands in parallel (-P4) and continuing even if some fail..."

      # Modified to continue processing even if some commands fail
      cat s1_commands_tile.txt | xargs -t -L1 -P4 -I{} bash -c '{} || echo "⚠️ Command failed with status $?: {} - but continuing..."'
      
      log_success "S1 .zip->.tif commands processing completed for tile ${TILE_ID}"
    else
      log_warning "No S1 commands for ${TILE_ID}."
    fi
  fi

  # Now do final S1 stacking for this tile
  log_info "Doing final S1 stack for tile ${TILE_ID}"
  STATUS3=$(check_tile_status "$TILE_ID")
  if [ "$STATUS3" = "ALL_DONE" ]; then
    log_info "Tile is now fully done, skipping stack."
    
    S1_END=$(date +%s)
    S1_DURATION=$((S1_END - S1_START))
    S1_TOTAL_TIME=$((S1_TOTAL_TIME + S1_DURATION))
    S1_TIME_FORMATTED=$(format_time $S1_DURATION)
    log_success "S1 processing completed in ${BOLD}${S1_TIME_FORMATTED}${RESET}"
    
    TILE_END=$(date +%s)
    TILE_DURATION=$((TILE_END - TILE_START))
    TILE_TIME_FORMATTED=$(format_time $TILE_DURATION)
    TILE_TIMES+=($TILE_DURATION)
    TILE_NAMES+=("$TILE_ID")
    
    log_success "Tile ${MAGENTA}${TILE_ID}${RESET} completed in ${BOLD}${TILE_TIME_FORMATTED}${RESET}"
    echo ""
    continue
  fi

  # get reference TIF
  RED_DIR="${S2_DATA_RAW}/${TILE_ID}/red"
  REF_TIF=$(ls -1 "${RED_DIR}"/*.tiff 2>/dev/null | head -n 1)
  if [ -z "$REF_TIF" ]; then
    log_warning "No reference TIF found in ${RED_DIR}, skipping stack."
    
    S1_END=$(date +%s)
    S1_DURATION=$((S1_END - S1_START))
    S1_TOTAL_TIME=$((S1_TOTAL_TIME + S1_DURATION))
    S1_TIME_FORMATTED=$(format_time $S1_DURATION)
    log_success "S1 processing completed in ${BOLD}${S1_TIME_FORMATTED}${RESET}"
    
    TILE_END=$(date +%s)
    TILE_DURATION=$((TILE_END - TILE_START))
    TILE_TIME_FORMATTED=$(format_time $TILE_DURATION)
    TILE_TIMES+=($TILE_DURATION)
    TILE_NAMES+=("$TILE_ID")
    
    log_success "Tile ${MAGENTA}${TILE_ID}${RESET} completed in ${BOLD}${TILE_TIME_FORMATTED}${RESET}"
    echo ""
    continue
  fi

  log_info "Running s1_stack for ${TILE_ID}..."
  $PYTHON s1_stack.py --tile_id "$TILE_ID" --reference_tiff "$REF_TIF" --sample_rate "$SAMPLE_RATE" \
    || { log_warning "s1_stack.py failed for ${TILE_ID} with status $? - but continuing..."; }

  S1_END=$(date +%s)
  S1_DURATION=$((S1_END - S1_START))
  S1_TOTAL_TIME=$((S1_TOTAL_TIME + S1_DURATION))
  S1_TIME_FORMATTED=$(format_time $S1_DURATION)
  log_success "S1 processing completed in ${BOLD}${S1_TIME_FORMATTED}${RESET}"

  TILE_END=$(date +%s)
  TILE_DURATION=$((TILE_END - TILE_START))
  TILE_TIME_FORMATTED=$(format_time $TILE_DURATION)
  TILE_TIMES+=($TILE_DURATION)
  TILE_NAMES+=("$TILE_ID")
  
  log_success "Tile ${MAGENTA}${TILE_ID}${RESET} completed in ${BOLD}${TILE_TIME_FORMATTED}${RESET}"
  echo ""
done < tile_list.txt

STEP_B_END=$(date +%s)
STEP_B_DURATION=$(calculate_time $STEP_B_START $STEP_B_END)
log_success "Step B completed in ${BOLD}${STEP_B_DURATION}${RESET}"

# Perform retiling for all tiles
log_header "RETILING ALL TILES"
RETILE_START=$(date +%s)

$PYTHON dpixel_retiler.py \
--tiff_path "$DOWNSTREAM_TIFF" \
--d_pixel_dir "$S2_DATA_PROCESSED" \
--patch_size 500 \
--out_dir "$RETILED_DATA" \
--data_raw_dir "$S2_DATA_RAW" \
--overwrite \
--only_non_zero \
|| { log_warning "dpixel_retiler.py failed but continuing..."; }
    
  
RETILE_END=$(date +%s)
RETILE_DURATION=$(calculate_time $RETILE_START $RETILE_END)
log_success "Retiling completed in ${BOLD}${RETILE_DURATION}${RESET}"

# Calculate script total runtime
SCRIPT_END=$(date +%s)
SCRIPT_DURATION=$(calculate_time $SCRIPT_START $SCRIPT_END)

# Print summary report
log_header "PIPELINE EXECUTION SUMMARY"
log_info "Total script runtime: ${BOLD}${SCRIPT_DURATION}${RESET}"
log_info "Tiles processed: ${PROCESSED_TILES}/${TOTAL_TILES} (${SKIPPED_TILES} already completed)"

if [ $PROCESSED_TILES -gt 0 ]; then
    S2_AVG_TIME=$(format_time $((S2_TOTAL_TIME / PROCESSED_TILES)))
    S1_AVG_TIME=$(format_time $((S1_TOTAL_TIME / PROCESSED_TILES)))
    log_info "Average S2 processing time: ${BOLD}${S2_AVG_TIME}${RESET}"
    log_info "Average S1 processing time: ${BOLD}${S1_AVG_TIME}${RESET}"
    
    # Find the slowest and fastest tiles
    MAX_TIME=0
    MIN_TIME=999999
    MAX_IDX=0
    MIN_IDX=0
    
    for i in "${!TILE_TIMES[@]}"; do
        if [ "${TILE_TIMES[$i]}" -gt "$MAX_TIME" ]; then
            MAX_TIME="${TILE_TIMES[$i]}"
            MAX_IDX=$i
        fi
        if [ "${TILE_TIMES[$i]}" -lt "$MIN_TIME" ]; then
            MIN_TIME="${TILE_TIMES[$i]}"
            MIN_IDX=$i
        fi
    done
    
    MAX_TIME_FORMATTED=$(format_time $MAX_TIME)
    MIN_TIME_FORMATTED=$(format_time $MIN_TIME)
    
    log_info "Fastest tile: ${MAGENTA}${TILE_NAMES[$MIN_IDX]}${RESET} (${BOLD}${MIN_TIME_FORMATTED}${RESET})"
    log_info "Slowest tile: ${MAGENTA}${TILE_NAMES[$MAX_IDX]}${RESET} (${BOLD}${MAX_TIME_FORMATTED}${RESET})"
fi

log_success "Pipeline execution completed successfully!"
