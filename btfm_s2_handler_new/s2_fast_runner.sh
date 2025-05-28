#!/usr/bin/env bash
#
# s2_flexible_runner.sh — Sentinel-2 Flexible Parallel Processing
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.9
# Usage:
#   bash s2_flexible_runner.sh --input_tiff roi.tif \
#                             --start_time 2023-10-01 --end_time 2024-09-30 \
#                             [--partitions 12] [--total_workers 24] [...]

set -euo pipefail

#######################################
# Help Information
#######################################
usage() {
  cat <<EOF
Usage: $0 --input_tiff <path> --start_time <YYYY-MM-DD> --end_time <YYYY-MM-DD> [options]

Required arguments:
  --input_tiff      ROI mask or template raster
  --start_time      Start date (YYYY-MM-DD[THH:MM:SS]) - inclusive
  --end_time        End date (YYYY-MM-DD[THH:MM:SS]) - inclusive

Optional arguments:
  --partitions      Number of parallel processes (default 4, i.e., quarterly)
  --total_workers   Total number of Dask workers (default 32)
  --output          Output directory (default sentinel2_output)
  --max_cloud       Maximum cloud cover percentage (default 90)
  --worker_memory   Memory per worker in GB (default 16)
  --chunksize       stackstac x/y chunk size (default 1024)
  --resolution      Output resolution (meters, default 10)
  --overwrite       Overwrite existing files
  --debug           Output debug logs
  --min_coverage    Minimum valid pixel coverage (default 10.0)

bash s2_fast_runner.sh \
--input_tiff /home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/cambridge_large/shp/cambridge_simplified.tiff \
--start_time 2017-01-01 \
--end_time 2017-12-31 \
--output /home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/cambridge_large/2017/data_raw

EOF
  exit 1
}

#######################################
# Defaults
#######################################
export TEMP_DIR="/local/zf281"
OUTPUT="sentinel2_output"
MAX_CLOUD=90
PARTITIONS=24
TOTAL_WORKERS=24
WORKER_MEMORY=32
CHUNKSIZE=1024
RESOLUTION=10
OVERWRITE=""
DEBUG=""
MIN_COVERAGE=10.0

#######################################
# Parse command line arguments
#######################################
mkdir -p "$TEMP_DIR"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_tiff)     INPUT_TIFF=$2; shift 2;;
    --start_time)     START_TIME=$2; shift 2;;
    --end_time)       END_TIME=$2; shift 2;;
    --partitions)     PARTITIONS=$2; shift 2;;
    --total_workers)  TOTAL_WORKERS=$2; shift 2;;
    --output)         OUTPUT=$2; shift 2;;
    --max_cloud)      MAX_CLOUD=$2; shift 2;;
    --worker_memory)  WORKER_MEMORY=$2; shift 2;;
    --chunksize)      CHUNKSIZE=$2; shift 2;;
    --resolution)     RESOLUTION=$2; shift 2;;
    --overwrite)      OVERWRITE="--overwrite"; shift 1;;
    --debug)          DEBUG="--debug"; shift 1;;
    --min_coverage)   MIN_COVERAGE=$2; shift 2;;
    -h|--help)        usage;;
    *)                echo "Unknown option: $1"; usage;;
  esac
done

[[ -z "${INPUT_TIFF:-}" || -z "${START_TIME:-}" || -z "${END_TIME:-}" ]] && usage

# Ensure correct time format (add time part if missing)
format_datetime() {
  local dt=$1
  if [[ $dt =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    if [[ $2 == "start" ]]; then
      echo "${dt}T00:00:00"
    else
      echo "${dt}T23:59:59"
    fi
  else
    # Already contains time
    echo $dt
  fi
}

START_TIME=$(format_datetime "$START_TIME" "start")
END_TIME=$(format_datetime "$END_TIME" "end")

#######################################
# Calculate time partitions
#######################################
# Convert timestamp to seconds
time_to_seconds() {
  date -d "$1" +%s
}

# Convert seconds to ISO time format
seconds_to_time() {
  date -d "@$1" +"%Y-%m-%dT%H:%M:%S"
}

# Calculate the start and end times for each partition
calculate_partitions() {
  local start_sec=$(time_to_seconds "$START_TIME")
  local end_sec=$(time_to_seconds "$END_TIME")
  local total_seconds=$((end_sec - start_sec + 1))  # +1 to include the end time
  local seconds_per_partition=$((total_seconds / PARTITIONS))
  
  for ((i=0; i<PARTITIONS; i++)); do
    local partition_start_sec
    local partition_end_sec
    
    if [[ $i -eq 0 ]]; then
      # First partition uses the exact start time
      partition_start_sec=$start_sec
    else
      # Other partitions start from the next second after the previous partition's end time
      partition_start_sec=$((start_sec + i * seconds_per_partition))
    fi
    
    if [[ $i -eq $((PARTITIONS - 1)) ]]; then
      # Last partition uses the exact end time
      partition_end_sec=$end_sec
    else
      # Other partitions' end times
      partition_end_sec=$((start_sec + (i + 1) * seconds_per_partition - 1))
    fi
    
    local p_start=$(seconds_to_time $partition_start_sec)
    local p_end=$(seconds_to_time $partition_end_sec)
    
    echo "$p_start,$p_end"
  done
}

#######################################
# Allocate worker count
#######################################
# Calculate the number of Dask workers per partition
calculate_workers_per_partition() {
  local workers_per_partition=$((TOTAL_WORKERS / PARTITIONS))
  local remaining_workers=$((TOTAL_WORKERS % PARTITIONS))
  
  for ((i=0; i<PARTITIONS; i++)); do
    if [[ $i -lt $remaining_workers ]]; then
      echo $((workers_per_partition + 1))
    else
      echo $workers_per_partition
    fi
  done
}

#######################################
# Generate Partition ID
#######################################
generate_partition_id() {
  local p_start=$1
  local p_end=$2
  local p_index=$3
  
  # Extract year and month
  local start_year=$(date -d "${p_start}" +%Y)
  local start_month=$(date -d "${p_start}" +%m)
  local start_day=$(date -d "${p_start}" +%d)
  local end_year=$(date -d "${p_end}" +%Y)
  local end_month=$(date -d "${p_end}" +%m)
  local end_day=$(date -d "${p_end}" +%d)
  
  # Generate partition ID
  if [[ "$start_year" == "$end_year" && "$start_month" == "$end_month" ]]; then
    if [[ "$start_day" == "$end_day" ]]; then
      # Same day
      echo "P${p_index}_${start_year}${start_month}${start_day}"
    else
      # Same month, different days
      echo "P${p_index}_${start_year}${start_month}${start_day}-${end_day}"
    fi
  elif [[ "$start_year" == "$end_year" ]]; then
    # Same year, different months
    echo "P${p_index}_${start_year}${start_month}${start_day}-${end_month}${end_day}"
  else
    # Across years
    echo "P${p_index}_${start_year}${start_month}${start_day}-${end_year}${end_month}${end_day}"
  fi
}

#######################################
# Real-time process monitoring function
#######################################
monitor_process() {
  # Periodically check the status of all processes
  while true; do
    local all_done=true
    
    for i in "${!pids[@]}"; do
      local pid=${pids[i]}
      if [[ -n "${finished_pids[$pid]:-}" ]]; then
        # This process is complete, no need to check
        continue
      fi
      
      # Check if the process is still running
      if kill -0 $pid 2>/dev/null; then
        all_done=false
      else
        # Process ended, record time and status
        local end_time=$(date +%s)
        local duration=$((end_time - ${start_times[i]}))
        local hours=$((duration / 3600))
        local minutes=$(( (duration % 3600) / 60 ))
        local seconds=$((duration % 60))
        local partition_id=${partition_ids[i]}
        
        # Check exit status
        wait $pid
        local exit_code=$?
        
        # Mark as completed
        finished_pids[$pid]=1
        
        if [ $exit_code -eq 0 ]; then
          echo "$(date '+%Y-%m-%d %H:%M:%S') ✅ Partition $partition_id completed successfully, took ${hours} hours ${minutes} minutes ${seconds} seconds"
          completed_partitions+=("$partition_id")
        else
          echo "$(date '+%Y-%m-%d %H:%M:%S') ❌ Partition $partition_id failed (exit code: $exit_code), took ${hours} hours ${minutes} minutes ${seconds} seconds"
          failed_partitions+=("$partition_id")
        fi
      fi
    done
    
    # If all processes are finished, exit the loop
    if $all_done; then
      break
    fi
    
    # Sleep and check again
    sleep 5
  done
}

#######################################
# Main processing logic
#######################################

mkdir -p "$OUTPUT"

echo "🚀 Starting Sentinel-2 Flexible Parallel Processing..."
echo "📊 Processing parameters:"
echo "   ROI File: $INPUT_TIFF"
echo "   Time Range: $START_TIME → $END_TIME"
echo "   Number of Partitions: $PARTITIONS"
echo "   Total Workers: $TOTAL_WORKERS"
echo "   Output Directory: $OUTPUT"
echo "   Maximum Cloud Cover: $MAX_CLOUD%"
echo "   Worker Memory: ${WORKER_MEMORY}GB"
echo "   Minimum Coverage: $MIN_COVERAGE%"
echo ""

# Generate partitions
mapfile -t partitions < <(calculate_partitions)
mapfile -t workers_per_partition < <(calculate_workers_per_partition)

echo "🗓️  Time Partition Scheme:"
for i in "${!partitions[@]}"; do
  p_start=${partitions[i]%,*}
  p_end=${partitions[i]#*,}
  workers=${workers_per_partition[i]}
  partition_id=$(generate_partition_id "$p_start" "$p_end" "$i")
  
  echo "   Partition $i ($partition_id): $p_start → $p_end (Workers: $workers)"
done

echo ""
echo "🌟 Starting parallel processing of ${#partitions[@]} partitions..."

# Start parallel processes
pids=()
partition_ids=()
log_files=()
start_times=()
finished_pids=()
completed_partitions=()
failed_partitions=()
for i in "${!partitions[@]}"; do
  p_range=${partitions[i]}
  p_start=${p_range%,*}
  p_end=${p_range#*,}
  workers=${workers_per_partition[i]}
  partition_id=$(generate_partition_id "$p_start" "$p_end" "$i")
  
  partition_ids+=("$partition_id")
  
  echo "🚀 Starting partition $partition_id ($i/${#partitions[@]})..."
  
  # Prepare log file
  log_file="$OUTPUT/s2_${partition_id}.log"
  log_files+=("$log_file")
  
  # Run processing in the background
  python3 s2_fast_processor.py \
    --input_tiff "$INPUT_TIFF" \
    --start_date "$p_start" \
    --end_date "$p_end" \
    --output "$OUTPUT" \
    --max_cloud "$MAX_CLOUD" \
    --dask_workers "$workers" \
    --worker_memory "$WORKER_MEMORY" \
    --chunksize "$CHUNKSIZE" \
    --resolution "$RESOLUTION" \
    --min_coverage "$MIN_COVERAGE" \
    --partition_id "$partition_id" \
    $OVERWRITE $DEBUG \
    > "$log_file" 2>&1 &
  
  pid=$!
  pids+=($pid)
  start_times+=("$(date +%s)")
  
  # Delay startup slightly to avoid simultaneous resource access conflicts
  sleep 5
done

echo ""
echo "⏳ Waiting for all partitions to complete..."

# Start the process monitor
monitor_process

# Summarize results
echo ""
echo "📊 Processing Summary:"
echo "   Total Partitions: ${#partitions[@]}"
echo "   Successful Partitions: ${#completed_partitions[@]}"
echo "   Failed Partitions: ${#failed_partitions[@]}"

if [[ ${#completed_partitions[@]} -gt 0 ]]; then
  echo "✅ Successfully completed partitions:"
  printf '   🗓️  %s\n' "${completed_partitions[@]}"
fi

if [[ ${#failed_partitions[@]} -gt 0 ]]; then
  echo "❌ Failed partitions:"
  printf '   🗓️  %s\n' "${failed_partitions[@]}"
  echo ""
  echo "💡 Check the corresponding log files for detailed error information:"
  for partition in "${failed_partitions[@]}"; do
    echo "   📄 $OUTPUT/s2_${partition}.log"
  done
  exit 1
fi

echo ""
echo "🎉 All partitions processed!"
echo "📁 Results saved in: $OUTPUT"

# Aggregate log files
echo ""
echo "📋 Aggregating log files..."
cat "$OUTPUT"/s2_*.log > "$OUTPUT/s2_combined.log"
echo "📄 Combined log saved: $OUTPUT/s2_combined.log"

echo "✅ Program execution completed"