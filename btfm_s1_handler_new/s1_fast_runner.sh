#!/usr/bin/env bash
#
# s1_fast_runner.sh — Sentinel-1 Flexible Parallel Processing
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.9
# Usage:
#   bash s1_fast_runner.sh --input_tiff roi.tif \
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
  --start_time      Start date (YYYY-MM-DD) - inclusive
  --end_time        End date (YYYY-MM-DD) - inclusive

Optional arguments:
  --partitions      Number of parallel processes (default 4)
  --total_workers   Total number of Dask workers (default 16)
  --output          Output directory (default sentinel1_output)
  --orbit_state     Orbit state (ascending/descending/both, default both)
  --worker_memory   Memory per worker in GB (default 16)
  --chunksize       stackstac x/y chunk size (default 1024)
  --overwrite       Overwrite existing files
  --debug           Output debug logs
  --min_coverage    Minimum valid pixel coverage percentage (default 10.0)

bash s1_fast_runner.sh \
--input_tiff /home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/austrian_crop/roi.tif \
--start_time 2022-01-01 \
--end_time 2022-12-31 \
--output /home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/austrian_crop/data_sar_raw

EOF
  exit 1
}

#######################################
# Default Values
#######################################
OUTPUT="sentinel1_output"
ORBIT_STATE="both"
PARTITIONS=12
TOTAL_WORKERS=24
WORKER_MEMORY=40
CHUNKSIZE=1024
OVERWRITE=""
DEBUG=""
MIN_COVERAGE=10.0

PYTHON_ENV="/home/zf281/rds/hpc-work/Softwares/anaconda3/envs/btfm-data-processing/bin/python"

#######################################
# Parse Command Line Arguments
#######################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_tiff)     INPUT_TIFF=$2; shift 2;;
    --start_time)     START_TIME=$2; shift 2;;
    --end_time)       END_TIME=$2; shift 2;;
    --partitions)     PARTITIONS=$2; shift 2;;
    --total_workers)  TOTAL_WORKERS=$2; shift 2;;
    --output)         OUTPUT=$2; shift 2;;
    --orbit_state)    ORBIT_STATE=$2; shift 2;;
    --worker_memory)  WORKER_MEMORY=$2; shift 2;;
    --chunksize)      CHUNKSIZE=$2; shift 2;;
    --overwrite)      OVERWRITE="--overwrite"; shift 1;;
    --debug)          DEBUG="--debug"; shift 1;;
    --min_coverage)   MIN_COVERAGE=$2; shift 2;;
    -h|--help)        usage;;
    *)                echo "Unknown option: $1"; usage;;
  esac
done

[[ -z "${INPUT_TIFF:-}" || -z "${START_TIME:-}" || -z "${END_TIME:-}" ]] && usage

#######################################
# Calculate Time Partitions
#######################################
# Convert timestamp to seconds
time_to_seconds() {
  date -d "$1" +%s
}

# Convert seconds to date format
seconds_to_date() {
  date -d "@$1" +"%Y-%m-%d"
}

# Calculate the start and end time for each partition
calculate_partitions() {
  local start_sec=$(time_to_seconds "$START_TIME")
  local end_sec=$(time_to_seconds "$END_TIME")
  local total_seconds=$((end_sec - start_sec + 86400))  # +1 day to ensure the end date is included
  local seconds_per_partition=$((total_seconds / PARTITIONS))
  
  for ((i=0; i<PARTITIONS; i++)); do
    local partition_start_sec
    local partition_end_sec
    
    if [[ $i -eq 0 ]]; then
      # The first partition uses the exact start time
      partition_start_sec=$start_sec
    else
      # Other partitions start from the day after the end time of the previous partition
      partition_start_sec=$((start_sec + i * seconds_per_partition))
    fi
    
    if [[ $i -eq $((PARTITIONS - 1)) ]]; then
      # The last partition uses the exact end time
      partition_end_sec=$end_sec
    else
      # End time of other partitions (subtract 1 second to avoid overlap)
      partition_end_sec=$((start_sec + (i + 1) * seconds_per_partition - 86400))
    fi
    
    local p_start=$(seconds_to_date $partition_start_sec)
    local p_end=$(seconds_to_date $partition_end_sec)
    
    echo "$p_start,$p_end"
  done
}

#######################################
# Allocate Worker Count
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
    # Different years
    echo "P${p_index}_${start_year}${start_month}${start_day}-${end_year}${end_month}${end_day}"
  fi
}

#######################################
# Main Processing Logic
#######################################

mkdir -p "$OUTPUT"

echo "🚀 Starting Sentinel-1 Flexible Parallel Processing..."
echo "📊 Processing Parameters:"
echo "   ROI File: $INPUT_TIFF"
echo "   Time Range: $START_TIME → $END_TIME"
echo "   Number of Partitions: $PARTITIONS"
echo "   Total Workers: $TOTAL_WORKERS"
echo "   Output Directory: $OUTPUT"
echo "   Orbit State: $ORBIT_STATE"
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

for i in "${!partitions[@]}"; do
  p_range=${partitions[i]}
  p_start=${p_range%,*}
  p_end=${p_range#*,}
  workers=${workers_per_partition[i]}
  partition_id=$(generate_partition_id "$p_start" "$p_end" "$i")
  
  partition_ids+=("$partition_id")
  
  echo "🚀 Starting partition $partition_id ($i/${#partitions[@]})..."
  
  # Run processing in the background
  $PYTHON_ENV s1_fast_processor.py \
    --input_tiff "$INPUT_TIFF" \
    --start_date "$p_start" \
    --end_date "$p_end" \
    --output "$OUTPUT" \
    --orbit_state "$ORBIT_STATE" \
    --dask_workers "$workers" \
    --worker_memory "$WORKER_MEMORY" \
    --chunksize "$CHUNKSIZE" \
    --min_coverage "$MIN_COVERAGE" \
    --partition_id "$partition_id" \
    $OVERWRITE $DEBUG \
    > "$OUTPUT/s1_${partition_id}.log" 2>&1 &
  
  pids+=($!)
  
  # Delay the start slightly to avoid simultaneous resource access conflicts
  sleep 5
done

echo ""
echo "⏳ Waiting for all partitions to complete..."

# Wait for all processes to complete
failed_partitions=()
completed_partitions=()

for i in "${!pids[@]}"; do
  pid=${pids[i]}
  partition_id=${partition_ids[i]}
  
  echo "🔄 Waiting for partition $partition_id (PID: $pid)..."
  
  if wait $pid; then
    completed_partitions+=("$partition_id")
    echo "✅ Partition $partition_id completed successfully"
  else
    failed_partitions+=("$partition_id")
    echo "❌ Partition $partition_id failed"
  fi
done

# Output final results
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
    echo "   📄 $OUTPUT/s1_${partition}.log"
  done
  exit 1
fi

echo ""
echo "🎉 All partitions processed successfully!"
echo "📁 Results saved in: $OUTPUT"

# Summarize log files
echo ""
echo "📋 Summarizing log files..."
cat "$OUTPUT"/s1_*.log > "$OUTPUT/s1_combined.log"
echo "📄 Combined log saved to: $OUTPUT/s1_combined.log"

echo "✅ Program execution completed"