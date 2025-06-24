#!/bin/bash

counter=0
max_runs=12

while [ $counter -lt $max_runs ]; do
    sbatch infer_all_tiles_QAT_global_file_system.slurm
    ((counter++))
    if [ $counter -lt $max_runs ]; then
        sleep 1800  # 30 minutes = 1800 seconds
    fi
done