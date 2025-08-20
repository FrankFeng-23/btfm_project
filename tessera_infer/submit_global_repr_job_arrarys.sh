#!/bin/bash

counter=0
max_runs=16

while [ $counter -lt $max_runs ]; do
    sbatch infer_all_tiles_QAT_global_file_system.slurm
    ((counter++))
    if [ $counter -lt $max_runs ]; then
        sleep 100  # 5 minutes
    fi
done