#!/bin/bash
#SBATCH --job-name=trajs_array
#SBATCH --array=1-20
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

echo "Array task ${SLURM_ARRAY_TASK_ID}"
python3 abm/monitoring/trajs.py ${SLURM_ARRAY_TASK_ID}