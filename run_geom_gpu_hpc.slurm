#!/bin/bash
#SBATCH --job-name=geom_gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/geom_gpu_%A_%a.out
#SBATCH --error=logs/geom_gpu_%A_%a.err

set -euo pipefail

mkdir -p logs

P_VALUES=(10 20 50 100 200)
P=${P_VALUES[$SLURM_ARRAY_TASK_ID]}

python -u measure_morala_gpu.py \
  --device cuda \
  --n 10000 \
  --p "$P" \
  --epochs 300 \
  --batch-size 512 \
  --repeats 10 \
  --data-seed 0 \
  --out-prefix "geom_gpu_noltpr_p${P}" \
  --checkpoint-every 1 \
  --ltpr-max-p 0
