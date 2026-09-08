#!/bin/bash -l
#SBATCH --output=/users/k21181837/PY/GPU_geom_%A_%a.txt
#SBATCH --job-name=GPU_Bridge_Geom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=k21181837@kcl.ac.uk

set -euo pipefail

echo "=================================================="
echo "Running on ${HOSTNAME}"
echo "Job started at $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="

module load cuda/11.8.0-gcc-13.2.0
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load r/4.3.0-gcc-13.2.0-withx-rmath-standalone-python-3.11.6

source /users/k21181837/python_env/bin/activate

cd /users/k21181837/PY
mkdir -p results

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export MPLBACKEND=Agg

echo ""
echo "========== Environment Check =========="
echo "Python path: $(which python)"
python --version
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version used by PyTorch:', torch.version.cuda)"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU AVAILABLE')"
echo "======================================="
echo ""

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"

P_VALUES=(5 10 20 50 75 100)
P=${P_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "========== Starting Geometric Equivalence experiment, p=${P} =========="
echo "Start time: $(date)"

python -u measure_morala_gpu.py \
  --device cuda \
  --n 10000 \
  --p "${P}" \
  --epochs 300 \
  --batch-size 512 \
  --repeats 10 \
  --data-seed 0 \
  --out-prefix "results/geom_gpu_noltpr_p${P}" \
  --checkpoint-every 1 \
  --ltpr-max-p 0

echo ""
echo "========== p=${P} experiment finished =========="
echo "Finish time: $(date)"
echo "=================================================="
echo "Array task completed successfully."
echo "=================================================="
