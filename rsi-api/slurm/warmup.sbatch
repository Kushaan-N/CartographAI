#!/bin/bash
#SBATCH --job-name=rsi-warmup
#SBATCH --gpus=1                  # any 1 GPU is plenty for Gemma-3-1B + LoRA
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=0-02:00:00        # SFT is ~10-30 min; 2h is headroom
#SBATCH --output=logs/warmup-%j.out
##SBATCH --account=<your_lab>    # uncomment + set if your lab requires an account
##SBATCH --constraint=vram16     # uncomment to force >=16GB VRAM, or sm_80 etc.

set -euo pipefail
mkdir -p logs

# --- environment (ADJUST to Unity's module names; see `module avail`) ----------
# Unity typically uses conda. Either of these patterns works — keep ONE:
# (a) venv created once on the login node:
#     module load python/3.11   # or whatever `module avail python` shows
#     source .venv/bin/activate
# (b) conda env:
#     module load conda/latest
#     conda activate rsi
source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before sbatch (export HF_TOKEN=hf_... )}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available())"

# Warm up the policy (constrained-selection SFT). Bigger --batch on a real GPU.
python run_warmup.py 200 --batch 16
echo "warmup done -> checkpoints/warmup"
