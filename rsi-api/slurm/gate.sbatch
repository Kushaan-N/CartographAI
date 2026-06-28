#!/bin/bash
#SBATCH --job-name=rsi-gate
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=0-08:00:00        # 800 episodes ~4-6h sequential; raise for more
#SBATCH --output=logs/gate-%j.out
##SBATCH --account=<your_lab>
##SBATCH --constraint=vram16

set -euo pipefail
mkdir -p logs

# --- environment (same as warmup.sbatch; keep ONE pattern) ---------------------
source .venv/bin/activate
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before sbatch}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available())"

# The GRPO gate. Loads checkpoints/warmup, runs same-seed contrastive groups,
# GRPO updates, epsilon decay, and prints pure_policy_cov inline.
# THE VERDICT = pure_policy_cov: rises above ~20% -> GRPO works; flat -> warmup
# is the ceiling; drops toward 7.1% -> degrading (kill + debug).
python run_local_smoke.py 800 \
    --checkpoint checkpoints/warmup \
    --eval-every-updates 3 \
    --metrics logs/gate_metrics.jsonl
echo "gate done -> logs/gate_metrics.jsonl"
