#!/bin/bash
# One-shot setup on a Slurm cluster LOGIN node (e.g. Unity HPC).
# Does: clone (if needed) -> venv -> install deps -> sanity-test.
# This is all LIGHT work (no GPU, no model download) and is safe on the login node.
#
# Usage:
#   # if you already cloned and are inside CartographAI/rsi-api:
#   bash setup_cluster.sh
#
#   # from scratch (clones for you):
#   git clone https://github.com/Kushaan-N/CartographAI.git && cd CartographAI/rsi-api && bash setup_cluster.sh
#
# After it finishes:  export HF_TOKEN=hf_...  then  sbatch slurm/warmup.sbatch
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/Kushaan-N/CartographAI.git}"

echo "=== [1/4] python ==="
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On Unity, load a module first, e.g.:"
  echo "    module avail python conda"
  echo "    module load python/3.11    # use whatever 'module avail' shows"
  echo "Then re-run this script."
  exit 1
fi
echo "python: $(python3 --version)"

echo "=== [2/4] repo ==="
if [ -f "run_local_smoke.py" ] && [ -d "configs" ]; then
  echo "already inside rsi-api; skipping clone"
else
  [ -d CartographAI ] || git clone "$REPO_URL"
  cd CartographAI/rsi-api
fi
echo "repo: $(pwd)"

echo "=== [3/4] venv + deps (this downloads torch etc; a few minutes) ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "=== [4/4] sanity test (no GPU, no model download) ==="
if PYTHONPATH="$(pwd)" python -m pytest tests/ -q; then
  TESTS="PASSED"
else
  TESTS="FAILED — check output above before running jobs"
fi

echo ""
echo "=== setup complete (tests: $TESTS) ==="
echo "next:"
echo "  export HF_TOKEN=hf_...        # accept the Gemma-3 license on HF first"
echo "  nvidia-smi                    # (on a GPU node) confirm a GPU is visible"
echo "  sbatch slurm/warmup.sbatch    # then, when done: sbatch slurm/gate.sbatch"
echo "  # adjust the 'module load'/'source .venv' lines in slurm/*.sbatch to your cluster"
