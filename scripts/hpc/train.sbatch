#!/bin/bash
#SBATCH --job-name=sqmg-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # ADJUST: partition name on the professor's cluster
#SBATCH --gres=gpu:1             # ADJUST: GPU request syntax may differ
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Generic SLURM template for RL training. Connection method, partition names,
# and the module system for the professor's private cluster are TBD — adjust the
# #SBATCH lines and the module loads below once access is established.

set -euo pipefail

# --- environment ---------------------------------------------------------
# module load cuda/12.4          # ADJUST or remove depending on cluster
source .env 2>/dev/null || true  # picks up MLFLOW_TRACKING_URI, MP_API_KEY, etc.

# Log to the MLflow server running on the main workstation. Set this in .env to
# the workstation's reachable address, e.g. http://10.0.0.5:5000
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

# --- run -----------------------------------------------------------------
echo "[hpc] node: $(hostname)  job: ${SLURM_JOB_ID:-local}"
uv run python -m synth_q_mat.rl.train ${ARGS:-}

echo "[hpc] training finished; sync checkpoints with: make push"
