#!/bin/bash
#SBATCH --job-name=perch2-server
#SBATCH --partition=a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/logs/perch2-server-%j.out

# ---------------------------------------------------------------------------
# Perch 2 FastAPI server — persistent Slurm job
#
# Usage:
#   sbatch slurm_job.sh
#
# Clients discover this server by reading the address file written below.
# ---------------------------------------------------------------------------

set -e

PORT=8002
REPO_DIR="$HOME/perch2-server"
ADDR_FILE="${PERCH2_ADDR_FILE:-$REPO_DIR/server.addr}"

# Write hostname:port so clients can find us
mkdir -p "$(dirname "$ADDR_FILE")"
echo "$(hostname):$PORT" > "$ADDR_FILE"
echo "Server address: $(cat $ADDR_FILE)"

# PERCH2_VERSION: integer Kaggle model version (default: 1).
# The model is downloaded automatically on first run via kagglehub.
export PERCH2_VERSION="${PERCH2_VERSION:-1}"

# TF GPU libraries are not available on this cluster; always use the CPU model.
export PERCH2_CPU=1

cd "$REPO_DIR"
mkdir -p logs

uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" soundfile librosa \
  "perch_hoplite @ git+https://github.com/google-research/perch-hoplite.git@f9f388d67a9153de6eaf83b669356e1f33ce6843" \
  "tensorflow[and-cuda]" ml_collections
uv pip install --python .venv/bin/python "kagglehub>=0.4.1"

export CUDA_VISIBLE_DEVICES=""
.venv/bin/uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
