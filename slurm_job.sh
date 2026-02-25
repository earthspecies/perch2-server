#!/bin/bash
#SBATCH --job-name=perch2-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/perch2-server-%j.out

# ---------------------------------------------------------------------------
# Perch 2 FastAPI server — persistent Slurm job
#
# Usage:
#   sbatch slurm_job.sh
#
# Clients discover this server by reading the address file written below.
# ---------------------------------------------------------------------------

set -e

PORT=8000
ADDR_FILE="${PERCH2_ADDR_FILE:-/home/anthony_earthspecies_org/dev/perch2-server/server.addr}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Write hostname:port so clients can find us
mkdir -p "$(dirname "$ADDR_FILE")"
echo "$(hostname):$PORT" > "$ADDR_FILE"
echo "Server address: $(cat $ADDR_FILE)"

# PERCH2_VERSION: integer Kaggle model version (default: 1).
# The model is downloaded automatically on first run via kagglehub.
export PERCH2_VERSION="${PERCH2_VERSION:-1}"

cd "$REPO_DIR"
mkdir -p logs

uv run uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
