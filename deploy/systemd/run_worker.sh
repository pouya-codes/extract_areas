#!/usr/bin/env bash
# Launch one model worker pinned to one GPU: run_worker.sh <model>:<gpu>
set -euo pipefail
SPEC="${1:?usage: run_worker.sh <model>:<gpu_index>}"
MODEL="${SPEC%%:*}"
GPU="${SPEC##*:}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/extract_areas}"

cd "$REPO_DIR"
export MODEL_NAME="$MODEL"
export CUDA_VISIBLE_DEVICES="$GPU"
export WORKER_HTTP_PORT="${WORKER_HTTP_PORT:-$((9100 + GPU))}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-gpu${GPU}}"
mkdir -p "$MPLCONFIGDIR"
if [[ -f "$REPO_DIR/.env" ]]; then set -a; source "$REPO_DIR/.env"; set +a; fi

echo "Starting worker model=${MODEL} gpu=${GPU} http=:${WORKER_HTTP_PORT}"
exec "$CONDA_ENV/bin/python" -m worker.main
