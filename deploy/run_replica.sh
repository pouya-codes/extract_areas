#!/bin/bash
# Launch one GPU-pinned replica of the inference API.
#
# Usage: run_replica.sh <gpu_index>
#
# Each replica is pinned to a single physical GPU via CUDA_VISIBLE_DEVICES, so
# torch.cuda.device_count() is 1 inside the process. Both DeepLIIF (gpu_ids: [])
# and HoVer-Net (gpu_ids: [0]) then resolve to that one card without any
# per-replica config changes.

set -euo pipefail

GPU_INDEX="${1:?usage: run_replica.sh <gpu_index>}"
BASE_PORT="${BASE_PORT:-8001}"
PORT=$((BASE_PORT + GPU_INDEX))

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/extract_areas}"

# deepliif_model.py does sys.path.append("module/DeepLiff"), a CWD-relative
# path, so the process must run from the repo root.
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export AI_REPLICA_GPU="$GPU_INDEX"
export AI_REPLICA_PORT="$PORT"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# python-javabridge (pulled in by DeepLIIF via bioformats) needs a JVM.
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"

# 32 cores across 4 replicas; keep BLAS from oversubscribing the box.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-gpu${GPU_INDEX}}"
mkdir -p "$MPLCONFIGDIR"

echo "Starting replica on GPU ${GPU_INDEX} -> 127.0.0.1:${PORT}"

exec "$CONDA_ENV/bin/uvicorn" app_refactored:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 75 \
    --log-level info
