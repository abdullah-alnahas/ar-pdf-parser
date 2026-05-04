#!/usr/bin/env bash
# Launch the Triton server hosting the surya_ocr Python backend.
#
# Usage:
#   triton/run.sh                # default: builds image if missing, runs
#   triton/run.sh --rebuild      # force rebuild before run
#   triton/run.sh --no-gpu       # FORCE_CPU=1 for laptops without CUDA
#
# Exposed ports:
#   8000 - HTTP, 8001 - gRPC (CLI uses this), 8002 - metrics
#
# Volumes:
#   triton/models -> /models                     (model repository)
#   $HOME/.cache/huggingface -> /root/.cache/... (surya weights persist)

set -euo pipefail

IMAGE="${TRITON_IMAGE:-apt-triton}"
TAG_BASE="${TRITON_BASE:-nvcr.io/nvidia/tritonserver:24.10-py3}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/triton/models"

REBUILD=0
GPU_FLAG="--gpus=all"
EXTRA_ENV=()
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --no-gpu)  GPU_FLAG=""; EXTRA_ENV+=("-e" "FORCE_CPU=1") ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$MODEL_DIR/surya_ocr/1" ]]; then
    echo "error: model dir $MODEL_DIR/surya_ocr/1 not found" >&2
    exit 1
fi

if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[run.sh] building $IMAGE from $TAG_BASE..."
    docker build -t "$IMAGE" "$REPO_ROOT/triton"
fi

mkdir -p "$HF_CACHE"

set -x
exec docker run --rm -it \
    $GPU_FLAG \
    --shm-size=2g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$MODEL_DIR:/models" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "${EXTRA_ENV[@]}" \
    "$IMAGE" \
    tritonserver --model-repository=/models --log-verbose=1
