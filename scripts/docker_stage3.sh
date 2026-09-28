#!/usr/bin/env bash
# Run a command inside the pre-existing pytorch image with exactly ONE GPU visible.
# Never builds/pulls images (root disk is full). All caches and outputs live on /raid.
#
#   PYLIBS=/raid/<you>/pylibs scripts/docker_stage3.sh <GPU-UUID> python scripts/check_nvml.py --expect-uuid <GPU-UUID>
#
# Inside the container the chosen GPU is always index 0, so use --gpu-index 0 (default).
set -euo pipefail
UUID="${1:?usage: docker_stage3.sh <GPU-UUID> <command...>}"; shift
: "${PYLIBS:?set PYLIBS to the directory holding the pip packages (on /raid)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${CACHE:-$REPO/../cache}"; mkdir -p "$CACHE" "$REPO/data"
nvidia-smi -L | grep -q "$UUID" || { echo "UUID $UUID not found on this host"; exit 1; }
exec docker run --rm --pull=never \
  --gpus "\"device=$UUID\"" \
  -v "$REPO":/work -v "$PYLIBS":/pylibs:ro -v "$CACHE":/cache \
  -e PYTHONPATH=/pylibs -e HF_HOME=/cache/hf -e TORCH_HOME=/cache/torch \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -w /work pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime "$@"
