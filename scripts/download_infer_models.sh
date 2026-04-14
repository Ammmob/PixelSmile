#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${1:-./weights}"

echo "[INFO] Target weights directory: ${WEIGHTS_DIR}"
mkdir -p "${WEIGHTS_DIR}"

echo "[INFO] Downloading Qwen base model..."
hf download Qwen/Qwen-Image-Edit-2511 \
  --local-dir "${WEIGHTS_DIR}/Qwen-Image-Edit-2511"

echo "[INFO] Downloading PixelSmile preview LoRA..."
hf download PixelSmile/PixelSmile PixelSmile-preview.safetensors \
  --local-dir "${WEIGHTS_DIR}"

echo "[INFO] Done."
