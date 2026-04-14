#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${1:-./weights}"

echo "[INFO] Target weights directory: ${WEIGHTS_DIR}"
mkdir -p "${WEIGHTS_DIR}"

echo "[INFO] Downloading CLIP (human)..."
hf download openai/clip-vit-large-patch14 \
  --local-dir "${WEIGHTS_DIR}/clip-vit-large-patch14"

echo "[INFO] Downloading CLIP (anime)..."
hf download OysterQAQ/DanbooruCLIP \
  --local-dir "${WEIGHTS_DIR}/DanbooruCLIP"

echo "[INFO] Done."
