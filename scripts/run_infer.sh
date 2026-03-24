#!/usr/bin/env bash

IMAGE_PATH="/path/to/input.jpg"
OUTPUT_DIR="/path/to/output"
MODEL_PATH="/path/to/Qwen-Image-Edit-2511"
LORA_PATH="/path/to/PixelSmile-preview.safetensors"
EXPRESSION="happy"
SCALES=(0.4 0.8 1.2)
SEED=42

python PixelSmile/infer.py \
  --image-path "${IMAGE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --lora-path "${LORA_PATH}" \
  --expression "${EXPRESSION}" \
  --scales "${SCALES[@]}" \
  --seed "${SEED}"
