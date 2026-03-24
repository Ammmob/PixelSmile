#!/usr/bin/env bash

DEFAULT_ARGS=(
  --image-path "/path/to/input.jpg"
  --output-dir "/path/to/output"
  --model-path "/path/to/Qwen-Image-Edit-2511"
  --lora-path "/path/to/PixelSmile-preview.safetensors"
  --expression "happy"
  --scales 0 0.5 1.0 1.5
  --data-type "human"
  --seed 42
)

python pixelsmile/infer.py "${DEFAULT_ARGS[@]}" "$@"