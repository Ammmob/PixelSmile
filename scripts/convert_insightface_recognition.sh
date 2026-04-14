#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:?Usage: bash scripts/convert_insightface_recognition.sh <glintr100.onnx> [glintr100.pth]}"
PTH_PATH="${2:-${ONNX_PATH%.onnx}.pth}"

echo "[INFO] Converting ${ONNX_PATH} -> ${PTH_PATH} via onnx2torch..."
python - "${ONNX_PATH}" "${PTH_PATH}" <<'PY'
import sys
import onnx
import torch
from onnx2torch import convert

onnx_path = sys.argv[1]
pth_path = sys.argv[2]
model = convert(onnx.load(onnx_path))
torch.save(model, pth_path)
print(f"[INFO] Saved {pth_path}")
PY

echo "[INFO] Done."
