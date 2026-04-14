#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${1:-./weights}"
INSIGHTFACE_ROOT="${WEIGHTS_DIR}/antelopev2"
INSIGHTFACE_ZIP="${WEIGHTS_DIR}/antelopev2.zip"
INSIGHTFACE_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Target weights directory: ${WEIGHTS_DIR}"
mkdir -p "${WEIGHTS_DIR}"
if [[ ! -w "${WEIGHTS_DIR}" ]]; then
  echo "[ERROR] No write permission for ${WEIGHTS_DIR}"
  echo "[ERROR] Please fix permissions first, then rerun this script."
  exit 1
fi

echo "[INFO] Downloading InsightFace antelopev2..."
# curl -L "${INSIGHTFACE_URL}" -o "${INSIGHTFACE_ZIP}"

# The zip already contains the top-level folder `antelopev2/`.
# Extract to WEIGHTS_DIR so final files land in ${WEIGHTS_DIR}/antelopev2/.
unzip -o "${INSIGHTFACE_ZIP}" -d "${WEIGHTS_DIR}"

SCRFD_ONNX="${INSIGHTFACE_ROOT}/scrfd_10g_bnkps.onnx"
GLINTR_ONNX="${INSIGHTFACE_ROOT}/glintr100.onnx"

if [[ ! -f "${SCRFD_ONNX}" || ! -f "${GLINTR_ONNX}" ]]; then
  echo "[ERROR] Expected files not found after unzip:"
  echo "        ${SCRFD_ONNX}"
  echo "        ${GLINTR_ONNX}"
  exit 1
fi

bash "${SCRIPT_DIR}/convert_insightface_recognition.sh" \
  "${GLINTR_ONNX}" \
  "${INSIGHTFACE_ROOT}/glintr100.pth"

echo "[INFO] Done."
