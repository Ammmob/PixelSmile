#!/usr/bin/env bash

set -e

TARGET_FILE=$(python3 -c "import importlib.util; import pathlib; spec = importlib.util.find_spec('diffusers'); assert spec and spec.origin, 'diffusers is not installed in the current environment'; root = pathlib.Path(spec.origin).resolve().parent; print(root / 'pipelines' / 'qwenimage' / 'pipeline_qwenimage_edit_plus.py')")

MATCH_LINE="if prompt_embeds_mask is not None and prompt_embeds_mask.all()"

sed -i "
/^[[:space:]]*$MATCH_LINE/ {
    /^[[:space:]]*#/! s/^/# /
    n
    /^[[:space:]]*#/! s/^/# /
}
" "$TARGET_FILE"

echo "[INFO] Patched Qwen diffusers bug in: $TARGET_FILE"
