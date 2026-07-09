#!/usr/bin/env bash
# Download a quantized Gemma model into models/model.gguf, which is bundled into
# the container image. Run before building the image.
#
# Usage:
#   ./download_model.sh 1b     Smaller and faster on CPU
#   ./download_model.sh 4b     More accurate, the default choice
#
# The source repository and file can be overridden:
#   REPO=unsloth/gemma-3-4b-it-GGUF FILE=gemma-3-4b-it-Q4_K_M.gguf ./download_model.sh
set -euo pipefail

SIZE="${1:-4b}"
mkdir -p models

if [[ "$SIZE" == "1b" ]]; then
  REPO="${REPO:-unsloth/gemma-3-1b-it-GGUF}"
  FILE="${FILE:-gemma-3-1b-it-Q4_K_M.gguf}"
else
  REPO="${REPO:-unsloth/gemma-3-4b-it-GGUF}"
  FILE="${FILE:-gemma-3-4b-it-Q4_K_M.gguf}"
fi

echo "Downloading $FILE from $REPO"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" "$FILE" --local-dir models
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" "$FILE" --local-dir models
else
  curl -L -o "models/$FILE" "https://huggingface.co/$REPO/resolve/main/$FILE?download=true"
fi

cp -f "models/$FILE" models/model.gguf
echo "Ready: models/model.gguf ($(du -h models/model.gguf | cut -f1))"
