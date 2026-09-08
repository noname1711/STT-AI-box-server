#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${VIT_STT_RELEASE_BIN_DIR:-$ROOT/bin}"
STT_CLI="${VIT_STT_CLI:-$BIN_DIR/stt-cli}"

if [[ ! -x "$STT_CLI" ]]; then
  if [[ -x "$ROOT/target/release/stt-cli" ]]; then
    STT_CLI="$ROOT/target/release/stt-cli"
  else
    echo "missing stt-cli. Set VIT_STT_CLI=/absolute/path/to/stt-cli" >&2
    exit 1
  fi
fi

echo "Preparing vit-stt models on server..."
"$STT_CLI" download-models
"$STT_CLI" verify-assets
"$STT_CLI" check-capu
"$STT_CLI" probe --model vit_stt_vi_v2
"$STT_CLI" probe --model vit_stt_en_v2

echo "Model preparation complete."
echo "Ensure LICENSE, THIRD_PARTY_NOTICES.md, MODEL_BOM.md, DEPENDENCY_BOM.md, and licenses/ are present in the deployment root."
