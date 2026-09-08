#!/usr/bin/env bash
set -euo pipefail

target_dir="${1:-}"
if [[ -z "$target_dir" ]]; then
  echo "Usage: scripts/package-legal-files.sh <target-dir>" >&2
  exit 1
fi

mkdir -p "$target_dir/licenses"
cp LICENSE "$target_dir/LICENSE"
cp THIRD_PARTY_NOTICES.md "$target_dir/THIRD_PARTY_NOTICES.md"
cp MODEL_BOM.md "$target_dir/MODEL_BOM.md"
cp DEPENDENCY_BOM.md "$target_dir/DEPENDENCY_BOM.md" 2>/dev/null || true
cp -R licenses/. "$target_dir/licenses/"
