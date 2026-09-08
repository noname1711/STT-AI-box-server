#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ENV="${1:-${SDK_ENV:-}}"
"$ROOT/scripts/build-stt-models.sh"
"$ROOT/scripts/build-translation-model.sh"
"$ROOT/scripts/build-native-stt.sh" "$SDK_ENV"
echo "PASS: all generated STT/translation assets are ready"
