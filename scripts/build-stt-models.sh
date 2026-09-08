#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/pins.env"

need() { command -v "$1" >/dev/null 2>&1 || { echo "FAIL: missing command: $1" >&2; exit 1; }; }
for c in curl tar sha256sum zstd; do need "$c"; done

WORK="$REPO_ROOT/.cache/stt-model-assets"
DL="$WORK/downloads"
EXTRACT="$WORK/extract"
PACKROOT="$WORK/packroot"
OUT="$REPO_ROOT/meta-meeting-server/recipes-ai/meeting-vit-stt/files/vit-stt-meeting-models.tar.zst"
mkdir -p "$DL" "$(dirname "$OUT")"
rm -rf "$EXTRACT" "$PACKROOT"
mkdir -p "$EXTRACT" "$PACKROOT"

download() {
    local url="$1" out="$2"
    if [[ ! -s "$out" ]]; then
        curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 -o "$out.part" "$url"
        mv -f "$out.part" "$out"
    fi
}

verify() {
    local expected="$1" file="$2"
    [[ -f "$file" ]] || { echo "FAIL: missing $file" >&2; exit 1; }
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "FAIL: SHA mismatch: $file" >&2
        echo " expected=$expected" >&2
        echo " actual=$actual" >&2
        exit 1
    }
}

VI_ARC="$DL/${VI_MODEL_DIR}.tar.bz2"
EN_ARC="$DL/${EN_MODEL_DIR}.tar.bz2"
download "$VI_MODEL_URL" "$VI_ARC"
download "$EN_MODEL_URL" "$EN_ARC"

tar -xjf "$VI_ARC" -C "$EXTRACT"
tar -xjf "$EN_ARC" -C "$EXTRACT"

VI="$EXTRACT/$VI_MODEL_DIR"
EN="$EXTRACT/$EN_MODEL_DIR"

verify "$VI_ENCODER_SHA256" "$VI/encoder-epoch-12-avg-8.int8.onnx"
verify "$VI_DECODER_SHA256" "$VI/decoder-epoch-12-avg-8.onnx"
verify "$VI_JOINER_SHA256" "$VI/joiner-epoch-12-avg-8.int8.onnx"
verify "$VI_TOKENS_SHA256" "$VI/tokens.txt"
verify "$VI_BPE_SHA256" "$VI/bpe.model"

verify "$EN_ENCODER_SHA256" "$EN/encoder.int8.onnx"
verify "$EN_DECODER_SHA256" "$EN/decoder.int8.onnx"
verify "$EN_JOINER_SHA256" "$EN/joiner.int8.onnx"
verify "$EN_TOKENS_SHA256" "$EN/tokens.txt"

mkdir -p "$PACKROOT/$VI_MODEL_DIR" "$PACKROOT/$EN_MODEL_DIR"
cp -a \
  "$VI/encoder-epoch-12-avg-8.int8.onnx" \
  "$VI/decoder-epoch-12-avg-8.onnx" \
  "$VI/joiner-epoch-12-avg-8.int8.onnx" \
  "$VI/tokens.txt" \
  "$VI/bpe.model" \
  "$PACKROOT/$VI_MODEL_DIR/"
cp -a \
  "$EN/encoder.int8.onnx" \
  "$EN/decoder.int8.onnx" \
  "$EN/joiner.int8.onnx" \
  "$EN/tokens.txt" \
  "$PACKROOT/$EN_MODEL_DIR/"

TMP="$OUT.tmp"
rm -f "$TMP" "$OUT"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner --format=gnu \
  -C "$PACKROOT" -cf - "$VI_MODEL_DIR" "$EN_MODEL_DIR" \
  | zstd -19 -T1 -q -o "$TMP"
mv -f "$TMP" "$OUT"

echo "PASS: STT model pack generated"
echo "OUTPUT=$OUT"
sha256sum "$OUT"
