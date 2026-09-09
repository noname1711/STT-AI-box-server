#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/pins.env"

WORK="$REPO_ROOT/.cache/envit5"
UPSTREAM="$WORK/upstream"
VENV="$WORK/venv"
CT2="$WORK/ct2"
PACKROOT="$WORK/packroot"
OUTPUT="$REPO_ROOT/meta-meeting-server/recipes-ai/meeting-translation-models/files/envit5-ct2.tar"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

verify() {
    local expected="$1"
    local file="$2"
    local actual

    [[ -f "$file" ]] || fail "missing $file"

    actual="$(sha256sum "$file" | awk '{print $1}')"

    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: SHA mismatch: $file" >&2
        echo " expected=$expected" >&2
        echo " actual=$actual" >&2
        exit 1
    fi
}

mkdir -p "$WORK" "$(dirname "$OUTPUT")"
rm -rf "$VENV" "$CT2" "$PACKROOT"

python3 -m venv "$VENV"

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

"$VENV/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0"

"$VENV/bin/python" -m pip install \
    "huggingface_hub==0.34.4" \
    "transformers==4.56.2" \
    "ctranslate2==4.8.1" \
    "sentencepiece==0.2.1"

if [[ ! -s "$UPSTREAM/pytorch_model.bin" ]] ||
   [[ ! -s "$UPSTREAM/spiece.model" ]]; then

    rm -rf "$UPSTREAM"

    "$VENV/bin/python" - "$UPSTREAM" "$ENVIT5_HF_REVISION" <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

snapshot_download(
    repo_id="VietAI/envit5-translation",
    revision=sys.argv[2],
    local_dir=str(Path(sys.argv[1])),
    allow_patterns=[
        "README.md",
        "config.json",
        "pytorch_model.bin",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)
PY
fi

verify "$ENVIT5_PT_SHA256" \
       "$UPSTREAM/pytorch_model.bin"

verify "$ENVIT5_SPIECE_SHA256" \
       "$UPSTREAM/spiece.model"

"$VENV/bin/ct2-transformers-converter" \
    --model "$UPSTREAM" \
    --output_dir "$CT2" \
    --quantization int8_float32 \
    --force

cp -a "$UPSTREAM/spiece.model" "$CT2/spiece.model"

#
# Production certification is defined by runtime payload bytes,
# not by container/provenance metadata.
#
verify "$ENVIT5_MODEL_SHA256" \
       "$CT2/model.bin"

verify "$ENVIT5_CONFIG_SHA256" \
       "$CT2/config.json"

verify "$ENVIT5_SHARED_VOCAB_SHA256" \
       "$CT2/shared_vocabulary.json"

verify "$ENVIT5_SPIECE_SHA256" \
       "$CT2/spiece.model"

cat > "$CT2/PROVENANCE.txt" <<EOF2
HL Meet EnViT5 translation asset
upstream_repository=$ENVIT5_HF_REPO
upstream_revision=$ENVIT5_HF_REVISION
source_pytorch_model_sha256=$ENVIT5_PT_SHA256
source_spiece_sha256=$ENVIT5_SPIECE_SHA256
runtime_model_sha256=$ENVIT5_MODEL_SHA256
runtime_config_sha256=$ENVIT5_CONFIG_SHA256
runtime_shared_vocabulary_sha256=$ENVIT5_SHARED_VOCAB_SHA256
converter_ctranslate2=4.8.1
converter_transformers=4.56.2
converter_torch=2.6.0+cpu
converter_sentencepiece=0.2.1
conversion_quantization=int8_float32
runtime_device=cpu
runtime_compute_type=int8_float32
translation_is_post_asr=true
canonical_stt_must_never_be_rewritten=true
runtime_certification=inner_sha256
EOF2

(
    cd "$CT2"

    sha256sum \
        config.json \
        model.bin \
        shared_vocabulary.json \
        spiece.model \
        > ASSETS.sha256
)

mkdir -p "$PACKROOT/envit5"
cp -a "$CT2"/. "$PACKROOT/envit5/"

rm -f "$OUTPUT"

tar \
    --sort=name \
    --mtime='@0' \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    --format=gnu \
    -C "$PACKROOT" \
    -cf "$OUTPUT" \
    envit5

ACTUAL_PACK="$(sha256sum "$OUTPUT" | awk '{print $1}')"

echo "REFERENCE_HISTORICAL_PACK_SHA256=$ENVIT5_REFERENCE_PACK_SHA256"
echo "ACTUAL_PACK_SHA256=$ACTUAL_PACK"
echo "PASS: EnViT5 runtime payload reproduced and certified"
echo "OUTPUT=$OUTPUT"
