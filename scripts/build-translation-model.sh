#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO_ROOT/.cache/envit5"
UPSTREAM="$WORK/upstream"
VENV="$WORK/venv"
CT2="$WORK/ct2"
PACKROOT="$WORK/packroot"
OUTPUT="$REPO_ROOT/meta-meeting-server/recipes-ai/meeting-translation-models/files/envit5-ct2.tar"

HF_REV="840bc88104d5a4277af740eaedb024df8c3093e7"
EXPECTED_PT="eef48b3eee23aae577e965ce8da5b2e9dcadfc4d08a85e2a302ef4b929fb613e"
EXPECTED_SP="3b4eda923bbac1726e8fda66254a8783ecc705be5577149ee8c98074efdb5de5"
EXPECTED_PACK="e5ddd932173cca21fd181eab4b2888955ff291ce0cbefcdca1fdb5333bd01588"

mkdir -p "$WORK" "$(dirname "$OUTPUT")"
rm -rf "$VENV" "$CT2" "$PACKROOT"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
"$VENV/bin/python" -m pip install \
    "huggingface_hub==0.34.4" \
    "transformers==4.56.2" \
    "ctranslate2==4.8.1" \
    "sentencepiece==0.2.1"

if [[ ! -s "$UPSTREAM/pytorch_model.bin" ]] || [[ ! -s "$UPSTREAM/spiece.model" ]]; then
    rm -rf "$UPSTREAM"
    "$VENV/bin/python" - "$UPSTREAM" "$HF_REV" <<'PY'
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

PT="$(sha256sum "$UPSTREAM/pytorch_model.bin" | awk '{print $1}')"
SP="$(sha256sum "$UPSTREAM/spiece.model" | awk '{print $1}')"
[[ "$PT" == "$EXPECTED_PT" ]]
[[ "$SP" == "$EXPECTED_SP" ]]

"$VENV/bin/ct2-transformers-converter" \
    --model "$UPSTREAM" \
    --output_dir "$CT2" \
    --quantization int8_float32 \
    --force

cp -a "$UPSTREAM/spiece.model" "$CT2/spiece.model"

cat > "$CT2/PROVENANCE.txt" <<EOF
HL Meet EnViT5 translation asset
upstream_repository=VietAI/envit5-translation
upstream_revision=$HF_REV
source_pytorch_model_sha256=$EXPECTED_PT
source_spiece_sha256=$EXPECTED_SP
converter_ctranslate2=4.8.1
converter_transformers=4.56.2
converter_torch=2.6.0+cpu
converter_sentencepiece=0.2.1
conversion_quantization=int8_float32
runtime_device=cpu
runtime_compute_type=int8_float32
translation_is_post_asr=true
canonical_stt_must_never_be_rewritten=true
EOF

(
    cd "$CT2"
    find . -type f ! -name ASSETS.sha256 -print0 | sort -z | xargs -0 sha256sum > ASSETS.sha256
)

mkdir -p "$PACKROOT/envit5"
cp -a "$CT2"/. "$PACKROOT/envit5/"

tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner --format=gnu \
    -C "$PACKROOT" -cf "$OUTPUT" envit5

ACTUAL="$(sha256sum "$OUTPUT" | awk '{print $1}')"
echo "EXPECTED_PACK_SHA256=$EXPECTED_PACK"
echo "ACTUAL_PACK_SHA256=$ACTUAL"
[[ "$ACTUAL" == "$EXPECTED_PACK" ]]
echo "PASS: deterministic EnViT5 pack reproduced"
echo "OUTPUT=$OUTPUT"
