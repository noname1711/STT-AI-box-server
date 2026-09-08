# Third-Party Notices for vit-stt

This product includes third-party software, models, and model assets.
Those components are licensed by their respective owners. VietInnotech's
commercial license applies only to VietInnotech-owned code and materials.

## Models and AI assets

### Gipformer 65M RNNT
- Source: g-group-ai-lab/gipformer-65M-rnnt
- License: MIT
- Purpose: Vietnamese automatic speech recognition
- Modified by VietInnotech: No model weight modification

### NVIDIA Parakeet / sherpa-onnx converted package
- Source: NVIDIA Parakeet model and sherpa-onnx packaged ONNX assets
- License: CC-BY-4.0 for NVIDIA Parakeet, plus applicable sherpa-onnx package notices
- Purpose: English automatic speech recognition
- Modified by VietInnotech: No model weight modification

### leakless/vibert-capu
- Source: leakless/vibert-capu
- Upstream CAPU model: dragonSwing/vibert-capu
- License: CC-BY-SA-4.0
- Purpose: Vietnamese capitalization and punctuation restoration
- Modified by VietInnotech: No model weight modification
- Note: This package consolidates CAPU model files and base tokenizer/config files.
  The CAPU model weights/code/vocabulary declare CC-BY-SA-4.0. The
  `base_model/` tokenizer/config files are derived from `FPTAI/vibert-base-cased`,
  for which no explicit source license metadata was observed in the 2026-06-30
  source check. Do not redistribute those `base_model/` files in an offline
  model pack until a license grant, permission, or replacement source is
  documented.

### ten-vad.int8.onnx
- Source: k2-fsa/sherpa-onnx release assets
- Purpose: Voice activity detection
- Modified by VietInnotech: No model weight modification
- License: Apache-2.0

## Server pre-seeded model deployment

In the standard offline/on-premise handoff, VietInnotech may download model
files onto the target server before transferring operation to the customer.

Those model files remain third-party materials under their original licenses.
VietInnotech's proprietary license applies only to VietInnotech-owned software.

The handoff server must include:

```text
THIRD_PARTY_NOTICES.md
MODEL_BOM.md
licenses/
```

Do not apply additional legal or technical restrictions to CC-BY-SA model files
that would prevent rights allowed by that license.

## Runtime libraries

### sherpa-onnx
- License: Apache-2.0
- Purpose: Offline speech recognition runtime

### Rust crates
This product includes Rust dependencies listed in Cargo.toml and Cargo.lock.
A generated dependency license report should be distributed with each release.

## Important distribution note

If model weights are not included in the release archive and are downloaded
during setup, keep this notice file in the product package anyway.

If model weights are bundled into a separate offline model pack, include this
notice file, MODEL_BOM.md, and all relevant license texts with that model pack.

The current offline model-pack restriction is specific to
`models/capu/vibert-capu/base_model/`: do not include those files in a
redistributable model pack until VietInnotech documents the source license,
permission, or a replacement source.
