# Model Bill of Materials for vit-stt

This file tracks model/runtime assets used by vit-stt for offline STT deployment.

| Component | Source | Local path | License | Modified weights? | Bundled in default release archive? | Pre-downloaded on handoff server? | Checksum source |
|---|---|---|---|---:|---:|---:|---|
| Gipformer encoder | g-group-ai-lab/gipformer-65M-rnnt | models/stt/gipformer-65M-rnnt/encoder-epoch-35-avg-6.int8.onnx | MIT | No | No | Yes | baselines/phase0/assets.lock.json |
| Gipformer decoder | g-group-ai-lab/gipformer-65M-rnnt | models/stt/gipformer-65M-rnnt/decoder-epoch-35-avg-6.int8.onnx | MIT | No | No | Yes | baselines/phase0/assets.lock.json |
| Gipformer joiner | g-group-ai-lab/gipformer-65M-rnnt | models/stt/gipformer-65M-rnnt/joiner-epoch-35-avg-6.int8.onnx | MIT | No | No | Yes | baselines/phase0/assets.lock.json |
| Gipformer tokens/BPE | g-group-ai-lab/gipformer-65M-rnnt | models/stt/gipformer-65M-rnnt/ | MIT | No | No | Yes | baselines/phase0/assets.lock.json |
| Parakeet ONNX package | sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming | models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/ | CC-BY-4.0 / sherpa-onnx package notices | No | No | Yes | baselines/phase0/assets.lock.json |
| viBERT-CaPU model weights/code/vocabulary | leakless/vibert-capu / dragonSwing/vibert-capu | models/capu/vibert-capu | CC-BY-SA-4.0 | No | No | Yes | baselines/phase0/assets.lock.json |
| viBERT-CaPU base tokenizer/config | FPTAI/vibert-base-cased, repackaged under leakless/vibert-capu/base_model | models/capu/vibert-capu/base_model/ | No explicit source license metadata observed | No | No | Yes | baselines/phase0/assets.lock.json |
| ten-vad | k2-fsa/sherpa-onnx release assets | models/vad/ten-vad.int8.onnx | Apache-2.0 | No | No | Yes | baselines/phase0/assets.lock.json |

## Distribution modes

### Default release archive

The default release archive should not include model weights unless explicitly approved.
The customer/operator may run the setup/download command to fetch model assets.

### Offline model pack

If a fully offline deployment requires pre-bundled model files, distribute them
as a separate model pack with:

- this MODEL_BOM.md;
- THIRD_PARTY_NOTICES.md;
- all relevant license texts;
- exact source URLs;
- checksums;
- confirmation that weights were not modified.

## Server pre-seeding

For normal customer delivery, model files are downloaded onto the target server
before handoff using `stt-cli download-models`.

This means the customer receives model files on the server. The deployment must
include this `MODEL_BOM.md`, `THIRD_PARTY_NOTICES.md`, and the relevant license
texts.

Weights are not modified by VietInnotech.

## Release legal posture

- `ten-vad.int8.onnx` is distributed as a k2-fsa/sherpa-onnx release asset.
  The source repository license is Apache-2.0, so releases that include this
  asset must include the Apache-2.0 license text and retain applicable notices.
- `leakless/vibert-capu` and its upstream `dragonSwing/vibert-capu` source
  declare CC-BY-SA-4.0. Releases or model packs that include those files must
  include the CC-BY-SA-4.0 license text, preserve attribution, identify the
  source repository, and avoid additional legal or technical restrictions that
  would conflict with CC-BY-SA-4.0.
- `models/capu/vibert-capu/base_model/` contains tokenizer/config files derived
  from `FPTAI/vibert-base-cased`. As of the 2026-06-30 source check, the
  Hugging Face API and model card did not expose explicit license metadata for
  that source. Do not include these base-model files in a redistributable
  offline model pack until VietInnotech has documented a license grant,
  permission, or replacement source. Server pre-seeding for a named customer
  deployment should keep this BOM and notice file with the deployed assets.
- No additional source-specific attribution text was observed beyond preserving
  source names, source URLs, license texts, checksums, and "no model weight
  modification" status in this BOM and `THIRD_PARTY_NOTICES.md`.
