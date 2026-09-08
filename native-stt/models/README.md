# Models

`vit-stt` expects model assets under this directory.

Current expected local layout:

```text
models/
  stt/
    gipformer-65M-rnnt/
    sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/
  vad/
    ten-vad.int8.onnx
  capu/
    vibert-capu/
      base_model/
```

The workspace currently uses these paths when assets are materialized locally:

- `models/stt/gipformer-65M-rnnt`
- `models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming`
- `models/vad/ten-vad.int8.onnx`
- `models/capu/vibert-capu`

The runtime-facing CAPU model is self-contained: the fine-tuned CAPU files live
at the model root and the ViBERT base config/vocab live under `base_model/`.
It is downloaded from `leakless/vibert-capu`. Older local directories such as
separately assembled CAPU/base-model folders are no longer the supported
production download path.

Asset versions are locked in:

```text
baselines/phase0/assets.lock.json
```

This repository intentionally does **not** commit large model assets or machine-local symlinks.

Experimental CAPU candidates and generated ONNX/IR artifacts belong outside
`models/`, for example under ignored `target/` subdirectories. Do not commit
downloaded candidate weights.

Materialize local assets with:

```bash
stt-cli download-models
```

That command follows `baselines/phase0/assets.lock.json` and downloads only
locked production assets. `scripts/download-assets` is a legacy Python helper
for maintainers; normal setup should use `stt-cli download-models`.
