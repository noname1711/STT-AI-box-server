# VAD Migration Guide: Silero VAD → TEN VAD

This document describes the migration from Silero VAD to TEN VAD in `vit-stt`.

## Why TEN VAD?

- **Smaller model file**: `ten-vad.int8.onnx` is ~130 KB vs `silero_vad.onnx` at ~630 KB (5× reduction).
- **Actively supported**: The `sherpa-onnx` crate has moved to TEN VAD as the default.
- **Int8 quantized**: Lower inference overhead while maintaining comparable accuracy.
- **Cleaner API**: TEN VAD exposes a `window_size` parameter that aligns directly with the VAD processing pipeline.

## What Changed

| Aspect | Before | After |
|---|---|---|
| Model file | `models/vad/silero_vad.onnx` | `models/vad/ten-vad.int8.onnx` |
| VAD profile | `custom` | `ten` |
| Rust config struct | `SileroVadModelConfig` | `TenVadModelConfig` |
| `VadModelConfig.silero_vad` | populated with config | `Default::default()` |
| `VadModelConfig.ten_vad` | `Default::default()` | populated with config |
| Window size | default | `256` |
| Asset lock ID | `vad.silero` | `vad.ten` |
| Download source | silero_vad.onnx URL | ten-vad.int8.onnx URL |

## Upgrade Steps

### 1. Download the new TEN VAD model

```bash
# Using the Rust CLI (preferred — follows the asset lock):
stt-cli download-models

# Or using the Python helper:
python scripts/download_assets.py vad --force
```

### 2. Update the VAD profile in `config/runtime.toml`

```toml
[vad]
enabled = true
profile = "ten"            # was "custom"
threshold = 0.5
min_silence = 0.5
min_speech = 0.05
max_speech = 14.0
```

### 3. Remove the old Silero VAD file (optional)

The old model file is not used after the migration but can be safely removed:

```bash
rm models/vad/silero_vad.onnx
```

The file is gitignored (`/models/vad/*`), so it will not reappear on fresh checkouts.

### 4. Verify the migration

```bash
# Check that TEN VAD is picked up and asset checksums match:
stt-cli verify-assets

# Run a probe transcription to confirm VAD works end-to-end:
stt-cli probe --model vit_stt_vi_v2
stt-cli probe --model vit_stt_en_v2
```

## Rollback

If you need to revert to Silero VAD:

1. Download `silero_vad.onnx` from the [k2-fsa sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx) and place it at `models/vad/silero_vad.onnx`.
2. Change `config/runtime.toml` profile back to `custom`.
3. In the Rust code, revert `recognizer.rs` to use `SileroVadModelConfig` instead of `TenVadModelConfig`.
4. Update `config.rs` to default the VAD model path to `models/vad/silero_vad.onnx`.

## Technical Details

### Code structure (`crates/stt-core/src/recognizer.rs`)

The VAD builder (`build_vad_for_model`) now constructs a `TenVadModelConfig`:

```rust
let ten = TenVadModelConfig {
    model: Some(model.vad_model_path.display().to_string()),
    threshold: model.vad.threshold,
    min_silence_duration: model.vad.min_silence,
    min_speech_duration: model.vad.min_speech,
    max_speech_duration: model.vad.max_speech,
    window_size: 256,
};

let config = VadModelConfig {
    silero_vad: Default::default(),  // unused, but required by the sherpa-onnx API
    ten_vad: ten,
    sample_rate: 16_000,
    num_threads: 1,
    provider: Some("cpu".to_string()),
    debug: false,
};
```

The `silero_vad: Default::default()` field is kept but unused — the `sherpa-onnx` crate's `VadModelConfig` struct requires it, but it is ignored when `ten_vad` is populated.

### Config default (`crates/stt-core/src/config.rs`)

```rust
fn default_vad_profile() -> String {
    "ten".to_string()  // was "custom"
}
```

The resolved VAD model path was updated from `models/vad/silero_vad.onnx` to `models/vad/ten-vad.int8.onnx`.

### Asset lock

The asset lock entry in `baselines/phase0/assets.lock.json` was replaced:

```json
{
  "id": "vad.ten",
  "path": "models/vad/ten-vad.int8.onnx",
  "sha256": "880c072f188efa169ea028b2159d1b3a438e153d080b87eac31b74ecad511e61",
  "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.int8.onnx"
}
```

The old `vad.silero` entry was removed entirely from the lock.
