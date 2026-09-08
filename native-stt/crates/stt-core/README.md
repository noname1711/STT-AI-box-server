# stt-core

Core runtime crate for `vit-stt`.

Responsibilities:

- model registry loading
- runtime config loading
- baseline checksum validation
- ffmpeg-based audio decode
- sherpa-onnx offline recognizer setup
- VAD-based segment transcription
- built-in postprocessing: `none`, `clean_lower`, `normalize`

Primary entry point:

- `SttRuntime`
