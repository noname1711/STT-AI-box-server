# stt-capu

CAPU postprocessing bridge for `vit-stt`.

Current implementation:

- probes for pure Rust/ONNX feasibility
- uses `leakless/vibert-capu` as the production CAPU package
- records that current supported CAPU assets are PyTorch-only
- runs CAPU through a private Python worker

The local model id is `vibert-capu`, resolved from
`models/capu/vibert-capu`. Candidate replacements benchmarked on
2026-05-27 did not beat this path for production use:

- `dragonSwing/xlm-roberta-capu`
- `welcomyou/vibert-capu-onnx`
- `tourmii/vietnamese-punc-cap-denorm-v1`

See `baselines/spikes/capu_candidate_benchmark_2026_05_27/report.md` before
changing the default CAPU model or runtime.

Experimental spike commands:

- `capu-worker/scripts/export_capu_onnx.py`
- `cargo run -p stt-capu --features onnx-spike --bin capu_onnx_spike -- benchmark --output baselines/spikes/capu_onnx_report.json`

Public Rust callers still see CAPU as a normal postprocessor.
