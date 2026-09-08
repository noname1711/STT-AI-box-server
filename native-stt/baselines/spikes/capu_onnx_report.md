# CAPU ONNX + Rust spike report

Date: 2026-05-25

## Scope

- export the current CAPU `Seq2LabelsModel` to ONNX
- run CAPU inference from Rust without the Python worker in the hot path
- compare final text output against the existing Python CAPU worker
- benchmark Python worker vs Rust + ONNX Runtime on the locked CAPU fixtures

## What was built

- Python exporter: `capu-worker/scripts/export_capu_onnx.py`
- Rust spike binary: `crates/stt-capu/src/bin/capu_onnx_spike.rs`
- Generated ONNX bundle: `models/capu/generated/dragonSwing--vibert-capu-onnx/`
- Machine-readable benchmark report: `baselines/spikes/capu_onnx_report.json`

## Export result

The custom CAPU `Seq2LabelsModel` exported successfully to ONNX.

Generated assets:

- `seq2labels.onnx`
- `export-metadata.json`
- `labels.txt`
- `d_tags.txt`
- `verb-form-vocab.txt`
- `vocab.txt`
- `tokenizer.json`

Export notes:

- exported opset: 18
- exported graph scope: neural model only (`input_ids`, `attention_mask`, `input_offsets` → `logits`, `detect_logits`)
- the iterative text-correction loop, chunk split/merge logic, and edit application were reimplemented in Rust for the spike

## Parity result

Exact final-text parity passed on all locked fixtures in `baselines/phase0/capu_snippets.jsonl`.

Fixtures checked:

1. `probe`
2. `long_vi_segment`
3. `legacy_expected_test_transcript`
4. `model_card_example`

Extra long-text chunking check also matched exactly between:

- `cargo run -p stt-cli -- postprocess-capu ...`
- `cargo run -p stt-capu --features onnx-spike --bin capu_onnx_spike -- run ...`

## Benchmark summary

Benchmark target: `model_card_example`

### Initialization

| Engine | Init ms |
| --- | ---: |
| Python CAPU worker | 1298.92 |
| Rust + ONNX Runtime | 627.39 |

### First inference

| Engine | First inference ms |
| --- | ---: |
| Python CAPU worker | 61.68 |
| Rust + ONNX Runtime | 111.65 |

### Warm latency over 100 runs

| Engine | Mean ms | P50 ms | P95 ms | Min ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python CAPU worker | 56.60 | 55.10 | 65.17 | 50.78 | 84.50 |
| Rust + ONNX Runtime | 96.59 | 96.53 | 98.85 | 93.47 | 103.44 |

## Interpretation

- The ONNX export is viable for this CAPU model.
- The Rust spike reproduced the current Python worker outputs exactly on the locked fixtures.
- Startup time improved materially in the Rust + ONNX path.
- Per-request latency was slower than the current Python worker on this machine for the tested text workload.

## Important limitation

This is a **Rust-hosted ONNX Runtime path**, not a pure-Rust neural inference engine.

It removes the Python worker from inference-time execution, but still depends on ONNX Runtime.

## Recommendation

- Keep the Python worker as the default production CAPU path for now.
- Treat the ONNX + Rust result as a validated spike proving that:
  - export is possible
  - Rust-side parity is possible
  - future integration work is reasonable
- Only promote the ONNX path after broader fixture coverage, packaging validation, and performance investigation.

## Validation commands used

```bash
capu-worker/.venv/bin/python capu-worker/scripts/export_capu_onnx.py
cargo run -p stt-capu --features onnx-spike --bin capu_onnx_spike -- run "rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia"
cargo run -p stt-capu --features onnx-spike --bin capu_onnx_spike -- benchmark --output baselines/spikes/capu_onnx_report.json
cargo run -p stt-cli -- postprocess-capu "...long text..."
cargo test -p stt-capu
```
