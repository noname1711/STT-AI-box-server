# CAPU ONNX long-audio performance reassessment

Date: 2026-05-25

## Scope

- reassess why the Rust + ONNX CAPU spike is slower than the Python CAPU worker
- benchmark CAPU on transcripts derived from longer Vietnamese audio assets
- measure slowdown across increasing effective audio lengths
- capture stage-level timing inside the Rust + ONNX path

## Benchmark inputs

Reference audio assets from `/Users/leakless/code/sherpa-onnx-vit/tests/assets`:

- `weanxinviec.mp3` — 30.04s
- `thoitiet936.mp3` — 232.61s
- `bomman.mp3` — 972.49s

Method:

1. Transcribe each full asset once with `gipformer-65M-rnnt` in `vit-stt`
2. Convert the raw ASR output to `clean_lower`
3. Build prefix benchmark cases that approximate increasing audio durations
4. Run CAPU with:
   - Python worker
   - Rust + ONNX Runtime spike
5. Record warm mean/p50/p95 and Rust stage timings

Important note:

- prefix durations are **estimated** from transcript character ratio, not exact VAD-aligned audio cuts

Machine-readable source:

- `baselines/spikes/capu_onnx_assets_report.json`

## Reassessment: why ONNX is slower

The measurements point to one dominant cause on this machine:

- **ONNX Runtime CPU inference time dominates almost the entire Rust pipeline**

Observed pattern from the stage timings:

- tokenization is small but grows with text length
- merge/edit/finalize are negligible
- `ort_ms` accounts for roughly **99%** of Rust warm latency on long cases

Examples:

- `weanxinviec/30s`
  - Rust mean: `278.6 ms`
  - ORT mean: `276.1 ms`
  - tokenizer mean: `2.3 ms`
- `thoitiet936/4m`
  - Rust mean: `4012.2 ms`
  - ORT mean: `3984.8 ms`
  - tokenizer mean: `25.5 ms`
- `bomman/full`
  - Rust mean: `38056.1 ms`
  - ORT mean: `37849.3 ms`
  - tokenizer mean: `187.9 ms`

That means the slowdown is **not primarily from Rust orchestration**. It is overwhelmingly the ONNX Runtime CPU execution path for this exported model.

Most likely contributors:

1. **CPU ORT path on Apple Silicon is weaker than PyTorch for this workload**
2. **Batch size is 1 and the model is invoked many times across chunked iterative passes**
3. **The exported model still uses dynamic-style execution behavior rather than an aggressively offline-optimized static graph**
4. **The Python worker uses the original PyTorch model path, which appears materially faster here for token classification at these sequence lengths**

Secondary contributor:

- The Hugging Face tokenizer cost is real, but small relative to ORT. On the largest case it is ~188 ms vs ~37.8 s in ORT.

## Detailed performance by length

### `weanxinviec.mp3` (~30s audio)

| Case | Estimated audio | Python mean ms | Rust+ONNX mean ms | Slowdown | ORT mean ms | Tokenizer mean ms | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30s | 29.4s | 61.6 | 278.6 | 4.52x | 276.1 | 2.3 | exact |
| 2m/full-equivalent | 30.0s | 68.2 | 336.4 | 4.94x | 333.5 | 2.7 | exact |

### `thoitiet936.mp3` (~3.9m audio)

| Case | Estimated audio | Python mean ms | Rust+ONNX mean ms | Slowdown | ORT mean ms | Tokenizer mean ms | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30s | 29.4s | 78.1 | 472.1 | 6.04x | 468.3 | 3.5 | exact |
| 2m | 119.8s | 245.8 | 2438.2 | 9.92x | 2422.5 | 14.6 | exact |
| 4m/full-equivalent | 232.6s | 379.9 | 4012.2 | 10.56x | 3984.8 | 25.5 | exact |

### `bomman.mp3` (~16.2m audio)

| Case | Estimated audio | Python mean ms | Rust+ONNX mean ms | Slowdown | ORT mean ms | Tokenizer mean ms | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30s | 29.9s | 120.8 | 881.8 | 7.30x | 875.5 | 5.9 | exact |
| 2m | 119.9s | 395.1 | 4161.3 | 10.53x | 4134.8 | 24.7 | exact |
| 4m | 239.8s | 675.7 | 7901.7 | 11.69x | 7852.4 | 45.7 | exact |
| 8m | 479.8s | 1422.1 | 17679.1 | 12.43x | 17579.9 | 91.4 | exact |
| 12m | 719.9s | 2099.8 | 27414.7 | 13.06x | 27260.0 | 141.6 | exact |
| full | 972.5s | 2939.8 | 38056.1 | 12.95x | 37849.3 | 187.9 | drift |

## How much longer does ONNX take?

Compared to the Python CAPU worker, the Rust + ONNX spike was:

- about **4.5x longer** on the shortest 30s-style case
- about **10x longer** around 2–4 minute content
- about **12–13x longer** on the longest `bomman` cases

Absolute gap examples:

- `weanxinviec/30s`: `+217.0 ms`
- `thoitiet936/4m`: `+3632.3 ms`
- `bomman/12m`: `+25314.9 ms`
- `bomman/full`: `+35116.3 ms`

## Parity result

Results:

- exact parity on every measured case **except one**
- one long-case drift at `bomman/full`

Observed drift:

- divergence appears near the very end of the transcript
- first visible difference is around the phrase:
  - Python: `Mấy chị em kiểu gọi là gà không lối thoát...`
  - Rust: `em kiểu gọi là gà không lối thoát...`

Interpretation:

- the long-form path is still very close, but not perfectly stable at the extreme length tested
- likely causes are chunk-merge edge behavior or tokenizer/input-offset behavior at a boundary near the tail

## What changed since the earlier spike

The reassessment added:

- Hugging Face `tokenizers`-based tokenization in Rust
- explicit ORT session settings:
  - `GraphOptimizationLevel::Level3`
  - `with_intra_threads(1)`
  - `with_memory_pattern(false)`
- long-audio-derived prefix benchmarks
- stage timing breakdowns
- progress logging for long runs

These changes improved confidence in the diagnosis, but **did not remove the core performance gap**.

## Conclusion

The main bottleneck is ONNX Runtime CPU execution, not Rust-side glue.

Current status:

- ONNX export: successful
- Rust long-form execution: successful
- Exact parity: strong but not perfect on the longest case
- Performance: significantly slower than the Python worker on this Apple Silicon CPU setup

## Recommendation

Do not replace the Python CAPU worker with the current CPU ONNX path yet.

Recommended next experiments if you want to keep pushing this:

1. export a more aggressively optimized static-shape ONNX variant
2. run ONNX Runtime transformer optimization offline
3. test ORT profiling output on representative medium and long cases
4. investigate CoreML EP or another Apple-optimized runtime path
5. fix the `bomman/full` tail drift before treating the ONNX path as parity-complete

## Hugging Face upload status

Not uploaded yet.

Current blockers:

- local `hf auth` token exists, but no target repo name was provided in the session
- the generated bundle currently includes:
  - converted ONNX artifacts derived from `dragonSwing/vibert-capu`
  - tokenizer/base-model artifacts derived from `FPTAI/vibert-base-cased`
- before upload, the repo target and packaging choice should be explicit

If you want the upload completed next, the safest target is a **private repo** containing only the generated ONNX bundle plus a clear conversion note and source attribution.
