# CAPU runtime alternatives report

Date: 2026-05-25

## Goal

Find a practical path to reduce or remove the Python CAPU dependency while
preserving the current Vietnamese CAPU behavior and moving toward a standalone
Rust-distributed STT runtime.

This report builds on:

- `baselines/spikes/capu_onnx_tuning_report_2026_05_25.md`
- `baselines/spikes/capu_onnx_tuning_2026_05_25/`

## Executive conclusion

There is no tested pure-Rust or ONNX-only replacement that is ready to become
the default CAPU runtime today.

The best immediate packaging candidate is:

```text
Rust + ONNX Runtime + ORT format model, intra_threads=6
```

It preserves locked snippet parity and improves short-snippet ONNX latency, but
it is still slower than the Python CAPU worker on long text.

The most promising performance candidate is:

```text
PyTorch/TorchScript on Apple MPS
```

It beats CPU PyTorch on warm snippet latency on this Mac, but it is not a
single-binary Rust solution. It implies a private libtorch/runtime bundle, or a
platform-specific helper.

The most interesting compression candidate is:

```text
ONNX dynamic unsigned 8-bit quantization
```

It shrinks the neural model from about `438 MB` to about `110 MB` and preserves
the current locked snippets, but it does not materially improve latency.

## Research notes

### ONNX quantization

ONNX Runtime documents dynamic and static post-training quantization. Dynamic
quantization calculates activation quantization parameters at inference time,
which can preserve accuracy but adds runtime cost. The docs also say dynamic
quantization is generally recommended for transformer-based models, and that
transformer-specific quantization benefits from running transformer optimization
before quantization.

Source: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### ORT format

ORT format is intended for reduced-size ONNX Runtime builds and produces a
required-operators config that can drive a minimal runtime build. The full ORT
runtime can also execute ORT format models.

Source: https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html

### Runtime optimization limits

ORT format is primarily a packaging/minimal-runtime tool. Runtime optimization
support differs from ONNX graph loading, and many optimizations should be baked
in at conversion time.

Source: https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-model-runtime-optimization.html

### Rust-native ML options

- Burn supports ONNX import as a complete model import path.
  Source: https://burn.dev/books/burn/import/index.html
- Candle is a Rust ML framework focused on performance and includes ONNX model
  evaluation support.
  Source: https://github.com/huggingface/candle
- tract is a self-contained Rust TensorFlow/ONNX inference toolkit.
  Source: https://github.com/sonos/tract
- `tch-rs` provides Rust bindings over the PyTorch C++ API/libtorch. It can use
  either a system libtorch, a manually installed libtorch, or a Python PyTorch
  install through `LIBTORCH_USE_PYTORCH=1`.
  Source: https://github.com/LaurentMazare/tch-rs

## Experiments

Raw outputs are under:

```text
baselines/spikes/capu_solution_experiments_2026_05_25/
```

Generated model artifacts are under:

```text
models/capu/generated/
```

These generated model artifacts are intentionally untracked. Do not commit them
without a packaging decision.

## Baseline for comparison

From the previous tuning pass, the best original ONNX setting was:

```text
original dynamic ONNX
ORT intra_threads=6
memory_pattern=false
```

Snippet benchmark:

| Runtime | Exact locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Original ONNX | yes | ~438 MB | 152.60 | 71.98 | 93.92 | 1.30x |

Long-form tuned benchmark from the previous pass:

| Case | Python mean ms | Rust ONNX mean ms | Slowdown | Rust ORT mean ms | Parity |
| --- | ---: | ---: | ---: | ---: | :---: |
| bomman/full | 3628.95 | 11974.53 | 3.30x | 11924.21 | yes |

## Experiment 1: dynamic QInt8 quantization

Command used:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="seq2labels.onnx",
    model_output="seq2labels.onnx",
    weight_type=QuantType.QInt8,
    use_external_data_format=True,
)
```

Result:

- model size: about `110 MB`
- locked parity: failed
- speed on matching subset: faster than original ONNX

Parity drift:

| Fixture | Result |
| --- | --- |
| probe | pass |
| long_vi_segment | fail: `Mười sáu` became `Mười Sáu` |
| legacy_expected_test_transcript | pass |
| model_card_example | pass |

Matching-subset snippet benchmark:

| Runtime | Exact full locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic QInt8 ONNX | no | ~110 MB | 77.06 | 50.47 | 62.69 | 1.24x |

Interpretation:

QInt8 is the fastest ONNX variant tested, but exact CAPU parity fails. It is
not safe as a default replacement unless the project accepts relaxed CAPU
equivalence or retrains/calibrates for parity.

## Experiment 2: dynamic QUInt8 quantization

Command used:

```python
quantize_dynamic(
    model_input="seq2labels.onnx",
    model_output="seq2labels.onnx",
    weight_type=QuantType.QUInt8,
    use_external_data_format=True,
)
```

Result:

- model size: about `110 MB`
- locked parity: passed on all 4 current snippets
- speed: roughly same as original ONNX

Snippet benchmark:

| Runtime | Exact locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic QUInt8 ONNX | yes | ~110 MB | 63.51 | 50.52 | 92.02 | 1.82x |

Interpretation:

This is useful for packaging size, not speed. It needs broader parity coverage
before use because quantization is not lossless.

## Experiment 3: dynamic QInt8 with reduce range

Result:

- model size: about `110 MB`
- locked parity: failed badly

Parity drift:

| Fixture | Result |
| --- | --- |
| probe | fail |
| long_vi_segment | fail |
| legacy_expected_test_transcript | pass |
| model_card_example | fail |

Interpretation:

Not viable.

## Experiment 4: static-shape ONNX exports

Two static variants were exported:

- `static_token_len=128`, `static_word_len=65`
- `static_token_len=64`, `static_word_len=65`

The spike runner was extended to read optional metadata:

```json
{
  "static_token_len": 64,
  "static_word_len": 65
}
```

and pad inputs to those fixed shapes.

Snippet benchmark:

| Runtime | Exact locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Static 64x65 ONNX | yes | ~438 MB | 121.16 | 50.46 | 94.99 | 1.88x |
| Static 128x65 ONNX | yes | ~438 MB | 123.84 | 51.80 | 182.57 | 3.52x |

Interpretation:

Static shape did not help. The 128-token model is much slower because it pads
every request to a larger sequence. The 64-token model is close to dynamic ONNX
but not better, and it is more brittle for longer or more heavily wordpiece-
split chunks.

## Experiment 5: ORT format

Command used:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort \
  --optimization_style Fixed \
  --enable_type_reduction \
  models/capu/generated/dragonSwing--vibert-capu-ort-format
```

The generated `required_operators_and_types.config` can be used later to build
a smaller ONNX Runtime package.

The spike runner was extended to read optional metadata:

```json
{
  "model_file": "seq2labels.ort"
}
```

Snippet benchmark:

| Runtime | Exact locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| ORT format FP32 | yes | ~438 MB | 272.00 | 51.51 | 81.64 | 1.58x |

Smoke long-form benchmark:

| Case | Python mean ms | Rust ORT-format mean ms | Slowdown | Rust ORT stage ms | Parity |
| --- | ---: | ---: | ---: | ---: | :---: |
| weanxinviec/30s | 77.30 | 103.95 | 1.34x | 103.28 | yes |
| thoitiet936/4m | 461.86 | 1286.39 | 2.79x | 1279.95 | yes |
| bomman/full | 3148.57 | 11930.03 | 3.79x | 11884.48 | yes |

Interpretation:

ORT format is a packaging win and a short-snippet latency win, but it does not
fix long-form performance. It should be considered the best ONNX packaging
candidate, not a complete performance solution.

## Experiment 6: QUInt8 plus ORT format

Result:

- model size: about `110 MB`
- locked parity: passed on current snippets
- speed: no improvement over QUInt8 ONNX

Snippet benchmark:

| Runtime | Exact locked parity | Model size | Init ms | Python mean ms | Rust mean ms | Slowdown |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| QUInt8 ORT format | yes | ~110 MB | 90.74 | 50.78 | 92.43 | 1.82x |

Interpretation:

This is the smallest tested exact-parity artifact. It is not a speed win.

## Experiment 7: TorchScript/libtorch feasibility

A TorchScript neural artifact was exported:

```text
models/capu/generated/dragonSwing--vibert-capu-torchscript/seq2labels.torchscript.pt
```

Raw neural-output comparison:

| Metric | Value |
| --- | ---: |
| max absolute difference vs eager PyTorch | 0.0 |
| eager raw neural mean | 23.74 ms |
| TorchScript raw neural mean | 21.62 ms |

Interpretation:

TorchScript can represent the neural CAPU model exactly for the tested input.
This makes `tch-rs`/libtorch plausible if the goal is "remove Python from the
hot path" rather than "avoid PyTorch entirely." It would still require:

- Rust-side CAPU loop using `tch-rs`
- bundled libtorch libraries
- platform-specific packaging
- licensing and size review

It is not a single static binary, but it may preserve PyTorch performance better
than ONNX Runtime.

## Experiment 8: PyTorch MPS

The Python CAPU model was tested directly on CPU and MPS for the
`model_card_example` snippet.

| Device | Init ms | First ms | Warm mean ms | Output parity |
| --- | ---: | ---: | ---: | :---: |
| CPU | 808.09 | 79.84 | 67.98 | yes |
| MPS | 950.21 | 1607.30 | 45.86 | yes |

Interpretation:

MPS is the first tested path that beats CPU PyTorch on warm snippet latency.
The first call is expensive, likely due to graph/kernel compilation. For a
long-lived worker, MPS may be worthwhile on macOS. For a short-lived CLI call,
the first-call penalty is too high.

This is a packaging/runtime strategy, not a Rust-native inference strategy.

## Alternatives not promoted

### CoreML through ONNX Runtime

Already tested in the previous tuning report. The provider covered only part of
the graph and was slower than CPU ORT for the sampled neural call.

### Burn, Candle, tract

These remain research candidates for a pure-Rust runtime, but they were not
promoted in this pass because the current CAPU graph uses BERT-style ops,
dynamic indexing through `input_offsets`, LayerNorm, softmax, and transformer
matmuls. The risk is not "can the framework run some ONNX," but "can it import
and optimize this exact graph with parity and better speed than ORT."

The most reasonable way to test these is a separate spike with an import-only
goal:

1. load `seq2labels.onnx`
2. run the four locked neural inputs
3. compare logits against PyTorch/ORT
4. only then wire the CAPU edit loop

## Ranking

| Rank | Path | Why |
| ---: | --- | --- |
| 1 | Keep Python worker, optionally enable MPS on macOS | Fastest proven warm path, exact parity, lowest implementation risk |
| 2 | Rust + ORT format FP32 | Best exact-parity ONNX packaging path, improves snippet latency, supports ORT minimal-build direction |
| 3 | Rust + QUInt8 ORT/ONNX | Best size reduction with current locked parity, but not a speed win and needs broader parity tests |
| 4 | Rust + libtorch/TorchScript via `tch-rs` | Plausible Python-free hot path with PyTorch-like speed, but packaging is heavier than ORT |
| 5 | QInt8 ONNX | Fastest ONNX variant, but exact parity fails |
| 6 | Static-shape ONNX | Tested and not faster |
| 7 | CoreML EP via ORT | Tested and slower due partial graph partitioning |
| 8 | Burn/Candle/tract | Needs dedicated import/parity spike before performance claims |

## Recommended next implementation plan

### Short term

Keep Python CAPU as production default.

Add a macOS runtime config option for:

```toml
[capu]
device = "mps"
```

but only enable it after adding a warmup call and documenting the first-call
latency. This gives the best local performance improvement without changing
CAPU semantics.

### Medium term

Promote ONNX Runtime as an experimental CAPU engine:

```text
engine = "onnx-ort"
model_file = "seq2labels.ort"
intra_threads = 6
```

Use FP32 ORT format first because it preserved parity and has the best
exact-parity ONNX latency in the short snippet benchmark.

Keep QUInt8 as a packaging-size variant only after broad fixture parity passes.

### Long term

Do not chase "single static binary" until CAPU is no longer a BERT-sized
transformer runtime problem.

A realistic llama.cpp-style CAPU target would require one of:

- a smaller/distilled CAPU model
- quantization-aware training that preserves Vietnamese CAPU behavior
- a custom Rust/native transformer inference path for this architecture
- a libtorch/TorchScript runtime bundle if PyTorch performance remains the
  target

## Validation commands

Representative commands used:

```bash
target/release/capu_onnx_spike \
  --export-dir models/capu/generated/dragonSwing--vibert-capu-ort-runtime \
  --ort-intra-threads 6 \
  benchmark \
  --iterations 20 \
  --benchmark-id model_card_example

target/release/capu_onnx_spike \
  --export-dir models/capu/generated/dragonSwing--vibert-capu-onnx-quant-dynamic-u8 \
  --ort-intra-threads 6 \
  benchmark \
  --iterations 20 \
  --benchmark-id model_card_example

target/release/capu_onnx_spike \
  --export-dir models/capu/generated/dragonSwing--vibert-capu-ort-runtime \
  --ort-intra-threads 6 \
  benchmark-assets \
  --iterations 1 \
  --warmup 0

cargo fmt -p stt-capu --check
cargo check -p stt-capu --features onnx-spike
cargo test -p stt-capu --features onnx-spike
```
