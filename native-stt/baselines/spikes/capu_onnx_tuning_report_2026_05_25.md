# CAPU ONNX Runtime tuning report

Date: 2026-05-25

## Scope

This pass investigated why the Rust-hosted ONNX CAPU path is slower than the
Python CAPU worker, with a focus on the knobs that were recommended after the
first spike:

- ONNX Runtime intra-op thread count
- ONNX Runtime memory pattern setting
- offline ONNX Runtime transformer optimization
- CoreML Execution Provider viability on Apple Silicon
- long-audio-derived CAPU behavior with the best CPU setting

The benchmark target was still CAPU postprocessing parity, not ASR decode
performance.

## Machine and runtime

- Host: Apple M5, macOS Darwin 25.5.0, arm64
- CPU topology reported locally: 10 CPUs, 4 `perflevel0` physical CPUs, 6
  `perflevel1` physical CPUs
- Python: 3.11.15
- PyTorch: 2.12.0
- PyTorch threads: 4 intra-op, 10 inter-op
- Rust benchmark binary: `target/release/capu_onnx_spike`
- Rust ONNX crate: `ort = 2.0.0-rc.12`
- Python ONNX Runtime installed for provider/optimizer probing:
  `onnxruntime==1.26.0`
- Python ONNX Runtime providers:
  `CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider`

## Raw artifacts

- Thread/memory matrix:
  `baselines/spikes/capu_onnx_tuning_2026_05_25/snippet_*.json`
- Tuned long-audio-derived benchmark:
  `baselines/spikes/capu_onnx_tuning_2026_05_25/assets_threads6_memfalse.json`
- Provider probe:
  `baselines/spikes/capu_onnx_tuning_2026_05_25/provider_probe.jsonl`
- Optimized ONNX artifact:
  `models/capu/generated/dragonSwing--vibert-capu-onnx-optimized/`

The generated ONNX directories are model artifacts and should not be committed
without an explicit packaging decision.

## Code change for this pass

The spike binary now accepts and records:

```bash
--ort-intra-threads <N>
--ort-memory-pattern
```

This only affects the `onnx-spike` benchmark binary. It does not change the
production Python-worker CAPU path.

## Snippet benchmark matrix

Benchmark:

```bash
target/release/capu_onnx_spike \
  --ort-intra-threads <N> \
  [--ort-memory-pattern] \
  benchmark \
  --iterations 20 \
  --benchmark-id model_card_example
```

All matrix runs passed locked snippet parity.

| ORT intra threads | Memory pattern | Python mean ms | Rust ONNX mean ms | Slowdown |
| ---: | :---: | ---: | ---: | ---: |
| 0 | false | 72.32 | 107.08 | 1.48x |
| 0 | true | 75.12 | 109.59 | 1.46x |
| 1 | false | 70.20 | 288.28 | 4.11x |
| 1 | true | 68.08 | 297.45 | 4.37x |
| 2 | false | 71.16 | 169.75 | 2.39x |
| 2 | true | 70.58 | 168.81 | 2.39x |
| 4 | false | 71.19 | 118.55 | 1.67x |
| 4 | true | 70.77 | 115.29 | 1.63x |
| 6 | false | 71.98 | 93.92 | 1.30x |
| 6 | true | 70.88 | 94.09 | 1.33x |
| 10 | false | 72.94 | 115.31 | 1.58x |
| 10 | true | 71.34 | 114.25 | 1.60x |

### Matrix interpretation

The previous spike used `with_intra_threads(1)`, which is the worst meaningful
setting in this matrix. Moving to 6 intra-op threads reduces snippet latency
from `288.28 ms` to `93.92 ms`, a `3.07x` improvement on the Rust ONNX side.

Memory pattern did not materially help. The best measured setting was:

```text
ORT intra threads = 6
memory pattern = false
```

The Rust ONNX path is still slower than the Python worker on the snippet, but
the gap is much smaller: `1.30x` instead of `4.11x`.

## Long-audio-derived benchmark with best setting

Benchmark:

```bash
target/release/capu_onnx_spike \
  --ort-intra-threads 6 \
  benchmark-assets \
  --iterations 3 \
  --warmup 1 \
  --output baselines/spikes/capu_onnx_tuning_2026_05_25/assets_threads6_memfalse.json
```

Prefix durations are estimated from transcript character ratio, not exact
VAD-aligned audio cuts.

| Case | Est. audio s | Python mean ms | Rust ONNX mean ms | Slowdown | ORT mean ms | Tokenizer mean ms | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| weanxinviec/30s | 29.4 | 89.39 | 103.17 | 1.15x | 102.54 | 0.51 | yes |
| weanxinviec/2m | 30.0 | 86.78 | 123.63 | 1.42x | 122.88 | 0.62 | yes |
| thoitiet936/30s | 29.4 | 95.07 | 171.38 | 1.80x | 170.45 | 0.79 | yes |
| thoitiet936/2m | 119.8 | 308.49 | 809.92 | 2.63x | 806.00 | 3.40 | yes |
| thoitiet936/4m | 232.6 | 501.18 | 1317.67 | 2.63x | 1310.82 | 5.82 | yes |
| bomman/30s | 29.9 | 146.35 | 304.05 | 2.08x | 302.26 | 1.53 | yes |
| bomman/2m | 119.9 | 508.51 | 1398.74 | 2.75x | 1391.56 | 6.14 | yes |
| bomman/4m | 239.8 | 864.12 | 2585.75 | 2.99x | 2573.70 | 9.97 | yes |
| bomman/8m | 479.8 | 1866.75 | 5692.64 | 3.05x | 5667.89 | 19.18 | yes |
| bomman/12m | 719.9 | 2731.51 | 8657.06 | 3.17x | 8619.44 | 28.77 | yes |
| bomman/full | 972.5 | 3628.95 | 11974.53 | 3.30x | 11924.21 | 37.18 | yes |

### Long-form interpretation

The tuned setting is a large improvement over the old `intra_threads=1` path,
but ONNX Runtime still dominates long-form latency. In the full `bomman` case,
`11924.21 ms` of `11974.53 ms` is inside ORT.

This confirms the previous diagnosis with better tuning:

- Rust glue is not the bottleneck.
- Tokenization is not the bottleneck.
- edit application, merge, and finalization are not the bottleneck.
- ONNX Runtime CPU execution is still the bottleneck.

## Offline ONNX transformer optimizer

Command:

```bash
capu-worker/.venv/bin/python -m onnxruntime.transformers.optimizer \
  --input models/capu/generated/dragonSwing--vibert-capu-onnx/seq2labels.onnx \
  --output models/capu/generated/dragonSwing--vibert-capu-onnx-optimized/seq2labels.onnx \
  --model_type bert \
  --num_heads 12 \
  --hidden_size 768 \
  --opt_level 1
```

Optimizer result:

- fused `BiasGelu`: 12
- fused `Attention`: 0
- fused `EmbedLayerNormalization`: 0
- fused `SkipLayerNormalization`: 0
- optimizer warning: attention was not fully optimized

Snippet benchmark with optimized artifact:

| Model | ORT threads | Python mean ms | Rust ONNX mean ms | Slowdown |
| --- | ---: | ---: | ---: | ---: |
| original ONNX | 6 | 71.98 | 93.92 | 1.30x |
| optimized ONNX | 6 | 69.30 | 92.39 | 1.33x |

The optimized graph is only marginally faster. It does not change the main
conclusion because attention fusion did not happen.

## CoreML provider probe

This was a narrow Python ONNX Runtime probe against the neural model only, not
the full Rust CAPU pipeline.

Input shape:

```text
input_ids:      [1, 54]
attention_mask: [1, 54]
input_offsets:  [1, 42]
```

| Provider request | Actual providers | Init ms | First ms | Warm mean ms |
| --- | --- | ---: | ---: | ---: |
| CPU | CPU | 127.31 | 23.54 | 19.92 |
| CoreML then CPU | CoreML, CPU | 966.79 | 205.30 | 53.68 |

CoreML provider warning:

```text
number of partitions supported by CoreML: 98
number of nodes in the graph: 441
number of nodes supported by CoreML: 237
```

CoreML is slower in this probe because the graph is only partially covered and
falls back across providers. It is not currently a promising path for this
exported graph.

## Why Python is still faster

The biggest correction from this pass is that the previous Rust benchmark was
handicapped by `intra_threads=1`. After fixing that, Rust ONNX is no longer
catastrophically slower on short snippets.

Python is still faster because:

1. PyTorch CPU is faster than ORT CPU for this exported ViBERT/CAPU workload on
   this Apple Silicon machine.
2. The ONNX graph remains dynamic and attention fusion did not happen.
3. CAPU invokes the neural model repeatedly across chunks and iterative passes,
   so every ORT inefficiency compounds on long text.
4. CoreML does not cover enough of this graph to beat CPU ORT.

## Recommendation

Do not promote the current ONNX Runtime CAPU path as the default yet.

Do promote the tuning fix into any future ONNX CAPU implementation:

```text
default ORT intra threads should not be 1
current best local setting: intra_threads = 6, memory_pattern = false
```

For packaging, the practical near-term route remains:

1. ship the Rust STT binary
2. bundle CAPU as a private runtime pack or optional runtime pack
3. keep the Python worker as the production CAPU path
4. keep the ONNX path as an experimental packaged alternative

For a future llama.cpp-style CAPU replacement, ONNX Runtime alone is probably
not enough. The next useful spike should target one of:

- static-shape exports for CAPU chunk sizes
- a graph/export rewrite that enables attention fusion
- quantized ONNX if parity holds
- a non-ORT native runtime strategy
- a smaller/distilled CAPU model specifically trained for this postprocessing
  task

Relevant ONNX Runtime documentation:

- Threading: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- Graph optimizations: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- Transformer optimizer: https://onnxruntime.ai/docs/performance/transformers-optimization.html
- CoreML Execution Provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
