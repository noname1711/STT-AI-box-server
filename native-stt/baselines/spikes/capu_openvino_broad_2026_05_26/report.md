# CAPU Binary Runtime Broad Benchmark - OpenVINO Candidate

Date: 2026-05-26  
Workspace: `/Users/leakless/code/vit-stt`  
Reference CAPU project: `/Users/leakless/code/sherpa-onnx-vit`

## Goal

Evaluate whether `vit-stt` can ship as a binary-style runtime with external model files while preserving CAPU output quality and staying comparable to the PyTorch baseline.

The candidate tested here is:

- Rust/application binary boundary: viable through OpenVINO C++/C/Python runtime APIs.
- External neural model artifact: OpenVINO IR, `seq2labels.xml` + `seq2labels.bin`.
- CAPU logic: Python `GecBERTModel` preprocessing and edit loop retained for this benchmark; only the neural `predict()` path is replaced by OpenVINO.

This benchmark is intentionally conservative for parity: exact string equality against live Python CAPU output is treated as the pass/fail criterion, with an additional normalized comparison reported only for diagnosis.

## External Runtime Notes

- OpenVINO Runtime is a C++ runtime with C and Python bindings and supports IR model deployment across CPU and other devices. This makes it a plausible fit for a binarized `vit-stt` runtime with external model assets. Source: [OpenVINO running inference docs](https://docs.openvino.ai/2026/openvino-workflow/running-inference.html).
- ONNX Runtime supports graph optimization, offline/online optimization, and thread/runtime tuning. The previous ONNX/Rust spike already used these tuning paths, but this model remained slower than PyTorch/OpenVINO in practical CAPU tests. Sources: [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
- PyTorch AOTInductor can export/compile `torch.export` programs into packaged artifacts, but the deployment story still keeps the PyTorch/Inductor runtime in the stack. Source: [PyTorch AOTInductor docs](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html).

## Artifacts

- Benchmark harness: `/Users/leakless/code/vit-stt/capu-worker/scripts/benchmark_openvino_capu.py`
- Primary full JSON report: `/Users/leakless/code/vit-stt/baselines/spikes/capu_openvino_broad_2026_05_26/full_report_i5_w2.json`
- Initial full JSON report: `/Users/leakless/code/vit-stt/baselines/spikes/capu_openvino_broad_2026_05_26/full_report.json`
- Quick JSON report: `/Users/leakless/code/vit-stt/baselines/spikes/capu_openvino_broad_2026_05_26/quick_report.json`
- Primary mismatch summary: `/Users/leakless/code/vit-stt/baselines/spikes/capu_openvino_broad_2026_05_26/mismatch_summary_i5_w2.json`
- Initial mismatch summary: `/Users/leakless/code/vit-stt/baselines/spikes/capu_openvino_broad_2026_05_26/mismatch_summary.json`
- OpenVINO IR: `/Users/leakless/code/vit-stt/models/capu/generated/dragonSwing--vibert-capu-openvino-ir/seq2labels.xml`
- OpenVINO weights: `/Users/leakless/code/vit-stt/models/capu/generated/dragonSwing--vibert-capu-openvino-ir/seq2labels.bin`
- Source ONNX: `/Users/leakless/code/vit-stt/models/capu/generated/dragonSwing--vibert-capu-onnx/seq2labels.onnx`

## Commands Run

```bash
capu-worker/.venv/bin/python capu-worker/scripts/benchmark_openvino_capu.py \
  --iterations 3 \
  --warmup 1 \
  --quick \
  --output baselines/spikes/capu_openvino_broad_2026_05_26/quick_report.json
```

```bash
capu-worker/.venv/bin/python capu-worker/scripts/benchmark_openvino_capu.py \
  --iterations 3 \
  --warmup 1 \
  --output baselines/spikes/capu_openvino_broad_2026_05_26/full_report.json
```

```bash
capu-worker/.venv/bin/python capu-worker/scripts/benchmark_openvino_capu.py \
  --iterations 5 \
  --warmup 2 \
  --output baselines/spikes/capu_openvino_broad_2026_05_26/full_report_i5_w2.json
```

```bash
capu-worker/.venv/bin/python -m py_compile capu-worker/scripts/benchmark_openvino_capu.py
```

## Environment

- OpenVINO: `2026.1.0-21367-63e31528c62-releases/2026/1`
- PyTorch: `2.12.0`
- Primary full benchmark warmup: 2
- Primary full benchmark measured iterations: 5
- Python CAPU init: 1153.44 ms
- OpenVINO-backed CAPU model init: 790.94 ms
- OpenVINO compile: 257.34 ms

## Feature Parity Result

| Scope | Cases | Exact Python parity | Expected fixture parity | Notes |
| --- | ---: | ---: | ---: | --- |
| Locked Phase 0 CAPU snippets | 4 | 4 / 4 | 4 / 4 | Exact match on all locked fixtures. |
| Asset prefix / long-form cases | 11 | 5 / 11 | 5 / 11 | Six exact mismatches, all punctuation/case-only under normalized comparison. |
| Full suite | 15 | 9 / 15 | 9 / 15 | Exact CAPU parity is not proven yet. |

The six mismatches all have `normalized_equal=true` after lowercasing and punctuation removal. Their similarity ratios are very high:

| Case | Similarity | Normalized equal | First observed drift |
| --- | ---: | --- | --- |
| `thoitiet936/30s` | 0.998267 | yes | Python: `tố, lốc`; OpenVINO: `tố lốc` |
| `bomman/2m` | 0.998476 | yes | Python: `tiền đô. Nết`; OpenVINO: `tiền đô, nết` |
| `bomman/4m` | 0.998989 | yes | Same punctuation/case boundary drift |
| `bomman/8m` | 0.998995 | yes | Same punctuation/case boundary drift |
| `bomman/12m` | 0.998829 | yes | Same punctuation/case boundary drift |
| `bomman/full` | 0.999104 | yes | Same punctuation/case boundary drift |

Interpretation: OpenVINO preserves lexical content on these mismatches, but it does not currently preserve exact CAPU punctuation/capitalization behavior on longer inputs.

## Timing Results

`slowdown` is `OpenVINO mean / Python mean`; values below `1.0` mean OpenVINO is faster.

| Group | Case | Chars | Python mean ms | OpenVINO mean ms | Slowdown | Exact match |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| locked | `probe` | 32 | 37.90 | 13.63 | 0.36x | yes |
| locked | `long_vi_segment` | 127 | 56.38 | 36.37 | 0.65x | yes |
| locked | `legacy_expected_test_transcript` | 52 | 30.72 | 22.59 | 0.74x | yes |
| locked | `model_card_example` | 200 | 64.85 | 47.41 | 0.73x | yes |
| asset | `weanxinviec/30s` | 197 | 71.33 | 51.61 | 0.72x | yes |
| asset | `weanxinviec/2m` | 201 | 79.24 | 57.99 | 0.73x | yes |
| asset | `thoitiet936/30s` | 281 | 89.48 | 75.08 | 0.84x | no |
| asset | `thoitiet936/2m` | 1143 | 274.07 | 286.74 | 1.05x | yes |
| asset | `thoitiet936/4m` | 2220 | 421.66 | 448.92 | 1.06x | yes |
| asset | `bomman/30s` | 485 | 136.73 | 124.79 | 0.91x | yes |
| asset | `bomman/2m` | 1947 | 451.33 | 474.19 | 1.05x | no |
| asset | `bomman/4m` | 3895 | 794.68 | 896.31 | 1.13x | no |
| asset | `bomman/8m` | 7792 | 1697.14 | 1887.40 | 1.11x | no |
| asset | `bomman/12m` | 11691 | 2550.34 | 2842.95 | 1.11x | no |
| asset | `bomman/full` | 15794 | 3383.35 | 3971.72 | 1.17x | no |

## Performance Read

OpenVINO is a strong small/medium text candidate:

- 1.4x to 2.4x faster than Python on the locked Phase 0 snippets.
- Faster on several short asset cases.
- Lower model init time than live Python CAPU, even before amortizing compile.

OpenVINO is not clearly faster on long form:

- Around parity on `thoitiet936/4m` and `bomman/30s`.
- 5% to 17% slower on the longer `bomman` cases in the primary run.
- About 5% slower on `thoitiet936/2m`.

This likely means neural inference alone is not the only bottleneck for long-form CAPU. The full CAPU flow repeatedly chunks text, runs model predictions, maps labels back to token edits, and applies punctuation/case edits. Once the input grows, loop structure, batching behavior, tensor conversion, and edit application become meaningful. A production Rust/OpenVINO implementation would need to port or reimplement that surrounding CAPU loop carefully, not only call the IR model.

## Why Rust + ONNX Was Slower Than Python

The benchmark history points to runtime/backend behavior, not Rust language overhead:

- The PyTorch baseline is already a compiled native stack below Python. The CAPU model spends most of its time in optimized tensor kernels, not in Python bytecode.
- The ONNX/Rust path crossed into ONNX Runtime kernels, graph scheduling, allocation, and thread pools for relatively small transformer-style inputs. For this CAPU workload, those costs were not amortized well.
- Thread tuning helped but did not close the gap. The best prior ONNX Runtime tuning used 6 intra-op threads with memory pattern disabled, but long-form CAPU still lagged.
- The ONNX export may not map this Vietnamese BERT sequence-labeling graph to the same CPU kernels and fusion choices as PyTorch or OpenVINO.

The practical takeaway is: moving the host application from Python to Rust does not automatically make inference faster. The runtime backend and graph lowering path dominate.

## Alternative Assessment

| Option | Binary app fit | External models | Quality/parity | Performance | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Keep Python CAPU sidecar | Medium | yes | best; current baseline | best long-form baseline | Use as default until replacement is exact. |
| OpenVINO IR + Rust/C++ runtime | strong | yes | near-exact; exact fails on six long cases | best short/medium; mixed long-form | Best candidate for experimental binary CAPU. |
| PyTorch AOTInductor package | medium | yes | raw neural diff previously very small | modest gain in raw neural probe | Keep as fallback research path, not the cleanest Rust binary path. |
| ONNX Runtime from Rust | strong | yes | usable but prior spike slower | slower than desired | Do not choose as default CAPU backend. |
| Pure Rust CAPU port with Candle/tract | strong | yes | unknown | unknown | Too much model/runtime risk for current phase. |
| CoreML | Apple-only | yes | untested | potentially good on Apple hardware | Not a general `vit-stt` default. |

## Decision

Do not replace the PyTorch CAPU baseline with OpenVINO yet if the requirement is exact CAPU output parity.

OpenVINO should remain the leading binary-runtime candidate because it satisfies the deployment shape better than PyTorch sidecar and performs well on short/medium snippets. However, the current data supports only this narrower claim:

> OpenVINO can provide external-model binary deployment with comparable lexical quality, but exact punctuation/case parity and long-form performance parity are not proven.

## Recommended Next Step

Add OpenVINO as an explicit experimental backend boundary, not as the default:

- `capu_backend=python` remains the production parity path.
- `capu_backend=openvino-experimental` is allowed for benchmarking and opt-in local runs.
- Parity tests should record both exact equality and normalized lexical equality.
- Promotion criteria should require exact parity on the locked snippets plus a broader agreed long-form suite, or an explicit product decision that punctuation/case-only drift is acceptable.

Before production promotion, the OpenVINO path needs one more focused spike:

1. Determine whether the punctuation/case drift comes from float precision, label tie-breaking, token chunk boundary behavior, or edit application differences.
2. Benchmark a non-Python CAPU loop so the timing reflects the actual Rust/C++ deployment target instead of a monkeypatched Python loop.
3. Expand long-form fixtures from real STT transcripts and preserve exact PyTorch outputs as locked baselines.
