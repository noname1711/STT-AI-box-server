# CAPU binary-runtime research note

Date: 2026-05-26

## Question

Is there another viable architecture for:

```text
vit-stt binary + external model files
```

while keeping CAPU inference performance and quality comparable to the current
PyTorch baseline?

## Short answer

Yes. The best new candidate is:

```text
Rust vit-stt binary + OpenVINO Runtime + OpenVINO IR model assets
```

This is now more attractive than the existing ONNX Runtime path for CAPU,
because local testing showed:

- OpenVINO can convert the CAPU ONNX neural model to IR.
- OpenVINO CPU raw neural inference is faster than PyTorch eager and ONNX
  Runtime for the sampled input.
- A CAPU harness using OpenVINO for neural inference and the same CAPU
  preprocessing/edit loop preserved locked final-text parity on all current
  clean-boundary snippets.
- OpenVINO exposes C and C++ runtime APIs, and there are Rust bindings.

The second-best candidate is:

```text
Rust vit-stt binary + PyTorch AOTInductor compiled package
```

This keeps the PyTorch compiler stack and produced a loadable non-Python
compiled artifact locally, with close numeric parity and faster raw neural
latency than eager PyTorch. It needs more packaging investigation than
OpenVINO.

## Local tests

Raw outputs:

```text
baselines/spikes/capu_solution_experiments_2026_05_25/
```

Generated model artifacts:

```text
models/capu/generated/
```

### OpenVINO

Installed for the probe:

```text
openvino==2026.1.0
```

Conversion:

```python
model = openvino.convert_model("seq2labels.onnx")
openvino.save_model(model, "seq2labels.xml")
```

Probe output:

```text
baselines/spikes/capu_solution_experiments_2026_05_25/openvino_probe.json
baselines/spikes/capu_solution_experiments_2026_05_25/openvino_capu_clean_harness.json
```

Raw neural sample:

| Runtime | Mean ms | P50 ms | Notes |
| --- | ---: | ---: | --- |
| OpenVINO CPU raw neural | 14.26 | 13.46 | sampled neural call only |
| AOTInductor raw neural | 19.48 | 18.98 | sampled neural call only |
| TorchScript raw neural | 21.62 | 20.58 | sampled neural call only |
| PyTorch eager raw neural | 22-24 | 21-23 | sampled neural call only |

Full CAPU harness on `clean_lower` boundary:

| Runtime | Locked final-text parity | First ms | Warm mean ms | P50 ms |
| --- | :---: | ---: | ---: | ---: |
| OpenVINO neural + CAPU loop | yes | 48.15 | 50.20 | 49.24 |

Important nuance:

- OpenVINO logits are not numerically identical to PyTorch logits
  (`max_abs_diff` around `0.19-0.21` on the sampled neural output).
- Final CAPU text still matched all current locked snippets when the same
  `clean_lower` input boundary was used.
- Therefore OpenVINO needs broader CAPU fixture coverage before replacing
  PyTorch, but it is no longer just theoretical.

### AOTInductor

Probe output:

```text
baselines/spikes/capu_solution_experiments_2026_05_25/aot_inductor_probe.json
baselines/spikes/capu_solution_experiments_2026_05_25/aot_inductor_runtime_probe.json
```

Result:

| Step | Result |
| --- | --- |
| `torch.export.export` | passed |
| AOTInductor compile/package | passed |
| Load compiled package | passed |
| Max abs diff vs eager PyTorch | `1.907e-05` |
| Eager raw neural mean | `22.43 ms` |
| AOTInductor raw neural mean | `19.48 ms` |

Interpretation:

AOTInductor is a strong "PyTorch-quality without Python at runtime" candidate.
It is less mature operationally for this repo than OpenVINO because it needs a
clear Rust/C++ loading story, package lifecycle, and platform matrix.

## Current ranking

| Rank | Architecture | Verdict |
| ---: | --- | --- |
| 1 | Rust + OpenVINO Runtime + OpenVINO IR | Best new candidate for binary + external models + comparable CAPU performance |
| 2 | Rust + AOTInductor package | Best PyTorch-native non-Python candidate; promising but packaging work remains |
| 3 | Rust + libtorch/TorchScript | Exact neural representation, but heavier runtime bundle |
| 4 | Rust + ONNX Runtime ORT format FP32 | Good packaging path, but slower long-form CAPU remains unresolved |
| 5 | Rust + QUInt8 ONNX/ORT | Useful for size, not speed |
| 6 | Rust-native Candle/Burn/tract | Still unproven for this exact CAPU graph and performance target |

## Recommended next spike

Implement a private `stt-capu-openvino` spike behind the existing CAPU boundary:

```text
PostprocessMode::Capu
  -> stt-capu engine = "openvino"
  -> external IR files under models/capu/generated/.../seq2labels.xml + .bin
  -> same Rust CAPU tokenizer/edit loop used by the ONNX spike
```

Success criteria:

1. Final text matches all locked CAPU snippets.
2. Final text matches long-audio-derived CAPU cases.
3. Warm latency is within `1.0x-1.3x` of Python CPU CAPU on snippets.
4. Long-form latency is materially better than Rust ONNX Runtime.
5. Runtime package does not require Python.
6. Model assets remain external and replaceable.

## Sources

- OpenVINO model preparation:
  https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html
- OpenVINO runtime:
  https://docs.openvino.ai/2024/openvino-workflow/running-inference.html
- OpenVINO C API:
  https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__c__api.html
- OpenVINO Apple Silicon support note:
  https://www.intel.com/content/www/us/en/support/articles/000101560.html
- Rust OpenVINO bindings:
  https://github.com/intel/openvino-rs
- PyTorch AOTInductor:
  https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_inductor.html
- PyTorch `torch.export`:
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- ExecuTorch:
  https://docs.pytorch.org/get-started/executorch/
