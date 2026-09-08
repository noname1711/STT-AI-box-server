@/Users/leakless/.codex/RTK.md

# AGENTS.md

Guidance for agents working in this repository.

This repository is the intended Rust STT workspace. The original Python STT
project lives at `/Users/leakless/code/sherpa-onnx-vit` and remains the
reference implementation during migration. Use `RUST_STT_SEPARATION_PLAN.md`
as the source of truth for the Rust separation strategy.

## Working Principles

- State assumptions when the request has multiple plausible interpretations.
- Ask before making broad architectural changes or changing migration
  direction.
- Prefer the smallest change that advances the current phase.
- Do not add speculative abstractions, providers, streaming APIs, packaging
  systems, or integration points before they are required by the plan.
- Touch only the files needed for the task. Do not clean up unrelated code,
  comments, formatting, or dead code.
- Every changed line should trace directly to the user request or a documented
  phase exit criterion.
- For non-trivial work, define the success criteria before implementation and
  verify them before reporting completion.

## Project Role

Current role:

- Build the reusable Rust STT runtime described in the separation plan.
- Preserve parity with the known-good Python implementation in
  `/Users/leakless/code/sherpa-onnx-vit`.
- Carry forward model registry knowledge, postprocessing behavior, CAPU
  behavior, probes, asset download behavior, and compatibility expectations.
- Keep the workspace separate from application-specific integration code.

Target Rust workspace from the plan:

```text
/Users/leakless/code/vit-stt
```

Expected workspace shape:

```text
vit-stt/
  Cargo.toml
  crates/
    stt-core/
    stt-capu/
    stt-http/
    stt-cli/
  capu-worker/
  models/
  scripts/
```

## Migration Boundaries

Move concepts, not the Python server structure.

Good source material from `/Users/leakless/code/sherpa-onnx-vit`:

- model registry fields from `models.local.json` and `models.example.json`
- recognizer setup from `src/sherpa_onnx_vit/services/recognizer.py`
- postprocessing and CAPU behavior from
  `src/sherpa_onnx_vit/services/postprocess.py`
- OpenAI-style HTTP response shape from `src/sherpa_onnx_vit/api/`
- asset download knowledge from `scripts/download_assets.py`
- decode and benchmark habits from `scripts/test_decode.py` and benchmark
  scripts

Do not port by default:

- FastAPI implementation details
- Python queue internals
- launchd wrapper scripts
- broad ffmpeg upload preprocessing beyond what `stt-http` needs
- unrelated operational assumptions from the Python server

## Required Behavior To Preserve

CAPU is required behavior, not a stretch goal. A Rust implementation may keep
CAPU behind a managed worker/runtime boundary at first, but callers should only
see a stable `postprocess_mode=capu` behavior.

First supported Rust behavior:

- offline transducer ASR
- 16 kHz mono PCM input
- flushed-segment decoding
- provider `cpu`
- postprocess modes: `none`, `clean_lower`, `normalize`, `capu`
- Vietnamese CAPU postprocessing parity
- model list endpoint
- JSON and text transcription responses
- no streaming/WebSocket unless a later phase requires it
- no authentication unless a consuming app requires it

Baseline Vietnamese model:

```text
models/stt/gipformer-65M-rnnt
```

Probe WAV (Legacy/Reference):

```text
models/stt/gipformer-65M-rnnt/test_wavs/0.wav
```
(Note: This file is absent in the current local assets copy. Use the locked replacement baseline instead.)

Probe WAV (Locked Replacement Baseline):

```text
baselines/phase0/vi_probe.wav
```

Expected raw recognizer output (Replacement Baseline):

```text
ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP
```

Expected clean-lower HTTP output (Replacement Baseline):

```text
định nghĩa thế nào là ăn mặc đẹp
```

Expected CAPU output (Replacement Baseline):

```text
Định nghĩa thế nào là ăn mặc đẹp?
```

Legacy reference outputs:
- Raw: `RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA`
- Clean-lower: `rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia`
- CAPU: `Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia.`

If model output differs, do not silently adjust expectations. Lock the actual
local model output with asset checksums and document the difference.

## Phase Order

Follow the plan order unless the user explicitly changes priorities:

1. Phase 0 - Baseline Lock [Completed]
2. Phase 1 - Rust Core Spike [Completed]
3. Phase 2 - CAPU Parity [Completed - via private Python worker fallback]
4. Phase 3 - CLI [Completed]
5. Phase 4 - HTTP Service [Completed]
6. Phase 5 - Standalone Runtime Hardening [Current]
7. Phase 6 - Packaging [Completed/Staged - portable releases generated for macOS, Windows, and Linux]
8. Phase 7 - Accelerators and Advanced Features [Next]

Immediate priority:

- Keep `vit-stt` usable as a standalone runtime and OpenAI-compatible HTTP service.
- Package standalone release directories and verify diagnostics like `doctor` and `check-capu` pass on target deployment platforms.
- Configurable model warm-start, admin warmup/status endpoints, and readiness gating on `/health`.

## Rust Workspace Guidance

When working in `/Users/leakless/code/vit-stt`:

- Use a Cargo workspace with small crates matching the planned boundaries.
- Keep `stt-core` independent of HTTP, CLI, and application adapters.
- Put model registry, recognizer setup, provider config, decode APIs,
  postprocessing interfaces, startup probes, and shared errors in `stt-core`.
- Put CAPU worker/client behavior and CAPU parity tests behind `stt-capu`.
- Put OpenAI-compatible endpoints only in `stt-http`.
- Put probe, transcription, model listing, CAPU postprocess, and benchmark
  commands only in `stt-cli`.
- Keep model assets external to binaries unless there is a documented packaging
  reason to embed them.
- Prefer explicit errors with user-actionable messages for missing model files,
  missing CAPU runtime, provider failures, or checksum mismatches.

## CAPU Guidance

- The current production CAPU package is `leakless/vibert-capu`, exposed in the
  registry as local model id `vibert-capu` at `models/capu/vibert-capu`.
- Do not switch the default CAPU package to `dragonSwing/xlm-roberta-capu`,
  `welcomyou/vibert-capu-onnx`, `tourmii/vietnamese-punc-cap-denorm-v1`, or any
  other candidate without updating locked baselines and recording a new
  benchmark. The 2026-05-27 candidate report rejected those three as default
  replacements.
- Do not assume CAPU can be ported to pure Rust until a focused spike proves it.
- Define the postprocessor boundary before wiring CAPU implementation details.
- Keep CAPU implementation details hidden behind a stable Rust interface.
- Compare CAPU output against the current Python `CapuPostprocessor` in
  `/Users/leakless/code/sherpa-onnx-vit`.
- Treat CAPU failures as explicit runtime/configuration failures, not silent
  fallback to weaker postprocessing.
- Do not import `/Users/leakless/code/sherpa-onnx-vit` as runtime code from
  this Rust workspace. Use fixtures, manifests, and parity tests instead.

## Commands

For this Rust workspace:

- `cargo fmt`
- `cargo clippy --all-targets --all-features`
- `cargo test`
- targeted CLI probes once `stt-cli` exists

For the original Python reference project, inspect its existing scripts before
adding new ones. Typical validation may include existing probes, tests, or asset
scripts when present.

## Validation Expectations

Before reporting completion, run the narrowest meaningful checks for the change.

For baseline work:

- record exact model paths
- record model file checksums
- record CAPU model/base-model paths and checksums
- capture raw, cleaned, and CAPU outputs for agreed fixtures

For Rust core work:

- verify the probe WAV decodes
- verify postprocess modes covered by the task
- verify tests pass for changed crates

For HTTP compatibility:

- verify `GET /health` returns `{"status":"ok","ready":true}` after warmup
- verify `GET /v1/models`
- verify `POST /v1/audio/transcriptions`
- verify JSON and text response shapes
- verify compatibility with the current OpenAI-style transcription contract

For warmup work:

- verify `warm_start_models` config is parsed correctly
- verify warm-start logs appear on server startup
- verify `POST /admin/warmup` returns timing and probe result
- verify `GET /admin/models/status` returns loaded state
- verify admin endpoints are reachable on localhost

If a check cannot be run because assets, dependencies, or hardware are missing,
state exactly what was skipped and why.

## Licensing And Assets

- Do not commit large model artifacts unless the user explicitly asks.
- Do not assume commercial redistribution is allowed. The plan notes the
  upstream Vietnamese model card may be `cc-by-nc-nd-4.0`; preserve that as an
  open legal/permission question.
- Prefer manifests, checksums, download scripts, and documentation over copied
  binaries.
- Keep model files inspectable and replaceable in `models/` or app resources.

## Documentation

- Keep documentation close to the phase being implemented.
- Update `RUST_STT_SEPARATION_PLAN.md` only when the plan itself changes.
- Document deviations from the plan with the reason, tradeoff, and verification
  impact.
- Keep generated or bulky output out of docs unless it is a required baseline
  fixture.
