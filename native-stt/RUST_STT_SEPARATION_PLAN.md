# Fresh Rust STT Repository Plan

## Goal

Create a fresh reusable speech-to-text repository that replaces the current
Python HTTP server shape with a standalone Rust runtime and service.

The target is a reusable Rust STT product that can be consumed in two modes:

- in-process by Rust applications that want one packaged app process
- as a sidecar HTTP service for other languages, remote machines, and
  compatibility with the current OpenAI-style transcription API

CAPU is required behavior, not a stretch goal. This fresh repo should preserve
Vietnamese CAPU postprocessing parity even if the initial implementation keeps
CAPU behind a managed worker/runtime boundary instead of a pure Rust model.

## Why Not Just Migrate The Original Project In Place?

Do not treat the original Python project as the final Rust home unless there is
a strong reason to preserve its current project identity.

Reasons:

1. The original project is shaped around a Python FastAPI server.
   The useful long-term unit is not the server. It is the recognizer runtime,
   model registry, provider setup, postprocessing, and test assets. Migrating
   in place would keep a lot of Python-server naming, layout, and operational
   assumptions around a Rust library that should be cleaner and more reusable.

2. The Rust target has different product boundaries.
   The Rust version should expose a core crate, an HTTP adapter, and a CLI.
   That is a workspace/product design, not a direct port of
   `src/sherpa_onnx_vit`.

3. The original project should stay valuable as a reference implementation
   during the migration.
   It contains known-good behavior, startup probes, model asset download logic,
   postprocessing behavior, CoreML quality notes, CAPU integration lessons, and
   regression tests. Keeping it intact lets the fresh Rust repo compare outputs
   against a working baseline.

4. A clean Rust workspace makes packaging easier.
   A new Rust-first repository can design release artifacts around binaries,
   resource directories, model packs, and platform-specific native libraries
   from the start.

5. In-place migration increases churn before the riskiest question is answered.
   The first risk to retire is whether the official `sherpa-onnx` Rust crate
   can reproduce the expected transcripts for the current Vietnamese and
   English models. That can be answered with a small Rust spike before moving
   any Python project structure.

Recommended decision:

- keep `/Users/leakless/code/sherpa-onnx-vit` as the Python reference and
  asset/test source during the Rust migration
- use this repository, `/Users/leakless/code/vit-stt`, as the fresh Rust
  workspace for the reusable STT runtime
- optionally retire or archive the original Python project after Rust reaches
  feature parity

## Proposed Workspace

Workspace location:

```text
/Users/leakless/code/vit-stt
```

Workspace shape:

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
    README.md
  scripts/
    download-assets
    package-release
```

Crate responsibilities:

| Crate | Responsibility |
| --- | --- |
| `stt-core` | Model registry, sherpa recognizer setup, decode APIs, provider config, postprocess, startup probes, shared errors |
| `stt-capu` | CAPU postprocessing interface, managed worker client, parity tests, CAPU error mapping |
| `stt-http` | OpenAI-compatible `/v1/models` and `/v1/audio/transcriptions` service |
| `stt-cli` | Local probes, file transcription, model listing, benchmark commands |
| `capu-worker` | Initial CAPU runtime implementation if Python/PyTorch remains necessary for parity |

## What Moves From The Original Project

Move concepts, not the Python structure.

From `/Users/leakless/code/sherpa-onnx-vit`:

- model registry fields from `models.local.json` / `models.example.json`
- transducer recognizer setup from `src/sherpa_onnx_vit/services/recognizer.py`
- simple postprocessing and CAPU behavior from `src/sherpa_onnx_vit/services/postprocess.py`
- OpenAI-style HTTP response shape from `src/sherpa_onnx_vit/api/`
- startup probe expectations from `AGENTS.md`
- STT and CAPU asset download knowledge from `scripts/download_assets.py`
- benchmark/probe habit from `scripts/test_decode.py` and benchmark scripts

Do not move initially:

- FastAPI implementation details
- Python queue implementation
- ffmpeg-heavy upload preprocessing beyond the minimum required by `stt-http`
- launchd wrapper scripts

## First Supported Behavior

Start with the smallest standalone behavior while still including required CAPU
parity:

- offline transducer ASR
- 16 kHz mono PCM input
- flushed-segment decoding
- provider `cpu`
- postprocess modes: `none`, `clean_lower`, `normalize`, `capu`
- CAPU postprocessing for Vietnamese
- model list endpoint
- JSON and text transcription responses
- no streaming/WebSocket
- no authentication unless a later deployment requires it

The default Vietnamese model should be:

```text
models/stt/gipformer-65M-rnnt
```

Expected probe WAV:

```text
models/stt/gipformer-65M-rnnt/test_wavs/0.wav
```

Expected raw recognizer output:

```text
RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA
```

Expected current clean-lower HTTP output:

```text
rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia
```

If the actual output differs by model revision, lock the Rust test to the
current local model output and document the exact model asset checksum.

## Public API Strategy

Keep the HTTP API as the primary integration contract.

Reasons:

- other apps can use the STT runtime without Rust integration
- it preserves the current `/v1/audio/transcriptions` contract
- it gives a stable compatibility test for the Rust port

Initial HTTP endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`

Defer:

- WebSocket streaming
- auth
- rate limiting
- TLS

## Packaging Strategy

A single executable is reasonable for code, but the model assets should remain
external files or bundled app resources.

Practical release layout:

```text
AI4Pro or stt-http binary
models/
  stt/
    gipformer-65M-rnnt/
      encoder...
      decoder...
      joiner...
      tokens.txt
      test_wavs/0.wav
  vad/
    ten-vad.int8.onnx
  capu/
    model/
    runtime/
runtime-libs/ or platform-native sherpa libs when needed
```

For macOS app packaging:

```text
AI4Pro.app/
  Contents/
    MacOS/AI4Pro
    Resources/models/...
```

For Linux:

```text
ai4pro/
  AI4Pro
  models/...
  runtime-libs/...
```

Avoid embedding large ONNX files directly into the binary unless there is a
specific deployment reason. File-based models are easier to inspect, replace,
hash, and license-audit.

## Provider Strategy

Phase 1:

- macOS Apple Silicon: `cpu`
- Linux x86_64: `cpu`

Phase 2:

- macOS Apple Silicon: optional `coreml`
- preserve the known `RequireStaticInputShapes=1` quality setting if the Rust
  API exposes provider options or if the native config can be extended

## CAPU Strategy

CAPU is required for `vit-stt`.

CAPU currently depends on Python, PyTorch, Transformers, custom Hugging Face
model code, and device-specific runtime library setup. A pure Rust CAPU port
should not be assumed until proven by a focused spike.

Current production package decision:

- use `leakless/vibert-capu` as the downloadable CAPU package
- expose it locally as model id `vibert-capu`
- keep `models/capu/vibert-capu` self-contained with `base_model/`
- do not replace it with `dragonSwing/xlm-roberta-capu`,
  `welcomyou/vibert-capu-onnx`, or `tourmii/vietnamese-punc-cap-denorm-v1`
  without a new benchmark and baseline update

Recommended CAPU path:

1. define a stable `Postprocessor` interface in `stt-core`
2. implement `none`, `clean_lower`, and `normalize` directly in Rust
3. implement `capu` through `stt-capu`
4. start with a managed local CAPU worker if that is the fastest parity path
5. keep the CAPU worker private to `vit-stt`, not the Python server
6. add parity tests that compare CAPU output against the original project's
   current `CapuPostprocessor`
7. only attempt a pure Rust or ONNX CAPU replacement after parity tests exist

The key boundary is that callers should not know whether CAPU is implemented by
Rust, ONNX Runtime, or a managed Python worker. They should only request
`postprocess_mode=capu` and receive the same response shape.

## Phased Plan

### Phase 0 - Baseline Lock

- record exact current model paths
- record model file checksums
- record exact CAPU model/base-model paths and checksums
- run the Python reference probe from `/Users/leakless/code/sherpa-onnx-vit`
  on `models/stt/gipformer-65M-rnnt/test_wavs/0.wav`
- save expected raw and cleaned output
- save expected CAPU output for the probe transcript and several real snippets

Exit criteria:

- known-good baseline outputs are documented
- STT and CAPU model assets are versioned by manifest/checksum

### Phase 1 - Rust Core Spike

- initialize this `vit-stt` workspace
- add `stt-core`
- add official `sherpa-onnx` Rust crate
- build offline transducer config for `vit_stt_vi_v2`
- decode `test_wavs/0.wav`
- implement `none`, `clean_lower`, and `normalize`
- define the postprocessor trait that CAPU will plug into

Exit criteria:

- `cargo test` passes
- Rust output matches the locked baseline closely enough for the selected
  postprocess mode

### Phase 2 - CAPU Parity

- add `stt-capu`
- add `capu-worker` if the initial implementation needs Python/PyTorch
- load the existing CAPU materialized assets
- support `postprocess_mode=capu` through the same `Postprocessor` trait
- compare output against the original project's Python `CapuPostprocessor`
- make CAPU failures explicit and user-actionable

Exit criteria:

- Rust repo can run ASR plus CAPU on the probe transcript
- CAPU output matches the locked Python baseline for selected snippets
- CAPU can be packaged from `vit-stt` without importing the original project as
  code

### Phase 3 - CLI

- add `stt-cli`
- commands:
  - `list-models`
  - `probe`
  - `transcribe-wav`
  - `postprocess-capu`
  - `benchmark`
- support config file path
- print timing and real-time factor

Exit criteria:

- CLI can replace `scripts/test_decode.py` for ASR probes
- CLI can run CAPU parity checks without starting HTTP

### Phase 4 - HTTP Service

- add `stt-http`
- implement `/health`, `/v1/models`, `/v1/audio/transcriptions`
- accept WAV first
- add text and JSON responses
- preserve OpenAI-style model field validation
- support model-configured `postprocess_mode=capu`

Exit criteria:

- OpenAI-style HTTP transcription works against locked real audio fixtures
- HTTP CAPU output matches the locked Python baseline

### Phase 5 - Standalone Runtime Hardening

- keep recognizers warm across requests
- keep blocking decode/inference work off async HTTP workers
- keep configuration and errors application-neutral
- preserve `stt-core` as the only in-process integration surface

Exit criteria:

- CLI and HTTP both reuse the same stable runtime behavior
- consumers can integrate over HTTP or build their own adapters on `stt-core`
- no application-specific adapter crate is required

### Phase 6 - Packaging

- package `stt-http` standalone
- package CAPU assets/runtime needed by the chosen CAPU implementation
- decide app-resource vs data-dir model placement
- add checksum validation at startup

Exit criteria:

- clean install on macOS Apple Silicon
- no dependency on the original Python project at runtime
- CAPU works from the packaged `vit-stt` layout

### Phase 7 - Accelerators and Advanced Features

- evaluate CoreML quality and speed
- decide whether to replace the managed CAPU worker with a pure Rust or ONNX
  implementation
- consider streaming recognizer only if partial transcripts become a product
  requirement

## Open Questions

- Is the final product allowed to bundle the Vietnamese model commercially?
  The upstream model card lists `cc-by-nc-nd-4.0`; this needs legal or upstream
  permission before commercial redistribution.
- Should model assets live in a shared system directory, app resources, or the
  app data directory?
- Should CAPU ship as a managed Python worker first, or should the project spend
  time up front trying to export/port CAPU away from Python?
- Which platforms are release targets first: macOS only, macOS plus Linux
  x86_64?

## Immediate Next Step

Initialize this `vit-stt` Rust workspace and implement Phase 0 plus Phase 1.

The first milestone is a standalone fresh Rust workspace that decodes the
current local Vietnamese test WAV, defines the postprocessor boundary, and
prepares locked CAPU parity fixtures. After that, implement CAPU before HTTP
and packaging work.
