# llama.cpp-like Distribution Smoke Test

Date: 2026-05-26  
Workspace: `/Users/leakless/code/vit-stt`

## Goal

Test whether `vit-stt` can be distributed like `llama.cpp`: platform-specific binaries plus external model files, without depending on the development checkout or the original Python reference repo at runtime.

This test validates the current practical shape:

```text
vit-stt/
  bin/
    stt-cli
    stt-http
  config/
    runtime.toml
    models.local.json
  baselines/
    phase0/
  runtime/
    capu/
      vit-stt-capu-worker-full-numpy/
  models/                 # either included, or mounted/copied externally
```

This is not a single static binary. It is a portable runtime directory with external model assets, which is the realistic equivalent of the `llama.cpp` deployment model for this repo while CAPU still depends on Python/PyTorch.

## Code Changes Made

### Optional packaged CAPU worker

`stt-capu` now supports an optional `capu.worker_exe` config field.

- If `worker_exe` is set, Rust launches that executable directly.
- If `worker_exe` is absent, Rust keeps the existing behavior: launch `python -m capu_worker`.

This preserves development behavior and allows release bundles to use a PyInstaller CAPU helper.

### Package script improvements

`scripts/package-release` now:

- stages the PyInstaller CAPU worker into `runtime/capu/` when present
- injects `worker_exe` into the staged `config/runtime.toml`
- copies only required CAPU model files under `models/capu/model`
- skips experimental generated CAPU artifacts under `models/capu/generated`
- supports code/runtime-only packages with:

```bash
VIT_STT_INCLUDE_MODELS=0 scripts/package-release <out-dir>
```

## Build

```bash
cargo build --release -p stt-cli -p stt-http
```

Result:

```text
Finished `release` profile [optimized]
```

## Test 1 - Portable Bundle With Models Included

Staged and copied outside the repo:

```bash
scripts/package-release release/vit-stt-portable-trimmed
cp -R release/vit-stt-portable-trimmed /tmp/vit-stt-portable-trimmed
```

Size:

```text
/tmp/vit-stt-portable-trimmed  2.8G
```

The size includes:

- Rust binaries
- config and baselines
- packaged CAPU worker, about 604 MB
- required STT/VAD/CAPU model files

It excludes experimental generated CAPU artifacts.

### CLI list models

```bash
/tmp/vit-stt-portable-trimmed/bin/stt-cli \
  --workspace-root /tmp/vit-stt-portable-trimmed \
  list-models
```

Result: passed. Returned all four configured models.

### CLI STT probe

```bash
/tmp/vit-stt-portable-trimmed/bin/stt-cli \
  --workspace-root /tmp/vit-stt-portable-trimmed \
  probe --model gipformer-65M-rnnt
```

Result:

```json
{
  "actual": "ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP",
  "expected": "ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP",
  "matches_expected": true,
  "model_id": "gipformer-65M-rnnt"
}
```

### CLI CAPU postprocess through packaged worker

```bash
/tmp/vit-stt-portable-trimmed/bin/stt-cli \
  --workspace-root /tmp/vit-stt-portable-trimmed \
  postprocess-capu 'rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia'
```

Result:

```text
Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia.
```

## Test 2 - HTTP Server From Portable Bundle

Started from the staged bundle directory:

```bash
env VIT_STT_SERVER_PORT=18082 /tmp/vit-stt-portable-trimmed/bin/stt-http
```

Important detail: `stt-http` currently uses the current directory as runtime root, so the server must be started with `cwd` set to the unpacked bundle root.

### Health

```bash
curl -sS --max-time 2 http://127.0.0.1:18082/health
```

Result:

```json
{"status":"ok"}
```

### Models

```bash
curl -sS --max-time 10 http://127.0.0.1:18082/v1/models
```

Result: passed. Returned the four configured models.

### CAPU transcription

```bash
curl -sS --max-time 60 \
  -F model=gipformer-65M-rnnt-capu \
  -F response_format=json \
  -F file=@/tmp/vit-stt-portable-trimmed/baselines/phase0/vi_probe.wav \
  http://127.0.0.1:18082/v1/audio/transcriptions
```

Result:

```json
{"text":"Định nghĩa thế nào là ăn mặc đẹp?"}
```

## Test 3 - Code/Runtime Package With External Models

Staged a code/runtime-only package:

```bash
env VIT_STT_INCLUDE_MODELS=0 scripts/package-release release/vit-stt-code-only-scripted
cp -R release/vit-stt-code-only-scripted /tmp/vit-stt-code-only-scripted
```

Mounted external models by replacing the empty `models/` directory with a symlink:

```bash
rm -rf /tmp/vit-stt-code-only-scripted/models
ln -s /Users/leakless/code/vit-stt/models /tmp/vit-stt-code-only-scripted/models
```

Size:

```text
/tmp/vit-stt-code-only-scripted    656M
/Users/leakless/code/vit-stt/models 6.9G
```

The 656 MB code/runtime package is dominated by the packaged PyTorch CAPU worker. A no-CAPU package would be much smaller.

### External-model STT probe

```bash
/tmp/vit-stt-code-only-scripted/bin/stt-cli \
  --workspace-root /tmp/vit-stt-code-only-scripted \
  probe --model gipformer-65M-rnnt
```

Result: passed with exact expected probe output.

### External-model CAPU probe

```bash
/tmp/vit-stt-code-only-scripted/bin/stt-cli \
  --workspace-root /tmp/vit-stt-code-only-scripted \
  postprocess-capu 'rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia'
```

Result:

```text
Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia.
```

## Validation

```bash
cargo test -p stt-capu capu_worker_matches_locked_output -- --nocapture
cargo test
```

Results:

```text
cargo test -p stt-capu ... 1 passed
cargo test: 8 passed (11 suites, 48.32s)
```

## Current Verdict

`vit-stt` can now be smoke-tested in a llama.cpp-like distribution shape:

- binaries are separate from models
- model files remain external and checksum-validated
- the package can run outside the repo
- CAPU can run without a system Python when the packaged worker is included
- HTTP and CLI both work from the staged bundle

The best current release shapes are:

1. `vit-stt-code-runtime-<platform>.tar.gz`
   - `bin/`, `config/`, `baselines/`, `runtime/capu/`
   - no models included
   - user supplies or downloads `models/`

2. `vit-stt-models-vi-capu.tar.gz`
   - external model pack with `models/stt`, `models/vad`, `models/capu/model`
   - checksums must match `baselines/phase0/assets.lock.json`

3. `vit-stt-full-local-<platform>.tar.gz`
   - code/runtime plus model pack
   - largest but easiest for internal deployment

## Remaining Gaps Before Real Cross-Platform Release

- The tested packaged CAPU worker is macOS arm64 only.
- Linux and Windows need their own PyInstaller or equivalent CAPU worker builds.
- `stt-http` still discovers runtime root from current working directory; a real release should add `--runtime-root` or `VIT_STT_RUNTIME_ROOT`.
- The current config expects models to appear at `<runtime-root>/models`; arbitrary model paths need either symlinks or config edits.
- Public redistribution rights for model assets remain unresolved.
- The package still depends on `ffmpeg` for broad non-WAV audio decoding.
- CI should run these smoke tests on clean target machines or containers.

## Recommendation

Proceed with a two-artifact distribution model:

- platform-specific code/runtime package
- separate external model pack or model install directory

This gets close to the `llama.cpp` operational model now, while keeping CAPU parity through the packaged worker. The next implementation step should be runtime-root discovery plus explicit model-directory overrides, so users do not need symlinks or hand-edited configs on other machines.
