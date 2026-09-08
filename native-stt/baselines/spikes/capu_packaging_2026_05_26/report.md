# CAPU Python Worker Packaging Test

Date: 2026-05-26  
Workspace: `/Users/leakless/code/vit-stt`

## Question

Can Python CAPU still work when `vit-stt` is distributed as a binary/library?

Yes. The cleanest shape is to keep Python outside the Rust library boundary and communicate with it as a managed worker process.

This report tests two variants:

- Option 1: Rust binary/library launches an unpacked Python CAPU sidecar from a venv.
- Option 3: Rust binary/library launches a packaged CAPU worker executable built with PyInstaller.

Both variants keep CAPU models external and replaceable.

## Option 1 - Python Sidecar From Venv

Existing implementation:

- Rust client: `/Users/leakless/code/vit-stt/crates/stt-capu/src/lib.rs`
- Python worker: `/Users/leakless/code/vit-stt/capu-worker/capu_worker/service.py`
- Runtime config: `/Users/leakless/code/vit-stt/config/runtime.toml`
- Protocol: newline-delimited JSON over stdin/stdout.

Rust currently starts:

```text
capu-worker/.venv/bin/python -m capu_worker --model-dir <overlay-model-dir> --device cpu
```

The worker supports:

```json
{"command":"ping","text":""}
{"command":"process_text","text":"..."}
```

### Option 1 Validation

Rust integration test:

```bash
cargo test -p stt-capu capu_worker_matches_locked_output -- --nocapture
```

Result:

```text
cargo test: 1 passed (2 suites, 1.86s)
```

Direct worker command:

```bash
zsh -lc 'printf ... | PYTHONPATH=capu-worker capu-worker/.venv/bin/python -m capu_worker --model-dir /Users/leakless/code/vit-stt/models/capu/model/dragonSwing--vibert-capu --device cpu'
```

Output:

```json
{"ok": true, "text": "pong"}
{"ok": true, "text": "Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia."}
```

## Option 3 - Packaged Python Worker Executable

Added entry point:

- `/Users/leakless/code/vit-stt/capu-worker/scripts/capu_worker_entry.py`

Build tool installed into the existing CAPU venv for this spike:

```text
pyinstaller==6.20.0
pyinstaller-hooks-contrib==2026.5
```

The minimal PyInstaller build succeeded but failed at runtime because dynamic CAPU model imports were not collected:

```json
{"ok": false, "error": "failed to initialize CAPU model: No module named 'difflib'"}
```

The next build collected PyTorch/Transformers but failed during CAPU inference because NumPy was not packaged in a way the Rust tokenizer extension could import:

```text
pyo3_runtime.PanicException: Failed to import NumPy module
```

The successful build explicitly collected NumPy, PyTorch, Transformers, tokenizers, and sentencepiece:

```bash
capu-worker/.venv/bin/pyinstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name vit-stt-capu-worker-full-numpy \
  --paths capu-worker \
  --hidden-import difflib \
  --hidden-import filelock \
  --hidden-import numpy \
  --hidden-import torch \
  --hidden-import transformers \
  --hidden-import sentencepiece \
  --collect-all numpy \
  --collect-all torch \
  --collect-all transformers \
  --collect-all tokenizers \
  --collect-all sentencepiece \
  --distpath baselines/spikes/capu_packaging_2026_05_26/dist \
  --workpath baselines/spikes/capu_packaging_2026_05_26/build \
  --specpath baselines/spikes/capu_packaging_2026_05_26 \
  capu-worker/scripts/capu_worker_entry.py
```

Packaged executable:

```text
/Users/leakless/code/vit-stt/baselines/spikes/capu_packaging_2026_05_26/dist/vit-stt-capu-worker-full-numpy/vit-stt-capu-worker-full-numpy
```

### Option 3 Validation

Command:

```bash
zsh -lc 'printf ... | baselines/spikes/capu_packaging_2026_05_26/dist/vit-stt-capu-worker-full-numpy/vit-stt-capu-worker-full-numpy --model-dir /Users/leakless/code/vit-stt/models/capu/model/dragonSwing--vibert-capu --device cpu'
```

Output:

```json
{"ok": true, "text": "pong"}
{"ok": true, "text": "Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia."}
```

## Size

| Artifact | Size |
| --- | ---: |
| Current CAPU venv | 920 MB |
| Minimal PyInstaller worker, not inference-capable | 19 MB |
| Full PyInstaller worker without explicit NumPy collection, runtime fails | 587 MB |
| Full working PyInstaller worker | 604 MB |
| CAPU model directory | 441 MB |
| ViBERT base-model config/vocab directory | 268 KB |

The working packaged-worker distribution plus current external CAPU model is roughly 1.0 GB before compression and before STT model assets.

## Timing Smoke Test

Same text was sent to both workers:

```text
rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia
```

| Worker | Startup to ping | Ping to first CAPU result | Output |
| --- | ---: | ---: | --- |
| venv Python worker | 1517.37 ms | 33.49 ms | exact expected |
| PyInstaller worker | 1488.55 ms | 44.72 ms | exact expected |

This is a smoke test, not a full benchmark. It shows the packaged worker preserves the protocol and output. It does not prove long-form performance beyond the already-tested Python CAPU baseline behavior.

## Recommendation

Option 1 should be the development and production-parity default for now:

- Lowest packaging complexity.
- Exact current CAPU behavior.
- Easy to update Python dependencies and model files.
- Already works through the Rust `stt-capu` client.

Option 3 is viable for binary-like distribution, but should be treated as a packaging layer around the same worker protocol:

- Keep the Rust side speaking JSON-lines to a worker process.
- Add config support for `worker_command` or `worker_exe` so Rust can launch either `python -m capu_worker` or the packaged executable.
- Keep CAPU model assets external.
- Maintain a PyInstaller spec instead of relying on CLI flags, because PyTorch/Transformers/tokenizers require explicit collection.
- Expect large artifacts and platform-specific builds.

Do not embed Python inside the Rust library for this phase. The worker-process boundary is simpler, easier to debug, and already passes the CAPU parity probe.
