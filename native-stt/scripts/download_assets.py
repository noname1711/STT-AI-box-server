#!/usr/bin/env python3
"""Download the model assets needed by vit-stt.

This script fetches the supported production assets directly from Hugging Face
and GitHub: Gipformer Vietnamese STT, English NeMo Parakeet STT, TEN VAD, and
the locked Vietnamese CAPU model.

After downloads, it regenerates ``config/models.local.json`` with the canonical
default model registry so a fresh checkout has working runtime aliases without
manual editing. Assets must be present in ``baselines/phase0/assets.lock.json``
before they are advertised by the default registry.

CAPU currently stays on the consolidated leakless/vibert-capu package, which
contains the accepted ViBERT CAPU weights plus the required base_model/ files.
Candidate CAPU models such as dragonSwing/xlm-roberta-capu, welcomyou/vibert-capu-onnx, and
tourmii/vietnamese-punc-cap-denorm-v1 are intentionally not downloaded here
because local benchmarks did not justify replacing the locked production model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
STT_ROOT = MODELS_ROOT / "stt"
VAD_ROOT = MODELS_ROOT / "vad"
CAPU_MODEL_ROOT = MODELS_ROOT / "capu"
CONFIG_ROOT = REPO_ROOT / "config"

# VI STT
GIPFORMER_REPO = "g-group-ai-lab/gipformer-65M-rnnt"
GIPFORMER_DIR = STT_ROOT / "gipformer-65M-rnnt"

# EN STT (NeMo Parakeet)
PARAKEET_EN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2"
)
PARAKEET_EN_DIR = STT_ROOT / "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"

# VAD
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.int8.onnx"
VAD_FILE = VAD_ROOT / "ten-vad.int8.onnx"

# CAPU production model. Keep this aligned with baselines/phase0/assets.lock.json
# and config/models.local.json. Do not switch it to experimental candidates
# without updating the locked baselines and CAPU benchmark report.
CAPU_MODEL_ID = "leakless/vibert-capu"
CAPU_CONSOLIDATED_DIR = CAPU_MODEL_ROOT / "vibert-capu"
CAPU_DIR = CAPU_CONSOLIDATED_DIR
CAPU_ALLOW_PATTERNS = (
    "*.py",
    "README.md",
    "config.json",
    "pytorch_model.bin",
    "verb-form-vocab.txt",
    "base_model/*.json",
    "base_model/*.txt",
    "vocabulary/*.txt",
)

# Canonical model registry aliases generated into config/models.local.json.
# The on-disk model directories still point at the original model downloads;
# only the runtime-facing IDs (aliases) are renamed.
DEFAULT_VI_STT_ID = "vit_stt_vi_v2"
DEFAULT_EN_STT_ID = "vit_stt_en_v2"


def _print_step(message: str) -> None:
    print(message)


def _download_file(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.part")
    _print_step(f"Downloading {label}...")
    _print_step(f"  URL: {url}")
    try:
        with urllib.request.urlopen(url) as response, tmp_dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_dest.replace(dest)
    except Exception as exc:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise SystemExit(f"Failed to download {label}: {exc}") from exc


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _extract_tar(archive_path: Path, dest_root: Path, label: str, mode: str) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    _print_step(f"Extracting {label}...")
    try:
        with tarfile.open(archive_path, mode) as archive:
            archive.extractall(path=dest_root)
    except Exception as exc:
        raise SystemExit(f"Failed to extract {label}: {exc}") from exc


def _strip_single_top_level_directory(source_root: Path, destination: Path) -> None:
    entries = list(source_root.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise SystemExit(f"Expected one top-level source directory in {source_root}")

    if destination.exists():
        _remove_path(destination)
    shutil.move(str(entries[0]), str(destination))


def _snapshot_download(
    repo_id: str,
    local_dir: Path,
    force: bool,
    allow_patterns: tuple[str, ...] | None = None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install requirements or run using the local virtual environment."
        ) from exc

    if force:
        _remove_path(local_dir)

    if local_dir.is_dir() and not force:
        _print_step(f"Skipping Hugging Face repo {repo_id}, already present at {local_dir}")
        return

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    _print_step(f"Downloading Hugging Face repo: {repo_id}")
    try:
        kwargs: dict[str, object] = {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "local_dir_use_symlinks": False,
        }
        if allow_patterns is not None:
            kwargs["allow_patterns"] = list(allow_patterns)
        snapshot_download(**kwargs)
    except Exception as exc:
        raise SystemExit(f"Failed to download Hugging Face repo '{repo_id}': {exc}") from exc


def download_vi_stt(force: bool) -> None:
    _snapshot_download(GIPFORMER_REPO, GIPFORMER_DIR, force)
    _print_step(f"Ready: {GIPFORMER_DIR}")


def download_en_stt_parakeet(force: bool) -> None:
    if PARAKEET_EN_DIR.is_dir() and not force:
        _print_step(f"Skipping NeMo Parakeet model, already present: {PARAKEET_EN_DIR}")
        return

    if force:
        _remove_path(PARAKEET_EN_DIR)

    with tempfile.TemporaryDirectory(prefix="vit-stt-parakeet-") as tmpdir:
        tmp_root = Path(tmpdir)
        archive_path = tmp_root / "parakeet.tar.bz2"
        extract_root = tmp_root / "extract"
        _download_file(PARAKEET_EN_URL, archive_path, "NeMo Parakeet English STT model")
        _extract_tar(archive_path, extract_root, "NeMo Parakeet English STT model", "r:bz2")
        _strip_single_top_level_directory(extract_root, PARAKEET_EN_DIR)

    _print_step(f"Ready: {PARAKEET_EN_DIR}")


def download_vad(force: bool) -> None:
    if VAD_FILE.is_file() and not force:
        _print_step(f"Skipping VAD model, already present: {VAD_FILE}")
        return
    if force:
        _remove_path(VAD_FILE)
    _download_file(VAD_URL, VAD_FILE, "TEN VAD model")
    _print_step(f"Ready: {VAD_FILE}")


def download_capu(force: bool) -> None:
    _snapshot_download(
        CAPU_MODEL_ID,
        CAPU_DIR,
        force,
        allow_patterns=CAPU_ALLOW_PATTERNS,
    )
    _print_step(f"Ready: {CAPU_DIR}")


def build_default_models_config() -> dict:
    """Return the canonical default model registry.

    The on-disk ``model_dir`` paths still point at the original model download
    directories; only the runtime-facing ``id`` aliases are renamed to the
    production aliases used by callers.
    """

    return {
        "capu_models": [
            {
                "id": "vibert-capu",
                "model_dir": "models/capu/vibert-capu",
            }
        ],
        "models": [
            {
                "id": DEFAULT_VI_STT_ID,
                "language": "vi",
                "model_dir": "models/stt/gipformer-65M-rnnt",
                "postprocess_mode": "capu",
                "capu_model_id": "vibert-capu",
                "vad_min_silence": 0.5,
                "vad_min_speech": 0.05,
                "vad_max_speech": 14.0,
                "startup_probe_wav_path": "baselines/phase0/vi_probe.wav",
                "startup_probe_expected_text": "ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP",
            },
            {
                "id": DEFAULT_EN_STT_ID,
                "language": "en",
                "model_dir": "models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
                "model_type": "nemo_transducer",
                "postprocess_mode": "none",
                "vad_min_silence": 0.5,
                "startup_probe_wav_path": "models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav",
                "startup_probe_expected_text": "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait",
            },
        ],
    }


def write_default_models_config(path: Path) -> None:
    """Write the canonical default model registry to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_default_models_config(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _print_step(f"Wrote default model registry: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model assets for vit-stt.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing local assets.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip regenerating config/models.local.json with the canonical defaults.",
    )
    parser.add_argument(
        "model",
        choices=["all", "vi-stt", "en-stt-parakeet", "vad", "capu"],
        default="all",
        nargs="?",
        help="Model to download (default: all)",
    )
    args = parser.parse_args()

    if args.model in ("all", "vi-stt"):
        download_vi_stt(args.force)
    if args.model in ("all", "en-stt-parakeet"):
        download_en_stt_parakeet(args.force)
    if args.model in ("all", "vad"):
        download_vad(args.force)
    if args.model in ("all", "capu"):
        download_capu(args.force)

    if not args.no_config:
        write_default_models_config(CONFIG_ROOT / "models.local.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
