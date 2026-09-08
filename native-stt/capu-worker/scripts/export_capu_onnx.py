from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
import torch


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir_default(root: Path) -> Path:
    return root / "models/capu/vibert-capu"


def base_model_dir_default(root: Path) -> Path:
    return root / "models/capu/vibert-capu/base_model"


def output_dir_default(root: Path) -> Path:
    return root / "target/capu-generated/vibert-capu-onnx"


def build_overlay(model_dir: Path, base_model_dir: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="vit-stt-capu-onnx-"))
    overlay = temp_root / model_dir.name
    overlay.mkdir(parents=True, exist_ok=True)

    for entry in model_dir.iterdir():
        destination = overlay / entry.name
        if entry.name == "config.json":
            config = json.loads(entry.read_text(encoding="utf-8"))
            config["pretrained_name_or_path"] = str(base_model_dir)
            destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        if entry.is_dir():
            os.symlink(entry, destination, target_is_directory=True)
        else:
            os.symlink(entry, destination)

    return overlay


def prepare_tokenizer_files(base_model_dir: Path, output_dir: Path) -> dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        do_basic_tokenize=False,
        do_lower_case=False,
        model_max_length=1024,
        use_fast=True,
    )
    tokenizer.add_tokens(["$START"])
    tokenizer.save_pretrained(str(output_dir))

    tokenizer_json = output_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise RuntimeError(f"expected tokenizer export at {tokenizer_json}")

    start_token_id = tokenizer.convert_tokens_to_ids("$START")
    return {
        "tokenizer_json": str(tokenizer_json),
        "start_token": "$START",
        "start_token_id": int(start_token_id),
        "tokenizer_vocab_size": int(len(tokenizer)),
    }


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, input_offsets):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_offsets=input_offsets,
            return_dict=False,
        )
        logits, detect_logits = outputs[:2]
        return logits, detect_logits


def export_model(model_dir: Path, base_model_dir: Path, output_dir: Path, opset: int) -> dict[str, object]:
    overlay = build_overlay(model_dir, base_model_dir)
    sys.path.insert(0, str(overlay))
    try:
        from modeling_seq2labels import Seq2LabelsModel  # type: ignore
    finally:
        sys.path.pop(0)

    model = Seq2LabelsModel.from_pretrained(str(overlay))
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()

    input_ids = torch.tensor([[38168, 337, 17, 125]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    input_offsets = torch.tensor([[0, 1, 2]], dtype=torch.long)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "seq2labels.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, input_offsets),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "input_offsets"],
            output_names=["logits", "detect_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "token_sequence"},
                "attention_mask": {0: "batch", 1: "token_sequence"},
                "input_offsets": {0: "batch", 1: "word_sequence"},
                "logits": {0: "batch", 1: "word_sequence"},
                "detect_logits": {0: "batch", 1: "word_sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    shutil.copy2(model_dir / "vocabulary/labels.txt", output_dir / "labels.txt")
    shutil.copy2(model_dir / "vocabulary/d_tags.txt", output_dir / "d_tags.txt")
    shutil.copy2(model_dir / "verb-form-vocab.txt", output_dir / "verb-form-vocab.txt")
    shutil.copy2(base_model_dir / "vocab.txt", output_dir / "vocab.txt")

    tokenizer_meta = prepare_tokenizer_files(base_model_dir, output_dir)

    metadata = {
        "format_version": 1,
        "source_model_dir": str(model_dir),
        "source_base_model_dir": str(base_model_dir),
        "exported_at_unix_seconds": int(Path(onnx_path).stat().st_mtime),
        "onnx_path": str(onnx_path),
        "opset": opset,
        "inputs": ["input_ids", "attention_mask", "input_offsets"],
        "outputs": ["logits", "detect_logits"],
        "start_token": "$START",
        "start_token_id": 38168,
        "max_len": 64,
        "min_len": 3,
        "iterations": 3,
        "min_error_probability": 0.0,
        "split_chunk": True,
        "chunk_size": 48,
        "overlap_size": 12,
        "min_words_cut": 6,
        "punctuation": [":", ".", ",", "?"],
        **tokenizer_meta,
    }
    metadata_path = output_dir / "export-metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=model_dir_default(root))
    parser.add_argument("--base-model-dir", type=Path, default=base_model_dir_default(root))
    parser.add_argument("--output-dir", type=Path, default=output_dir_default(root))
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    metadata = export_model(
        model_dir=args.model_dir.expanduser().resolve(),
        base_model_dir=args.base_model_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        opset=args.opset,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
