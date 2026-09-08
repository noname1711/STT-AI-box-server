from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Suppress FutureWarning from transformers (torch.utils._pytree deprecation).
# The pinned transformers<4.35 uses a deprecated PyTorch API that still works.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*_register_pytree_node.*"
)

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_model(model_dir: Path, device: str):
    sys.path.insert(0, str(model_dir))

    # Patch: transformers >=4.34 requires ModelOutput subclasses to be @dataclass.
    # The model's Seq2LabelsOutput is not decorated, so we apply it at runtime.
    from dataclasses import dataclass

    import modeling_seq2labels

    if not hasattr(modeling_seq2labels.Seq2LabelsOutput, "__dataclass_fields__"):
        modeling_seq2labels.Seq2LabelsOutput = dataclass(
            modeling_seq2labels.Seq2LabelsOutput
        )

    # Suppress SyntaxWarning from gec_model.py (Hugging Face model):
    # invalid escape sequence '\{' on line 111. The local model copy is
    # gitignored, so we suppress at runtime instead of patching the file.
    warnings.filterwarnings(
        "ignore", category=SyntaxWarning, message=".*invalid escape sequence.*"
    )
    import gec_model  # type: ignore

    return gec_model.GecBERTModel(
        vocab_path=str(model_dir / "vocabulary"),
        model_paths=str(model_dir),
        device=device,
        split_chunk=True,
    )


def respond(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    try:
        model = build_model(model_dir, args.device)
    except Exception as exc:  # noqa: BLE001
        respond({"ok": False, "error": f"failed to initialize CAPU model: {exc}"})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            command = payload.get("command")
            if command == "ping":
                respond({"ok": True, "text": "pong"})
                continue
            if command != "process_text":
                respond({"ok": False, "error": f"unsupported command: {command}"})
                continue
            text = str(payload.get("text", ""))
            output = model(text)
            respond({"ok": True, "text": str(output[0])})
        except Exception as exc:  # noqa: BLE001
            respond({"ok": False, "error": str(exc)})

    return 0
