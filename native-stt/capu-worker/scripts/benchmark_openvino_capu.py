from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_overlay(root: Path, model_dir: Path, base_model_dir: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="vit-stt-capu-openvino-"))
    overlay = temp_root / model_dir.name
    overlay.mkdir(parents=True, exist_ok=True)
    for entry in model_dir.iterdir():
        destination = overlay / entry.name
        if entry.name == "config.json":
            config = json.loads(entry.read_text(encoding="utf-8"))
            config["pretrained_name_or_path"] = str(base_model_dir)
            destination.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        elif entry.is_dir():
            os.symlink(entry, destination, target_is_directory=True)
        else:
            os.symlink(entry, destination)
    return overlay


def load_gec_model(overlay: Path, device: str):
    sys.path.insert(0, str(overlay))
    try:
        import gec_model  # type: ignore
    finally:
        sys.path.pop(0)
    return gec_model.GecBERTModel(
        vocab_path=str(overlay / "vocabulary"),
        model_paths=str(overlay),
        device=device,
        split_chunk=True,
    )


def convert_openvino_if_needed(onnx_path: Path, ir_xml_path: Path) -> None:
    if ir_xml_path.exists() and ir_xml_path.with_suffix(".bin").exists():
        return
    ir_xml_path.parent.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(ir_xml_path))


def patch_openvino_predict(model: Any, compiled_model: Any) -> None:
    request = compiled_model.create_infer_request()
    input_names = {item.get_any_name() for item in compiled_model.inputs}

    def openvino_predict(self: Any, batches: list[dict[str, torch.Tensor]]):
        predictions = []
        for batch in batches:
            feeds = {
                key: value.detach().cpu().numpy().astype(np.int64)
                for key, value in batch.items()
                if key in input_names
            }
            outputs = request.infer(feeds)
            values = list(outputs.values())
            logits = torch.from_numpy(np.asarray(values[0]))
            detect_logits = torch.from_numpy(np.asarray(values[1]))
            predictions.append(
                {
                    "logits": logits,
                    "detect_logits": detect_logits,
                    "max_error_probability": torch.ones(logits.size(0)),
                }
            )
        return self._convert(predictions)

    model.predict = types.MethodType(openvino_predict, model)


def latency_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    sorted_samples = sorted(samples)
    index = lambda p: round((len(sorted_samples) - 1) * p)
    return {
        "mean_ms": sum(sorted_samples) / len(sorted_samples),
        "p50_ms": sorted_samples[index(0.50)],
        "p95_ms": sorted_samples[index(0.95)],
        "min_ms": sorted_samples[0],
        "max_ms": sorted_samples[-1],
    }


def benchmark_callable(fn: Any, text: str, iterations: int, warmup: int) -> tuple[str, float, dict[str, float]]:
    started = time.perf_counter()
    first_output = str(fn(text)[0])
    first_ms = (time.perf_counter() - started) * 1000.0

    samples: list[float] = []
    for run_index in range(iterations + warmup):
        started = time.perf_counter()
        fn(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if run_index >= warmup:
            samples.append(elapsed_ms)
    return first_output, first_ms, latency_summary(samples)


def load_locked_snippet_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        cases.append(
            {
                "group": "locked_snippets",
                "label": item["id"],
                "text": item["clean_lower"],
                "expected": item["capu"],
                "source": str(path),
                "char_count": len(item["clean_lower"]),
            }
        )
    return cases


def load_asset_cases(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for asset in report["assets"]:
        for item in asset["cases"]:
            text = item["python"]["sample_text"]
            cases.append(
                {
                    "group": "asset_prefixes",
                    "label": item["label"],
                    "text": text,
                    "expected": item["python"]["sample_output"],
                    "source": str(path),
                    "char_count": len(text),
                    "estimated_audio_seconds": item.get("estimated_audio_seconds"),
                }
            )
    return cases


def main() -> int:
    root = workspace_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=root / "models/capu/vibert-capu")
    parser.add_argument("--base-model-dir", type=Path, default=root / "models/capu/vibert-capu/base_model")
    parser.add_argument("--onnx-path", type=Path, default=root / "target/capu-generated/vibert-capu-onnx/seq2labels.onnx")
    parser.add_argument("--ir-xml-path", type=Path, default=root / "target/capu-generated/vibert-capu-openvino-ir/seq2labels.xml")
    parser.add_argument("--snippets", type=Path, default=root / "baselines/phase0/capu_snippets.jsonl")
    parser.add_argument("--asset-report", type=Path, default=root / "baselines/spikes/capu_onnx_tuning_2026_05_25/assets_threads6_memfalse.json")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "baselines/spikes/capu_openvino_broad_2026_05_26/report.json")
    args = parser.parse_args()

    convert_openvino_if_needed(args.onnx_path, args.ir_xml_path)

    overlay = build_overlay(root, args.model_dir, args.base_model_dir)
    python_started = time.perf_counter()
    python_model = load_gec_model(overlay, "cpu")
    python_init_ms = (time.perf_counter() - python_started) * 1000.0

    openvino_model_started = time.perf_counter()
    openvino_capu_model = load_gec_model(overlay, "cpu")
    openvino_capu_model_init_ms = (time.perf_counter() - openvino_model_started) * 1000.0

    core = ov.Core()
    compile_started = time.perf_counter()
    compiled_model = core.compile_model(str(args.ir_xml_path), "CPU")
    openvino_compile_ms = (time.perf_counter() - compile_started) * 1000.0
    patch_openvino_predict(openvino_capu_model, compiled_model)

    cases = load_locked_snippet_cases(args.snippets) + load_asset_cases(args.asset_report)
    if args.quick:
        cases = [
            case
            for case in cases
            if case["group"] == "locked_snippets"
            or case["label"].endswith("/30s")
            or case["label"].endswith("/4m")
            or case["label"].endswith("/full")
        ]

    results = []
    for case in cases:
        print(f"[openvino-capu] benchmarking {case['label']} chars={case['char_count']}", file=sys.stderr)
        text = case["text"]
        expected = case["expected"]
        python_output, python_first_ms, python_warm = benchmark_callable(
            python_model,
            text,
            args.iterations,
            args.warmup,
        )
        openvino_output, openvino_first_ms, openvino_warm = benchmark_callable(
            openvino_capu_model,
            text,
            args.iterations,
            args.warmup,
        )
        results.append(
            {
                **{key: value for key, value in case.items() if key != "text"},
                "sample_text": text,
                "expected": expected,
                "python": {
                    "first_ms": python_first_ms,
                    "warm_summary": python_warm,
                    "output": python_output,
                    "matches_expected": python_output == expected,
                },
                "openvino": {
                    "first_ms": openvino_first_ms,
                    "warm_summary": openvino_warm,
                    "output": openvino_output,
                    "matches_expected": openvino_output == expected,
                    "matches_python": openvino_output == python_output,
                    "slowdown_vs_python": (
                        openvino_warm["mean_ms"] / python_warm["mean_ms"]
                        if python_warm["mean_ms"] > 0
                        else None
                    ),
                },
            }
        )

    mismatches = [
        item
        for item in results
        if not item["openvino"]["matches_expected"] or not item["openvino"]["matches_python"]
    ]
    report = {
        "openvino_version": ov.__version__,
        "python_torch_version": torch.__version__,
        "onnx_path": str(args.onnx_path),
        "ir_xml_path": str(args.ir_xml_path),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "python_init_ms": python_init_ms,
        "openvino_capu_model_init_ms": openvino_capu_model_init_ms,
        "openvino_compile_ms": openvino_compile_ms,
        "case_count": len(results),
        "mismatch_count": len(mismatches),
        "all_openvino_matches_python": len(mismatches) == 0,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
