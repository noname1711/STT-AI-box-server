#!/usr/bin/env python3
"""Benchmark Vietnamese CAPU candidate models.

The benchmark intentionally treats the current locked CAPU output as the
production reference. That gives us a reproducible migration-risk score even
when public candidate models do not publish comparable benchmark sets.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "baselines" / "spikes" / "capu_candidate_benchmark_2026_05_27"
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "target" / "capu-candidate-models"
CURRENT_CAPU_DIR = REPO_ROOT / "models" / "capu" / "vibert-capu"
SNIPPETS_PATH = REPO_ROOT / "baselines" / "phase0" / "capu_snippets.jsonl"
LONG_CASES_PATH = (
    REPO_ROOT
    / "baselines"
    / "spikes"
    / "capu_onnx_tuning_2026_05_25"
    / "assets_threads6_memfalse.json"
)


GEC_ALLOW_PATTERNS = [
    "*.py",
    "config.json",
    "pytorch_model.bin",
    "verb-form-vocab.txt",
    "vocabulary/*.txt",
]

TOURMII_ALLOW_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "dict.txt",
]


@dataclass(frozen=True)
class Candidate:
    key: str
    repo_id: str
    kind: str
    local_dir: Path
    description: str
    allow_patterns: list[str] | None


def candidates(candidate_root: Path) -> list[Candidate]:
    return [
        Candidate(
            key="vibert-capu-current",
            repo_id="leakless/vibert-capu",
            kind="gec",
            local_dir=CURRENT_CAPU_DIR,
            description="Current production CAPU baseline, consolidated ViBERT Seq2Labels.",
            allow_patterns=None,
        ),
        Candidate(
            key="xlm-roberta-capu",
            repo_id="dragonSwing/xlm-roberta-capu",
            kind="gec",
            local_dir=candidate_root / "dragonSwing--xlm-roberta-capu",
            description="dragonSwing XLM-RoBERTa Seq2Labels CAPU candidate.",
            allow_patterns=GEC_ALLOW_PATTERNS,
        ),
        Candidate(
            key="tourmii-vietnamese-punc-cap-denorm-v1",
            repo_id="tourmii/vietnamese-punc-cap-denorm-v1",
            kind="seq2seq",
            local_dir=candidate_root / "tourmii--vietnamese-punc-cap-denorm-v1",
            description="mBART text2text punc/cap/denormalization candidate.",
            allow_patterns=TOURMII_ALLOW_PATTERNS,
        ),
    ]


def load_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for line in SNIPPETS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            {
                "id": f"snippet/{row['id']}",
                "input": row["clean_lower"],
                "reference": row["capu"],
                "source": "locked_snippet",
            }
        )

    if LONG_CASES_PATH.is_file():
        data = json.loads(LONG_CASES_PATH.read_text(encoding="utf-8"))
        for asset in data.get("assets", []):
            for case in asset.get("cases", []):
                label = str(case["label"])
                # Keep the local run bounded while still covering long-form behavior.
                if not any(label.endswith(suffix) for suffix in ("/30s", "/2m", "/4m")):
                    continue
                python_case = case["python"]
                cases.append(
                    {
                        "id": f"asset/{label}",
                        "input": python_case["sample_text"],
                        "reference": python_case["sample_output"],
                        "source": "previous_audio_derived_python_capu",
                    }
                )
    return cases


def snapshot(candidate: Candidate, force: bool) -> None:
    if candidate.key == "vibert-capu-current":
        if not candidate.local_dir.is_dir():
            raise SystemExit(
                f"Current CAPU model missing at {candidate.local_dir}. Run `stt-cli download-models` first."
            )
        patch_gec_config(candidate.local_dir)
        return

    if force and candidate.local_dir.exists():
        shutil.rmtree(candidate.local_dir)
    if candidate.local_dir.is_dir():
        patch_gec_config(candidate.local_dir)
        return

    from huggingface_hub import snapshot_download

    candidate.local_dir.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": candidate.repo_id,
        "local_dir": str(candidate.local_dir),
        "local_dir_use_symlinks": False,
    }
    if candidate.allow_patterns:
        kwargs["allow_patterns"] = candidate.allow_patterns
    snapshot_download(**kwargs)
    patch_gec_config(candidate.local_dir)


def patch_gec_config(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    repo_hint = str(config.get("pretrained_name_or_path", ""))
    if "xlm" in model_dir.name and not repo_hint:
        config["pretrained_name_or_path"] = "xlm-roberta-base"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_gec_modules() -> None:
    for name in [
        "gec_model",
        "modeling_seq2labels",
        "configuration_seq2labels",
        "utils",
        "vocabulary",
    ]:
        sys.modules.pop(name, None)


def build_runner(candidate: Candidate):
    started = time.perf_counter()
    if candidate.kind == "gec":
        clear_gec_modules()
        sys.path.insert(0, str(candidate.local_dir))
        try:
            gec_model = importlib.import_module("gec_model")
            model = gec_model.GecBERTModel(
                vocab_path=str(candidate.local_dir / "vocabulary"),
                model_paths=str(candidate.local_dir),
                device="cpu",
                split_chunk=True,
            )
        finally:
            try:
                sys.path.remove(str(candidate.local_dir))
            except ValueError:
                pass

        def run(text: str) -> str:
            return str(model(text)[0])

    elif candidate.kind == "seq2seq":
        from transformers import pipeline

        pipe = pipeline("text2text-generation", model=str(candidate.local_dir), device=-1)

        def run(text: str) -> str:
            # Keep enough headroom for punctuation, capitalization, and denormalization.
            max_length = min(1024, max(64, int(len(text.split()) * 2.5) + 32))
            return str(
                pipe(
                    text,
                    max_length=max_length,
                    num_beams=4,
                    do_sample=False,
                    truncation=True,
                )[0]["generated_text"]
            )

    else:
        raise ValueError(f"unsupported candidate kind: {candidate.kind}")
    return run, (time.perf_counter() - started) * 1000.0


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def punctuation_signature(text: str) -> str:
    return "".join(ch for ch in text if ch in ".,:?!")


def uppercase_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ch.upper() == ch and ch.lower() != ch)


def model_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": math.nan, "p50_ms": math.nan, "min_ms": math.nan, "max_ms": math.nan}
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def benchmark_candidate(candidate: Candidate, cases: list[dict[str, str]], iterations: int) -> dict[str, Any]:
    run, init_ms = build_runner(candidate)
    first_started = time.perf_counter()
    first_output = run(cases[0]["input"])
    first_ms = (time.perf_counter() - first_started) * 1000.0

    case_reports: list[dict[str, Any]] = []
    all_times: list[float] = []
    for case in cases:
        timings: list[float] = []
        output = ""
        for _ in range(iterations):
            started = time.perf_counter()
            output = run(case["input"])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timings.append(elapsed_ms)
            all_times.append(elapsed_ms)
        reference = case["reference"]
        distance = levenshtein(output, reference)
        case_reports.append(
            {
                "id": case["id"],
                "source": case["source"],
                "input_chars": len(case["input"]),
                "input_words": len(case["input"].split()),
                "reference": reference,
                "output": output,
                "exact_match": output == reference,
                "char_edit_distance": distance,
                "char_edit_rate": distance / max(1, len(reference)),
                "punctuation_reference": punctuation_signature(reference),
                "punctuation_output": punctuation_signature(output),
                "punctuation_exact_match": punctuation_signature(output) == punctuation_signature(reference),
                "uppercase_delta": uppercase_count(output) - uppercase_count(reference),
                "timing": summarize(timings),
            }
        )

    exact = sum(1 for case in case_reports if case["exact_match"])
    punct_exact = sum(1 for case in case_reports if case["punctuation_exact_match"])
    return {
        "key": candidate.key,
        "repo_id": candidate.repo_id,
        "kind": candidate.kind,
        "description": candidate.description,
        "local_dir": str(candidate.local_dir),
        "model_size_mb": model_size_bytes(candidate.local_dir) / 1_000_000.0,
        "init_ms": init_ms,
        "first_inference_ms": first_ms,
        "first_output": first_output,
        "overall_timing": summarize(all_times),
        "case_count": len(case_reports),
        "exact_match_count": exact,
        "exact_match_rate": exact / max(1, len(case_reports)),
        "punctuation_exact_match_count": punct_exact,
        "punctuation_exact_match_rate": punct_exact / max(1, len(case_reports)),
        "mean_char_edit_rate": statistics.fmean(case["char_edit_rate"] for case in case_reports),
        "cases": case_reports,
    }


def write_report(result: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Vietnamese CAPU candidate benchmark")
    lines.append("")
    lines.append(f"Generated at: `{result['generated_at']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This benchmark compares candidate Vietnamese capitalization and punctuation postprocessors "
        "against the current locked `leakless/vibert-capu` production outputs. The score is therefore "
        "a replacement-risk benchmark, not an independent linguistic gold benchmark."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Candidate | Size MB | Init ms | First ms | Mean ms | Exact | Punctuation exact | Mean char edit rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in result["models"]:
        lines.append(
            "| {key} | {size:.1f} | {init:.1f} | {first:.1f} | {mean:.1f} | {exact}/{count} | {punct}/{count} | {cer:.3f} |".format(
                key=model["key"],
                size=model["model_size_mb"],
                init=model["init_ms"],
                first=model["first_inference_ms"],
                mean=model["overall_timing"]["mean_ms"],
                exact=model["exact_match_count"],
                punct=model["punctuation_exact_match_count"],
                count=model["case_count"],
                cer=model["mean_char_edit_rate"],
            )
        )
    lines.append("")
    lines.append("## Case-level output drift")
    for model in result["models"]:
        lines.append("")
        lines.append(f"### {model['key']}")
        lines.append("")
        lines.append("| Case | Exact | Punct exact | Edit rate | Mean ms | Output |")
        lines.append("| --- | :---: | :---: | ---: | ---: | --- |")
        for case in model["cases"]:
            output = case["output"].replace("|", "\\|")
            if len(output) > 180:
                output = output[:177] + "..."
            lines.append(
                "| {id} | {exact} | {punct} | {cer:.3f} | {mean:.1f} | {output} |".format(
                    id=case["id"],
                    exact="yes" if case["exact_match"] else "no",
                    punct="yes" if case["punctuation_exact_match"] else "no",
                    cer=case["char_edit_rate"],
                    mean=case["timing"]["mean_ms"],
                    output=output,
                )
            )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "Keep `leakless/vibert-capu` as the default unless another candidate wins on both "
        "quality and operational constraints. A candidate should not replace it solely because it has "
        "similar public model-card metrics; it must pass the local locked snippets and long transcript cases."
    )
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Candidate keys to run. Default runs all candidates.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = candidates(args.candidate_dir)
    if args.models:
        wanted = set(args.models)
        selected = [candidate for candidate in selected if candidate.key in wanted]
        missing = wanted.difference(candidate.key for candidate in selected)
        if missing:
            raise SystemExit(f"Unknown candidate key(s): {', '.join(sorted(missing))}")

    cases = load_cases()
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "iterations": args.iterations,
        "case_count": len(cases),
        "cases": cases,
        "models": [],
    }

    for candidate in selected:
        print(f"[benchmark] preparing {candidate.key}", flush=True)
        snapshot(candidate, args.force_download)
        print(f"[benchmark] running {candidate.key}", flush=True)
        model_result = benchmark_candidate(candidate, cases, args.iterations)
        result["models"].append(model_result)
        gc.collect()

    json_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, report_path)
    print(f"[benchmark] wrote {json_path}")
    print(f"[benchmark] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
