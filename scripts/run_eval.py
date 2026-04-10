#!/usr/bin/env python
"""Evaluation runner for Polish LM checkpoints.

Reads `config/eval.yaml`, picks a suite, and runs each benchmark by
kind. The first iteration supports `perplexity` (held out Polish shard)
and `multiple_choice` (simple log likelihood scorer over MMLU style
records). Both kinds write aggregate metrics into a single JSON report.

Usage:
    python scripts/run_eval.py --checkpoint $GEMMA4_PL_CHECKPOINTS/sft --suite polish
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from gemma4_pl.config import load_config

log = logging.getLogger("scripts.run_eval")


@dataclass
class BenchmarkReport:
    name: str
    kind: str
    metric: str
    value: float
    extra: dict


def _run_perplexity(name: str, bench_cfg, checkpoint: Path) -> BenchmarkReport:
    from gemma4_pl.eval.perplexity import compute_perplexity

    result = compute_perplexity(
        checkpoint=checkpoint,
        dataset_path=bench_cfg.path,
        text_field=getattr(bench_cfg, "text_field", "text"),
        max_seq_length=int(getattr(bench_cfg, "max_seq_length", 2048)),
        batch_size=int(getattr(bench_cfg, "batch_size", 1)),
    )
    return BenchmarkReport(
        name=name,
        kind="perplexity",
        metric="perplexity",
        value=result.perplexity,
        extra={
            "loss": result.loss,
            "n_tokens": result.n_tokens,
        },
    )


def _run_multiple_choice(name: str, bench_cfg, checkpoint: Path) -> BenchmarkReport:
    """Simple MCQA scorer via log likelihood of each option.

    Expects JSONL with rows of the form
    `{"question": "...", "choices": ["A", "B", ...], "answer_index": 1}`.
    Picks the choice with the highest conditional log likelihood.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    correct = 0
    total = 0
    path = Path(bench_cfg.path)
    if not path.exists():
        log.warning("benchmark %s not found at %s, skipping", name, path)
        return BenchmarkReport(name=name, kind="multiple_choice", metric="accuracy", value=float("nan"), extra={"skipped": True})

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row["question"]
            choices = row["choices"]
            gold = int(row["answer_index"])
            scores = []
            with torch.no_grad():
                for choice in choices:
                    prompt = f"{question}\nOdpowiedz: {choice}"
                    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
                    out = model(**enc, labels=enc["input_ids"])
                    scores.append(float(out.loss.item()) * enc["input_ids"].shape[1])
            pred = int(min(range(len(choices)), key=lambda i: scores[i]))
            if pred == gold:
                correct += 1
            total += 1

    acc = correct / total if total else 0.0
    return BenchmarkReport(
        name=name,
        kind="multiple_choice",
        metric="accuracy",
        value=acc,
        extra={"correct": correct, "total": total},
    )


def run(checkpoint: Path, suite: str, output: Path, config_path: Path = Path("config/eval.yaml")) -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config_path)

    suite_benchmarks = cfg.suites[suite]
    reports: list[BenchmarkReport] = []
    for name in suite_benchmarks:
        bench_cfg = cfg.benchmarks[name]
        kind = getattr(bench_cfg, "kind", "perplexity")
        log.info("running benchmark %s kind=%s", name, kind)
        if kind == "perplexity":
            reports.append(_run_perplexity(name, bench_cfg, checkpoint))
        elif kind == "multiple_choice":
            reports.append(_run_multiple_choice(name, bench_cfg, checkpoint))
        else:
            log.warning("unknown benchmark kind %s, skipping", kind)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "suite": suite,
        "results": [
            {
                "name": r.name,
                "kind": r.kind,
                "metric": r.metric,
                "value": r.value,
                "extra": r.extra,
            }
            for r in reports
        ],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Polish LM eval suite")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--suite", type=str, default="polish")
    p.add_argument("--output", type=Path, default=Path("logs/eval.json"))
    p.add_argument("--config", type=Path, default=Path("config/eval.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        checkpoint=args.checkpoint,
        suite=args.suite,
        output=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
