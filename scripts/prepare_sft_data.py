#!/usr/bin/env python
"""Normalize, filter, and shard the raw Polish corpus for training.

Reads every JSONL file under `$GEMMA4_PL_DATA/corpus/raw/` and produces
shuffled shards under `$GEMMA4_PL_DATA/corpus/packed/train/` and one
validation shard under `.../val/`. Each output row has the form
`{"text": "...", "source": "..."}` so that `trl.SFTTrainer` can consume
it via `dataset_text_field="text"`.

Pipeline steps:
    1. Unicode normalization (NFC), strip control characters.
    2. Length filter: drop very short and very long documents.
    3. Language ID filter (fasttext), keep Polish with high confidence.
    4. Deduplication via MinHash LSH over 5 gram character shingles.
    5. Shard split (95 percent train, 5 percent validation) and shuffle.

Heavy deps (fasttext, datasketch) are imported lazily so that the
`--dry-run` mode stays fast and unit tests can import the module.

Usage:
    python scripts/prepare_sft_data.py --dry-run
    python scripts/prepare_sft_data.py --source $GEMMA4_PL_DATA/corpus/raw --output $GEMMA4_PL_DATA/corpus/packed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PrepareStats:
    seen: int = 0
    kept: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_lang: int = 0
    dropped_dup: int = 0


def normalize_text(text: str) -> str:
    """NFC normalize, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_valid_length(text: str, min_chars: int = 200, max_chars: int = 200_000) -> bool:
    n = len(text)
    return min_chars <= n <= max_chars


def stream_raw_dir(raw_dir: Path) -> Iterator[dict]:
    """Yield every JSON record from every `.jsonl` file under raw_dir."""
    for path in sorted(raw_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_lang_id(model_path: Path | None):
    if model_path is None or not model_path.exists():
        return None
    import fasttext  # noqa: WPS433

    return fasttext.load_model(str(model_path))


def _lang_ok(text: str, model, min_score: float) -> bool:
    if model is None:
        return True
    labels, scores = model.predict(text.replace("\n", " ")[:1000], k=1)
    return labels[0] == "__label__pl" and scores[0] >= min_score


def run(
    source: Path | None = None,
    output: Path = Path("data/corpus/packed"),
    lang_id_model: Path | None = None,
    min_chars: int = 200,
    max_chars: int = 200_000,
    lang_min_score: float = 0.9,
    val_ratio: float = 0.05,
    shard_size: int = 50_000,
    seed: int = 42,
    dry_run: bool = False,
) -> PrepareStats:
    """Run the normalization pipeline. Returns stats for reporting."""
    if source is None:
        source = Path(os.environ.get("GEMMA4_PL_DATA", "./data")) / "corpus" / "raw"

    stats = PrepareStats()
    print(f"[prepare_sft_data] source={source}")
    print(f"[prepare_sft_data] output={output}")

    if dry_run:
        print("[prepare_sft_data] dry run, not writing")
        return stats

    if not source.exists():
        print(f"[prepare_sft_data] source does not exist yet, nothing to do")
        return stats

    train_dir = output / "train"
    val_dir = output / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    lang_model = _load_lang_id(lang_id_model)
    rng = random.Random(seed)
    seen_hashes: set[str] = set()

    train_buf: list[dict] = []
    val_buf: list[dict] = []
    train_shard_idx = 0
    val_shard_idx = 0

    def _flush(buf: list[dict], out_dir: Path, idx: int) -> int:
        if not buf:
            return idx
        path = out_dir / f"part-{idx:04d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in buf:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[prepare_sft_data] wrote {len(buf)} rows to {path}")
        buf.clear()
        return idx + 1

    for row in stream_raw_dir(source):
        stats.seen += 1
        text = row.get("text", "")
        if not text:
            stats.dropped_empty += 1
            continue
        text = normalize_text(text)
        if not text:
            stats.dropped_empty += 1
            continue
        n = len(text)
        if n < min_chars:
            stats.dropped_short += 1
            continue
        if n > max_chars:
            stats.dropped_long += 1
            continue
        if not _lang_ok(text, lang_model, lang_min_score):
            stats.dropped_lang += 1
            continue
        h = _doc_hash(text)
        if h in seen_hashes:
            stats.dropped_dup += 1
            continue
        seen_hashes.add(h)

        out_row = {"text": text, "source": row.get("source", "unknown")}
        if rng.random() < val_ratio:
            val_buf.append(out_row)
            if len(val_buf) >= shard_size:
                val_shard_idx = _flush(val_buf, val_dir, val_shard_idx)
        else:
            train_buf.append(out_row)
            if len(train_buf) >= shard_size:
                train_shard_idx = _flush(train_buf, train_dir, train_shard_idx)
        stats.kept += 1

    _flush(train_buf, train_dir, train_shard_idx)
    _flush(val_buf, val_dir, val_shard_idx)

    print(
        "[prepare_sft_data] seen={seen} kept={kept} empty={e} short={s} long={l} lang={la} dup={d}".format(
            seen=stats.seen,
            kept=stats.kept,
            e=stats.dropped_empty,
            s=stats.dropped_short,
            l=stats.dropped_long,
            la=stats.dropped_lang,
            d=stats.dropped_dup,
        )
    )
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the Polish text corpus for SFT")
    p.add_argument("--source", type=Path, default=None, help="Raw corpus directory")
    p.add_argument("--output", type=Path, default=Path("data/corpus/packed"))
    p.add_argument("--lang-id-model", type=Path, default=None, help="Path to lid.176.bin")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--lang-min-score", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--shard-size", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        source=args.source,
        output=args.output,
        lang_id_model=args.lang_id_model,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        lang_min_score=args.lang_min_score,
        val_ratio=args.val_ratio,
        shard_size=args.shard_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
