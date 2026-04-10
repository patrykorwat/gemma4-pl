"""Dataset loaders for the Polish text corpus and the optional CKE slice.

All data is kept as JSONL for easy streaming, sharding across ranks, and
compatibility with HF `datasets.load_dataset("json", ...)`. Each record
schema is documented inline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Text corpus record schema (primary training data):
#   {
#     "text": "...",                  # normalized Polish text
#     "source": "speakleash" | "hplt" | "wikipedia" | "cke" | ...,
#     "doc_id": "...",                # optional, stable per document
#     "lang_score": 0.98,              # optional, fasttext pl confidence
#     "n_chars": 1234                  # optional, length in characters
#   }
#
# CKE instruction record schema (optional small slice):
#   {
#     "prompt": "Treść zadania maturalnego...",
#     "response": "Oczekiwana odpowiedź.",
#     "year": 2023,
#     "level": "podstawowa" | "rozszerzona",
#     "source": "cke"
#   }


@dataclass
class TextRecord:
    text: str
    source: str = "unknown"
    doc_id: str | None = None
    lang_score: float | None = None
    n_chars: int | None = None


@dataclass
class CkeRecord:
    prompt: str
    response: str
    year: int | None = None
    level: str | None = None
    source: str = "cke"


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file line by line."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_text_dataset(paths: Iterable[str | Path]) -> list[TextRecord]:
    """Load one or more text corpus JSONL shards into TextRecord objects."""
    records: list[TextRecord] = []
    for path in paths:
        for row in load_jsonl(path):
            if "text" not in row:
                continue
            records.append(
                TextRecord(
                    text=row["text"],
                    source=row.get("source", "unknown"),
                    doc_id=row.get("doc_id"),
                    lang_score=row.get("lang_score"),
                    n_chars=row.get("n_chars"),
                )
            )
    return records


def load_cke_dataset(paths: Iterable[str | Path]) -> list[CkeRecord]:
    """Load the optional CKE instruction slice."""
    records: list[CkeRecord] = []
    for path in paths:
        for row in load_jsonl(path):
            if "prompt" not in row or "response" not in row:
                continue
            records.append(
                CkeRecord(
                    prompt=row["prompt"],
                    response=row["response"],
                    year=row.get("year"),
                    level=row.get("level"),
                    source=row.get("source", "cke"),
                )
            )
    return records
