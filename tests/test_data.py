"""Unit tests for the data loader module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma4_pl.data.loaders import (
    CkeRecord,
    TextRecord,
    load_cke_dataset,
    load_jsonl,
    load_text_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tiny.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    rows = list(load_jsonl(path))
    assert rows == [{"a": 1}, {"a": 2}]


def test_load_jsonl_raises_on_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(load_jsonl(path))


def test_load_text_dataset(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "Witaj swiecie", "source": "speakleash", "doc_id": "abc"},
            {"text": "Ala ma kota", "source": "wikipedia"},
            {"source": "no text here"},
        ],
    )
    records = load_text_dataset([path])
    assert len(records) == 2
    assert all(isinstance(r, TextRecord) for r in records)
    assert records[0].text == "Witaj swiecie"
    assert records[0].source == "speakleash"
    assert records[0].doc_id == "abc"
    assert records[1].source == "wikipedia"


def test_load_cke_dataset(tmp_path: Path) -> None:
    path = tmp_path / "cke.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": "Ile wynosi 2+2?",
                "response": "4",
                "year": 2020,
                "level": "podstawowa",
            },
            {"prompt": "brak odpowiedzi"},
        ],
    )
    records = load_cke_dataset([path])
    assert len(records) == 1
    assert isinstance(records[0], CkeRecord)
    assert records[0].year == 2020
