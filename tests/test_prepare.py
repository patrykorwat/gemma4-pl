"""Unit tests for the text corpus preparation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_sft_data import (  # noqa: E402
    is_valid_length,
    normalize_text,
    run,
    stream_raw_dir,
)


def test_normalize_text_strips_control_and_collapses_whitespace() -> None:
    dirty = "Ala\x00  ma\nkota\r\n"
    assert normalize_text(dirty) == "Ala ma kota"


def test_normalize_text_nfc() -> None:
    # "a" followed by combining acute accent vs precomposed.
    decomposed = "a\u0301"
    assert normalize_text(decomposed) == "\u00e1"


def test_is_valid_length() -> None:
    assert not is_valid_length("short", min_chars=10, max_chars=100)
    assert is_valid_length("x" * 50, min_chars=10, max_chars=100)
    assert not is_valid_length("x" * 200, min_chars=10, max_chars=100)


def test_stream_raw_dir(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        '{"text": "jeden"}\n{"text": "dwa"}\n', encoding="utf-8"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.jsonl").write_text('{"text": "trzy"}\n', encoding="utf-8")
    rows = list(stream_raw_dir(tmp_path))
    assert [r["text"] for r in rows] == ["jeden", "dwa", "trzy"]


def test_run_end_to_end_without_lang_id(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    long_text = "To jest polski tekst. " * 20  # well above min_chars=50
    (raw / "shard.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"text": long_text, "source": "test"}),
                json.dumps({"text": long_text, "source": "test"}),  # duplicate
                json.dumps({"text": "za krotki"}),
                json.dumps({"text": long_text + "unique"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "packed"
    stats = run(
        source=raw,
        output=out,
        lang_id_model=None,
        min_chars=50,
        max_chars=10_000,
        val_ratio=0.0,
        shard_size=100,
        seed=1,
    )
    assert stats.seen == 4
    assert stats.dropped_short == 1
    assert stats.dropped_dup == 1
    assert stats.kept == 2

    train_files = sorted((out / "train").glob("*.jsonl"))
    assert len(train_files) == 1
    with train_files[0].open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 2
    assert all(row["text"] for row in rows)
