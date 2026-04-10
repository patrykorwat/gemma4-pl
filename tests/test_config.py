"""Smoke test for the config loader."""

from __future__ import annotations

from pathlib import Path

from gemma4_pl.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_base() -> None:
    cfg = load_config(REPO_ROOT / "config" / "base.yaml")
    assert cfg.model.name_or_path == "google/gemma-4-E4B"
    assert cfg.project.seed == 42


def test_load_sft_inherits_base() -> None:
    cfg = load_config(REPO_ROOT / "config" / "sft.yaml")
    assert cfg.stage == "sft"
    assert cfg.model.name_or_path == "google/gemma-4-E4B"
    assert cfg.data.max_seq_length == 4096
    assert cfg.data.packing is True
    assert cfg.training.num_train_epochs == 1


def test_load_eval_inherits_base() -> None:
    cfg = load_config(REPO_ROOT / "config" / "eval.yaml")
    assert cfg.stage == "eval"
    assert "polish_ppl" in cfg.benchmarks
