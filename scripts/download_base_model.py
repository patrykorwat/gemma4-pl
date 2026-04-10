#!/usr/bin/env python
"""Download the base model to the Helios SCRATCH filesystem.

The HF repo for Gemma 4 E4B is gated, so `HF_TOKEN` must be set in `.env`
or in the environment. This script uses `snapshot_download` so that a
partial failure can be resumed.

Usage:
    python scripts/download_base_model.py --repo google/gemma-4-E4B --dest $BIELIK_R_MODELS
    python scripts/download_base_model.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download base model to local cache")
    p.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("BASE_MODEL", "google/gemma-4-E4B"),
        help="HF repo id",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("BIELIK_R_MODELS", "./data/models")),
        help="Destination directory on SCRATCH",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision or tag",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the plan, do not download",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    target = args.dest / args.repo.replace("/", "__")
    print(f"[download_base_model] repo={args.repo} revision={args.revision}")
    print(f"[download_base_model] target={target}")
    if args.dry_run:
        print("[download_base_model] dry run, not downloading")
        return

    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "tokenizer.*",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )
    print(f"[download_base_model] done, {target}")


if __name__ == "__main__":
    run()
