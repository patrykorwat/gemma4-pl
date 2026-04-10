#!/usr/bin/env python
"""Download Polish text corpora to the local data directory.

Supports a small set of pre configured sources. By default it pulls the
SpeakLeash Bielik mini shard (fast to validate the pipeline) and Polish
Wikipedia. Add more shards by passing `--source` multiple times, or by
editing the `SOURCES` registry below.

Usage:
    python scripts/download_datasets.py --dest $BIELIK_R_DATA
    python scripts/download_datasets.py --source speakleash --source wikipedia
    python scripts/download_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorpusSource:
    name: str
    repo: str
    kind: str  # "dataset" for HF datasets, "dataset_snapshot" for raw file pull
    revision: str = "main"
    subset: str | None = None
    split: str = "train"
    text_field: str = "text"
    description: str = ""


# Pre configured Polish corpus sources. The exact HF handles should be
# verified at download time because some of these evolve over time.
SOURCES: dict[str, CorpusSource] = {
    "speakleash": CorpusSource(
        name="speakleash",
        repo="speakleash/Bielik-data",
        kind="dataset_snapshot",
        description="SpeakLeash Bielik Polish web and literary corpus",
    ),
    "hplt_pl": CorpusSource(
        name="hplt_pl",
        repo="HPLT/HPLT2.0_cleaned",
        subset="pol_Latn",
        kind="dataset",
        description="HPLT v2 Polish cleaned subset",
    ),
    "culturax_pl": CorpusSource(
        name="culturax_pl",
        repo="uonlp/CulturaX",
        subset="pl",
        kind="dataset",
        description="CulturaX Polish subset",
    ),
    "wikipedia": CorpusSource(
        name="wikipedia",
        repo="wikimedia/wikipedia",
        subset="20231101.pl",
        kind="dataset",
        description="Polish Wikipedia dump",
    ),
    "oscar_pl": CorpusSource(
        name="oscar_pl",
        repo="oscar-corpus/OSCAR-2301",
        subset="pl",
        kind="dataset",
        description="OSCAR 2301 Polish subset",
    ),
    "cke": CorpusSource(
        name="cke",
        repo="speakleash/cke-matura",
        kind="dataset_snapshot",
        description="Optional CKE matura examples for an instruction slice",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Polish text corpora")
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("BIELIK_R_DATA", "./data")),
        help="Root dataset directory on SCRATCH",
    )
    p.add_argument(
        "--source",
        action="append",
        choices=list(SOURCES),
        help="Which source to pull, can be repeated. Defaults to speakleash + wikipedia",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without downloading",
    )
    return p.parse_args()


def _pull_dataset(src: CorpusSource, out_dir: Path) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_datasets] load_dataset repo={src.repo} subset={src.subset} split={src.split}")
    ds = load_dataset(
        src.repo,
        src.subset,
        split=src.split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    target = out_dir / f"{src.name}.jsonl"
    print(f"[download_datasets] writing {len(ds)} rows to {target}")
    with target.open("w", encoding="utf-8") as fh:
        import orjson

        for row in ds:
            text = row.get(src.text_field, "")
            if not text:
                continue
            fh.write(
                orjson.dumps(
                    {"text": text, "source": src.name},
                    option=orjson.OPT_APPEND_NEWLINE,
                ).decode()
            )


def _pull_snapshot(src: CorpusSource, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / src.name
    print(f"[download_datasets] snapshot_download repo={src.repo} -> {target}")
    snapshot_download(
        repo_id=src.repo,
        repo_type="dataset",
        revision=src.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )


def run(dest: Path, sources: list[str], dry_run: bool) -> None:
    selected = sources or ["speakleash", "wikipedia"]
    raw_dir = dest / "corpus" / "raw"
    cke_dir = dest / "cke"

    print(f"[download_datasets] dest={dest}")
    print(f"[download_datasets] selected={selected}")

    for key in selected:
        src = SOURCES[key]
        print(f"[download_datasets] {src.name}: {src.description}")
        if dry_run:
            continue
        out = cke_dir if src.name == "cke" else raw_dir
        if src.kind == "dataset":
            _pull_dataset(src, out)
        elif src.kind == "dataset_snapshot":
            _pull_snapshot(src, out)
        else:
            raise ValueError(f"unknown kind {src.kind}")

    print("[download_datasets] done")


def main() -> None:
    args = parse_args()
    run(dest=args.dest, sources=args.source or [], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
