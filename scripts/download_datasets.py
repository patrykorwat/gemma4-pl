#!/usr/bin/env python
"""Download Polish text corpora to the local data directory.

The primary source is SpeakLeash (the Bielik text corpus). SpeakLeash is
not published as a HuggingFace dataset, it is streamed via the
`speakleash` Python package. This script downloads SpeakLeash shards by
name and writes them out as JSONL into `$GEMMA4_PL_DATA/corpus/raw/`.

Auxiliary HuggingFace sources (HPLT v2 PL, CulturaX PL, Polish Wikipedia,
OSCAR PL) are also registered and can be pulled with `--source`.

Usage:
    # default: pull the core SpeakLeash shards (plwiki + a few forums)
    python scripts/download_datasets.py --dest $GEMMA4_PL_DATA

    # pull every SpeakLeash shard (warning: hundreds of GB)
    python scripts/download_datasets.py --source speakleash --speakleash-all

    # pull specific SpeakLeash shards
    python scripts/download_datasets.py \
        --source speakleash \
        --speakleash-shard plwiki --speakleash-shard forum_wolnepodroze

    # mix with a HuggingFace source
    python scripts/download_datasets.py --source speakleash --source wikipedia

    python scripts/download_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CorpusSource:
    name: str
    kind: str  # "speakleash", "hf_dataset", or "hf_snapshot"
    repo: str = ""
    revision: str = "main"
    subset: str | None = None
    split: str = "train"
    text_field: str = "text"
    description: str = ""
    default_shards: list[str] = field(default_factory=list)


# Default SpeakLeash shards. Keep this list small for the default path so
# that a first run validates the pipeline quickly. The full corpus is
# hundreds of GB and should only be pulled with --speakleash-all.
DEFAULT_SPEAKLEASH_SHARDS = [
    "plwiki",
    "forum_wolnepodroze",
    "forum_gazeta",
    "wolnelektury_pl_txt",
]


SOURCES: dict[str, CorpusSource] = {
    "speakleash": CorpusSource(
        name="speakleash",
        kind="speakleash",
        description="SpeakLeash Polish text corpus (the Bielik training data)",
        default_shards=DEFAULT_SPEAKLEASH_SHARDS,
    ),
    "hplt_pl": CorpusSource(
        name="hplt_pl",
        kind="hf_dataset",
        repo="HPLT/HPLT2.0_cleaned",
        subset="pol_Latn",
        description="HPLT v2 Polish cleaned subset",
    ),
    "culturax_pl": CorpusSource(
        name="culturax_pl",
        kind="hf_dataset",
        repo="uonlp/CulturaX",
        subset="pl",
        description="CulturaX Polish subset",
    ),
    "wikipedia": CorpusSource(
        name="wikipedia",
        kind="hf_dataset",
        repo="wikimedia/wikipedia",
        subset="20231101.pl",
        description="Polish Wikipedia dump (HuggingFace mirror)",
    ),
    "oscar_pl": CorpusSource(
        name="oscar_pl",
        kind="hf_dataset",
        repo="oscar-corpus/OSCAR-2301",
        subset="pl",
        description="OSCAR 2301 Polish subset (gated, needs HF_TOKEN)",
    ),
    "cke": CorpusSource(
        name="cke",
        kind="hf_snapshot",
        repo="speakleash/cke-matura",
        description="Optional CKE matura examples for an instruction slice",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Polish text corpora")
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("GEMMA4_PL_DATA", "./data")),
        help="Root dataset directory (usually $GEMMA4_PL_DATA on the cluster)",
    )
    p.add_argument(
        "--source",
        action="append",
        choices=list(SOURCES),
        help="Which source to pull, can be repeated. Defaults to speakleash",
    )
    p.add_argument(
        "--speakleash-shard",
        action="append",
        default=[],
        help="SpeakLeash shard name, can be repeated. Overrides the defaults.",
    )
    p.add_argument(
        "--speakleash-all",
        action="store_true",
        help="Pull every SpeakLeash shard. Warning: hundreds of GB.",
    )
    p.add_argument(
        "--speakleash-cache",
        type=Path,
        default=None,
        help="Where SpeakLeash should cache its shard files. "
        "Defaults to <dest>/corpus/speakleash_cache",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without downloading",
    )
    return p.parse_args()


def _write_jsonl_row(fh, text: str, source: str, doc_id: str | None = None) -> None:
    import orjson

    payload: dict[str, object] = {"text": text, "source": source}
    if doc_id is not None:
        payload["doc_id"] = doc_id
    fh.write(
        orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE).decode()
    )


def _pull_hf_dataset(src: CorpusSource, out_dir: Path) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[download_datasets] load_dataset repo={src.repo} "
        f"subset={src.subset} split={src.split}"
    )
    ds = load_dataset(
        src.repo,
        src.subset,
        split=src.split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    target = out_dir / f"{src.name}.jsonl"
    print(f"[download_datasets] writing {len(ds)} rows to {target}")
    with target.open("w", encoding="utf-8") as fh:
        for row in ds:
            text = row.get(src.text_field, "")
            if not text:
                continue
            _write_jsonl_row(fh, text, src.name)


def _pull_hf_snapshot(src: CorpusSource, out_dir: Path) -> None:
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


def _pull_speakleash(
    src: CorpusSource,
    out_dir: Path,
    cache_dir: Path,
    shards: list[str],
    pull_all: bool,
) -> None:
    """Stream SpeakLeash shards via the speakleash Python package.

    Each shard is written to its own JSONL file under
    `<out_dir>/speakleash/<shard_name>.jsonl` so that the preparation
    stage (dedup, language filter, packing) can walk them with a simple
    `*.jsonl` glob.
    """
    try:
        from speakleash import Speakleash
    except ImportError as exc:
        raise RuntimeError(
            "The `speakleash` package is required to pull SpeakLeash shards. "
            "Install it with `pip install speakleash`."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_out_dir = out_dir / "speakleash"
    shard_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_datasets] speakleash cache={cache_dir}")
    sl = Speakleash(str(cache_dir))

    available = {d.name: d for d in sl.datasets}
    print(f"[download_datasets] speakleash: {len(available)} shards available")

    if pull_all:
        selected_names = sorted(available)
    else:
        selected_names = shards or list(src.default_shards)

    missing = [name for name in selected_names if name not in available]
    if missing:
        raise ValueError(
            f"Unknown SpeakLeash shards: {missing}. "
            f"Available examples: {sorted(available)[:10]}..."
        )

    for name in selected_names:
        ds = available[name]
        target = shard_out_dir / f"{name}.jsonl"
        n_docs = getattr(ds, "documents", None)
        n_chars = getattr(ds, "characters", None)
        print(
            f"[download_datasets] speakleash shard={name} "
            f"docs={n_docs} chars={n_chars} -> {target}"
        )

        kept = 0
        with target.open("w", encoding="utf-8") as fh:
            # `ext_data` yields (text, meta) pairs with richer info.
            # Fall back to `data` (plain text) if ext_data is absent on
            # this shard, which can happen for older manifests.
            iterator = getattr(ds, "ext_data", None) or ds.data
            for item in iterator:
                if isinstance(item, tuple):
                    text, meta = item
                    doc_id = None
                    if isinstance(meta, dict):
                        doc_id = meta.get("url") or meta.get("id") or meta.get("title")
                else:
                    text = item
                    doc_id = None
                if not text:
                    continue
                _write_jsonl_row(fh, text, f"speakleash/{name}", doc_id)
                kept += 1
        print(f"[download_datasets] speakleash shard={name} wrote={kept} rows")


def run(
    dest: Path,
    sources: list[str],
    dry_run: bool,
    speakleash_shards: list[str] | None = None,
    speakleash_all: bool = False,
    speakleash_cache: Path | None = None,
) -> None:
    selected = sources or ["speakleash"]
    raw_dir = dest / "corpus" / "raw"
    cke_dir = dest / "cke"
    sl_cache = speakleash_cache or (dest / "corpus" / "speakleash_cache")

    print(f"[download_datasets] dest={dest}")
    print(f"[download_datasets] selected={selected}")
    if "speakleash" in selected:
        mode = "all" if speakleash_all else (speakleash_shards or DEFAULT_SPEAKLEASH_SHARDS)
        print(f"[download_datasets] speakleash shards={mode}")

    for key in selected:
        src = SOURCES[key]
        print(f"[download_datasets] {src.name}: {src.description}")
        if dry_run:
            continue
        out = cke_dir if src.name == "cke" else raw_dir
        if src.kind == "speakleash":
            _pull_speakleash(
                src,
                out,
                cache_dir=sl_cache,
                shards=speakleash_shards or [],
                pull_all=speakleash_all,
            )
        elif src.kind == "hf_dataset":
            _pull_hf_dataset(src, out)
        elif src.kind == "hf_snapshot":
            _pull_hf_snapshot(src, out)
        else:
            raise ValueError(f"unknown kind {src.kind}")

    print("[download_datasets] done")


def main() -> None:
    args = parse_args()
    run(
        dest=args.dest,
        sources=args.source or [],
        dry_run=args.dry_run,
        speakleash_shards=args.speakleash_shard,
        speakleash_all=args.speakleash_all,
        speakleash_cache=args.speakleash_cache,
    )


if __name__ == "__main__":
    main()
