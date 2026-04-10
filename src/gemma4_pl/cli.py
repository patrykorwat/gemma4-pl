"""Typer CLI entry points, registered in `pyproject.toml`.

These are thin wrappers around the underlying modules so that common
developer operations have short command names once `pip install -e .` has
been run:

    gemma4-pl-prepare ...
    gemma4-pl-sft ...
    gemma4-pl-eval ...
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="gemma4-pl CLI")


@app.command()
def prepare(
    output: Path = typer.Option(Path("data/corpus/packed"), help="Where to write prepared JSONL"),
    dry_run: bool = typer.Option(False, help="Only print what would be written"),
) -> None:
    """Prepare the Polish text corpus (normalize, filter, shard)."""
    from scripts.prepare_sft_data import run

    run(output=output, dry_run=dry_run)


@app.command()
def sft(
    config: Path = typer.Option(Path("config/sft.yaml")),
    output_dir: Path = typer.Option(Path("checkpoints/sft")),
) -> None:
    """Run SFT locally (for smoke tests, real runs go through SLURM)."""
    from gemma4_pl.training.sft import main as run_sft

    import sys

    sys.argv = [
        "gemma4_pl.training.sft",
        "--config",
        str(config),
        "--output_dir",
        str(output_dir),
    ]
    run_sft()


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Checkpoint to evaluate"),
    suite: str = typer.Option("all", help="Eval suite name from config/eval.yaml"),
    output: Path = typer.Option(Path("logs/eval.json")),
) -> None:
    """Evaluate a checkpoint against the Polish LM suite."""
    from scripts.run_eval import run

    run(checkpoint=checkpoint, suite=suite, output=output)


if __name__ == "__main__":
    app()
