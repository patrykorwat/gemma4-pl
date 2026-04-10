# gemma4-fine-tuning

Polish language adaptation of `google/gemma-4-E4B` on the Helios cluster. The main training corpus is a Polish text database (web, literary, and curated Polish text). The project produces a Polish fluent base model suitable for downstream instruction tuning.

## Scope

The fine tuning stage targets language quality in Polish, not reasoning. Specifically:

1. Continued training on a large Polish text corpus (primary data source).
2. Optional light instruction mix drawn from CKE matura examples at most (no Formulo reasoning traces, no synthetic chain of thought data).
3. Evaluation on Polish language benchmarks (perplexity on held out Polish text, PolEval tasks, MMLU PL, a few Matura style sanity checks).

RLVR, GenRM, and RLHF stages are explicitly out of scope for the first iteration and are not scaffolded in the active configs.

## Target cluster: Helios (PLGrid)

The build targets the aarch64 ARM side of Helios (GH200 Grace Hopper superchip) with the SLURM scheduler.

Modules used: `GCC/13.2.0`, `Python/3.11.5`, `CUDA/12.9.1`, `cuDNN/9.19.1.2-CUDA-12.9.1`, `NVHPC/25.9-CUDA-12.9.1`. Datasets and checkpoints live on `$SCRATCH`; code lives in `$HOME`. The canonical module load sequence is in `slurm/env.sh`.

PLGrid docs: https://guide.plgrid.pl/

## Repository layout

```
gemma4-fine-tuning/
  README.md                 this file
  PLAN.md                   training plan
  pyproject.toml            package metadata, dependencies
  requirements-helios.txt   pinned versions for aarch64 GH200
  Makefile                  common targets
  .env.example              environment variables template

  config/                   YAML configs
    base.yaml
    sft.yaml                primary config for Polish corpus training
    eval.yaml

  slurm/                    Helios SLURM submission scripts
    env.sh                  module loads, venv activation
    sft.sbatch
    eval.sbatch
    download_data.sbatch

  scripts/                  CLI entry points
    download_base_model.py
    download_datasets.py
    prepare_sft_data.py
    run_eval.py

  src/bielik_r/             Python package
    data/                   dataset loaders and text pipeline
    training/               causal LM trainer
    eval/                   benchmark harnesses

  tests/                    pytest unit tests
  data/                     datasets (gitignored, symlink to $SCRATCH on Helios)
  checkpoints/              training outputs (gitignored)
  logs/                     run logs (gitignored)
```

## Quickstart on Helios

```bash
cd $HOME/gemma4-fine-tuning
source slurm/env.sh
python -m venv $SCRATCH/venvs/bielik-r
source $SCRATCH/venvs/bielik-r/bin/activate
pip install -U pip wheel
pip install -r requirements-helios.txt
pip install -e .

sbatch slurm/download_data.sbatch
sbatch slurm/sft.sbatch
sbatch slurm/eval.sbatch
```

## Local smoke tests (without a GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
python scripts/prepare_sft_data.py --dry-run
```

## Status

Scaffolding stage. The training loop, Polish corpus adapter, and evaluation harness are in place at the interface level; real data paths and hyperparameters are pending a first smoke run on Helios.
