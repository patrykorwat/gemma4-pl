# gemma4-pl

Polish language adaptation of `google/gemma-4-E4B`. The main training corpus is SpeakLeash, the same Polish text pool used for the Bielik models (web, literary, forums, curated text). The project produces a Polish fluent base model suitable for downstream instruction tuning.

## Scope

The fine tuning stage targets language quality in Polish, not reasoning. Specifically:

1. Continued training on a large Polish text corpus (primary data source).
2. Optional light instruction mix drawn from CKE matura examples at most (no reasoning trace data, no synthetic chain of thought).
3. Evaluation on Polish language benchmarks (perplexity on a held out Polish text shard, PolEval tasks, MMLU PL, a few Matura style sanity checks).

## Target hardware

The build targets aarch64 ARM GH200 Grace Hopper nodes with a SLURM scheduler. Account, partition, and storage paths are configured via environment variables and are not hard coded.

Modules loaded by `slurm/env.sh`: `GCC/13.2.0`, `Python/3.11.5`, `CUDA/12.9.1`, `cuDNN/9.19.1.2-CUDA-12.9.1`, `NVHPC/25.9-CUDA-12.9.1`. Datasets and checkpoints live on `$SCRATCH`; code lives in `$HOME`.

## Repository layout

```
gemma4-pl/
  README.md                 this file
  PLAN.md                   training plan
  pyproject.toml            package metadata, dependencies
  requirements-cluster.txt  pinned versions for aarch64 GH200
  Makefile                  common targets
  .env.example              environment variables template

  config/                   YAML configs
    base.yaml
    sft.yaml                primary config for Polish corpus training
    eval.yaml

  slurm/                    SLURM submission scripts
    env.sh                  module loads, venv activation
    sft.sbatch
    eval.sbatch
    download_data.sbatch

  scripts/                  CLI entry points
    download_base_model.py
    download_datasets.py
    prepare_sft_data.py
    run_eval.py

  src/gemma4_pl/             Python package
    data/                   dataset loaders and text pipeline
    training/               causal LM trainer
    eval/                   benchmark harnesses

  tests/                    pytest unit tests
  data/                     datasets (gitignored, symlink to $SCRATCH)
  checkpoints/              training outputs (gitignored)
  logs/                     run logs (gitignored)
```

## Quickstart on the cluster

```bash
cd $HOME/gemma4-pl
source slurm/env.sh
python -m venv $SCRATCH/venvs/gemma4-pl
source $SCRATCH/venvs/gemma4-pl/bin/activate
pip install -U pip wheel
pip install -r requirements-cluster.txt
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

Scaffolding stage. The training loop, Polish corpus adapter, and evaluation harness are in place at the interface level; real data paths and hyperparameters are pending a first smoke run on the cluster.
