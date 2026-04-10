"""Causal LM trainer for the Polish text corpus.

This module is entered via `python -m gemma4_pl.training.sft` from the
SLURM scripts. The entry point:

  1. Parses `--config` and optional dotlist overrides
  2. Loads and resolves the YAML config
  3. Sets up accelerator, tokenizer, model
  4. Runs the training loop
  5. Saves the final checkpoint to `--output_dir`
"""
