"""Trainers for SFT, RLVR and RLHF stages.

These modules are entered via `python -m bielik_r.training.<stage>` from
the Helios SLURM scripts. Each entry point:

  1. Parses `--config` and optional dotlist overrides
  2. Loads and resolves the YAML config
  3. Sets up accelerator, tokenizer, model
  4. Runs the training loop
  5. Saves the final checkpoint to `--output_dir`
"""
