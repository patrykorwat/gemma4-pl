"""Causal LM fine tuning on a Polish text corpus.

This wraps `trl.SFTTrainer` in the simplest possible way for next token
prediction on packed text. Each training record is a JSONL row with a
single `text` field produced by `scripts/prepare_sft_data.py`.

Usage (invoked by `slurm/sft.sbatch`):

    accelerate launch -m bielik_r.training.sft \
        --config config/sft.yaml \
        --output_dir $BIELIK_R_CHECKPOINTS/sft
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from bielik_r.config import load_config

log = logging.getLogger("bielik_r.training.sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma 4 Polish SFT trainer")
    p.add_argument("--config", type=Path, required=True, help="Path to SFT YAML config")
    p.add_argument("--output_dir", type=Path, required=True, help="Checkpoint output directory")
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="OmegaConf dotlist overrides, e.g. training.num_train_epochs=1",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config, overrides=args.override)
    log.info("Loaded config stage=%s model=%s", cfg.stage, cfg.model.name_or_path)

    # Imports are lazy so that unit tests on the data format do not require
    # torch or transformers to be installed.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.name_or_path,
        padding_side=cfg.tokenizer.padding_side,
        truncation_side=cfg.tokenizer.truncation_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.use_cache = False

    train_files = [str(p) for p in cfg.data.train_files]
    eval_files = [str(p) for p in cfg.data.validation_files]

    text_field = cfg.data.get("text_field", "text") if hasattr(cfg.data, "get") else getattr(cfg.data, "text_field", "text")

    train_ds = load_dataset("json", data_files=train_files, split="train")
    eval_ds = load_dataset("json", data_files=eval_files, split="train")

    # Drop any rows with empty text so that packing does not waste positions.
    def _non_empty(row: dict) -> bool:
        return bool(row.get(text_field, "").strip())

    train_ds = train_ds.filter(_non_empty)
    eval_ds = eval_ds.filter(_non_empty)

    train_cfg = cfg.training
    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        bf16=train_cfg.bf16,
        tf32=train_cfg.tf32,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        optim=train_cfg.optim,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        logging_steps=train_cfg.logging_steps,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        dataloader_pin_memory=train_cfg.dataloader_pin_memory,
        report_to=list(cfg.logging.report_to),
        max_seq_length=cfg.data.max_seq_length,
        packing=cfg.data.packing,
        dataset_text_field=text_field,
        completion_only_loss=cfg.loss.completion_only,
        seed=cfg.project.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
    )

    log.info(
        "Starting training, %d train rows, %d eval rows",
        len(train_ds),
        len(eval_ds),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    log.info("SFT complete, checkpoint at %s", args.output_dir)


if __name__ == "__main__":
    main()
