"""Perplexity on a held out Polish text shard.

Loads a JSONL file with one `text` field per record, tokenizes it with
the checkpoint tokenizer, runs the model in eval mode, and returns the
token level negative log likelihood together with perplexity.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PerplexityResult:
    checkpoint: str
    dataset: str
    n_tokens: int
    nll_sum: float
    loss: float
    perplexity: float

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def compute_perplexity(
    checkpoint: str | Path,
    dataset_path: str | Path,
    *,
    text_field: str = "text",
    max_seq_length: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
) -> PerplexityResult:
    """Compute perplexity for a causal LM on a JSONL text shard.

    The function is lazy about heavy imports so that unit tests can
    import the module without torch installed.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    # Stream the JSONL as a simple list. Callers typically point us at a
    # held out shard of a few MB, not the full corpus.
    rows: list[str] = []
    with Path(dataset_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if text:
                rows.append(text)

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=True,
            ).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            # `out.loss` is the mean over non masked tokens. Convert to
            # a sum so that averages across variable length batches are
            # correct.
            valid_tokens = int((labels != -100).sum().item())
            nll_sum += float(out.loss.item()) * valid_tokens
            n_tokens += valid_tokens

    loss = nll_sum / max(n_tokens, 1)
    return PerplexityResult(
        checkpoint=str(checkpoint),
        dataset=str(dataset_path),
        n_tokens=n_tokens,
        nll_sum=nll_sum,
        loss=loss,
        perplexity=math.exp(loss),
    )
