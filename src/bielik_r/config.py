"""Config loading and merging for Bielik-R.

All stage configs live under `config/` as YAML files that resolve a
`defaults: [base]` list at load time. We use OmegaConf directly instead of
Hydra main decorators because the trainers are entered via `python -m` with
explicit `--config` arguments from SLURM scripts, and Hydra's working
directory rewriting interferes with SLURM log paths.

Example:

    from bielik_r.config import load_config
    cfg = load_config("config/sft.yaml")
    print(cfg.model.name_or_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _resolve_defaults(cfg: DictConfig, config_dir: Path) -> DictConfig:
    """Merge configs referenced under `defaults: [name]` in priority order.

    The list is applied left to right, with the current file having the
    highest priority (it is merged last).
    """
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg

    merged = OmegaConf.create({})
    for entry in defaults:
        if isinstance(entry, str):
            ref_path = config_dir / f"{entry}.yaml"
            loaded = OmegaConf.load(ref_path)
            loaded = _resolve_defaults(loaded, config_dir)
            merged = OmegaConf.merge(merged, loaded)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported defaults entry: {entry!r}")

    merged = OmegaConf.merge(merged, cfg)
    return merged


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config and apply the `defaults` chain and CLI overrides."""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    raw = OmegaConf.load(config_path)
    merged = _resolve_defaults(raw, config_dir)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return merged


def as_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert to a plain dict, useful for wandb.config and HF TrainingArguments."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
