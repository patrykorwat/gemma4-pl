"""gemma4-pl: Polish language adaptation of Gemma 4 E4B.

Top level package. Sub modules:

    gemma4_pl.data       text corpus and CKE slice loaders
    gemma4_pl.training   causal LM trainer
    gemma4_pl.eval       evaluation harnesses
    gemma4_pl.cli        Typer CLI entry points
    gemma4_pl.config     config loading via OmegaConf

Start with `from gemma4_pl.config import load_config`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gemma4-pl")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
