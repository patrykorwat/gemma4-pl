"""Bielik-R: Polish reasoning model on top of Gemma 4 E4B.

Top level package. Sub modules:

    bielik_r.data       dataset loaders and trace format conversion
    bielik_r.training   SFT, RLVR and RLHF trainers
    bielik_r.rewards    verifiable reward functions (math, code, format)
    bielik_r.eval       benchmark harnesses
    bielik_r.cli        Typer CLI entry points
    bielik_r.config     config loading via OmegaConf

Start with `from bielik_r.config import load_config`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bielik-r")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
