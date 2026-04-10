"""Polish language evaluation harnesses.

The first iteration measures perplexity on a held out Polish text shard
and optionally runs a few sanity generation probes. Richer benchmarks
(PolEval, MMLU PL) are invoked by the standalone runner in
`scripts/run_eval.py`.
"""

from bielik_r.eval.perplexity import PerplexityResult, compute_perplexity

__all__ = ["PerplexityResult", "compute_perplexity"]
