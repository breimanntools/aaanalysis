"""
Protein design / engineering: per-mutation and per-sequence ΔCPP feature-impact analysis
and multi-objective directed-evolution optimization.

Public objects: AAMut(+Plot), SeqMut(+Plot), SeqOpt(+Plot).
Consumes CPP feature impact from ``feature_engineering`` to score amino-acid mutations
(``AAMut``) and whole-sequence variants (``SeqMut``); ``SeqOpt`` searches the variant space
(NSGA-II) for the multi-objective Pareto front. Each is paired with a plot class. ``SeqOpt`` is
a core class — only its SHAP-guided ``mode="impact"`` needs ``aaanalysis[pro]`` (imported lazily).

See ``.claude/rules/code-conventions.md`` for conventions, ``CONTEXT.md`` for domain
terms (protein-design (mutation / ΔCPP) + SeqOpt directed-evolution vocabulary).
"""
from ._aamut import AAMut
from ._aamut_plot import AAMutPlot
from ._seqmut import SeqMut
from ._seqmut_plot import SeqMutPlot
from ._seqopt import SeqOpt
from ._seqopt_plot import SeqOptPlot

__all__ = [
    "AAMut",
    "AAMutPlot",
    "SeqMut",
    "SeqMutPlot",
    "SeqOpt",
    "SeqOptPlot",
]