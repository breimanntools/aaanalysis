"""
Protein design: per-mutation and per-sequence ΔCPP feature-impact analysis.

Public objects: AAMut(+Plot), SeqMut(+Plot).
Consumes CPP feature impact from ``feature_engineering`` to score amino-acid mutations
(``AAMut``) and whole-sequence variants (``SeqMut``); each is paired with a plot class
for visualization.

See ``.claude/rules/code-conventions.md`` for conventions, ``CONTEXT.md`` for domain
terms (protein-design (mutation / ΔCPP) vocabulary).
"""
from ._aamut import AAMut
from ._aamut_plot import AAMutPlot
from ._seqmut import SeqMut
from ._seqmut_plot import SeqMutPlot

__all__ = [
    "AAMut",
    "AAMutPlot",
    "SeqMut",
    "SeqMutPlot",
]