"""
Protein design (pro): SHAP-guided multi-objective directed-evolution optimization.

Public objects: SeqOpt(+Plot).
The search/optimization counterpart to ``protein_design``'s scoring classes: ``SeqOpt``
evolves multi-mutation variants of one wild-type toward a multi-objective Pareto front,
reusing a model-bound ``SeqMut`` as the fitness engine and ``ShapModel`` for per-generation
residue guidance. Gated behind the ``pro`` extra (imports SHAP).

See ``.claude/rules/pro-core-boundary.md`` for the pro gating, ``CONTEXT.md`` for domain
terms (SeqOpt directed-evolution vocabulary).
"""
from ._seqopt import SeqOpt
from ._seqopt_plot import SeqOptPlot

__all__ = [
    "SeqOpt",
    "SeqOptPlot",
]
