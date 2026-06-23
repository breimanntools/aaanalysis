"""
Positive-unlabeled learning: identify reliable negatives from unlabeled data.

Public objects: dPULearn(+Plot).
``dPULearn`` (deterministic PU learning) labels reliable negatives in a dataset that has
only positives and unlabeled samples; those labels feed ``explainable_ai.TreeModel``.
Paired with ``dPULearnPlot`` for visualization.

See ``.claude/rules/code-conventions.md`` for conventions, ``reproducibility.md`` for the
``random_state`` contract, ``CONTEXT.md`` for domain terms (prediction-task taxonomy).
"""
from ._dpulearn import dPULearn
from ._dpulearn_plot import dPULearnPlot

__all__ = [
    "dPULearn",
    "dPULearnPlot",
]