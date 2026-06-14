"""
Sequence analysis: window sampling and amino-acid sequence logos.

Public objects: AAlogo(+Plot), AAWindowSampler.
``AAWindowSampler`` turns ``df_seq`` (from ``data_handling.load_dataset``) into
fixed-length windows / labels for ``feature_engineering.SequenceFeature``; ``AAlogo``
(+ ``AAlogoPlot``) renders position-specific sequence logos.

See ``.claude/rules/code-conventions.md`` for conventions, ``reproducibility.md`` for the
``seed`` contract, ``CONTEXT.md`` for domain terms (window sampling vocabulary).
"""
from ._aalogo import AAlogo
from ._aalogo_plot import AAlogoPlot
from ._aa_window_sampler import AAWindowSampler

__all__ = [
    "AAlogo",
    "AAlogoPlot",
    "AAWindowSampler",
]
