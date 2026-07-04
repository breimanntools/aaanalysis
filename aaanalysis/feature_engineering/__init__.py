"""
Feature engineering: scale-based amino-acid features for interpretable prediction.

Public objects: AAclust(+Plot), SequenceFeature, NumericalFeature, CPP, CPPGrid, CPPPlot.
``AAclust`` reduces a scale set (``df_scales`` from ``data_handling.load_scales``);
``SequenceFeature`` / ``NumericalFeature`` build ``df_parts``; ``CPP`` / ``CPPGrid`` derive
the interpretable feature matrix ``df_feat``, which feeds ``explainable_ai.TreeModel``,
``protein_engineering``, and the paired ``CPPPlot``.

See ``.claude/rules/code-conventions.md`` / ``frontend-backend.md`` for conventions,
``CONTEXT.md`` for domain terms (df_parts, part, split, scale; CPP split vocabulary).
"""
# DEV NOTE:
# Suppress matplotlib tight_layout warning caused by manual colorbar axes (fig.add_axes).
# This is intentional for precise layout control in CPPPlot and related visualizations.
# The layout is correct; the warning is noise in notebooks/docs.
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Axes that are not compatible with tight_layout.*"
)

from ._aaclust import AAclust
from ._aaclust_plot import AAclustPlot
from ._cpp_plot import CPPPlot
from ._cpp import CPP
from ._cpp_grid import CPPGrid
from ._sequence_feature import SequenceFeature
from ._numerical_feature import NumericalFeature

__all__ = [
    "AAclust",
    "AAclustPlot",
    "SequenceFeature",
    "NumericalFeature",
    "CPP",
    "CPPGrid",
    "CPPPlot",
]
