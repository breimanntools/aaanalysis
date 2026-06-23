"""
Pro explainable AI: CPP-SHAP feature explanations (``pro`` extra).

Public objects: ShapModel.
Gated behind the ``pro`` extra (needs ``shap``). Wraps a fitted model (typically
``explainable_ai.TreeModel``) to compute SHAP values, which ``feature_engineering.CPPPlot``
renders as per-feature impact. Imported lazily from the top-level package and replaced
by an install-hint stub when ``shap`` is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms (explainability (CPP-SHAP) vocabulary).
"""
from ._shap_model import ShapModel

__all__ = [
    "ShapModel",
]
