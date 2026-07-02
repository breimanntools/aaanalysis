"""
Pro explainable AI: CPP-SHAP feature explanations (``pro`` extra).

Public objects: ShapModel, shap_to_feat_imp.
Gated behind the ``pro`` extra (needs ``shap``). ``ShapModel`` wraps a fitted model (typically
``explainable_ai.TreeModel``) to compute SHAP values, which ``feature_engineering.CPPPlot``
renders as per-feature impact; ``shap_to_feat_imp`` converts a per-sample SHAP vector into signed
feature impact / absolute importance. Imported lazily from the top-level package and replaced by an
install-hint stub when ``shap`` is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms (explainability (CPP-SHAP) vocabulary).
"""
from ._shap_model import ShapModel
from ._shap_model_plot import shap_to_feat_imp

# NOTE: shap_to_feat_imp is intentionally NOT yet re-exported from the top-level
# aaanalysis namespace.
__all__ = [
    "ShapModel",
    "shap_to_feat_imp",
]
