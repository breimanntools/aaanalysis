"""
Prediction: evaluate and deploy sequence-based prediction models.

Public objects: AAPred, AAPredPlot.
Downstream of feature engineering (``CPP`` / ``CPPGrid`` produce ``df_feat`` and the feature
matrix ``X``): ``AAPred`` evaluates one or more scikit-learn models across metrics by
cross-validation and an optional held-out set (``eval``), and fits them for deployment, then
scores raw sequences at the whole-protein, domain, or residue-window level (``fit`` /
``predict``); ``AAPredPlot`` visualizes the evaluation table (``eval``) and per-sample
predictions (``predict``). Complements ``explainable_ai.TreeModel`` (tree-ensemble
feature importance) — this subpackage owns the general evaluate-and-deploy path.

See ``.claude/rules/code-conventions.md`` for conventions and ``CONTEXT.md`` for domain terms.
"""
from ._aa_pred import AAPred
from ._aa_pred_plot import AAPredPlot

__all__ = [
    "AAPred",
    "AAPredPlot",
]
