"""
Prediction: evaluate and deploy sequence-based prediction models.

Public objects: AAPred, AAPredPlot, ReliabilityModel, ReliabilityModelPlot, ModelEvaluator,
ModelEvaluatorPlot, bind_groups, audit_leakage.
Downstream of feature engineering (``CPP`` / ``CPPGrid`` produce ``df_feat`` and the feature
matrix ``X``): ``AAPred`` evaluates one or more scikit-learn models across metrics by
cross-validation and an optional held-out set (``eval``), and fits them for deployment, then
scores raw sequences at the whole-protein, domain, or residue-window level (``fit`` /
``predict``); ``AAPredPlot`` visualizes the evaluation table (``eval``) and per-sample
predictions (``predict``). ``ModelEvaluator`` adds a rigorous evaluation surface — repeated
stratified cross-validation with bootstrap confidence intervals (``run``) and paired model
comparison with a signed delta and a Wilcoxon significance test (``eval``) — visualized by
``ModelEvaluatorPlot``; ``bind_groups`` binds group labels (protein accession, family, or an
externally computed homology cluster) to any scikit-learn splitter, so dependent samples stay
within one fold of ``eval`` / ``run`` instead of leaking across them, while ``audit_leakage``
inspects a dataset and its folds after the fact and reports the leakage risks it can see
(duplicate sequences, a protein or group split across a fold, a feature tracking the label, a
skewed fold) as a plain table of findings. Complements ``explainable_ai.TreeModel`` (tree-ensemble feature
importance) — this subpackage owns the general evaluate-and-deploy path.

See ``.claude/rules/code-conventions.md`` for conventions and ``CONTEXT.md`` for domain terms.
"""
from ._aa_pred import AAPred
from ._aa_pred_plot import AAPredPlot
from ._reliability_model import ReliabilityModel
from ._reliability_model_plot import ReliabilityModelPlot
from ._model_evaluator import ModelEvaluator
from ._model_evaluator_plot import ModelEvaluatorPlot
from ._bind_groups import bind_groups
from ._audit_leakage import audit_leakage

__all__ = [
    "AAPred",
    "AAPredPlot",
    "ReliabilityModel",
    "ReliabilityModelPlot",
    "ModelEvaluator",
    "ModelEvaluatorPlot",
    "bind_groups",
    "audit_leakage",
]
