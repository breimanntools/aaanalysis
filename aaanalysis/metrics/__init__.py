"""
Metrics: scoring helpers for clustering, ranking, and site-localization quality.

Public objects: comp_auc_adjusted, comp_bic_score, comp_kld, comp_per_protein_ap,
comp_detection_metrics, comp_bootstrap_ci, comp_smooth_scores.
A cross-cutting subpackage (not a pipeline stage): the ``comp_*`` functions score the
outputs of ``feature_engineering.AAclust`` (BIC), model rankings from ``explainable_ai``,
and per-protein site detection.

See ``.claude/rules/code-conventions.md`` for conventions, ``CONTEXT.md`` for domain
terms (scoring vocabulary, site-localization metrics vocabulary).
"""
from ._metrics import (comp_auc_adjusted, comp_bic_score, comp_kld,
                       comp_per_protein_ap, comp_detection_metrics,
                       comp_bootstrap_ci, comp_smooth_scores)

__all__ = [
    "comp_auc_adjusted",
    "comp_bic_score", "comp_kld",
    "comp_per_protein_ap", "comp_detection_metrics",
    "comp_bootstrap_ci", "comp_smooth_scores",
]
