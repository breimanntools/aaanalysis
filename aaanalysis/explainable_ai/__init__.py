"""
Explainable AI: interpretable tree-based prediction over CPP features.

Public objects: TreeModel.
Consumes a CPP feature matrix (``df_feat``) from ``feature_engineering.CPP`` plus labels
(e.g. reliable negatives from ``pu_learning.dPULearn``) or embeddings ``X`` from
``data_handling.EmbeddingPreprocessor``; produces a fitted model whose outputs feed
``explainable_ai_pro.ShapModel``.

See ``.claude/rules/code-conventions.md`` for conventions, ``reproducibility.md`` for the
``random_state`` contract, ``CONTEXT.md`` for domain terms (explainability vocabulary).
"""
from ._tree_model import TreeModel

__all__ = [
    "TreeModel",
]
