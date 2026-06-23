"""This is a script for the sub-minute smoke gate: the critical CPP path end to end.

A single, tiny, seeded ``load -> parts -> CPP.run -> feature_matrix -> TreeModel.fit``
run that exercises the spine most changes touch. It is a fast local sanity check —
``pytest -m smoke -c tests/pytest.ini`` should finish in well under a minute and catch a
gross break (import error, broken CPP output, model that won't fit) before the full suite.

Marked ``smoke`` (registered in ``tests/pytest.ini``); deliberately NOT a separate CI job
(the full matrix already covers these paths). It reuses ``tests/_pipeline`` so the call
pattern stays defined in one place.
"""
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from tests import _pipeline

pytestmark = pytest.mark.smoke


def test_pipeline_load_cpp_treemodel():
    """The full spine runs and produces a model fitted on real CPP features."""
    art = _pipeline.build_pipeline(n=5, n_filter=30, n_scales=15)
    df_feat, X, labels = art["df_feat"], art["X"], art["labels"]

    # CPP produced a ranked, non-empty feature set and a matching design matrix.
    assert df_feat.shape[0] > 0
    assert X.shape[0] == len(labels)
    assert X.shape[1] == df_feat.shape[0]

    # df_feat carries the canonical contract columns (ties to the df_feat data dictionary).
    missing = [c for c in ut.LIST_COLS_FEAT if c not in df_feat.columns]
    assert not missing, f"df_feat missing contract columns: {missing}"

    # The downstream model fits on those features and reports feature importances,
    # which land back on df_feat as the contracted post-fit column.
    tm = aa.TreeModel(random_state=0)
    tm.fit(X, labels=labels, n_rounds=2, n_cv=3, n_feat_min=10, n_feat_max=20)
    df_feat_imp = tm.add_feat_importance(df_feat=df_feat)
    assert ut.COL_FEAT_IMPORT in df_feat_imp.columns


def test_pipeline_predict_proba_shape():
    """A fitted model predicts class probabilities with the expected shape."""
    art = _pipeline.build_pipeline(n=5, n_filter=30, n_scales=15)
    X, labels = art["X"], art["labels"]
    tm = aa.TreeModel(random_state=0)
    tm.fit(X, labels=labels, n_rounds=2, n_cv=3, n_feat_min=10, n_feat_max=20)
    pred, proba = tm.predict_proba(X)
    assert len(pred) == X.shape[0]
    assert len(proba) == X.shape[0]
