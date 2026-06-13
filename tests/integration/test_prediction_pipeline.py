"""This is a script to test the prediction seams (real components, no mocks).

Integration tier (ADR-0031). Negatives are composition failures, properties are
pipeline invariants / reproducibility.

Seams covered:
  6. CPP df_feat -> feature_matrix -> TreeModel.fit / .eval
  7. load_dataset (PU) -> dPULearn.fit -> carved labels (0)
  8. dPULearn carved labels -> TreeModel.fit
"""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from tests import _pipeline

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.integration


def _pu_feature_matrix(n=12, n_filter=15):
    """Build (X, labels) for the PU dataset, reusing the DOM_GSEC feature set.

    DOM_GSEC_PU is labelled 1 (positive) / 2 (unlabeled). We engineer features on
    the labelled DOM_GSEC split, then project the PU sequences onto the same
    feature set so the matrices are comparable.
    """
    base = _pipeline.build_pipeline(n=10, n_filter=n_filter)
    df_seq_pu = aa.load_dataset(name="DOM_GSEC_PU", n=n)
    labels_pu = df_seq_pu["label"].to_list()
    df_parts_pu = aa.SequenceFeature().get_df_parts(df_seq=df_seq_pu)
    X_pu = _pipeline.feature_matrix(df_feat=base["df_feat"], df_parts=df_parts_pu,
                                    df_scales=base["df_scales"])
    return np.asarray(X_pu), labels_pu


# ---------------------------------------------------------------------------
# Seam 6: CPP df_feat -> feature_matrix -> TreeModel
# ---------------------------------------------------------------------------
class TestFeaturesToTreeModel:
    """The CPP feature matrix trains and evaluates a TreeModel."""

    def test_fit_sets_importance_over_features(self, pipeline):
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            pipeline["X"], labels=pipeline["labels"], use_rfe=False, n_cv=2, n_rounds=2)
        assert len(tm.feat_importance) == np.asarray(pipeline["X"]).shape[1]
        assert len(tm.is_selected_) == 2          # one selection mask per round
        assert len(tm.list_models_) == 2          # one model bundle per round

    def test_eval_metrics_in_range(self, pipeline):
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            pipeline["X"], labels=pipeline["labels"], use_rfe=False, n_cv=2, n_rounds=2)
        df_eval = tm.eval(pipeline["X"], labels=pipeline["labels"],
                          list_is_selected=[tm.is_selected_], n_cv=2,
                          list_metrics=["accuracy", "f1"])
        num = df_eval.select_dtypes("number")
        assert ((num.values >= -1e-9) & (num.values <= 1 + 1e-9)).all()

    def test_same_seed_same_importance(self, pipeline):
        # Reproducibility property over the feature-matrix -> model seam.
        kws = dict(use_rfe=False, n_cv=2, n_rounds=2)
        tm_a = aa.TreeModel(verbose=False, random_state=0).fit(
            pipeline["X"], labels=pipeline["labels"], **kws)
        tm_b = aa.TreeModel(verbose=False, random_state=0).fit(
            pipeline["X"], labels=pipeline["labels"], **kws)
        assert np.allclose(tm_a.feat_importance, tm_b.feat_importance)
        assert np.array_equal(np.asarray(tm_a.is_selected_), np.asarray(tm_b.is_selected_))

    def test_label_length_mismatch_rejected(self, pipeline):
        # Composition failure: X rows and labels must align across the seam.
        labels_short = pipeline["labels"][:-1]
        with pytest.raises(ValueError, match="should contain"):
            aa.TreeModel(verbose=False, random_state=0).fit(
                pipeline["X"], labels=labels_short, use_rfe=False, n_cv=2, n_rounds=2)


# ---------------------------------------------------------------------------
# Seam 7: load_dataset (PU) -> dPULearn.fit -> carved labels
# ---------------------------------------------------------------------------
class TestPUToDPULearn:
    """dPULearn carves reliable negatives (0) from the PU feature matrix."""

    def test_carves_negatives(self):
        X_pu, labels_pu = _pu_feature_matrix()
        dpul = aa.dPULearn(verbose=False, random_state=0).fit(
            X=X_pu, labels=labels_pu, n_unl_to_neg=3)
        out = np.asarray(dpul.labels_)
        assert out.shape[0] == X_pu.shape[0]
        assert (out == 0).sum() == 3           # exactly the requested negatives
        assert set(np.unique(out)).issubset({0, 1, 2})

    def test_same_seed_same_carve(self):
        # Reproducibility property of the PU carve.
        X_pu, labels_pu = _pu_feature_matrix()
        a = aa.dPULearn(verbose=False, random_state=0).fit(X=X_pu, labels=labels_pu, n_unl_to_neg=3)
        b = aa.dPULearn(verbose=False, random_state=0).fit(X=X_pu, labels=labels_pu, n_unl_to_neg=3)
        assert np.array_equal(np.asarray(a.labels_), np.asarray(b.labels_))

    def test_standard_01_labels_rejected(self, pipeline):
        # Composition failure: dPULearn expects PU encoding {1, 2}, not {0, 1}.
        with pytest.raises(ValueError, match="does not contain required values"):
            aa.dPULearn(verbose=False, random_state=0).fit(
                X=pipeline["X"], labels=pipeline["labels"], n_unl_to_neg=2)


# ---------------------------------------------------------------------------
# Seam 8: dPULearn carved labels -> TreeModel
# ---------------------------------------------------------------------------
class TestDPULearnToTreeModel:
    """The negatives dPULearn identifies become a trainable {0, 1} set for TreeModel."""

    def test_carved_labels_train_a_model(self):
        X_pu, labels_pu = _pu_feature_matrix(n=20, n_filter=15)
        dpul = aa.dPULearn(verbose=False, random_state=0).fit(
            X=X_pu, labels=labels_pu, n_unl_to_neg=8)
        carved = np.asarray(dpul.labels_)
        # Keep only the resolved {positive=1, reliable-negative=0} samples.
        keep = carved != 2
        X_train, y_train = np.asarray(X_pu)[keep], carved[keep]
        assert set(np.unique(y_train)) == {0, 1}
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            X_train, labels=y_train.tolist(), use_rfe=False, n_cv=2, n_rounds=2)
        assert len(tm.feat_importance) == X_train.shape[1]

    @settings(max_examples=3, deadline=None)
    @given(n_unl_to_neg=some.integers(min_value=2, max_value=6))
    def test_carve_count_matches_request(self, n_unl_to_neg):
        # Property: the carve produces exactly n_unl_to_neg negatives, feeding a {0,1} set.
        X_pu, labels_pu = _pu_feature_matrix(n=20, n_filter=15)
        dpul = aa.dPULearn(verbose=False, random_state=0).fit(
            X=X_pu, labels=labels_pu, n_unl_to_neg=n_unl_to_neg)
        assert (np.asarray(dpul.labels_) == 0).sum() == n_unl_to_neg
