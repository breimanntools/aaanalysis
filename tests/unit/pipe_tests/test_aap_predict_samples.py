"""This is a script to test the aaanalysis.pipe.predict_samples() golden pipeline."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import aaanalysis as aa
import aaanalysis.pipe as aap

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# Every test here runs cross-validated model fits (tens of seconds each), so the whole module is
# the `slow` tier: deselected from the fast PR unit matrix (-m "not slow"), run by coverage + nightly.
pytestmark = pytest.mark.slow

# Shared seeded fixture data (small DOM_GSEC slice)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat = aa.load_features().head(6)
_METRIC_COLS = ("balanced_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc")


class TestPredictSamples:
    """Positive and negative tests for aap.predict_samples(), one parameter per test."""

    # Positive tests
    def test_returns_triple(self):
        result = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=0)
        assert isinstance(result, tuple) and len(result) == 3
        predictors, figs, df_eval = result
        assert isinstance(predictors, dict)
        assert figs is None
        assert isinstance(df_eval, pd.DataFrame)

    def test_default_models_compared(self):
        predictors, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, random_state=0)
        assert len(df_eval) == 4
        assert len(predictors) == 4

    def test_df_eval_metric_columns(self):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=0)
        for m in _METRIC_COLS:
            assert f"{m}_mean" in df_eval.columns
            assert f"{m}_std" in df_eval.columns
        for col in ("feature_set", "model", "n_features", "is_shap_ready", "is_best"):
            assert col in df_eval.columns

    def test_single_df_feat_named(self):
        predictors, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=0)
        assert list(df_eval["feature_set"].unique()) == ["features"]
        assert ("features", "rf") in predictors

    def test_list_df_feat_auto_named(self):
        _, _, df_eval = aap.predict_samples([df_feat, df_feat.head(4)], df_seq, labels,
                                            models=["rf"], random_state=0)
        assert set(df_eval["feature_set"]) == {"feat1", "feat2"}

    def test_dict_df_feat_keeps_names(self):
        predictors, _, df_eval = aap.predict_samples({"a": df_feat, "b": df_feat.head(4)}, df_seq,
                                                     labels, models=["rf"], random_state=0)
        assert set(df_eval["feature_set"]) == {"a", "b"}
        assert ("a", "rf") in predictors and ("b", "rf") in predictors

    def test_models_string(self):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf", "svm"], random_state=0)
        assert set(df_eval["model"]) == {"rf", "svm"}

    def test_models_instance(self):
        predictors, _, _ = aap.predict_samples(df_feat, df_seq, labels,
                                               models=[RandomForestClassifier()], random_state=0)
        assert ("features", "RandomForestClassifier") in predictors

    def test_models_dict_keeps_names(self):
        predictors, _, df_eval = aap.predict_samples(df_feat, df_seq, labels,
                                                     models={"forest": RandomForestClassifier()},
                                                     random_state=0)
        assert list(df_eval["model"]) == ["forest"]
        assert ("features", "forest") in predictors

    @settings(max_examples=3, deadline=None)
    @given(random_state=st.integers(min_value=0, max_value=50))
    def test_random_state_parameter(self, random_state):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=random_state)
        assert np.isfinite(df_eval["balanced_accuracy_mean"]).all()

    @settings(max_examples=3, deadline=None)
    @given(n_cv=st.integers(min_value=2, max_value=5))
    def test_n_cv_parameter(self, n_cv):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], n_cv=n_cv, random_state=0)
        assert np.isfinite(df_eval["balanced_accuracy_mean"]).all()

    def test_n_jobs_parameter(self):
        for n_jobs in [None, 1]:
            _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"],
                                                n_jobs=n_jobs, random_state=0)
            assert np.isfinite(df_eval["balanced_accuracy_mean"]).all()

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            predictors, _, _ = aap.predict_samples(df_feat, df_seq, labels, models=["rf"],
                                                   verbose=verbose, random_state=0)
            assert len(predictors) == 1

    def test_is_best_exactly_one(self):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, random_state=0)
        assert int(df_eval["is_best"].sum()) == 1

    def test_is_shap_ready_flags_tree_models(self):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels,
                                            models={"rf": RandomForestClassifier(), "lr": LogisticRegression()},
                                            random_state=0)
        flags = dict(zip(df_eval["model"], df_eval["is_shap_ready"]))
        assert flags["rf"] is True or flags["rf"] == True  # tree-based -> feature_importances_
        assert not flags["lr"]

    # Negative tests
    def test_invalid_df_feat(self):
        with pytest.raises(ValueError):
            aap.predict_samples("invalid", df_seq, labels)
        with pytest.raises(ValueError):
            aap.predict_samples(pd.DataFrame({"not_feature": [1, 2]}), df_seq, labels)

    def test_invalid_df_seq(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, "invalid", labels)

    def test_invalid_labels_length(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels[:-1], models=["rf"])

    def test_invalid_n_cv(self):
        for n_cv in [1, 0, -1, "invalid"]:
            with pytest.raises(ValueError):
                aap.predict_samples(df_feat, df_seq, labels, models=["rf"], n_cv=n_cv)

    def test_invalid_random_state(self):
        for random_state in [-1, "invalid", 1.5]:
            with pytest.raises(ValueError):
                aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=random_state)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models=["rf"], verbose="invalid")

    def test_invalid_models_string(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models=["not_a_model"])

    def test_invalid_models_type(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models=[123])

    def test_empty_list_df_feat(self):
        with pytest.raises(ValueError):
            aap.predict_samples([], df_seq, labels, models=["rf"])

    def test_empty_models(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models=[])


class TestPredictSamplesComplex:
    """Combinations and grid / invariant contracts."""

    # Positive tests
    def test_grid_shape(self):
        predictors, _, df_eval = aap.predict_samples({"a": df_feat, "b": df_feat.head(4)}, df_seq, labels,
                                                     models=["rf", "log_reg"], random_state=2)
        assert df_eval.shape[0] == 4
        assert len(predictors) == 4
        assert int(df_eval["is_best"].sum()) == 1

    def test_returned_predictors_are_fitted(self):
        predictors, _, _ = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=2)
        est = predictors[("features", "rf")]
        sf = aa.SequenceFeature()
        X = sf.feature_matrix(features=df_feat["feature"], df_parts=sf.get_df_parts(df_seq=df_seq))
        preds = est.predict(X)
        assert len(preds) == len(labels)

    def test_reproducible_same_seed(self):
        _, _, e1 = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=7)
        _, _, e2 = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=7)
        assert e1.equals(e2)

    def test_mixed_str_and_instance_models(self):
        predictors, _, df_eval = aap.predict_samples(df_feat, df_seq, labels,
                                                     models=["rf", SVC(probability=True)], random_state=1)
        assert ("features", "rf") in predictors
        assert ("features", "SVC") in predictors

    def test_n_features_matches_feature_set_size(self):
        _, _, df_eval = aap.predict_samples({"six": df_feat, "four": df_feat.head(4)}, df_seq, labels,
                                            models=["rf"], random_state=1)
        sizes = dict(zip(df_eval["feature_set"], df_eval["n_features"]))
        assert sizes["six"] == 6 and sizes["four"] == 4

    def test_metrics_in_unit_range(self):
        _, _, df_eval = aap.predict_samples(df_feat, df_seq, labels, models=["rf"], random_state=1)
        for m in _METRIC_COLS:
            vals = df_eval[f"{m}_mean"].to_numpy()
            assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    # Negative tests
    def test_combined_invalid_parameters(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, n_cv=-5, random_state="bad")

    def test_invalid_df_feat_in_dict(self):
        with pytest.raises(ValueError):
            aap.predict_samples({"good": df_feat, "bad": pd.DataFrame({"x": [1]})}, df_seq, labels,
                                models=["rf"])

    def test_n_cv_exceeds_class_count(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models=["rf"], n_cv=999)

    def test_empty_dict_models(self):
        with pytest.raises(ValueError):
            aap.predict_samples(df_feat, df_seq, labels, models={})

    def test_invalid_labels_in_grid(self):
        with pytest.raises(ValueError):
            aap.predict_samples({"a": df_feat, "b": df_feat.head(4)}, df_seq, labels[:-2], models=["rf"])
