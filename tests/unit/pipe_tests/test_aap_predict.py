"""This script tests the aaanalysis.pipe.predict() golden pipeline."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import aaanalysis as aa
import aaanalysis.pipe as aap

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False


# Shared seeded fixture data (small DOM_GSEC slice)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat = aa.load_features().head(8)
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)


def _manual_predict(random_state=1, n_cv=5, list_model_classes=None):
    """The explicit primitive chain aap.predict is supposed to mirror byte-for-byte."""
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    tm = aa.TreeModel(list_model_classes=list_model_classes, random_state=random_state, verbose=False)
    tm.fit(X, labels=labels, n_cv=n_cv)
    df_eval = tm.eval(X, labels=labels, list_is_selected=[tm.is_selected_], n_cv=n_cv)
    return tm, df_eval


class TestPredict:
    """Positive and negative tests for aap.predict(), one parameter per test."""

    # Positive tests
    def test_returns_treemodel_and_df_eval(self):
        model, df_eval = aap.predict(df_feat, df_parts, labels, random_state=0)
        assert isinstance(model, aa.TreeModel)
        assert isinstance(df_eval, pd.DataFrame)
        assert len(df_eval) == 1

    def test_df_eval_has_metric_columns(self):
        _, df_eval = aap.predict(df_feat, df_parts, labels, random_state=0)
        for col in ("name", "accuracy", "precision", "recall", "f1"):
            assert col in df_eval.columns

    @settings(max_examples=3, deadline=None)
    @given(random_state=st.integers(min_value=0, max_value=50))
    def test_random_state_parameter(self, random_state):
        model, df_eval = aap.predict(df_feat, df_parts, labels, random_state=random_state)
        assert model.is_selected_ is not None
        assert np.isfinite(df_eval["accuracy"]).all()

    @settings(max_examples=3, deadline=None)
    @given(n_cv=st.integers(min_value=2, max_value=5))
    def test_n_cv_parameter(self, n_cv):
        _, df_eval = aap.predict(df_feat, df_parts, labels, n_cv=n_cv, random_state=0)
        assert np.isfinite(df_eval["accuracy"]).all()

    def test_list_model_classes_parameter(self):
        model, _ = aap.predict(df_feat, df_parts, labels,
                               list_model_classes=[RandomForestClassifier, ExtraTreesClassifier],
                               random_state=0)
        assert isinstance(model, aa.TreeModel)

    def test_n_jobs_parameter(self):
        for n_jobs in [None, 1]:
            _, df_eval = aap.predict(df_feat, df_parts, labels, n_jobs=n_jobs, random_state=0)
            assert np.isfinite(df_eval["accuracy"]).all()

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            model, _ = aap.predict(df_feat, df_parts, labels, verbose=verbose, random_state=0)
            assert isinstance(model, aa.TreeModel)

    # Negative tests
    def test_invalid_df_feat(self):
        with pytest.raises(ValueError):
            aap.predict("invalid", df_parts, labels)
        with pytest.raises(ValueError):
            aap.predict(pd.DataFrame({"not_feature": [1, 2]}), df_parts, labels)

    def test_invalid_df_parts(self):
        with pytest.raises(ValueError):
            aap.predict(df_feat, "invalid", labels)

    def test_invalid_n_cv(self):
        for n_cv in [1, 0, -1, "invalid"]:
            with pytest.raises(ValueError):
                aap.predict(df_feat, df_parts, labels, n_cv=n_cv)

    def test_invalid_random_state(self):
        for random_state in [-1, "invalid", 1.5]:
            with pytest.raises(ValueError):
                aap.predict(df_feat, df_parts, labels, random_state=random_state)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aap.predict(df_feat, df_parts, labels, verbose="invalid")

    def test_invalid_labels_length(self):
        with pytest.raises(ValueError):
            aap.predict(df_feat, df_parts, labels[:-1])


class TestPredictComplex:
    """Combinations and the byte-identical parity contract."""

    def test_byte_identical_to_manual_chain(self):
        model_m, df_eval_m = _manual_predict(random_state=1, n_cv=5)
        model_a, df_eval_a = aap.predict(df_feat, df_parts, labels, random_state=1, n_cv=5)
        assert df_eval_m.equals(df_eval_a)
        assert np.array_equal(np.asarray(model_m.is_selected_), np.asarray(model_a.is_selected_))
        assert np.array_equal(np.asarray(model_m.feat_importance), np.asarray(model_a.feat_importance))

    def test_byte_identical_with_explicit_models(self):
        models = [RandomForestClassifier, ExtraTreesClassifier]
        model_m, df_eval_m = _manual_predict(random_state=3, n_cv=4, list_model_classes=models)
        model_a, df_eval_a = aap.predict(df_feat, df_parts, labels, list_model_classes=models,
                                         random_state=3, n_cv=4)
        assert df_eval_m.equals(df_eval_a)

    def test_reproducible_same_seed(self):
        _, df_eval_1 = aap.predict(df_feat, df_parts, labels, random_state=7)
        _, df_eval_2 = aap.predict(df_feat, df_parts, labels, random_state=7)
        assert df_eval_1.equals(df_eval_2)

    def test_combined_valid_parameters(self):
        model, df_eval = aap.predict(df_feat, df_parts, labels,
                                     list_model_classes=[RandomForestClassifier],
                                     n_cv=3, random_state=2, n_jobs=1, verbose=False)
        assert isinstance(model, aa.TreeModel)
        assert np.isfinite(df_eval["accuracy"]).all()

    def test_combined_invalid_parameters(self):
        with pytest.raises(ValueError):
            aap.predict(df_feat, df_parts, labels, n_cv=-5, random_state="bad")
