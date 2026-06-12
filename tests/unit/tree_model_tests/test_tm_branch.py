"""This is a script to test TreeModel branch arcs through the public API.

Focused branch-coverage tests: each test drives a public TreeModel method into a
specific validation guard or verbose/quiet arc that the existing per-method test
files leave un-hit. Everything goes through ``import aaanalysis as aa`` — no
private backend calls. Inputs are kept small (few features, few samples) so the
forest fits are fast.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Small, shared fixtures (read-only across tests)
N_FEAT = 8
df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
valid_labels = df_seq["label"].to_list()
_df_feat = aa.load_features(name="DOM_GSEC").head(N_FEAT)
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=_df_feat["feature"], df_parts=df_parts)

ARGS_FIT = dict(use_rfe=False, n_cv=2, n_rounds=2)


def fitted_tm(verbose=False):
    """Return a freshly fitted, seeded TreeModel."""
    tm = aa.TreeModel(random_state=0, verbose=verbose)
    tm.fit(valid_X, labels=valid_labels, **ARGS_FIT)
    return tm


def get_df_feat():
    return aa.load_features(name="DOM_GSEC").head(N_FEAT)


def make_list_is_selected(n_features=N_FEAT, n_rows=1, n_arrays=2):
    """Valid 2D boolean selection arrays, all features selected."""
    return [np.ones((n_rows, n_features), dtype=bool) for _ in range(n_arrays)]


# Main Test Classes
class TestFitBranch:
    """fit() validation guards reached only with specific arguments."""

    # Negative Tests
    def test_n_feat_min_greater_than_max(self):
        # L36: n_feat_min > n_feat_max raises, regardless of use_rfe.
        tm = aa.TreeModel(random_state=0, verbose=False)
        with pytest.raises(ValueError, match="n_feat_min"):
            tm.fit(valid_X, labels=valid_labels, n_cv=2, n_rounds=1,
                   n_feat_min=20, n_feat_max=5)

    def test_labels_three_classes_rejected(self):
        # check_models L15: check_match_labels_X requires exactly 2 unique labels.
        tm = aa.TreeModel(random_state=0, verbose=False)
        n = len(valid_labels)
        # Three distinct labels, exactly n entries (check_labels needs len match).
        labels3 = [(i % 3) for i in range(n)]
        assert len(set(labels3)) == 3 and len(labels3) == n
        with pytest.raises(ValueError, match="2 unique labels"):
            tm.fit(valid_X, labels=labels3, n_cv=2, n_rounds=1)

    # Positive Tests
    @settings(max_examples=3, deadline=None)
    @given(n_rounds=st.integers(min_value=1, max_value=2))
    def test_fit_verbose_true_runs(self, n_rounds):
        # Drives the verbose=True progress arcs in the fit/eval backends.
        tm = aa.TreeModel(random_state=0, verbose=True)
        tm.fit(valid_X, labels=valid_labels, use_rfe=False, n_cv=2, n_rounds=n_rounds)
        assert tm.feat_importance is not None


class TestEvalBranch:
    """eval() list_is_selected validation + verbose arcs (eval backend)."""

    # Positive Tests
    def test_eval_verbose_true(self):
        # tree_model_eval L18/L26/L36 -> the verbose=True arcs.
        tm = aa.TreeModel(random_state=0, verbose=True)
        lis = make_list_is_selected()
        df_eval = tm.eval(valid_X, labels=valid_labels, list_is_selected=lis,
                          n_cv=2, list_metrics=["accuracy"])
        assert isinstance(df_eval, pd.DataFrame)

    def test_eval_verbose_false(self):
        # tree_model_eval 18->20 / 26->29 / 36->39 -> the verbose=False (skip) arcs.
        tm = aa.TreeModel(random_state=0, verbose=False)
        lis = make_list_is_selected()
        df_eval = tm.eval(valid_X, labels=valid_labels, list_is_selected=lis,
                          n_cv=2, list_metrics=["accuracy"])
        assert isinstance(df_eval, pd.DataFrame)

    # Negative Tests
    def test_convert_1d_to_2d_non_bool_array(self):
        # L74: convert_1d_to_2d=True, but a 1D array is not boolean dtype.
        tm = aa.TreeModel(random_state=0, verbose=False)
        bad = [np.arange(N_FEAT)]  # int dtype, 1D
        with pytest.raises(ValueError, match="not a boolean array"):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=bad,
                    convert_1d_to_2d=True, n_cv=2, list_metrics=["accuracy"])

    def test_convert_1d_to_2d_size_mismatch(self):
        # L76: convert_1d_to_2d=True, 1D bool array of the wrong length.
        tm = aa.TreeModel(random_state=0, verbose=False)
        bad = [np.ones(N_FEAT - 1, dtype=bool)]
        with pytest.raises(ValueError, match="does not match"):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=bad,
                    convert_1d_to_2d=True, n_cv=2, list_metrics=["accuracy"])

    def test_element_not_ndarray(self):
        # L81: an element of list_is_selected is not a numpy array (a Python list).
        tm = aa.TreeModel(random_state=0, verbose=False)
        bad = [[[True] * N_FEAT]]  # list, not ndarray
        with pytest.raises(ValueError, match="not a numpy array"):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=bad,
                    n_cv=2, list_metrics=["accuracy"])

    def test_element_not_bool_dtype(self):
        # L83: a 2D element is not boolean dtype.
        tm = aa.TreeModel(random_state=0, verbose=False)
        bad = [np.ones((1, N_FEAT), dtype=int)]
        with pytest.raises(ValueError, match="not a boolean array"):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=bad,
                    n_cv=2, list_metrics=["accuracy"])

    def test_element_shape_mismatch(self):
        # L87: a 2D boolean element has the wrong number of feature columns.
        tm = aa.TreeModel(random_state=0, verbose=False)
        bad = [np.ones((1, N_FEAT + 3), dtype=bool)]
        with pytest.raises(ValueError, match="does not match the number of features"):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=bad,
                    n_cv=2, list_metrics=["accuracy"])


class TestPredictProbaBranch:
    """predict_proba() feature-count guard."""

    # Negative Tests
    def test_X_feature_count_mismatch(self):
        # check_models L27: X passed to predict_proba has a different feature count
        # than the fitted is_selected_ masks.
        tm = fitted_tm()
        X_wrong = valid_X[:, : N_FEAT - 2]  # fewer features than fitted
        with pytest.raises(ValueError, match="does not match"):
            tm.predict_proba(X_wrong)


class TestAddFeatImportanceBranch:
    """add_feat_importance() guards on fitted state and pre-existing columns."""

    # Negative Tests
    def test_unfitted_feat_importance_none(self):
        # L96: add_feat_importance before fit -> feat_importance is None.
        tm = aa.TreeModel(random_state=0, verbose=False)
        with pytest.raises(ValueError, match="feat_importance"):
            tm.add_feat_importance(df_feat=get_df_feat())

    def test_df_feat_length_mismatch(self):
        # L103/L105: df_feat row count differs from fitted feat_importance length.
        tm = fitted_tm()
        df_short = get_df_feat().head(N_FEAT - 3)
        with pytest.raises(ValueError, match="Mismatch of number of features"):
            tm.add_feat_importance(df_feat=df_short)

    def test_std_column_already_present(self):
        # L111: feat_importance_std column already present (import col absent), drop=False.
        tm = fitted_tm()
        df = get_df_feat()
        df = df[[c for c in df.columns
                 if c not in ["feat_importance", "feat_importance_std"]]].copy()
        df["feat_importance_std"] = 0.0
        with pytest.raises(ValueError, match="feat_importance_std"):
            tm.add_feat_importance(df_feat=df, drop=False)
