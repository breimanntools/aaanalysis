"""This script tests the TreeModel.eval() method."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
import hypothesis.extra.numpy as npst
aa.options["verbose"] = False


# Helper functions
def check_invalid_conditions(X, min_samples=3, min_unique_features=2, check_unique=True):
    n_samples, n_features = X.shape
    # Check for a minimum number of unique values in each feature
    unique_features_count = sum([len(set(X[:, col])) > 1 for col in range(n_features)])
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 3, f"n_features={n_features} should be >= 3"),
        (unique_features_count < min_unique_features, f"Not enough unique features: found {unique_features_count}, require at least {min_unique_features}")
                  ]
    if check_unique:
        n_unique_samples = len(set(map(tuple, X)))
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False

def create_labels(size):
    labels = np.array([1, 1, 0, 0] + list(np.random.choice([1, 0], size=size-4)))
    return labels

def create_list_is_selected(n_features=None, n_rows=1, n_arrays=2, d1=False):
    if d1:
        list_is_selected = [np.random.choice([True, False], size=n_features) for _ in range(n_arrays)]
    else:
        list_is_selected = [np.random.choice([True, False], size=(n_rows, n_features)) for _ in range(n_arrays)]
    return list_is_selected

ARGS = dict(n_cv=2, list_metrics=["accuracy"])

# Create valid X
df_seq = aa.load_dataset(name="DOM_GSEC")
df_feat = aa.load_features()
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)


# Main functions
class TestEval:
    """Test the eval method with positive test cases for each parameter."""

    # Positive tests
    @settings(max_examples=12, deadline=4000)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=20),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        tm = aa.TreeModel()
        X = X.round(0)
        size, n_feat = X.shape
        if size > 3 and n_feat > 3 and not check_invalid_conditions(X):
            # Ensure unique features
            for i in range(size):
                X[i, 0] = i + 1
            for i in range(n_feat):
                X[0, i] = i + 1
            labels = create_labels(X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat)
            df_eval = tm.eval(X, labels=labels, list_is_selected=list_is_selected, **ARGS)
            assert isinstance(df_eval, pd.DataFrame)

    def test_labels_parameter(self):
        for i in range(3):
            tm = aa.TreeModel()
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat)
            df_eval = tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected, **ARGS)
            assert isinstance(df_eval, pd.DataFrame)

    def test_list_is_selected_parameter(self):
        for i in range(3):
            tm = aa.TreeModel()
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat)
            df_eval = tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected, **ARGS)
            assert isinstance(df_eval, pd.DataFrame)

    def test_convert_1d_to_2d(self):
        tm = aa.TreeModel()
        size, n_feat = valid_X.shape
        labels = create_labels(valid_X.shape[0])
        list_is_selected = create_list_is_selected(n_features=n_feat, d1=True)
        df_eval = tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected, convert_1d_to_2d=True, **ARGS)
        assert isinstance(df_eval, pd.DataFrame)

    def test_names_feature_selections_parameter(self):
        for names in [["a", "b"], ["set a", "set a", "set c"], ["set"]]:
            tm = aa.TreeModel()
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat, n_arrays=len(names))
            df_eval = tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected,
                              names_feature_selections=names, **ARGS)
            assert isinstance(df_eval, pd.DataFrame)

    @settings(max_examples=3, deadline=5000)
    @given(metrics=st.lists(st.sampled_from(["accuracy", "f1", "precision", "recall", "roc_auc"]),
                            min_size=1, max_size=2))
    def test_list_metrics_parameter(self, metrics):
        tm = aa.TreeModel()
        size, n_feat = valid_X.shape
        list_is_selected = create_list_is_selected(n_features=n_feat)
        df_eval = tm.eval(valid_X, labels=valid_labels, list_metrics=metrics,
                          list_is_selected=list_is_selected, n_cv=2)

        assert isinstance(df_eval, pd.DataFrame)

    def test_n_cv_parameter(self):
        for i in [2, 7]:
            tm = aa.TreeModel()
            size, n_feat = valid_X.shape
            list_is_selected = create_list_is_selected(n_features=n_feat)
            df_eval = tm.eval(valid_X, labels=valid_labels, list_is_selected=list_is_selected,
                              n_cv=i, list_metrics=["accuracy"])
            assert isinstance(df_eval, pd.DataFrame)

    # Negative tests
    def test_invalid_X_parameter(self):
        tm = aa.TreeModel()
        invalid_X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        n_feat = invalid_X.shape[1]
        list_is_selected = create_list_is_selected(n_features=n_feat)
        with pytest.raises(ValueError):
            tm.eval(invalid_X, labels=valid_labels, list_is_selected=list_is_selected)

    def test_invalid_labels_parameter(self):
        tm = aa.TreeModel()
        n_feat = valid_X.shape[1]
        list_is_selected = create_list_is_selected(n_features=n_feat)
        invalid_labels = [1]  # Not matching the number of samples in valid_X
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=invalid_labels, list_is_selected=list_is_selected)

    def test_invalid_list_is_selected_parameter(self):
        tm = aa.TreeModel()
        invalid_list_is_selected = [np.array([True])]  # Not matching the number of features
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, list_is_selected=invalid_list_is_selected)

    def test_invalid_convert_1d_to_2d(self):
        tm = aa.TreeModel()
        size, n_feat = valid_X.shape
        labels = create_labels(valid_X.shape[0])
        list_is_selected = create_list_is_selected(n_features=n_feat, d1=True)
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected, convert_1d_to_2d=False, **ARGS)
        list_is_selected = create_list_is_selected(n_features=n_feat, d1=False)
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=labels, list_is_selected=list_is_selected, convert_1d_to_2d=True, **ARGS)

    def test_invalid_names_feature_selections_parameter(self):
        tm = aa.TreeModel()
        n_feat = valid_X.shape[1]
        list_is_selected = create_list_is_selected(n_features=n_feat)
        invalid_names = [None, "set"]  # One of the names is not a string
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, names_feature_selections=invalid_names,
                    list_is_selected=list_is_selected)

    def test_invalid_list_metrics_parameter(self):
        tm = aa.TreeModel()
        n_feat = valid_X.shape[1]
        list_is_selected = create_list_is_selected(n_features=n_feat)
        invalid_metrics = ["not_a_metric"]
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, list_metrics=invalid_metrics,
                    list_is_selected=list_is_selected)

    def test_invalid_n_cv_parameter(self):
        tm = aa.TreeModel()
        n_samp, n_feat = valid_X.shape
        list_is_selected = create_list_is_selected(n_features=n_feat)
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, n_cv=0, list_is_selected=list_is_selected)
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, n_cv="sdf", list_is_selected=list_is_selected)
        with pytest.raises(ValueError):
            tm.eval(valid_X, labels=valid_labels, n_cv=n_samp, list_is_selected=list_is_selected)
