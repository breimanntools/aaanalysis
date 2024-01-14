"""This script tests the TreeModel.predict_proba() method."""
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

# Create valid X
df_seq = aa.load_dataset(name="DOM_GSEC")
df_feat = aa.load_features()
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)

N_ROUNDS = 2
ARGS = dict(use_rfe=False, n_cv=2, n_rounds=N_ROUNDS)

class TestPredictProba:
    """
    Test the predict_proba method with positive and negative test cases for each parameter individually.
    """

    # Positive tests for X parameter
    @settings(max_examples=10, deadline=4000)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(allow_nan=False, allow_infinity=False)))
    def test_positive_X_parameter(self, X):
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
            tm.fit(X, labels=labels, **ARGS)
            pred, pred_std = tm.predict_proba(X)
            assert len(pred) == X.shape[0] and len(pred_std) == X.shape[0]

    def test_X(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        pred, pred_std = tm.predict_proba(valid_X)
        assert len(pred) == len(valid_labels) and len(pred_std) == valid_X.shape[0]

    # Negative tests for X parameter
    def test_negative_X_parameter(self):
        tm = aa.TreeModel()
        with pytest.raises(ValueError):
            tm.predict_proba(None)
        with pytest.raises(ValueError):
            tm.predict_proba({})
        with pytest.raises(ValueError):
            tm.predict_proba("str")
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            tm.predict_proba(None)
        with pytest.raises(ValueError):
            tm.predict_proba({})
        with pytest.raises(ValueError):
            tm.predict_proba("str")
