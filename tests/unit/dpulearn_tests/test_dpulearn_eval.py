"""
This is a script for testing the aa.dPULearn.eval() method.
"""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pandas as pd
import aaanalysis as aa
import random

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


def create_labels(size):
    labels = np.array([0, 1] + list(np.random.choice([0, 1], size=size-2)))
    return labels


def check_invalid_conditions(X, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False


class TestdPULearnEval:
    """Test dPULearn.eval() method for each parameter individually."""

    # Positive tests
    @settings(deadline=350, max_examples=100)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X(self, X):
        """Test 'X' with valid inputs."""
        dpul = aa.dPULearn()
        size = X.shape[0]
        if size >= 2:
            is_invalid = check_invalid_conditions(X, min_samples=3)
            if not is_invalid:
                i = random.randint(2, 10)  # Random number between 2 and 10
                list_labels = [create_labels(X.shape[0]) for j in range(0, i)]
                df_eval = dpul.eval(X=X, list_labels=list_labels)
                assert isinstance(df_eval, pd.DataFrame)

    def test_list_labels(self):
        """Test 'list_labels' with valid inputs."""
        for i in range(3, 7):
            list_labels = [create_labels(i) for j in range(0, i)]
            X = np.random.rand(len(list_labels[0]), 100)
            dpul = aa.dPULearn()
            assert isinstance(dpul.eval(X, list_labels=list_labels), pd.DataFrame)

    @settings(max_examples=10)
    @given(names_datasets=st.lists(st.text(), min_size=2))
    def test_names_datasets(self, names_datasets):
        """Test 'names_datasets' with valid inputs."""
        X = np.random.rand(100, 5)
        names_datasets = [x.replace("_", "X").replace("$", "X") for x in names_datasets]
        list_labels = [np.random.randint(0, 3, size=100) for _ in names_datasets]
        dpul = aa.dPULearn()
        size = X.shape[0]
        if size >= 2:
            is_invalid = check_invalid_conditions(X, min_samples=3)
            if not is_invalid:
                assert isinstance(dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets), pd.DataFrame)

    @settings(max_examples=10)
    @given(X_neg=npst.arrays(dtype=np.float64, shape=(100, 5), elements=st.floats(allow_nan=False, allow_infinity=False)))
    def test_X_neg(self, X_neg):
        """Test 'X_neg' with valid inputs."""
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=100)]
        dpul = aa.dPULearn()
        size = X.shape[0]
        if size >= 2:
            is_invalid = check_invalid_conditions(X, min_samples=3)
            if not is_invalid:
                assert isinstance(dpul.eval(X, list_labels=list_labels, X_neg=X_neg), pd.DataFrame)

    @settings(max_examples=10, deadline=1000)
    @given(comp_kld=st.booleans())
    def test_comp_kld(self, comp_kld):
        """Test 'comp_kld' with valid inputs."""
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=100)]
        dpul = aa.dPULearn()
        size = X.shape[0]
        if size >= 2:
            is_invalid = check_invalid_conditions(X, min_samples=3)
            if not is_invalid:
                assert isinstance(dpul.eval(X, list_labels=list_labels, comp_kld=comp_kld), pd.DataFrame)

    # Negative tests
    @settings(max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(1, 5), elements=st.floats(allow_nan=False, allow_infinity=False)))
    def test_X_invalid_shape(self, X):
        """Test 'X' with invalid shape (not enough samples)."""
        dpul = aa.dPULearn()
        list_labels = [np.random.randint(0, 3, size=X.shape[0])]
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels)

    @settings(max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(100, 1), elements=st.floats(allow_nan=False, allow_infinity=False)))
    def test_X_invalid_features(self, X):
        """Test 'X' with invalid number of features (too few)."""
        dpul = aa.dPULearn()
        list_labels = [np.random.randint(0, 3, size=X.shape[0])]
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels)

    @settings(max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(100, 5), elements=st.just(np.nan)))
    def test_X_nan_values(self, X):
        """Test 'X' with NaN values."""
        dpul = aa.dPULearn()
        list_labels = [np.random.randint(0, 3, size=X.shape[0])]
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels)

    def test_list_labels_invalid_lengths(self):
        """Test 'list_labels' with invalid lengths."""
        dpul = aa.dPULearn()
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=99)]  # Invalid length
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels)

    def test_list_labels_invalid_values(self):
        """Test 'list_labels' with invalid values."""
        dpul = aa.dPULearn()
        X = np.random.rand(100, 5)
        list_labels = [np.array([3, 4, 5] * 33)]  # Invalid label values
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels)

    @settings(max_examples=10)
    @given(names_datasets=st.lists(st.text(), min_size=2))
    def test_names_datasets_invalid(self, names_datasets):
        """Test 'names_datasets' with invalid sizes."""
        names_datasets = [x.replace("_", "X").replace("$", "X") for x in names_datasets]
        dpul = aa.dPULearn()
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=100) for _ in range(len(names_datasets) - 1)]  # Mismatch in sizes
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels, names_datasets)

    @settings(max_examples=10)
    @given(
        X_neg=npst.arrays(dtype=np.float64, shape=(99, 5), elements=st.floats(allow_nan=False, allow_infinity=False)))
    def test_X_neg_invalid_shape(self, X_neg):
        """Test 'X_neg' with invalid shape."""
        dpul = aa.dPULearn()
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=100)]
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels, X_neg)

    def test_comp_kld_invalid_type(self):
        """Test 'comp_kld' with invalid type."""
        dpul = aa.dPULearn()
        X = np.random.rand(100, 5)
        list_labels = [np.random.randint(0, 3, size=100)]
        with pytest.raises(ValueError):
            dpul.eval(X, list_labels, comp_kld="invalid_type")  # Passing a string instead of


class TestdPULearnEvalComplex:
    """Complex test cases for dPULearn.eval() method combining multiple parameters."""

    @settings(max_examples=10, deadline=4000)
    @given(
        X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=100),
                      elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
        X_neg=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=100),
                          elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
    )
    def test_complex_valid_combinations(self, X, X_neg):
        """Test valid combinations of 'X', 'X_neg', and 'comp_kld'."""
        dpul = aa.dPULearn()
        n_samples = X.shape[0]
        list_labels = [create_labels(n_samples) for _ in range(random.randint(2, 10))]
        names_datasets = ["dataset_" + str(i) for i in range(len(list_labels))]
        # Check if conditions are valid before proceeding
        if not check_invalid_conditions(X, min_samples=3) and not check_invalid_conditions(X_neg, min_samples=3):
            if X.shape[1] == X_neg.shape[1]:
                df_eval = dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets, X_neg=X_neg)
                assert isinstance(df_eval, pd.DataFrame)

    @settings(max_examples=10)
    @given(
        X=npst.arrays(dtype=np.float64, shape=(10, 5), elements=st.floats(allow_nan=False, allow_infinity=False)),
        X_neg=npst.arrays(dtype=np.float64, shape=(9, 11), elements=st.floats(allow_nan=False, allow_infinity=False))
    )
    def test_complex_invalid_combinations(self, X, X_neg):
        """Test invalid combinations of 'X' and 'X_neg' with mismatched shapes."""
        dpul = aa.dPULearn()
        list_labels = [create_labels(X.shape[0]) for _ in range(random.randint(2, 10))]
        names_datasets = ["dataset_" + str(i) for i in range(len(list_labels))]
        # Check if number of features are not matching
        if X.shape[1] != X_neg.shape[1]:
            with pytest.raises(ValueError):
                dpul.eval(X, list_labels=list_labels, names_datasets=names_datasets, X_neg=X_neg)
