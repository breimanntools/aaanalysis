"""This is a script to test the comp_correlation() method."""
from hypothesis import given, settings
import hypothesis.strategies as some
from hypothesis.extra import numpy as npst
import numpy as np
import aaanalysis as aa  # You might need to modify the import based on the actual module name.
import pytest
import warnings
import pandas as pd
from typing import List, Optional

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Helper function
def check_invalid_conditions(X, labels, min_samples=2, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (len(labels) != n_samples, "Length of labels should match n_samples."),
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False


# Normal Cases
class TestCompCorrelation:
    """Test comp_correlation function of the AAclust class."""

    @settings(deadline=500)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
                         elements=some.floats(allow_nan=True, allow_infinity=True)))
    def test_valid_X(self, X):
        """Test a valid 'X' parameter."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    aa.AAclust.comp_correlation(X, labels=labels)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result_df, labels_sorted = aa.AAclust.comp_correlation(X, labels=labels)
                assert isinstance(result_df, pd.DataFrame)
                assert len(result_df) == len(labels_sorted)
                assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=500)
    @given(X_ref=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
                             elements=some.floats(allow_nan=True, allow_infinity=True)))
    def test_invalid_X_ref(self, X_ref):
        """Test an invalid 'X_ref' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([1, 2, 1])
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels)

    @settings(deadline=300, max_examples=25)
    @given(labels_ref=npst.arrays(dtype=np.int32,
                                  shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=10, max_side=30),
                                  elements=some.integers(min_value=0, max_value=30)))
    def test_valid_labels_ref(self, labels_ref):
        """Test a valid 'labels_ref' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_ref = np.random.rand(labels_ref.size, 2)
        labels = np.array([1, 2, 1])
        is_invalid = check_invalid_conditions(X, labels) or check_invalid_conditions(X_ref, labels_ref, min_samples=1, check_unique=False)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref)
        else:
            result_df, labels_sorted = aa.AAclust.comp_correlation(X, labels=labels, X_ref=X_ref, labels_ref=labels_ref)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(labels_sorted)
            assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=500)
    @given(labels_ref=npst.arrays(dtype=np.float64,
                                  shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=5),
                                  elements=some.floats(allow_nan=True, allow_infinity=True)))
    def test_invalid_labels_ref_dtype(self, labels_ref):
        """Test invalid 'labels_ref' datatype."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, labels=np.array([1, 2, 1]), X_ref=X_ref, labels_ref=labels_ref)

    @settings(deadline=500)
    @given(names=some.lists(some.text(), min_size=1, max_size=5))
    def test_valid_names(self, names):
        """Test a valid 'names' parameter."""
        X = np.random.rand(len(names), 2)
        labels = np.arange(len(names))
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust.comp_correlation(X, labels=labels, names=names)
        else:
            warnings.simplefilter("ignore", RuntimeWarning)
            result_df, labels_sorted = aa.AAclust.comp_correlation(X, labels=labels, names=names)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(labels_sorted)
            assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=500)
    @given(names=some.lists(some.integers(), min_size=1, max_size=5))
    def test_invalid_names_dtype(self, names):
        """Test invalid 'names' datatype."""
        X = np.random.rand(len(names), 2)
        labels = np.arange(len(names))
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                aa.AAclust.comp_correlation(X, labels=labels, names=names)
        else:
            aa.AAclust.comp_correlation(X, labels=labels, names=names)
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, labels=labels, names={})
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, labels=labels, names=pd.DataFrame())

    @settings(deadline=500)
    @given(names_ref=some.lists(some.text(), min_size=1, max_size=5))
    def test_valid_names_ref(self, names_ref):
        """Test a valid 'names_ref' parameter."""
        X = np.random.rand(3, 2)
        X_ref = np.random.rand(len(names_ref), 2)
        labels = np.array([1, 2, 1])
        labels_ref = np.arange(len(names_ref))
        is_invalid = check_invalid_conditions(X, labels) or check_invalid_conditions(X_ref, labels_ref, min_samples=1, check_unique=False)
        if is_invalid:
            with pytest.raises(ValueError):
                aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref, names_ref=names_ref)
        else:
            result_df, labels_sorted = aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref, names_ref=names_ref)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(labels_sorted)
            assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=500)
    @given(names_ref=some.lists(some.integers(), min_size=1, max_size=5))
    def test_invalid_names_ref_dtype(self, names_ref):
        """Test invalid 'names_ref' datatype."""
        X = np.random.rand(3, 2)
        X_ref = np.random.rand(len(names_ref), 2)
        labels = np.array([1, 2, 1])
        labels_ref = np.arange(len(names_ref))
        is_invalid = check_invalid_conditions(X, labels) or check_invalid_conditions(X_ref, labels_ref, min_samples=1, check_unique=False)
        if is_invalid:
            with pytest.raises(ValueError):
                aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref, names_ref=names_ref)


# Complex Cases
class TestCompCorrelationComplex:
    """Test comp_correlation function of the AAclust class for Complex Cases."""

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_valid_parameters(self, X):
        """Test combination of valid parameters."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        X_ref = np.random.rand(X.shape[0], X.shape[1])
        labels_ref = np.random.randint(1, X_ref.shape[0] + 1, size=X_ref.shape[0])
        names = [f"Sample_{i}" for i in range(X.shape[0])]
        names_ref = [f"Ref_{i}" for i in range(X_ref.shape[0])]
        is_invalid = check_invalid_conditions(X, labels) or check_invalid_conditions(X_ref, labels_ref, min_samples=1, check_unique=False)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref, names=names, names_ref=names_ref)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result_df, labels_sorted = aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref, names=names, names_ref=names_ref)
                assert isinstance(result_df, pd.DataFrame)
                assert len(result_df) == len(labels_sorted)
                assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_with_missing_labels(self, X):
        """Test combination with missing 'labels' parameter."""
        X_ref = np.random.rand(X.shape[0], X.shape[1])
        names = [f"Sample_{i}" for i in range(X.shape[0])]
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, X_ref=X_ref, names=names)

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_with_incompatible_X_and_X_ref(self, X):
        """Test combination where 'X' and 'X_ref' have different number of features."""
        X_ref = np.random.rand(X.shape[0], X.shape[1] + 1)
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        labels_ref = np.random.randint(1, X_ref.shape[0] + 1, size=X_ref.shape[0])
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, X_ref=X_ref, labels=labels, labels_ref=labels_ref)

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_with_duplicate_names(self, X):
        """Test combination with duplicate 'names'."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        names = [f"Sample_0" for _ in range(X.shape[0])]
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                aa.AAclust.comp_correlation(X, labels=labels, names=names)
        else:
            result_df, labels_sorted = aa.AAclust.comp_correlation(X, labels=labels, names=names)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(labels_sorted)
            assert isinstance(labels_sorted, np.ndarray)

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_with_mismatched_labels_and_names(self, X):
        """Test combination where number of 'labels' and 'names' do not match."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0] - 1)
        names = [f"Sample_{i}" for i in range(X.shape[0])]
        with pytest.raises(ValueError):
            aa.AAclust.comp_correlation(X, labels=labels, names=names)

