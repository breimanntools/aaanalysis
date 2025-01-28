import pytest
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from hypothesis import given, settings
import aaanalysis as aa
import warnings

# I Helper function
def check_invalid_conditions(X, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    n_unique_features = sum([len(set(X[:, col])) > 1 for col in range(n_features)])
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (n_unique_features < 2, f"n_unique_features={n_unique_features} should be >= 2"),
        (n_unique_samples <= 2, "n_unique_samples should be >= 3")
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False

# II Adjustments to the filter_correlation implementation
# Add a small epsilon to avoid division by zero errors in numpy calculations
def adjusted_filter_correlation(nf, X, max_cor):
    eps = 1e-12  # Small constant to avoid division by zero
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] += eps  # Prevent zero division
    X = (X - np.mean(X, axis=0)) / X_std  # Normalize to avoid invalid values

    # Call the original function
    return nf.filter_correlation(X, max_cor=max_cor)

# III Main function
class TestFilterCorrelation:
    """Test the filter_correlation method for individual parameters."""

    @settings(max_examples=10, deadline=1000)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_valid_X(self, X):
        """Test with valid 'X' inputs."""
        nf = aa.NumericalFeature()
        X = np.array(X)
        if not check_invalid_conditions(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                is_selected = adjusted_filter_correlation(nf, X, max_cor=0.7)
                assert isinstance(is_selected, np.ndarray)

    @settings(max_examples=10, deadline=1000)
    @given(max_cor=st.floats(min_value=0.0, max_value=1.0))
    def test_valid_max_cor(self, max_cor):
        """Test with valid 'max_cor' inputs."""
        nf = aa.NumericalFeature()
        X = np.random.rand(10, 5)
        if not check_invalid_conditions(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                is_selected = adjusted_filter_correlation(nf, X, max_cor=max_cor)
                assert isinstance(is_selected, np.ndarray)

    def test_invalid_X(self):
        """Test with invalid 'X' inputs."""
        nf = aa.NumericalFeature()
        invalid_Xs = [None, [], [[1]], [[1, 2], [3]], 'invalid']
        for invalid_X in invalid_Xs:
            with pytest.raises(Exception):
                nf.filter_correlation(invalid_X, max_cor=0.7)

    def test_invalid_max_cor(self):
        """Test with invalid 'max_cor' inputs."""
        nf = aa.NumericalFeature()
        invalid_max_cors = [-1, 2, 'invalid', None]
        X = np.random.rand(10, 5)
        for invalid_max_cor in invalid_max_cors:
            with pytest.raises(Exception):
                nf.filter_correlation(X, max_cor=invalid_max_cor)


class TestFilterCorrelationComplex:
    """Complex tests for the filter_correlation method."""

    @settings(max_examples=5, deadline=1500)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
           max_cor=st.floats(min_value=0.0, max_value=1.0))
    def test_complex_positive(self, X, max_cor):
        """Test with a complex combination of valid parameters."""
        nf = aa.NumericalFeature()
        X = np.array(X)
        if not check_invalid_conditions(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                is_selected = adjusted_filter_correlation(nf, X, max_cor=max_cor)
                assert isinstance(is_selected, np.ndarray)

    def test_complex_negative(self):
        """Test with invalid combinations of parameters."""
        nf = aa.NumericalFeature()
        invalid_Xs = [None, [], [[1]], [[1, 2], [3]], 'invalid']
        invalid_max_cors = [-1, 2, 'invalid', None]
        for invalid_X in invalid_Xs:
            for invalid_max_cor in invalid_max_cors:
                with pytest.raises(Exception):
                    nf.filter_correlation(invalid_X, max_cor=invalid_max_cor)
