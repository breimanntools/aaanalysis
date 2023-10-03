"""This is a script to test AAclust().comp_medoids()"""
from hypothesis import given, settings
import hypothesis.strategies as some
from hypothesis.extra import numpy as npst
import numpy as np
import aaanalysis as aa
import pytest
import warnings

# Helper function
def check_invalid_conditions(X, labels):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    n_classes = len(set(labels))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."),
        (n_samples < 3 or n_unique_samples < 3, f"n_samples={n_samples} and n_unique_samples={n_unique_samples} should be >= 3"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (len(set(labels)) == 1 or len(labels) != n_samples, "Length of labels should match n_samples and not have all identical values."),
        (n_classes == 1 or n_classes >= n_samples, f"n_classes should be > 1 and < n_samples")
    ]
    for condition, msg in conditions:
        if condition:
            return True
    return False


# Normal Cases
class TestCompCenters:
    """Test comp_medoids function of the AAclust class."""

    def test_valid_X(self):
        """Test a valid 'X' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([1, 2, 1])
        result_medoids, result_labels = aa.AAclust().comp_medoids(X, labels)
        assert isinstance(result_medoids, np.ndarray)
        assert isinstance(result_labels, np.ndarray)

    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
                         elements=some.floats(allow_nan=True, allow_infinity=True)))
    def test_invalid_X(self, X):
        """Test an invalid 'X' parameter."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().comp_medoids(X, labels)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().comp_medoids(X, labels)


    @given(labels=npst.arrays(dtype=np.int32, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=10, max_side=30),
                              elements=some.integers(min_value=0, max_value=30)))
    def test_valid_labels(self, labels):
        """Test a valid 'labels' parameter."""
        X = np.random.rand(labels.size, 2)
        n_unique_samples = len(set(map(tuple, X)))
        n_unique_labels = len(set(labels))
        if n_unique_samples > 2 and n_unique_labels > 1:
            result_medoids, result_labels = aa.AAclust().comp_medoids(X, labels)
            assert isinstance(result_medoids, np.ndarray)
            assert isinstance(result_labels, np.ndarray)

    @given(labels=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=3, max_side=10),
                              elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_invalid_labels_dtype(self, labels):
        """Test invalid 'labels' datatype."""
        X = np.random.rand(labels.size, 2)
        is_invalid  = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().comp_medoids(X, labels)

    def test_labels_none(self):
        """Test 'labels' parameter set to None."""
        X = np.random.rand(5, 2)
        with pytest.raises(ValueError):
            aa.AAclust().comp_medoids(X, None)

# Complex Cases
class TestCompCentersComplex:
    """Test comp_medoids function of the AAclust class for Complex Cases."""

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_combination_valid_parameters(self, X):
        """Test combination of valid parameters."""
        labels = np.random.randint(1, X.shape[0] + 1, size=X.shape[0])
        is_invalid = check_invalid_conditions(X, labels)
        if is_invalid:
            with pytest.raises(ValueError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    aa.AAclust().comp_medoids(X, labels)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result_medoids, result_labels = aa.AAclust().comp_medoids(X, labels)
                assert isinstance(result_medoids, np.ndarray)
                assert isinstance(result_labels, np.ndarray)

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, max_side=5),
                         elements=some.floats(allow_nan=True, allow_infinity=True)),
           labels=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1),
                              elements=some.floats()))
    def test_combination_invalid_parameters(self, X, labels):
        """Test combination of invalid parameters."""
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().comp_medoids(X, labels)
