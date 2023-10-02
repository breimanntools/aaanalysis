"""
This is a script for testing the AAclust.evaluate() functions.
"""
from hypothesis import given, settings
import hypothesis.strategies as some
from hypothesis.extra import numpy as npst
import numpy as np
import aaanalysis as aa
import warnings
import pytest



class TestAAclustEvaluate:
    """Test evaluate function of the TARGET FUNCTION"""

    def test_X(self):
        """Test the 'X' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([1, 2, 1])
        assert isinstance(aa.AAclust().eval(X, labels), tuple)

    def test_invalid_X(self):
        """Test with an invalid 'X' value."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError):
            aa.AAclust().eval(X)

    def test_labels(self):
        """Test the 'labels' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([1, 2, 1])
        assert isinstance(aa.AAclust().eval(X, labels), tuple)

    def test_invalid_labels_dtype(self):
        """Test with invalid 'labels' datatype."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([1.5, 2.5, 1.5])  # Floating point labels, should be int
        with pytest.raises(ValueError):
            aa.AAclust().eval(X, labels)

    def test_labels_none(self):
        """Test with 'labels' parameter set to None."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError):
            aa.AAclust().eval(X)

    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50),
                         elements=some.floats(allow_nan=False, allow_infinity=False, min_value=1)),
        labels_length=some.integers(min_value=10, max_value=50))
    def test_matching_X_labels(self, X, labels_length):
        """Test evaluate with matching X and labels dimensions."""
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        labels = np.random.randint(1, 5, size=labels_length)  # Random label assignments
        n_classes = len(set(labels))

        # Check for invalid X conditions that should raise a ValueError
        invalid_conditions = [(len(set(labels)) == 1 or len(labels) != n_samples,
                               "Length of labels should match n_samples and not have all identical values."),
            (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."), (
            n_samples < 3 or n_unique_samples < 3,
            f"n_samples={n_samples} and n_unique_samples={n_unique_samples} should be >= 3"),
            (n_features < 2, f"n_features={n_features} should be >= 2"),
            (n_classes == 1 or n_classes >= n_samples, f"n_classes should be > 1 and < n_samples")]

        for condition, msg in invalid_conditions:
            if condition:
                with pytest.raises(ValueError):
                    aa.AAclust().eval(X, labels)
                return  # exit early since an invalid condition was met

        # If no invalid conditions are met
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = aa.AAclust().eval(X, labels)
            assert isinstance(result, tuple)


class TestAAclustEvaluateComplex:
    """Test evaluate function of the TARGET FUNCTION for Complex Cases"""

    @settings(deadline=1000, max_examples=25)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, max_side=50),
                         elements=some.floats(allow_nan=False, allow_infinity=False)),
           labels=npst.arrays(dtype=np.int32, shape=some.integers(min_value=1, max_value=100).map(lambda x: (x,)),
                              elements=some.integers(min_value=0, max_value=10)))
    def test_return_values_range(self, X, labels):
        """Test the range of the return values from the 'evaluate' function."""
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        n_classes = len(set(labels))

        # Check for invalid X conditions that should raise a ValueError
        invalid_conditions = [(np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."), (
            n_samples < 3 or n_unique_samples < 3,
            f"n_samples={n_samples} and n_unique_samples={n_unique_samples} should be >= 3"),
            (n_features < 2, f"n_features={n_features} should be >= 2"), (
            len(set(labels)) == 1 or len(labels) != n_samples,
            "Length of labels should match n_samples and not have all identical values."),
            (n_classes == 1 or n_classes >= n_samples, f"n_classes should be > 1 and < n_samples")]

        for condition, msg in invalid_conditions:
            if condition:
                with pytest.raises(ValueError):
                    aa.AAclust().eval(X, labels)
                return  # exit early since an invalid condition was met

        # If no invalid conditions are met
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            BIC, CH, SC = aa.AAclust().eval(X, labels)
            assert -np.inf <= BIC <= np.inf
            assert 0 <= CH
            assert -1 <= SC <= 1

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, max_side=50),
                         elements=some.floats(allow_nan=False, allow_infinity=False)),
           labels=npst.arrays(dtype=np.int32, shape=some.integers(min_value=1, max_value=100).map(lambda x: (x,)),
                              elements=some.integers(min_value=0, max_value=10)))
    def test_evaluate_large_dataset(self, X, labels):
        """Test the 'evaluate' function with large datasets."""
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        n_classes = len(set(labels))

        # Check for invalid X and labels conditions that should raise a ValueError
        invalid_conditions = [(np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."), (
            n_samples < 3 or n_unique_samples < 3,
            f"n_samples={n_samples} and n_unique_samples={n_unique_samples} should be >= 3"),
            (n_features < 2, f"n_features={n_features} should be >= 2"), (
            len(set(labels)) == 1 or len(labels) != n_samples,
            "Length of labels should match n_samples and not have all identical values."),
            (n_classes == 1 or n_classes >= n_samples, f"n_classes should be > 1 and < n_samples")]

        for condition, msg in invalid_conditions:
            if condition:
                with pytest.raises(ValueError):
                    aa.AAclust().eval(X, labels)
                return  # exit early since an invalid condition was met

        # If no invalid conditions are met
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert isinstance(aa.AAclust().eval(X, labels), tuple)

    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_small_X(self, X):
        """Test 'evaluate' with a very small 'X'."""
        with pytest.raises(ValueError):
            aa.AAclust().eval(X)
