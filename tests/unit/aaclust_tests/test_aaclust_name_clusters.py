"""This is a script to test AAclust().name_clusters()"""
from hypothesis import given, settings
import hypothesis.strategies as some
from hypothesis.extra import numpy as npst
import numpy as np
import aaanalysis as aa
import warnings
import pytest


class TestNameClusters:
    """Test name_clusters function of the TARGET FUNCTION"""

    def test_valid_X(self):
        """Test a valid 'X' parameter."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = [2, 3, 2]
        names = ['scale'+str(i) for i in range(X.shape[0])]
        result = aa.AAclust().name_clusters(X, labels, names)
        assert isinstance(result, list)

    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_invalid_X(self, X):
        """Test an invalid 'X' parameter."""
        labels = np.random.randint(1, 5, size=X.shape[0])
        names = ['scale'+str(i) for i in range(X.shape[0])]
        with pytest.raises(ValueError):
            aa.AAclust().name_clusters(X, labels, names)

    @given(labels=npst.arrays(dtype=np.int32, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=2),
                              elements=some.integers(min_value=0, max_value=10)))
    def test_valid_labels(self, labels):
        """Test a valid 'labels' parameter."""
        X = np.random.rand(labels.size, 2)
        n_samples, n_feature = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        names = ['scale' + str(i) for i in range(labels.size)]
        if n_samples < 2 or n_feature < 2 or n_unique_samples < 3:
            with pytest.raises(ValueError):
                aa.AAclust().name_clusters(X, labels, names)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = aa.AAclust().name_clusters(X, labels, names)
                assert isinstance(result, list)

    @given(labels=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=2),
                              elements=some.floats()))
    def test_invalid_labels_dtype(self, labels):
        """Test invalid 'labels' datatype."""
        X = np.random.rand(labels.size, 2)
        names = ['scale'+str(i) for i in range(labels.size)]
        with pytest.raises(ValueError):
            aa.AAclust().name_clusters(X, labels, names)

    def test_labels_none(self):
        """Test 'labels' parameter set to None."""
        X = np.random.rand(5, 2)
        names = ['scale'+str(i) for i in range(5)]
        with pytest.raises(ValueError):
            aa.AAclust().name_clusters(X, None, names)

    @given(names=some.lists(elements=some.text(), min_size=1))
    def test_valid_names(self, names):
        """Test a valid 'names' parameter."""
        X = np.random.rand(len(names), 2)
        labels = np.random.randint(1, 5, size=len(names))
        n_samples, n_feature = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        if n_samples < 2 or n_feature < 2 or n_unique_samples < 3:
            with pytest.raises(ValueError):
                aa.AAclust().name_clusters(X, labels, names)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = aa.AAclust().name_clusters(X, labels, names)
                assert isinstance(result, list)

    @given(names=some.lists(elements=some.text(), min_size=1))
    def test_invalid_names_length(self, names):
        """Test 'names' with a length not matching 'X'."""
        X = np.random.rand(len(names) + 1, 2)
        labels = np.random.randint(1, 5, size=len(names) + 1)
        with pytest.raises(ValueError):
            aa.AAclust().name_clusters(X, labels, names)

    def test_names_none(self):
        """Test 'names' parameter set to None."""
        X = np.random.rand(5, 2)
        labels = np.random.randint(1, 5, size=5)
        with pytest.raises(ValueError):
            aa.AAclust().name_clusters(X, labels, None)


class TestNameClustersComplex:
    """Test name_clusters function of the TARGET FUNCTION for Complex Cases"""

    @settings(deadline=1000, max_examples=5)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(allow_nan=False, allow_infinity=False)),
           labels=npst.arrays(dtype=np.int32, shape=npst.array_shapes(min_dims=1, max_dims=1),
                              elements=some.integers(min_value=0, max_value=10)))
    def test_combination_valid_parameters(self, X, labels):
        """Test combination of valid parameters."""
        names = ['scale'+str(i) for i in range(X.shape[0])]
        n_samples, n_feature = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        n_unique_labels = len(set(labels))
        if n_samples < 2 or n_feature < 2 or n_unique_samples < 3 or len(labels) != n_samples or n_unique_samples < 2:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().name_clusters(X, labels, names)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = aa.AAclust().name_clusters(X, labels, names)
                assert isinstance(result, list)

    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, max_side=50),
                         elements=some.floats(allow_nan=True, allow_infinity=True)))
    def test_combination_invalid_parameters(self, X):
        """Test combination of invalid parameters."""
        labels = np.random.randint(1, 5, size=X.shape[0])
        names = ['scale'+str(i) for i in range(X.shape[0])]
        n_samples, n_feature = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        if n_samples < 2 or n_feature < 2 or n_unique_samples < 3:
            with pytest.raises(ValueError):
                warnings.simplefilter("ignore", RuntimeWarning)
                aa.AAclust().name_clusters(X, labels, names)
