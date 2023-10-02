"""
This is a script for testing the AAclust.fit() method.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
from hypothesis.extra import numpy as npst
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import aaanalysis as aa
import aaanalysis.utils as ut

import pytest
from sklearn.exceptions import ConvergenceWarning
import warnings


class TestAAclust:
    """Test AAclust class"""

    # Simple cases for initialization
    def test_initialization(self):
        """
        Test initialization of the AAclust class without any arguments.
        """
        aac = aa.AAclust()
        assert aac.model_class == KMeans
        assert aac._model_kwargs == dict(n_init="auto")
        assert aac._verbose == False
        assert aac.model == None

    # Property-based testing for positive cases
    @given(verbose=some.booleans())
    def test_verbose_init(self, verbose):
        """Test the 'verbose' parameter during initialization."""
        aac = aa.AAclust(verbose=verbose)
        assert aac._verbose == verbose

    @given(n_clusters=some.integers(min_value=1, max_value=10))
    def test_fit_n_clusters(self, n_clusters):
        """Test the 'n_clusters' parameter during the 'fit' method."""
        X = np.random.rand(100, 10)  # 100 samples, 10 features
        aac = aa.AAclust()
        aac.fit(X, n_clusters=n_clusters)
        assert len(set(aac.labels_)) == n_clusters

    @settings(deadline=1000, max_examples=5)
    @given(min_th=some.floats(min_value=0, max_value=0.3))
    def test_fit_min_th(self, min_th):
        """Test the 'min_th' parameter during the 'fit' method."""
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        # No direct assertions on 'min_th' effect, as its role is internal and complex.  # This test ensures no error is raised.
        aac.fit(X, min_th=min_th)

    # Property-based testing for negative cases
    @given(n_clusters=some.integers(max_value=0))
    def test_fit_invalid_n_clusters(self, n_clusters):
        """Test with an invalid 'n_clusters' value."""
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.fit(X, n_clusters=n_clusters)

    @given(min_th=some.floats(max_value=-0.1, min_value=-10))
    def test_fit_invalid_min_th(self, min_th):
        """Test with an invalid 'min_th' value."""
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.fit(X, min_th=min_th)

    # Additional tests
    def test_fit_results(self):
        """
        Ensure attributes are correctly set after 'fit' method.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X)
        assert isinstance(aac.n_clusters, int)
        assert isinstance(aac.labels_, np.ndarray)
        assert isinstance(aac.centers_, np.ndarray)
        assert isinstance(aac.center_labels_, np.ndarray)
        assert isinstance(aac.medoids_, np.ndarray)
        assert isinstance(aac.medoid_labels_, np.ndarray)
        assert isinstance(aac.is_medoid_, np.ndarray)

    @settings(max_examples=10)
    @given(names=some.lists(some.text(min_size=1, max_size=10), min_size=10, max_size=50))
    def test_fit_names(self, names):
        """
        Test the 'names' parameter during the 'fit' method.
        """
        X = np.random.rand(len(names), 10)
        aac = aa.AAclust()
        aac.fit(X, names=names, n_clusters=5)
        assert len(aac.medoid_names_) == aac.n_clusters

    def test_medoids_labels_match(self):
        """
        Ensure medoid labels match their respective cluster.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X, n_clusters=5)
        for i, label in enumerate(aac.medoid_labels_):
            assert aac.is_medoid_[np.where(aac.labels_ == label)[0]].sum() == 1  # Only one medoid per cluster.

    def test_fit_no_names(self):
        """
        Ensure 'medoid_names_' is None when 'names' is not provided.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X, n_clusters=5)
        assert aac.medoid_names_ is None

    def test_merge_metric(self):
        """
        Test the 'merge_metric' parameter with its different options.
        """
        valid_merge_metrics = ["correlation", None, "manhattan",  "euclidean", "cosine"]
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        for merge_metric in valid_merge_metrics:
            aac.fit(X, merge_metric=merge_metric)

    @settings(max_examples=10)
    @given(merge_metric=some.text(min_size=1).filter(lambda x: x not in ["correlation", None, "manhattan", "euclidean", "cosine"]))
    def test_invalid_merge_metric(self, merge_metric):
        """
        Test that using an invalid 'merge_metric' value raises an error.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        with pytest.raises(ValueError):  # Or whatever error you expect
            aac.fit(X, n_clusters=5, merge_metric=merge_metric)


class TestAAclustComplex:
    """Test AAclust class"""

    # Property-based testing for positive cases
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=12, max_side=100),
                         elements=some.floats(allow_nan=False, allow_infinity=False)),
           n_clusters=some.integers(min_value=2, max_value=5))
    def test_fit_with_n_clusters(self, X, n_clusters):
        """Test the fit method with a pre-defined number of clusters."""
        model = aa.AAclust(model_class=KMeans)
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))

        # Check for invalid X conditions that should raise a ValueError
        invalid_conditions = [(np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."),
            (n_samples < 3, f"n_samples={n_samples} should be >= 3"),
            (n_unique_samples < 3, f"n_unique_samples={n_unique_samples} should be >= 3"),
            (n_samples <= n_clusters, f"n_samples={n_samples} should be > n_clusters"),
            (n_unique_samples < n_clusters, f"n_unique_samples={n_unique_samples} should be >= n_clusters"),
            (n_features < 2, f"n_features={n_features} should be >= 2")]

        for condition, msg in invalid_conditions:
            if condition:
                with pytest.raises(ValueError):
                    model.fit(X, n_clusters=n_clusters)
                return  # exit early since an invalid condition was met

        # If no invalid conditions are met
        with warnings.catch_warnings(record=True) as w:
            model.fit(X, n_clusters=n_clusters)
            # Check if ConvergenceWarning was raised and handle
            convergence_warned = any([issubclass(warn.category, ConvergenceWarning) for warn in w])
            if convergence_warned:
                assert len(set(model.labels_)) <= n_clusters
                assert len(model.medoids_) <= n_clusters
            else:
                assert len(set(model.labels_)) == n_clusters
                assert len(model.medoids_) == n_clusters

    @settings(deadline=2000, max_examples=20)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50),
                         elements=some.floats(allow_nan=False, allow_infinity=False)))
    def test_fit_without_n_clusters(self, X):
        """Test the fit method without a pre-defined number of clusters."""
        model = aa.AAclust()
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        # Check for invalid X conditions that should raise a ValueError
        invalid_conditions = [(np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."),
            (n_samples < 3, f"n_samples={n_samples} should be >= 3"),
            (n_unique_samples < 3, f"n_unique_samples={n_unique_samples} should be >= 3"),
            (n_features < 2, f"n_features={n_features} should be >= 2")]
        for condition, msg in invalid_conditions:
            if condition:
                with pytest.raises(ValueError):
                    model.fit(X)
                return  # exit early since an invalid condition was met

        # If no invalid conditions are met
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X)
            assert isinstance(model.labels_, (list, np.ndarray))
            assert len(model.medoids_) == len(set(model.labels_))

    # Property-based testing for negative cases
    @settings(max_examples=20)
    @given(n_clusters=some.integers(max_value=0))
    def test_fit_invalid_n_clusters(self, n_clusters):
        """Test the fit method with an invalid number of clusters."""
        with pytest.raises(ValueError):
            model = aa.AAclust()
            model.fit([[1, 2], [2, 3]], n_clusters=n_clusters)

    # Additional Negative Tests
    def test_fit_empty_data(self):
        """Test the fit method with empty data."""
        with pytest.raises(ValueError):
            model = aa.AAclust()
            model.fit([])

    def test_fit_non_list_data(self):
        """Test the fit method with non-list data."""
        with pytest.raises(ValueError):
            model = aa.AAclust()
            model.fit("invalid")

    @given(merge_metric=some.text())
    @example(merge_metric="invalid_option")
    def test_fit_invalid_merge_metric(self, merge_metric):
        """Test the fit method with an invalid merge metric value."""
        if merge_metric not in ["euclidean", "pearson", None]:
            with pytest.raises(ValueError):
                model = aa.AAclust()
                model.fit([[1, 2], [2, 3]], merge_metric=merge_metric)