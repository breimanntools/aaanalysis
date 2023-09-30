"""
This is a script for testing the AAclust.fit() method.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import aaanalysis as aa
import aaanalysis.utils as ut
import pytest


class TestAAclust:
    """Test AAclust class"""

    # Simple cases for initialization
    def test_initialization(self):
        """
        Test initialization of the AAclust class without any arguments.
        """
        aac = aa.AAclust()
        assert aac.model_class == AgglomerativeClustering
        assert aac._model_kwargs == {}
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

    @settings(deadline=1000)
    @given(min_th=some.floats(min_value=0, max_value=1))
    def test_fit_min_th(self, min_th):
        """Test the 'min_th' parameter during the 'fit' method."""
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X,
               min_th=min_th)  # No direct assertions on 'min_th' effect, as its role is internal and complex.  # This test ensures no error is raised.

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

    @given(names=some.lists(some.text(min_size=1, max_size=10), min_size=50, max_size=100))
    def test_fit_names(self, names):
        """
        Test the 'names' parameter during the 'fit' method.
        """
        X = np.random.rand(len(names), 10)
        aac = aa.AAclust()
        aac.fit(X, names=names)
        assert len(aac.medoid_names_) == aac.n_clusters

    # Complex scenarios
    def test_fit_complex_scenario(self):
        """
        Test a complex scenario for 'fit' method with multiple parameters.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X, n_clusters=5, min_th=0.5, on_center=True, merge_metric="euclidean")
        assert len(set(aac.labels_)) <= 5

    def test_medoids_labels_match(self):
        """
        Ensure medoid labels match their respective cluster.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X)

        for i, label in enumerate(aac.medoid_labels_):
            assert aac.is_medoid_[np.where(aac.labels_ == label)[0]].sum() == 1  # Only one medoid per cluster.

    def test_fit_no_names(self):
        """
        Ensure 'medoid_names_' is None when 'names' is not provided.
        """
        X = np.random.rand(100, 10)
        aac = aa.AAclust()
        aac.fit(X)
        assert aac.medoid_names_ is None

    def test_merge_metric(self):
        """
        Test the 'merge_metric' parameter with its different options.
        """
        X = np
        df = aa.load_dataset(name="SEQ_LOCATION", n=10, min_len=5, max_len=200, non_canonical_aa="remove")
        assert len(df) == 10 * 2
        assert all(5 <= len(seq) <= 200 for seq in df[ut.COL_SEQ])
        assert all(seq.isalpha() for seq in df[ut.COL_SEQ])


class TestAAclusComplex:
    """Test AAclust class"""

    # Property-based testing for positive cases
    @settings(max_examples=10)
    @given(X=some.lists(some.lists(some.floats(allow_nan=False, allow_infinity=False),
                                   min_size=1, max_size=10), min_size=1, max_size=100),
           n_clusters=some.integers(min_value=1, max_value=10))
    def test_fit_with_n_clusters(self, X, n_clusters):
        """Test the fit method with a pre-defined number of clusters."""
        model = aa.AAclust()
        model.fit(X, n_clusters=n_clusters)
        assert len(set(model.labels_)) == n_clusters
        assert len(model.medoids_) == n_clusters

    @settings(max_examples=3)
    @given(X=some.lists(some.lists(some.floats(allow_nan=False, allow_infinity=False),
                                   min_size=1, max_size=10), min_size=1, max_size=100))
    def test_fit_without_n_clusters(self, X):
        """Test the fit method without a pre-defined number of clusters."""
        model = aa.AAclust()
        model.fit(X)
        assert isinstance(model.labels_, list)
        assert len(model.medoids_) == len(set(model.labels_))

    @settings(max_examples=3)
    @given(X=some.lists(some.lists(some.floats(allow_nan=False, allow_infinity=False),
                                   min_size=1, max_size=10), min_size=1, max_size=100),
           min_th=some.floats(min_value=0.5, max_value=1))
    def test_fit_with_min_th(self, X, min_th):
        """Test the fit method with a threshold value."""
        model = aa.AAclust()
        model.fit(X, min_th=min_th, n_clusters=50)
        assert isinstance(model.labels_, list)

    # Property-based testing for negative cases
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

