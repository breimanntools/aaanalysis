"""
This is a script for testing the  aa.AAclustPlot().center() method.
"""
import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import aaanalysis as aa
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


def call_aaclust_plot_center(X=None, labels=None, **kwargs):
    """"""
    n_samples, _ = X.shape
    n_clusters = max(2, int(n_samples / 2))  # Ensure at least 2 clusters
    if labels is None:
        labels = [i % n_clusters for i in range(n_samples)]
    aac_plot = aa.AAclustPlot()
    # Check if at least 3 unique values exist and non is nan
    all_vals = X.flatten()[1:].tolist()
    if not np.isnan(X).any() and len(set(all_vals)) > 2 and not np.asarray_chkfinite(X).any():
        ax, df_components = aac_plot.centers(X, labels=labels, **kwargs)
        assert isinstance(ax, plt.Axes) and isinstance(df_components, pd.DataFrame)
        plt.close()


class TestAAclustPlotCenter:
    """Test the 'center' method of the AAclustPlot class."""

    # Positive Tests
    @settings(max_examples=10, deadline=1000)
    @given(X=some.lists(some.lists(some.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                                   min_size=10, max_size=10), min_size=10, max_size=15).map(np.array))
    def test_valid_X(self, X):
        """Test the 'X' parameter with a valid 2D array."""
        call_aaclust_plot_center(X=X)

    @settings(max_examples=10, deadline=1000)
    @given(labels=some.lists(some.integers(min_value=0, max_value=10), min_size=5, max_size=20))
    def test_valid_labels(self, labels):
        """Test the 'labels' parameter with a valid list of integers."""
        X = np.random.rand(len(labels), 5)
        call_aaclust_plot_center(X=X, labels=labels)

    @settings(max_examples=10, deadline=1000)
    @given(component_x=some.integers(min_value=1, max_value=5))
    def test_valid_component_x(self, component_x):
        """Test the 'component_x' parameter with a valid integer."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, component_x=component_x)

    @settings(max_examples=10, deadline=1000)
    @given(component_y=some.integers(min_value=1, max_value=5))
    def test_valid_component_y(self, component_y):
        """Test the 'component_y' parameter with a valid integer."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, component_y=component_y)

    @settings(max_examples=10, deadline=1000)
    @given(figsize=some.tuples(some.integers(min_value=1, max_value=20), some.integers(min_value=1, max_value=20)))
    def test_valid_figsize(self, figsize):
        """Test the 'figsize' parameter with a valid tuple."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, figsize=figsize)


    @settings(max_examples=10, deadline=1000)
    @given(dot_alpha=some.floats(min_value=0.0, max_value=1.0))
    def test_valid_dot_alpha(self, dot_alpha):
        """Test the 'dot_alpha' parameter with a valid float."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, dot_alpha=dot_alpha)

    @settings(max_examples=10, deadline=1000)
    @given(dot_size=some.integers(min_value=1))
    def test_valid_dot_size(self, dot_size):
        """Test the 'dot_size' parameter with a valid integer."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, dot_size=dot_size)


    @settings(max_examples=3, deadline=1000)
    @given(legend=some.booleans())
    def test_valid_legend(self, legend):
        """Test the 'legend' parameter with a valid boolean."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, legend=legend)


    @settings(max_examples=10, deadline=1000)
    @given(palette=some.sampled_from(["viridis", "plasma", "inferno", "magma", "cividis"]))
    def test_valid_palette(self, palette):
        """Test the 'palette' parameter with a valid string."""
        X = np.random.rand(10, 5)
        call_aaclust_plot_center(X=X, palette=palette)

    # Negative Tests
    @settings(max_examples=10, deadline=1000)
    @given(X=some.just([]))
    def test_invalid_X_empty(self, X):
        """Test with an empty X array."""
        with pytest.raises(ValueError):
            call_aaclust_plot_center(X=np.array(X))

    @settings(max_examples=10, deadline=1000)
    @given(X=some.lists(some.just("not_a_number"), min_size=1, max_size=10))
    def test_invalid_X_non_numeric(self, X):
        """Test with a non-numeric X array."""
        with pytest.raises(ValueError):
            call_aaclust_plot_center(X=np.array(X))

    @settings(max_examples=10, deadline=1000)
    @given(X=some.lists(some.floats(min_value=-100, max_value=100), min_size=10, max_size=10).map(np.array))
    def test_invalid_X_one_dimensional(self, X):
        """Test with a 1-dimensional X array."""
        with pytest.raises(ValueError):
            call_aaclust_plot_center(X=np.array(X))

    @settings(max_examples=10, deadline=1000)
    @given(X=some.lists(some.lists(some.floats(allow_nan=True, allow_infinity=True),
                                   min_size=10, max_size=10), min_size=10, max_size=15).map(np.array))
    def test_invalid_X_with_nan_inf(self, X):
        """Test with NaN or infinity in X."""
        n_samples, _ = X.shape
        n_clusters = max(2, int(n_samples / 2))  # Ensure at least 2 clusters
        labels = [i % n_clusters for i in range(n_samples)]
        aac_plot = aa.AAclustPlot()
        # Check if at least 3 unique values exist and non is nan
        if np.isnan(X).any() or np.isinf(X).any():
            with pytest.raises(ValueError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    aac_plot.centers(X, labels=labels)

    def test_invalid_labels_empty(self):
        """Test with empty labels."""
        labels = []
        X = np.random.rand(10, 5)
        with pytest.raises(ValueError):
            aac_plot = aa.AAclustPlot()
            # Check if at least 3 unique values exist and non is nan
            ax, df_components = aac_plot.centers(X, labels=labels)
            assert isinstance(ax, plt.Axes) and isinstance(df_components, pd.DataFrame)
            plt.close()

    def test_invalid_labels_non_integer(self):
        """Test with non-integer labels."""
        labels = ["s", "d", "e"]
        X = np.random.rand(len(labels), 5)
        with pytest.raises(ValueError):
            aac_plot = aa.AAclustPlot()
            # Check if at least 3 unique values exist and non is nan
            ax, df_components = aac_plot.centers(X, labels=labels)
            assert isinstance(ax, plt.Axes) and isinstance(df_components, pd.DataFrame)
            plt.close()

    @settings(max_examples=10, deadline=1000)
    @given(component_x=some.integers(max_value=0))
    def test_invalid_component_x(self, component_x):
        """Test with an invalid component_x."""
        X = np.random.rand(10, 5)
        labels = [1, 4, 2, 4, 3]
        with pytest.raises(ValueError):
            aac_plot = aa.AAclustPlot()
            ax, df_components = aac_plot.centers(X, labels=labels, component_x=component_x)
            assert isinstance(ax, plt.Axes) and isinstance(df_components, pd.DataFrame)
            plt.close()

    @settings(max_examples=10, deadline=1000)
    @given(component_y=some.integers(max_value=0))
    def test_invalid_component_y(self, component_y):
        """Test with an invalid component_y."""
        X = np.random.rand(10, 5)
        labels = [1, 4, 2, 4, 3]
        with pytest.raises(ValueError):
            aac_plot = aa.AAclustPlot()
            ax, df_components = aac_plot.centers(X, labels=labels, component_y=component_y)
            assert isinstance(ax, plt.Axes) and isinstance(df_components, pd.DataFrame)
            plt.close()


class TestAAclustPlotCenterComplex:
    """Test combinations of parameters in the 'center' method of the AAclustPlot class."""

    @settings(max_examples=10, deadline=1000)
    @given(X=some.lists(some.lists(some.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                                   min_size=10, max_size=10), min_size=10, max_size=15).map(np.array),
           labels=some.lists(some.integers(min_value=0, max_value=10), min_size=5, max_size=20),
           component_x=some.integers(min_value=1, max_value=5),
           component_y=some.integers(min_value=1, max_value=5),
           dot_size=some.integers(min_value=1),
           palette=some.sampled_from(["viridis", "plasma", "inferno", "magma", "cividis"]))
    def test_complex_case(self, X, labels, component_x, component_y, dot_size, palette):
        """Test a combination of parameters."""
        call_aaclust_plot_center(X=X, labels=labels, component_x=component_x, component_y=component_y,
                                 dot_size=dot_size, palette=palette)
