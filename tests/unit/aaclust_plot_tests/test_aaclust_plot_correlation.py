import pytest
import pandas as pd
import numpy as np
import hypothesis.strategies as some
from hypothesis import given, settings
from matplotlib import pyplot as plt
import aaanalysis as aa


class TestAAclustPlotCorrelation:
    """Test the 'correlation' method of the AAclustPlot class."""

    # Positive Tests
    @settings(max_examples=25, deadline=1000)
    @given(df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_valid_df_corr(self, df_corr):
        """Test the 'df_corr' parameter with valid DataFrame."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        ax = aac_plot.correlation(df_corr=df_corr, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()

    # Test for 'labels' parameter
    @settings(max_examples=10, deadline=1000)
    @given(labels=some.lists(some.integers(), min_size=1, max_size=10))
    def test_valid_labels(self, labels):
        """Test the 'labels' parameter with valid list."""
        df_corr = pd.DataFrame(np.random.rand(len(labels), len(labels)))
        aac_plot = aa.AAclustPlot()
        ax = aac_plot.correlation(df_corr=df_corr, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()


    # Test 'cluster_x' parameter
    @settings(max_examples=10)
    @given(cluster_x=some.booleans(), df_corr=some.lists(
        some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_cluster_x(self, cluster_x, df_corr):
        """Test 'cluster_x' with boolean values."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, cluster_x=cluster_x), plt.Axes)
        plt.close()

    # Test 'method' parameter
    @settings(max_examples=10)
    @given(method=some.sampled_from(["single", "complete", "average", "weighted", "centroid", "median", "ward"]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_method(self, method, df_corr):
        """Test 'method' with valid clustering methods."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, method=method), plt.Axes)
        plt.close()


    # Test 'xtick_label_rotation' parameter
    @settings(max_examples=10)
    @given(xtick_label_rotation=some.integers(min_value=0, max_value=360),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_xtick_label_rotation(self, xtick_label_rotation, df_corr):
        """Test 'xtick_label_rotation' with integers."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, xtick_label_rotation=xtick_label_rotation), plt.Axes)
        plt.close()

    # Test 'ytick_label_rotation' parameter
    @settings(max_examples=10)
    @given(ytick_label_rotation=some.integers(min_value=0, max_value=360),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_ytick_label_rotation(self, ytick_label_rotation, df_corr):
        """Test 'ytick_label_rotation' with integers."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, ytick_label_rotation=ytick_label_rotation, labels=labels), plt.Axes)
        plt.close()

    # Test 'bar_position' parameter
    @settings(max_examples=10)
    @given(bar_position=some.sampled_from(["left", "right", "top", "bottom", None]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_position(self, bar_position, df_corr):
        """Test 'bar_position' with valid positions."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, bar_position=bar_position, labels=labels), plt.Axes)
        plt.close()

    # Test 'bar_colors' parameter
    @settings(max_examples=10)
    @given(bar_colors=some.sampled_from(["red", "green", "blue", ["red", "green", "blue"]]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_colors(self, bar_colors, df_corr):
        """Test 'bar_colors' with valid colors."""
        aac_plot = aa.AAclustPlot()
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, bar_colors=bar_colors), plt.Axes)

# TODO finish test and add negative tests TODO HERE