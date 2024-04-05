import pytest
import pandas as pd
import numpy as np
import hypothesis.strategies as some
from hypothesis import given, settings
from matplotlib import pyplot as plt
import aaanalysis as aa


# Helper function
def adjust_vmin_vmax(vmin=None, vmax=None):
    if vmin is not None:
        vmin = -10000 if vmin < -10000 else vmin
    if vmax is not None:
        vmax = 10000 if vmax > 10000 else vmax
    return vmin, vmax


def call_aaclust_plot_correlation(df_corr=None, **kwargs):
    """"""
    aac_plot = aa.AAclustPlot(verbose=False)
    df_corr = df_corr.fillna(1)
    n_samples, n_clusters = df_corr.shape
    labels = [i % n_clusters for i in range(n_samples)]
    # Check if not all values are the same
    all_vals = df_corr.to_numpy().flatten()[1:].tolist()
    if not df_corr.isna().any().any() and len(set(all_vals)) != 1:
        assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, **kwargs), plt.Axes)
        plt.close()


class TestAAclustPlotCorrelation:
    """Test the 'correlation' method of the AAclustPlot class."""

    # Positive Tests
    @settings(max_examples=10, deadline=2500)
    @given(df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_valid_df_corr(self, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr)

    @settings(max_examples=10, deadline=1000)
    @given(labels=some.lists(some.integers(), min_size=1, max_size=10))
    def test_valid_labels(self, labels):
        # Labels should contain more than one distinct value
        if len(set(labels)) > 1:
            df_corr = pd.DataFrame(np.random.rand(len(labels), len(labels)))
            aac_plot = aa.AAclustPlot(verbose=False)
            ax = aac_plot.correlation(df_corr=df_corr, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()


    @settings(max_examples=10, deadline=1250)
    @given(labels=some.lists(some.integers(), min_size=10, max_size=20))
    def test_valid_labels_ref(self, labels):
        # Labels should contain more than one distinct value
        if len(set(labels)) > 10:
            labels_ref = list(sorted(set(labels)))
            df_corr = pd.DataFrame(np.random.rand(len(labels), len(labels_ref)))
            aac_plot = aa.AAclustPlot(verbose=False)
            ax = aac_plot.correlation(df_corr=df_corr, labels=labels)
            assert isinstance(ax, plt.Axes)
            plt.close()
            aac_plot = aa.AAclustPlot(verbose=False)
            ax = aac_plot.correlation(df_corr=df_corr, labels=labels, labels_ref=labels_ref)
            assert isinstance(ax, plt.Axes)
            plt.close()


    @settings(max_examples=10, deadline=1000)
    @given(cluster_x=some.booleans(), df_corr=some.lists(
        some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_cluster_x(self, cluster_x, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, cluster_x=cluster_x)

    @settings(max_examples=10, deadline=1000)
    @given(method=some.sampled_from(["single", "complete", "average", "weighted", "centroid", "median", "ward"]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_method(self, method, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, method=method)

    @settings(max_examples=10, deadline=1000)
    @given(xtick_label_rotation=some.integers(min_value=0, max_value=360),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_xtick_label_rotation(self, xtick_label_rotation, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, xtick_label_rotation=xtick_label_rotation)

    @settings(max_examples=10, deadline=1000)
    @given(ytick_label_rotation=some.integers(min_value=0, max_value=360),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_ytick_label_rotation(self, ytick_label_rotation, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, ytick_label_rotation=ytick_label_rotation)

    @settings(max_examples=10, deadline=1000)
    @given(bar_position=some.sampled_from(["left", "right", "top", "bottom"]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_position(self, bar_position, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, bar_position=bar_position)

    @settings(max_examples=10, deadline=1000)
    @given(bar_colors=some.sampled_from(["red", "green", "blue", ["red", "green", "blue"]]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, width=16, allow_nan=False, allow_infinity=False),
                   min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_colors(self, bar_colors, df_corr):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = df_corr.fillna(1)
        n_samples, n_clusters = df_corr.shape
        labels = [i % n_clusters for i in range(n_samples)]
        n_unique = len(set(labels))
        if type(bar_colors) is list:
            bar_colors *= n_unique
        all_vals = df_corr.to_numpy().flatten()[1:].tolist()
        if not df_corr.isna().any().any() and len(set(all_vals)) != 1 and n_samples > n_clusters:
            assert isinstance(aac_plot.correlation(df_corr=df_corr, labels=labels, bar_colors=bar_colors), plt.Axes)

    @settings(max_examples=10, deadline=1000)
    @given(bar_width_x=some.floats(min_value=0.01, max_value=1.0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_width_x(self, bar_width_x, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, bar_width_x=bar_width_x)

    @settings(max_examples=10, deadline=1000)
    @given(bar_spacing_x=some.floats(min_value=0.01, max_value=1.0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_spacing_x(self, bar_spacing_x, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, bar_spacing_x=bar_spacing_x)

    @settings(max_examples=10, deadline=1000)
    @given(bar_width_y=some.floats(min_value=0.01, max_value=1.0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_width_y(self, bar_width_y, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, bar_width_y=bar_width_y)

    @settings(max_examples=10, deadline=1000)
    @given(bar_spacing_y=some.floats(min_value=0.01, max_value=1.0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_bar_spacing_y(self, bar_spacing_y, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, bar_spacing_y=bar_spacing_y)

    @settings(max_examples=10, deadline=1000)
    @given(vmin=some.floats(min_value=-1.0, max_value=0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_vmin(self, vmin, df_corr):
        vmin, vmax = adjust_vmin_vmax(vmin=vmin)
        call_aaclust_plot_correlation(df_corr=df_corr, vmin=vmin)

    @settings(max_examples=10, deadline=1000)
    @given(vmax=some.floats(min_value=0, max_value=1.0),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_vmax(self, vmax, df_corr):
        vmin, vmax = adjust_vmin_vmax(vmax=vmax)
        call_aaclust_plot_correlation(df_corr=df_corr, vmax=vmax)

    @settings(max_examples=10, deadline=1000)
    @given(cmap=some.sampled_from(["viridis", "plasma", "inferno", "magma", "cividis"]),
           df_corr=some.lists(some.lists(some.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                                         min_size=3, max_size=10), min_size=3, max_size=10).map(pd.DataFrame))
    def test_cmap(self, cmap, df_corr):
        call_aaclust_plot_correlation(df_corr=df_corr, cmap=cmap)

    def test_kwargs_heatmap(self):
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        call_aaclust_plot_correlation(df_corr=df_corr, kwargs_heatmap=dict(linecolor="black"))

    # Negative Tests
    def test_invalid_df_corr_type(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr="not_a_dataframe", labels=[1, 2, 3])

    def test_df_corr_with_nans(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame([[np.nan, 1], [1, 1]])
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2])

    def test_labels_incorrect_length(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2])

    @settings(max_examples=10, deadline=1000)
    @given(labels=some.lists(some.integers(), min_size=10, max_size=20))
    def test_invalid_labels_ref(self, labels):
        # Labels should contain more than one distinct value
        if len(set(labels)) > 10:
            labels_ref = list(sorted(set(labels)))
            df_corr = pd.DataFrame(np.random.rand(len(labels), len(labels_ref)))
            aac_plot = aa.AAclustPlot(verbose=False)
            with pytest.raises(ValueError):
                aac_plot.correlation(df_corr=df_corr, labels=labels, labels_ref=["wrong"])
                plt.close()


    @settings(max_examples=10, deadline=1000)
    @given(labels=some.lists(some.integers(), min_size=10, max_size=20))
    def test_invalid_labels_ref_from_df_corr(self, labels):
        # Labels should contain more than one distinct value
        if len(set(labels)) > 10:
            labels_ref = list(sorted(set(labels)))
            df_corr = pd.DataFrame(np.random.rand(len(labels), len(labels_ref)))
            df_corr.columns = [str(i) for i in list(df_corr)]
            aac_plot = aa.AAclustPlot(verbose=False)
            with pytest.raises(ValueError):
                aac_plot.correlation(df_corr=df_corr, labels=labels, labels_ref=["wrong"])
                plt.close()

    def test_cluster_x_invalid_type(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], cluster_x="not_a_boolean")

    def test_invalid_method(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], method="invalid_method")

    def test_xtick_label_rotation_invalid_type(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], xtick_label_rotation="not_an_integer")

    def test_ytick_label_rotation_invalid_type(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], ytick_label_rotation="not_an_integer")

    def test_invalid_bar_position(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], bar_position="invalid_position")

    def test_non_positive_bar_width_x(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], bar_width_x=-1)

    def test_invalid_cmap(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], cmap="invalid_cmap")

    def test_invalid_kwargs_heatmap(self):
        aac_plot = aa.AAclustPlot(verbose=False)
        df_corr = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            aac_plot.correlation(df_corr=df_corr, labels=[1, 2, 3], kwargs_heatmap=dict(invalid=None))
