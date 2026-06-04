"""This is a script to test AAclustPlot render paths that the existing
hypothesis tests don't reach: centers()/medoids() actually rendering (the
existing guard `not np.asarray_chkfinite(X).any()` skips the real call for all
realistic X), the PCA component path, and the correlation() check branches
(labels_ref mismatch warning, cluster_x validation, bar_colors list/error/short).

Deterministic inputs; Agg backend; figures closed after each test.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


# I Helper Functions
def _X_labels(n=20, f=8, k=4):
    rng = np.random.default_rng(0)
    X = rng.random((n, f))
    labels = [i % k for i in range(n)]
    return X, labels


def _df_corr(n=6, k=3):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.random((n, k)) * 2 - 1, columns=list(range(k)))
    return df


# II Test Classes
class TestAAclustPlotCentersRender:
    """centers() renders -> covers the frontend body + backend plot_center_or_medoid + PCA."""

    def test_centers_returns_ax_and_components(self):
        X, labels = _X_labels()
        ax, dfc = aa.AAclustPlot().centers(X, labels=labels)
        assert isinstance(ax, plt.Axes)
        assert isinstance(dfc, pd.DataFrame)
        assert dfc.shape[0] == 20
        # PCA columns are renamed to 'PC..(xx.x%)'
        assert any("%" in str(c) for c in dfc.columns)
        plt.close("all")

    def test_centers_no_legend(self):
        X, labels = _X_labels()
        ax, _ = aa.AAclustPlot().centers(X, labels=labels, legend=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_centers_custom_palette_and_components(self):
        X, labels = _X_labels()
        palette = ["red", "green", "blue", "orange"]
        ax, _ = aa.AAclustPlot().centers(X, labels=labels, component_x=1,
                                         component_y=2, palette=palette)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_centers_on_passed_ax(self):
        X, labels = _X_labels()
        fig, ax0 = plt.subplots()
        ax, _ = aa.AAclustPlot().centers(X, labels=labels, ax=ax0)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestAAclustPlotMedoidsRender:
    """medoids() renders -> covers the medoid branch of plot_center_or_medoid."""

    def test_medoids_euclidean(self):
        X, labels = _X_labels()
        ax, dfc = aa.AAclustPlot().medoids(X, labels=labels, metric="euclidean")
        assert isinstance(ax, plt.Axes)
        assert isinstance(dfc, pd.DataFrame)
        plt.close("all")

    def test_medoids_correlation_metric(self):
        X, labels = _X_labels()
        ax, _ = aa.AAclustPlot().medoids(X, labels=labels, metric="correlation")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_medoids_no_legend(self):
        X, labels = _X_labels()
        ax, _ = aa.AAclustPlot().medoids(X, labels=labels, legend=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestAAclustPlotEvalRender:
    """eval() renders the 4-panel barplot."""

    def test_eval_renders(self):
        df_eval = pd.DataFrame({
            "name": ["A", "B", "C"],
            "n_clusters": [3, 4, 5],
            "BIC": [-10.0, -5.0, -2.0],
            "CH": [50.0, 60.0, 45.0],
            "SC": [0.4, 0.5, 0.3],
        })
        fig, axes = aa.AAclustPlot.eval(df_eval=df_eval)
        assert fig is not None and len(axes) == 4
        plt.close("all")


class TestAAclustPlotCorrelationBranches:
    """correlation() check-branch coverage."""

    def test_correlation_basic(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        ax = aa.AAclustPlot(verbose=False).correlation(
            df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_correlation_all_bar_positions(self):
        # Drives plot_add_bars for every side -> _get_xy_hava top/bottom/right
        # branches + _add_bar_labels in utils_plot_elements.
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        ax = aa.AAclustPlot(verbose=False).correlation(
            df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
            bar_position=["left", "right", "top", "bottom"])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_correlation_cluster_x(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        ax = aa.AAclustPlot(verbose=False).correlation(
            df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2], cluster_x=True)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_correlation_bar_colors_list(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        ax = aa.AAclustPlot(verbose=False).correlation(
            df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
            bar_colors=["red", "green", "blue"])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_correlation_bar_colors_short_warns(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aa.AAclustPlot(verbose=False).correlation(
                df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
                bar_colors=["red", "green"])  # 2 < 3 clusters -> warn
        assert any("bar_colors" in str(x.message) for x in w)
        plt.close("all")

    def test_correlation_labels_ref_mismatch_warns(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # labels_ref set differs from labels set + verbose -> warning
            aa.AAclustPlot(verbose=True).correlation(
                df_corr=df_corr, labels=labels, labels_ref=[7, 8, 9])
        assert any("does not match" in str(x.message) for x in w)
        plt.close("all")

    # ----- NEGATIVES -----
    def test_correlation_invalid_bar_colors_type(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="bar_colors"):
            aa.AAclustPlot(verbose=False).correlation(
                df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
                bar_colors=123)

    def test_correlation_cluster_x_all_same_raises(self):
        df_corr = pd.DataFrame(np.ones((6, 3)), columns=[0, 1, 2])
        labels = [0, 1, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="same values"):
            aa.AAclustPlot(verbose=False).correlation(
                df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
                cluster_x=True)

    def test_correlation_invalid_method(self):
        df_corr = _df_corr()
        labels = [0, 1, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="method"):
            aa.AAclustPlot(verbose=False).correlation(
                df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2],
                method="not_a_method")
