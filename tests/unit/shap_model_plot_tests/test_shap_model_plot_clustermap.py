"""This script tests the ShapModelPlot.clustermap() method."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pytest

pytest.importorskip("shap")  # ShapModelPlot lives in the pro extra (needs shap to import)
from aaanalysis.explainable_ai_pro import ShapModelPlot


def _shap_matrix(n_pos=5, n_neg=6, n_features=25, seed=0):
    """Two class-structured blocks so the correlation clustermap is well separated."""
    rng = np.random.default_rng(seed)
    base_pos, base_neg = rng.normal(size=(1, n_features)), rng.normal(size=(1, n_features))
    pos = base_pos + rng.normal(scale=0.4, size=(n_pos, n_features))
    neg = base_neg + rng.normal(scale=0.4, size=(n_neg, n_features))
    shap_values = np.vstack([pos, neg])
    labels = [1] * n_pos + [0] * n_neg
    return shap_values, labels


class TestClustermap:
    """Normal cases, one parameter per test."""

    def test_returns_clustergrid(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels)
        assert isinstance(grid, sns.matrix.ClusterGrid)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_heatmap_is_square_n_samples(self):
        sv, labels = _shap_matrix(n_pos=5, n_neg=6)
        grid = ShapModelPlot.clustermap(sv, labels=labels)
        # The correlation matrix is (n_samples, n_samples)
        assert grid.data2d.shape == (11, 11)
        plt.close("all")

    def test_linkage_available(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels)
        assert grid.dendrogram_row.linkage.shape == (len(labels) - 1, 4)
        plt.close("all")

    def test_default_names(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels)
        assert "Protein0" in list(grid.data2d.columns)
        plt.close("all")

    def test_custom_names(self):
        sv, labels = _shap_matrix()
        names = [f"P{i}" for i in range(len(labels))]
        grid = ShapModelPlot.clustermap(sv, labels=labels, names=names)
        assert set(names) == set(grid.data2d.columns)
        plt.close("all")

    def test_labels_pred(self):
        sv, labels = _shap_matrix()
        labels_pred = [0] * 5 + [1] * 6
        grid = ShapModelPlot.clustermap(sv, labels=labels, labels_pred=labels_pred)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_custom_dict_color(self):
        sv, labels = _shap_matrix()
        dict_color = {0: "tab:gray", 1: "tab:red"}
        grid = ShapModelPlot.clustermap(sv, labels=labels, dict_color=dict_color)
        assert grid.ax_heatmap is not None
        plt.close("all")

    @pytest.mark.parametrize("method", ["complete", "average", "single", "ward"])
    def test_method(self, method):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, method=method)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_figsize(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, figsize=(4, 4))
        assert grid.ax_heatmap is not None
        plt.close("all")

    @pytest.mark.parametrize("cmap", ["GnBu", "viridis", "coolwarm"])
    def test_cmap(self, cmap):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, cmap=cmap)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_vmin_vmax(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, vmin=0, vmax=1)
        assert grid.ax_heatmap is not None
        plt.close("all")

    @pytest.mark.parametrize("tick_labels", [1, 2, 5])
    def test_tick_labels(self, tick_labels):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, tick_labels=tick_labels)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_tree_linewidth(self):
        sv, labels = _shap_matrix()
        grid = ShapModelPlot.clustermap(sv, labels=labels, tree_linewidth=2.5)
        assert grid.ax_heatmap is not None
        plt.close("all")

    def test_deterministic(self):
        sv, labels = _shap_matrix()
        g1 = ShapModelPlot.clustermap(sv, labels=labels)
        link1 = g1.dendrogram_row.linkage.copy()
        plt.close("all")
        g2 = ShapModelPlot.clustermap(sv, labels=labels)
        assert np.array_equal(link1, g2.dendrogram_row.linkage)
        plt.close("all")


class TestClustermapErrors:
    """Negative cases, one wrong parameter per test."""

    def test_single_sample_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(np.ones((1, 5)), labels=[1])

    def test_1d_shap_values_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(np.ones(5), labels=[1, 0])

    def test_labels_length_mismatch_raises(self):
        sv, _ = _shap_matrix()
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=[1, 0])

    def test_names_length_mismatch_raises(self):
        sv, labels = _shap_matrix()
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=labels, names=["a", "b"])

    def test_duplicate_names_raises(self):
        sv, labels = _shap_matrix(n_pos=1, n_neg=1, n_features=5)
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=labels, names=["dup", "dup"])

    def test_dict_color_missing_class_raises(self):
        sv, labels = _shap_matrix()
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=labels, dict_color={1: "tab:red"})

    def test_bad_tick_labels_raises(self):
        sv, labels = _shap_matrix()
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=labels, tick_labels=0)

    def test_negative_tree_linewidth_raises(self):
        sv, labels = _shap_matrix()
        with pytest.raises(ValueError):
            ShapModelPlot.clustermap(sv, labels=labels, tree_linewidth=-1)
