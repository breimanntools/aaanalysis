"""This script tests the ShapModelPlot.get_clusters() method."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")  # ShapModelPlot lives in the pro extra (needs shap to import)
from aaanalysis.explainable_ai_pro import ShapModelPlot


def _shap_matrix(n_pos=5, n_neg=6, n_features=25, seed=0):
    rng = np.random.default_rng(seed)
    base_pos, base_neg = rng.normal(size=(1, n_features)), rng.normal(size=(1, n_features))
    pos = base_pos + rng.normal(scale=0.4, size=(n_pos, n_features))
    neg = base_neg + rng.normal(scale=0.4, size=(n_neg, n_features))
    return np.vstack([pos, neg])


class TestGetClusters:
    """Normal cases, one parameter per test."""

    def test_returns_dataframe_columns(self):
        df = ShapModelPlot.get_clusters(_shap_matrix())
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "cluster"]

    def test_row_per_sample(self):
        sv = _shap_matrix(n_pos=4, n_neg=4)
        df = ShapModelPlot.get_clusters(sv)
        assert len(df) == 8

    def test_default_two_clusters(self):
        df = ShapModelPlot.get_clusters(_shap_matrix())
        assert df["cluster"].nunique() == 2

    def test_n_clusters(self):
        df = ShapModelPlot.get_clusters(_shap_matrix(), n_clusters=3)
        assert df["cluster"].nunique() == 3

    def test_separates_class_blocks(self):
        # Well-separated blocks -> the two clusters recover the two class blocks
        df = ShapModelPlot.get_clusters(_shap_matrix(n_pos=5, n_neg=6), n_clusters=2)
        clusters = df["cluster"].to_list()
        assert len(set(clusters[:5])) == 1 and len(set(clusters[5:])) == 1
        assert clusters[0] != clusters[5]

    def test_color_threshold(self):
        df = ShapModelPlot.get_clusters(_shap_matrix(), color_threshold=1e9)
        # A huge threshold collapses everything into a single cluster
        assert df["cluster"].nunique() == 1

    def test_custom_names(self):
        sv = _shap_matrix(n_pos=2, n_neg=2, n_features=5)
        names = ["a", "b", "c", "d"]
        df = ShapModelPlot.get_clusters(sv, names=names)
        assert df["name"].to_list() == names

    def test_deterministic(self):
        sv = _shap_matrix()
        d1 = ShapModelPlot.get_clusters(sv, n_clusters=3)
        d2 = ShapModelPlot.get_clusters(sv, n_clusters=3)
        assert d1.equals(d2)

    def test_cluster_ids_are_int(self):
        df = ShapModelPlot.get_clusters(_shap_matrix())
        assert all(isinstance(c, int) for c in df["cluster"])


class TestGetClustersErrors:
    """Negative cases, one wrong parameter per test."""

    def test_both_criteria_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.get_clusters(_shap_matrix(), n_clusters=2, color_threshold=1.0)

    def test_n_clusters_out_of_range_raises(self):
        sv = _shap_matrix(n_pos=3, n_neg=3)
        with pytest.raises(ValueError):
            ShapModelPlot.get_clusters(sv, n_clusters=99)

    def test_1d_shap_values_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.get_clusters(np.ones(5))

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.get_clusters(_shap_matrix(), names=["a", "b"])

    def test_negative_color_threshold_raises(self):
        with pytest.raises(ValueError):
            ShapModelPlot.get_clusters(_shap_matrix(), color_threshold=-1)
