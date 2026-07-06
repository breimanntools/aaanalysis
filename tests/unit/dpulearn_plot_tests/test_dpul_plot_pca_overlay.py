"""
Tests the dPULearnPlot.pca() projected-group overlay (issue #352).

``pca`` gains optional ``df_pu_add`` / ``names_add`` / ``colors_add`` to overlay one or more groups
of held-out samples (projected via :meth:`dPULearn.project`) on top of the three ``df_pu`` groups.
The key contracts: with ``df_pu_add=None`` the figure is byte-identical to the three-group plot, and
each extra group adds exactly one scatter collection with its own name/color in the legend.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


# Helper functions
def _fit_and_project(n_pos=8, n_unl=12, n_features=30, n_new=5, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(0.0, 1.0, size=(n_pos, n_features))
    X_unl = rng.normal(0.6, 1.0, size=(n_unl, n_features))
    dpul = aa.dPULearn(random_state=42, verbose=False)
    dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_unl_to_neg=4)
    X_new = rng.normal(0.2, 1.0, size=(n_new, n_features))
    df_proj = dpul.project(X_new)
    return dpul, df_proj


def _pc_cols(df_pu):
    return [c for c in df_pu.columns if "PC" in c and "abs" not in c]


# Normal Cases Test Class
class TestPCAOverlay:
    """Test the projected-group overlay parameters."""

    def test_default_no_overlay_is_unchanged(self):
        """df_pu_add=None: one scatter collection per unique label, offsets == df_pu_ PC values."""
        dpul, _ = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        fig, ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels)
        assert len(ax.collections) == len(set(labels))
        cols = _pc_cols(dpul.df_pu_)
        # Each collection's offsets must come straight from df_pu_ (default rendering regression)
        plotted = np.vstack([c.get_offsets().data for c in ax.collections])
        expected = dpul.df_pu_[[cols[0], cols[1]]].to_numpy()
        assert np.allclose(np.sort(plotted[:, 0]), np.sort(expected[:, 0]))
        plt.close(fig)

    def test_single_extra_group_adds_one_collection(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        fig0, ax0 = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels)
        n0 = len(ax0.collections)
        plt.close(fig0)
        fig, ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels,
                                        df_pu_add=df_proj, names_add="Held-out")
        assert len(ax.collections) == n0 + 1
        # Extra group's offsets equal the projected coordinates
        cols = _pc_cols(dpul.df_pu_)
        added = ax.collections[-1].get_offsets().data
        assert np.allclose(np.sort(added[:, 0]), np.sort(df_proj[cols[0]].to_numpy()))
        plt.close(fig)

    @pytest.mark.parametrize("n_extra", [1, 2, 3])
    def test_multiple_extra_groups(self, n_extra):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        n0 = len(set(labels))
        list_add = [df_proj] * n_extra
        fig, ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels, df_pu_add=list_add)
        assert len(ax.collections) == n0 + n_extra
        plt.close(fig)

    def test_names_and_colors_defaults(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        # Should not raise with names_add / colors_add omitted (defaults applied)
        fig, ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels, df_pu_add=df_proj)
        plt.close(fig)

    def test_colors_add_as_string(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        fig, ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels, df_pu_add=df_proj,
                                        names_add="G", colors_add="tab:red")
        plt.close(fig)


# Negative Cases Test Class
class TestPCAOverlayNegative:
    """Test the overlay rejects mismatched or malformed extra groups."""

    def test_names_add_length_mismatch(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        with pytest.raises(ValueError):
            aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels,
                                  df_pu_add=[df_proj, df_proj], names_add=["only-one"])

    def test_colors_add_length_mismatch(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        with pytest.raises(ValueError):
            aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels,
                                  df_pu_add=[df_proj, df_proj],
                                  names_add=["a", "b"], colors_add=["tab:red"])

    def test_missing_pc_column(self):
        dpul, df_proj = _fit_and_project()
        labels = np.asarray(dpul.labels_)
        bad = df_proj.rename(columns={df_proj.columns[1]: "not_a_pc"})
        with pytest.raises(ValueError):
            aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=labels, df_pu_add=bad, pc_x=1, pc_y=2)
