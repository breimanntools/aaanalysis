"""This is a script to test the plot_eval_heatmap() house-preset evaluation heatmap (#310)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pytest

import aaanalysis as aa
from aaanalysis.plotting import plot_eval_heatmap

aa.options["verbose"] = False


# Helper functions
def _df(n_rows=2, n_cols=3, seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(55, 95, size=(n_rows, n_cols))
    return pd.DataFrame(vals,
                        index=[f"row{i}" for i in range(n_rows)],
                        columns=[f"col{j}" for j in range(n_cols)])


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotEvalHeatmap:
    """Normal cases for plot_eval_heatmap."""

    def test_returns_ax(self):
        ax = plot_eval_heatmap(df_eval=_df())
        assert isinstance(ax, plt.Axes)

    def test_is_public_export(self):
        assert "plot_eval_heatmap" in aa.__all__
        assert aa.plot_eval_heatmap is plot_eval_heatmap

    def test_draws_on_passed_ax(self):
        fig0, ax0 = plt.subplots()
        ax = plot_eval_heatmap(df_eval=_df(), ax=ax0)
        assert ax is ax0

    def test_vmin_vmax_respected(self):
        ax = plot_eval_heatmap(df_eval=_df(), vmin=40, vmax=90)
        assert ax.collections[0].get_clim() == (40.0, 90.0)

    def test_default_vmin_vmax(self):
        ax = plot_eval_heatmap(df_eval=_df())
        assert ax.collections[0].get_clim() == (50.0, 100.0)

    def test_xlabel_ylabel_set(self):
        ax = plot_eval_heatmap(df_eval=_df(), xlabel="Scales", ylabel="Parts")
        assert ax.get_xlabel() == "Scales" and ax.get_ylabel() == "Parts"

    def test_labels_default_none_keeps_seaborn(self):
        # With unnamed axes and xlabel/ylabel=None, no custom label is forced.
        ax = plot_eval_heatmap(df_eval=_df())
        assert ax.get_xlabel() == "" and ax.get_ylabel() == ""

    def test_cbar_label_default(self):
        ax = plot_eval_heatmap(df_eval=_df())
        # The colorbar lives on a sibling axes of the same figure.
        cbar_axes = [a for a in ax.figure.axes if a is not ax]
        assert any(a.get_ylabel() == "Balanced accuracy [%]" for a in cbar_axes)

    def test_cbar_label_custom(self):
        ax = plot_eval_heatmap(df_eval=_df(), cbar_label="F1 [%]")
        cbar_axes = [a for a in ax.figure.axes if a is not ax]
        assert any(a.get_ylabel() == "F1 [%]" for a in cbar_axes)

    def test_cbar_label_none_no_colorbar_label(self):
        ax = plot_eval_heatmap(df_eval=_df(), cbar_label=None)
        cbar_axes = [a for a in ax.figure.axes if a is not ax]
        assert all(a.get_ylabel() == "" for a in cbar_axes)

    def test_annotation_count_matches_cells(self):
        df = _df(n_rows=2, n_cols=3)
        ax = plot_eval_heatmap(df_eval=df)
        assert len(ax.texts) == df.size

    def test_ticklabels_horizontal(self):
        ax = plot_eval_heatmap(df_eval=_df())
        assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())
        assert all(t.get_rotation() == 0 for t in ax.get_yticklabels())

    def test_single_cell(self):
        ax = plot_eval_heatmap(df_eval=pd.DataFrame([[88.0]]))
        assert isinstance(ax, plt.Axes) and len(ax.texts) == 1


class TestPlotEvalHeatmapEquivalence:
    """KPI #310: equivalent to the hand-built seaborn block it consolidates."""

    def test_matches_raw_seaborn_block(self):
        # The exact block duplicated in gamma-secretase notebook cells 12/25.
        df = _df()
        fig_raw, ax_raw = plt.subplots()
        sns.heatmap(df, ax=ax_raw, vmin=50, vmax=100, cmap="viridis", annot=True,
                    fmt=".0f", linewidth=0.1, cbar_kws=dict(label="Balanced accuracy [%]"))
        ax_raw.tick_params(left=False, bottom=False)
        ax_new = plot_eval_heatmap(df_eval=df)
        # Same heatmap data, color limits, colormap, and annotations.
        raw_mesh, new_mesh = ax_raw.collections[0], ax_new.collections[0]
        assert np.allclose(raw_mesh.get_array(), new_mesh.get_array())
        assert raw_mesh.get_clim() == new_mesh.get_clim()
        assert raw_mesh.get_cmap().name == new_mesh.get_cmap().name == "viridis"
        assert ([t.get_text() for t in ax_raw.texts]
                == [t.get_text() for t in ax_new.texts])


class TestPlotEvalHeatmapErrors:
    """Negative cases — bad input raises ValueError."""

    def test_not_a_dataframe(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval="not a frame")

    def test_none_df(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=None)

    def test_empty_df(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=pd.DataFrame())

    def test_non_numeric_df(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]}))

    def test_vmin_not_below_vmax(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=_df(), vmin=100, vmax=50)

    def test_bad_xlabel_type(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=_df(), xlabel=123)

    def test_bad_ax_type(self):
        with pytest.raises(ValueError):
            plot_eval_heatmap(df_eval=_df(), ax="not an ax")
