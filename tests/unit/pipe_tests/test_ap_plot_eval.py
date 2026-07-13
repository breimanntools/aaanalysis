"""This script tests the aaanalysis.pipe.plot_eval() publication eval-figure decomposition."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.pipe as ap
from aaanalysis.pipe._eval_plot import (_resolve_score_col, _ranked_axes, _axis_impact,
                                        _sensitivity_view, _axis_levels, _level_label, _best_row,
                                        plot_eval)

aa.options["verbose"] = False


def _make_eval(parts=("tmd",), pats=("none",), nsmax=(15,), scales=("explain:30",),
               nfilt=(100,), metrics=("balanced_accuracy",)):
    """Build a synthetic v2 find_features-style sensitivity table over the given level lists."""
    rows = []
    for lp, pm, ns, sc, nf in itertools.product(parts, pats, nsmax, scales, nfilt):
        row = {"stage": "sensitivity", "list_parts": lp, "split_types": pm, "pattern_mode": pm,
               "n_split_max": ns, "scale": sc, "n_filter": nf, "n_features": nf}
        for m in metrics:
            row[m + "_mean"] = 0.5 + 0.1 * np.cos(nf / 40.0) + 0.02 * len(pm) + 0.01 * len(sc)
            row[m + "_std"] = 0.03
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_pareto"] = False
    df["rank"] = df[metrics[0] + "_mean"].rank(ascending=False, method="first").astype(int)
    df["is_selected"] = False
    df.loc[df[metrics[0] + "_mean"].idxmax(), "is_selected"] = True
    return df


# Per-dimensionality fixtures (0/1/2/3 structural axes varying).
_DF_0AX = _make_eval()
_DF_1AX = _make_eval(pats=("none", "p1", "p2", "p1+p2"))
_DF_2AX = _make_eval(pats=("none", "p1", "p2", "p1+p2"), scales=("explain:30", "explain:50"))
_DF_3AX = _make_eval(parts=("tmd", "tmd_jmd", "tmd_c"), pats=("none", "p1", "p2"),
                     scales=("explain:30", "explain:50"))
_DF_2METRIC = _make_eval(pats=("none", "p1"), scales=("explain:30", "explain:50"),
                         metrics=("balanced_accuracy", "f1"))


def _make_eval_njmd(njmd=(6, 10, 14), pats=("none", "p1"), metric="balanced_accuracy"):
    """Synthetic sensitivity table whose structural axes are the symmetric JMD length and splits."""
    rows = []
    for nj, pm in itertools.product(njmd, pats):
        row = {"stage": "sensitivity", "list_parts": "tmd", "split_types": pm, "pattern_mode": pm,
               "n_split_max": 15, "scale": "explain:30", "n_jmd": nj, "n_filter": 100,
               "n_features": 100}
        # Score varies clearly with n_jmd so the axis registers a non-zero marginal impact.
        row[metric + "_mean"] = 0.5 + 0.02 * nj + 0.01 * len(pm)
        row[metric + "_std"] = 0.03
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_pareto"] = False
    df["rank"] = df[metric + "_mean"].rank(ascending=False, method="first").astype(int)
    df["is_selected"] = False
    df.loc[df[metric + "_mean"].idxmax(), "is_selected"] = True
    return df


_DF_NJMD = _make_eval_njmd()


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotEval:
    """Positive tests: the publication-figure decomposition."""

    def test_single_config_returns_empty(self):
        assert plot_eval(_DF_0AX) == []

    def test_returns_list_of_figures(self):
        figs = plot_eval(_DF_2AX)
        assert isinstance(figs, list) and all(isinstance(f, Figure) for f in figs)

    def test_two_axes_one_heatmap_plus_marginal(self):
        figs = plot_eval(_DF_2AX)
        n_heatmaps = sum(1 for f in figs for ax in f.axes if ax.images)
        assert n_heatmaps == 1   # 2 axes -> a single heatmap slice
        assert len(figs) >= 2     # heatmap + marginal panel

    def test_three_axes_multiple_separate_heatmaps(self):
        figs = plot_eval(_DF_3AX)
        heatmap_figs = [f for f in figs if any(ax.images for ax in f.axes)]
        assert len(heatmap_figs) >= 2   # one separate figure per slice level

    def test_each_heatmap_is_own_figure(self):
        figs = plot_eval(_DF_3AX)
        for f in figs:
            assert sum(1 for ax in f.axes if ax.images) <= 1   # never crammed into one figure

    def test_heatmaps_use_viridis(self):
        cmaps = {ax.images[0].get_cmap().name
                 for f in plot_eval(_DF_2AX) for ax in f.axes if ax.images}
        assert cmaps == {"viridis"}

    def test_heatmaps_share_color_scale(self):
        clims = {ax.images[0].get_clim()
                 for f in plot_eval(_DF_3AX) for ax in f.axes if ax.images}
        assert len(clims) == 1   # every slice on one shared scale

    def test_best_marked(self):
        # The selected configuration is boxed (a Rectangle patch) on a heatmap (image) panel.
        marked = any(len(ax.patches) > 0
                     for f in plot_eval(_DF_2AX) for ax in f.axes if ax.images)
        assert marked

    def test_marginal_panel_present(self):
        # The marginal panel has bar patches and no image.
        figs = plot_eval(_DF_2AX)
        assert any(any(ax.patches for ax in f.axes) and not any(ax.images for ax in f.axes)
                   for f in figs)

    def test_metric_parameter_selects_score(self):
        figs = plot_eval(_DF_2METRIC, metric="f1")
        assert len(figs) >= 1

    def test_score_col_parameter(self):
        figs = plot_eval(_DF_2AX, score_col="balanced_accuracy_mean")
        assert len(figs) >= 1

    def test_figsize_parameter(self):
        figs = plot_eval(_DF_2AX, figsize=(5, 4))
        assert len(figs) >= 1

    def test_n_jmd_rendered_as_heatmap_axis(self):
        # A sweep table carrying the symmetric JMD-length axis renders it on a heatmap (x or y).
        figs = plot_eval(_DF_NJMD)
        heatmap_axes = [ax for f in figs for ax in f.axes if ax.images]
        assert heatmap_axes  # at least one heatmap exists
        labels = {ax.get_xlabel() for ax in heatmap_axes} | {ax.get_ylabel() for ax in heatmap_axes}
        assert "JMD length (n_jmd)" in labels

    def test_n_jmd_is_numeric_axis(self):
        # n_jmd levels sort numerically (not lexically) for the eval grid.
        assert _axis_levels(_DF_NJMD, "n_jmd") == [6, 10, 14]

    def test_most_informative_axes_on_heatmap(self):
        # The lowest-impact axis becomes the slice; the top-2 are the heatmap x/y.
        axes = _ranked_axes(_sensitivity_view(_DF_3AX), "balanced_accuracy_mean")
        figs = plot_eval(_DF_3AX)
        heatmap_figs = [f for f in figs if any(ax.images for ax in f.axes)]
        # number of separate heatmaps == number of levels of the least-informative axis
        assert len(heatmap_figs) == _DF_3AX[axes[-1]].nunique()


class TestPlotEvalNFilter:
    """The n_filter refinement panel (its own stage in df_eval)."""

    @pytest.mark.slow
    def test_nfilter_panel_from_real_sweep(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
        labels = df_seq["label"].to_list()
        _, _, df_eval = ap.find_features(labels, df_seq=df_seq, search="balanced",
                                          kws={"n_explain": 30, "n_split_max": 15}, plot=False,
                                          random_state=0, n_jobs=1)
        figs = plot_eval(df_eval)
        # n_filter stage varies -> a line panel exists (lines, no image).
        assert any(any(ax.lines for ax in f.axes) and not any(ax.images for ax in f.axes)
                   for f in figs)


class TestPlotEvalErrors:
    """Negative tests: one rejected input per test."""

    def test_df_eval_not_dataframe(self):
        with pytest.raises(ValueError):
            plot_eval([1, 2, 3])

    def test_df_eval_empty(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2AX.iloc[0:0])

    def test_no_mean_column(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2AX.rename(columns={"balanced_accuracy_mean": "score"}))

    def test_score_col_missing(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2AX, score_col="not_a_column")

    def test_all_nan_score_rejected(self):
        df = _DF_2AX.copy()
        df["balanced_accuracy_mean"] = np.nan
        with pytest.raises(ValueError):
            plot_eval(df)


class TestPlotEvalHelpers:
    """Unit tests for the score/axis helpers."""

    def test_resolve_score_col_first_mean(self):
        assert _resolve_score_col(_DF_2AX) == "balanced_accuracy_mean"

    def test_resolve_score_col_metric(self):
        assert _resolve_score_col(_DF_2METRIC, metric="f1") == "f1_mean"

    def test_resolve_score_col_explicit(self):
        assert _resolve_score_col(_DF_2AX, score_col="x") == "x"

    def test_axis_impact_nonneg(self):
        assert _axis_impact(_DF_2AX, "pattern_mode", "balanced_accuracy_mean") >= 0

    def test_ranked_axes_by_impact(self):
        axes = _ranked_axes(_sensitivity_view(_DF_3AX), "balanced_accuracy_mean")
        impacts = [_axis_impact(_DF_3AX, a, "balanced_accuracy_mean") for a in axes]
        assert impacts == sorted(impacts, reverse=True)

    def test_sensitivity_view_filters_stage(self):
        df = pd.concat([_DF_2AX, _DF_2AX.assign(stage="n_filter")], ignore_index=True)
        assert (_sensitivity_view(df)["stage"] == "sensitivity").all()

    def test_axis_levels_none_last(self):
        df = _make_eval(scales=("explain:30",), nsmax=(5, 15))
        assert _axis_levels(df, "n_split_max") == [5, 15]

    def test_level_label_numeric(self):
        assert _level_label("n_split_max", 15) == "15"

    def test_best_row_uses_is_selected(self):
        assert bool(_DF_2AX.loc[_best_row(_DF_2AX, "balanced_accuracy_mean"), "is_selected"])
