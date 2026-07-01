"""This is a script to test the (internal) plot_prediction_hist() class-separated score histogram (#312)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.plotting._plot_prediction_hist import plot_prediction_hist

aa.options["verbose"] = False


# Helper functions
def _df(n_sub=10, n_hold=5, n_non=15, seed=0, scale=100.0):
    rng = np.random.default_rng(seed)
    n = n_sub + n_hold + n_non
    return pd.DataFrame({"score": rng.random(n) * scale,
                         "group": ["substrate"] * n_sub + ["hold-out"] * n_hold
                                  + ["non-substrate"] * n_non})


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotPredictionHist:
    """Normal cases for plot_prediction_hist."""

    def test_returns_fig_ax(self):
        fig, ax = plot_prediction_hist(df_pred=_df())
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    def test_returns_axes(self):
        _, ax = plot_prediction_hist(df_pred=_df())
        assert isinstance(ax, plt.Axes)

    def test_draws_bars(self):
        fig, ax = plot_prediction_hist(df_pred=_df())
        assert len(ax.patches) > 0

    def test_xlim_matches_binrange(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), binrange=(0, 100))
        assert ax.get_xlim() == (0.0, 100.0)

    def test_custom_binrange_applied(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), binrange=(0, 50))
        assert ax.get_xlim() == (0.0, 50.0)

    def test_auto_rescale_probability_to_percent(self):
        # scores in [0, 1] should map onto the [0, 100] axis (bars beyond x=1)
        fig, ax = plot_prediction_hist(df_pred=_df(scale=1.0))
        right_edges = [p.get_x() + p.get_width() for p in ax.patches if p.get_height() > 0]
        assert max(right_edges) > 1.0

    def test_percent_scores_not_rescaled(self):
        # An already-percent score (max > 1) must be left untouched (no x100 blow-up).
        df = pd.DataFrame({"score": [2.0, 40.0, 95.0], "group": ["a", "a", "a"]})
        fig, ax = plot_prediction_hist(df_pred=df, binrange=(0, 100))
        right_edges = [p.get_x() + p.get_width() for p in ax.patches if p.get_height() > 0]
        assert max(right_edges) <= 100.0  # would be 9500 if wrongly rescaled

    def test_stacked_true_uses_stack(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), stacked=True)
        assert len(ax.patches) > 0

    def test_layer_mode_runs(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), stacked=False)
        assert len(ax.patches) > 0

    def test_kde_adds_lines(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), kde=True)
        assert len(ax.get_lines()) > 0

    def test_no_kde_no_lines(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), kde=False)
        assert len(ax.get_lines()) == 0

    def test_legend_drawn(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), legend=True)
        assert ax.get_legend() is not None

    def test_legend_off(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), legend=False)
        assert ax.get_legend() is None

    def test_custom_columns(self):
        df = _df().rename(columns={"score": "s", "group": "g"})
        fig, ax = plot_prediction_hist(df_pred=df, col_score="s", col_group="g")
        assert len(ax.patches) > 0

    def test_draws_on_passed_ax(self):
        fig0, ax0 = plt.subplots()
        fig, ax = plot_prediction_hist(df_pred=_df(), ax=ax0)
        assert ax is ax0 and fig is fig0

    def test_figsize_applied(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), figsize=(10, 4))
        assert tuple(fig.get_size_inches()) == (10.0, 4.0)

    def test_labels_custom(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), xlabel="Score X", ylabel="N")
        assert ax.get_xlabel() == "Score X" and ax.get_ylabel() == "N"

    def test_fontsize_labels_applied(self):
        fig, ax = plot_prediction_hist(df_pred=_df(), fontsize_labels=15)
        assert ax.xaxis.label.get_size() == 15

    def test_canonical_substrate_color_is_green(self):
        fig, ax = plot_prediction_hist(df_pred=_df(),
                                       group_order=["substrate", "hold-out", "non-substrate"])
        green = matplotlib.colors.to_rgb(ut.COLOR_POS)
        facecolors = {tuple(np.round(p.get_facecolor()[:3], 4)) for p in ax.patches}
        assert tuple(np.round(green, 4)) in facecolors

    def test_custom_dict_color_applied(self):
        fig, ax = plot_prediction_hist(df_pred=_df(n_sub=5, n_hold=0, n_non=5),
                                       group_order=["substrate", "non-substrate"],
                                       dict_color={"substrate": "#000000", "non-substrate": "#ffffff"})
        facecolors = {tuple(np.round(p.get_facecolor()[:3], 4)) for p in ax.patches}
        assert (0.0, 0.0, 0.0) in facecolors


class TestPlotPredictionHistComplex:
    """Negative cases and combinations."""

    def test_missing_score_col_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df().drop(columns=["score"]))

    def test_missing_group_col_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df().drop(columns=["group"]))

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(n_sub=0, n_hold=0, n_non=0))

    def test_bad_score_name_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), col_score="nope")

    def test_bad_group_name_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), col_group="nope")

    def test_group_order_missing_group_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), group_order=["substrate"])

    def test_binwidth_zero_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), binwidth=0)

    def test_binwidth_negative_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), binwidth=-5)

    @pytest.mark.parametrize("bad", [(0,), (0, 50, 100), "rng"])
    def test_binrange_wrong_shape_raises(self, bad):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), binrange=bad)

    def test_binrange_low_ge_high_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), binrange=(100, 0))

    def test_stacked_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), stacked="yes")

    def test_kde_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), kde="yes")

    def test_legend_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), legend="yes")

    def test_figsize_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), figsize="big")

    def test_xlabel_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), xlabel=123)

    def test_fontsize_labels_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), fontsize_labels="big")

    def test_ax_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), ax="not_an_axes")

    def test_dict_color_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), dict_color="red")

    def test_group_order_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=_df(), group_order="substrate")

    def test_single_group(self):
        df = pd.DataFrame({"score": [10, 50, 90], "group": ["a", "a", "a"]})
        fig, ax = plot_prediction_hist(df_pred=df)
        assert len(ax.patches) > 0

    def test_non_numeric_score_raises(self):
        # A non-numeric score column must fail loudly, not be silently coerced to NaN.
        df = pd.DataFrame({"score": ["high", "low", "mid"], "group": ["a", "a", "a"]})
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=df)

    def test_all_nan_score_raises(self):
        # An all-NaN score column must fail with a clear message, not a cryptic seaborn error.
        df = pd.DataFrame({"score": [np.nan, np.nan], "group": ["a", "b"]})
        with pytest.raises(ValueError):
            plot_prediction_hist(df_pred=df)

    def test_integer_yticks(self):
        # Counts are integers; every y-tick within the data range must be a whole number.
        fig, ax = plot_prediction_hist(df_pred=_df())
        _, top = ax.get_ylim()
        assert all(float(t).is_integer() for t in ax.get_yticks() if 0 <= t <= top)
