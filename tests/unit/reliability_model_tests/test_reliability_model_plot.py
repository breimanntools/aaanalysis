"""Unit tests for ReliabilityModelPlot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.datasets import make_classification
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

import aaanalysis as aa


def _fitted():
    X, y = make_classification(n_samples=100, n_features=8, n_informative=5, random_state=0)
    rm = aa.ReliabilityModel(random_state=0).fit(X[:80], y[:80], n_bootstrap=5)
    return rm, X[80:], y[80:], X[:80], y[:80]


def _df_rel():
    rm, X_new, *_ = _fitted()
    return rm.predict(X_new)


def _df_eval():
    rm, _, _, Xt, yt = _fitted()
    return rm.eval(X=Xt, labels=yt)


@pytest.fixture(scope="module")
def df_eval():
    """One fitted eval table, reused by the per-parameter reliability_diagram tests."""
    return _df_eval()


def _is_fig_ax(res):
    fig, ax = res
    return isinstance(fig, Figure) and isinstance(ax, Axes)


class TestRanking:
    def test_returns_fig_ax(self):
        assert _is_fig_ax(aa.ReliabilityModelPlot().ranking(df_rel=_df_rel()))
        plt.close("all")

    def test_params(self):
        df = _df_rel()
        fig, ax = plt.subplots()
        res = aa.ReliabilityModelPlot().ranking(
            df_rel=df, names=[f"p{i}" for i in range(len(df))], figsize=(5, 6),
            top_n=5, title="rank", ax=ax)
        assert res[1] is ax
        plt.close("all")

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="'names'"):
            aa.ReliabilityModelPlot().ranking(df_rel=_df_rel(), names=["only-one"])

    @pytest.mark.parametrize("top_n", [0, -2])
    def test_bad_top_n_raises(self, top_n):
        with pytest.raises(ValueError, match="'top_n'"):
            aa.ReliabilityModelPlot().ranking(df_rel=_df_rel(), top_n=top_n)

    def test_missing_col_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().ranking(df_rel=pd.DataFrame({"score": [0.5]}))


class TestReliabilityDiagram:
    def test_returns_fig_ax(self):
        assert _is_fig_ax(aa.ReliabilityModelPlot().reliability_diagram(df_eval=_df_eval()))
        plt.close("all")

    # figsize / color / label / title / ax: positive, asserting the visual effect
    @settings(max_examples=5, deadline=None)
    @given(w=some.floats(min_value=3, max_value=8), h=some.floats(min_value=3, max_value=8))
    def test_figsize_applied(self, df_eval, w, h):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, figsize=(w, h))
        assert tuple(fig.get_size_inches()) == pytest.approx((w, h))
        plt.close("all")

    @pytest.mark.parametrize("color", ["tab:red", "green", "#1b9e77"])
    def test_color_applied(self, df_eval, color):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, color=color)
        assert to_rgba(ax.get_lines()[-1].get_color()) == to_rgba(color)   # [0] is the diagonal
        plt.close("all")

    @settings(max_examples=5, deadline=None)
    @given(label=some.text(alphabet="abcdefgh ", min_size=1, max_size=12))
    def test_label_valid(self, label):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=_df_eval(), label=label)
        labels = [line.get_label() for line in ax.get_lines()]
        assert label in labels
        plt.close("all")

    @settings(max_examples=5, deadline=None)
    @given(title=some.text(alphabet="abcdefgh ", min_size=1, max_size=12))
    def test_title_applied(self, df_eval, title):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, title=title)
        assert ax.get_title() == title
        plt.close("all")

    def test_ax_reused(self, df_eval):
        fig, ax = plt.subplots()
        n_figs = len(plt.get_fignums())
        res = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, ax=ax)
        assert res[1] is ax and res[0] is fig
        assert len(plt.get_fignums()) == n_figs            # drew onto the passed ax, no new figure
        plt.close("all")

    # df_eval / figsize / color / label / title / ax: negative
    def test_bad_df_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=pd.DataFrame({"x": [1]}))

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="should be a pandas DataFrame"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=[1, 2, 3])

    @pytest.mark.parametrize("figsize", [(0, 5), (5,), "big", (None, 5), 5])
    def test_figsize_invalid(self, df_eval, figsize):
        with pytest.raises(ValueError, match="figsize"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, figsize=figsize)

    @pytest.mark.parametrize("color", ["not-a-color", 5, None, ["tab:blue"]])
    def test_color_invalid(self, df_eval, color):
        with pytest.raises(ValueError, match="color"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, color=color)

    @pytest.mark.parametrize("label", [None, 1, ["model"]])
    def test_label_invalid(self, label):
        with pytest.raises(ValueError, match="'label'"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=_df_eval(), label=label)

    @pytest.mark.parametrize("title", [5, ["cal"], 1.5, True])
    def test_title_invalid(self, df_eval, title):
        with pytest.raises(ValueError, match="title"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, title=title)

    @pytest.mark.parametrize("ax", ["ax", 5, [1, 2]])
    def test_ax_invalid(self, df_eval, ax):
        with pytest.raises(ValueError, match="ax"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, ax=ax)

    def test_figure_as_ax_invalid(self, df_eval):
        with pytest.raises(ValueError, match="ax"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, ax=plt.figure())


class TestReliabilityDiagramComplex:
    """Metric annotation, raw-vs-calibrated overlays, and the ax / figsize interplay."""

    # Positive interactions
    def test_metrics_annotated_in_legend(self):
        rm, _, _, Xt, yt = _fitted()
        df_eval = rm.eval(X=Xt, labels=yt, add_metrics=True, use_calibrated=True)
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval,
                                                                label="calibrated")
        brier = float(df_eval.loc[df_eval["bin"] == "brier", "mean_score"].iloc[0])
        ece = float(df_eval.loc[df_eval["bin"] == "ece", "mean_score"].iloc[0])
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert f"calibrated (Brier {brier:.3f}, ECE {ece:.3f})" in texts
        plt.close("all")

    def test_no_annotation_without_metric_rows(self):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=_df_eval())
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["perfect calibration", "model"]
        plt.close("all")

    def test_metric_rows_not_plotted_as_points(self):
        rm, _, _, Xt, yt = _fitted()
        df_eval = rm.eval(X=Xt, labels=yt, add_metrics=True)
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval)
        n_bins = int((df_eval["n_samples"].iloc[:5] > 0).sum())
        assert len(ax.get_lines()[-1].get_xdata()) == n_bins
        plt.close("all")

    def test_overlay_raw_and_calibrated_single_diagonal(self):
        rm, _, _, Xt, yt = _fitted()
        raw = rm.eval(X=Xt, labels=yt, add_metrics=True)
        cal = rm.eval(X=Xt, labels=yt, add_metrics=True, use_calibrated=True)
        rm_plot = aa.ReliabilityModelPlot()
        fig, ax = rm_plot.reliability_diagram(df_eval=raw, label="raw", color="tab:red")
        fig2, ax2 = rm_plot.reliability_diagram(df_eval=cal, label="calibrated", ax=ax)
        assert ax2 is ax and fig2 is fig
        labels = [line.get_label() for line in ax.get_lines()]
        assert labels.count("perfect calibration") == 1
        assert len(labels) == 3
        plt.close("all")

    def test_overlay_keeps_the_two_colors_apart(self):
        rm, _, _, Xt, yt = _fitted()
        raw = rm.eval(X=Xt, labels=yt)
        cal = rm.eval(X=Xt, labels=yt, use_calibrated=True)
        rm_plot = aa.ReliabilityModelPlot()
        fig, ax = rm_plot.reliability_diagram(df_eval=raw, label="raw", color="tab:red")
        rm_plot.reliability_diagram(df_eval=cal, label="calibrated", color="tab:green", ax=ax)
        colors = [to_rgba(line.get_color()) for line in ax.get_lines()[-2:]]
        assert colors == [to_rgba("tab:red"), to_rgba("tab:green")]
        plt.close("all")

    def test_overlay_with_metrics_annotates_each_curve(self):
        rm, _, _, Xt, yt = _fitted()
        raw = rm.eval(X=Xt, labels=yt, add_metrics=True)
        cal = rm.eval(X=Xt, labels=yt, add_metrics=True, use_calibrated=True)
        rm_plot = aa.ReliabilityModelPlot()
        fig, ax = rm_plot.reliability_diagram(df_eval=raw, label="raw", color="tab:red")
        rm_plot.reliability_diagram(df_eval=cal, label="calibrated", color="tab:green", ax=ax)
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert sum(("Brier" in t and "ECE" in t) for t in texts) == 2
        plt.close("all")

    def test_all_params_together_on_a_new_figure(self, df_eval):
        fig, ax = aa.ReliabilityModelPlot().reliability_diagram(
            df_eval=df_eval, figsize=(4, 4), color="tab:purple", label="cal", title="calibration")
        assert tuple(fig.get_size_inches()) == pytest.approx((4, 4))
        assert ax.get_title() == "calibration"
        assert to_rgba(ax.get_lines()[-1].get_color()) == to_rgba("tab:purple")
        assert "cal" in [line.get_label() for line in ax.get_lines()]
        plt.close("all")

    def test_ax_wins_over_figsize(self, df_eval):
        fig, ax = plt.subplots(figsize=(3, 3))
        res = aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, figsize=(9, 2), ax=ax)
        assert res[0] is fig and res[1] is ax
        assert tuple(fig.get_size_inches()) == pytest.approx((3, 3))   # the passed ax keeps its size
        plt.close("all")

    # Negative interactions
    def test_invalid_color_with_valid_ax_raises(self, df_eval):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="color"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, color="chartreuse!",
                                                          label="cal", ax=ax)
        plt.close("all")

    def test_invalid_figsize_with_valid_ax_raises(self, df_eval):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="figsize"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, figsize=(0, 3), ax=ax)
        plt.close("all")

    def test_invalid_label_on_a_metrics_table_raises(self):
        rm, _, _, Xt, yt = _fitted()
        df_eval = rm.eval(X=Xt, labels=yt, add_metrics=True)
        with pytest.raises(ValueError, match="'label'"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, label=None,
                                                          color="tab:red")

    def test_invalid_title_with_every_other_param_valid_raises(self, df_eval):
        with pytest.raises(ValueError, match="title"):
            aa.ReliabilityModelPlot().reliability_diagram(
                df_eval=df_eval, figsize=(4, 4), color="tab:red", label="cal", title=7)

    def test_figure_as_ax_with_every_other_param_valid_raises(self, df_eval):
        with pytest.raises(ValueError, match="ax"):
            aa.ReliabilityModelPlot().reliability_diagram(
                df_eval=df_eval, figsize=(4, 4), color="tab:red", label="cal", ax=plt.figure())
        plt.close("all")

    def test_metrics_table_without_empirical_pos_raises(self):
        rm, _, _, Xt, yt = _fitted()
        df_eval = rm.eval(X=Xt, labels=yt, add_metrics=True).drop(columns=["empirical_pos"])
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, label="cal")

    def test_calibrated_table_without_bin_column_raises(self):
        rm, _, _, Xt, yt = _fitted()
        df_eval = rm.eval(X=Xt, labels=yt, use_calibrated=True).drop(columns=["bin"])
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=df_eval, color="tab:red",
                                                          label="cal")


class TestOodHist:
    def test_returns_fig_ax(self):
        assert _is_fig_ax(aa.ReliabilityModelPlot().ood_hist(df_rel=_df_rel()))
        plt.close("all")

    def test_params(self):
        fig, ax = plt.subplots()
        res = aa.ReliabilityModelPlot().ood_hist(
            df_rel=_df_rel(), figsize=(5, 3), bins=10, color="tab:blue", title="ood", ax=ax)
        assert res[1] is ax
        plt.close("all")

    @pytest.mark.parametrize("bins", [0, -3, 2.5])
    def test_bad_bins_raises(self, bins):
        with pytest.raises(ValueError, match="'bins'"):
            aa.ReliabilityModelPlot().ood_hist(df_rel=_df_rel(), bins=bins)

    def test_missing_col_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().ood_hist(df_rel=pd.DataFrame({"score": [0.5]}))


class TestTrustMap:
    def test_returns_fig_ax(self):
        assert _is_fig_ax(aa.ReliabilityModelPlot().trust_map(df_rel=_df_rel()))
        plt.close("all")

    def test_params(self):
        fig, ax = plt.subplots()
        res = aa.ReliabilityModelPlot().trust_map(
            df_rel=_df_rel(), figsize=(5, 5), title="trust", ax=ax)
        assert res[1] is ax
        plt.close("all")

    def test_missing_col_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ReliabilityModelPlot().trust_map(df_rel=pd.DataFrame({"score": [0.5]}))
