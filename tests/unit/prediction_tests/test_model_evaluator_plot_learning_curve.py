"""This is a script to test ModelEvaluatorPlot.learning_curve()."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

METRICS = ["accuracy", "mcc", "f1"]


# I Helper Functions
def _data(n_per_class=20, n_feat=4, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.8, 1.0, size=(n_per_class, n_feat)),
                   rng.normal(-0.8, 1.0, size=(n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


@pytest.fixture(scope="module")
def df_curve():
    X, labels = _data()
    me = aa.ModelEvaluator(models=["log_reg", "svm"], random_state=0, verbose=False)
    return me.learning_curve(X, labels, train_sizes=[0.25, 0.5, 1.0], metrics=METRICS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# II Main Functions
class TestPlotLearningCurve:
    """Normal cases: one parameter per test (positive and negative)."""

    # Positive tests
    def test_df_curve(self, df_curve):
        out = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        assert isinstance(out, ut.FigAxResult)
        fig, ax = out
        assert ax.get_figure() is fig
        assert len(ax.get_lines()) == 2

    @settings(max_examples=3, deadline=None)
    @given(metric=some.sampled_from(METRICS))
    def test_metric(self, df_curve, metric):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric=metric)
        assert ax.get_ylabel() == metric
        sub = df_curve[(df_curve[ut.COL_METRIC] == metric) & (df_curve[ut.COL_MODEL] == "log_reg")]
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), sub[ut.COL_SCORE].to_numpy())
        plt.close("all")

    def test_metric_default(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        assert ax.get_ylabel() == "mcc"
        df_no_mcc = df_curve[df_curve[ut.COL_METRIC] != "mcc"]
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_no_mcc)
        assert ax.get_ylabel() == "accuracy"

    @settings(max_examples=3, deadline=None)
    @given(width=some.floats(min_value=2, max_value=10), height=some.floats(min_value=2, max_value=8))
    def test_figsize(self, df_curve, width, height):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=(width, height))
        np.testing.assert_allclose(fig.get_size_inches(), [width, height])
        plt.close("all")

    def test_colors(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=["tab:red", "tab:green"])
        colors = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()]
        assert colors == [matplotlib.colors.to_hex("tab:red"), matplotlib.colors.to_hex("tab:green")]

    @settings(max_examples=2, deadline=None)
    @given(show_ci=some.booleans())
    def test_show_ci(self, df_curve, show_ci):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, show_ci=show_ci)
        assert len(ax.collections) == (2 if show_ci else 0)
        plt.close("all")

    def test_x_is_train_size(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [8, 16, 32])
        assert "Training size" in ax.get_xlabel()

    # Negative tests
    def test_invalid_df_curve(self, df_curve):
        for df in [None, "df", df_curve.drop(columns=ut.COL_TRAIN_SIZE),
                   df_curve.drop(columns=ut.COL_CI_LOW), df_curve.iloc[0:0]]:
            with pytest.raises(ValueError):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df)

    def test_invalid_metric(self, df_curve):
        for metric in ["roc_auc", "not_a_metric", 1]:
            with pytest.raises(ValueError):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric=metric)

    def test_invalid_figsize(self, df_curve):
        for figsize in [(0, 0), (-1, 4), "big", (4,)]:
            with pytest.raises(ValueError):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=figsize)

    def test_invalid_colors(self, df_curve):
        for colors in [["tab:red"], ["not_a_color", "tab:red"], "tab:red"]:
            with pytest.raises(ValueError):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=colors)

    def test_invalid_show_ci(self, df_curve):
        for show_ci in [None, 1, "yes"]:
            with pytest.raises(ValueError):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, show_ci=show_ci)


class TestPlotLearningCurveComplex:
    """Combinations and edge interactions."""

    # Positive tests
    def test_single_model_without_legend(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="log_reg", random_state=0, verbose=False)
        df = me.learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"])
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df)
        assert ax.get_legend() is None
        assert len(ax.get_lines()) == 1

    def test_multi_model_legend(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="f1")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["log_reg", "svm"]

    def test_ci_none_falls_back_to_std_band(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["log_reg", "svm"], random_state=0, verbose=False)
        df = me.learning_curve(X, labels, train_sizes=[0.25, 1.0], metrics=["mcc"], ci=None)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=True)
        for coll in ax.collections:
            assert np.isfinite(coll.get_paths()[0].vertices).all()

    def test_all_params_combined(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="accuracy", figsize=(5, 3),
                                                       colors=["black", "gray", "red"], show_ci=False)
        assert len(ax.collections) == 0 and ax.get_ylabel() == "accuracy"

    def test_unsorted_rows_are_plotted_by_size(self, df_curve):
        df_shuffled = df_curve.sample(frac=1.0, random_state=0)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_shuffled, metric="mcc")
        for line in ax.get_lines():
            assert np.all(np.diff(line.get_xdata()) > 0)

    # Negative tests
    def test_invalid_metric_after_filtering(self, df_curve):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve[df_curve[ut.COL_METRIC] == "mcc"],
                                                 metric="f1")

    def test_invalid_colors_with_show_ci(self, df_curve):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=["red"], show_ci=True)

    def test_invalid_df_from_run(self):
        X, labels = _data()
        df_eval = aa.ModelEvaluator(models="log_reg", random_state=0, verbose=False).run(X, labels)
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_eval)

    def test_invalid_figsize_with_metric(self, df_curve):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="mcc", figsize=None)

    def test_invalid_show_ci_with_colors(self, df_curve):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=["red", "blue"], show_ci="false")
