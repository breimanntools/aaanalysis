"""Unit tests for the ModelEvaluatorPlot class (CI bars + paired-comparison delta bars)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False


def _data(n_per_class=20, n_feat=6, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.6, 1.0, size=(n_per_class, n_feat)),
                   rng.normal(-0.6, 1.0, size=(n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


@pytest.fixture(scope="module")
def evaluated():
    X, labels = _data()
    me = aa.ModelEvaluator(models=["rf", "svm", "log_reg"], random_state=0)
    df_eval = me.run(X, labels, n_rounds=2)
    df_cmp = me.eval(metric="mcc")
    return df_eval, df_cmp


class TestModelEvaluatorPlotScores:
    def test_returns_fig_ax(self, evaluated):
        df_eval, _ = evaluated
        out = aa.ModelEvaluatorPlot.scores(df_eval=df_eval)
        assert isinstance(out, ut.FigAxResult)
        plt.close("all")

    def test_figsize_and_colors(self, evaluated):
        df_eval, _ = evaluated
        fig, ax = aa.ModelEvaluatorPlot.scores(df_eval=df_eval, figsize=(5, 3),
                                            colors=["tab:blue", "tab:orange", "tab:green"])
        assert fig is not None
        plt.close("all")

    def test_metrics_subset(self, evaluated):
        df_eval, _ = evaluated
        fig, ax = aa.ModelEvaluatorPlot.scores(df_eval=df_eval, metrics=["mcc"])
        assert fig is not None
        plt.close("all")

    def test_invalid_missing_columns(self):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.scores(df_eval=None)

    def test_invalid_figsize(self, evaluated):
        df_eval, _ = evaluated
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.scores(df_eval=df_eval, figsize=(0, 0))


class TestModelEvaluatorPlotCompare:
    def test_returns_fig_ax(self, evaluated):
        _, df_cmp = evaluated
        out = aa.ModelEvaluatorPlot.compare(df_eval=df_cmp)
        assert isinstance(out, ut.FigAxResult)
        plt.close("all")

    def test_colors_and_alpha(self, evaluated):
        _, df_cmp = evaluated
        fig, ax = aa.ModelEvaluatorPlot.compare(df_eval=df_cmp, figsize=(5, 3),
                                             colors=["tab:green", "tab:red"], alpha=0.1)
        assert fig is not None
        plt.close("all")

    def test_invalid_colors_length(self, evaluated):
        _, df_cmp = evaluated
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.compare(df_eval=df_cmp, colors=["tab:green"])

    def test_invalid_alpha(self, evaluated):
        _, df_cmp = evaluated
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.compare(df_eval=df_cmp, alpha=2.0)

    def test_invalid_missing_columns(self):
        with pytest.raises(ValueError):
            aa.ModelEvaluatorPlot.compare(df_eval=None)
