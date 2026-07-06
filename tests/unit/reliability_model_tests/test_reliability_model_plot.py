"""Unit tests for ReliabilityModelPlot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().ranking(df_rel=_df_rel(), names=["only-one"])

    @pytest.mark.parametrize("top_n", [0, -2])
    def test_bad_top_n_raises(self, top_n):
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().ranking(df_rel=_df_rel(), top_n=top_n)

    def test_missing_col_raises(self):
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().ranking(df_rel=pd.DataFrame({"score": [0.5]}))


class TestReliabilityDiagram:
    def test_returns_fig_ax(self):
        assert _is_fig_ax(aa.ReliabilityModelPlot().reliability_diagram(df_eval=_df_eval()))
        plt.close("all")

    def test_params(self):
        fig, ax = plt.subplots()
        res = aa.ReliabilityModelPlot().reliability_diagram(
            df_eval=_df_eval(), figsize=(4, 4), color="tab:red", title="cal", ax=ax)
        assert res[1] is ax
        plt.close("all")

    def test_bad_df_raises(self):
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=pd.DataFrame({"x": [1]}))

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().reliability_diagram(df_eval=[1, 2, 3])


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
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().ood_hist(df_rel=_df_rel(), bins=bins)

    def test_missing_col_raises(self):
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            aa.ReliabilityModelPlot().trust_map(df_rel=pd.DataFrame({"score": [0.5]}))
