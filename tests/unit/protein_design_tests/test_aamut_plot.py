"""Tests for AAMutPlot (substitution_matrix / scale_ranking / aa_comparison)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestAAMutPlotInit:
    def test_default(self):
        amp = aa.AAMutPlot()
        assert amp._verbose is False

    def test_verbose(self):
        amp = aa.AAMutPlot(verbose=True)
        assert amp._verbose is True

    def test_df_scales(self):
        df = ut.load_default_scales()
        amp = aa.AAMutPlot(df_scales=df)
        assert amp.df_scales is df

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aa.AAMutPlot(verbose="x")


class TestSubstitutionMatrix:
    def test_returns_axes(self, df_impact):
        ax = aa.AAMutPlot().substitution_matrix(df_impact=df_impact)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_on_passed_ax(self, df_impact):
        fig, ax = plt.subplots()
        out = aa.AAMutPlot().substitution_matrix(df_impact=df_impact, ax=ax)
        assert out is ax
        plt.close("all")

    def test_figsize(self, df_impact):
        ax = aa.AAMutPlot().substitution_matrix(df_impact=df_impact, figsize=(4, 4))
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_cmap(self, df_impact):
        ax = aa.AAMutPlot().substitution_matrix(df_impact=df_impact, cmap="magma")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            aa.AAMutPlot().substitution_matrix(df_impact=None)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aa.AAMutPlot().substitution_matrix(df_impact=pd.DataFrame(columns=ut.COLS_AAMUT))


class TestScaleRanking:
    def test_returns_axes(self, df_impact):
        ax = aa.AAMutPlot().scale_ranking(df_impact=df_impact, top_n=3)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_top_n(self, df_impact):
        ax = aa.AAMutPlot().scale_ranking(df_impact=df_impact, top_n=2)
        assert len([p for p in ax.patches]) <= 4
        plt.close("all")

    def test_color(self, df_impact):
        ax = aa.AAMutPlot().scale_ranking(df_impact=df_impact, color="blue")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_on_passed_ax(self, df_impact):
        fig, ax = plt.subplots()
        out = aa.AAMutPlot().scale_ranking(df_impact=df_impact, ax=ax)
        assert out is ax
        plt.close("all")

    def test_invalid_top_n(self, df_impact):
        with pytest.raises(ValueError):
            aa.AAMutPlot().scale_ranking(df_impact=df_impact, top_n=0)


class TestAAComparison:
    def test_returns_axes(self, df_impact):
        ax = aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="M", to_aa="V")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_top_n(self, df_impact):
        ax = aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="M", to_aa="V", top_n=2)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_figsize(self, df_impact):
        ax = aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="M", to_aa="A",
                                          figsize=(5, 5))
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_missing_pair_raises(self, df_impact):
        with pytest.raises(ValueError):
            aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="W", to_aa="C")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            aa.AAMutPlot().aa_comparison(df_impact=None, from_aa="M", to_aa="V")
