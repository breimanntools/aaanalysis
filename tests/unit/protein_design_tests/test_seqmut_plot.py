"""Tests for SeqMutPlot (mutation_landscape / residue_mutation_impact)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa


class TestSeqMutPlotInit:
    def test_default(self):
        assert aa.SeqMutPlot()._verbose is False

    def test_verbose(self):
        assert aa.SeqMutPlot(verbose=True)._verbose is True

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aa.SeqMutPlot(verbose="x")


class TestMutationLandscape:
    def test_returns_axes(self, df_scan):
        ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_entry(self, df_scan):
        ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan, entry="P2")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_on_passed_ax(self, df_scan):
        fig, ax = plt.subplots()
        out = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan, ax=ax)
        assert out is ax
        plt.close("all")

    def test_figsize(self, df_scan):
        ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan, figsize=(6, 4))
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_cmap(self, df_scan):
        ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan, cmap="magma")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_unknown_entry_raises(self, df_scan):
        with pytest.raises(ValueError):
            aa.SeqMutPlot().mutation_landscape(df_scan=df_scan, entry="NOPE")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMutPlot().mutation_landscape(df_scan=None)


class TestResidueMutationImpact:
    def test_returns_axes(self, df_scan):
        ax = aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=12)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_entry_and_pos(self, df_scan):
        ax = aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, entry="P2", pos=12)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_color(self, df_scan):
        ax = aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=12, color="green")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_figsize(self, df_scan):
        ax = aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=12, figsize=(5, 4))
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_invalid_pos_raises(self, df_scan):
        with pytest.raises(ValueError):
            aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=0)

    def test_absent_pos_raises(self, df_scan):
        with pytest.raises(ValueError):
            aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=9999)
