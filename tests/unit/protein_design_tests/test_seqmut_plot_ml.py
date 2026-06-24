"""Tests for the model-aware SeqMutPlot methods: the ΔP mutation_landscape, variant_impact
(ranked-variant bar) and epistasis (pairwise non-additivity)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df_scan_model(df_seq_pos, df_feat, model_tuple):
    return aa.SeqMut(model=model_tuple).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")


class TestMutationLandscapeModel:
    def test_returns_fig_ax(self, df_scan_model):
        out = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan_model)
        fig, ax = out
        assert isinstance(fig, plt.Figure)

    def test_title_reports_wt_prediction(self, df_scan_model):
        _, ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan_model, entry="P1")
        assert "Mutation Scan for P1" in ax.get_title() and "%" in ax.get_title()

    def test_class_names_in_title(self, df_scan_model):
        _, ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan_model,
                                                   class_names=("NONSUB", "SUB"))
        assert ("NONSUB" in ax.get_title()) or ("SUB" in ax.get_title())

    def test_cmap_and_figsize(self, df_scan_model):
        _, ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan_model, cmap="coolwarm",
                                                   figsize=(8, 4))
        assert ax is not None

    def test_ax_passthrough(self, df_scan_model):
        _, ax0 = plt.subplots()
        _, ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan_model, ax=ax0)
        assert ax is ax0

    def test_model_free_uses_delta_cpp(self, df_scan):
        _, ax = aa.SeqMutPlot().mutation_landscape(df_scan=df_scan)
        assert ax is not None


class TestVariantImpact:
    def test_returns_fig_ax(self, df_variant):
        fig, ax = aa.SeqMutPlot().variant_impact(df_variant=df_variant)
        assert isinstance(fig, plt.Figure)

    def test_top_n(self, df_variant):
        _, ax = aa.SeqMutPlot().variant_impact(df_variant=df_variant, n=2)
        assert len(ax.patches) == 2

    def test_entry_and_figsize(self, df_variant):
        _, ax = aa.SeqMutPlot().variant_impact(df_variant=df_variant, entry="P1", figsize=(6, 4))
        assert ax is not None

    def test_ax_passthrough(self, df_variant):
        _, ax0 = plt.subplots()
        _, ax = aa.SeqMutPlot().variant_impact(df_variant=df_variant, ax=ax0)
        assert ax is ax0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMutPlot().variant_impact(df_variant=pd.DataFrame(columns=ut.COLS_SEQMUT_VARIANT))


class TestEpistasis:
    def test_returns_fig_ax(self, df_variant):
        fig, ax = aa.SeqMutPlot().epistasis(df_variant=df_variant)
        assert isinstance(fig, plt.Figure)

    def test_entry_cmap_figsize(self, df_variant):
        _, ax = aa.SeqMutPlot().epistasis(df_variant=df_variant, entry="P1", cmap="bwr",
                                          figsize=(5, 5))
        assert "epistasis" in ax.get_title().lower()

    def test_ax_passthrough(self, df_variant):
        _, ax0 = plt.subplots()
        _, ax = aa.SeqMutPlot().epistasis(df_variant=df_variant, ax=ax0)
        assert ax is ax0

    def test_too_few_singles_raises(self, df_seq_pos, df_feat, model_tuple):
        variants = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_VARIANT: ["a"],
                                 ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df_v = aa.SeqMut(model=model_tuple).combine(df_seq=df_seq_pos, variants=variants,
                                                    df_feat=df_feat)
        with pytest.raises(ValueError):
            aa.SeqMutPlot().epistasis(df_variant=df_v)
