"""This is a script to test residual protein_engineering (AAMut / SeqMut) branch arms.

All exercised exclusively through the public API (aa.AAMut / aa.AAMutPlot /
aa.SeqMut / aa.SeqMutPlot). Fixtures come from the subpackage conftest.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


# I AAMut branch arms
class TestAAMutBranch:
    """Cover the verbose-output arm and the df_cat lookup arm of AAMut.run."""

    def test_run_verbose(self):
        # _aamut.py:127 — verbose print branch; backend aam.py:14 df_cat lookup taken
        df_impact = aa.AAMut(verbose=True).run(from_aa="M", to_aa="V")
        assert len(df_impact) > 0
        assert ut.COL_CAT in df_impact.columns


# II AAMutPlot.aa_comparison branch arms
class TestAAMutPlotBranch:
    """Cover the empty-subset raise and the signed-color arm of aa_comparison."""

    def test_aa_comparison_no_rows(self, df_impact):
        # _aamut_plot.py:192-193 — no rows for the requested from->to pair
        with pytest.raises(ValueError, match="No rows"):
            aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="W", to_aa="W")
        plt.close("all")

    def test_aa_comparison_signed_colors(self, df_impact):
        # _aamut_plot.py:195-196 — bars colored by sign (positive + negative deltas)
        _, ax = aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="M", to_aa="V", top_n=5)
        assert ax is not None
        plt.close("all")

    def test_aa_comparison_provided_ax(self, df_impact):
        # _aamut_plot.py:196->198 — pre-supplied ax (skip the ax-is-None creation)
        _, ax_in = plt.subplots()
        _, ax = aa.AAMutPlot().aa_comparison(df_impact=df_impact, from_aa="M", to_aa="V", ax=ax_in)
        assert ax is ax_in
        plt.close("all")


# III SeqMut backend region arms (jmd_n / jmd_c)
class TestSeqMutScanRegionBranch:
    """Cover the jmd_n / jmd_c region arms of get_region_positions + verbose."""

    def test_scan_region_jmd_n(self, df_seq_pos, df_feat_multipart):
        # seqm.py:46 — region 'jmd_n' positions
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat_multipart, region="jmd_n")
        assert len(df_scan) > 0
        assert set(df_scan[ut.COL_REGION]) <= {"jmd_n"}

    def test_scan_region_jmd_c(self, df_seq_pos, df_feat_multipart):
        # seqm.py:48 — region 'jmd_c' positions
        df_scan = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat_multipart, region="jmd_c")
        assert len(df_scan) > 0
        assert set(df_scan[ut.COL_REGION]) <= {"jmd_c"}

    def test_scan_verbose(self, df_seq_pos, df_feat):
        # _seqmut.py:268 — verbose print branch in scan
        df_scan = aa.SeqMut(verbose=True).scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert len(df_scan) > 0


# IV SeqMut.suggest / eval branch arms
class TestSeqMutSuggestEvalBranch:
    """Cover the empty-region raise in suggest and the verbose arm in eval."""

    def test_suggest_empty_region(self, df_seq_pos, df_feat):
        # _seqmut.py:332 — no scannable positions for the given region
        with pytest.raises(ValueError, match="No scannable positions"):
            aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region=[9999])

    def test_eval_verbose(self, df_scan):
        # _seqmut.py:374 — verbose print branch in eval
        df_eval = aa.SeqMut(verbose=True).eval(df_scan=df_scan)
        assert len(df_eval) > 0


# V SeqMutPlot branch arms
class TestSeqMutPlotBranch:
    """Cover the empty-df_scan raise and the no-mutations-at-pos raise."""

    def test_landscape_empty_df_scan(self):
        # _seqmut_plot.py:17 — empty df_scan raises
        empty = pd.DataFrame(columns=ut.COLS_SEQMUT_SCAN)
        with pytest.raises(ValueError, match="should not be empty"):
            aa.SeqMutPlot().mutation_landscape(df_scan=empty)
        plt.close("all")

    def test_residue_provided_ax(self, df_scan):
        # _seqmut_plot.py:153->155 — pre-supplied ax (skip the ax-is-None creation)
        pos = int(df_scan[ut.COL_POS].iloc[0])
        _, ax_in = plt.subplots()
        _, ax = aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=pos, ax=ax_in)
        assert ax is ax_in
        plt.close("all")

    def test_residue_no_mutations_at_pos(self, df_scan):
        # _seqmut_plot.py:153 — no scanned mutations at the requested position
        bad_pos = int(df_scan[ut.COL_POS].max()) + 50
        with pytest.raises(ValueError, match="No scanned mutations"):
            aa.SeqMutPlot().residue_mutation_impact(df_scan=df_scan, pos=bad_pos)
        plt.close("all")
