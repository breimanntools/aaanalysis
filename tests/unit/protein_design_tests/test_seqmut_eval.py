"""Tests for SeqMut.eval (stable-vs-disruptive classification, per-region summary)."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestSeqMutEval:
    def test_columns(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        assert list(df.columns) == ut.COLS_SEQMUT_EVAL

    def test_one_row_per_entry_region(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        assert len(df) == df_scan.groupby([ut.COL_ENTRY, ut.COL_REGION]).ngroups

    def test_frac_in_unit_interval(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        assert ((df[ut.COL_FRAC_DISRUPTIVE] >= 0) & (df[ut.COL_FRAC_DISRUPTIVE] <= 1)).all()

    def test_counts_consistent(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        assert (df[ut.COL_N_DISRUPTIVE] <= df[ut.COL_N_MUT]).all()

    def test_threshold_explicit(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan, th=0.0)
        # th=0 -> every mutation disruptive
        assert (df[ut.COL_N_DISRUPTIVE] == df[ut.COL_N_MUT]).all()

    def test_high_threshold_none_disruptive(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan, th=10.0)
        assert (df[ut.COL_N_DISRUPTIVE] == 0).all()

    def test_n_mut_sums_to_scan(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        assert df[ut.COL_N_MUT].sum() == len(df_scan)

    # Negative cases
    def test_none_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMut().eval(df_scan=None)

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            aa.SeqMut().eval(df_scan=pd.DataFrame({"x": [1]}))

    def test_negative_threshold_raises(self, df_scan):
        with pytest.raises(ValueError):
            aa.SeqMut().eval(df_scan=df_scan, th=-1.0)


class TestSeqMutEvalGoldenValues:
    def test_default_threshold_is_tertile(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        th = float(np.quantile(df_scan[ut.COL_DELTA_CPP], 2 / 3))
        n_disruptive = int((df_scan[ut.COL_DELTA_CPP] >= th).sum())
        assert df[ut.COL_N_DISRUPTIVE].sum() == n_disruptive

    def test_mean_delta_cpp(self, df_scan):
        df = aa.SeqMut().eval(df_scan=df_scan)
        entry, region = df[ut.COL_ENTRY].iloc[0], df[ut.COL_REGION].iloc[0]
        sub = df_scan[(df_scan[ut.COL_ENTRY] == entry) & (df_scan[ut.COL_REGION] == region)]
        assert df[ut.COL_MEAN_DELTA_CPP].iloc[0] == pytest.approx(sub[ut.COL_DELTA_CPP].mean())
