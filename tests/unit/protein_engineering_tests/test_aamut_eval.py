"""Tests for AAMut.eval (per-scale substitution-sensitivity ranking)."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestAAMutEval:
    """Normal cases."""

    def test_returns_dataframe(self, df_impact):
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        assert isinstance(df_eval, pd.DataFrame)

    def test_columns(self, df_impact):
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        assert list(df_eval.columns) == [ut.COL_SCALE_ID, ut.COL_MEAN_DELTA_CPP]

    def test_one_row_per_scale(self, df_impact):
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        assert len(df_eval) == df_impact[ut.COL_SCALE_ID].nunique()

    def test_sorted_descending(self, df_impact):
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        vals = df_eval[ut.COL_MEAN_DELTA_CPP].to_numpy()
        assert np.all(np.diff(vals) <= 1e-12)

    def test_nonnegative(self, df_impact):
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        assert (df_eval[ut.COL_MEAN_DELTA_CPP] >= 0).all()

    def test_none_raises(self):
        with pytest.raises(ValueError):
            aa.AAMut().eval(df_impact=None)

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            aa.AAMut().eval(df_impact=pd.DataFrame({"x": [1]}))


class TestAAMutEvalGoldenValues:
    """Hand-computed expectations."""

    def test_mean_abs_delta(self):
        df_impact = aa.AAMut().run(from_aa=["M", "L"], to_aa="V")
        df_eval = aa.AAMut().eval(df_impact=df_impact)
        scale = df_eval[ut.COL_SCALE_ID].iloc[0]
        expected = df_impact[df_impact[ut.COL_SCALE_ID] == scale][ut.COL_ABS_DELTA].mean()
        assert df_eval[ut.COL_MEAN_DELTA_CPP].iloc[0] == pytest.approx(expected)
