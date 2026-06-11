"""Tests for SeqMut.suggest (top mutations shifting toward the test-class profile)."""
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestSeqMutSuggest:
    def test_columns(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert list(df.columns) == ut.COLS_SEQMUT_SCAN

    @pytest.mark.parametrize("n", [1, 3, 5, 15])
    def test_n_limits_rows(self, df_seq_pos, df_feat, n):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, n=n, region="tmd")
        assert len(df) <= n

    def test_sorted_by_shift_score(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, n=10, region="tmd")
        assert np.all(np.diff(df[ut.COL_SHIFT_SCORE].to_numpy()) <= 1e-9)

    def test_weight_feat_importance(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                                 weight="feat_importance")
        assert len(df) > 0

    def test_weight_abs_auc(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                                 weight="abs_auc")
        assert len(df) > 0

    def test_region_restriction(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert set(df[ut.COL_REGION]) == {ut.COL_TMD}

    def test_to_aa(self, df_seq_pos, df_feat):
        df = aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                                 to_aa=["A", "L"])
        assert set(df[ut.COL_TO_AA]).issubset({"A", "L"})

    # Negative cases
    def test_invalid_n_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, n=0)

    def test_invalid_weight_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat, weight="bogus")

    def test_missing_weight_column_raises(self, df_seq_pos, df_feat):
        df_feat2 = df_feat.drop(columns=[ut.COL_FEAT_IMPORT])
        with pytest.raises(ValueError):
            aa.SeqMut().suggest(df_seq=df_seq_pos, df_feat=df_feat2, weight="feat_importance")


class TestSeqMutSuggestGoldenValues:
    def test_shift_score_definition(self, df_seq_pos, df_feat):
        # shift_score sign should follow mean_dif-aligned movement; the engine output is
        # internally consistent with scan's shift_score on the same plan.
        sm = aa.SeqMut()
        df_sug = sm.suggest(df_seq=df_seq_pos, df_feat=df_feat, n=5, region="tmd")
        df_scan = sm.scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        top = df_sug.iloc[0]
        match = df_scan[df_scan[ut.COL_MUTATION] == top[ut.COL_MUTATION]]
        assert match[ut.COL_SHIFT_SCORE].iloc[0] == pytest.approx(top[ut.COL_SHIFT_SCORE])
