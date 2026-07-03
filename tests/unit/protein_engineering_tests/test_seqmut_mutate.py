"""Tests for SeqMut.mutate (apply specific single/batch mutations + ΔCPP)."""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _muts(entries=("P1",), positions=(12,), to=("K",)):
    return pd.DataFrame({ut.COL_ENTRY: list(entries), ut.COL_POS: list(positions),
                         ut.COL_TO_AA: list(to)})


class TestSeqMutMutate:
    def test_returns_history(self, df_seq_pos):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts())
        assert ut.COL_SEQ_MUT in df.columns and ut.COL_FROM_AA in df.columns

    def test_mutation_applied(self, df_seq_pos):
        muts = _muts(("P1",), (12,), ("K",))
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=muts)
        assert df[ut.COL_SEQ_MUT].iloc[0][11] == "K"

    def test_from_aa_derived(self, df_seq_pos):
        seq = df_seq_pos.set_index(ut.COL_ENTRY).loc["P1", ut.COL_SEQ]
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("K",)))
        assert df[ut.COL_FROM_AA].iloc[0] == seq[11]

    def test_mutation_label(self, df_seq_pos):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("K",)))
        r = df.iloc[0]
        assert r[ut.COL_MUTATION] == f"{r[ut.COL_FROM_AA]}12K"

    def test_batch(self, df_seq_pos):
        muts = _muts(("P1", "P2"), (12, 13), ("K", "D"))
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=muts)
        assert len(df) == 2

    def test_with_df_feat_adds_delta(self, df_seq_pos, df_feat):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("K",)),
                                df_feat=df_feat)
        assert ut.COL_DELTA_CPP in df.columns and ut.COL_SHIFT_SCORE in df.columns

    def test_without_df_feat_no_delta(self, df_seq_pos):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts())
        assert ut.COL_DELTA_CPP not in df.columns

    def test_jmd_n_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("K",)),
                                df_feat=df_feat, jmd_n_len=8)
        assert len(df) == 1

    def test_jmd_c_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("K",)),
                                df_feat=df_feat, jmd_c_len=8)
        assert len(df) == 1

    # Negative cases
    def test_unknown_entry_raises(self, df_seq_pos):
        with pytest.raises(ValueError):
            aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("NOPE",), (1,), ("A",)))

    def test_out_of_range_pos_raises(self, df_seq_pos):
        with pytest.raises(ValueError):
            aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (9999,), ("A",)))

    def test_bad_to_aa_raises(self, df_seq_pos):
        with pytest.raises(ValueError):
            aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=_muts(("P1",), (12,), ("Z",)))

    def test_missing_columns_raises(self, df_seq_pos):
        with pytest.raises(ValueError):
            aa.SeqMut().mutate(df_seq=df_seq_pos, mutations=pd.DataFrame({ut.COL_ENTRY: ["P1"]}))


class TestSeqMutMutateGoldenValues:
    def test_delta_matches_scan(self, df_seq_pos, df_feat):
        sm = aa.SeqMut()
        # A single mutation's delta_cpp equals its value in a full scan.
        muts = _muts(("P1",), (12,), ("K",))
        df_m = sm.mutate(df_seq=df_seq_pos, mutations=muts, df_feat=df_feat)
        df_scan = sm.scan(df_seq=df_seq_pos, df_feat=df_feat, region=[12])
        scan_val = df_scan[(df_scan[ut.COL_ENTRY] == "P1") &
                           (df_scan[ut.COL_TO_AA] == "K") &
                           (df_scan[ut.COL_POS] == 12)][ut.COL_DELTA_CPP].iloc[0]
        assert df_m[ut.COL_DELTA_CPP].iloc[0] == pytest.approx(scan_val, abs=1e-9)
