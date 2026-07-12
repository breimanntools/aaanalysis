"""Tests for SeqMut.combine (score combined multi-mutation variants)."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _variants():
    """Two variants on P1: a double (11A, 12P) and a single (13K)."""
    return pd.DataFrame({
        ut.COL_ENTRY: ["P1", "P1", "P1"],
        ut.COL_VARIANT: ["v1", "v1", "v2"],
        ut.COL_POS: [11, 12, 13],
        ut.COL_TO_AA: ["A", "P", "K"],
    })


class TestSeqMutCombine:
    def test_columns_model_free(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert list(df.columns) == ut.COLS_SEQMUT_VARIANT

    def test_columns_with_model(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).combine(df_seq=df_seq_pos, variants=_variants(),
                                                  df_feat=df_feat)
        assert ut.COL_DELTA_PRED in df.columns

    def test_one_row_per_variant(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert len(df) == 2

    def test_variant_label_joins_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        labels = set(df[ut.COL_VARIANT])
        assert any("+" in lab and lab.count("+") == 1 for lab in labels)  # the double

    def test_n_mut_counts_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        assert set(df[ut.COL_N_MUT]) == {2, 1}

    def test_combined_sequence_carries_all_mutations(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat)
        double = df[df[ut.COL_N_MUT] == 2].iloc[0]
        assert double[ut.COL_SEQ_MUT][10] == "A" and double[ut.COL_SEQ_MUT][11] == "P"

    def test_sorted_by_score(self, df_seq_pos, df_feat, model_tuple):
        df = aa.SeqMut(model=model_tuple).combine(df_seq=df_seq_pos, variants=_variants(),
                                                  df_feat=df_feat)
        assert np.all(np.diff(df[ut.COL_DELTA_PRED].to_numpy()) <= 1e-9)

    def test_jmd_n_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                 jmd_n_len=8)
        assert len(df) == 2

    def test_jmd_c_len(self, df_seq_pos, df_feat):
        df = aa.SeqMut().combine(df_seq=df_seq_pos, variants=_variants(), df_feat=df_feat,
                                 jmd_c_len=8)
        assert len(df) == 2

    # Negative cases
    def test_duplicate_position_in_variant_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1", "P1"], ut.COL_VARIANT: ["v", "v"],
                            ut.COL_POS: [11, 11], ut.COL_TO_AA: ["A", "P"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)

    def test_missing_variant_column_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)

    def test_unknown_entry_raises(self, df_seq_pos, df_feat):
        bad = pd.DataFrame({ut.COL_ENTRY: ["NOPE"], ut.COL_VARIANT: ["v"],
                            ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        with pytest.raises(ValueError):
            aa.SeqMut().combine(df_seq=df_seq_pos, variants=bad, df_feat=df_feat)


class TestSeqMutCombineGoldenValues:
    def test_single_variant_matches_mutate(self, df_seq_pos, df_feat, model_tuple):
        # A 1-mutation "variant" must score identically to the same point mutation via mutate.
        sm = aa.SeqMut(model=model_tuple)
        single = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_VARIANT: ["v"],
                               ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df_c = sm.combine(df_seq=df_seq_pos, variants=single, df_feat=df_feat)
        muts = pd.DataFrame({ut.COL_ENTRY: ["P1"], ut.COL_POS: [11], ut.COL_TO_AA: ["A"]})
        df_m = sm.mutate(df_seq=df_seq_pos, mutations=muts, df_feat=df_feat)
        assert df_c[ut.COL_DELTA_PRED].iloc[0] == pytest.approx(df_m[ut.COL_DELTA_PRED].iloc[0])
        assert df_c[ut.COL_DELTA_CPP].iloc[0] == pytest.approx(df_m[ut.COL_DELTA_CPP].iloc[0])
