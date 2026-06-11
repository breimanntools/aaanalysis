"""Tests for SeqMut.__init__ and SeqMut.scan (exhaustive ΔCPP landscape)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut


class TestSeqMutInit:
    def test_default(self):
        sm = aa.SeqMut()
        assert sm.df_scales.shape[0] == 20

    @settings(max_examples=4, deadline=None)
    @given(verbose=some.booleans())
    def test_verbose(self, verbose):
        assert aa.SeqMut(verbose=verbose)._verbose == verbose

    def test_custom_df_scales(self):
        df = ut.load_default_scales().iloc[:, :5]
        assert list(aa.SeqMut(df_scales=df).df_scales.columns) == list(df.columns)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aa.SeqMut(verbose="no")


class TestSeqMutScan:
    """Normal cases for SeqMut.scan."""

    def test_columns(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert list(df.columns) == ut.COLS_SEQMUT_SCAN

    def test_region_tmd_count(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        # TMD is 10 residues per protein, 19 substitutions, 2 proteins
        assert len(df) == 10 * 19 * 2

    def test_region_none_covers_more(self, df_seq_pos, df_feat):
        n_tmd = len(aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd"))
        n_all = len(aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region=None))
        assert n_all > n_tmd

    def test_region_jmd_n(self, df_seq_pos, df_feat_multipart):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat_multipart, region="jmd_n")
        assert set(df[ut.COL_REGION]) == {ut.COL_JMD_N}

    def test_region_positions_list(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region=[11, 12])
        assert set(df[ut.COL_POS]).issubset({11, 12})

    def test_to_aa_subset(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                              to_aa=["A", "G"])
        assert set(df[ut.COL_TO_AA]).issubset({"A", "G"})

    def test_sorted_by_delta_cpp(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert np.all(np.diff(df[ut.COL_DELTA_CPP].to_numpy()) <= 1e-9)

    def test_delta_cpp_nonnegative(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert (df[ut.COL_DELTA_CPP] >= 0).all()

    def test_no_self_mutation(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert (df[ut.COL_FROM_AA] != df[ut.COL_TO_AA]).all()

    def test_mutation_label_format(self, df_seq_pos, df_feat):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        row = df.iloc[0]
        assert row[ut.COL_MUTATION] == f"{row[ut.COL_FROM_AA]}{row[ut.COL_POS]}{row[ut.COL_TO_AA]}"

    @pytest.mark.parametrize("jmd_n_len", [5, 8, 10])
    def test_jmd_n_len(self, df_seq_pos, df_feat, jmd_n_len):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                              jmd_n_len=jmd_n_len, jmd_c_len=10)
        assert len(df) > 0

    @pytest.mark.parametrize("jmd_c_len", [5, 8, 10])
    def test_jmd_c_len(self, df_seq_pos, df_feat, jmd_c_len):
        df = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd",
                              jmd_n_len=10, jmd_c_len=jmd_c_len)
        assert len(df) > 0

    # Negative cases
    def test_part_based_df_seq_raises(self, df_feat):
        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1"], "jmd_n": ["A" * 10],
                               "tmd": ["L" * 10], "jmd_c": ["K" * 10]})
        with pytest.raises(ValueError):
            aa.SeqMut().scan(df_seq=df_seq, df_feat=df_feat, region="tmd")

    def test_df_feat_none_raises(self, df_seq_pos):
        with pytest.raises(ValueError):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=None)

    def test_invalid_region_name_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="nonsense")

    def test_invalid_to_aa_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, to_aa=["Z"])

    def test_empty_region_raises(self, df_seq_pos, df_feat):
        with pytest.raises(ValueError):
            aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region=[10000])


class TestSeqMutScanComplex:
    def test_region_list_equivalent_to_tmd(self, df_seq_pos, df_feat):
        tmd_positions = list(range(11, 21))
        df_list = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region=tmd_positions)
        df_tmd = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        assert len(df_list) == len(df_tmd)

    def test_to_aa_count_scaling(self, df_seq_pos, df_feat):
        df1 = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd", to_aa=["A"])
        df2 = aa.SeqMut().scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd", to_aa=["A", "G"])
        assert len(df2) >= len(df1)


class TestSeqMutScanGoldenValues:
    """Hand-computed ΔCPP against the SequenceFeature builder."""

    def test_delta_cpp_matches_manual(self, df_seq_pos, df_feat):
        sm = aa.SeqMut()
        df = sm.scan(df_seq=df_seq_pos, df_feat=df_feat, region="tmd")
        # Reconstruct one mutation manually via SequenceFeature.
        from aaanalysis.feature_engineering._sequence_feature import SequenceFeature
        sf = SequenceFeature(verbose=False)
        ds = ut.load_default_scales()
        feats = list(df_feat[ut.COL_FEATURE])
        row = df.iloc[0]
        entry = row[ut.COL_ENTRY]
        seq = df_seq_pos.set_index(ut.COL_ENTRY).loc[entry, ut.COL_SEQ]
        pos, to_aa = int(row[ut.COL_POS]), row[ut.COL_TO_AA]
        mut_seq = seq[:pos - 1] + to_aa + seq[pos:]
        ts = int(df_seq_pos.set_index(ut.COL_ENTRY).loc[entry, ut.COL_TMD_START])
        te = int(df_seq_pos.set_index(ut.COL_ENTRY).loc[entry, ut.COL_TMD_STOP])
        df_wt = pd.DataFrame({ut.COL_ENTRY: ["x"], ut.COL_SEQ: [seq],
                              ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
        df_mt = pd.DataFrame({ut.COL_ENTRY: ["x"], ut.COL_SEQ: [mut_seq],
                              ut.COL_TMD_START: [ts], ut.COL_TMD_STOP: [te]})
        parts = ["tmd"]
        X_wt = sf.feature_matrix(features=feats, df_parts=sf.get_df_parts(df_seq=df_wt, list_parts=parts), df_scales=ds)
        X_mt = sf.feature_matrix(features=feats, df_parts=sf.get_df_parts(df_seq=df_mt, list_parts=parts), df_scales=ds)
        expected = np.abs(X_mt - X_wt).sum()
        assert row[ut.COL_DELTA_CPP] == pytest.approx(expected, abs=1e-6)
