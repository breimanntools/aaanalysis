"""This is a script to test the SequenceFeature().get_df_parts() method ."""
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import pytest
import random

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Normal Cases
class TestGetDfParts:
    """Test get_df_parts function of the SequenceFeature class."""

    def test_default_format(self):
        """Test all valid formats"""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC")
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        df_parts = sf.get_df_parts(df_seq=df_seq, jmd_c_len=10, jmd_n_len=10, list_parts=list_parts).reset_index(drop=True)
        default_true = (df_seq[list_parts] == df_parts[list_parts]).all().all()
        assert default_true

    def test_pos_based_format(self):
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC")
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        cols_position_based = ["entry", "sequence", "tmd_start", "tmd_stop"]
        df_parts = sf.get_df_parts(df_seq=df_seq[cols_position_based], list_parts=list_parts).reset_index(drop=True)
        pos_based_true = (df_seq[list_parts] == df_parts[list_parts]).all().all()
        assert pos_based_true

    def test_part_based_format(self):
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC")
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        cols_part_based = ["entry", "jmd_n", "tmd", "jmd_c"]
        df_parts = sf.get_df_parts(df_seq=df_seq[cols_part_based], list_parts=list_parts).reset_index(drop=True)
        part_based_true = (df_seq[list_parts] == df_parts[list_parts]).all().all()
        assert part_based_true

    def test_seq_tmd_based_format(self):
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC")
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        cols_sequence_tmd_based = ["entry", "sequence", "tmd"]
        df_parts = sf.get_df_parts(df_seq=df_seq[cols_sequence_tmd_based], list_parts=list_parts).reset_index(drop=True)
        seq_tmd_based_true = (df_seq[list_parts] == df_parts[list_parts]).all().all()
        assert seq_tmd_based_true

    def test_seq_based_format(self):
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC")
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        cols_sequence_based = ["entry", "sequence"]
        df_parts = sf.get_df_parts(df_seq=df_seq[cols_sequence_based], list_parts=list_parts).reset_index(drop=True)
        seq = df_parts["jmd_n"] + df_parts["tmd"] + df_parts["jmd_c"]
        seq_based_true = (df_seq["sequence"] == seq).all()
        assert seq_based_true

    def test_valid_df_seq(self):
        """Test a valid 'df_seq' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = df_info["Dataset"].to_list()
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            df_parts = sf.get_df_parts(df_seq=df_seq)
            assert isinstance(df_parts, pd.DataFrame)
        # Test different formats
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        cols_pos_format = ["entry", "sequence", "tmd_start", "tmd_stop"]
        cols_part_format = ["entry", "jmd_n", "tmd", "jmd_c"]
        cols_seq_format = ["entry", "sequence"]
        cols_seq_tmd_format = ["entry", "sequence", "tmd"]
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_seq_format]), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_pos_format]), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_part_format]), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_part_format], jmd_n_len=0, jmd_c_len=0), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_seq_tmd_format]), pd.DataFrame)

    @settings(max_examples=10, deadline=None)
    @given(list_parts=some.lists(some.sampled_from(['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
                                                    'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']),
                                 min_size=1))
    def test_valid_list_parts(self, list_parts):
        """Test a valid 'list_parts' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = random.sample(df_info["Dataset"].to_list(), 2)
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, list_parts=list_parts), pd.DataFrame)

    def test_valid_all_parts(self):
        """Test a valid 'all_parts' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = df_info["Dataset"].to_list()
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, all_parts=True, jmd_n_len=10, jmd_c_len=10), pd.DataFrame)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, all_parts=True, jmd_n_len=2, jmd_c_len=2), pd.DataFrame)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, all_parts=True, jmd_n_len=2, jmd_c_len=40), pd.DataFrame)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, all_parts=True, jmd_n_len=0, jmd_c_len=10), pd.DataFrame)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, all_parts=True, jmd_n_len=5, jmd_c_len=0), pd.DataFrame)

    def test_valid_remove_entries_with_gaps(self):
        """Test a valid 'all_parts' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = df_info["Dataset"].to_list()
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50, aa_window_size=21)
            assert isinstance(sf.get_df_parts(df_seq=df_seq, remove_entries_with_gaps=True, jmd_n_len=2, jmd_c_len=2), pd.DataFrame)
            df_seq["sequence"] += "A"
            assert isinstance(sf.get_df_parts(df_seq=df_seq, remove_entries_with_gaps=True, jmd_n_len=5, jmd_c_len=0), pd.DataFrame)

    @settings(max_examples=10, deadline=None)
    @given(jmd_n_len=some.integers(min_value=1), jmd_c_len=some.integers(min_value=1))
    def test_valid_jmd_len(self, jmd_n_len, jmd_c_len):
        """Test a valid 'jmd_n_len' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = random.sample(df_info["Dataset"].to_list(), 2)
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            min_n = min(df_seq["sequence"].apply(len))
            if min_n > jmd_n_len + jmd_c_len:
                assert isinstance(sf.get_df_parts(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len), pd.DataFrame)

    def test_valid_replace_non_canonical_aa(self):
        """Test a valid 'replace_non_canonical_aa' parameter."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = random.sample(df_info["Dataset"].to_list(), 2)
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            df_seq = df_seq[[x for x in list(df_seq) if x not in list_parts]]
            df_seq_modified = df_seq.copy()
            df_seq_modified["sequence"] = df_seq_modified["sequence"].str.replace("A", "X", regex=True)
            df_seq_modified["sequence"] = df_seq_modified["sequence"].str.replace("G", "Z", regex=True)
            df_parts = sf.get_df_parts(df_seq=df_seq_modified, replace_non_canonical_aa=True, all_parts=True)
            for col in list_parts:
                if col in df_parts.columns:
                    assert not df_parts[col].str.contains("[^ACDEFGHIKLMNPQRSTVWY-]").any()

    # Negative tests for each parameter (not exhaustive)
    def test_invalid_df_seq(self):
        """Test an invalid 'df_seq' parameter."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=None)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=[])
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=dict())
        df_seq = aa.load_dataset(name="SEQ_LOCATION", n=50)
        df_seq["sequence"] = 1
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq)
        df_seq = df_seq.drop("sequence", axis=1)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq)
        df_seq = aa.load_dataset(name="SEQ_LOCATION", n=100)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_c_len=None)
        # Invalid format
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        cols_seq_format = ["entry", "sequence"]
        cols_pos_format = ["entry", "sequence", "tmd_start", "tmd_stop"]
        cols_part_format = ["entry", "jmd_n", "tmd", "jmd_c"]
        cols_seq_tmd_format = ["entry", "sequence", "tmd"]
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq[cols_seq_format], jmd_n_len=None, jmd_c_len=None)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq[cols_pos_format], jmd_n_len=None, jmd_c_len=None)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq[cols_part_format], jmd_n_len=10, jmd_c_len=None)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq[cols_seq_tmd_format], jmd_n_len=None, jmd_c_len=None)

    def test_invalid_list_parts(self):
        """Test an invalid 'list_parts' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, list_parts=["TMD"])
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, list_parts=["tmd-e"])
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_e", 1])
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_e", "jmd_c", "jmd_x"])

    def test_invalid_jmd_len(self):
        """Test an invalid 'jmd_n_len' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len=None, jmd_c_len=10)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=None)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len="A")
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len="", jmd_c_len="A")
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=dict())

    def test_invalid_all_parts(self):
        """Test an invalid 'jmd_n_len' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq, all_parts=None)

    def test_invalid_tmd_start_pos_tmd_stop_pos(self):
        """Test if fails when TMD position is >= 0"""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        cols_pos_format = ["entry", "sequence", "tmd_start", "tmd_stop"]
        # TMD start smaller than 1
        df_seq_a = df_seq[cols_pos_format].copy()
        df_seq_a["tmd_start"] = 0
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq_a)
        # TMD start higher than sequence length
        df_seq_b = df_seq[cols_pos_format].copy()
        df_seq_b["tmd_start"] = 100
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq_b)
        # TMD stop higher than sequence length
        df_seq_b1 = df_seq[cols_pos_format].copy()
        df_seq_b1["tmd_stop"] = 1000000
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq_b1)
        # TMD stop higher than start
        df_seq_c = df_seq[cols_pos_format].copy()
        df_seq_c["tmd_start"] = 1000
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq_c)
        # TMD longer than sequence
        df_seq_d = df_seq[cols_pos_format].copy().head(1)
        df_seq_d["tmd_stop"] = 15001
        df_seq_d["tmd_start"] = 5000
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df_seq_d)

    def test_invalid_replace_non_canonical_aa(self):
        """Test an invalid 'replace_non_canonical_aa' parameter."""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        # Test with invalid value type
        for invalid_args in [None, "True", 1, [True]]:
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=df_seq, replace_non_canonical_aa=invalid_args)


# Golden values & TMD coordinate convention (issue #17)
class TestGetDfPartsGoldenValues:
    """Pin the 1-based, start- & stop-inclusive TMD coordinate convention (CONTEXT.md).

    These tests guard against a future edit re-introducing a 0-based / exclusive-stop
    construction: ``tmd_stop`` is the 1-based inclusive last residue, so a length-L TMD
    satisfies ``tmd_stop - tmd_start + 1 == L``.
    """

    def test_get_tmd_positions_golden(self):
        """Hand-computed 1-based inclusive [tmd_start, tmd_stop] from a tmd substring."""
        from aaanalysis.feature_engineering._backend.check_feature import _get_tmd_positions
        import aaanalysis.utils as ut
        # seq = AAAA | LMVF | CCCC ; TMD 'LMVF' occupies 1-based positions 5..8
        row = pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "AAAALMVFCCCC", ut.COL_TMD: "LMVF"})
        start, stop = _get_tmd_positions(row)
        assert (start, stop) == (5, 8)
        assert stop - start + 1 == len("LMVF")

    def test_get_tmd_positions_len1_accepted(self):
        """A length-1 TMD is valid: tmd_start == tmd_stop, not rejected as 'empty'."""
        from aaanalysis.feature_engineering._backend.check_feature import _get_tmd_positions
        import aaanalysis.utils as ut
        row = pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "AAAAMCCCC", ut.COL_TMD: "M"})
        start, stop = _get_tmd_positions(row)
        assert (start, stop) == (5, 5)

    def test_get_tmd_positions_boundaries(self):
        """TMD at the sequence start (tmd_start==1) and end (tmd_stop==len(seq))."""
        from aaanalysis.feature_engineering._backend.check_feature import _get_tmd_positions
        import aaanalysis.utils as ut
        row_start = pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "LMCCCC", ut.COL_TMD: "LM"})
        assert tuple(_get_tmd_positions(row_start)) == (1, 2)
        row_end = pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "AAAACC", ut.COL_TMD: "CC"})
        start, stop = _get_tmd_positions(row_end)
        assert (start, stop) == (5, 6) and stop == len("AAAACC")

    def test_get_tmd_positions_empty_and_missing_rejected(self):
        """Empty TMD and a TMD absent from the sequence both raise (same message)."""
        from aaanalysis.feature_engineering._backend.check_feature import _get_tmd_positions
        import aaanalysis.utils as ut
        with pytest.raises(ValueError, match="not contained"):
            _get_tmd_positions(pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "ABC", ut.COL_TMD: ""}))
        with pytest.raises(ValueError, match="not contained"):
            _get_tmd_positions(pd.Series({ut.COL_ENTRY: "X", ut.COL_SEQ: "ABC", ut.COL_TMD: "ZZ"}))

    def test_roundtrip_all_formats_identical(self):
        """ACCEPTANCE: position / part / seq-TMD / sequence-only formats give identical parts.

        Clean fixture (jmd_n_len == jmd_c_len == 4, no terminal gaps) so all four
        construction paths must agree residue-for-residue.
        """
        sf = aa.SequenceFeature()
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        seq = "AAAALMVFCCCC"  # jmd_n=AAAA, tmd=LMVF, jmd_c=CCCC
        pos = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "tmd_start": [5], "tmd_stop": [8]})
        part = pd.DataFrame({"entry": ["P1"], "jmd_n": ["AAAA"], "tmd": ["LMVF"], "jmd_c": ["CCCC"]})
        seq_tmd = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "tmd": ["LMVF"]})
        seq_only = pd.DataFrame({"entry": ["P1"], "sequence": [seq]})
        kw = dict(list_parts=list_parts, jmd_n_len=4, jmd_c_len=4)
        out_pos = sf.get_df_parts(df_seq=pos, **kw).reset_index(drop=True)
        out_part = sf.get_df_parts(df_seq=part, **kw).reset_index(drop=True)
        out_seq_tmd = sf.get_df_parts(df_seq=seq_tmd, **kw).reset_index(drop=True)
        out_seq_only = sf.get_df_parts(df_seq=seq_only, **kw).reset_index(drop=True)
        expected = pd.Series({"jmd_n": "AAAA", "tmd": "LMVF", "jmd_c": "CCCC"})
        for out in (out_pos, out_part, out_seq_tmd, out_seq_only):
            assert (out[list_parts].iloc[0] == expected).all()

    def test_roundtrip_on_benchmark_no_value_shift(self):
        """Position-, part-, and seq-TMD-derived parts agree on a real dataset (no drift)."""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        out_pos = sf.get_df_parts(df_seq=df_seq[["entry", "sequence", "tmd_start", "tmd_stop"]],
                                  list_parts=list_parts).reset_index(drop=True)
        out_part = sf.get_df_parts(df_seq=df_seq[["entry", "jmd_n", "tmd", "jmd_c"]],
                                   list_parts=list_parts).reset_index(drop=True)
        out_seq_tmd = sf.get_df_parts(df_seq=df_seq[["entry", "sequence", "tmd"]],
                                      list_parts=list_parts).reset_index(drop=True)
        assert (out_pos[list_parts] == out_part[list_parts]).all().all()
        assert (out_pos[list_parts] == out_seq_tmd[list_parts]).all().all()


# Complex Cases
class TestGetDfPartsComplex:
    """Test get_df_parts function of the SequenceFeature class for Complex Cases."""

    @settings(max_examples=10, deadline=None)
    @given(list_parts=some.lists(some.sampled_from(['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
                                                    'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']),
                                 min_size=1),
           all_parts=some.booleans(),
           jmd_n_len=some.integers(min_value=1),
           jmd_c_len=some.integers(min_value=1))
    def test_valid_combination(self, list_parts, all_parts, jmd_n_len, jmd_c_len):
        """Test valid combinations of parameters."""
        sf = aa.SequenceFeature()
        df_info = aa.load_dataset()
        list_name = random.sample(df_info["Dataset"].to_list(), 2)
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            min_n = min(df_seq["sequence"].apply(len))
            if min_n > jmd_n_len + jmd_c_len:
                assert isinstance(sf.get_df_parts(df_seq=df_seq, list_parts=list_parts, all_parts=all_parts, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len), pd.DataFrame)

