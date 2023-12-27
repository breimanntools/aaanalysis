"""This is a script to test the SequenceFeature().get_df_parts() method ."""
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import pytest
import random

# Helper function to create a mock DataFrame


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
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_part_format], jmd_n_len=None, jmd_c_len=None), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df_seq[cols_seq_tmd_format]), pd.DataFrame)

    @settings(max_examples=10, deadline=1000)
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

    @settings(max_examples=10, deadline=1000)
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


# Complex Cases
class TestGetDfPartsComplex:
    """Test get_df_parts function of the SequenceFeature class for Complex Cases."""

    @settings(max_examples=10, deadline=1000)
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
