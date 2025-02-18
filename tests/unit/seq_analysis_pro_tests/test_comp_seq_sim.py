"""
This is a script for testing the aa.comp_seq_sim function.
"""
from hypothesis import given, settings, strategies as st
import pytest
import pandas as pd
import aaanalysis as aa

# Register hypothesis profile for continuous integration
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TestCompSeqSim:
    """Test class for the 'aa.comp_seq_sim' function."""

    @settings(max_examples=10, deadline=1000)
    @given(seq1=st.text(min_size=10, max_size=100, alphabet=ALPHABET), seq2=st.text(min_size=10, max_size=100, alphabet=ALPHABET))
    def test_valid_seqs(self, seq1, seq2):
        """Test valid sequence inputs."""
        result = aa.comp_seq_sim(seq1=seq1, seq2=seq2)
        assert isinstance(result, float)
        
    @settings(max_examples=10, deadline=1000)
    @given(seq1=st.text(min_size=10, max_size=100, alphabet=ALPHABET))
    def test_invalid_none_seqs(self, seq1):
        """Test invalid None sequences."""
        for seq2 in [None, 324, []]:
            with pytest.raises(ValueError):
                aa.comp_seq_sim(seq1=seq1, seq2=seq2)
        with pytest.raises(ValueError):
            aa.comp_seq_sim(seq1=None, seq2=None)

    def test_valid_df_seq(self):
        """Test valid DataFrame input."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
        result = aa.comp_seq_sim(df_seq=df_seq)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_df_seq(self):
        """Test invalid None DataFrame."""
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=None)
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=[])
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=dict())
        df_seq = aa.load_dataset(name="SEQ_LOCATION", n=25)
        df_seq["sequence"] = 1
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=df_seq)
        df_seq = df_seq.drop("sequence", axis=1)
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=df_seq)


class TestCompSeqSimComplex:
    """Complex tests for the 'aa.comp_seq_sim' function."""

    def test_valid_seqs_and_df_seq(self):
        """Test valid sequences with alignment mode."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
        seq1 = "ACGT"
        seq2 = "ACGT"
        result = aa.comp_seq_sim(df_seq=df_seq, seq1=seq1, seq2=seq2)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_seqs_and_df_seq(self):
        """Test valid sequences with alignment mode."""
        seq2 = "ACGT"
        with pytest.raises(ValueError):
            aa.comp_seq_sim(df_seq=None, seq1=None, seq2=seq2)


