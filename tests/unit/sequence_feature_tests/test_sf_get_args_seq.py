"""This is a script to test the SequenceFeature().get_args_seq() method."""
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import pytest

# Set default deadline
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False

EXPECTED_KEYS = ["jmd_n_seq", "tmd_seq", "jmd_c_seq"]

# Fixtures
df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
ENTRIES = df_seq["entry"].to_list()


class TestGetArgsSeq:
    """Positive and negative cases, one parameter per test."""

    # Positive tests
    def test_df_seq_returns_seq_keys(self):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=ENTRIES[0])
        assert isinstance(args_seq, dict)
        assert list(args_seq.keys()) == EXPECTED_KEYS
        assert all(isinstance(v, str) for v in args_seq.values())

    @settings(max_examples=5, deadline=None)
    @given(i=some.integers(min_value=0, max_value=9))
    def test_sample_entry_name(self, i):
        sf = aa.SequenceFeature()
        entry = ENTRIES[i]
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=entry)
        assert list(args_seq.keys()) == EXPECTED_KEYS

    @settings(max_examples=5, deadline=None)
    @given(pos=some.integers(min_value=0, max_value=9))
    def test_sample_position(self, pos):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=pos)
        assert list(args_seq.keys()) == EXPECTED_KEYS

    def test_entry_and_position_agree(self):
        # Selecting by entry == selecting by that entry's row position
        sf = aa.SequenceFeature()
        entry = ENTRIES[2]
        pos = list(df_seq["entry"]).index(entry)
        assert sf.get_args_seq(df_seq=df_seq, sample=entry) == sf.get_args_seq(df_seq=df_seq, sample=pos)

    @settings(max_examples=5, deadline=None)
    @given(jmd_len=some.integers(min_value=1, max_value=10))
    def test_jmd_n_len(self, jmd_len):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=ENTRIES[0], jmd_n_len=jmd_len)
        assert len(args_seq["jmd_n_seq"]) == jmd_len

    @settings(max_examples=5, deadline=None)
    @given(jmd_len=some.integers(min_value=1, max_value=10))
    def test_jmd_c_len(self, jmd_len):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=ENTRIES[0], jmd_c_len=jmd_len)
        assert len(args_seq["jmd_c_seq"]) == jmd_len

    def test_numpy_int_position(self):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=np.int64(1))
        assert list(args_seq.keys()) == EXPECTED_KEYS

    # Golden value: parts match get_df_parts slicing for the same protein
    def test_matches_get_df_parts(self):
        sf = aa.SequenceFeature()
        entry = ENTRIES[0]
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=entry)
        assert args_seq["jmd_n_seq"] == df_parts.loc[entry, "jmd_n"]
        assert args_seq["tmd_seq"] == df_parts.loc[entry, "tmd"]
        assert args_seq["jmd_c_seq"] == df_parts.loc[entry, "jmd_c"]

    # Negative tests
    def test_invalid_entry_name(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_args_seq(df_seq=df_seq, sample="NOT_AN_ENTRY")

    def test_position_out_of_range(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_args_seq(df_seq=df_seq, sample=999)

    def test_negative_position(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_args_seq(df_seq=df_seq, sample=-1)

    def test_invalid_sample_type(self):
        sf = aa.SequenceFeature()
        for bad in [1.5, None, ["P05067"], {"a": 1}]:
            with pytest.raises(ValueError):
                sf.get_args_seq(df_seq=df_seq, sample=bad)

    def test_invalid_df_seq(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_args_seq(df_seq="not_a_df", sample=ENTRIES[0])


class TestGetArgsSeqComplex:
    """Combinations and edge interactions."""

    def test_splat_into_plot_kwargs(self):
        # The returned dict must splat as keyword arguments without collision
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=ENTRIES[0])

        def _consumer(jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None):
            return (jmd_n_seq, tmd_seq, jmd_c_seq)

        out = _consumer(**args_seq)
        assert all(isinstance(x, str) for x in out)

    def test_jmd_lengths_combined(self):
        sf = aa.SequenceFeature()
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=ENTRIES[1], jmd_n_len=7, jmd_c_len=3)
        assert len(args_seq["jmd_n_seq"]) == 7
        assert len(args_seq["jmd_c_seq"]) == 3

    def test_part_based_df_seq(self):
        # Part-based df_seq (jmd_n/tmd/jmd_c columns) also resolves
        sf = aa.SequenceFeature()
        df_parts_seq = pd.DataFrame({
            "entry": ["A", "B"],
            "jmd_n": ["AAAAAAAAAA", "CCCCCCCCCC"],
            "tmd": ["LIVMFWLIVM", "GGGGGGGGGG"],
            "jmd_c": ["KKKKKKKKKK", "DDDDDDDDDD"],
        })
        args_seq = sf.get_args_seq(df_seq=df_parts_seq, sample="A")
        assert args_seq["tmd_seq"] == "LIVMFWLIVM"
        assert args_seq["jmd_n_seq"] == "AAAAAAAAAA"

    def test_bool_sample_rejected(self):
        # bool must not be treated as an int position
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_args_seq(df_seq=df_seq, sample=True)
