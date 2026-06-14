"""This is a script to test the SequenceFeature().get_seq_kws() method."""
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


def _df_parts(list_parts=None):
    sf = aa.SequenceFeature(verbose=False)
    return sf.get_df_parts(df_seq=df_seq, list_parts=list_parts) if list_parts else sf.get_df_parts(df_seq=df_seq)


def _manual_parts(pos):
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[pos, ["jmd_n", "tmd", "jmd_c"]]
    return dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)


class TestGetSeqKws:
    """Normal cases, one parameter per test."""

    def test_returns_seq_keys(self):
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=ENTRIES[0])
        assert list(seq_kws) == EXPECTED_KEYS
        assert all(isinstance(v, str) for v in seq_kws.values())

    @settings(max_examples=5, deadline=None)
    @given(i=some.integers(min_value=0, max_value=len(ENTRIES) - 1))
    def test_sample_entry_name(self, i):
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=ENTRIES[i])
        assert seq_kws == _manual_parts(i)

    @settings(max_examples=5, deadline=None)
    @given(pos=some.integers(min_value=0, max_value=len(ENTRIES) - 1))
    def test_sample_position(self, pos):
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=pos)
        assert seq_kws == _manual_parts(pos)

    def test_entry_and_position_agree(self):
        sf = aa.SequenceFeature(verbose=False)
        dp = _df_parts()
        for pos, entry in enumerate(ENTRIES):
            assert sf.get_seq_kws(df_seq=df_seq, df_parts=dp, sample=entry) == \
                   sf.get_seq_kws(df_seq=df_seq, df_parts=dp, sample=pos)

    def test_numpy_int_position(self):
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=np.int64(1))
        assert seq_kws == _manual_parts(1)

    def test_extended_df_parts(self):
        """Default CPP df_parts (tmd/jmd_n_tmd_n/tmd_c_jmd_c) yields the basic parts."""
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=0)
        assert seq_kws == _manual_parts(0)

    def test_basic_df_parts(self):
        """df_parts already holding the basic parts is used directly."""
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(["jmd_n", "tmd", "jmd_c"]), sample=0)
        assert seq_kws == _manual_parts(0)

    def test_tmd_only_df_parts_empty_jmd(self):
        """A df_parts without JMD parts returns empty JMD strings (no error)."""
        sf = aa.SequenceFeature(verbose=False)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(["tmd"]), sample=0)
        assert seq_kws["jmd_n_seq"] == "" and seq_kws["jmd_c_seq"] == ""
        assert seq_kws["tmd_seq"] == _manual_parts(0)["tmd_seq"]

    def test_invalid_entry_name(self):
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError, match="sample"):
            sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample="NOT_AN_ENTRY")

    def test_position_out_of_range(self):
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError):
            sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=999)

    def test_negative_position(self):
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError):
            sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=-1)

    def test_invalid_sample_type(self):
        sf = aa.SequenceFeature(verbose=False)
        for bad in [3.5, None, True, [0]]:
            with pytest.raises(ValueError):
                sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=bad)

    def test_invalid_df_seq(self):
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError):
            sf.get_seq_kws(df_seq="not_a_df", df_parts=_df_parts(), sample=ENTRIES[0])

    def test_invalid_df_parts(self):
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError):
            sf.get_seq_kws(df_seq=df_seq, df_parts="not_a_df", sample=ENTRIES[0])


class TestGetSeqKwsComplex:
    """Combinations and edge interactions, incl. the df_seq<->df_parts binding."""

    def test_splat_into_plot_methods(self):
        sf = aa.SequenceFeature(verbose=False)
        df_feat = aa.load_features().head(10)
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=ENTRIES[0])
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, **seq_kws)
        assert ax is not None

    def test_matches_get_df_parts_basic(self):
        sf = aa.SequenceFeature(verbose=False)
        dp_basic = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        entry = ENTRIES[2]
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=_df_parts(), sample=entry)
        row = dp_basic.loc[entry]
        assert seq_kws == {"jmd_n_seq": row["jmd_n"], "tmd_seq": row["tmd"], "jmd_c_seq": row["jmd_c"]}

    def test_seq_based_df_seq(self):
        """A sequence-based df_seq (no stored parts) still cross-checks via the full sequence."""
        sf = aa.SequenceFeature(verbose=False)
        df_seq_seq = df_seq[["entry", "sequence"]].copy()
        seq_kws = sf.get_seq_kws(df_seq=df_seq_seq, df_parts=_df_parts(), sample=0)
        assert seq_kws == _manual_parts(0)

    def test_mismatch_df_seq_df_parts_raises(self):
        """df_seq from different proteins than df_parts -> ValueError."""
        sf = aa.SequenceFeature(verbose=False)
        df_seq_other = df_seq.copy()
        df_seq_other["entry"] = ["X" + str(i) for i in range(len(df_seq_other))]
        with pytest.raises(ValueError, match="do not match|not in the"):
            sf.get_seq_kws(df_seq=df_seq_other, df_parts=_df_parts(), sample=0)

    def test_part_mismatch_raises(self):
        """Part-only df_seq whose stored parts disagree with df_parts -> ValueError."""
        sf = aa.SequenceFeature(verbose=False)
        # Part-only df_seq (no 'sequence' column) so the check compares stored parts, not the full sequence
        df_seq_parts = df_seq[["entry", "jmd_n", "tmd", "jmd_c"]].copy()
        df_seq_parts.loc[0, "tmd"] = "AAAAAAAAAA"
        with pytest.raises(ValueError, match="do not match"):
            sf.get_seq_kws(df_seq=df_seq_parts, df_parts=_df_parts(["jmd_n", "tmd", "jmd_c"]), sample=0)
