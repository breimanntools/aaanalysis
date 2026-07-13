"""This is a script to test the SequencePreprocessor().pad_parts() method."""
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
import aaanalysis as aa

# Set default deadline
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

AA = "ACDEFGHIKLMNPQRSTVWY"


def _df():
    """Small two-part df_parts fixture (tmd per-part max 5, jmd per-part max 3)."""
    return pd.DataFrame({"tmd": ["AC", "ACDEF"], "jmd": ["A", "ABC"]})


# Normal Cases: df_parts input
class TestPadPartsInput:
    """Test the 'df_parts' input (type + string-column validation)."""

    @settings(max_examples=15, deadline=None)
    @given(seqs=st.lists(st.text(alphabet=AA, min_size=1, max_size=30), min_size=1, max_size=20))
    def test_df_parts_valid(self, seqs):
        """A DataFrame of sequence-part columns is padded to a uniform per-column length."""
        seqp = aa.SequencePreprocessor()
        df = pd.DataFrame({"tmd": seqs})
        out = seqp.pad_parts(df_parts=df)
        assert isinstance(out, pd.DataFrame)
        max_len = max(len(s) for s in seqs)
        assert all(len(s) == max_len for s in out["tmd"])

    def test_df_parts_invalid_type(self):
        """A non-DataFrame 'df_parts' raises a ValueError (list/str no longer accepted)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=["AC", "ACDEF"])
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts="ACDEF")
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=None)
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=dict())

    def test_df_parts_non_string_column(self):
        """A selected column with non-string values raises a clear ValueError."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=pd.DataFrame({"tmd": [1, 2, 3]}))

    def test_df_parts_non_string_only_checked_when_selected(self):
        """A non-string column that is NOT selected does not trigger the string check."""
        seqp = aa.SequencePreprocessor()
        df = pd.DataFrame({"tmd": ["AC", "ACDEF"], "meta": [1, 2]})
        out = seqp.pad_parts(df_parts=df, list_parts=["tmd"])
        assert out["tmd"].tolist() == ["AC---", "ACDEF"]
        assert out["meta"].tolist() == [1, 2]


class TestPadPartsLength:
    """Test the 'length' parameter."""

    @settings(max_examples=15, deadline=None)
    @given(length=st.integers(min_value=1, max_value=60))
    def test_length_valid(self, length):
        """A valid 'length' (>= longest value) pads every selected column to that length."""
        seqp = aa.SequencePreprocessor()
        df = _df()
        length = max(length, 5)  # >= longest value across columns
        out = seqp.pad_parts(df_parts=df, length=length)
        assert all(len(s) == length for s in out["tmd"])
        assert all(len(s) == length for s in out["jmd"])

    def test_length_none_per_part_max(self):
        """length=None pads each column to its OWN max string length (per-part)."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=None)
        assert out["tmd"].tolist() == ["AC---", "ACDEF"]   # per-part max 5
        assert out["jmd"].tolist() == ["A--", "ABC"]        # per-part max 3

    def test_length_explicit(self):
        """An explicit 'length' overrides the per-part max for all selected columns."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6)
        assert out["tmd"].tolist() == ["AC----", "ACDEF-"]
        assert out["jmd"].tolist() == ["A-----", "ABC---"]

    def test_length_too_short_raises(self):
        """A selected value longer than 'length' raises a ValueError."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=_df(), length=3)   # 'ACDEF' (5) > 3

    def test_length_invalid(self):
        """Invalid 'length' values raise a ValueError."""
        seqp = aa.SequencePreprocessor()
        for bad in [0, -3, 2.5, "5"]:
            with pytest.raises(ValueError):
                seqp.pad_parts(df_parts=_df(), length=bad)


class TestPadPartsGap:
    """Test the 'gap' parameter."""

    @settings(max_examples=15, deadline=None)
    @given(gap=st.text(min_size=1, max_size=1))
    def test_gap_valid(self, gap):
        """A single-character 'gap' is used for padding."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), gap=gap)
        assert out["tmd"].tolist()[0] == "AC" + gap * 3

    def test_gap_invalid(self):
        """Invalid 'gap' values raise a ValueError."""
        seqp = aa.SequencePreprocessor()
        for bad in [None, "", "--"]:
            with pytest.raises(ValueError):
                seqp.pad_parts(df_parts=_df(), gap=bad)


class TestPadPartsPadAt:
    """Test the 'pad_at' parameter."""

    @settings(max_examples=10, deadline=None)
    @given(pad_at=st.sampled_from(["N", "C", "both"]))
    def test_pad_at_valid(self, pad_at):
        """'N', 'C', and 'both' are accepted 'pad_at' options."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), pad_at=pad_at)
        assert all(len(s) == 5 for s in out["tmd"])

    def test_pad_at_invalid(self):
        """Invalid 'pad_at' values raise a ValueError."""
        seqp = aa.SequencePreprocessor()
        for bad in [None, "", "X", "c"]:
            with pytest.raises(ValueError):
                seqp.pad_parts(df_parts=_df(), pad_at=bad)


class TestPadPartsListParts:
    """Test the 'list_parts' column selection."""

    def test_list_parts_none_pads_all(self):
        """list_parts=None pads every column."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), list_parts=None)
        assert out["tmd"].tolist() == ["AC---", "ACDEF"]
        assert out["jmd"].tolist() == ["A--", "ABC"]

    def test_list_parts_single_str(self):
        """A single column name selects only that column; the rest are untouched."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), list_parts="tmd")
        assert out["tmd"].tolist() == ["AC---", "ACDEF"]
        assert out["jmd"].tolist() == ["A", "ABC"]

    def test_list_parts_list(self):
        """A list of column names selects those columns."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6, list_parts=["tmd", "jmd"])
        assert out["tmd"].tolist() == ["AC----", "ACDEF-"]
        assert out["jmd"].tolist() == ["A-----", "ABC---"]

    def test_list_parts_leaves_others_exact(self):
        """Non-selected columns are byte-identical to the input."""
        seqp = aa.SequencePreprocessor()
        df = _df()
        out = seqp.pad_parts(df_parts=df, list_parts=["tmd"])
        assert out["jmd"].tolist() == df["jmd"].tolist()

    def test_list_parts_invalid_column(self):
        """A 'list_parts' entry not among the columns raises a ValueError."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=_df(), list_parts=["nope"])
        with pytest.raises(ValueError):
            seqp.pad_parts(df_parts=_df(), list_parts=["tmd", "nope"])


class TestPadPartsInvariants:
    """Copy semantics, non-mutation, and index preservation."""

    def test_returns_copy_not_same_object(self):
        seqp = aa.SequencePreprocessor()
        df = _df()
        out = seqp.pad_parts(df_parts=df)
        assert out is not df

    def test_input_not_mutated(self):
        seqp = aa.SequencePreprocessor()
        df = _df()
        seqp.pad_parts(df_parts=df, length=8)
        assert df["tmd"].tolist() == ["AC", "ACDEF"]
        assert df["jmd"].tolist() == ["A", "ABC"]

    def test_index_preserved(self):
        seqp = aa.SequencePreprocessor()
        df = pd.DataFrame({"tmd": ["AC", "ACDEF"]}, index=[7, 11])
        out = seqp.pad_parts(df_parts=df)
        assert out.index.tolist() == [7, 11]

    def test_columns_and_order_preserved(self):
        seqp = aa.SequencePreprocessor()
        df = _df()
        out = seqp.pad_parts(df_parts=df, list_parts=["tmd"])
        assert list(out.columns) == list(df.columns)

    def test_original_residues_preserved(self):
        """Padding never alters the original residues; it only adds gaps."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=8, pad_at="C")
        assert [s.rstrip("-") for s in out["tmd"]] == ["AC", "ACDEF"]


# pad_at="both" symmetric / centered padding
class TestPadPartsBoth:
    """Test symmetric ('both') padding: floor(k/2) N-terminal, remainder C-terminal."""

    def test_ref_both_even(self):
        """Golden: an even gap count splits evenly."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=pd.DataFrame({"tmd": ["AC"]}), length=6, pad_at="both")
        assert out["tmd"].tolist() == ["--AC--"]

    def test_ref_both_odd(self):
        """Golden: an odd gap count puts floor(k/2) at N, remainder at C."""
        seqp = aa.SequencePreprocessor()
        # k = 7 - 2 = 5 -> 2 at N, 3 at C
        out = seqp.pad_parts(df_parts=pd.DataFrame({"tmd": ["AC"]}), length=7, pad_at="both")
        assert out["tmd"].tolist() == ["--AC---"]

    def test_ref_both_mixed_lengths(self):
        """Golden: per-value gap counts; 'ACDEF' (k=1) -> 0 N + 1 C."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6, pad_at="both")
        assert out["tmd"].tolist() == ["--AC--", "ACDEF-"]

    def test_both_zero_gap_noop(self):
        """No gaps needed -> value unchanged under 'both'."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=pd.DataFrame({"tmd": ["ACDEF"]}), length=5, pad_at="both")
        assert out["tmd"].tolist() == ["ACDEF"]


# Golden / byte-identical reference values (assert exact equality, no tolerance)
class TestPadPartsReference:
    """Byte-identical reference cases pinning the exact output."""

    def test_ref_c_terminal(self):
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6, pad_at="C")
        assert out["tmd"].tolist() == ["AC----", "ACDEF-"]

    def test_ref_n_terminal(self):
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6, pad_at="N")
        assert out["tmd"].tolist() == ["----AC", "-ACDEF"]

    def test_ref_length_none_per_part(self):
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df())
        assert out["tmd"].tolist() == ["AC---", "ACDEF"]
        assert out["jmd"].tolist() == ["A--", "ABC"]

    def test_ref_custom_gap(self):
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6, gap="*", pad_at="C")
        assert out["tmd"].tolist() == ["AC****", "ACDEF*"]

    def test_ref_default_gap_and_pad_at(self):
        """Defaults are gap='-' and pad_at='C'."""
        seqp = aa.SequencePreprocessor()
        out = seqp.pad_parts(df_parts=_df(), length=6)
        assert out["tmd"].tolist() == ["AC----", "ACDEF-"]


# Backend byte-exactness guard: the length=None default is a no-op for encode_*
class TestPadSequencesByteExactness:
    """The additive 'length' param must not change pad_sequences' default behavior."""

    @settings(max_examples=25, deadline=None)
    @given(
        list_seq=st.lists(st.text(alphabet=AA, min_size=1, max_size=25), min_size=1, max_size=15),
        pad_at=st.sampled_from(["N", "C"]),
        gap=st.text(min_size=1, max_size=1),
    )
    def test_default_matches_legacy(self, list_seq, pad_at, gap):
        """pad_sequences with length omitted == length=None == pad-to-max (unchanged N/C)."""
        from aaanalysis.data_handling._backend.seq_preproc._utils import pad_sequences
        legacy = pad_sequences(list_seq, pad_at=pad_at, gap=gap)
        with_none = pad_sequences(list_seq, pad_at=pad_at, gap=gap, length=None)
        max_len = max(len(s) for s in list_seq)
        assert legacy == with_none
        assert all(len(s) == max_len for s in legacy)
