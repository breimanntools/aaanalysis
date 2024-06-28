"""This is a script to test the encode_one_hot function."""
import numpy as np
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Normal Cases
class TestEncodeOneHot:
    """Test encode_one_hot function."""

    @settings(max_examples=10, deadline=1000)
    @given(list_seq=st.lists(st.text(alphabet="ACDEFGHIKLMNPQRSTVWY-", min_size=1, max_size=50), min_size=1, max_size=20))
    def test_list_seq_valid(self, list_seq):
        """Test a valid 'list_seq' parameter."""
        sp = aa.SequencePreprocessor()
        result = sp.encode_one_hot(list_seq=list_seq)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    def test_list_seq_invalid(self):
        """Test an invalid 'list_seq' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.encode_one_hot(list_seq=None)
        with pytest.raises(ValueError):
            sp.encode_one_hot(list_seq=[])
        with pytest.raises(ValueError):
            sp.encode_one_hot(list_seq=dict())
        with pytest.raises(ValueError):
            sp.encode_one_hot(list_seq=["INVALIDSEQUENCE"])

    @settings(max_examples=10, deadline=1000)
    @given(alphabet=st.text(min_size=1, alphabet="ACDEFGHIKLMNPQRSTVWY"))
    def test_alphabet_valid(self, alphabet):
        """Test a valid 'alphabet' parameter."""
        sp = aa.SequencePreprocessor()
        valid_seq = "".join(np.random.choice(list(alphabet), size=10))
        result = sp.encode_one_hot(list_seq=[valid_seq], alphabet=alphabet)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    def test_alphabet_invalid(self):
        """Test an invalid 'alphabet' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.encode_one_hot(alphabet=None)
        with pytest.raises(ValueError):
            sp.encode_one_hot(alphabet="")
        with pytest.raises(ValueError):
            sp.encode_one_hot(alphabet=123)

    @settings(max_examples=10, deadline=1000)
    @given(gap=st.text(min_size=1, max_size=1).filter(lambda g: g not in "ACDEFGHIKLMNPQRSTVWY"))
    def test_gap_valid(self, gap):
        """Test a valid 'gap' parameter."""
        sp = aa.SequencePreprocessor()
        result = sp.encode_one_hot(list_seq=["ARND"], gap=gap)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    def test_gap_invalid(self):
        """Test an invalid 'gap' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.encode_one_hot(gap=None)
        with pytest.raises(ValueError):
            sp.encode_one_hot(gap="")
        with pytest.raises(ValueError):
            sp.encode_one_hot(gap="INVALID")

    @settings(max_examples=10, deadline=1000)
    @given(pad_at=st.sampled_from(["N", "C"]))
    def test_pad_at_valid(self, pad_at):
        """Test a valid 'pad_at' parameter."""
        sp = aa.SequencePreprocessor()
        result = sp.encode_one_hot(list_seq=["ARND"], pad_at=pad_at)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    def test_pad_at_invalid(self):
        """Test an invalid 'pad_at' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.encode_one_hot(pad_at=None)
        with pytest.raises(ValueError):
            sp.encode_one_hot(pad_at="")
        with pytest.raises(ValueError):
            sp.encode_one_hot(pad_at="INVALID")


# Complex Cases
class TestEncodeOneHotComplex:
    """Test encode_one_hot function for Complex Cases."""

    @settings(max_examples=10, deadline=1000)
    @given(list_seq=st.lists(st.text(alphabet="ACDEFGHIKLMNPQRSTVWY-", min_size=1), min_size=1),
           alphabet=st.text(min_size=1, alphabet="ACDEFGHIKLMNPQRSTVWY").filter(lambda a: "-" not in a),
           gap=st.text(min_size=1, max_size=1).filter(lambda g: g not in "ACDEFGHIKLMNPQRSTVWY"),
           pad_at=st.sampled_from(["N", "C"]))
    def test_valid_combination(self, list_seq, alphabet, gap, pad_at):
        """Test valid combinations of parameters."""
        sp = aa.SequencePreprocessor()
        # Filter list_seq to include only characters in the alphabet or the gap
        list_seq = ["".join([char if char in alphabet else gap for char in seq]) for seq in list_seq]
        result = sp.encode_one_hot(list_seq=list_seq, alphabet=alphabet, gap=gap, pad_at=pad_at)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], list)

    @settings(max_examples=10, deadline=1000)
    @given(
        list_seq=st.none(),
        alphabet=st.text(min_size=0),
        gap=st.text(min_size=0, max_size=2),
        pad_at=st.text(min_size=0)
    )
    def test_invalid_combination(self, list_seq, alphabet, gap, pad_at):
        """Test invalid combinations of parameters."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.encode_one_hot(list_seq=list_seq, alphabet=alphabet, gap=gap, pad_at=pad_at)
