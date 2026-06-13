"""This is a script to test the encode_one_hot function."""
import numpy as np
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Normal Cases
class TestEncodeOneHot:
    """Test encode_one_hot function."""

    @settings(max_examples=10, deadline=None)
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

    @settings(max_examples=10, deadline=None)
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

    @settings(max_examples=10, deadline=None)
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

    @settings(max_examples=10, deadline=None)
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

    @settings(max_examples=10, deadline=None)
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


def _ref_one_hot(list_seq, alphabet="ACDEFGHIKLMNPQRSTVWY", gap="-", pad_at="C"):
    """Reference (per-residue) one-hot encoder used to pin the vectorized output."""
    from aaanalysis.data_handling._backend.seq_preproc._utils import pad_sequences
    padded = pad_sequences(list_seq, pad_at=pad_at, gap=gap)
    max_length, num = len(padded[0]), len(alphabet)
    feats = [f"{i}{aa}" for i in range(1, max_length + 1) for aa in alphabet]
    fm = np.zeros((len(padded), max_length * num), dtype=int)
    d = {a: i for i, a in enumerate(alphabet)}
    for r, seq in enumerate(padded):
        rows = []
        for aa_ in seq:
            v = np.zeros(num, dtype=int)
            if aa_ != gap:
                v[d[aa_]] = 1
            rows.append(v)
        fm[r, :] = np.array(rows).flatten()
    return fm, feats


class TestEncodeOneHotEquivalence:
    """The vectorized encoder must return output identical to the per-residue reference."""

    @pytest.mark.parametrize("pad_at", ["C", "N"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 7])
    def test_matches_reference(self, pad_at, seed):
        rng = np.random.default_rng(seed)
        aas = "ACDEFGHIKLMNPQRSTVWY"
        seqs = ["".join(rng.choice(list(aas), int(rng.integers(5, 40)))) for _ in range(60)]
        sp = aa.SequencePreprocessor()
        fm, feats = sp.encode_one_hot(list_seq=seqs, pad_at=pad_at)
        ref_fm, ref_feats = _ref_one_hot(seqs, pad_at=pad_at)
        assert feats == ref_feats
        assert fm.dtype == ref_fm.dtype
        assert np.array_equal(fm, ref_fm)

    @settings(max_examples=10, deadline=None)
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
