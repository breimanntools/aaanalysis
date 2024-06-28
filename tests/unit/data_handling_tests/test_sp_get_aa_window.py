"""This is a script to test the get_aa_window function."""
from typing import Optional
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Normal Cases
class TestGetAAWindow:
    """Test get_aa_window function."""

    @settings(max_examples=10, deadline=1000)
    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY-", min_size=10, max_size=100))
    def test_seq_valid(self, seq):
        """Test a valid 'seq' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq=seq, pos_start=0, pos_stop=5)
        assert isinstance(window, str)

    def test_seq_invalid(self):
        """Test an invalid 'seq' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq=None, pos_start=0, pos_stop=5)
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="", pos_start=0, pos_stop=5, accept_gap=False)

    @settings(max_examples=10, deadline=1000)
    @given(pos_start=st.integers(min_value=0, max_value=50))
    def test_pos_start_valid(self, pos_start):
        """Test a valid 'pos_start' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=pos_start, pos_stop=pos_start + 5)
        assert isinstance(window, str)

    def test_pos_start_invalid(self):
        """Test an invalid 'pos_start' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=-1, pos_stop=5)
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=100, pos_stop=5)

    @settings(max_examples=10, deadline=1000)
    @given(pos_stop=st.integers(min_value=0, max_value=50))
    def test_pos_stop_valid(self, pos_stop):
        """Test a valid 'pos_stop' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=pos_stop)
        assert isinstance(window, str)

    def test_pos_stop_invalid(self):
        """Test an invalid 'pos_stop' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=-1)
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=100, accept_gap=False)

    @settings(max_examples=10, deadline=1000)
    @given(window_size=st.integers(min_value=1, max_value=50))
    def test_window_size_valid(self, window_size):
        """Test a valid 'window_size' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, window_size=window_size)
        assert isinstance(window, str)

    def test_window_size_invalid(self):
        """Test an invalid 'window_size' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, window_size=0)
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, window_size=-5)

    @settings(max_examples=10, deadline=1000)
    @given(index1=st.booleans())
    def test_index1_valid(self, index1):
        """Test a valid 'index1' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=1 if index1 else 0, pos_stop=5, index1=index1)
        assert isinstance(window, str)

    def test_index1_invalid(self):
        """Test an invalid 'index1' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=1, pos_stop=5, index1=None)

    @settings(max_examples=10, deadline=1000)
    @given(gap=st.text(min_size=1, max_size=1).filter(lambda g: g not in "ACDEFGHIKLMNPQRSTVWY"))
    def test_gap_valid(self, gap):
        """Test a valid 'gap' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=5, gap=gap)
        assert isinstance(window, str)

    def test_gap_invalid(self):
        """Test an invalid 'gap' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=5, gap="")

    @settings(max_examples=10, deadline=1000)
    @given(accept_gap=st.booleans())
    def test_accept_gap_valid(self, accept_gap):
        """Test a valid 'accept_gap' parameter."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=5, accept_gap=accept_gap)
        assert isinstance(window, str)

    def test_accept_gap_invalid(self):
        """Test an invalid 'accept_gap' parameter."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(seq="ACDEFGHIKLMNPQRSTVWY", pos_start=0, pos_stop=5, accept_gap=None)


# Complex Cases
class TestGetAAWindowComplex:
    """Test get_aa_window function for Complex Cases."""

    @settings(max_examples=10, deadline=1000)
    @given(
        seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=10, max_size=100),
        pos_start=st.integers(min_value=1, max_value=50),
        window_size=st.integers(min_value=1, max_value=50),
        index1=st.booleans(),
        gap=st.text(min_size=1, max_size=1).filter(lambda g: g not in "ACDEFGHIKLMNPQRSTVWY"),
    )
    def test_valid_combination(self, seq, pos_start, window_size, index1, gap):
        """Test valid combinations of parameters."""
        sp = aa.SequencePreprocessor()
        window = sp.get_aa_window(seq=seq,
                                  pos_start=pos_start,
                                  window_size=window_size,
                                  index1=index1,
                                  gap=gap)
        assert isinstance(window, str)

    @settings(max_examples=10, deadline=1000)
    @given(
        seq=st.none(),
        pos_start=st.integers(min_value=-10, max_value=-1),
        pos_stop=st.integers(min_value=-10, max_value=-1),
        window_size=st.integers(min_value=-10, max_value=0),
        index1=st.none(),
        gap=st.text(min_size=0, max_size=2),
        accept_gap=st.none()
    )
    def test_invalid_combination(self, seq, pos_start, pos_stop, window_size, index1, gap, accept_gap):
        """Test invalid combinations of parameters."""
        sp = aa.SequencePreprocessor()
        with pytest.raises(ValueError):
            sp.get_aa_window(
                seq=seq,
                pos_start=pos_start,
                pos_stop=pos_stop,
                window_size=window_size,
                index1=index1,
                gap=gap,
                accept_gap=accept_gap
            )
