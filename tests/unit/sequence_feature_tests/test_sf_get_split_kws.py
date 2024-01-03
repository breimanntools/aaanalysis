"""This is a script to test the SequenceFeature().get_df_parts() method ."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
aa.options["verbose"] = False


class TestGetSplitKws:
    """Test the get_split_kws static method."""

    @settings(max_examples=10)
    @given(split_types=st.sampled_from(
        [None, "Segment", "Pattern", "PeriodicPattern", ["Segment", "Pattern"], ["Pattern", "PeriodicPattern"],
         ["Segment", "PeriodicPattern"]]))
    def test_split_types(self, split_types):
        """Test different 'split_types'."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(split_types=split_types)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_split_min=st.integers(min_value=1, max_value=14))
    def test_n_split_min(self, n_split_min):
        """Test 'n_split_min' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_split_min=n_split_min)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_split_max=st.integers(min_value=2, max_value=15))
    def test_n_split_max(self, n_split_max):
        """Test 'n_split_max' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_split_max=n_split_max)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(steps_pattern=st.lists(st.integers(min_value=1), min_size=1, max_size=8))
    def test_steps_pattern(self, steps_pattern):
        """Test 'steps_pattern' with various list sizes."""
        sf = aa.SequenceFeature()
        if len(steps_pattern) > 0:
            result = sf.get_split_kws(steps_pattern=steps_pattern, len_max=steps_pattern[0]+1)
            assert isinstance(result, dict)
        result = sf.get_split_kws(steps_pattern=[9, 15], len_max=10)
        assert isinstance(result, dict)


    @settings(max_examples=10)
    @given(n_min=st.integers(min_value=1, max_value=4))
    def test_n_min(self, n_min):
        """Test 'n_min' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_min=n_min)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_max=st.integers(min_value=2, max_value=4))
    def test_n_max(self, n_max):
        """Test 'n_max' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_max=n_max)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(len_max=st.integers(min_value=4, max_value=15))
    def test_len_max(self, len_max):
        """Test 'len_max' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(len_max=len_max)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(steps_periodicpattern=st.lists(st.integers(min_value=1), min_size=2, max_size=2))
    def test_steps_periodicpattern(self, steps_periodicpattern):
        """Test 'steps_periodicpattern' with various list sizes."""
        sf = aa.SequenceFeature()
        if len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    # Negative tests for each parameter
    def test_invalid_split_types(self):
        """Test invalid 'split_types' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types="InvalidType")
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types=["Segment", "InvalidType"])

    def test_invalid_n_split_min(self):
        """Test invalid 'n_split_min' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=0)

    def test_invalid_n_split_max(self):
        """Test invalid 'n_split_max' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_max=1, n_split_min=2)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_max=0)

    def test_invalid_steps_pattern(self):
        """Test invalid 'steps_pattern' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=-1)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=["a", "b", "c"])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[0])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[-4, 10])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[3, None])

    def test_invalid_n_min_max(self):
        """Test invalid 'n_min' and 'n_max' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_min=5, n_max=4)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_min=0, n_max=3)

    def test_invalid_len_max(self):
        """Test invalid 'len_max' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(len_max=0)
        with pytest.raises(ValueError):
            sf.get_split_kws(len_max=3, steps_pattern=[4, 5])

    def test_invalid_steps_periodicpattern(self):
        """Test invalid 'steps_periodicpattern' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=-1)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=["a", "b", "c"])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[0])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[-4, 10])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[3, 4, 5])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[3, None])


class TestGetSplitKwsComplex:
    """Test complex combinations of parameters in get_split_kws."""

    @settings(max_examples=5, deadline=500)
    @given(
        split_types=st.sampled_from([None, "Segment", "Pattern", "PeriodicPattern", ["Segment", "Pattern"], ["Pattern", "PeriodicPattern"], ["Segment", "PeriodicPattern"]]),
        n_split_min=st.integers(min_value=1, max_value=14),
        n_split_max=st.integers(min_value=2, max_value=15),
        steps_pattern=st.lists(st.integers(min_value=1), min_size=1, max_size=8),
        n_min=st.integers(min_value=1, max_value=4),
        n_max=st.integers(min_value=2, max_value=4),
        len_max=st.integers(min_value=4, max_value=15),
        steps_periodicpattern=st.lists(st.integers(min_value=1), min_size=2, max_size=2)
    )
    def test_valid_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test valid combinations of parameters."""
        sf = aa.SequenceFeature()
        if n_split_min > n_split_max:
            n_split_min, n_split_max = n_split_max, n_split_min  # Ensure min <= max
        if n_min > n_max:
            n_min, n_max = n_max, n_min  # Ensure n_min <= n_max
        if steps_pattern and len_max <= min(steps_pattern):
            len_max = min(steps_pattern) + 1  # Ensure len_max > min(steps_pattern)
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    @settings(max_examples=5, deadline=500)
    @given(
        split_types=st.sampled_from(["Segment", ["Pattern", "PeriodicPattern"]]),
        n_split_min=st.integers(min_value=1, max_value=3),
        n_split_max=st.integers(min_value=10, max_value=15),
        steps_pattern=st.just([3, 4, 5]),
        n_min=st.integers(min_value=1, max_value=2),
        n_max=st.integers(min_value=3, max_value=4),
        len_max=st.integers(min_value=5, max_value=10),
        steps_periodicpattern=st.just([3, 4])
    )
    def test_edge_case_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test edge case combinations of parameters."""
        sf = aa.SequenceFeature()
        if n_split_min > n_split_max:
            n_split_min, n_split_max = n_split_max, n_split_min  # Ensure min <= max
        if n_min > n_max:
            n_min, n_max = n_max, n_min  # Ensure n_min <= n_max
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    @settings(max_examples=5, deadline=500)
    @given(
        split_types=st.sampled_from([None, ["Segment", "Pattern"]]),
        n_split_min=st.integers(min_value=5, max_value=7),
        n_split_max=st.integers(min_value=8, max_value=10),
        steps_pattern=st.lists(st.integers(min_value=1, max_value=2), min_size=2, max_size=4),
        n_min=st.integers(min_value=1, max_value=2),
        n_max=st.integers(min_value=2, max_value=5),
        len_max=st.integers(min_value=11, max_value=15),
        steps_periodicpattern=st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=4)
    )
    def test_random_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test random valid combinations of parameters."""
        sf = aa.SequenceFeature()
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    # Negative complex cases
    def test_invalid_combinations(self):
        """Test invalid combinations of parameters."""
        sf = aa.SequenceFeature()
        # Example of an invalid combination
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=15, n_split_max=14)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[1, 2], len_max=1)

    def test_invalid_random_combinations(self):
        """Test invalid random combinations of parameters."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=10, n_split_max=5, steps_pattern=[5, 2, 3], len_max=1)
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types=["Invalid", "Segment"], n_min=4, n_max=3)
