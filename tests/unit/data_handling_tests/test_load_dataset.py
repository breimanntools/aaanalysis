"""
This is a script for testing the aa.load_dataset function.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
import aaanalysis.utils as ut
import aaanalysis as aa
import pytest

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


class TestLoadDataset:
    """Test load_dataset function"""

    # Property-based testing for positive cases
    def test_df_seq_output_columns(self):
        """"""
        all_data_set_names = aa.load_dataset()["Dataset"].to_list()
        for name in all_data_set_names:
            df = aa.load_dataset(name=name)
            assert set(ut.COLS_SEQ_INFO).issubset(set(df))

    @settings(max_examples=10, deadline=350)
    @given(n=some.integers(min_value=1, max_value=100))
    def test_load_dataset_n_value(self, n):
        """Test the 'n' parameter for limiting rows."""
        max_n = aa.load_dataset(name="SEQ_LOCATION")["label"].value_counts().min()
        if max_n > n:
            df = aa.load_dataset(name="SEQ_LOCATION", n=n)
            assert len(df) == (n * 2)

    @settings(max_examples=10, deadline=350)
    @given(min_len=some.integers(min_value=400, max_value=1000))
    def test_load_dataset_min_len(self, min_len):
        """Test the 'min_len' parameter for filtering sequences."""
        df = aa.load_dataset(name="SEQ_LOCATION", min_len=min_len)
        assert all(len(seq) >= min_len for seq in df[ut.COL_SEQ])

    @settings(max_examples=10, deadline=350)
    @given(max_len=some.integers(min_value=50, max_value=100))
    def test_load_dataset_max_len(self, max_len):
        """Test the 'max_len' parameter for filtering sequences."""
        df = aa.load_dataset(name="SEQ_LOCATION", max_len=max_len)
        assert all(len(seq) <= max_len for seq in df[ut.COL_SEQ])

    # Property-based testing for negative cases
    @settings(max_examples=10, deadline=350)
    @given(n=some.integers(max_value=0))
    def test_load_dataset_invalid_n(self, n):
        """Test with an invalid 'n' value."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", n=n)

    @settings(max_examples=10, deadline=350)
    @given(min_len=some.integers(max_value=0))
    def test_load_dataset_invalid_min_len(self, min_len):
        """Test with an invalid 'min_len' value."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", min_len=min_len)
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_AMYLO", min_len=10)

    @settings(max_examples=10, deadline=350)
    @given(max_len=some.integers(max_value=0))
    def test_load_dataset_invalid_max_len(self, max_len):
        """Test with an invalid 'max_len' value."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", max_len=max_len)

    # Additional Negative Tests
    @settings(max_examples=10, deadline=350)
    @given(n=some.integers(min_value=1000, max_value=1050))
    def test_load_dataset_n_value_too_high(self, n):
        """Test the 'n' parameter for limiting rows."""
        max_n = aa.load_dataset(name="SEQ_LOCATION")["label"].value_counts().min()
        if max_n < n:
            with pytest.warns(UserWarning):
                df = aa.load_dataset(name="SEQ_LOCATION", n=n)

    @settings(max_examples=10, deadline=350)
    @given(negative_n=some.integers(min_value=-100, max_value=-1))
    def test_load_dataset_negative_n(self, negative_n):
        """Test with a negative 'n' value."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", n=negative_n)

    @settings(max_examples=10, deadline=350)
    @given(non_canonical_aa=some.text())
    @example(non_canonical_aa="invalid_option")
    def test_load_dataset_invalid_non_canonical_aa(self, non_canonical_aa):
        """Test with an invalid 'non_canonical_aa' value."""
        if non_canonical_aa not in ["remove", "keep", "gap"]:
            with pytest.raises(ValueError):
                aa.load_dataset(name="SEQ_LOCATION", non_canonical_aa=non_canonical_aa)


class TestLoadDatasetComplex:
    """Test load_dataset function with complex scenarios"""

    def test_load_dataset_n_and_min_len(self):
        """Test the 'n' and 'min_len' parameters together."""
        df = aa.load_dataset(name="SEQ_LOCATION", n=10, min_len=5)
        assert len(df) == 10 * 2
        assert all(len(seq) >= 5 for seq in df[ut.COL_SEQ])

    def test_load_dataset_n_and_max_len(self):
        """Test the 'n' and 'max_len' parameters together."""
        df = aa.load_dataset(name="SEQ_LOCATION", n=10, max_len=200)
        assert len(df) == 10 * 2
        assert all(len(seq) <= 200 for seq in df[ut.COL_SEQ])

    def test_load_dataset_min_max_len(self):
        """Test both 'min_len' and 'max_len' together."""
        df = aa.load_dataset(name="SEQ_LOCATION", min_len=5, max_len=200)
        assert all(5 <= len(seq) <= 200 for seq in df[ut.COL_SEQ])

    def test_load_dataset_min_max_len_and_n(self):
        """Test 'min_len', 'max_len', and 'n' together."""
        df = aa.load_dataset(name="SEQ_LOCATION", min_len=5, max_len=200, n=10)
        assert len(df) == 10 * 2
        assert all(5 <= len(seq) <= 200 for seq in df[ut.COL_SEQ])

    def test_load_dataset_all_filters(self):
        """Test all filters together ('n', 'min_len', 'max_len', 'non_canonical_aa')."""
        df = aa.load_dataset(name="SEQ_LOCATION", n=10, min_len=5, max_len=200, non_canonical_aa="remove")
        assert len(df) == 10 * 2
        assert all(5 <= len(seq) <= 200 for seq in df[ut.COL_SEQ])
        # Add your assertion to check if non-canonical amino acids are removed

    def test_load_dataset_invalid_min_max_len(self):
        """Test with 'min_len' greater than 'max_len'."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", min_len=10, max_len=5)

    def test_load_dataset_invalid_min_max_len_and_n(self):
        """Test with 'min_len' greater than 'max_len' and a valid 'n'."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", min_len=10, max_len=5, n=10)

    def test_load_dataset_invalid_all_filters(self):
        """Test with all invalid filters ('n', 'min_len', 'max_len', 'non_canonical_aa')."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", n=-1, min_len=10, max_len=5, non_canonical_aa="invalid_option")

    # Invalid dataset name (lists valid options)
    def test_load_dataset_invalid_name(self):
        """Unknown 'name' raises a ValueError listing valid datasets."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="NOT_A_DATASET")

    # Non-canonical amino acid handling
    def test_load_dataset_non_canonical_keep(self):
        """non_canonical_aa='keep' returns the dataset unchanged (no filtering)."""
        df_keep = aa.load_dataset(name="SEQ_LOCATION", non_canonical_aa="keep")
        df_remove = aa.load_dataset(name="SEQ_LOCATION", non_canonical_aa="remove")
        assert len(df_keep) >= len(df_remove)

    def test_load_dataset_non_canonical_gap(self):
        """non_canonical_aa='gap' replaces non-canonical residues with the gap symbol."""
        df = aa.load_dataset(name="SEQ_LOCATION", non_canonical_aa="gap")
        allowed = set(ut.LIST_CANONICAL_AA) | {ut.STR_AA_GAP}
        assert all(set(seq).issubset(allowed) for seq in df[ut.COL_SEQ])

    # AA-level windowing
    def test_load_dataset_aa_window_size_none_returns_unwindowed(self):
        """aa_window_size=None on an AA dataset returns the unfiltered residue table."""
        df = aa.load_dataset(name="AA_LDR", aa_window_size=None)
        assert set(ut.COLS_SEQ_INFO).issubset(set(df))
        assert len(df) > 0

    def test_load_dataset_aa_window_odd(self):
        """Odd aa_window_size on a non-cleavage AA dataset yields fixed-length windows."""
        size = 9
        df = aa.load_dataset(name="AA_LDR", aa_window_size=size, n=20)
        assert all(len(seq) == size for seq in df[ut.COL_SEQ])

    def test_load_dataset_aa_window_even_cleavage_site(self):
        """Even aa_window_size is allowed for cleavage-site datasets and yields fixed windows."""
        size = 8
        df = aa.load_dataset(name="AA_CASPASE3", aa_window_size=size, n=20)
        assert all(len(seq) == size for seq in df[ut.COL_SEQ])

    def test_load_dataset_aa_window_even_non_cleavage_invalid(self):
        """Even aa_window_size on a non-cleavage AA dataset raises ValueError."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="AA_LDR", aa_window_size=8)

    def test_load_dataset_aa_window_too_large_invalid(self):
        """aa_window_size larger than the shortest sequence raises ValueError (odd path)."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="AA_LDR", aa_window_size=9999)

    def test_load_dataset_aa_window_even_too_large_invalid(self):
        """Even aa_window_size larger than the shortest sequence raises (cleavage even path)."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="AA_CASPASE3", aa_window_size=9998)

    # Filters that remove everything
    def test_load_dataset_max_len_removes_all(self):
        """A max_len below every sequence length removes all rows and raises."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", max_len=1)

    # Random balanced selection
    def test_load_dataset_random_selection(self):
        """random=True returns n balanced samples per class."""
        df = aa.load_dataset(name="SEQ_LOCATION", n=5, random=True)
        assert len(df) == 5 * 2
        assert set(df[ut.COL_LABEL]) == {0, 1}

