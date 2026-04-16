"""
This is a script to test the AAlogo.get_df_logo method.
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# Helper
def get_df_parts(n=50):
    """Load default df_parts from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return sf.get_df_parts(df_seq=df_seq)


def get_labels(n=50):
    """Load labels from DOM_GSEC dataset."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq["label"].values


# ===========================================================================
# I Test get_df_logo: df_parts
# ===========================================================================
class TestGetDfLogoDfParts:
    """Test get_df_logo 'df_parts' parameter."""

    def test_valid_df_parts(self):
        """Test valid 'df_parts' returns a DataFrame."""
        df_parts = get_df_parts()
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts)
        assert isinstance(df_logo, pd.DataFrame)

    def test_valid_df_parts_single_part(self):
        """Test that df_parts with only one valid part column works."""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_parts_tmd = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts_tmd)
        assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_df_parts_none(self):
        """Test that df_parts=None raises ValueError."""
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=None)

    def test_invalid_df_parts_empty(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=pd.DataFrame())

    def test_invalid_df_parts_no_valid_columns(self):
        """Test that df_parts with no valid part columns raises ValueError."""
        df_bad = pd.DataFrame({"invalid_col": ["ACDE", "FGHI"]})
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_bad)

    def test_invalid_df_parts_wrong_type(self):
        """Test that non-DataFrame raises ValueError."""
        for df_parts in ["invalid", 1, [1, 2, 3]]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(df_parts=df_parts)

    def test_df_parts_not_mutated(self):
        """Test that get_df_logo does not mutate the input df_parts."""
        df_parts = get_df_parts()
        df_parts_copy = df_parts.copy()
        aa.AAlogo().get_df_logo(df_parts=df_parts)
        pd.testing.assert_frame_equal(df_parts, df_parts_copy)


# ===========================================================================
# II Test get_df_logo: labels and label_test
# ===========================================================================
class TestGetDfLogoLabels:
    """Test get_df_logo 'labels' and 'label_test' parameters."""

    def test_valid_labels(self):
        """Test valid 'labels' filters correctly."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [0, 1]:
            df_logo = aa.AAlogo().get_df_logo(
                df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(df_logo, pd.DataFrame)

    def test_valid_labels_none(self):
        """Test that labels=None uses all samples."""
        df_parts = get_df_parts()
        df_logo_all = aa.AAlogo().get_df_logo(df_parts=df_parts, labels=None)
        assert isinstance(df_logo_all, pd.DataFrame)

    def test_invalid_labels_wrong_length(self):
        """Test that labels with wrong length raises ValueError."""
        df_parts = get_df_parts()   # n=50
        for labels in [np.array([1, 0]), np.array([1] * 3)]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(df_parts=df_parts, labels=labels)

    def test_invalid_labels_wrong_type(self):
        """Test that non-array labels raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, labels="invalid")

    def test_valid_label_test(self):
        """Test valid 'label_test' integer values."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [0, 1]:
            df_logo = aa.AAlogo().get_df_logo(
                df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_label_test_type(self):
        """Test that non-integer label_test raises ValueError."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [None, 1.5, "invalid"]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(
                    df_parts=df_parts, labels=labels, label_test=label_test)

    def test_label_test_not_in_labels_raises(self):
        """Test that label_test absent from labels raises ValueError (empty-filter guard)."""
        df_parts = get_df_parts()
        labels = get_labels()
        # label_test=99 passes check_number_val (any int is valid) but
        # no samples have label=99, so the empty-filter guard raises
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, labels=labels, label_test=99)


# ===========================================================================
# III Test get_df_logo: tmd_len
# ===========================================================================
class TestGetDfLogoTmdLen:
    """Test get_df_logo 'tmd_len' parameter."""

    def test_valid_tmd_len_none(self):
        """Test that tmd_len=None uses the maximum TMD length."""
        df_parts = get_df_parts()
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=None)
        assert isinstance(df_logo, pd.DataFrame)

    def test_valid_tmd_len_values(self):
        """Test valid integer tmd_len values up to the maximum in df_parts."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        for tmd_len in [1, max_tmd_len // 2, max_tmd_len]:
            df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=tmd_len)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_tmd_len_zero(self):
        """Test that tmd_len=0 raises ValueError (min_val=1)."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=0)

    def test_invalid_tmd_len_negative(self):
        """Test that negative tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=-1)

    def test_invalid_tmd_len_float(self):
        """Test that float tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=1.5)

    def test_invalid_tmd_len_wrong_type(self):
        """Test that non-numeric tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len="invalid")

    def test_invalid_tmd_len_exceeds_max(self):
        """Test that tmd_len exceeding the maximum TMD length in df_parts raises ValueError."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=max_tmd_len + 1)


# ===========================================================================
# IV Test get_df_logo: start_n
# ===========================================================================
class TestGetDfLogoStartN:
    """Test get_df_logo 'start_n' parameter."""

    def test_valid_start_n(self):
        """Test valid 'start_n' values."""
        df_parts = get_df_parts()
        for start_n in [True, False]:
            df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts, start_n=start_n)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_start_n(self):
        """Test that non-bool start_n raises ValueError."""
        df_parts = get_df_parts()
        for start_n in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(df_parts=df_parts, start_n=start_n)

    def test_start_n_effect_on_truncation(self):
        """Test that start_n changes which end is kept when tmd_len forces truncation."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        short_tmd = max(1, max_tmd_len - 5)
        df_logo_n = aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=short_tmd, start_n=True)
        df_logo_c = aa.AAlogo().get_df_logo(df_parts=df_parts, tmd_len=short_tmd, start_n=False)
        # Both have the same shape; values differ when truncation is active
        assert df_logo_n.shape == df_logo_c.shape
        assert not df_logo_n.equals(df_logo_c)


# ===========================================================================
# V Test get_df_logo: characters_to_ignore
# ===========================================================================
class TestGetDfLogoCharactersToIgnore:
    """Test get_df_logo 'characters_to_ignore' parameter."""

    def test_valid_characters_to_ignore(self):
        """Test valid 'characters_to_ignore' string values."""
        df_parts = get_df_parts()
        for chars in [".-", ".", "-", ""]:
            df_logo = aa.AAlogo().get_df_logo(
                df_parts=df_parts, characters_to_ignore=chars)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_characters_to_ignore(self):
        """Test that non-string characters_to_ignore raises ValueError."""
        df_parts = get_df_parts()
        for chars in [None, 123, [], [".", "-"]]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(df_parts=df_parts, characters_to_ignore=chars)

    def test_gap_excluded_by_default(self):
        """Test that '-' does not appear as a column when using default characters_to_ignore."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        # Use short tmd_len to force gap padding
        short_tmd = max(1, max_tmd_len - 5)
        df_logo = aa.AAlogo().get_df_logo(
            df_parts=df_parts, tmd_len=short_tmd, characters_to_ignore=".-")
        assert "-" not in df_logo.columns


# ===========================================================================
# VI Test get_df_logo: pseudocount
# ===========================================================================
class TestGetDfLogoPseudocount:
    """Test get_df_logo 'pseudocount' parameter."""

    def test_valid_pseudocount(self):
        """Test valid 'pseudocount' values."""
        df_parts = get_df_parts()
        for pseudocount in [0.0, 0.1, 0.5, 1.0]:
            df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts, pseudocount=pseudocount)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_pseudocount(self):
        """Test that negative or non-numeric pseudocount raises ValueError."""
        df_parts = get_df_parts()
        for pseudocount in [-0.1, -1, "invalid", None]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo(df_parts=df_parts, pseudocount=pseudocount)

    def test_pseudocount_smooths_distribution(self):
        """Test that higher pseudocount reduces the maximum probability."""
        df_parts = get_df_parts()
        df_logo_raw = aa.AAlogo().get_df_logo(df_parts=df_parts, pseudocount=0.0)
        df_logo_smooth = aa.AAlogo().get_df_logo(df_parts=df_parts, pseudocount=1.0)
        assert df_logo_smooth.max().max() < df_logo_raw.max().max()


# ===========================================================================
# VII Complex: valid parameter combinations
# ===========================================================================
class TestGetDfLogoComplex:
    """Test get_df_logo with valid parameter combinations."""

    @settings(max_examples=5)
    @given(
        logo_type=st.sampled_from(["probability", "weight", "counts", "information"]),
        tmd_len=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
        start_n=st.booleans(),
        pseudocount=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_valid_combinations(self, logo_type, tmd_len, start_n, pseudocount):
        """Test valid parameter combinations return a DataFrame."""
        df_parts = get_df_parts()
        df_logo = aa.AAlogo(logo_type=logo_type).get_df_logo(
            df_parts=df_parts, tmd_len=tmd_len,
            start_n=start_n, pseudocount=pseudocount)
        assert isinstance(df_logo, pd.DataFrame)
        assert len(df_logo) > 0
