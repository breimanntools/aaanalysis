"""
This is a script to test the AALogo class.
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

# Set default deadline from 200 to 20000
settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# Helper
def get_df_parts(n=10):
    """Load default df_parts from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return sf.get_df_parts(df_seq=df_seq)


def get_labels(n=10):
    """Load labels from DOM_GSEC dataset."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq["label"].values


def get_df_logo_info(n=10):
    """Get pre-computed df_logo_info for conservation tests."""
    df_parts = get_df_parts(n=n)
    aalogo = aa.AAlogo()
    return aalogo.get_df_logo_info(df_parts=df_parts)


# ===========================================================================
# I Test AALogo.__init__
# ===========================================================================
class TestAALogoInit:
    """Test AALogo class initialization."""

    def test_valid_logo_type(self):
        """Test valid 'logo_type' parameter."""
        for logo_type in ["probability", "weight", "counts", "information"]:
            aalogo = aa.AAlogo(logo_type=logo_type)
            assert aalogo._logo_type == logo_type

    def test_invalid_logo_type(self):
        """Test invalid 'logo_type' parameter."""
        for logo_type in [None, 0, "invalid", "bits", []]:
            with pytest.raises(ValueError):
                aa.AAlogo(logo_type=logo_type)


# ===========================================================================
# II Test AALogo.get_df_logo
# ===========================================================================
class TestGetDfLogo:
    """Test AALogo.get_df_logo method."""

    def test_valid_df_parts(self):
        """Test valid 'df_parts' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        df_logo = aalogo.get_df_logo(df_parts=df_parts)
        assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_df_parts(self):
        """Test invalid 'df_parts' parameter."""
        aalogo = aa.AAlogo()
        for df_parts in [None, pd.DataFrame(), pd.DataFrame({"invalid": ["ACDE", "FGHI"]}), "invalid", 1]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts)

    def test_valid_labels(self):
        """Test valid 'labels' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        df_logo = aalogo.get_df_logo(df_parts=df_parts, labels=labels, label_test=1)
        assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_labels(self):
        """Test invalid 'labels' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for labels in [np.array([1, 0]), "invalid", [1, 2, 3]]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, labels=labels)

    def test_valid_label_test(self):
        """Test valid 'label_test' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        for label_test in [0, 1]:
            df_logo = aalogo.get_df_logo(df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_label_test(self):
        """Test invalid 'label_test' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        for label_test in [99, "invalid", None]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, labels=labels, label_test=label_test)

    def test_valid_tmd_len(self):
        """Test valid 'tmd_len' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for tmd_len in [None, 5, 10, 20]:
            df_logo = aalogo.get_df_logo(df_parts=df_parts, tmd_len=tmd_len)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_tmd_len(self):
        """Test invalid 'tmd_len' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for tmd_len in [0, -1, 99999, "invalid", 1.5]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, tmd_len=tmd_len)

    def test_valid_start_n(self):
        """Test valid 'start_n' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for start_n in [True, False]:
            df_logo = aalogo.get_df_logo(df_parts=df_parts, start_n=start_n)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_start_n(self):
        """Test invalid 'start_n' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for start_n in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, start_n=start_n)

    def test_valid_characters_to_ignore(self):
        """Test valid 'characters_to_ignore' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for characters_to_ignore in [".-", ".", "-", ""]:
            df_logo = aalogo.get_df_logo(df_parts=df_parts, characters_to_ignore=characters_to_ignore)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_characters_to_ignore(self):
        """Test invalid 'characters_to_ignore' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for characters_to_ignore in [None, 123, [], [".", "-"]]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, characters_to_ignore=characters_to_ignore)

    def test_valid_pseudocount(self):
        """Test valid 'pseudocount' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for pseudocount in [0.0, 0.5, 1.0]:
            df_logo = aalogo.get_df_logo(df_parts=df_parts, pseudocount=pseudocount)
            assert isinstance(df_logo, pd.DataFrame)

    def test_invalid_pseudocount(self):
        """Test invalid 'pseudocount' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for pseudocount in [-0.1, -1, "invalid", None]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo(df_parts=df_parts, pseudocount=pseudocount)


# ===========================================================================
# III Test AALogo.get_df_logo_info
# ===========================================================================
class TestGetDfLogoInfo:
    """Test AALogo.get_df_logo_info method."""

    def test_valid_df_parts(self):
        """Test valid 'df_parts' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        result = aalogo.get_df_logo_info(df_parts=df_parts)
        assert isinstance(result, pd.Series)
        assert result.index.name == "pos"

    def test_invalid_df_parts(self):
        """Test invalid 'df_parts' parameter."""
        aalogo = aa.AAlogo()
        for df_parts in [None, pd.DataFrame(), pd.DataFrame({"invalid": ["ACDE", "FGHI"]}), "invalid", 1]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts)

    def test_valid_labels(self):
        """Test valid 'labels' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        for label_test in [0, 1]:
            result = aalogo.get_df_logo_info(df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(result, pd.Series)

    def test_invalid_labels(self):
        """Test invalid 'labels' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for labels in [np.array([1, 0]), "invalid", [1, 2, 3]]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts, labels=labels)

    def test_valid_label_test(self):
        """Test valid 'label_test' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        for label_test in [0, 1]:
            result = aalogo.get_df_logo_info(df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(result, pd.Series)

    def test_invalid_label_test(self):
        """Test invalid 'label_test' parameter."""
        df_parts = get_df_parts()
        labels = get_labels()
        aalogo = aa.AAlogo()
        for label_test in [99, "invalid", None]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts, labels=labels, label_test=label_test)

    def test_valid_tmd_len(self):
        """Test valid 'tmd_len' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for tmd_len in [None, 5, 10, 20]:
            result = aalogo.get_df_logo_info(df_parts=df_parts, tmd_len=tmd_len)
            assert isinstance(result, pd.Series)

    def test_invalid_tmd_len(self):
        """Test invalid 'tmd_len' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for tmd_len in [0, -1, 99999, "invalid", 1.5]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts, tmd_len=tmd_len)

    def test_valid_start_n(self):
        """Test valid 'start_n' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for start_n in [True, False]:
            result = aalogo.get_df_logo_info(df_parts=df_parts, start_n=start_n)
            assert isinstance(result, pd.Series)

    def test_invalid_start_n(self):
        """Test invalid 'start_n' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for start_n in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts, start_n=start_n)

    def test_valid_pseudocount(self):
        """Test valid 'pseudocount' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for pseudocount in [0.0, 0.5, 1.0]:
            result = aalogo.get_df_logo_info(df_parts=df_parts, pseudocount=pseudocount)
            assert isinstance(result, pd.Series)

    def test_invalid_pseudocount(self):
        """Test invalid 'pseudocount' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        for pseudocount in [-0.1, -1, "invalid", None]:
            with pytest.raises(ValueError):
                aalogo.get_df_logo_info(df_parts=df_parts, pseudocount=pseudocount)

    def test_logo_type_unchanged(self):
        """Test that logo_type attribute is not mutated by get_df_logo_info."""
        df_parts = get_df_parts()
        for logo_type in ["probability", "counts", "weight"]:
            aalogo = aa.AAlogo(logo_type=logo_type)
            aalogo.get_df_logo_info(df_parts=df_parts)
            assert aalogo._logo_type == logo_type

    def test_values_non_negative(self):
        """Test that information content values are non-negative."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        result = aalogo.get_df_logo_info(df_parts=df_parts)
        assert (result >= 0).all()


# ===========================================================================
# IV Test AALogo.get_conservation
# ===========================================================================
class TestGetConservation:
    """Test AALogo.get_conservation method."""

    def test_valid_df_logo_info(self):
        """Test valid 'df_logo_info' parameter."""
        df_logo_info = get_df_logo_info()
        result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info)
        assert isinstance(result, float)

    def test_invalid_df_logo_info(self):
        """Test invalid 'df_logo_info' parameter."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo()
        df_logo = aalogo.get_df_logo(df_parts=df_parts)
        for df_logo_info in [None, df_logo, "invalid", 1]:
            with pytest.raises(ValueError):
                aa.AAlogo.get_conservation(df_logo_info=df_logo_info)

    def test_invalid_df_logo_info_wrong_index_name(self):
        """Test that df_logo_info with wrong index name raises ValueError."""
        df_logo_info = get_df_logo_info()
        df_logo_info.index.name = "wrong"
        with pytest.raises(ValueError):
            aa.AAlogo.get_conservation(df_logo_info=df_logo_info)

    def test_valid_value_type(self):
        """Test valid 'value_type' parameter."""
        df_logo_info = get_df_logo_info()
        for value_type in ["min", "mean", "median", "max"]:
            result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type=value_type)
            assert isinstance(result, float)

    def test_invalid_value_type(self):
        """Test invalid 'value_type' parameter."""
        df_logo_info = get_df_logo_info()
        for value_type in [None, "sum", "average", 0, []]:
            with pytest.raises(ValueError):
                aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type=value_type)

    def test_result_non_negative(self):
        """Test that conservation score is non-negative."""
        df_logo_info = get_df_logo_info()
        result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info)
        assert result >= 0

    def test_min_leq_mean_leq_max(self):
        """Test that min <= mean <= max ordering holds."""
        df_logo_info = get_df_logo_info()
        val_min = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="min")
        val_mean = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="mean")
        val_max = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="max")
        assert val_min <= val_mean <= val_max


# ===========================================================================
# V Complex Cases
# ===========================================================================
class TestAALogoComplex:
    """Test AALogo for complex parameter combinations."""

    @settings(max_examples=5)
    @given(logo_type=st.sampled_from(["probability", "weight", "counts", "information"]),
           tmd_len=st.one_of(st.none(), st.integers(min_value=1, max_value=20)),
           start_n=st.booleans(),
           pseudocount=st.floats(min_value=0.0, max_value=1.0))
    def test_valid_get_df_logo_combinations(self, logo_type, tmd_len, start_n, pseudocount):
        """Test valid combinations of get_df_logo parameters."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo(logo_type=logo_type)
        df_logo = aalogo.get_df_logo(df_parts=df_parts, tmd_len=tmd_len,
                                     start_n=start_n, pseudocount=pseudocount)
        assert isinstance(df_logo, pd.DataFrame)

    @settings(max_examples=5)
    @given(logo_type=st.sampled_from(["probability", "weight", "counts", "information"]),
           tmd_len=st.one_of(st.none(), st.integers(min_value=1, max_value=20)),
           start_n=st.booleans(),
           pseudocount=st.floats(min_value=0.0, max_value=1.0))
    def test_valid_get_df_logo_info_combinations(self, logo_type, tmd_len, start_n, pseudocount):
        """Test valid combinations of get_df_logo_info parameters."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo(logo_type=logo_type)
        result = aalogo.get_df_logo_info(df_parts=df_parts, tmd_len=tmd_len,
                                         start_n=start_n, pseudocount=pseudocount)
        assert isinstance(result, pd.Series)
        assert result.index.name == "pos"
        assert aalogo._logo_type == logo_type

    @settings(max_examples=5)
    @given(value_type=st.sampled_from(["min", "mean", "median", "max"]))
    def test_valid_get_conservation_combinations(self, value_type):
        """Test valid combinations of get_conservation parameters."""
        df_logo_info = get_df_logo_info()
        result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type=value_type)
        assert isinstance(result, float)
        assert result >= 0