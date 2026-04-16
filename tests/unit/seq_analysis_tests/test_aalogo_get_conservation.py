"""
This is a script to test the AAlogo.get_conservation method.
"""
import pytest
import pandas as pd
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# Helper
def get_df_logo_info(n=50):
    """Get pre-computed df_logo_info from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    df_parts = sf.get_df_parts(df_seq=df_seq)
    return aa.AAlogo().get_df_logo_info(df_parts=df_parts)


# ===========================================================================
# I Test get_conservation: df_logo_info
# ===========================================================================
class TestGetConservationDfLogoInfo:
    """Test get_conservation 'df_logo_info' parameter."""

    def test_valid_df_logo_info(self):
        """Test valid 'df_logo_info' returns a float."""
        df_logo_info = get_df_logo_info()
        result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info)
        assert isinstance(result, float)

    def test_invalid_df_logo_info_none(self):
        """Test that df_logo_info=None raises ValueError."""
        with pytest.raises(ValueError):
            aa.AAlogo.get_conservation(df_logo_info=None)

    def test_invalid_df_logo_info_dataframe(self):
        """Test that a DataFrame (not Series) raises ValueError."""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts)
        with pytest.raises(ValueError):
            aa.AAlogo.get_conservation(df_logo_info=df_logo)

    def test_invalid_df_logo_info_wrong_type(self):
        """Test that non-Series raises ValueError."""
        for df_logo_info in ["invalid", 1, [1.0, 2.0]]:
            with pytest.raises(ValueError):
                aa.AAlogo.get_conservation(df_logo_info=df_logo_info)

    def test_invalid_df_logo_info_wrong_index_name(self):
        """Test that Series with index.name != 'pos' raises ValueError."""
        df_logo_info = get_df_logo_info()
        df_logo_info.index.name = "wrong"
        with pytest.raises(ValueError):
            aa.AAlogo.get_conservation(df_logo_info=df_logo_info)

    def test_valid_df_logo_info_index_name_pos(self):
        """Test that Series with index.name == 'pos' passes."""
        df_logo_info = get_df_logo_info()
        assert df_logo_info.index.name == "pos"
        result = aa.AAlogo.get_conservation(df_logo_info=df_logo_info)
        assert isinstance(result, float)


# ===========================================================================
# II Test get_conservation: value_type
# ===========================================================================
class TestGetConservationValueType:
    """Test get_conservation 'value_type' parameter."""

    def test_valid_value_type(self):
        """Test all valid 'value_type' options return a float."""
        df_logo_info = get_df_logo_info()
        for value_type in ["min", "mean", "median", "max"]:
            result = aa.AAlogo.get_conservation(
                df_logo_info=df_logo_info, value_type=value_type)
            assert isinstance(result, float)

    def test_invalid_value_type(self):
        """Test that invalid 'value_type' raises ValueError."""
        df_logo_info = get_df_logo_info()
        for value_type in [None, "sum", "average", "std", 0, []]:
            with pytest.raises(ValueError):
                aa.AAlogo.get_conservation(
                    df_logo_info=df_logo_info, value_type=value_type)

    def test_default_value_type_is_mean(self):
        """Test that the default value_type is 'mean'."""
        df_logo_info = get_df_logo_info()
        result_default = aa.AAlogo.get_conservation(df_logo_info=df_logo_info)
        result_mean = aa.AAlogo.get_conservation(
            df_logo_info=df_logo_info, value_type="mean")
        assert result_default == result_mean


# ===========================================================================
# III Test get_conservation: behavioral
# ===========================================================================
class TestGetConservationBehavior:
    """Test get_conservation behavioral properties."""

    def test_result_non_negative(self):
        """Test that all value_type options return a non-negative result."""
        df_logo_info = get_df_logo_info()
        for value_type in ["min", "mean", "median", "max"]:
            result = aa.AAlogo.get_conservation(
                df_logo_info=df_logo_info, value_type=value_type)
            assert result >= 0

    def test_min_leq_mean_leq_median_leq_max(self):
        """Test that min <= mean <= max ordering holds."""
        df_logo_info = get_df_logo_info()
        val_min = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="min")
        val_mean = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="mean")
        val_max = aa.AAlogo.get_conservation(df_logo_info=df_logo_info, value_type="max")
        assert val_min <= val_mean <= val_max

    def test_matches_pandas_aggregation(self):
        """Test that each value_type matches the corresponding pandas method directly."""
        df_logo_info = get_df_logo_info()
        expected = {
            "min":    df_logo_info.min(),
            "mean":   df_logo_info.mean(),
            "median": df_logo_info.median(),
            "max":    df_logo_info.max(),
        }
        for value_type, expected_val in expected.items():
            result = aa.AAlogo.get_conservation(
                df_logo_info=df_logo_info, value_type=value_type)
            assert result == expected_val, (
                f"value_type='{value_type}': got {result}, expected {expected_val}")


# ===========================================================================
# IV Complex: valid parameter combinations
# ===========================================================================
class TestGetConservationComplex:
    """Test get_conservation with valid parameter combinations."""

    @settings(max_examples=5)
    @given(value_type=st.sampled_from(["min", "mean", "median", "max"]))
    def test_valid_combinations(self, value_type):
        """Test valid parameter combinations return a non-negative float."""
        df_logo_info = get_df_logo_info()
        result = aa.AAlogo.get_conservation(
            df_logo_info=df_logo_info, value_type=value_type)
        assert isinstance(result, float)
        assert result >= 0
