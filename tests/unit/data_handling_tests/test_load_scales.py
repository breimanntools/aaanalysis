"""
This is a script for testing the aa.load_scales function.
"""
from hypothesis import given, settings, example
import hypothesis.strategies as some
import aaanalysis as aa
from pandas import DataFrame
import pytest

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


class TestLoadScales:
    """Test load_scales function"""

    # Basic positive tests
    def test_load_scales_default(self):
        """Test the default parameters."""
        df = aa.load_scales()
        assert isinstance(df, DataFrame)

    def test_load_scales_names(self):
        """Test different dataset names."""
        for name in ["scales", "scales_raw", "scales_cat", "scales_pc", "top60", "top60_eval"]:
            df = aa.load_scales(name=name)
            assert isinstance(df, DataFrame)

    def test_load_scales_just_aaindex(self):
        """Test the 'just_aaindex' parameter."""
        df = aa.load_scales(just_aaindex=True)
        assert isinstance(df, DataFrame)

    def test_load_scales_unclassified_in(self):
        """Test the 'unclassified_in' parameter."""
        df = aa.load_scales(unclassified_out=True)
        assert isinstance(df, DataFrame)

    @settings(max_examples=10, deadline=1000)
    @given(top60_n=some.integers(min_value=1, max_value=60))
    def test_load_scales_top60_n(self, top60_n):
        """Test the 'top60_n' parameter."""
        df = aa.load_scales(name="scales", top60_n=top60_n)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_name(self):
        """Test with an invalid dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name="invalid_name")

    def test_invalid_name_type(self):
        """Test with a non-string dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name=123)

    def test_empty_name(self):
        """Test with an empty string as dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name="")


class TestLoadScalesComplex:
    """Test load_scales function with complex scenarios"""

    # Positive tests
    def test_load_scales_both_filters(self):
        """Test both 'just_aaindex' and 'unclassified_in' together."""
        df = aa.load_scales(just_aaindex=True, unclassified_out=True)
        assert isinstance(df, DataFrame)

    def test_load_scales_all_params(self):
        """Test all parameters together."""
        df = aa.load_scales(name="scales", just_aaindex=True, unclassified_out=True, top60_n=10)
        assert isinstance(df, DataFrame)

    def test_load_scales_name_and_filters(self):
        """Test 'name' with 'just_aaindex' and 'unclassified_in' together."""
        df = aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_combination(self):
        """Test with all invalid parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False, top60_n=-5)

    def test_invalid_complex_scenario_1(self):
        """Test a complex combination of parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="hydrophobicity", just_aaindex="yes", top60_n="60")


class TestLoadScalesVeryComplex:
    """Test load_scales function with very complex scenarios"""

    # Positive tests
    def test_load_scales_all_filters_with_top60(self):
        """Test all filters ('just_aaindex', 'unclassified_in', 'top60_n')."""
        df = aa.load_scales(just_aaindex=True, unclassified_out=False, top60_n=5)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_all_filters(self):
        """Test with all invalid filters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False, top60_n=100)

    def test_invalid_very_complex_scenario_1(self):
        """Test a very complex scenario with extreme values."""
        with pytest.raises(ValueError):
            aa.load_scales(name="hydrophobicity" * 1000, just_aaindex=True)

    def test_invalid_very_complex_scenario_2(self):
        """Test a very complex scenario with conflicting parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="some_name", top60_n=-100, unclassified_out="yes")

    def test_invalid_very_complex_scenario_3(self):
        """Test a very complex scenario with both invalid and out-of-bounds values."""
        with pytest.raises(ValueError):
            aa.load_scales(name="some_invalid_name", just_aaindex="not a boolean", top60_n=2000)
