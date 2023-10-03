"""
This is a script for testing the aa.plot_get_cdict function.
"""
from hypothesis import given, example
import hypothesis.strategies as some

import aaanalysis.plotting.plot_get_cdict_
import aaanalysis.utils as ut
import pytest
import aaanalysis as aa

class TestPlotGetCDict:
    """Test plot_get_cdict function"""

    # Positive test cases
    @given(name=some.sampled_from(["DICT_COLOR", "DICT_CAT"]))
    def test_valid_dict_names(self, name):
        """Test function with valid dictionary names."""
        color_dict = aaanalysis.plotting.plot_get_cdict_.plot_get_cdict(name=name)
        if name == "DICT_COLOR":
            assert color_dict == ut.DICT_COLOR
        else:
            assert color_dict == ut.DICT_COLOR_CAT

    # Negative test cases
    @given(invalid_name=some.text().filter(lambda x: x not in ["DICT_COLOR", "DICT_CAT"]))
    @example(invalid_name="INVALID_NAME")
    def test_invalid_dict_names(self, invalid_name):
        """Test function with invalid dictionary names."""
        with pytest.raises(ValueError):
            aaanalysis.plotting.plot_get_cdict_.plot_get_cdict(name=invalid_name)

    @given(invalid_type=some.one_of(some.integers(), some.floats(), some.lists(some.text())))
    def test_invalid_types(self, invalid_type):
        """Test function with unexpected types as dictionary name."""
        with pytest.raises(ValueError):
            aaanalysis.plotting.plot_get_cdict_.plot_get_cdict(name=invalid_type)

    @given(empty_str=some.just(""))
    def test_empty_string_as_name(self, empty_str):
        """Test function with an empty string as dictionary name."""
        with pytest.raises(ValueError):
            aaanalysis.plotting.plot_get_cdict_.plot_get_cdict(name=empty_str)
