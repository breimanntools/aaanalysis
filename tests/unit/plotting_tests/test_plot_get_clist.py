"""
This is a script for testing the aa.plot_get_clist function.
"""
from hypothesis import given, strategies as st
import pytest
import aaanalysis as aa

class TestPlotGetCList:
    """Test plot_get_clist function"""

    # Positive testing
    @given(n_colors=st.integers(min_value=2, max_value=9))
    def test_valid_n_colors(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors)
        assert len(colors) == n_colors

    def test_return_type(self):
        colors = aa.plot_get_clist(n_colors=7)
        assert all(isinstance(color, str) for color in colors)

    # Negative testing
    @given(n_colors=st.integers(min_value=-1000, max_value=1))
    def test_invalid_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=n_colors)

    @given(n_colors=st.floats())
    def test_non_integer_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=n_colors)
