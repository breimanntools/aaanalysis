"""
This is a script for testing the aa.plot_get_cmap function.
"""
from hypothesis import given, strategies as st
from typing import List, Tuple, Union
import pytest
import aaanalysis as aa

STR_CMAP_CPP = "CPP"
STR_CMAP_SHAP = "SHAP"
STR_CMAP_TAB = "TAB"


class TestPlotGetCMap:
    """Test plot_get_cmap function"""

    # Positive testing
    @given(n_colors=st.integers(min_value=3, max_value=1000))
    def test_valid_n_colors_cpp(self, n_colors):
        colors = aa.plot_get_cmap(name=STR_CMAP_CPP, n_colors=n_colors)
        assert len(colors) == n_colors

    @given(n_colors=st.integers(min_value=3, max_value=1000))
    def test_valid_n_colors_shap(self, n_colors):
        colors = aa.plot_get_cmap(name=STR_CMAP_SHAP, n_colors=n_colors)
        assert len(colors) == n_colors

    @given(n_colors=st.integers(min_value=2, max_value=9))
    def test_valid_n_colors_tab(self, n_colors):
        colors = aa.plot_get_cmap(name=STR_CMAP_TAB, n_colors=n_colors)
        assert len(colors) == n_colors

    @given(name=st.sampled_from([STR_CMAP_CPP, STR_CMAP_SHAP, STR_CMAP_TAB]))
    def test_valid_name(self, name):
        colors = aa.plot_get_cmap(name=name, n_colors=5)
        assert colors is not None

    def test_facecolor_dark_toggle(self):
        light_colors = aa.plot_get_cmap(name=STR_CMAP_CPP, facecolor_dark=False)
        dark_colors = aa.plot_get_cmap(name=STR_CMAP_CPP, facecolor_dark=True)
        assert light_colors != dark_colors

    def test_return_type_cpp(self):
        colors = aa.plot_get_cmap(name=STR_CMAP_CPP)
        assert all(isinstance(color, tuple) and len(color) == 3 for color in colors)

    def test_return_type_tab(self):
        colors = aa.plot_get_cmap(name=STR_CMAP_TAB, n_colors=7)
        assert all(isinstance(color, str) for color in colors)

    # Negative testing
    @given(name=st.text())
    def test_invalid_name(self, name):
        if name not in [STR_CMAP_CPP, STR_CMAP_SHAP, STR_CMAP_TAB]:
            with pytest.raises(ValueError):
                aa.plot_get_cmap(name=name)

    @given(n_colors=st.integers(min_value=-1000, max_value=1))
    def test_invalid_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(n_colors=n_colors)


    @given(n_colors=st.integers(min_value=-1000, max_value=1))
    def test_invalid_n_colors_for_cpp(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=STR_CMAP_CPP, n_colors=n_colors)

    @given(n_colors=st.one_of(st.integers(min_value=-1000, max_value=1), st.integers(min_value=10, max_value=1000)))
    def test_invalid_n_colors_for_tab(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=STR_CMAP_TAB, n_colors=n_colors)

    @given(n_colors=st.floats())
    def test_non_integer_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(n_colors=n_colors)

    @given(facecolor_dark=st.one_of(st.text(), st.integers(), st.floats(), st.lists(st.booleans())))
    def test_invalid_facecolor_dark_type(self, facecolor_dark):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=STR_CMAP_CPP, facecolor_dark=facecolor_dark)

    @given(n_colors=st.integers(min_value=-1000, max_value=2))
    def test_invalid_n_colors_for_shap(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=STR_CMAP_SHAP, n_colors=n_colors)

    @given(name=st.text(), n_colors=st.integers(min_value=-1000, max_value=1))
    def test_invalid_name_and_n_colors(self, name, n_colors):
        if name not in [STR_CMAP_CPP, STR_CMAP_SHAP, STR_CMAP_TAB]:
            with pytest.raises(ValueError):
                aa.plot_get_cmap(name=name, n_colors=n_colors)

    def test_name_as_none(self):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=None)

    @given(name=st.sampled_from([STR_CMAP_CPP, STR_CMAP_SHAP]))
    def test_zero_n_colors(self, name):
        with pytest.raises(ValueError):
            aa.plot_get_cmap(name=name, n_colors=0)


