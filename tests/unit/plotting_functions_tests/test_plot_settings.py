"""
This is a script for testing the aa.plot_settings function.
"""
import pytest
from hypothesis import given, strategies as st, example, settings
import matplotlib as mpl
import aaanalysis as aa


class TestPlotSettings:

    # Positive Tests
    def test_default_params(self):
        aa.plot_settings()
        assert mpl.rcParams["font.family"] == ["sans-serif"]

    def test_font(self):
        aa.plot_settings(font="Verdana")
        assert "Verdana" in mpl.rcParams["font.sans-serif"]

    def test_weight_bold(self):
        aa.plot_settings(weight_bold=True)
        assert mpl.rcParams["axes.labelweight"] == "bold"

    def test_adjust_only_font(self):
        aa.plot_settings(adjust_only_font=True)
        assert mpl.rcParams["font.family"] == ["sans-serif"]

    def test_grid_axis_y(self):
        aa.plot_settings(grid_axis="y")
        assert mpl.rcParams["axes.grid.axis"] == "y"

    def test_no_ticks(self):
        aa.plot_settings(no_ticks=True)
        assert mpl.rcParams["xtick.major.size"] == 0
        assert mpl.rcParams["ytick.major.size"] == 0

    def test_grid_true(self):
        aa.plot_settings(grid=True)
        assert mpl.rcParams["axes.grid"] == True

    def test_adjust_further_elements(self):
        aa.plot_settings(adjust_further_elements=True)
        assert mpl.rcParams["errorbar.capsize"] == 10
        assert mpl.rcParams["legend.frameon"] == False

    # Negative Tests
    def test_negative_font_scale(self):
        with pytest.raises(ValueError):
            aa.plot_settings(font_scale=-1.0)

    def test_invalid_font(self):
        with pytest.raises(ValueError):
            aa.plot_settings(font="InvalidFont")

    def test_invalid_grid_axis(self):
        with pytest.raises(ValueError):
            aa.plot_settings(grid_axis="z")

    def test_both_no_ticks_and_short_ticks(self):
        with pytest.warns(UserWarning):
            aa.plot_settings(no_ticks=True, short_ticks=True)

    def test_both_no_ticks_x_and_short_ticks_x(self):
        with pytest.warns(UserWarning):
            aa.plot_settings(no_ticks_x=True, short_ticks_x=True)

    def test_both_no_ticks_y_and_short_ticks_y(self):
        with pytest.warns(UserWarning):
            aa.plot_settings(no_ticks_y=True, short_ticks_y=True)

class TestPlotSettingsComplexCases:

    @given(st.floats(min_value=0, allow_nan=False, allow_infinity=False),
           st.sampled_from(["Arial", "Verdana", "Helvetica", "DejaVu Sans"]),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.sampled_from(["x", "y", "both"]))
    @example(1.5, "Arial", True, False, False, True, False, "y")
    @example(1.0, "Verdana", False, True, True, False, True, "x")
    @settings(max_examples=5)
    def test_complex_positive_cases(self, font_scale, font, weight_bold, adjust_only_font, adjust_further_elements, grid, no_ticks, grid_axis):
        aa.plot_settings(font_scale=font_scale, font=font, weight_bold=weight_bold, adjust_only_font=adjust_only_font, adjust_further_elements=adjust_further_elements, grid=grid, no_ticks=no_ticks, grid_axis=grid_axis)

    @given(st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
           st.text(),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.booleans(),
           st.text())
    @example(-1.0, "InvalidFont", True, False, False, True, False, "z")
    @settings(max_examples=5)
    def test_complex_negative_cases(self, font_scale, font, weight_bold, adjust_only_font, adjust_further_elements, grid, no_ticks, grid_axis):
        with pytest.raises(Exception):
            aa.plot_settings(font_scale=font_scale, font=font, weight_bold=weight_bold, adjust_only_font=adjust_only_font, adjust_further_elements=adjust_further_elements, grid=grid, no_ticks=no_ticks, grid_axis=grid_axis)

