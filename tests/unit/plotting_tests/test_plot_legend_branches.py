"""This is a script to test the marker / hatch / linestyle / marker_size
validation + warning branches of aa.plot_legend (backend _utils/plotting.py)
that the main test_plot_legend.py doesn't exercise, plus plot_gco's error path.

Agg backend; figures closed after each test.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

DICT3 = {"A": "tab:red", "B": "tab:blue", "C": "tab:green"}


def _ax():
    fig, ax = plt.subplots()
    return ax


class TestPlotLegendMarkers:
    def test_real_marker(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o")
        plt.close("all")

    def test_line_marker_with_lw(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1.5)
        plt.close("all")

    def test_marker_list(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker=["o", "-", "s"], lw=1.0)
        plt.close("all")

    def test_default_marker_none(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker=None)
        plt.close("all")

    def test_invalid_marker_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="@@@")
        plt.close("all")

    def test_invalid_marker_list_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker=["o", "@@@", "s"])
        plt.close("all")

    def test_invalid_marker_list_length(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker=["o", "s"])
        plt.close("all")

    def test_line_marker_zero_lw_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=0)
        assert any("lw" in str(x.message) for x in w)
        plt.close("all")


class TestPlotLegendHatch:
    def test_valid_hatch(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch="/")
        plt.close("all")

    def test_valid_hatch_list(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch=["/", "x", "o"])
        plt.close("all")

    def test_invalid_hatch_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch="ZZ")
        plt.close("all")

    def test_invalid_hatch_list_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch=["/", "ZZ", "o"])
        plt.close("all")

    def test_invalid_hatch_list_length(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch=["/", "x"])
        plt.close("all")

    def test_hatch_with_marker_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aa.plot_legend(ax=_ax(), dict_color=DICT3, hatch="/", marker="o")
        assert any("hatch" in str(x.message) for x in w)
        plt.close("all")


class TestPlotLegendLinestyle:
    def test_valid_linestyle(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1, linestyle="--")
        plt.close("all")

    def test_valid_linestyle_list(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1,
                       linestyle=["-", "--", ":"])
        plt.close("all")

    def test_invalid_linestyle_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1, linestyle="bad")
        plt.close("all")

    def test_invalid_linestyle_list_value(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1,
                           linestyle=["-", "bad", ":"])
        plt.close("all")

    def test_invalid_linestyle_list_length(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1,
                           linestyle=["-", "--"])
        plt.close("all")

    def test_linestyle_with_nonline_marker_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o", linestyle="--")
        assert any("linestyle" in str(x.message) for x in w)
        plt.close("all")


class TestPlotLegendMarkerSize:
    def test_marker_size_list(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o",
                       marker_size=[8, 10, 12])
        plt.close("all")

    def test_invalid_marker_size_type(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o", marker_size="big")
        plt.close("all")


class TestPlotLegendListCatLabels:
    def test_list_cat_subset(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, list_cat=["A", "B"])
        plt.close("all")

    def test_invalid_list_cat_missing_key(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, list_cat=["A", "Z"])
        plt.close("all")

    def test_invalid_labels_length(self):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=_ax(), dict_color=DICT3, labels=["only_one"])
        plt.close("all")


class TestPlotLegendPlacement:
    def test_loc_out(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, loc_out=True)
        plt.close("all")

    def test_xy_anchor(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, x=0.5, y=-0.2)
        plt.close("all")


class TestPlotGco:
    def test_valid_option(self):
        assert ut.plot_gco(option="font.size") > 0

    def test_invalid_option_raises(self):
        with pytest.raises(ValueError, match="should be one of"):
            ut.plot_gco(option="__no_such_option__")
