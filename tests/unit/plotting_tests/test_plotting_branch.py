"""Branch-coverage tests for the public plotting API.

Exercises option / edge arms in the backend plotting helpers
(_utils/plotting.py, plotting/_plot_*.py, show_html/_display_df.py) strictly
through the public ``import aaanalysis as aa`` surface -- never importing a
private helper directly. Each test pushes one option value so both arms of a
branch run across the class.

Agg backend; figures closed after each test.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

DICT3 = {"A": "tab:red", "B": "tab:blue", "C": "tab:green"}


def _ax():
    fig, ax = plt.subplots()
    return ax


# --------------------------------------------------------------------------
# _utils/plotting.py -- _marker_has / _marker_has_no list+string arms,
# _check_list_cat length guard, plot_legend_ ax=None / existing-legend /
# title_align_left arms (lines 47,54,56,58,72,139,228,238,259).
# --------------------------------------------------------------------------
class TestPlotLegendMarkerBranch:
    def test_marker_list_with_line_and_default(self):
        # list arm of _marker_has (find "-") and _marker_has_no (find non-"-")
        aa.plot_legend(ax=_ax(), dict_color=DICT3,
                       marker=["-", "o", None], lw=1.0)
        plt.close("all")

    def test_marker_list_all_lines(self):
        # _marker_has_no list arm: every element == "-" -> any(!= "-") is False
        aa.plot_legend(ax=_ax(), dict_color=DICT3,
                       marker=["-", "-", "-"], lw=1.0)
        plt.close("all")

    def test_marker_string_dash(self):
        # string arm of _marker_has / _marker_has_no (marker == / != val)
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="-", lw=1.0)
        plt.close("all")

    def test_marker_string_non_dash(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o")
        plt.close("all")

    def test_marker_none_default(self):
        # marker is None -> _marker_has*/None arm returns False
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker=None)
        plt.close("all")

    @settings(max_examples=5, deadline=None)
    @given(ms=some.floats(min_value=1, max_value=30))
    def test_marker_size_scalar(self, ms):
        # int/float arm of _check_marker_size (line ~134/139)
        aa.plot_legend(ax=_ax(), dict_color=DICT3, marker="o", marker_size=ms)
        plt.close("all")


class TestPlotLegendListCatBranch:
    def test_list_cat_default_none(self):
        # _check_list_cat: not list_cat -> keys of dict_color
        aa.plot_legend(ax=_ax(), dict_color=DICT3, list_cat=None)
        plt.close("all")

    def test_list_cat_longer_than_dict_raises(self):
        # line 72: len(dict_color) < len(list_cat)
        small = {"A": "tab:red"}
        with pytest.raises(ValueError, match="must contain"):
            aa.plot_legend(ax=_ax(), dict_color=small, list_cat=["A", "A"])
        plt.close("all")


class TestPlotLegendBackendBranch:
    def test_ax_none_uses_gca(self):
        # plotting.py 228 + _plot_legend.py 143: ax is None -> plt.gca()
        plt.figure()
        aa.plot_legend(ax=None, dict_color=DICT3)
        plt.close("all")

    def test_keep_existing_legend(self):
        # 238 + 259/262: an existing legend present, keep_legend=True
        ax = _ax()
        ax.plot([0, 1], [0, 1], label="prior")
        ax.legend()
        aa.plot_legend(ax=ax, dict_color=DICT3, keep_legend=True)
        plt.close("all")

    def test_remove_existing_legend(self):
        # 238: legend present with lines, keep_legend=False -> removed
        ax = _ax()
        ax.plot([0, 1], [0, 1], label="prior")
        ax.legend()
        aa.plot_legend(ax=ax, dict_color=DICT3, keep_legend=False)
        plt.close("all")

    def test_title_align_left_false(self):
        # 259: title_align_left=False skips the align branch
        aa.plot_legend(ax=_ax(), dict_color=DICT3, title="T",
                       title_align_left=False)
        plt.close("all")

    def test_title_align_left_true_with_title(self):
        aa.plot_legend(ax=_ax(), dict_color=DICT3, title="T",
                       title_align_left=True)
        plt.close("all")

    def test_fontsize_title_set(self):
        # title_fontproperties branch with fontsize_title given
        aa.plot_legend(ax=_ax(), dict_color=DICT3, title="T",
                       fontsize_title=12)
        plt.close("all")


# --------------------------------------------------------------------------
# plotting/_plot_gcfs.py line 36 -- invalid option.
# --------------------------------------------------------------------------
class TestPlotGcfsBranch:
    def test_font_size_default(self):
        assert aa.plot_gcfs() > 0

    def test_axes_linewidth_option(self):
        assert aa.plot_gcfs(option="axes.linewidth") >= 0

    def test_invalid_option_raises(self):
        with pytest.raises(ValueError, match="should be one of"):
            aa.plot_gcfs(option="__nope__")


# --------------------------------------------------------------------------
# plotting/_plot_settings.py lines 183/191 -- short_ticks x/y elif arms.
# --------------------------------------------------------------------------
class TestPlotSettingsBranch:
    def test_short_ticks_x(self):
        # 183: short_ticks_x -> short tick size on x
        aa.plot_settings(short_ticks_x=True)
        plt.close("all")

    def test_short_ticks_y(self):
        # 191: short_ticks_y -> short tick size on y
        aa.plot_settings(short_ticks_y=True)
        plt.close("all")

    def test_short_ticks_both(self):
        aa.plot_settings(short_ticks=True)
        plt.close("all")

    def test_no_ticks_default_arms(self):
        # else arms (default tick sizes) on both axes
        aa.plot_settings(no_ticks=False, short_ticks=False)
        plt.close("all")

    def test_no_ticks_x_only(self):
        aa.plot_settings(no_ticks_x=True)
        plt.close("all")


# --------------------------------------------------------------------------
# plotting/_plot_rank.py line 153 -- empty subgroup (continue) + group_order.
# --------------------------------------------------------------------------
def _df_rank():
    return pd.DataFrame({"score": [3.0, 1.0, 2.0, 0.5],
                         "group": ["x", "x", "y", "y"]})


class TestPlotRankBranch:
    def test_group_order_with_absent_group(self):
        # 153: a group in group_order with no rows -> continue arm
        df = _df_rank()
        fig, ax = aa.plot_rank(df_rank=df, group_order=["x", "y", "z"])
        plt.close("all")

    def test_group_order_default(self):
        df = _df_rank()
        aa.plot_rank(df_rank=df)
        plt.close("all")

    def test_group_order_missing_raises(self):
        df = _df_rank()
        with pytest.raises(ValueError, match="missing groups"):
            aa.plot_rank(df_rank=df, group_order=["x"])
        plt.close("all")

    def test_threshold_list(self):
        df = _df_rank()
        aa.plot_rank(df_rank=df, threshold=[1.0, 2.0])
        plt.close("all")

    def test_ax_provided(self):
        df = _df_rank()
        ax = _ax()
        aa.plot_rank(df_rank=df, ax=ax)
        plt.close("all")


# --------------------------------------------------------------------------
# show_html/_display_df.py lines 24,50,52,60,62,146,151 -- _adjust_df char
# arms, _select_row/_select_col int+str arms, n_rows/n_cols pos+neg arms,
# show_shape.
# --------------------------------------------------------------------------
def _df():
    return pd.DataFrame({"a": ["x" * 80, "y", "z"], "b": [1, 2, 3]})


class TestDisplayDfBranch:
    def test_char_limit_none_skips_truncation(self):
        # 24: char_limit is None -> truncation loop skipped
        aa.display_df(df=_df(), char_limit=None)

    def test_char_limit_truncates_long(self):
        # _adjust_df truncate_string arm for a >char_limit string
        aa.display_df(df=_df(), char_limit=10)

    def test_row_to_show_int(self):
        # 50: _select_row int arm
        aa.display_df(df=_df(), row_to_show=0)

    def test_row_to_show_str(self):
        # 52: _select_row str arm (index label)
        df = _df()
        df.index = ["r0", "r1", "r2"]
        aa.display_df(df=df, row_to_show="r0")

    def test_col_to_show_int(self):
        # 60: _select_col int arm
        aa.display_df(df=_df(), col_to_show=0)

    def test_col_to_show_str(self):
        # 62: _select_col str arm
        aa.display_df(df=_df(), col_to_show="a")

    def test_n_rows_positive(self):
        # 146: n_rows > 0 -> head
        aa.display_df(df=_df(), n_rows=2)

    def test_n_rows_negative(self):
        # 146 else: n_rows < 0 -> tail
        aa.display_df(df=_df(), n_rows=-2)

    def test_n_cols_positive(self):
        # 151: n_cols > 0 -> T.head().T
        aa.display_df(df=_df(), n_cols=1)

    def test_n_cols_negative(self):
        # 151 else: n_cols < 0 -> T.tail().T
        aa.display_df(df=_df(), n_cols=-1)

    def test_show_shape(self):
        # show_shape=True print arm
        aa.display_df(df=_df(), show_shape=True)

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=3))
    def test_n_rows_property(self, n):
        aa.display_df(df=_df(), n_rows=n)

    def test_invalid_row_to_show_type_raises(self):
        with pytest.raises(ValueError, match="should be int"):
            aa.display_df(df=_df(), row_to_show=1.5)

    def test_invalid_row_to_show_name_raises(self):
        with pytest.raises(ValueError, match="should be one of"):
            aa.display_df(df=_df(), row_to_show="__nope__")
