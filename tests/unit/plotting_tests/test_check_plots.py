"""This is a script to test the plot-checking validators in _utils/check_plots.py
(check_fig / check_ax / check_figsize / check_lim / check_dict_xlims / check_color /
check_list_colors / check_dict_color / check_cmap / check_palette / check_vmin_vmax).

These are pure validators exposed via ``ut``; tested directly to reach their
error / accept-none / list branches. Doubles as user-facing error-message coverage.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import aaanalysis.utils as ut


class TestCheckFig:
    def test_valid_fig(self):
        fig = plt.figure()
        assert ut.check_fig(fig=fig) is None
        plt.close("all")

    def test_none_accepted(self):
        assert ut.check_fig(fig=None, accept_none=True) is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Figure"):
            ut.check_fig(fig="not a fig")


class TestCheckAx:
    def test_valid_ax(self):
        fig, ax = plt.subplots()
        assert ut.check_ax(ax=ax) is ax
        plt.close("all")

    def test_none_accepted(self):
        assert ut.check_ax(ax=None, accept_none=True) is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Axes"):
            ut.check_ax(ax=123)


class TestCheckFigsize:
    def test_valid(self):
        assert ut.check_figsize(figsize=(6, 4)) is None

    def test_none_accepted(self):
        assert ut.check_figsize(figsize=None, accept_none=True) is None

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            ut.check_figsize(figsize=(0, 4))


class TestCheckLim:
    def test_valid(self):
        assert ut.check_lim(name="xlim", val=(0, 1)) is None

    def test_none_not_accepted_raises(self):
        with pytest.raises(ValueError, match="xlim"):
            ut.check_lim(name="xlim", val=None, accept_none=False)

    def test_min_ge_max_raises(self):
        with pytest.raises(ValueError):
            ut.check_lim(name="xlim", val=(2, 1))


class TestCheckDictXlims:
    def test_n_ax_none_raises(self):
        with pytest.raises(ValueError, match="n_ax"):
            ut.check_dict_xlims(dict_xlims={0: (0, 1)}, n_ax=None)

    def test_none_dict_ok(self):
        assert ut.check_dict_xlims(dict_xlims=None, n_ax=4) is None

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="invalid keys"):
            ut.check_dict_xlims(dict_xlims={9: (0, 1)}, n_ax=4)

    def test_valid(self):
        assert ut.check_dict_xlims(dict_xlims={0: (0, 1)}, n_ax=4) is None


class TestCheckColors:
    def test_valid_named(self):
        assert ut.check_color(name="c", val="tab:blue") is None

    def test_valid_hex(self):
        assert ut.check_color(name="c", val="#1A2B3C") is None

    def test_invalid_color(self):
        with pytest.raises(ValueError, match="valid color"):
            ut.check_color(name="c", val="not_a_color")

    def test_list_colors_valid(self):
        assert ut.check_list_colors(name="c", val=["red", "blue"]) is None

    def test_list_colors_max_n(self):
        with pytest.raises(ValueError, match="no more than"):
            ut.check_list_colors(name="c", val=["red", "blue", "green"], max_n=2)

    def test_list_colors_min_n(self):
        with pytest.raises(ValueError, match="at least"):
            ut.check_list_colors(name="c", val=["red"], min_n=2)

    def test_dict_color_valid(self):
        assert ut.check_dict_color(val={"a": "red"}) is None

    def test_dict_color_min_n(self):
        with pytest.raises(ValueError, match="at least"):
            ut.check_dict_color(val={"a": "red"}, min_n=2)

    def test_dict_color_max_n(self):
        with pytest.raises(ValueError, match="no more than"):
            ut.check_dict_color(val={"a": "red", "b": "blue"}, max_n=1)


class TestCheckCmapPalette:
    def test_cmap_valid(self):
        assert ut.check_cmap(name="cmap", val="viridis") is None

    def test_cmap_invalid(self):
        with pytest.raises(ValueError, match="valid cmap"):
            ut.check_cmap(name="cmap", val="__no_cmap__")

    def test_palette_str(self):
        assert ut.check_palette(name="p", val="viridis") is None

    def test_palette_list(self):
        assert ut.check_palette(name="p", val=["red", "blue"]) is None


class TestCheckVminVmax:
    def test_valid(self):
        assert ut.check_vmin_vmax(vmin=-1, vmax=1) is None

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            ut.check_vmin_vmax(vmin=1, vmax=-1)
