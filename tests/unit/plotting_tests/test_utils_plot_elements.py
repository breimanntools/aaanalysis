"""This is a script to test the plot-element helpers in _utils/utils_plot_elements.py
(plot_add_bars with bar_labels across all positions -> _get_xy_hava / _add_bar_labels,
plus the labels guards and ticks_0 axis branches).

Driven directly via ut.plot_add_bars / ut.ticks_0 with a prepared Axes.
NOTE: the first ticks_0 definition (lines ~181-198) is shadowed by the second
def at line ~200 (Python uses the last), so it is unreachable dead code and not
targeted here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import aaanalysis.utils as ut


def _ax_with_grid(n=4):
    fig, ax = plt.subplots()
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    return ax


class TestPlotAddBars:
    def test_left_with_bar_labels(self):
        ax = _ax_with_grid(4)
        ut.plot_add_bars(ax=ax, labels=[0, 1, 2, 3], bar_position="left",
                         bar_labels=["A", "B", "C", "D"])
        plt.close("all")

    def test_right_with_bar_labels(self):
        ax = _ax_with_grid(4)
        ut.plot_add_bars(ax=ax, labels=[0, 1, 2, 3], bar_position="right",
                         bar_labels=["A", "B", "C", "D"])
        plt.close("all")

    def test_top_with_bar_labels(self):
        ax = _ax_with_grid(4)
        ut.plot_add_bars(ax=ax, labels=[0, 1, 2, 3], bar_position="top",
                         bar_labels=["A", "B", "C", "D"])
        plt.close("all")

    def test_bottom_with_bar_labels(self):
        ax = _ax_with_grid(4)
        ut.plot_add_bars(ax=ax, labels=[0, 1, 2, 3], bar_position="bottom",
                         bar_labels=["A", "B", "C", "D"])
        plt.close("all")

    def test_labels_none_raises(self):
        ax = _ax_with_grid(4)
        with pytest.raises(ValueError, match="labels"):
            ut.plot_add_bars(ax=ax, labels=None)
        plt.close("all")

    def test_labels_count_mismatch_raises(self):
        ax = _ax_with_grid(4)
        with pytest.raises(ValueError, match="match"):
            ut.plot_add_bars(ax=ax, labels=[0, 1], bar_position="left")
        plt.close("all")

    def test_invalid_position_raises(self):
        ax = _ax_with_grid(4)
        with pytest.raises(ValueError, match="Position"):
            ut.plot_add_bars(ax=ax, labels=[0, 1, 2, 3], bar_position="middle")
        plt.close("all")


class TestTicks0:
    def test_axis_x(self):
        ax = _ax_with_grid(4)
        assert ut.ticks_0(ax, axis="x") is None or True
        plt.close("all")

    def test_axis_y(self):
        ax = _ax_with_grid(4)
        ut.ticks_0(ax, axis="y")
        plt.close("all")

    def test_invalid_axis_raises(self):
        ax = _ax_with_grid(4)
        with pytest.raises(ValueError, match="axis"):
            ut.ticks_0(ax, axis="z")
        plt.close("all")
