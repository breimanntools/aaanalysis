"""
This is a script for testing the plot_set_legend function.
"""
from hypothesis import given, example
from hypothesis import strategies as st
import hypothesis.strategies as some
import aaanalysis as aa
import matplotlib.pyplot as plt
import pytest


class TestPlotSetLegend:
    """Test plot_set_legend function"""

    # Test if function returns plt.Axes when return_handles=False
    def test_return_type_without_handles(self):
        fig, ax = plt.subplots()
        ax_returned = aa.plot_set_legend(ax=ax)
        assert isinstance(ax_returned, plt.Axes)

    # Test if function returns handles and labels when return_handles=True
    def test_return_type_with_handles(self):
        fig, ax = plt.subplots()
        handles, labels = aa.plot_set_legend(ax=ax, return_handles=True)
        assert isinstance(handles, list)
        assert isinstance(labels, list)

    # Test if a legend is created in the ax
    def test_legend_created(self):
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"})
        assert ax.get_legend() is not None

    # Test if the legend gets removed when remove_legend=True
    def test_remove_legend(self):
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"})
        assert ax.get_legend() is not None
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category B": "blue"}, remove_legend=True)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "Category A" not in labels

    @given(loc_out=some.booleans())
    def test_location_outside(self, loc_out):
        """Test the 'loc_out' parameter."""
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, loc_out=loc_out)
        if loc_out:
            assert ax.get_legend()._loc == (0, -0.25)
        else:
            assert ax.get_legend()._loc != (0, -0.25)

    # Property-based testing for negative cases
    @given(lw=some.floats(min_value=-5, max_value=0))
    def test_invalid_line_width(self, lw):
        """Test with an invalid 'lw' value."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, lw=lw)

    @given(marker_size=some.floats(min_value=-5, max_value=0))
    def test_invalid_marker_size(self, marker_size):
        """Test with an invalid 'marker_size' value."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, marker_size=marker_size)

    @given(marker=some.text())
    @example(marker="invalid_marker")
    def test_invalid_marker(self, marker):
        """Test with an invalid 'marker' value."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, marker=marker)

    @given(linestyle=some.text())
    @example(linestyle="invalid_linestyle")
    def test_invalid_linestyle(self, linestyle):
        """Test with an invalid 'linestyle' value."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, marker_linestyle=linestyle)

    @given(st.floats(1, 10))
    def test_plot_set_legend_ncol(self, ncol):
        """Test the 'ncol' parameter."""
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, ncol=int(ncol))
        assert isinstance(result, plt.Axes)

    @given(st.floats(0, 5))
    def test_plot_set_legend_labelspacing(self, labelspacing):
        """Test the 'labelspacing' parameter."""
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, labelspacing=labelspacing)
        assert isinstance(result, plt.Axes)

    @given(st.floats(0, 5))
    def test_plot_set_legend_columnspacing(self, columnspacing):
        """Test the 'columnspacing' parameter."""
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, columnspacing=columnspacing)
        assert isinstance(result, plt.Axes)

    @given(st.floats(0, 5))
    def test_plot_set_legend_handletextpad(self, handletextpad):
        """Test the 'handletextpad' parameter."""
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, handletextpad=handletextpad)
        assert isinstance(result, plt.Axes)

    @given(some.booleans())
    def test_plot_set_legend_remove_legend(self, remove_legend):
        """Test the 'remove_legend' parameter."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        ax.legend()
        aa.plot_set_legend(ax=ax, remove_legend=remove_legend)
        if remove_legend:
            assert ax.get_legend() is None
        else:
            assert ax.get_legend() is not None

    @given(some.booleans())
    def test_plot_set_legend_return_handles(self, return_handles):
        """Test the 'return_handles' parameter."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        result = aa.plot_set_legend(ax=ax, return_handles=return_handles)
        if return_handles:
            assert isinstance(result, tuple)
        else:
            assert isinstance(result, plt.Axes)

    @given(some.booleans())
    def test_plot_set_legend_title_align_left(self, title_align_left):
        """Test the 'title_align_left' parameter."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        legend = aa.plot_set_legend(ax=ax, title="Legend Title", title_align_left=title_align_left)
        if title_align_left:
            assert legend._legend_box.align == "left"
        else:
            assert legend._legend_box.align != "left"

    def test_legend_positioning(self):
        """Test the positioning with x, y, and loc_out."""
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, loc_out=True, x=0, y=0)
        assert isinstance(result, plt.Axes)

    def test_invalid_combination(self):
        """Test with invalid combinations of parameters."""
        with pytest.raises(ValueError):
            ax = plt.gca()
            aa.plot_set_legend(ax=ax, loc_out=True, x=None, y=None)

    def test_plot_set_legend_loc_outside(self):
        """Test 'loc_out' parameter."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        aa.plot_set_legend(ax=ax, loc_out=True)
        assert ax.get_legend().get_bbox_to_anchor().y0 <= 0

    def test_plot_set_legend_invalid_fontsize(self):
        """Test with 'fontsize' less than 0."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, fontsize=-5)

    def test_plot_set_legend_invalid_marker_size(self):
        """Test with negative 'marker_size'."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, marker_size=-10)

    def test_plot_set_legend_color_and_category(self):
        """Test 'dict_color' and 'list_cat' parameters together."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        color_dict = {"Sample Line": "red"}
        categories = ["Sample Line"]
        aa.plot_set_legend(ax=ax, dict_color=color_dict, list_cat=categories)
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(categories) == set(legend_texts)

    def test_plot_set_legend_invalid_color_and_category(self):
        """Test with invalid 'dict_color' and 'list_cat'."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        color_dict = {"Wrong Label": "red"}
        categories = ["Sample Line"]
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color=color_dict, list_cat=categories)


# II. Complex Cases Test Class
class TestPlotSetLegendComplex:
    """Test plot_set_legend function with complex scenarios."""

    @given(st.floats(0, 10))
    def test_plot_set_legend_ncol(self, ncol):
        ax = plt.gca()
        result = aa.plot_set_legend(ax=ax, ncol=int(ncol))
        assert isinstance(result, plt.Axes)

    @given(st.integers(0, 10), st.integers(0, 10))
    def test_plot_set_legend_order(self, idx1, idx2):
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category A": "red", "Category B": "blue"})
        handles, labels = ax.get_legend_handles_labels()
        ax = aa.plot_set_legend(ax=ax, order=[idx1 % 2, idx2 % 2])
        new_handles, new_labels = ax.get_legend_handles_labels()
        assert handles[idx1 % 2] == new_handles[0]
        assert handles[idx2 % 2] == new_handles[1]

    @given(st.lists(st.text(), min_size=1, max_size=5))
    def test_plot_set_legend_custom_labels(self, labels):
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={label: "red" for label in labels})
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(legend_labels) == set(labels)

    @given(st.floats(0, 1))
    def test_plot_set_legend_alpha(self, alpha):
        fig, ax = plt.subplots()
        ax = aa.plot_set_legend(ax=ax, dict_color={"Category A": "red"}, alpha=alpha)
        legend = ax.get_legend()
        for handle in legend.legendHandles:
            assert handle.get_alpha() == alpha


    def test_handles_generation(self):
        """Test handles based on dict_color and list_cat."""
        ax = plt.gca()
        dict_color = {"Category 1": "red", "Category 2": "blue"}
        list_cat = ["Category 1", "Category 2"]
        result = aa.plot_set_legend(ax=ax, dict_color=dict_color, list_cat=list_cat)
        assert isinstance(result, plt.Axes)

    def test_remove_existing_legend(self):
        """Test removing of existing legend."""
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], label="Test")
        plt.legend()
        aa.plot_set_legend(ax=ax, remove_legend=True)
        assert ax.get_legend() is None

    def test_return_handles_and_labels(self):
        """Test return_handles functionality."""
        ax = plt.gca()
        handles, labels = aa.plot_set_legend(ax=ax, return_handles=True)
        assert isinstance(handles, list)
        assert isinstance(labels, list)
