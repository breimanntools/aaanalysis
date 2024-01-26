"""
This is a script for testing the plot_set_legend function.
"""
from hypothesis import given, example, settings
from hypothesis import strategies as st
import hypothesis.strategies as some
import aaanalysis as aa
import matplotlib.pyplot as plt
import pytest
import matplotlib.lines as mlines

plt.rcParams['figure.max_open_warning'] = -1

@pytest.fixture(scope="module")
def dict_color():
    return dict(zip(["Class A", "Class B", "Class C"], ["r", "b", "g"]))


class TestPlotSetLegend:
    """Test plot_set_legend function"""

    @pytest.fixture(autouse=True)
    def create_fig_and_ax(self):
        self.fig, self.ax = plt.subplots()

    @pytest.mark.parametrize("dict_color", [
        dict(zip(["Class A", "Class B", "Class C"], ["r", "b", "g"])),
        {"Category A": "red"}
    ])
    def test_return_type_without_handles(self, dict_color):
        ax_returned = aa.plot_legend(ax=self.ax, dict_color=dict_color)
        assert isinstance(ax_returned, plt.Axes)


    def test_legend_created(self, dict_color):
        ax = aa.plot_legend(ax=self.ax, dict_color=dict_color)
        assert ax.get_legend() is not None


    def test_invalid_line_width(self, dict_color):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=self.ax, dict_color=dict_color, lw="not right")

    @settings(max_examples=7, deadline=500)
    @given(marker_size=st.floats(min_value=-5, max_value=-1))
    def test_invalid_marker_size(self, marker_size, dict_color):
        with pytest.raises(ValueError):
            aa.plot_legend(ax=self.ax, dict_color=dict_color, marker_size=marker_size)

    @settings(max_examples=5, deadline=500)
    @given(marker=st.text(min_size=1, max_size=5))
    @example(marker="invalid_marker")
    def test_invalid_marker(self, marker, dict_color):
        valid_markers = [None, "-"] + list(mlines.Line2D.markers.keys())
        if marker not in valid_markers:
            with pytest.raises(ValueError):
                aa.plot_legend(ax=self.ax, dict_color=dict_color, marker=marker)

    @settings(max_examples=5, deadline=500)
    @given(n_cols=st.integers(min_value=1, max_value=10))
    def test_plot_set_legend_n_cols(self, n_cols, dict_color):
        """Test the 'n_cols' parameter."""
        ax = plt.gca()
        result = aa.plot_legend(ax=ax, n_cols=n_cols, dict_color=dict_color)
        assert isinstance(result, plt.Axes)


    @settings(max_examples=5, deadline=500)
    @given(labelspacing=st.floats(0, 5))
    def test_plot_set_legend_labelspacing(self, labelspacing, dict_color):
        """Test the 'labelspacing' parameter."""
        ax = plt.gca()
        result = aa.plot_legend(ax=ax, labelspacing=labelspacing, dict_color=dict_color)
        assert isinstance(result, plt.Axes)


    @settings(max_examples=5, deadline=500)
    @given(columnspacing=st.floats(0, 5))
    def test_plot_set_legend_columnspacing(self, columnspacing, dict_color):
        """Test the 'columnspacing' parameter."""
        ax = plt.gca()
        result = aa.plot_legend(ax=ax, columnspacing=columnspacing, dict_color=dict_color)
        assert isinstance(result, plt.Axes)


    @settings(max_examples=5, deadline=500)
    @given(handletextpad=st.floats(0, 5))
    def test_plot_set_legend_handletextpad(self, handletextpad, dict_color):
        """Test the 'handletextpad' parameter."""
        ax = plt.gca()
        result = aa.plot_legend(ax=ax, handletextpad=handletextpad, dict_color=dict_color)
        assert isinstance(result, plt.Axes)


    def test_legend_positioning(self, dict_color):
        """Test the positioning with x, y, and loc_out."""
        ax = plt.gca()
        result = aa.plot_legend(ax=ax, loc_out=True, x=0, y=0, dict_color=dict_color)
        assert isinstance(result, plt.Axes)


    def test_plot_set_legend_loc_outside(self, dict_color):
        """Test 'loc_out' parameter."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        aa.plot_legend(ax=ax, loc_out=True, dict_color=dict_color)
        assert ax.get_legend().get_bbox_to_anchor().y0 <= 0


    def test_plot_set_legend_invalid_fontsize(self, dict_color):
        """Test with 'fontsize' less than 0."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        with pytest.raises(ValueError):
            aa.plot_legend(ax=ax, dict_color=dict_color, fontsize=-5)


    def test_plot_set_legend_invalid_marker_size(self, dict_color):
        """Test with negative 'marker_size'."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        with pytest.raises(ValueError):
            aa.plot_legend(ax=ax, dict_color=dict_color, marker_size=-10)


    def test_plot_set_legend_color_and_category(self, dict_color):
        """Test 'dict_color' and 'list_cat' parameters together."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        color_dict = {"Sample Line": "red"}
        categories = ["Sample Line"]
        aa.plot_legend(ax=ax, dict_color=color_dict, list_cat=categories)
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(categories) == set(legend_texts)


    def test_plot_set_legend_invalid_color_and_category(self, dict_color):
        """Test with invalid 'dict_color' and 'list_cat'."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 2], label="Sample Line")
        color_dict = {"Wrong Label": "red"}
        categories = ["Sample Line"]
        with pytest.raises(ValueError):
            aa.plot_legend(ax=ax, dict_color=color_dict, list_cat=categories)


    # Negative Test: check that if dict_color is not provided, it should raise an error
    def test_missing_dict_color(self):
        fig, ax = plt.subplots()
        ax.plot(range(5))
        with pytest.raises(ValueError):
            aa.plot_legend(ax=ax, list_cat=['A'])


    # Negative Test: check that if list_cat items are not in dict_color, it should raise an error
    @settings(max_examples=5, deadline=500)
    @given(random_cat=some.text())
    def test_invalid_list_cat(self, random_cat):
        fig, ax = plt.subplots()
        ax.plot(range(5))
        with pytest.raises(ValueError):
            aa.plot_legend(ax=ax, dict_color={'A': 'red'}, list_cat=[random_cat])



# II. Complex Cases Test Class
class TestPlotSetLegendComplex:
    """Test plot_set_legend function with complex scenarios."""

    @settings(max_examples=5, deadline=500)
    @given(st.floats(1, 10))
    def test_plot_set_legend_n_cols(self, n_cols):
        ax = plt.gca()
        dict_color = {str(i): "r" for i in range(0, 10)}
        result = aa.plot_legend(ax=ax, n_cols=int(n_cols), dict_color=dict_color)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=5, deadline=500)
    @given(st.lists(st.text(), min_size=1, max_size=5))
    def test_plot_set_legend_custom_labels(self, labels):
        fig, ax = plt.subplots()
        ax = aa.plot_legend(ax=ax, dict_color={label: "red" for label in labels})
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert set(legend_labels) == set(labels)


    def test_handles_generation(self, dict_color):
        """Test handles based on dict_color and list_cat."""
        ax = plt.gca()
        dict_color = {"Category 1": "red", "Category 2": "blue"}
        list_cat = ["Category 1", "Category 2"]
        result = aa.plot_legend(ax=ax, dict_color=dict_color, list_cat=list_cat)
        assert isinstance(result, plt.Axes)



