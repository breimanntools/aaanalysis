"""
This is a script for testing the plot_set_legend function.
"""
from hypothesis import given, example
import hypothesis.strategies as some
import aaanalysis.utils as ut
import aaanalysis as aa
import pytest
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional


class TestPlotSetLegend:
    """Test plot_set_legend function"""

    # Initialize a test axis to work with
    @pytest.fixture
    def test_axis(self):
        fig, ax = plt.subplots()
        return ax

    # Property-based testing for positive cases
    @given(dict_color=some.dictionaries(keys=some.text(min_size=1, max_size=10), values=some.sampled_from(['red', 'blue', 'green'])),
           list_cat=some.lists(some.text(min_size=1, max_size=10), min_size=1, max_size=5))
    def test_legend_creation(self, test_axis, dict_color, list_cat):
        """Test the basic legend creation."""
        legend_axis = aa.plot_set_legend(ax=test_axis, dict_color=dict_color, list_cat=list_cat)
        assert legend_axis.get_legend() is not None

    @given(loc=some.sampled_from(['upper left', 'upper right', 'lower left', 'lower right']),
           title=some.text())
    def test_legend_position_and_title(self, test_axis, loc, title):
        """Test the legend position and title."""
        legend_axis = aa.plot_set_legend(ax=test_axis, loc=loc, title=title)
        assert legend_axis.get_legend().get_title().get_text() == title
        assert legend_axis.get_legend()._loc == loc

    @given(marker=some.sampled_from(['o', 's', '^', 'D']),
           marker_size=some.floats(min_value=1, max_value=20))
    def test_legend_marker(self, test_axis, marker, marker_size):
        """Test the marker styles in the legend."""
        dict_color = {"Cat1": "red"}
        list_cat = ["Cat1"]
        legend_axis = aa.plot_set_legend(ax=test_axis, dict_color=dict_color, list_cat=list_cat, marker=marker, marker_size=marker_size)
        handle = legend_axis.get_legend().legendHandles[0]
        assert handle.get_marker() == marker
        assert handle.get_markersize() == marker_size

    @given(fontsize=some.floats(min_value=8, max_value=20),
           weight=some.sampled_from(['normal', 'bold']))
    def test_legend_font(self, test_axis, fontsize, weight):
        """Test the font properties in the legend."""
        dict_color = {"Cat1": "red"}
        list_cat = ["Cat1"]
        legend_axis = aa.plot_set_legend(ax=test_axis, dict_color=dict_color, list_cat=list_cat, fontsize=fontsize, weight=weight)
        label = legend_axis.get_legend().get_texts()[0]
        assert label.get_fontsize() == fontsize
        assert label.get_fontweight() == weight

    # Property-based testing for negative cases
    @given(loc=some.text())
    @example(loc="invalid_location")
    def test_invalid_legend_location(self, test_axis, loc):
        """Test with an invalid legend location."""
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=test_axis, loc=loc)

    @given(marker=some.text())
    @example(marker="invalid_marker")
    def test_invalid_legend_marker(self, test_axis, marker):
        """Test with an invalid marker."""
        with pytest.raises(ValueError):
            dict_color = {"Cat1": "red"}
            list_cat = ["Cat1"]
            aa.plot_set_legend(ax=test_axis, dict_color=dict_color, list_cat=list_cat, marker=marker)

    def test_load_dataset_invalid_min_max_len_and_n(self):
        """Test with 'min_len' greater than 'max_len' and a valid 'n'."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", min_len=10, max_len=5, n=10)

    def test_load_dataset_invalid_non_canonical_aa_and_n(self):
        """Test with an invalid 'non_canonical_aa' value and a valid 'n'."""
        with pytest.raises(ValueError):
            aa.load_dataset(name="SEQ_LOCATION", non_canonical_aa="invalid_option", n=10)



    # Basic function tests
    def test_default(self):
        """Test the default functionality without any arguments."""
        ax = aa.plot_set_legend()
        assert ax is not None

    @given(dict_color=some.dictionaries(keys=some.text(min_size=1, max_size=5),
                                        values=some.text(min_size=1, max_size=7)))
    def test_dict_color(self, dict_color):
        """Test the 'dict_color' parameter."""
        ax = aa.plot_set_legend(dict_color=dict_color)
        assert ax is not None

    @given(list_cat=some.lists(some.text(min_size=1, max_size=5)))
    def test_list_cat(self, list_cat):
        """Test the 'list_cat' parameter."""
        ax = aa.plot_set_legend(list_cat=list_cat)
        assert ax is not None

    @given(title=some.text(min_size=1, max_size=50))
    def test_title(self, title):
        """Test the 'title' parameter."""
        ax = aa.plot_set_legend(title=title)
        assert ax.get_legend().get_title().get_text() == title

    @given(fontsize=some.integers(min_value=8, max_value=20))
    def test_fontsize(self, fontsize):
        """Test the 'fontsize' parameter."""
        ax = aa.plot_set_legend(fontsize=fontsize)
        assert any(item.get_fontsize() == fontsize for item in ax.get_legend().get_texts())

    # Negative Tests
    @given(loc=some.text())
    @example(loc="invalid_location")
    def test_invalid_loc(self, loc):
        """Test with an invalid 'loc' value."""
        with pytest.raises(ValueError):
            aa.plot_set_legend(loc=loc)

    def setup_method(self):
        self.ax = plt.gca()

    def teardown_method(self):
        plt.close('all')

    @given(dict_color=some.dictionaries(keys=some.text(min_size=1, max_size=10),
                                        values=some.sampled_from(['red', 'blue', 'green']), min_size=1, max_size=5))
    def test_legend_colors(self, dict_color):
        """Test the 'dict_color' parameter for setting colors."""
        aa.plot_set_legend(ax=self.ax, dict_color=dict_color, list_cat=list(dict_color.keys()))
        legend = self.ax.get_legend()
        for i, text in enumerate(legend.get_texts()):
            assert text.get_color() == dict_color[text.get_text()]

    @given(x=some.floats(min_value=-1, max_value=1), y=some.floats(min_value=-1, max_value=1))
    def test_legend_position(self, x, y):
        """Test the 'x' and 'y' parameters for setting legend position."""
        aa.plot_set_legend(ax=self.ax, x=x, y=y, dict_color={'Cat': 'red'}, list_cat=['Cat'])
        legend = self.ax.get_legend()
        bbox = legend.get_bbox_to_anchor().inverse_transformed(self.ax.transAxes)
        assert bbox.x0 == x and bbox.y0 == y

    def test_return_handles(self):
        """Test the 'return_handles' parameter."""
        handles, labels = aa.plot_set_legend(ax=self.ax, dict_color={'Cat': 'red'}, list_cat=['Cat'],
                                             return_handles=True)
        assert len(handles) == len(labels) == 1

    @given(title=some.text(min_size=1, max_size=50))
    def test_legend_title(self, title):
        """Test the 'title' parameter for setting legend title."""
        aa.plot_set_legend(ax=self.ax, title=title, dict_color={'Cat': 'red'}, list_cat=['Cat'])
        legend = self.ax.get_legend()
        assert legend.get_title().get_text() == title

    # Additional Negative Tests
    @given(dict_color=some.dictionaries(keys=some.text(min_size=1, max_size=10),
                                        values=some.sampled_from(['red', 'blue', 'green']), min_size=1, max_size=5))
    def test_invalid_list_cat(self, dict_color):
        """Test mismatched 'list_cat' and 'dict_color'."""
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=self.ax, dict_color=dict_color, list_cat=['Invalid'])

    @given(loc=some.text())
    @example(loc="invalid_location")
    def test_invalid_loc(self, loc):
        """Test invalid 'loc' parameter."""
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=self.ax, dict_color={'Cat': 'red'}, list_cat=['Cat'], loc=loc)

