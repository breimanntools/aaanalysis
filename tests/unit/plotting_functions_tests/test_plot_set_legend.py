"""
This is a script for testing the aa.aa.plot_set_legend function.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import hypothesis.strategies as some
from hypothesis import given, settings, example
import aaanalysis as aa
import pytest


# Helper function
def random_colors():
    """Generate random colors."""
    return some.tuples(some.integers(min_value=0, max_value=255),
                       some.integers(min_value=0, max_value=255),
                       some.integers(min_value=0, max_value=255)).map(lambda x: '#%02X%02X%02X' % x)


class TestPlotSetLegend:
    """Test aa.plot_set_legend function"""

    # Property-based testing for positive cases
    @given(x=some.floats(min_value=0, max_value=1), y=some.floats(min_value=-1, max_value=1),
        ncol=some.integers(min_value=1, max_value=5), fontsize=some.integers(min_value=8, max_value=15),
        lw=some.floats(min_value=0, max_value=2), labelspacing=some.floats(min_value=0, max_value=1),
        columnspacing=some.floats(min_value=0, max_value=2))
    def test_basic_legend(self, x, y, ncol, fontsize, lw, labelspacing, columnspacing):
        """Test basic legend functionality."""
        dict_color = {'Cat1': 'red', 'Cat2': 'blue'}
        fig, ax = plt.subplots()
        aa.plot_set_legend(ax=ax, dict_color=dict_color, x=x, y=y, ncol=ncol, fontsize=fontsize, lw=lw,
                        labelspacing=labelspacing, columnspacing=columnspacing)
        # Assert legend created
        assert ax.get_legend() is not None

    @given(dict_color=some.dictionaries(keys=some.text(min_size=1, max_size=5), values=random_colors(), min_size=2,
                                        max_size=5), shape=some.sampled_from(['o', 's', '^', 'v', 'D', '*', 'X']))
    def test_color_and_shapes(self, dict_color, shape):
        """Test different colors and marker shapes."""
        fig, ax = plt.subplots()
        aa.plot_set_legend(ax=ax, dict_color=dict_color, shape=shape)
        # Assert legend created with specified colors and shapes
        assert ax.get_legend() is not None  # TODO: Add more specific assertions to check the colors and shapes


class TestPlotSetLegendComplex:
    """Test aa.plot_set_legend function with complex scenarios"""

    def test_combined_parameters(self):
        """Test a combination of parameters."""
        dict_color = {'Cat1': 'red', 'Cat2': 'blue', 'Cat3': 'green'}
        labels = ['Category 1', 'Category 2', 'Category 3']
        fig, ax = plt.subplots()
        aa.plot_set_legend(ax=ax, dict_color=dict_color, labels=labels, x=0.8, y=-0.5, ncol=2, fontsize=13, weight="bold",
                        lw=1.5, labelspacing=0.5, columnspacing=1.2, title="Test Title", fontsize_legend=15,
                        title_align_left=False, shape='s')
        # Assert legend created with the combined settings
        assert ax.get_legend() is not None  # TODO: Add specific assertions to verify each of the settings

    def test_invalid_input_combinations(self):
        """Test invalid combinations of parameters."""
        dict_color = {'Cat1': 'red', 'Cat2': 'blue'}
        fig, ax = plt.subplots()

        # Invalid number of labels
        labels = ['Category 1']
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color=dict_color, labels=labels)

        # Invalid x and y combination
        with pytest.raises(ValueError):
            aa.plot_set_legend(ax=ax, dict_color=dict_color, x=2, y=2)

        # TODO: Add more invalid combinations to test robustness
