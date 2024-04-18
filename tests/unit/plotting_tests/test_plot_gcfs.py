"""
This is a script for testing the aa.plot_gcfs function.
"""
import seaborn as sns
import aaanalysis as aa


class TestPlotGCFS:
    """Test plot_gcfs function"""

    # Positive test cases
    def test_retrieve_current_font_size(self):
        """Test that function retrieves the current font size."""
        current_context = sns.plotting_context()
        expected_font_size = current_context['font.size']

        font_size = aa.plot_gcfs()
        assert font_size == expected_font_size

    def test_font_size_not_none(self):
        """Test that function doesn't return None."""
        font_size = aa.plot_gcfs()
        assert font_size is not None
