"""
This is a script for testing the  aa.AAclustPlot().eval function.
"""
import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
from matplotlib import pyplot as plt
import numpy as np
import aaanalysis as aa


class TestAAclustPlotEval:
    """Test  aa.AAclustPlot().eval function"""

    # Positive tests
    @settings(max_examples=10, deadline=1000)
    @given(n_samples=some.integers(min_value=1, max_value=20))
    def test_data_input(self, n_samples):
        """Test the 'data' parameter with valid data."""
        data = np.random.randn(n_samples, 4)
        fig, axes =  aa.AAclustPlot().eval(data_eval=data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    def test_names_input(self):
        """Test the 'names' parameter with valid data."""
        data = np.random.randn(3, 4)
        names = [f"Set {i}" for i in range(3)]
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, names=names)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()


    def test_dict_xlims_input(self):
        """Test the 'dict_xlims' parameter with valid data."""
        dict_xlims = {"BIC": (2, 5), "CH": (3, 5)}
        data = np.random.randn(5, 4)
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, dict_xlims=dict_xlims)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=5, deadline=1000)
    @given(figsize=some.tuples(some.integers(min_value=4, max_value=20), some.integers(min_value=4, max_value=20)))
    def test_figsize_input(self, figsize):
        """Test the 'figsize' parameter with valid data."""
        data = np.random.randn(5, 4)
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, figsize=figsize)
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        plt.close()


    # Negative test
    @settings(max_examples=5)
    @given(data=some.lists(some.lists(some.floats(allow_nan=True, allow_infinity=True), min_size=2, max_size=10), min_size=2, max_size=10))
    def test_data_with_nans_and_infs(self, data):
        """Test the 'data' parameter with NaN and Inf."""
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(data_eval=data)
            plt.close()


    def test_invalid_data_shape(self):
        """Test with invalid data shape."""
        data = [[0.5, 0.4], [0.3]]
        with pytest.raises(ValueError):
            aa.AAclustPlot().eval(data_eval=data)
            plt.close()

    def test_names_but_no_data(self):
        """Test with names provided but no data."""
        data = []
        names = ["Set 1", "Set 2", "Set 3"]
        with pytest.raises(ValueError):
            aa.AAclustPlot().eval(data_eval=data, names=names)
            plt.close()

    @settings(max_examples=5)
    @given(names=some.lists(some.text(), min_size=1, max_size=1))
    def test_insufficient_names_input(self, names):
        """Test the 'names' parameter with insufficient data."""
        data = np.random.randn(5, 5)
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(data_eval=data, names=names)
            plt.close()

    def test_invalid_dict_xlims(self):
        """Test the 'dict_xlims' parameter with valid data."""
        dict_xlims = {"BIC": (2, -5), "CH": (3, -5)}
        data = np.random.randn(5, 4)
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(data_eval=data, dict_xlims=dict_xlims)
            plt.close()


class TestEvalComplex:
    """Test eval function with complex scenarios"""

    def test_combination_of_params(self):
        """Test with a valid use case."""
        data = [[0.5, 0.4, 5, 6], [0.3, 0.7, 1, 2], [0.9, 0.1, 5, 3]]
        names = ["Set 1", "Set 2", "Set 3"]
        dict_xlims = {"n_clusters": (2, 4), "BIC": (0.1, 1.0), "CH": (0.2, 0.8), "SC": (0.3, 0.9)}
        figsize = (8, 7)
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, names=names, dict_xlims=dict_xlims, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, dict_xlims=dict_xlims, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(data_eval=data, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(data_eval=data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    # Negative tests
    def test_combination_with_invalid_params(self):
        """Test multiple parameters together with some invalid."""
        data = np.random.randn(5, 4)
        names = ["Set 1", "Set 2", "Set 3"]  # Insufficient names
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(data_eval=data, names=names)
            plt.close()

    def test_combination_with_nan_data(self):
        """Test combination of parameters with NaN data."""
        data = np.array([[np.nan, 1.0], [1.0, np.nan]])
        names = ["Set 1", "Set 2"]
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(data_eval=data, names=names)
            plt.close()
