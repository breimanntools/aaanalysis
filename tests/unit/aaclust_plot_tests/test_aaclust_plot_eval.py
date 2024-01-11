"""
This is a script for testing the  aa.AAclustPlot().eval() method.
"""
import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import aaanalysis as aa

COLS_EVAL = ["n_clusters", "BIC", "CH", "SC"]

# Helper function
def _add_names_to_df_eval(df_eval=None, names_datasets=None):
    """Add names column to df_eval"""
    if names_datasets is None:
        n_datasets = len(df_eval)
        names_datasets = [f"Set {i}" for i in range(1, n_datasets + 1)]
    df_eval.insert(0, "name", names_datasets)
    return df_eval

class TestAAclustPlotEval:
    """Test  aa.AAclustPlot().eval function"""

    # Positive tests
    @settings(max_examples=10, deadline=1000)
    @given(n_samples=some.integers(min_value=1, max_value=20))
    def test_data_input(self, n_samples):
        """Test the 'data' parameter with valid data."""
        data = np.random.randn(n_samples, 4)
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval)
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    def test_names_input(self):
        """Test the 'names' parameter with valid data."""
        data = np.random.randn(3, 4)
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval)
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()


    def test_dict_xlims_input(self):
        """Test the 'dict_xlims' parameter with valid data."""
        dict_xlims = {0: (2, 5), 1: (3, 5)}
        data = np.random.randn(5, 4)
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval)
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, dict_xlims=dict_xlims)
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
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval)
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, figsize=figsize)
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        plt.close()


    # Negative test
    @settings(max_examples=5, deadline=1000)
    @given(data=some.lists(some.lists(some.floats(allow_nan=True, allow_infinity=True), min_size=2, max_size=10), min_size=2, max_size=10))
    def test_data_with_nans_and_infs(self, data):
        """Test the 'data' parameter with NaN and Inf."""
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(df_eval=data)
            plt.close()

    def test_invalid_data_shape(self):
        """Test with invalid data shape."""
        data = [[0.5, 0.4], [0.3]]
        with pytest.raises(ValueError):
            aa.AAclustPlot().eval(df_eval=data)
            plt.close()

    def test_names_but_no_data(self):
        """Test with names provided but no data."""
        data = []
        names = ["Set 1", "Set 2", "Set 3"]
        with pytest.raises(ValueError):
            aa.AAclustPlot().eval(df_eval=data)
            plt.close()

    def test_invalid_dict_xlims(self):
        """Test the 'dict_xlims' parameter with valid data."""
        dict_xlims = {0: (2, -5), 1: (3, -5)}
        data = np.random.randn(5, 4)
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, dict_xlims=dict_xlims)
            plt.close()
        dict_xlims = {"0": (2, -5), "1": (3, -5)}
        with pytest.raises(ValueError):
            fig, axes = aa.AAclustPlot().eval(df_eval=df_eval, dict_xlims=dict_xlims)
            plt.close()


class TestEvalComplex:
    """Test eval function with complex scenarios"""

    def test_combination_of_params(self):
        """Test with a valid use case."""
        data = [[0.5, 0.4, 5, 6], [0.3, 0.7, 1, 2], [0.9, 0.1, 5, 3]]
        names = ["Set 1", "Set 2", "Set 3"]
        dict_xlims = {0: (2, 4), 1: (0.1, 1.0), 2: (0.2, 0.8), 3: (0.3, 0.9)}
        figsize = (8, 7)
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval, names_datasets=names)
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, dict_xlims=dict_xlims, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, dict_xlims=dict_xlims, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval, figsize=figsize)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()
        fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 4
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    # Negative tests
    def test_combination_with_nan_data(self):
        """Test combination of parameters with NaN data."""
        data = np.array([[np.nan, 1.0, 1, 3], [1.0, np.nan, 3, 5]])
        df_eval = pd.DataFrame(data, columns=COLS_EVAL)
        df_eval = _add_names_to_df_eval(df_eval=df_eval)
        with pytest.raises(ValueError):
            fig, axes =  aa.AAclustPlot().eval(df_eval=df_eval)
            plt.close()
