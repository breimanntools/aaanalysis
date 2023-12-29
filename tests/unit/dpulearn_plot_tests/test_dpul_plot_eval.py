import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import aaanalysis as aa
import random

# Sample columns for testing
COLS_EVAL_REQUIERED = ['name', 'avg_STD', 'avg_IQR', 'avg_abs_AUC_pos', 'avg_abs_AUC_unl']
COLS_EVAL = COLS_EVAL_REQUIERED + ["avg_abs_AUC_neg", "avg_KLD_pos", "avg_KLD_unl", "avg_KLD_neg"]
VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']


def _create_sample_df_eval(n_rows=5, just_ned=True, random_order=False):
    cols = COLS_EVAL_REQUIERED if just_ned else COLS_EVAL
    # Generate random data
    data_matrix = [[random.uniform(0, 1) for _ in cols] for _ in range(n_rows)]
    df_eval = pd.DataFrame(data_matrix, columns=cols)
    if random_order:
        shuffled_cols = random.sample(cols, len(cols))
        df_eval = df_eval[shuffled_cols]
    return df_eval


class TestdPULearnPlotEval:
    """Test aa.dPULearnPlot.eval() function for individual parameters."""

    # Positive tests
    @settings(max_examples=10, deadline=4000)
    @given(n_samples=some.integers(min_value=2, max_value=10))
    def test_df_eval_valid(self, n_samples):
        df_eval = _create_sample_df_eval(n_rows=n_samples)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)
        df_eval = _create_sample_df_eval(n_rows=n_samples, just_ned=False)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)
        df_eval = _create_sample_df_eval(n_rows=n_samples, just_ned=False, random_order=True)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)

    @settings(max_examples=10, deadline=3000)
    @given(figsize=some.tuples(some.floats(min_value=4, max_value=20), some.floats(min_value=4, max_value=20)))
    def test_figsize_valid(self, figsize):
        df_eval = _create_sample_df_eval()
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, figsize=figsize)
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        plt.close(fig)

    def test_legend_valid(self):
        df_eval = _create_sample_df_eval()
        for legend in [True, False]:
            fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, legend=legend)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert isinstance(axes[0], plt.Axes)
            plt.close(fig)


    @settings(max_examples=10, deadline=3000)
    @given(legend_y=some.floats(min_value=-1, max_value=1))
    def test_legend_y_valid(self, legend_y):
        df_eval = _create_sample_df_eval()
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, legend_y=legend_y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)

    def test_colors_valid(self):
        # Test 15 random color lists
        for i in range(15):
            df_eval = _create_sample_df_eval()
            colors = random.sample(VALID_COLORS, 4)
            fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, colors=colors)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert isinstance(axes[0], plt.Axes)
            plt.close(fig)

    # Negative tests for each parameter
    def test_df_eval_invalid(self):
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval="invalid")
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=None)
        df_eval = _create_sample_df_eval()
        df_eval.columns = [c.upper() for c in list(df_eval)]
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval)
        df_eval.columns = ["name" for i in list(df_eval)]
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval)

    def test_figsize_invalid(self):
        df_eval = _create_sample_df_eval()
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, figsize=(0, 4))

    def test_legend_y_invalid(self):
        df_eval = _create_sample_df_eval()
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, legend_y="invalid")

    def test_colors_invalid(self):
        df_eval = _create_sample_df_eval()
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, colors="invalid")
        colors = random.sample(VALID_COLORS, 2)
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, colors=colors)
        colors = [None] * 4
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, colors=colors)


class TestdPULearnPlotEvalComplex:
    """Test aa.dPULearnPlot.eval() function for combinations of parameters."""

    def test_valid_combination(self):
        df_eval = _create_sample_df_eval()
        figsize = (8, 6)
        legend = True
        legend_y = -0.2
        colors = ['blue', 'green', 'red', 'yellow']
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, figsize=figsize, legend=legend, legend_y=legend_y, colors=colors)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)

    def test_valid_combination_diff_legend_pos(self):
        """Test with a valid combination of parameters and different legend positions."""
        df_eval = _create_sample_df_eval()
        figsize = (8, 6)
        legend = True
        legend_y = 0.5  # Different from the usual -0.2
        colors = ['blue', 'green', 'red', 'yellow']
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, figsize=figsize, legend=legend, legend_y=legend_y, colors=colors)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        plt.close(fig)

    def test_invalid_combination(self):
        df_eval = _create_sample_df_eval()
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, figsize=(0, 4), legend=123, legend_y='invalid')

    def test_invalid_color_list(self):
        """Test with an invalid color list."""
        df_eval = _create_sample_df_eval()
        figsize = (8, 6)
        legend = True
        legend_y = -0.2
        colors = ['blue']  # Only one color for multiple data categories
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=df_eval, figsize=figsize, legend=legend, legend_y=legend_y, colors=colors)