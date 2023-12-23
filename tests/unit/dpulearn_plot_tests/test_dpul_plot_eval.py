import pandas as pd
import matplotlib.pyplot as plt
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import numpy as np
import aaanalysis as aa


# Helper function for creating a DataFrame for testing
COLS_REQUIRED = ['name', 'avg_STD', 'avg_IQR'] + ['avg_abs_AUC_pos', 'avg_KLD_pos', "avg_abs_AUC_unl", "avg_KLD_unl"]
COLS_REQUIRED_NEG = COLS_REQUIRED + ["avg_abs_AUC_neg", "avg_KLD_neg"]

def create_df_eval(n_rows, n_cols, columns):
    return pd.DataFrame(np.random.randn(n_rows, n_cols), columns=columns)

"""
# Test class for normal cases
class TestdPULearnPlotEval:
    # Constants for DataFrame column names

    # Test df_eval parameter
    def test_df_eval(self):
        for i in range(2, 20):
            df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
            fig, axes = aa.dPULearnPlot.eval(df_eval)

            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, plt.Axes)
            df_eval = create_df_eval(5, len(COLS_REQUIRED_NEG), COLS_REQUIRED_NEG)
            fig, axes = aa.dPULearnPlot.eval(df_eval)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, plt.Axes)

    # Test figsize parameter
    @settings(max_examples=10, deadline=200)
    @given(figsize=st.tuples(st.floats(min_value=4, max_value=20), st.floats(min_value=4, max_value=20)))
    def test_figsize(self, figsize):
        df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
        fig, _ = aa.dPULearnPlot.eval(df_eval, figsize=figsize)
        assert fig.get_size_inches() == figsize

    # Test legend parameter
    @settings(max_examples=10, deadline=200)
    @given(legend=st.booleans())
    def test_legend(self, legend):
        df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
        fig, axes = aa.dPULearnPlot.eval(df_eval, legend=legend)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)
        
        
    # Test legend_y parameter
    @settings(max_examples=10, deadline=200)
    @given(legend_y=st.floats())
    def test_legend_y(self, legend_y):
        df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
        fig, axes = aa.dPULearnPlot.eval(df_eval, legend_y=legend_y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

    # Test colors parameter
    @settings(max_examples=10, deadline=200)
    @given(colors=st.lists(st.text(), min_size=4))
    def test_colors(self, colors):
        df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
        fig, axes = aa.dPULearnPlot.eval(df_eval, colors=colors)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

    # Negative tests
    @settings(max_examples=10, deadline=200)
    def test_invalid_df_eval(self):
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=None)


    @settings(max_examples=10, deadline=200)
    def test_invalid_figsize_(self):
        df_eval = create_df_eval(5, len(COLS_REQUIRED), COLS_REQUIRED)
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval, figsize=(-5, -5))

    @given(figsize=st.one_of(st.tuples(st.floats(max_value=0), st.floats(min_value=1)),
                             st.tuples(st.floats(min_value=1), st.floats(max_value=0))))
    def test_invalid_figsize(self, figsize):
        df_eval = pd.DataFrame(np.random.randn(5, len(COLS_REQUIRED)), columns=COLS_REQUIRED)
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval, figsize=figsize)

    @settings(max_examples=10, deadline=200)
    @given(legend_y=st.one_of(st.text(), st.none()))
    def test_invalid_legend_y(self, legend_y):
        df_eval = pd.DataFrame(np.random.randn(5, len(COLS_REQUIRED)), columns=COLS_REQUIRED)
        with pytest.raises(TypeError):
            aa.dPULearnPlot.eval(df_eval, legend_y=legend_y)

    @settings(max_examples=10, deadline=200)
    @given(colors=st.lists(st.integers(), min_size=4))
    def test_invalid_colors(self, colors):
        df_eval = pd.DataFrame(np.random.randn(5, len(COLS_REQUIRED)), columns=COLS_REQUIRED)
        with pytest.raises(TypeError):
            aa.dPULearnPlot.eval(df_eval, colors=colors)

    @settings(max_examples=10, deadline=200)
    def test_missing_required_columns(self):
        df_eval = pd.DataFrame(np.random.randn(5, 3), columns=['col1', 'col2', 'col3'])
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval)


# Test class for complex cases
class TestdPULearnPlotEvalComplex:
    
    # Test with valid combinations of parameters
    def test_valid_combinations(self):
        df_eval = create_df_eval(5, 5, COLS_REQUIRED)
        figsize = (10, 8)
        colors = ['red', 'green', 'blue', 'yellow']
        fig, axes = aa.dPULearnPlot.eval(df_eval, figsize=figsize, colors=colors)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

    # Negative tests for invalid combinations
    @settings(max_examples=10, deadline=200)
    def test_invalid_combinations(self):
        with pytest.raises(ValueError):
            df_eval = pd.DataFrame({'wrong_column': [1, 2, 3]})
            aa.dPULearnPlot.eval(df_eval)
            
            
    # Complex test: Valid combinations with varying data sizes and formats
    @settings(max_examples=10, deadline=200)
    @given(n_rows=st.integers(min_value=1, max_value=10),
           figsize=st.tuples(st.floats(min_value=4, max_value=20), st.floats(min_value=4, max_value=20)),
           legend=st.booleans(), legend_y=st.floats(min_value=-1, max_value=1),
           colors=st.lists(st.sampled_from(['red', 'green', 'blue', 'yellow']), min_size=4))
    def test_valid_combinations_varied(self, n_rows, figsize, legend, legend_y, colors):
        df_eval = pd.DataFrame(np.random.randn(n_rows, len(COLS_REQUIRED)), columns=COLS_REQUIRED)
        fig, axes = aa.dPULearnPlot.eval(df_eval=df_eval, figsize=figsize, legend=legend, legend_y=legend_y,
                                      colors=colors)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

    # Complex test: Invalid combinations with mismatched data and parameters
    @settings(max_examples=10, deadline=200)
    def test_invalid_combinations_mismatched(self):
        df_eval = pd.DataFrame(np.random.randn(5, 2),
                               columns=['avg_std', 'avg_iqr'])  # Missing some required columns
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval, figsize=(10, -5), colors=['red', 123])

    # Complex test: Handling of NaN and Inf values in df_eval
    @settings(max_examples=10, deadline=200)
    @given(data=st.lists(st.lists(st.floats(allow_nan=True, allow_infinity=True)), min_size=1, max_size=10))
    def test_nan_inf_in_df_eval(self, data):
        with pytest.raises(ValueError):
            aa.dPULearnPlot.eval(df_eval=data)

    # Complex test: Extreme values for legend_y and figsize
    @settings(max_examples=10, deadline=200)
    def test_extreme_values_for_parameters(self):
        df_eval = pd.DataFrame(np.random.randn(5, len(COLS_REQUIRED)), columns=COLS_REQUIRED)
        extreme_figsize = (100, 100)  # Unusually large figsize
        extreme_legend_y = 100  # Unusually large legend_y
        with pytest.raises(Exception):  # Expect some form of exception due to extreme values
            aa.dPULearnPlot.eval(df_eval, figsize=extreme_figsize, legend_y=extreme_legend_y)
"""