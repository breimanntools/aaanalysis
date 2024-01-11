"""
This is a script for testing the CPPPlot().eval() method.
"""
import warnings

import hypothesis.strategies as st
from hypothesis import given, settings, assume
import pytest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import aaanalysis as aa

# Required columns for df_eval
REQUIRED_COLS = ['name', 'n_features', 'avg_ABS_AUC', 'range_ABS_AUC',
                 'avg_MEAN_DIF', 'n_clusters', 'avg_n_feat_per_clust', 'std_n_feat_per_clust']

# Helpe functions
def get_df_eval_default():
    """"""
    df_seq = aa.load_dataset(name="DOM_GSEC_PU", n=63)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales()
    split_kws = aa.SequenceFeature().get_split_kws()
    cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
    df_feat = aa.load_features()
    list_df_feat = [df_feat, df_feat.head(125), df_feat.head(100), df_feat.head(50), df_feat.head(25)]
    df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, label_ref=2, min_th=0)
    return df_eval

def create_valid_df_eval(n_rows=5, n_cat=8):
    data = {
        'name': [f'Set {i+1}' for i in range(n_rows)],
        'n_features': [(np.random.randint(20, 150), np.random.randint(0, 60, size=n_cat).tolist()) for _ in range(n_rows)],
        'avg_ABS_AUC': np.random.rand(n_rows),
        'range_ABS_AUC': [np.sort(np.random.rand(5)).tolist() for _ in range(n_rows)],
        'avg_MEAN_DIF': [(np.random.rand(), -np.random.rand()) for _ in range(n_rows)],
        'n_clusters': np.random.randint(1, 30, size=n_rows),
        'avg_n_feat_per_clust': np.random.uniform(1, 10, size=n_rows),
        'std_n_feat_per_clust': np.random.uniform(0, 10, size=n_rows)
    }
    return pd.DataFrame(data)


class TestCPPPlotEval:
    """
    Test class for CPPPlot().eval() method focusing on individual parameters.
    """

    # Positive tests
    def test_df_eval_default(self):
        df_eval = get_df_eval_default()
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        plt.close()

    @settings(max_examples=5, deadline=5000)
    @given(n_rows=st.integers(min_value=2, max_value=7))
    def test_df_eval_input(self, n_rows):
        df_eval = create_valid_df_eval(n_rows=n_rows)
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.eval(df_eval=df_eval)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        plt.close()

    @settings(max_examples=5, deadline=2500)
    @given(figsize=st.tuples(st.integers(min_value=4, max_value=20), st.integers(min_value=4, max_value=20)))
    def test_figsize_input(self, figsize):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axes = cpp_plot.eval(df_eval=df_eval, figsize=figsize)
            assert fig.get_size_inches()[0] == figsize[0]
            assert fig.get_size_inches()[1] == figsize[1]
            plt.close()

    def test_dict_xlims_input(self):
        """Test the 'dict_xlims' parameter with valid data."""
        dict_xlims = {0: (2, 5), 1: (3, 5), 4: (0, 10)}
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axes = cpp_plot.eval(df_eval=df_eval, dict_xlims=dict_xlims)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()


    def test_legend_input(self):
        for legend in [True, False]:
            df_eval = create_valid_df_eval(n_rows=5)
            cpp_plot = aa.CPPPlot()
            fig, axes = cpp_plot.eval(df_eval=df_eval, legend=legend)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()

    @settings(max_examples=5, deadline=2500)
    @given(legend_y=st.floats(min_value=-0.5, max_value=0.5))
    def test_legend_y_input(self, legend_y):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.eval(df_eval=df_eval, legend_y=legend_y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        plt.close()

    def test_dict_color_input(self):
        list_colors = ["red", "blue", "yellow", "orange", "green", "gray", "black", "white",
                       "tab:red", "tab:blue", "tab:pink", "tab:orange", "tab:green", "tab:gray", "gold"]
        for i in range(8, 12):
            list_cat = [f"Cat {j}" for j in range(0, i)]
            dict_color = dict(zip(list_cat, list_colors[0:i]))
            df_eval = create_valid_df_eval(n_rows=5, n_cat=i)
            cpp_plot = aa.CPPPlot()
            fig, axes = cpp_plot.eval(df_eval=df_eval, dict_color=dict_color, list_cat=list_cat)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()

    def test_list_cat(self):
        valid_list_cats = ['ASA/Volume', 'Composition', 'Conformation', 'Energy', 'Others', 'Polarity', 'Shape', 'Structure-Activity']
        for i in range(2, len(valid_list_cats)):
            df_eval = create_valid_df_eval(n_rows=5, n_cat=i)
            cpp_plot = aa.CPPPlot()
            fig, axes = cpp_plot.eval(df_eval=df_eval, list_cat=valid_list_cats[0:i])
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()

    # Negative tests
    def test_df_eval_invalid_input(self):
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval='invalid_input')

    def test_figsize_invalid_input(self):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, figsize=('invalid', 'input'))

    def test_invlaid_dict_xlims(self):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, dict_xlims=('invalid', 'input'))
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, dict_xlims={10: (10, 12)})
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, dict_xlims={2: ("10", 12)})

    def test_dict_color_invalid_input(self):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        invalid_dict_color = {"invalid": "color"}
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, dict_color=invalid_dict_color)

    def test_dict_color_insufficient_colors(self):
        df_eval = create_valid_df_eval(n_rows=5, n_cat=10)
        cpp_plot = aa.CPPPlot()
        insufficient_dict_color = {f"Cat {i}": "red" for i in range(1, 5)}
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, dict_color=insufficient_dict_color)

    def test_list_cat_invalid(self):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        invalid_list_cat = ['Invalid Category 1', 'Invalid Category 2']
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, list_cat=invalid_list_cat)

    def test_df_eval_missing_columns(self):
        df_eval = pd.DataFrame({'name': ['Set 1', 'Set 2'], 'avg_ABS_AUC': [0.5, 0.6],  # Missing other required columns
        })
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval)

    def test_df_eval_wrong_data_types(self):
        df_eval = create_valid_df_eval(n_rows=5)
        df_eval['range_ABS_AUC'] = 'invalid data type'  # Corrupting data type
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval)

    def test_invalid_legend_y_input(self):
        df_eval = create_valid_df_eval(n_rows=5)
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            fig, axes = cpp_plot.eval(df_eval=df_eval, legend_y="dasf")
        with pytest.raises(ValueError):
            fig, axes = cpp_plot.eval(df_eval=df_eval, legend_y={})


class TestCPPPlotEvalComplex:
    """
    Test class for CPPPlot().eval() method focusing on complex parameter combinations.
    """

    def test_valid_complex_case(self):
        # Create a valid df_eval with specific parameters
        df_eval = create_valid_df_eval(n_rows=5, n_cat=8)

        # Define specific complex input parameters
        figsize = (12, 8)
        legend = True
        legend_y = -0.2
        list_colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray"]
        list_cat = [f"Cat {j + 1}" for j in range(8)]
        dict_color = dict(zip(list_cat, list_colors))

        # Create a CPPPlot instance and call eval with complex parameters
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.eval(df_eval=df_eval, figsize=figsize, legend=legend, legend_y=legend_y, dict_color=dict_color,
                                  list_cat=list_cat)

        # Assertions to check if the plot is correctly generated
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        # Additional assertions can be added as needed
        plt.close()


    def test_invalid_complex_case(self):
        # Create a valid df_eval with specific parameters
        df_eval = create_valid_df_eval(n_rows=5, n_cat=10)

        # Define specific complex input parameters with some invalid values
        figsize = (12, "invalid")  # Invalid figsize value
        legend = "not a boolean"  # Invalid legend value
        dict_color = {"invalid": "color"}  # Invalid dict_color value

        # Create a CPPPlot instance and call eval with complex parameters
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.eval(df_eval=df_eval, figsize=figsize, legend=legend, dict_color=dict_color)