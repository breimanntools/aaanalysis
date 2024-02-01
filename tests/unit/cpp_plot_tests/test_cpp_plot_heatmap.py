"""
This script tests the heatmap() method.
"""
import pandas as pd
import matplotlib.pyplot as plt
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

# Constants and Helper functions
N_SEQ = 10
COL_FEAT_IMPACT_TEST = "feat_impact_test"
COL_MEAN_DIF_TEST = "mean_dif_test"


VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
INVALID_COLORS = ["invalid-color", "tab:black", 234, [], {}]
VALID_LITERALS = ['normal', 'bold']
INVALID_LITERALS = ['light', 'italic', 123]
VALID_COL_CATS = ['category', 'subcategory', 'scale_name']
INVALID_COL_CATS = ['cat', 123, [], {}]
VALID_COL_VALS = ['mean_dif', COL_MEAN_DIF_TEST, 'abs_mean_dif', 'abs_auc', 'feat_importance', COL_FEAT_IMPACT_TEST]
INVALID_COL_VALS = ['diff', 'mean', 123, [], {}]
LIST_CAT = ['ASA/Volume', 'Conformation', 'Energy', 'Polarity', 'Shape', 'Composition', 'Structure-Activity', 'Others']
DICT_COLOR = dict(zip(LIST_CAT, VALID_COLORS))

def get_args_seq(n=0):
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SEQ)
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    return args_seq


def get_df_feat(n=10):
    aa.options["verbose"] = False
    df_feat = aa.load_features().head(n)
    df_feat.insert(0, COL_FEAT_IMPACT_TEST, [2]*len(df_feat))
    df_feat.insert(0, COL_MEAN_DIF_TEST, [1]*len(df_feat))
    return df_feat


class TestHeatmap:
    """Normal test cases for the heatmap method, focusing on individual parameters."""

    # Positive tests: Data and Plot Type
    def test_df_feat(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_shap_plot(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, shap_plot=True, col_val=COL_FEAT_IMPACT_TEST,
                                   **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_col_cat(self):
        for col_cat in VALID_COL_CATS:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, col_cat=col_cat)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=2000)
    @given(col_val=st.sampled_from(VALID_COL_VALS))
    def test_col_val(self, col_val):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, col_val=col_val)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_normalize(self):
        for normalize in [True, False]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, normalize=normalize)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_names(self):
        for valid_names in ["res", "Protein", "AA AA"]:
            df_feat = get_df_feat()
            cpp_plot = aa.CPPPlot()
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, name_test=valid_names, name_ref=valid_names)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(figsize=st.tuples(st.floats(min_value=4.0, max_value=20.0), st.floats(min_value=5.0, max_value=20.0)))
    def test_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, figsize=figsize)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()
            
    # Positive tests: Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=5000)
    @given(tmd_color=st.sampled_from(VALID_COLORS))
    def test_tmd_color(self, tmd_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, tmd_color=tmd_color)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(grid_linewidth=st.floats(min_value=0.0, max_value=10.0))
    def test_grid_linewidth(self, grid_linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, grid_linewidth=grid_linewidth)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Positive tests: Legend, Axis, and Grid Configurations


    # Negative tests: Data and Plot Type

    # Negative tests: Appearance of Parts (TMD-JMD)

    # Negative tests: Legend, Axis, and Grid Configurations
