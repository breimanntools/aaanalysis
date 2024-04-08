"""
This script tests the heatmap() method.
"""
import pandas as pd
import matplotlib.pyplot as plt
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa
import random

# Constants and Helper functions
N_SEQ = 10
COL_FEAT_IMPACT_TEST = "feat_impact_test"
COL_MEAN_DIF_TEST = "mean_dif_test"


VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
INVALID_COLORS = ["invalid-color", "tab:black", 234, [], {}]
VALID_WEIGHT = ['normal', 'bold']
INVALID_WEIGHT = ['light', 'italic', 123]
VALID_COL_CATS = ['category', 'subcategory', 'scale_name']
INVALID_COL_CATS = ['cat', 123, [], {}]
VALID_COL_VALS = ['mean_dif', 'abs_mean_dif', 'abs_auc', 'feat_importance']
INVALID_COL_VALS = ['diff', 'mean', 123, [], {}]
LIST_CAT = ['ASA/Volume', 'Conformation', 'Energy', 'Polarity', 'Shape', 'Composition', 'Structure-Activity', 'Others']
DICT_COLOR = dict(zip(LIST_CAT, VALID_COLORS))


# Helper functions
def adjust_vmin_vmax(vmin=None, vmax=None):
    if vmin is not None:
        vmin = -10000 if vmin < -10000 else vmin
        vmin = 10000 if vmin > 10000 else vmin
    if vmax is not None:
        vmax = -10000 if vmin < -10000 else vmin
        vmax = 10000 if vmax > 10000 else vmax
    return vmin, vmax



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


class TestCCPlotHeatmap:
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
        for col_val in [COL_FEAT_IMPACT_TEST, COL_MEAN_DIF_TEST]:
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, shap_plot=True, col_val=col_val,
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
    @given(start=st.integers(min_value=0, max_value=1000))
    def test_start(self, start):
        if start <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, start=start)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_len=st.integers(min_value=20, max_value=100))
    def test_tmd_len(self, tmd_len):
        if tmd_len <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, tmd_len=tmd_len)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_sequence_parameters(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_color=st.sampled_from(VALID_COLORS),
           jmd_color=st.sampled_from(VALID_COLORS),
           tmd_seq_color=st.sampled_from(VALID_COLORS),
           jmd_seq_color=st.sampled_from(VALID_COLORS))
    def test_color_parameters(self, tmd_color, jmd_color, tmd_seq_color, jmd_seq_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat,
                                   tmd_color=tmd_color, jmd_color=jmd_color,
                                   tmd_seq_color=tmd_seq_color,
                                   jmd_seq_color=jmd_seq_color)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(seq_size=st.floats(min_value=8.0, max_value=14.0),
           fontsize=st.floats(min_value=8.0, max_value=14.0))
    def test_font_sizes(self, seq_size, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, seq_size=seq_size,
                                   fontsize_tmd_jmd=fontsize,
                                   fontsize_labels=fontsize)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=10, deadline=5000)
    @given(weight_tmd_jmd=st.sampled_from(VALID_WEIGHT))
    def test_weight_tmd_jmd(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, weight_tmd_jmd=weight_tmd_jmd)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_add_xticks_pos(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for add_xticks_pos in [True, False]:
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, add_xticks_pos=add_xticks_pos)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()
        
    # Positive tests: Legend, Axis, and Grid Configurations
    @settings(max_examples=3, deadline=5000)
    @given(linewidth=st.floats(min_value=0.0, max_value=10.0))
    def test_linewidth(self, linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, grid_linewidth=linewidth,
                                   border_linewidth=linewidth)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_linecolor(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        random_colors = random.sample(VALID_COLORS, 3)
        for lc in random_colors:
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, grid_linecolor=lc)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_facecolor_dark(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for facecolor_dark in [True, False, None]:
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, facecolor_dark=facecolor_dark)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(vmin=st.one_of(st.none(), st.integers(min_value=0), st.floats(min_value=0.0)),
           vmax=st.one_of(st.none(), st.integers(min_value=1), st.floats(min_value=1.0)))
    def test_vmin_vmax(self, vmin, vmax):
        vmin, vmax = adjust_vmin_vmax(vmin=vmin, vmax=vmax)
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        if vmax is None or vmin is None or (vmax is not None and vmin is not None and vmin < vmax):
            fig, ax = cpp_plot.heatmap(df_feat=df_feat, vmin=vmin, vmax=vmax)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cmap=st.one_of(st.none(), st.sampled_from(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])))
    def test_cmap(self, cmap):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, cmap=cmap)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cmap_n_colors=st.integers(min_value=2, max_value=200))
    def test_cmap_n_colors(self, cmap_n_colors):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, cmap_n_colors=cmap_n_colors)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_cbar_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_cbar_kws = {'orientation': 'horizontal'}
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, cbar_kws=valid_cbar_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cbar_pct=st.booleans())
    def test_cbar_pct(self, cbar_pct):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, cbar_pct=cbar_pct)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cbar_xywh=st.tuples(st.floats(min_value=0.0, max_value=1.0),
                               st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
                               st.floats(min_value=0.0, max_value=1.0),
                               st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))))
    def test_cbar_xywh(self, cbar_xywh):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, cbar_xywh=cbar_xywh)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_dict_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, dict_color=DICT_COLOR)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        legend_kws = {'title': 'Legend', 'loc': 'upper right'}
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, legend_kws=legend_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(legend_xy=st.tuples(st.floats(min_value=-1.0, max_value=1.0), st.floats(min_value=-1.0, max_value=1.0)))
    def test_legend_xy(self, legend_xy):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, legend_xy=legend_xy)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(xtick_size=st.floats(min_value=8.0, max_value=14.0),
           xtick_width=st.floats(min_value=0.5, max_value=2.0),
           xtick_length=st.floats(min_value=3.0, max_value=10.0))
    def test_xy_tick_styling(self, xtick_size, xtick_width, xtick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, xtick_size=xtick_size, xtick_width=xtick_width,
                                   xtick_length=xtick_length)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Negative tests: Data and Plot Type
    def test_invalid_df_feat(self):
        cpp_plot = aa.CPPPlot()
        df_feat = "invalid_df_feat"  # This should be a DataFrame
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat)
        plt.close()
        df_feat = get_df_feat(n=150)
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, tmd_len=2)
        plt.close()
        cpp_plot = aa.CPPPlot(jmd_c_len=2)
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(shap_plot=st.text(min_size=1))
    def test_invalid_shap_plot(self, shap_plot):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, shap_plot=shap_plot)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(col_cat=st.sampled_from(INVALID_COL_CATS))
    def test_invalid_col_cat(self, col_cat):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, col_cat=col_cat)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(col_val=st.sampled_from(INVALID_COL_VALS))
    def test_invalid_col_val(self, col_val):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, col_val=col_val)
        plt.close()

    def test_invalid_names(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, name_test=None, name_ref=None)
        for valid_names in [123, [], {}]:
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, name_test=valid_names, name_ref=valid_names)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(figsize=st.tuples(st.just(-10.0), st.just(-10.0)))
    def test_invalid_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, figsize=figsize)
        plt.close()

    def test_zero_or_negative_figsize(self):
        for figsize in [(None, None), [], "asdf", (-12, 1)]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, figsize=figsize)
            plt.close()

    # Negative tests: Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=5000)
    @given(start=st.integers(max_value=-1))  # Invalid negative start
    def test_invalid_start(self, start):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, start=start)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_len=st.integers(max_value=0))  # Invalid non-positive tmd_len
    def test_invalid_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, tmd_len=tmd_len)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_seq=st.text(), jmd_n_seq=st.text(), jmd_c_seq=st.text())
    def test_invalid_sequences(self, tmd_seq, jmd_n_seq, jmd_c_seq):
        if not isinstance(tmd_seq, str) or not isinstance(jmd_n_seq, str) or not isinstance(jmd_c_seq, str):
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(color=st.sampled_from(INVALID_COLORS))
    def test_invalid_color_parameters(self, color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, tmd_color=color, jmd_color=color, tmd_seq_color=color, jmd_seq_color=color)
        plt.close()

    # Negative tests for Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=5000)
    @given(seq_size=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_seq_size(self, seq_size):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, seq_size=seq_size)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(fontsize_tmd_jmd=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_fontsize_tmd_jmd(self, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, fontsize_tmd_jmd=fontsize_tmd_jmd)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(weight_tmd_jmd=st.sampled_from(INVALID_WEIGHT))
    def test_invalid_weight_tmd_jmd(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, weight_tmd_jmd=weight_tmd_jmd)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(fontsize_labels=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_fontsize_labels(self, fontsize_labels):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, fontsize_labels=fontsize_labels)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(add_xticks_pos=st.text(min_size=1))  # Invalid input for boolean parameter
    def test_invalid_add_xticks_pos(self, add_xticks_pos):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, add_xticks_pos=add_xticks_pos)
        plt.close()

    # Negative tests: Legend, Axis, and Grid Configurations
    @settings(max_examples=3, deadline=5000)
    @given(grid_linewidth=st.floats(max_value=-0.01))  # Invalid negative grid_linewidth
    def test_invalid_grid_linewidth(self, grid_linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, grid_linewidth=grid_linewidth)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(grid_linecolor=st.sampled_from(INVALID_COLORS))  # Invalid grid_linecolor
    def test_invalid_grid_linecolor(self, grid_linecolor):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, grid_linecolor=grid_linecolor)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(border_linewidth=st.floats(max_value=-0.01))  # Invalid negative border_linewidth
    def test_invalid_border_linewidth(self, border_linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, border_linewidth=border_linewidth)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(facecolor_dark=st.text(min_size=1))  # Invalid input for optional boolean parameter
    def test_invalid_facecolor_dark(self, facecolor_dark):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, facecolor_dark=facecolor_dark)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(vmin=st.floats(min_value=10.0), vmax=st.floats(max_value=0.0))  # Invalid vmin > vmax
    def test_invalid_vmin_vmax(self, vmin, vmax):
        vmin, vmax = adjust_vmin_vmax(vmin=vmin, vmax=vmax)
        if vmin > vmax:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, vmin=vmin, vmax=vmax)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cmap=st.text(min_size=1))  # Invalid colormap name
    def test_invalid_cmap(self, cmap):
        if cmap not in VALID_COLORS:  # Adjust this condition based on your colormap validation logic
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, cmap=cmap)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cmap_n_colors=st.integers(max_value=0))  # Invalid non-positive cmap_n_colors
    def test_invalid_cmap_n_colors(self, cmap_n_colors):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, cmap_n_colors=cmap_n_colors)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cbar_pct=st.text(min_size=1))  # Invalid input for boolean parameter
    def test_invalid_cbar_pct(self, cbar_pct):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, cbar_pct=cbar_pct)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cbar_xywh=st.tuples(st.just(-0.1), st.just(-0.1), st.just(-0.1), st.just(-0.1)))  # Invalid cbar_xywh values
    def test_invalid_cbar_xywh(self, cbar_xywh):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, cbar_xywh=cbar_xywh)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(cbar_kws=st.text(min_size=1))  # Invalid input for dict parameter
    def test_invalid_cbar_kws(self, cbar_kws):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, cbar_kws=cbar_kws)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(dict_color=st.just({'invalid_cat': 'blue'}))
    def test_invalid_dict_color(self, dict_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, dict_color=dict_color)
        plt.close()
        
    def test_invalid_dict_color_structure(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_dict_color = {'invalid_cat': 123}  # Value should be a valid color representation, not a number
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, dict_color=invalid_dict_color)
        plt.close()

    def test_invalid_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_legend_kws = 'not_a_dict'  # This should be a dictionary
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, legend_kws=invalid_legend_kws)
        plt.close()

    def test_invalid_legend_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for legend_xy in [(), ("q23", 123), (123, 123,1 )]:
            with pytest.raises(ValueError):
                cpp_plot.heatmap(df_feat=df_feat, legend_xy=legend_xy)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(xtick_size=st.just(-1), xtick_width=st.just(-1),
           xtick_length=st.just(-1))
    def test_invalid_tick_styling(self, xtick_size, xtick_width, xtick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, xtick_size=xtick_size,
                             xtick_width=xtick_width, xtick_length=xtick_length)
        plt.close()

class TestCCPlotHeatmapComplex:

    # Positive test
    def test_complex_positive(self):
        """Complex positive test with multiple valid parameters."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.heatmap(
            df_feat=df_feat,
            shap_plot=True,
            col_cat='subcategory',
            col_val=COL_FEAT_IMPACT_TEST,
            name_test='Test Protein',
            name_ref='Ref Protein',
            figsize=(12, 12),
            start=5,
            tmd_len=25,
            tmd_color='mediumspringgreen',
            jmd_color='blue',
            tmd_seq_color='black',
            jmd_seq_color='white',
            seq_size=12,
            fontsize_tmd_jmd=14,
            weight_tmd_jmd='bold',
            fontsize_labels=12,
            add_xticks_pos=True,
            grid_linewidth=0.5,
            grid_linecolor='gray',
            border_linewidth=3,
            facecolor_dark=False,
            vmin=0,
            vmax=5,
            cmap='viridis',
            cmap_n_colors=200,
            cbar_pct=True,
            cbar_xywh=(0.85, 0.1, 0.05, 0.8),
            dict_color=DICT_COLOR,
            legend_kws={'title': 'Categories', 'loc': 'upper left'},
            legend_xy=(-0.15, 1.05),
            xtick_size=11,
            xtick_width=2,
            xtick_length=5,
            **args_seq
        )
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Negative test
    def test_complex_negative_positive(self):
        """Complex positive test with multiple valid parameters."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        with pytest.raises(ValueError):
            cpp_plot.heatmap(df_feat=df_feat, shap_plot='invalid',  # Invalid shap_plot
                col_cat='invalid_category',  # Invalid col_cat
                col_val='invalid_col_val',  # Invalid col_val
                name_test=123,  # Invalid name_test
                name_ref=123,  # Invalid name_ref
                figsize=(-5, -5),  # Invalid figsize
                start=-10,  # Invalid start
                tmd_len=-20,  # Invalid tmd_len
                tmd_color='invalid_color',  # Invalid tmd_color
                jmd_color='invalid_color',  # Invalid jmd_color
                tmd_seq_color='invalid_color',  # Invalid tmd_seq_color
                jmd_seq_color='invalid_color',  # Invalid jmd_seq_color
                seq_size=-1,  # Invalid seq_size
                fontsize_tmd_jmd=-1,  # Invalid fontsize_tmd_jmd
                weight_tmd_jmd='invalid',  # Invalid weight_tmd_jmd
                fontsize_labels=-1,  # Invalid fontsize_labels
                add_xticks_pos='invalid',  # Invalid add_xticks_pos
                grid_linewidth=-1,  # Invalid grid_linewidth
                grid_linecolor='invalid_color',  # Invalid grid_linecolor
                border_linewidth=-1,  # Invalid border_linewidth
                facecolor_dark='invalid',  # Invalid facecolor_dark
                vmin=5,  # Invalid vmin > vmax
                vmax=0,  # Invalid vmin > vmax
                cmap='invalid_cmap',  # Invalid cmap
                cmap_n_colors=-100,  # Invalid cmap_n_colors
                cbar_pct='invalid',  # Invalid cbar_pct
                cbar_xywh=(-0.1, -0.1, -0.1, -0.1),  # Invalid cbar_xywh
                dict_color={'invalid_cat': 'blue'},  # Invalid dict_color
                legend_kws='not_a_dict',  # Invalid legend_kws
                legend_xy=(-2, -2),  # Invalid legend_xy
                xtick_size=-1,  # Invalid xtick_size
                xtick_width=-1,  # Invalid xtick_width
                xtick_length=-1,  # Invalid xtick_length
                **args_seq)
        plt.close()
