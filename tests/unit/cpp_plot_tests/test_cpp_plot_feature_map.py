"""
This script tests the heatmap() method.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa
import random

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Constants and Helper functions
N_SEQ = 10
COL_FEAT_IMPORTANCE_TEST = "feat_importance_test"
COL_MEAN_DIF_TEST = "mean_dif_test"

VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
INVALID_COLORS = ["invalid-color", "tab:black", 234, [], {}]
VALID_WEIGHT = ['normal', 'bold']
INVALID_WEIGHT = ['light', 'italic', 123]
VALID_COL_CATS = ['category', 'subcategory', 'scale_name']
INVALID_COL_CATS = ['cat', 123, [], {}]
VALID_COL_VALS = ['mean_dif', 'abs_mean_dif', 'abs_auc', 'feat_importance']
INVALID_COL_VALS = ['diff', 'mean', 123, [], {}]
LIST_CAT = ['ASA/Volume', 'Conformation', 'Energy', 'Polarity', 'Shape', 'Composition', 'Structure-Activity', 'Others']
DICT_COLOR = dict(zip(LIST_CAT, VALID_COLORS))


# Helper function
def adjust_vmin_vmax(vmin=None, vmax=None):
    if vmin is not None:
        vmin = -10000 if vmin < -10000 else vmin
        vmin = 10000 if vmin > 10000 else vmin
    if vmax is not None:
        vmax = -10000 if vmax < -10000 else vmax
        vmax = 10000 if vmax > 10000 else vmax
    return vmin, vmax


def get_args_seq(n=0):
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SEQ)
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    return args_seq


COL_FEAT_IMPACT_TEST = "feat_impact_test"
SHAP_POS = mcolors.to_rgba("#FF0D57")
SHAP_NEG = mcolors.to_rgba("#1E88E5")
FEAT_IMP_GRAY = mcolors.to_rgba("#7F7F7F")


def get_df_feat(n=10):
    aa.options["verbose"] = False
    df_feat = aa.load_features().head(n)
    df_feat.insert(0, COL_FEAT_IMPORTANCE_TEST, [2] * len(df_feat))
    df_feat.insert(0, COL_MEAN_DIF_TEST, [1] * len(df_feat))
    return df_feat


def get_df_feat_shap(n=10):
    """Feature table with a signed sample-level SHAP feature-impact column."""
    df_feat = get_df_feat(n=n).reset_index(drop=True)
    # Alternating +/- signed impact so both red and blue stack segments appear
    signs = [1 if i % 2 == 0 else -1 for i in range(len(df_feat))]
    df_feat[COL_FEAT_IMPACT_TEST] = [s * (1 + i) for i, s in enumerate(signs)]
    return df_feat


def get_bar_facecolors(fig):
    """Collect rounded RGBA face colors of all bar patches across a figure."""
    colors = set()
    for ax in fig.axes:
        for patch in ax.patches:
            colors.add(tuple(round(x, 3) for x in patch.get_facecolor()))
    return colors


def _rgba(color):
    return tuple(round(x, 3) for x in color)


def _right_importance_bar(fig, hm):
    """The narrow importance bar to the RIGHT of the heatmap, robust to figure size.

    It is glued to the heatmap's right edge and shares its vertical extent (sharey), so
    identify it by that y-extent match rather than an absolute height fraction -- the latter
    changes as the constant-cell sizer resizes the figure. Returns the axes, or None."""
    hp = hm.get_position()
    for a in fig.get_axes():
        if a is hm:
            continue
        p = a.get_position()
        if (p.x0 >= hp.x1 - 0.02 and p.width < hp.width
                and abs(p.y0 - hp.y0) < 0.02 and abs(p.height - hp.height) < 0.02):
            return a
    return None


class TestCCPlotFeatureMap:
    """Normal test cases for the heatmap method, focusing on individual parameters."""

    # Positive tests: Data and Plot Type
    def test_df_feat(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_col_cat(self):
        for col_cat in VALID_COL_CATS:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, col_cat=col_cat)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_valid_col_imp(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, col_imp=COL_FEAT_IMPORTANCE_TEST)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(col_val=st.sampled_from(VALID_COL_VALS))
    def test_col_val(self, col_val):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, col_val=col_val)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_names(self):
        for valid_names in ["res", "Protein", "AA AA"]:
            df_feat = get_df_feat()
            cpp_plot = aa.CPPPlot()
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, name_test=valid_names, name_ref=valid_names)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(figsize=st.tuples(st.floats(min_value=4.0, max_value=20.0), st.floats(min_value=5.0, max_value=20.0)))
    def test_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, figsize=figsize)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Positive tests: Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=None)
    @given(start=st.integers(min_value=0, max_value=1000))
    def test_start(self, start):
        if start <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, start=start)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(tmd_len=st.integers(min_value=20, max_value=100))
    def test_tmd_len(self, tmd_len):
        if tmd_len <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, tmd_len=tmd_len)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_sequence_parameters(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(tmd_color=st.sampled_from(VALID_COLORS), jmd_color=st.sampled_from(VALID_COLORS),
           tmd_seq_color=st.sampled_from(VALID_COLORS), jmd_seq_color=st.sampled_from(VALID_COLORS))
    def test_color_parameters(self, tmd_color, jmd_color, tmd_seq_color, jmd_seq_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, tmd_color=tmd_color, jmd_color=jmd_color,
                                   tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(seq_size=st.floats(min_value=8.0, max_value=14.0),
           fontsize=st.floats(min_value=8.0, max_value=14.0))
    def test_font_sizes(self, seq_size, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, seq_size=seq_size,
                                       fontsize_tmd_jmd=fontsize,
                                       fontsize_labels=fontsize,
                                       fontsize_annotations=fontsize,
                                       fontsize_titles=fontsize,
                                       fontsize_imp_bar=fontsize)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=10, deadline=None)
    @given(weight_tmd_jmd=st.sampled_from(VALID_WEIGHT))
    def test_weight_tmd_jmd(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, weight_tmd_jmd=weight_tmd_jmd)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_add_xticks_pos(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for add_xticks_pos in [True, False]:
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, add_xticks_pos=add_xticks_pos)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    # Positive tests: Legend, Axis, and Grid Configurations
    @settings(max_examples=3, deadline=None)
    @given(linewidth=st.floats(min_value=0.0, max_value=10.0))
    def test_linewidth(self, linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, grid_linewidth=linewidth, border_linewidth=linewidth)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_linecolor(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        random_colors = random.sample(VALID_COLORS, 3)
        for lc in random_colors:
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, grid_linecolor=lc)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_facecolor_dark(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for facecolor_dark in [True, False]:
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, facecolor_dark=facecolor_dark)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(vmin=st.one_of(st.none(), st.integers(min_value=0), st.floats(min_value=0.0)),
           vmax=st.one_of(st.none(), st.integers(min_value=1), st.floats(min_value=1.0)))
    def test_vmin_vmax(self, vmin, vmax):
        vmin, vmax = adjust_vmin_vmax(vmin=vmin, vmax=vmax)
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        if vmax is None or vmin is None or (vmax is not None and vmin is not None and vmin < vmax):
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, vmin=vmin, vmax=vmax)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cmap=st.one_of(st.none(), st.sampled_from(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])))
    def test_cmap(self, cmap):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, cmap=cmap)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cmap_n_colors=st.integers(min_value=2, max_value=200))
    def test_cmap_n_colors(self, cmap_n_colors):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, cmap_n_colors=cmap_n_colors)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cbar_pct=st.booleans())
    def test_cbar_pct(self, cbar_pct):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, cbar_pct=cbar_pct)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_cbar_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_cbar_kws = {'orientation': 'horizontal'}
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, cbar_kws=valid_cbar_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cbar_xywh=st.tuples(st.floats(min_value=0.0, max_value=1.0),
                               st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
                               st.floats(min_value=0.0, max_value=1.0),
                               st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))))
    def test_cbar_xywh(self, cbar_xywh):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, cbar_xywh=cbar_xywh)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_dict_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, dict_color=DICT_COLOR)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        legend_kws = {'title': 'Legend', 'loc': 'upper right'}
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, legend_kws=legend_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(legend_xy=st.tuples(st.floats(min_value=-1.0, max_value=1.0), st.floats(min_value=-1.0, max_value=1.0)))
    def test_legend_xy(self, legend_xy):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, legend_xy=legend_xy)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_imp_legend_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_legend_imp_xy = (1.5, 0.5)
        fig, ax = cpp_plot.feature_map(df_feat=df_feat,
                                       legend_imp_xy=valid_legend_imp_xy)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_imp_ths(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_imp_ths = (0.1, 0.4, 0.9)
        fig, ax = cpp_plot.feature_map(df_feat=df_feat,
                                       imp_ths=valid_imp_ths)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_imp_marker_sizes(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_imp_marker_sizes = (1, 4, 6)
        fig, ax = cpp_plot.feature_map(df_feat=df_feat,
                                       imp_marker_sizes=valid_imp_marker_sizes)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_add_imp_bar_top(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for add_imp_bar_top in [True, False]:
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, add_imp_bar_top=add_imp_bar_top)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    def test_valid_imp_bar_th(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        valid_imp_bar_th = 0.5
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, imp_bar_th=valid_imp_bar_th)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_valid_imp_bar_label_type(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for imp_bar_label_type in ["short", "long", None]:
            fig, ax = cpp_plot.feature_map(df_feat=df_feat, imp_bar_label_type=imp_bar_label_type)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(xtick_size=st.floats(min_value=8.0, max_value=14.0), xtick_width=st.floats(min_value=0.5, max_value=2.0),
           xtick_length=st.floats(min_value=3.0, max_value=10.0))
    def test_x_tick_styling(self, xtick_size, xtick_width, xtick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, xtick_size=xtick_size, xtick_width=xtick_width,
                                   xtick_length=xtick_length)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Negative tests: Data and Plot Type
    def test_invalid_df_feat(self):
        cpp_plot = aa.CPPPlot()
        df_feat = "invalid_df_feat"  # This should be a DataFrame
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat)
        plt.close()
        df_feat = get_df_feat(n=150)
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, tmd_len=2)
        plt.close()
        cpp_plot = aa.CPPPlot(jmd_c_len=2)
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(col_cat=st.sampled_from(INVALID_COL_CATS))
    def test_invalid_col_cat(self, col_cat):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, col_cat=col_cat)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(col_val=st.sampled_from(INVALID_COL_VALS))
    def test_invalid_col_val(self, col_val):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, col_val=col_val)
        plt.close()

    def test_invalid_col_imp(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_col_imp = 'invalid_col'
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, col_imp=invalid_col_imp)
        plt.close()

    def test_invalid_names(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, name_test=None, name_ref=None)
        for valid_names in [123, [], {}]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, name_test=valid_names, name_ref=valid_names)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(figsize=st.tuples(st.just(-10.0), st.just(-10.0)))
    def test_invalid_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, figsize=figsize)
        plt.close()

    def test_zero_or_negative_figsize(self):
        for figsize in [(None, None), [], "asdf", (-12, 1)]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, figsize=figsize)
            plt.close()

    # Negative tests: Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=None)
    @given(start=st.integers(max_value=-1))  # Invalid negative start
    def test_invalid_start(self, start):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, start=start)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(tmd_len=st.integers(max_value=0))  # Invalid non-positive tmd_len
    def test_invalid_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, tmd_len=tmd_len)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(tmd_seq=st.text(), jmd_n_seq=st.text(), jmd_c_seq=st.text())
    def test_invalid_sequences(self, tmd_seq, jmd_n_seq, jmd_c_seq):
        if not isinstance(tmd_seq, str) or not isinstance(jmd_n_seq, str) or not isinstance(jmd_c_seq, str):
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(color=st.sampled_from(INVALID_COLORS))
    def test_invalid_color_parameters(self, color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, tmd_color=color, jmd_color=color, tmd_seq_color=color,
                             jmd_seq_color=color)
        plt.close()

    # Negative tests for Appearance of Parts (TMD-JMD)
    @settings(max_examples=3, deadline=None)
    @given(seq_size=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_seq_size(self, seq_size):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, seq_size=seq_size)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_tmd_jmd=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_fontsize_tmd_jmd(self, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, fontsize_tmd_jmd=fontsize_tmd_jmd)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(weight_tmd_jmd=st.sampled_from(INVALID_WEIGHT))
    def test_invalid_weight_tmd_jmd(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, weight_tmd_jmd=weight_tmd_jmd)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(fontsize_labels=st.one_of(st.integers(max_value=-1), st.floats(max_value=-0.1)))
    def test_invalid_fontsize_labels(self, fontsize_labels):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat,
                                 fontsize_labels=fontsize_labels,
                                 fontsize_annotations=fontsize_labels,
                                 fontsize_titles=fontsize_labels,
                                 fontsize_imp_bar=fontsize_labels)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(add_xticks_pos=st.text(min_size=1))  # Invalid input for boolean parameter
    def test_invalid_add_xticks_pos(self, add_xticks_pos):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, add_xticks_pos=add_xticks_pos)
        plt.close()

    # Negative tests: Legend, Axis, and Grid Configurations
    @settings(max_examples=3, deadline=None)
    @given(grid_linewidth=st.floats(max_value=-0.01))  # Invalid negative grid_linewidth
    def test_invalid_grid_linewidth(self, grid_linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, grid_linewidth=grid_linewidth)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(grid_linecolor=st.sampled_from(INVALID_COLORS))  # Invalid grid_linecolor
    def test_invalid_grid_linecolor(self, grid_linecolor):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, grid_linecolor=grid_linecolor)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(border_linewidth=st.floats(max_value=-0.01))  # Invalid negative border_linewidth
    def test_invalid_border_linewidth(self, border_linewidth):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, border_linewidth=border_linewidth)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(facecolor_dark=st.text(min_size=1))  # Invalid input for optional boolean parameter
    def test_invalid_facecolor_dark(self, facecolor_dark):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, facecolor_dark=facecolor_dark)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(vmin=st.floats(min_value=10.0), vmax=st.floats(max_value=0.0))  # Invalid vmin > vmax
    def test_invalid_vmin_vmax(self, vmin, vmax):
        vmin, vmax = adjust_vmin_vmax(vmin=vmin, vmax=vmax)
        if vmin > vmax:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, vmin=vmin, vmax=vmax)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cmap=st.text(min_size=1))  # Invalid colormap name
    def test_invalid_cmap(self, cmap):
        if cmap not in VALID_COLORS:  # Adjust this condition based on your colormap validation logic
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, cmap=cmap)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cmap_n_colors=st.integers(max_value=0))  # Invalid non-positive cmap_n_colors
    def test_invalid_cmap_n_colors(self, cmap_n_colors):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, cmap_n_colors=cmap_n_colors)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cbar_pct=st.text(min_size=1))  # Invalid input for boolean parameter
    def test_invalid_cbar_pct(self, cbar_pct):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, cbar_pct=cbar_pct)
        plt.close()

    def test_invalid_cbar_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_cbar_kws = 'not_a_dict'  # Should be a dictionary
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, cbar_kws=invalid_cbar_kws)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cbar_xywh=st.tuples(st.just(-0.1), st.just(-0.1), st.just(-0.1), st.just(-0.1)))  # Invalid cbar_xywh values
    def test_invalid_cbar_xywh(self, cbar_xywh):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, cbar_xywh=cbar_xywh)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(cbar_kws=st.text(min_size=1))  # Invalid input for dict parameter
    def test_invalid_cbar_kws(self, cbar_kws):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, cbar_kws=cbar_kws)
        plt.close()

    @settings(max_examples=3, deadline=None)
    @given(dict_color=st.just({'invalid_cat': 'blue'}))
    def test_invalid_dict_color(self, dict_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, dict_color=dict_color)
        plt.close()

    def test_invalid_dict_color_structure(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_dict_color = {'invalid_cat': 123}  # Value should be a valid color representation, not a number
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, dict_color=invalid_dict_color)
        plt.close()

    def test_invalid_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_legend_kws = 'not_a_dict'  # This should be a dictionary
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, legend_kws=invalid_legend_kws)
        plt.close()

    def test_invalid_legend_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for legend_xy in [(), ("q23", 123), (123, 123, 1)]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, legend_xy=legend_xy)
            plt.close()

    def test_invalid_legend_imp_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_legend_imp_xy = 'not_a_tuple'  # Should be a tuple
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, legend_imp_xy=invalid_legend_imp_xy)
        plt.close()

    def test_invalid_imp_ths(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_imp_ths = (1.1, 0.4, 0.9)  # Values should be in a certain range or order
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, imp_ths=invalid_imp_ths)
        plt.close()

    def test_invalid_imp_marker_sizes(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_imp_marker_sizes = (1.1, 0.4, 0.9)  # Values should be in a certain range or order
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, imp_marker_sizes=invalid_imp_marker_sizes)
        plt.close()

    def test_invalid_add_imp_bar_top(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for add_imp_bar_top in [None, "adsf", 123, pd.DataFrame, {}]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, add_imp_bar_top=add_imp_bar_top)
            plt.close()

    def test_invalid_imp_bar_th(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_imp_bar_th = -0.5  # Negative values might be invalid
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, imp_bar_th=invalid_imp_bar_th)
        plt.close()

    def test_invalid_imp_bar_label_type(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        for imp_bar_label_type in ["adsf", 123, pd.DataFrame, {}]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, imp_bar_label_type=imp_bar_label_type)
            plt.close()

    @settings(max_examples=3, deadline=None)
    @given(xtick_size=st.just(-1), xtick_width=st.just(-1), xtick_length=st.just(-1))
    def test_invalid_tick_styling(self, xtick_size, xtick_width, xtick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        plt.close()


class TestCCPlotFeatureMapComplex:

    # Positive test
    def test_complex_positive(self):
        """Complex positive test with multiple valid parameters."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, col_cat='subcategory', col_val="abs_mean_dif",
                                       name_test='Test Protein', name_ref='Ref Protein', figsize=(12, 12), start=5, tmd_len=25,
                                       tmd_color='mediumspringgreen', jmd_color='blue', tmd_seq_color='black', jmd_seq_color='white', seq_size=12,
                                       fontsize_tmd_jmd=14, weight_tmd_jmd='bold', fontsize_labels=12, add_xticks_pos=True, grid_linewidth=0.5,
                                       grid_linecolor='gray', border_linewidth=3, facecolor_dark=False, vmin=0, vmax=5, cmap='viridis',
                                       cmap_n_colors=200, cbar_pct=True, cbar_xywh=(0.85, 0.1, 0.05, 0.8), dict_color=DICT_COLOR,
                                       legend_kws={'title': 'Categories', 'loc': 'upper left'}, legend_xy=(-0.15, 1.05), xtick_size=11,
                                       xtick_width=2, xtick_length=5, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    # Negative test
    def test_complex_negative_positive(self):
        """Complex positive test with multiple valid parameters."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat,
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


class TestCCPlotFeatureMapShap:
    """Tests for the SHAP mode (shap_plot) of the feature_map method (issue #63)."""

    # Positive tests
    def test_shap_plot_false_returns_fig_ax(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=False)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_shap_plot_true_returns_fig_ax(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                       col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_FEAT_IMPACT_TEST)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_shap_impact_bar_aligns_with_correct_row(self):
        # Regression: the per-subcategory SHAP impact bar must sit on its OWN row, not be
        # compacted onto the top rows. Put all impact in a single subcategory and check the
        # bar's y-position matches that subcategory's heatmap row index. col_val is a
        # mean-difference column (not a feat_impact column) so the side bars are shown.
        cpp_plot = aa.CPPPlot()
        df_feat = aa.load_features("DOM_GSEC")
        subcats = list(dict.fromkeys(df_feat["subcategory"]))[:12]
        df_feat = df_feat[df_feat["subcategory"].isin(subcats)].copy()
        target = subcats[-1]  # a subcategory near the bottom of the row order
        df_feat[COL_FEAT_IMPACT_TEST] = [5.0 if s == target else 0.0 for s in df_feat["subcategory"]]
        fig, hm = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                       col_val="mean_dif", col_imp=COL_FEAT_IMPACT_TEST)
        fig.canvas.draw()
        ylabels = [t.get_text() for t in hm.get_yticklabels()]
        row = ylabels.index(target)
        bar_ax = _right_importance_bar(fig, hm)
        assert bar_ax is not None, "right importance bar not found"
        y_centers = {round(p.get_y() + p.get_height() / 2)
                     for p in bar_ax.patches if getattr(p, "get_width", lambda: 0)() > 0.01}
        assert y_centers == {row}, (y_centers, "expected only row", row)
        plt.close()

    def test_shap_impact_bars_map_one_to_one_to_rows(self):
        # Stronger alignment guard: give several distinct subcategories impact and verify
        # each bar sits on exactly its own heatmap row (no compaction, no reordering).
        cpp_plot = aa.CPPPlot()
        df_feat = aa.load_features("DOM_GSEC")
        subcats = list(dict.fromkeys(df_feat["subcategory"]))[:14]
        df_feat = df_feat[df_feat["subcategory"].isin(subcats)].copy()
        targets = {subcats[1], subcats[6], subcats[-1]}  # well-separated rows
        df_feat[COL_FEAT_IMPACT_TEST] = [3.0 if s in targets else 0.0
                                         for s in df_feat["subcategory"]]
        fig, hm = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                       col_val="mean_dif", col_imp=COL_FEAT_IMPACT_TEST)
        fig.canvas.draw()
        ylabels = [t.get_text() for t in hm.get_yticklabels()]
        expected_rows = {ylabels.index(t) for t in targets}
        bar_ax = _right_importance_bar(fig, hm)
        assert bar_ax is not None, "right importance bar not found"
        bar_rows = {round(p.get_y() + p.get_height() / 2)
                    for p in bar_ax.patches if getattr(p, "get_width", lambda: 0)() > 0.01}
        assert bar_rows == expected_rows, (sorted(bar_rows), "expected", sorted(expected_rows))
        plt.close()

    def test_importance_bar_labels_inside_bars(self):
        # The cumulative-importance % labels sit INSIDE the bars (right-aligned at each bar
        # tip, white), not outside to the right of the tip. Guard: every % label's right
        # edge is at/inside the bar region (<= the axis max), so it does not spill out past
        # the bar tip. Only bars at/above the threshold are annotated (long enough to hold it).
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, hm = cpp_plot.feature_map(df_feat=df_feat)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        a = _right_importance_bar(fig, hm)  # narrow bar glued to the heatmap's right edge
        assert a is not None, "right importance bar not found"
        checked = 0
        x_max = a.get_xlim()[1]
        for t in a.texts:
            if t.get_text().strip().endswith("%"):
                ext = t.get_window_extent(renderer)
                right = a.transData.inverted().transform((ext.x1, ext.y0))[0]
                assert right <= x_max + 1e-6, f"label spills right past the bar region: {right} > {x_max}"
                checked += 1
        assert checked, "no % importance-bar labels found"
        plt.close()

    def test_default_bars_are_gray_not_signed(self):
        """shap_plot=False keeps the gray cumulative bars and shows no SHAP +/- colors."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, _ = cpp_plot.feature_map(df_feat=df_feat)
        colors = get_bar_facecolors(fig)
        assert _rgba(FEAT_IMP_GRAY) in colors
        assert _rgba(SHAP_POS) not in colors and _rgba(SHAP_NEG) not in colors
        plt.close()

    def test_shap_bars_are_red_and_blue(self):
        """With a mean_dif heatmap, the impact bars stack red (positive) and blue (negative)."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                      col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        colors = get_bar_facecolors(fig)
        assert _rgba(SHAP_POS) in colors and _rgba(SHAP_NEG) in colors
        assert _rgba(FEAT_IMP_GRAY) not in colors
        plt.close()

    def test_shap_bars_are_stacked_one_direction(self):
        """SHAP impact bars are cumulative stacks in ONE direction, not diverging: every
        red (positive) / blue (negative) segment has a non-negative extent -- the right
        per-subcategory bars never extend left of the baseline (no negative width) and the
        top per-position bars never extend down (no negative height). Positive and negative
        feature impacts pile up as stacked segments rather than extending opposite ways.
        (Both colors being present is guarded by test_shap_bars_are_red_and_blue.)"""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()  # alternating +/- impact -> both signs present
        fig, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                      col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        shap_rgba = {_rgba(SHAP_POS), _rgba(SHAP_NEG)}
        widths, heights = [], []
        for ax in fig.axes:
            for p in ax.patches:
                if tuple(round(x, 3) for x in p.get_facecolor()) in shap_rgba:
                    widths.append(round(p.get_width(), 6))
                    heights.append(round(p.get_height(), 6))
        assert widths and heights
        assert all(w >= 0 for w in widths), "subcategory bars must not extend left (negative width)"
        assert all(h >= 0 for h in heights), "position bars must not extend down (negative height)"
        plt.close()

    def test_shap_bars_interleave_by_feature_order_not_grouped(self):
        """Segments stack in feature order (red/blue interleaved), NOT grouped all-positive-
        then-all-negative. Six alternating-sign features in one subcategory must render a
        right bar whose segments alternate along the axis -- i.e. a blue segment precedes a
        later red one, which sign-grouping (all reds, then all blues) could never produce."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat(n=6).reset_index(drop=True)
        df_feat["category"] = df_feat["category"].iloc[0]
        df_feat["subcategory"] = df_feat["subcategory"].iloc[0]
        df_feat[COL_FEAT_IMPACT_TEST] = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]
        fig, hm = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                       col_val=COL_MEAN_DIF_TEST, col_imp=COL_FEAT_IMPACT_TEST)
        fig.canvas.draw()
        bar_ax = _right_importance_bar(fig, hm)
        assert bar_ax is not None, "right importance bar not found"
        code = {_rgba(SHAP_POS): "R", _rgba(SHAP_NEG): "B"}
        segs = sorted((p for p in bar_ax.patches
                       if _rgba(p.get_facecolor()) in code and p.get_width() > 0.01),
                      key=lambda p: p.get_x())
        seq = "".join(code[_rgba(p.get_facecolor())] for p in segs)
        assert seq.count("R") >= 2 and seq.count("B") >= 2, seq
        assert any(seq[k] == "B" and "R" in seq[k + 1:] for k in range(len(seq))), \
            f"segments grouped by sign, not interleaved: {seq}"
        plt.close()

    def test_shap_markers_present(self):
        """Magnitude markers (squares) are still drawn in SHAP mode."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                      col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_FEAT_IMPACT_TEST)
        n_squares = sum(1 for ax in fig.axes for t in ax.texts if t.get_text() == "■")
        assert n_squares > 0
        plt.close()

    def test_shap_with_top_bar(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, add_imp_bar_top=True,
                                      col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        colors = get_bar_facecolors(fig)
        assert _rgba(SHAP_POS) in colors and _rgba(SHAP_NEG) in colors
        plt.close()

    def test_shap_without_top_bar(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, add_imp_bar_top=False,
                                      col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        colors = get_bar_facecolors(fig)
        assert _rgba(SHAP_POS) in colors and _rgba(SHAP_NEG) in colors
        plt.close()

    def test_shap_col_val_mean_dif_name(self):
        """SHAP impact bars combine with a sample-level mean_dif_'name' heatmap."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                       col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_shap_impact_heatmap_switches_bars_off(self):
        """When col_val is a feat_impact column, impact is shown in the heatmap and the
        cumulative-impact bars are switched off (heatmap-only layout)."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        # bars on (mean_dif heatmap) -> more axes than bars off (impact heatmap)
        fig_on, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                         col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_MEAN_DIF_TEST)
        n_on = len(fig_on.axes)
        plt.close()
        fig_off, _ = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                          col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_FEAT_IMPACT_TEST)
        n_off = len(fig_off.axes)
        # No SHAP-colored bars in the bars-off layout
        assert _rgba(SHAP_POS) not in get_bar_facecolors(fig_off)
        assert n_off < n_on
        plt.close()

    # Negative tests
    def test_invalid_shap_plot(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        for shap_plot in ["yes", 1, None, []]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, shap_plot=shap_plot,
                                     col_imp=COL_FEAT_IMPACT_TEST, col_val=COL_FEAT_IMPACT_TEST)
            plt.close()

    def test_shap_rejects_importance_col_imp(self):
        """In SHAP mode col_imp must follow feat_impact_'name', not feat_importance."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                 col_imp=COL_FEAT_IMPORTANCE_TEST, col_val=COL_FEAT_IMPACT_TEST)
        plt.close()

    def test_shap_rejects_plain_col_val(self):
        """In SHAP mode col_val must follow feat_impact_'name' / mean_dif_'name'."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        # Values lacking both 'feat_impact' and 'mean_dif' substrings are rejected
        for col_val in ["abs_auc", "feat_importance"]:
            with pytest.raises(ValueError):
                cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,
                                     col_imp=COL_FEAT_IMPACT_TEST, col_val=col_val)
            plt.close()

    def test_non_shap_rejects_impact_col_imp(self):
        """Without shap_plot, a feat_impact_'name' col_imp is rejected."""
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat_shap()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, shap_plot=False, col_imp=COL_FEAT_IMPACT_TEST)
        plt.close()


class TestFeatureMapSeqCharFill:
    """seq_char_fill: opt-in edge-to-edge residue characters (no whitespace, no overlap)."""

    @staticmethod
    def _seq_kws():
        df = aa.load_dataset(name="DOM_GSEC", n=6)
        seq = df["sequence"][0]
        return dict(tmd_seq=seq[10:30], jmd_n_seq=seq[:10], jmd_c_seq=seq[30:40])

    def _max_seq_fs(self, ax):
        fs = [t.get_fontsize() for t in ax.get_xticklabels(minor=True)]
        fs += [t.get_fontsize() for t in ax.get_xticklabels()]
        return max(fs)

    def test_fill_true_grows_font(self):
        df_feat = aa.load_features(name="DOM_GSEC").head(40)
        cpp = aa.CPPPlot(df_scales=aa.load_scales())
        kws = self._seq_kws()
        _, ax_off = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=False, **kws)
        off = self._max_seq_fs(ax_off); plt.close("all")
        _, ax_on = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=True, **kws)
        on = self._max_seq_fs(ax_on); plt.close("all")
        assert on >= off  # fill never shrinks; grows toward touching characters

    def test_fill_default_is_true(self):
        # Default is edge-to-edge fill (seq_char_fill=True): default matches fill=True,
        # and is at least as large as the explicit no-fill spacing.
        df_feat = aa.load_features(name="DOM_GSEC").head(40)
        cpp = aa.CPPPlot(df_scales=aa.load_scales())
        kws = self._seq_kws()
        _, ax_def = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, **kws)
        def_fs = self._max_seq_fs(ax_def); plt.close("all")
        _, ax_on = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=True, **kws)
        on_fs = self._max_seq_fs(ax_on); plt.close("all")
        _, ax_off = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=False, **kws)
        off_fs = self._max_seq_fs(ax_off); plt.close("all")
        assert def_fs == on_fs and def_fs >= off_fs

    def test_fill_draws_seamless_full_width_cells(self):
        # seq_char_fill=True paints one full-width (1.0 data unit) colored cell per residue
        # behind the letters, so the sequence band is gap-free and aligned with the heatmap
        # columns. The legacy glyph-sized text bbox left hairline gaps between narrow residues.
        from matplotlib.patches import Rectangle
        df_feat = aa.load_features(name="DOM_GSEC").head(40)
        cpp = aa.CPPPlot(df_scales=aa.load_scales())
        kws = self._seq_kws()
        n_res = sum(len(kws[k]) for k in ("jmd_n_seq", "tmd_seq", "jmd_c_seq"))

        def _full_cells(ax):
            ax.figure.canvas.draw()
            return [p for p in ax.patches if isinstance(p, Rectangle) and abs(p.get_width() - 1.0) < 1e-6]

        _, ax_on = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=True, **kws)
        cells = _full_cells(ax_on)
        lefts = sorted(round(p.get_x(), 3) for p in cells)
        plt.close("all")
        assert len(cells) == n_res                          # one seamless cell per residue
        assert lefts == [float(i) for i in range(n_res)]    # contiguous 0..n-1 -> no gaps

        _, ax_off = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=False, **kws)
        cells_off = _full_cells(ax_off)
        plt.close("all")
        assert len(cells_off) == 0                          # fill=False keeps the legacy glyph bbox

    def test_fill_cell_flush_to_heatmap_edge(self):
        # The colored cell's inner edge snaps exactly onto a heatmap border (a y-limit), so there
        # is no gap between the sequence band and the grid above it. (The band's far edge is a slim
        # margin past the letters; that margin is a visual constant, not asserted here because the
        # font's bounding box includes empty descender space below the caps.)
        from matplotlib.patches import Rectangle
        df_feat = aa.load_features(name="DOM_GSEC").head(40)
        cpp = aa.CPPPlot(df_scales=aa.load_scales())
        _, ax = cpp.feature_map(df_feat=df_feat, add_imp_bar_top=False, seq_char_fill=True, **self._seq_kws())
        ax.figure.canvas.draw()
        cells = [p for p in ax.patches if isinstance(p, Rectangle) and abs(p.get_width() - 1.0) < 1e-6]
        ys = [p.get_y() for p in cells] + [p.get_y() + p.get_height() for p in cells]
        cy0, cy1 = min(ys), max(ys)
        ylim = ax.get_ylim()
        plt.close("all")
        flush = min(min(abs(cy0 - e), abs(cy1 - e)) for e in ylim)
        assert flush < 1e-6                                       # one band edge is flush to the grid

    def test_fill_bad_type_raises(self):
        df_feat = aa.load_features(name="DOM_GSEC").head(40)
        cpp = aa.CPPPlot(df_scales=aa.load_scales())
        with pytest.raises(ValueError):
            cpp.feature_map(df_feat=df_feat, seq_char_fill="yes", **self._seq_kws())
