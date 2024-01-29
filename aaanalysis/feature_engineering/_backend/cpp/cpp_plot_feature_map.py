"""
This is a script for the backend of the CPPPlot.feature_map method.
"""

import statistics
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis.utils as ut


from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions
from ._utils_cpp_plot_map import plot_heatmap_


# I Helper Functions
def add_feature_title_single_fig(y=None, fontsize_title=None, pad_factor=2.0):
    """Add title for feature onto single figure"""
    f_space = lambda x: " "*x
    plt.text(0, y, "Scale (subcategory)" + f_space(3), size=fontsize_title, weight="bold", ha="right")
    plt.text(0, y, f_space(3) + "Positions", size=fontsize_title, weight="bold", ha="left")
    plt.text(0, y * pad_factor, "Feature", size=fontsize_title + 1, weight="bold", ha="center")
    plt.text(0, y, "+", size=fontsize_title, weight="bold", ha="center")


def plot_feat_importance_bars(ax=None, df=None, top_n=5, top_pcp=None, sum_imp=None, show_sum_pcp=True,
                              label=ut.LABEL_FEAT_IMPORT, col_imp=None,
                              legendsize=12, labelsize=12, top_pcp_size=12, top_pcp_weight="bold", titlesize=12):
    """"""
    plt.sca(ax)
    df["pos_start"] = [int(x.split(",")[0]) if "," in x else int(x) for x in df[ut.COL_POSITION]]
    df["subcat_lower"] = [x.lower() for x in df[ut.COL_SUBCAT]]
    df = df.sort_values(by=[ut.COL_CAT, "subcat_lower", "pos_start", col_imp],
                        ascending=[True, True, True, False])
    df_imp = df[[ut.COL_SUBCAT, col_imp]].groupby(by=ut.COL_SUBCAT).sum()
    dict_imp = dict(zip(df_imp.index, df_imp[col_imp]))
    # Case non-sensitive sorted list of subcat
    list_subcat = list(
        df.sort_values(by=[ut.COL_CAT, ut.COL_SUBCAT], key=lambda x: x.str.lower())[ut.COL_SUBCAT].drop_duplicates())
    list_imp = [dict_imp[x] for x in list_subcat]
    ax.barh(list_subcat, list_imp, color="tab:gray", edgecolor="white", align="edge")
    plt.xlabel(label, size=titlesize, weight="bold", ha="center")
    ax.xaxis.set_label_position('top')
    # Add annotations
    top_pcp = max(list_imp)/2 if top_pcp is None else top_pcp
    for i, val in enumerate(list_imp):
        if val >= top_pcp:
            plt.text(val, i+0.45, f"{round(val, 1)}% ", va="center", ha="right",
                     weight=top_pcp_weight, color="white", size=top_pcp_size)
    # Adjust ticks
    ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    ax.tick_params(axis='x', which='both', labelsize=labelsize, pad=-1)
    sns.despine(ax=ax, bottom=True, top=False)
    plt.xlim(0, max(list_imp))
    plt.tight_layout()


# Add importance map (for feature map)
def _add_importance_map(ax=None, df_feat=None, df_cat=None, start=None, args_len=None, col_cat=None, col_imp=None):
    """"""
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat,
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=True)
    _df = pd.melt(df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, "position", col_imp]
    _list_sub_cat = _df[ut.COL_SUBCAT].unique()
    for i, sub_cat in enumerate(_list_sub_cat):
        _dff = _df[_df[ut.COL_SUBCAT] == sub_cat]
        for pos, val in enumerate(_dff[col_imp]):
            _symbol = "■"
            color = "black"
            size = 7 if val >= 1 else (5 if val >= 0.5 else 3)
            _args_symbol = dict(ha="center", va="center", color=color, size=size)
            if val >= 0.2:
                ax.text(pos + 0.5, i + 0.5, _symbol, **_args_symbol)


def _add_importance_map_legend(ax=None, y=None, x=None, fontsize=None):
    """"""
    # Now create a list of Patch instances for the second legend
    # Define the sizes for the legend markers
    list_labels = ["  >0.2%", "  >0.5%", "  >1.0%"]
    list_sizes = [3, 5, 7]
    # Create the legend handles manually
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label=label, markersize=size, markerfacecolor='black', linewidth=0)
        for label, size in zip(list_labels, list_sizes)]
    # Create the second legend
    second_legend = ax.legend(handles=legend_handles, title="Feature importance",
                              loc='lower left',
                              bbox_to_anchor=(x, y), frameon=False, fontsize=fontsize,
                              labelspacing=0,
                              columnspacing=0, handletextpad=0, handlelength=0,
                              borderpad=0)
    # Add the second legend to the plot
    ax.add_artist(second_legend)


# II Main Functions
# TODO check if fontsize_labels, fontsize_text and cbar_ticksize is needed
def plot_feature_map(df_feat=None, df_cat=None,
                     col_cat="subcategory", col_val="mean_dif", col_imp="feat_importance",
                     normalize=False,
                     name_test="TEST", name_ref="REF",
                     figsize=(8, 8),
                     start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                     tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None, fontsize_labels=11, fontsize_text=11,
                     fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                     add_xticks_pos=False,
                     grid_linewidth=0.01, grid_linecolor=None,
                     border_linewidth=2,
                     facecolor_dark=False, vmin=None, vmax=None,
                     cmap=None, cmap_n_colors=None, cbar_kws=None,
                     cbar_ticksize=11, cbar_pct=True,
                     dict_color=None, legend_kws=None,
                     xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                     ytick_size=None):
    """
    Plot a feature map of the selected value column with scale information (y-axis) versus sequence position (x-axis).
    """
    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    # Plot
    width, height = figsize
    n_subcat = len(set(df_feat["subcategory"]))
    fig, axes = plt.subplots(1, 2, sharey=True,
                             gridspec_kw={'width_ratios': [6, min(1*height/width, 1)], "wspace": 0},
                             figsize=figsize, layout="constrained")
    plot_feat_importance_bars(ax=axes[1], df=df_feat.copy(), legendsize=fontsize_text, titlesize=fontsize_text - 2,
                              labelsize=fontsize_text, top_pcp_size=fontsize_text - 5,
                              top_pcp_weight="bold", col_imp=col_imp,
                              label="Cumulative feature\n  importance [%]")

    # Set cat legend arguments
    _legend_kws = dict(fontsize=12, fontsize_title=12, y=-0.06, x=-0.1)
    if legend_kws is not None:
        _legend_kws.update(legend_kws)
    # Create cbar axes: [left, bottom, width, height]
    cbar_ax_pos = (0.5, 0.02, 0.2, 0.015)
    cbar_ax = fig.add_axes(cbar_ax_pos)
    _cbar_kws = dict(ticksize=cbar_ticksize, label=f"Feature value\n{name_test} - {name_ref}")
    if cbar_kws is not None:
        _cbar_kws.update(cbar_kws)
    ax = plot_heatmap_(df_feat=df_feat, df_cat=df_cat,
                       col_cat=col_cat, col_val=col_val, normalize=normalize,
                       ax=axes[0], figsize=figsize,
                       start=start, **args_len, **args_seq,
                       **args_part_color, **args_seq_color,
                       seq_size=seq_size, fontsize_labels=fontsize_labels,
                       fontsize_tmd_jmd=fontsize_tmd_jmd, weight_tmd_jmd=weight_tmd_jmd,
                       add_xticks_pos=add_xticks_pos,
                       grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                       border_linewidth=border_linewidth,
                       facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                       cmap=cmap, cmap_n_colors=cmap_n_colors,
                       cbar_ax=cbar_ax, cbar_pct=cbar_pct, cbar_kws=_cbar_kws,
                       dict_color=dict_color, legend_kws=_legend_kws,
                       **args_xtick, ytick_size=ytick_size)

    # Add importance map
    _add_importance_map(df_feat=df_feat, df_cat=df_cat, col_cat=col_cat, col_imp=col_imp,
                        ax=ax, start=start, args_len=args_len)
    _add_importance_map_legend(fontsize=fontsize_text, ax=cbar_ax, y=0, x=1.5)
    # Add feature title
    add_feature_title_single_fig(y=-n_subcat / 80, fontsize_title=fontsize_text, pad_factor=3.2)
    fig.tight_layout()#pad=3.0)
    plt.subplots_adjust(wspace=0, bottom=0.15, top=0.92)#, left=0.3)
    plt.tight_layout()
    return ax
