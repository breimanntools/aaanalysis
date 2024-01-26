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

import aaanalysis
import aaanalysis.utils as ut

from .cpp_plot_heatmap import plot_heatmap
from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions


# I Helper Functions
def add_feature_title_single_fig(y=None, fontsize_title=None, pad_factor=2.0):
    """Add title for feature onto single figure"""
    f_space = lambda x: " "*x
    plt.text(0, y, "Scale (subcategory)" + f_space(3), size=fontsize_title, weight="bold", ha="right")
    plt.text(0, y, f_space(3) + "Positions", size=fontsize_title, weight="bold", ha="left")
    plt.text(0, y * pad_factor, "Feature", size=fontsize_title + 1, weight="bold", ha="center")
    plt.text(0, y, "+", size=fontsize_title, weight="bold", ha="center")


def _bars(ax=None, df=None, top_n=5, top_pcp=None, sum_imp=None, show_sum_pcp=True, label=ut.LABEL_FEAT_IMPORT,
          legendsize=12, labelsize=12, top_pcp_size=12, top_pcp_weight="bold", titlesize=12):
    """"""
    plt.sca(ax)
    df["pos_start"] = [int(x.split(",")[0]) if "," in x else int(x) for x in df[ut.COL_POSITION]]
    df["subcat_lower"] = [x.lower() for x in df[ut.COL_SUBCAT]]
    df = df.sort_values(by=[ut.COL_CAT, "subcat_lower", "pos_start", ut.COL_FEAT_IMPORT],
                        ascending=[True, True, True, False])
    df_imp = df[[ut.COL_SUBCAT, ut.COL_FEAT_IMPORT]].groupby(by=ut.COL_SUBCAT).sum()
    dict_imp = dict(zip(df_imp.index, df_imp[ut.COL_FEAT_IMPORT]))
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


def _add_importance_map_legend(ax=None, y=None, x=None, fontsize=None, tick_fontsize=None, title_weight="bold"):
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
    # You can position it using the loc and bbox_to_anchor parameters
    second_legend = ax.legend(handles=legend_handles, title="Feature importance", loc='lower left',
                              bbox_to_anchor=(x, y), frameon=False, fontsize=fontsize,
                              labelspacing=0,
                              columnspacing=0, handletextpad=0, handlelength=0,
                              borderpad=0)

    # Add the second legend to the plot
    ax.add_artist(second_legend)


# II Main Functions
def plot_feature_map(df_feat=None, df_cat=None, y="subcategory", col_value="mean_dif", value_type="mean", normalize=False,
                     figsize=(8, 8), ax=None, dict_color=None,
                     vmin=None, vmax=None, grid_on=True, cmap="RdBu_r", cmap_n_colors=None, cbar_kws=None,
                     cbar_pos=(0.5, 0.01, 0.2, 0.015), legend_xy=(-0.5, -0.04),  # TODO add
                     facecolor_dark=False, add_jmd_tmd=True,
                     tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1,
                     tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None, linecolor=None,
                     tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None, fontsize_tmd_jmd=None, add_xticks_pos=False, xtick_size=11.0, xtick_width=2.0,
                     xtick_length=5.0, ytick_size=None, add_legend_cat=True, legend_kws=None, cbar_pct=True,
                     name_test="TEST", name_ref="REF",
                     fontsize_text=11, cbar_ticksize=11):
    """
    Plot a feature map of the selected value column with scale information (y-axis) versus sequence position (x-axis).
    """
    # Group arguments
    # Plot
    ####
    width, height = figsize
    n_subcat = len(set(df_feat["subcategory"]))
    fig, axes = plt.subplots(1, 2, sharey=True,
                             gridspec_kw={'width_ratios': [6, min(1*height/width, 1)], "wspace": 0},
                             figsize=figsize, layout="constrained")
    _bars(ax=axes[1], df=df_feat.copy(), legendsize=fontsize_text, titlesize=fontsize_text - 2,
          labelsize=fontsize_text, top_pcp_size=fontsize_text - 5, top_pcp_weight="bold",
          label="Cumulative feature\n  importance [%]")
    info_weight = "normal"
    _label = f"Feature value\n{name_test} - {name_ref}"
    cbar_kws = dict(use_gridspec=False,
                    orientation="horizontal",
                    ticksize=cbar_ticksize,
                    label=_label,
                    pad=0,
                    panchor=(0, 0))
    # TODO!! Get args to adjust cat, cbar and feat importance together
    y_pos = min(1/n_subcat, 0.1)
    cbar_pos = (0.51, 0, 0.2, 0.015)
    legend_pos = (0, -y_pos) #-1/n_subcat)
    legend_kws = dict(fontsize=fontsize_text, loc=9,
                      title=ut.LABEL_SCALE_CAT,
                      title_fontproperties={'weight': info_weight, "size": 1 + fontsize_text},
                      bbox_to_anchor=legend_pos)
    ####
    cbar_ax_pos = (0.5, 0.01, 0.2, 0.015)
    fig = plt.gcf()
    cbar_ax = fig.add_axes(cbar_ax_pos)
    ax = plot_heatmap(df_feat=df_feat, df_cat=df_cat, ax=axes[0],
                      col_cat=y, col_value=col_value, value_type=value_type, normalize=normalize,
                      figsize=figsize, dict_color=dict_color,
                      vmin=vmin, vmax=vmax, grid_on=grid_on,
                      facecolor_dark=facecolor_dark, add_jmd_tmd=add_jmd_tmd,
                      tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start,
                      tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq, linecolor=linecolor,
                      tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                      jmd_seq_color=jmd_seq_color, add_importance_map=True,
                      seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd, add_xticks_pos=add_xticks_pos,
                      xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length,
                      ytick_size=ytick_size, add_legend_cat=add_legend_cat, legend_kws=legend_kws,
                      cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws, cbar_ax=cbar_ax,  #cbar_ax_pos=cbar_pos,
                      cbar_pct=cbar_pct)
    # Add importance map
    _add_importance_map_legend(fontsize=fontsize_text,
                               ax=cbar_ax, y=0, x=1.5,
                               #ax=ax, y=n_subcat+3, x=plt.xlim()[1], pad_factor=1,
                               tick_fontsize=cbar_ticksize,
                               title_weight=info_weight)
    add_feature_title_single_fig(y=-n_subcat / 80, fontsize_title=fontsize_text, pad_factor=3.2)
    fig.tight_layout()#pad=3.0)
    plt.subplots_adjust(wspace=0, bottom=0.15, top=0.92)#, left=0.3)
    plt.tight_layout()
    return ax

"""
    n = 100
    label_fontsize = 11
    name_test = "TEST"
    name_ref = "REF"
    tick_fontsize = 11
    df_feat = df_feat.copy().sort_values(ut.COL_FEAT_IMPORT, ascending=False).head(n)
    n_subcat = len(set(df_feat["subcategory"]))

    fig, axes = plt.subplots(1, 2, sharey=True,
                             gridspec_kw={'width_ratios': [6, 1], "wspace": 0},
                             figsize=figsize, layout="constrained")
    _bars(ax=axes[1], df=df_feat.copy(), legendsize=label_fontsize, titlesize=label_fontsize - 2,
          labelsize=label_fontsize, top_pcp_size=label_fontsize - 5, top_pcp_weight="bold",
          label="Cumulative feature\n  importance [%]")


    info_weight = "normal"
    width, height = figsize
    bbox_x = 0.5
    bbox_y = height * 0.0225 - 0.17 #0.2175
    bbox_y = 0.01
    _label = f"Feature value\n{name_test} - {name_ref}"
    cbar_kws = dict(use_gridspec=False,
                    orientation="horizontal",
                    ticksize=tick_fontsize,
                    label=_label,
                    pad=2,
                    panchor=(0, 0))
    # TODO!! Get args to adjust cat, cbar and feat importance together
    legend_kws = dict(fontsize=tick_fontsize,
                      title=ut.LABEL_SCALE_CAT,
                      title_fontproperties={'weight': info_weight, "size": label_fontsize},
                      bbox_to_anchor=(-0.5, -0.04), loc=2)
    ####
    ax = plot_heatmap(df_feat=df_feat, df_cat=df_cat, ax=axes[0],
                      y=y, col_value=col_value, value_type=value_type, normalize=normalize,
                      figsize=figsize, dict_color=dict_color,
                      vmin=vmin, vmax=vmax, grid_on=grid_on,
                      facecolor_dark=facecolor_dark, add_jmd_tmd=add_jmd_tmd,
                      tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start,
                      tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq, linecolor=linecolor,
                      tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                      jmd_seq_color=jmd_seq_color, add_importance_map=True,
                      seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd, xticks_pos=xticks_pos,
                      xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length,
                      ytick_size=ytick_size, add_legend_cat=add_legend_cat, legend_kws=legend_kws,
                      cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws, cbar_ax_pos=cbar_ax_pos,
                      cbar_pct=cbar_pct)
    #cb = ax.collections[0].colorbar
    #cb.set_label(label=cbar_kws["label"], weight=info_weight, size=label_fontsize - 1)
    #cb.ax.xaxis.set_ticks_position('top')
    #cb.ax.xaxis.set_label_position('top')
    # Add importance map
    _add_importance_map_legend(ax=ax, x=plt.xlim()[1],
                               y=n_subcat + 3, fontsize=label_fontsize,
                               tick_fontsize=tick_fontsize,
                               title_weight=info_weight, pad_factor=0.9)
    add_feature_title(y=-n_subcat / 80, fontsize_title=label_fontsize, pad_factor=3.2)
    fig.tight_layout()#pad=3.0)
    plt.subplots_adjust(wspace=0, bottom=0.15, top=0.92)#, left=0.3)
    plt.tight_layout()
"""
