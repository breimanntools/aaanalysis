"""
This is a script for the backend of the CPPPlot.feature_map method.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPartPositions
from ._utils_cpp_plot_map import plot_heatmap_
from ._utils_cpp_plot import get_sorted_list_cat_


# I Helper functions


# Add feature importance plot elements
def plot_feat_importance_bars_subcat(ax=None,
                                     df_feat=None,
                                     df_cat=None,
                                     col_cat=None,
                                     col_imp=None,
                                     annotation_th=None,
                                     label=None,
                                     fontsize_label=12,
                                     fontsize_annotations=11,
                                     fontsize_imp_bar=9,
                                     ha="left",
                                     position=(1, 0),
                                     multialignment="right",
                                     weight_annotation="bold"):
    """Display a vertical bar plot (y-axis) for feature importance sorted by categories"""
    # Get feature importance per scale class
    df_imp = df_feat[[col_cat, col_imp]].groupby(by=col_cat).sum()
    dict_imp = dict(zip(df_imp.index, df_imp[col_imp]))
    list_cat = get_sorted_list_cat_(df_cat=df_cat,
                                    list_cat=df_feat[col_cat].to_list(),
                                    col_cat=col_cat)
    list_imp = [dict_imp[x] for x in list_cat]

    # Plot bars
    ax.barh(list_cat, list_imp, color=ut.COLOR_FEAT_IMP, edgecolor=None, align="edge")
    sns.despine(ax=ax, bottom=True, top=False, left=False)

    # Add label
    ax.set_xlabel(label, size=fontsize_label, weight="bold",
                  ha=ha, position=position, multialignment=multialignment)
    ax.xaxis.set_label_position("top")
    # Add annotations
    v_max = int(np.ceil(max(list_imp)))
    annotation_th = v_max / 2 if annotation_th is None else annotation_th
    for i, val in enumerate(list_imp):
        if val >= annotation_th:
            ax.text(val, i + 0.45, f"{round(val, 1)}% ",
                    va="center", ha="right",
                    weight=weight_annotation,
                    color="white",
                    size=fontsize_imp_bar)

    # Adjust ticks
    ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    ax.tick_params(axis='x', which='both', labelsize=fontsize_annotations,
                   pad=0, length=3)
    for label in ax.get_yticklabels():
        label.set_visible(False)

    ax.set_xlim(0, v_max)
    # Adjust plot size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.tight_layout()


def plot_feat_importance_bars_pos(ax=None,
                                  df_feat=None,
                                  df_cat=None,
                                  col_cat=None,
                                  col_imp=None,
                                  start=1,
                                  tmd_len=20,
                                  jmd_n_len=10,
                                  jmd_c_len=10):
    """Display a horizontal (x-axis) bar plot for feature importance per position"""
    # Get feature importance per position class
    pp = PlotPartPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(),
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=False)
    # Plot bars
    list_imp = list(df_pos.sum())
    x_ticks = list(range(0, len(list_imp)))
    ax.bar(x_ticks, list_imp,
           color=ut.COLOR_FEAT_IMP, edgecolor=None, align="edge")
    sns.despine(ax=ax, bottom=False, top=True, left=True, right=False)
    # Adjust ticks
    ax.set_xticks([])
    v_max = int(np.ceil(max(list_imp)))
    ax.set_ylim(0, v_max)
    ax.tick_params(axis='y', length=3, pad=1)


def add_feat_importance_map(ax=None, df_feat=None, df_cat=None,
                            col_cat=None, col_imp=None,
                            imp_ths=(0.2, 0.5, 1),
                            imp_marker_size=(3, 5.5, 8),
                            start=None, args_len=None):
    """Overlay feature importance symbols on the heatmap based on the importance values."""
    th1, th2, th3 = imp_ths
    ms1, ms2, ms3 = imp_marker_size
    pp = PlotPartPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat,
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=False)
    _df = pd.melt(df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, ut.COL_POSITION, col_imp]
    _list_sub_cat = _df[ut.COL_SUBCAT].unique()
    for i, sub_cat in enumerate(_list_sub_cat):
        _dff = _df[_df[ut.COL_SUBCAT] == sub_cat]
        for pos, val in enumerate(_dff[col_imp]):
            _symbol = "■"
            color = "black"
            size = ms3 if val >= th3 else (ms2 if val >= th2 else ms1)
            _args_symbol = dict(ha="center", va="center", color=color, size=size)
            if val >= th1:
                ax.text(pos + 0.5, i + 0.5, _symbol, **_args_symbol)


def add_feat_importance_legend(ax=None,
                               legend_imp_xy=None,
                               imp_ths=(0.2, 0.5, 1),
                               label=None,
                               fontsize_title=None,
                               fontsize_annotations=None):
    """Add a custom legend indicating the meaning of feature importance symbols."""
    # Define the sizes for the legend markers
    list_labels = [f"  >{float(x)}%" for x in imp_ths]
    list_imp_marker_sizes = [3, 5, 7]

    # Create the legend handles manually
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label=label,
                   markersize=size, markerfacecolor='black', linewidth=0)
        for label, size in zip(list_labels, list_imp_marker_sizes)]

    # Create the second legend
    ax.legend(handles=legend_handles,
              title=label,
              loc='lower left',
              bbox_to_anchor=legend_imp_xy,
              frameon=False,
              title_fontsize=fontsize_title,
              fontsize=fontsize_annotations,
              labelspacing=0.25,
              columnspacing=0, handletextpad=0, handlelength=0,
              borderpad=0)


# II Main Functions
# TODO stacked bar charts for SHAP (later)
def plot_feature_map(df_feat=None, df_cat=None,
                     col_cat="subcategory", col_val="mean_dif", col_imp="feat_importance",
                     name_test="TEST", name_ref="REF",
                     figsize=(8, 8),
                     add_imp_bar_top=True,
                     imp_bar_th=None,
                     imp_bar_label_type="long",
                     imp_ths=(0.2, 0.5, 1), imp_marker_sizes=(3, 5, 8),
                     start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                     tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None,
                     fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                     fontsize_titles=11, fontsize_labels=12,
                     fontsize_annotations=11,
                     fontsize_imp_bar=9,
                     add_xticks_pos=False,
                     grid_linewidth=0.01, grid_linecolor=None,
                     border_linewidth=2,
                     facecolor_dark=False, vmin=None, vmax=None,
                     cmap=None, cmap_n_colors=101,
                     cbar_pct=True, cbar_kws=None, cbar_xywh=(0.5, None, 0.2, None),
                     dict_color=None, legend_kws=None, legend_xy=(-0.1, -0.01),
                     legend_imp_xy=(1.25, 0),
                     xtick_size=11.0, xtick_width=2.0, xtick_length=5.0):
    """Create a comprehensive feature map with a heatmap, feature importance bars, and custom legends."""
    # Get fontsize
    pe = PlotElements()
    fs = ut.plot_gco()
    fs_titles = fs-1 if fontsize_titles is None else fontsize_titles
    fs_labels = fs if fontsize_labels is None else fontsize_labels
    fs_annotations = fs-1 if fontsize_annotations is None else fontsize_annotations
    fs_imp_bar = fs-3 if fontsize_imp_bar is None else fontsize_imp_bar

    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_fs = dict(seq_size=seq_size,
                   fontsize_labels=fs_labels,
                   fontsize_tmd_jmd=fontsize_tmd_jmd)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

    # Plot
    width, height = figsize
    width_ratio = [6, min(1*height/width, 1)]
    height_ratio = (width_ratio[1], width_ratio[0])
    gridspc_kw = {'width_ratios': width_ratio, "wspace": 0, "hspace": 0}
    if add_imp_bar_top:
        gridspc_kw["height_ratios"] = height_ratio
    fig, axes = plt.subplots(figsize=figsize,
                             nrows=2 if add_imp_bar_top else 1,
                             ncols=2,
                             gridspec_kw=gridspc_kw,
                             layout="constrained")
    if add_imp_bar_top:
        ax_hm, ax_bt, ax_br, ax_empty = axes[1, 0], axes[0, 0], axes[1, 1], axes[0, 1]
        ax_hm.sharex(ax_bt)
    else:
        ax_hm, ax_br = axes[0], axes[1]
    ax_hm.sharey(ax_br)
    # Set colorbar and legend arguments
    label_cbar = f"Feature value\n{name_test} - {name_ref}"
    _cbar_kws, cbar_ax = pe.adjust_cbar_kws(fig=fig,
                                            cbar_kws=cbar_kws,
                                            cbar_xywh=cbar_xywh,
                                            label=label_cbar,
                                            fontsize_labels=fs_labels)

    n_cat = len(set(df_feat[ut.COL_CAT]))
    _legend_kws = pe.adjust_cat_legend_kws(legend_kws=legend_kws,
                                           n_cat=n_cat,
                                           legend_xy=legend_xy,
                                           fontsize_labels=fs_labels)
    # Plot feat importance bars
    label_imp_bar = "Cumulative\nfeature\nimportance" if add_imp_bar_top else "Cumulative feature\nimportance [%]"
    fs_label_bar = fs_titles - 1 if add_imp_bar_top else fs_titles
    ha_bar = "left" if add_imp_bar_top else "right"
    position_bar = (0.1, 0) if add_imp_bar_top else (1, 0)
    multialignment_bar = "center" if add_imp_bar_top else "right"
    if add_imp_bar_top and imp_bar_label_type != "long":
        label_imp_bar = "FI" if imp_bar_label_type == "short" else None
        position_bar = (0.5, 0)
    plot_feat_importance_bars_subcat(ax=ax_br,
                                     df_feat=df_feat.copy(),
                                     df_cat=df_cat.copy(),
                                     col_imp=col_imp,
                                     col_cat=col_cat,
                                     label=label_imp_bar,
                                     annotation_th=imp_bar_th,
                                     fontsize_label=fs_label_bar,
                                     fontsize_imp_bar=fs_imp_bar,
                                     fontsize_annotations=fontsize_annotations,
                                     ha=ha_bar,
                                     position=position_bar,
                                     multialignment=multialignment_bar,
                                     weight_annotation="bold")
    show_only_max = add_imp_bar_top != "long"
    args_ticks_0 = dict(show_zero=False, show_only_max=show_only_max, precision=1)
    ut.ticks_0(ax_br, **args_ticks_0)

    if add_imp_bar_top:
        plot_feat_importance_bars_pos(ax=ax_bt,
                                      df_feat=df_feat.copy(),
                                      df_cat=df_cat.copy(),
                                      col_imp=col_imp,
                                      col_cat=col_cat,
                                      start=start,
                                      **args_len)
        ut.ticks_0(ax_bt, axis="y", **args_ticks_0)

    # Plot heatmap
    plot_heatmap_(df_feat=df_feat.copy(), df_cat=df_cat,
                  col_cat=col_cat, col_val=col_val,
                  ax=ax_hm, figsize=figsize,
                  start=start, **args_len, **args_seq,
                  **args_part_color, **args_seq_color,
                  **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                  add_xticks_pos=add_xticks_pos,
                  grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                  border_linewidth=border_linewidth,
                  facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                  cmap=cmap, cmap_n_colors=cmap_n_colors,
                  cbar_ax=cbar_ax, cbar_pct=cbar_pct, cbar_kws=_cbar_kws,
                  dict_color=dict_color, legend_kws=_legend_kws,
                  **args_xtick)
    # Add feature position title
    sns.despine(ax=ax_hm, top=True, left=True, right=True, bottom=True)
    if add_imp_bar_top:
        ax_hm.set_title("Scale (subcategory)  ", x=0, ha="right", weight="bold", fontsize=fs_titles)
    else:
        ax_hm.set_title(ut.LABEL_FEAT_POS, x=0, weight="bold", fontsize=fs_titles)
    # Add feature importance map
    add_feat_importance_map(df_feat=df_feat, df_cat=df_cat,
                            col_cat=col_cat, col_imp=col_imp,
                            imp_ths=imp_ths,
                            imp_marker_size=imp_marker_sizes,
                            ax=ax_hm, start=start, args_len=args_len)

    legend_imp_xy_default = (1.25, 0)
    _legend_imp_xy = ut.adjust_tuple_elements(tuple_in=legend_imp_xy,
                                              tuple_default=legend_imp_xy_default)
    label_feat_imp = "Feature importance"
    add_feat_importance_legend(ax=cbar_ax,
                               imp_ths=imp_ths,
                               legend_imp_xy=_legend_imp_xy,
                               label=label_feat_imp,
                               fontsize_title=fs_labels,
                               fontsize_annotations=fs_annotations)
    if add_imp_bar_top:
        ax_empty.axis("off")
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.sca(ax_hm)
    ax = ax_hm
    return fig, ax
