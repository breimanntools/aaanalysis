"""
This is a script for the backend of the CPPPlot.feature_map method.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import aaanalysis.utils as ut

from ._utils_cpp_plot_positions import PlotPositions
from ._utils_cpp_plot_map import plot_heatmap_


# I Helper Functions
def _adjust_fontsize(fontsize_labels=None, fontsize_annotations=None):
    """Adjust font size for labels and annotations to defaults if not specified."""
    fs = ut.plot_gco()
    fontsize_labels = fs if fontsize_labels is None else fontsize_labels
    fontsize_annotations = fs - 2 if fontsize_annotations is None else fontsize_annotations
    return fontsize_labels, fontsize_annotations


def _adjust_legend_kws(legend_kws=None, n_cat=None,
                       legend_xy=None, fontsize_labels=None):
    """Optimize legend position and appearance based on the number of categories and provided keywords."""
    n_cols = 2 if legend_kws is None else legend_kws.get("n_cols", 2)
    n_rows = np.floor(n_cat / n_cols)
    if legend_xy is not None:
        x, y = legend_xy
        title = ut.LABEL_SCALE_CAT
    else:
        x, y = -0.1, -0.01
        str_space = "\n" * int((6-n_rows))
        title = f"{str_space}{ut.LABEL_SCALE_CAT}"
    _legend_kws = dict(fontsize=fontsize_labels,
                       fontsize_title=fontsize_labels,
                       n_cols=n_cols,
                       title=title,
                       x=x, y=y)
    if legend_kws is not None:
        _legend_kws.update(legend_kws)
    return _legend_kws


def _adjust_cbar_kws(fig=None, cbar_kws=None, cbar_xywh=None,
                     name_test=None, name_ref=None,
                     fontsize_labels=None):
    """Set color bar position, appearance, and label with default or provided keywords."""
    # Use default cbar position and size if cbar_xywh is not provided
    default_cbar_ax_pos = (0.5, 0.01, 0.2, 0.015)
    cbar_ax_pos = cbar_xywh if cbar_xywh is not None else default_cbar_ax_pos

    # Create cbar axes: [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbar_ax_pos)
    _cbar_kws = dict(ticksize=fontsize_labels,
                     label=f"Feature value\n{name_test} - {name_ref}")
    if cbar_kws is not None:
        _cbar_kws.update(cbar_kws)
    return _cbar_kws, cbar_ax


# Add feature importance plot elements
def plot_feat_importance_bars(ax=None, df=None, top_pcp=None,
                              label=ut.LABEL_FEAT_IMPORT, col_imp=None,
                              titlesize=12, labelsize=12, top_pcp_size=12,
                              top_pcp_weight="bold"):
    """Display a horizontal bar plot for feature importance sorted by categories."""
    col_start = "pos_start"
    col_subcat_lower = "subcat_lower"
    plt.sca(ax)
    df[col_start] = [int(x.split(",")[0]) if "," in x else int(x) for x in df[ut.COL_POSITION]]
    df[col_subcat_lower] = [x.lower() for x in df[ut.COL_SUBCAT]]
    df = df.sort_values(by=[ut.COL_CAT, col_subcat_lower, col_start, col_imp],
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
            plt.text(val, i+0.45, f"{round(val, 1)}% ",
                     va="center", ha="right",
                     weight=top_pcp_weight,
                     color="white",
                     size=top_pcp_size)
    # Adjust ticks
    ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    ax.tick_params(axis='x', which='both', labelsize=labelsize, pad=-1)
    sns.despine(ax=ax, bottom=True, top=False)
    plt.xlim(0, max(list_imp))
    plt.tight_layout()


# TODO adjust 3 th by args
def add_feat_importance_map(ax=None, df_feat=None, df_cat=None, start=None, args_len=None,
                            col_cat=None, col_imp=None, normalize=False):
    """Overlay feature importance symbols on the heatmap based on the importance values."""
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat,
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=normalize)
    _df = pd.melt(df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, ut.COL_POSITION, col_imp]
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


def add_feat_importance_legend(ax=None, legend_imp_xy=None, fontsize_title=None, fontsize=None):
    """Add a custom legend indicating the meaning of feature importance symbols."""
    if legend_imp_xy is not None:
        x, y = legend_imp_xy
    else:
        x = 1.25
        y = 0
    # Define the sizes for the legend markers
    list_labels = ["  >0.2%", "  >0.5%", "  >1.0%"]
    list_sizes = [3, 5, 7]
    # Create the legend handles manually
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label=label,
                   markersize=size, markerfacecolor='black', linewidth=0)
        for label, size in zip(list_labels, list_sizes)]
    # Create the second legend
    ax.legend(handles=legend_handles,
              title="Feature importance",
              loc='lower left',
              bbox_to_anchor=(x, y),
              frameon=False,
              title_fontsize=fontsize_title,
              fontsize=fontsize,
              labelspacing=0.25,
              columnspacing=0, handletextpad=0, handlelength=0,
              borderpad=0)


# II Main Functions
def plot_feature_map(df_feat=None, df_cat=None,
                     col_cat="subcategory", col_val="mean_dif", col_imp="feat_importance",
                     normalize=False,
                     name_test="TEST", name_ref="REF",
                     figsize=(8, 8),
                     start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                     tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None,
                     fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                     fontsize_labels=12, fontsize_annotations=11,
                     add_xticks_pos=False,
                     grid_linewidth=0.01, grid_linecolor=None,
                     border_linewidth=2,
                     facecolor_dark=False, vmin=None, vmax=None,
                     cmap=None, cmap_n_colors=None,
                     cbar_pct=True, cbar_kws=None,
                     dict_color=None, legend_kws=None,
                     xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                     ytick_size=None):
    """Create a comprehensive feature map with a heatmap, feature importance bars, and custom legends."""
    # Get fontsize
    fs_labels, fs_annotations = _adjust_fontsize(fontsize_labels=fontsize_labels,
                                                 fontsize_annotations=fontsize_annotations)
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
    fig, axes = plt.subplots(figsize=figsize,
                             nrows=1, ncols=2,
                             sharey=True,
                             gridspec_kw={'width_ratios': width_ratio, "wspace": 0},
                             layout="constrained")

    # Set color bar and legend arguments
    _cbar_kws, cbar_ax = _adjust_cbar_kws(fig=fig,
                                          cbar_kws=cbar_kws,
                                          cbar_xywh=None,
                                          name_test=name_test,
                                          name_ref=name_ref,
                                          fontsize_labels=fs_labels)

    n_cat = len(set(df_feat[ut.COL_CAT]))
    _legend_kws = _adjust_legend_kws(legend_kws=legend_kws,
                                     n_cat=n_cat,
                                     legend_xy=None,
                                     fontsize_labels=fs_labels)

    # Plot feat importance bars
    plot_feat_importance_bars(ax=axes[1], df=df_feat.copy(),
                              titlesize=fs_annotations,
                              labelsize=fs_annotations,
                              top_pcp_size=fs_annotations-2,
                              top_pcp_weight="bold",
                              col_imp=col_imp,
                              label="Cumulative feature\n  importance [%]")

    ax = plot_heatmap_(df_feat=df_feat, df_cat=df_cat,
                       col_cat=col_cat, col_val=col_val,
                       normalize=normalize,
                       ax=axes[0], figsize=figsize,
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
                       **args_xtick, ytick_size=ytick_size)

    # Add feature position title
    title_positions = ("                  Feature                  \n"
                       "Scale (subcategory)  +  Positions                 ")
    plt.title(title_positions, x=0, weight="bold", fontsize=fs_annotations)


    # Add feature importance map
    add_feat_importance_map(df_feat=df_feat, df_cat=df_cat,
                            col_cat=col_cat, col_imp=col_imp,
                            normalize=normalize,
                            ax=ax, start=start, args_len=args_len)

    add_feat_importance_legend(ax=cbar_ax,
                               legend_imp_xy=None,
                               fontsize_title=fs_labels,
                               fontsize=fs_annotations)


    return fig, ax
