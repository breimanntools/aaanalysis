"""
This is the common utility script for the backend of the CPPPlot.heatmap() and
CPPPlot.feat_map() methods.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPartPositions


WIDTH_NORM_FACTOR = 110


# I Helper Functions
def _get_value_type(col_val="abs_auc"):
    """Determine the value type based on column value."""
    dict_val_type = ut.DICT_VALUE_TYPE
    if ut.COL_FEAT_IMPACT in col_val:
        val_type = "sum"
    elif ut.COL_MEAN_DIF in col_val:
        val_type = "mean"
    else:
        val_type = dict_val_type.get(col_val, "mean")
    return val_type


def _adjust_df_feat(df_feat=None, col_val=None, cbar_pct=True):
    """Adjusts feature values in `df_feat` based on percentage scaling and sets the limits for difference columns."""
    df_feat = df_feat.copy()
    val_type = _get_value_type(col_val=col_val)
    if val_type == "sum":
        sum_val = df_feat[col_val].abs().sum()
        adjust_col_val = sum_val < 1
    else:
        min_val = df_feat[col_val].min()
        max_val = df_feat[col_val].max()
        n_th = 1 if max_val < 0 or 0 < min_val else 2
        adjust_col_val = max(df_feat[col_val]) - min(df_feat[col_val]) < n_th
    if cbar_pct and adjust_col_val:
        df_feat[col_val] *= 100
    return df_feat


def _get_vmin_vmax(df_pos=None, vmin=None, vmax=None):
    """Obtain minimum and maximum values for the heatmap."""
    if vmax is not None and vmin is not None:
        return vmin, vmax   # Skip adjustment
    vmin = df_pos.min().min() if vmin is None else vmin
    vmax = df_pos.max().max() if vmax is None else vmax
    # Create centered vmin and vmax
    val = max((abs(vmin), abs(vmax)))
    vmax = val if vmax > 0 else 0
    vmin = -val if vmin < 0 else 0
    return vmin, vmax


def _get_bar_width(fig=None, len_seq=None):
    """Get consistent bar width to for category bars"""
    width, height = fig.get_size_inches()
    width_factor = 1 / width * 8
    bar_width = len_seq / WIDTH_NORM_FACTOR * width_factor
    return bar_width


# Get colormap
def _get_cmap_heatmap(df_pos=None, cmap=None, n_colors=101, facecolor_dark=True, vmin=None, vmax=None):
    """Create a sequential or diverging colormap for a heatmap based on data properties."""
    n_colors = n_colors if n_colors is not None else 101
    v_max = df_pos.max().max() if vmax is None else vmax
    v_min = df_pos.min().min() if vmin is None else vmin
    only_neg = v_max <= 0
    only_pos = v_min >= 0
    args = dict(n_colors=n_colors, facecolor_dark=facecolor_dark,
                only_neg=only_neg, only_pos=only_pos)
    cmap = ut.STR_CMAP_CPP if cmap is None else cmap
    cmap = ut.plot_get_cmap_(cmap=cmap, **args)
    return cmap


# Get and set colorbar arguments
def _get_cbar_ticks_heatmap(df_pos=None):
    """Calculate colorbar ticks for the heatmap."""
    stop_legend = df_pos.values.max()
    if type(stop_legend) is int:
        # Ticks for count datasets
        cbar_ticks = [int(x) for x in np.linspace(1, stop_legend, num=stop_legend)]
    else:
        cbar_ticks = None
    return cbar_ticks


def _get_cbar_args_heatmap(df_pos=None, cbar_kws=None):
    """Create colorbar arguments for the heatmap."""
    cbar_ticksize = ut.plot_gco() - 2
    cbar_ticks = _get_cbar_ticks_heatmap(df_pos=df_pos)
    dict_cbar = {"ticksize": cbar_ticksize, "labelsize": cbar_ticksize, "labelweight": "normal"}
    cbar_kws_ = dict(orientation="horizontal", ticks=cbar_ticks)
    if cbar_kws is not None:
        # Catch colorbar arguments that must be set manually
        for arg in dict_cbar:
            if arg in cbar_kws:
                dict_cbar[arg] = cbar_kws.pop(arg)
        cbar_kws_.update(**cbar_kws)
    return dict_cbar, cbar_kws_


def _set_cbar_heatmap(ax=None, dict_cbar=None, cbar_kws=None,
                      vmin=None, vmax=None, cbar_pct=True, weight="normal", fontsize=11):
    """Configure the heatmap's colorbar."""
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=dict_cbar["ticksize"])
    if "label" in cbar_kws:
        cbar.set_label(label=cbar_kws["label"], weight="bold", size=dict_cbar["labelsize"])
    cbar.ax.yaxis.label.set_size(dict_cbar["labelsize"])
    cbar_ticks = [x for x in cbar.get_ticks() if vmin <= x <= vmax]
    cbar.set_ticks(cbar_ticks)
    str_zero = "[0]"
    str_pct = "%" if cbar_pct else ""
    f = lambda x: int(x) if int(x) == float(x) else round(x, 1)
    cbar.set_ticklabels([f"{f(x)}{str_pct}" if float(x) != 0 else str_zero for x in cbar_ticks])
    if "label" in cbar_kws:
        cbar.set_label(label=cbar_kws["label"], weight=weight, size=fontsize)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    # Customization for the thin line on top or right of the colorbar
    orientation = cbar_kws.get("orientation", None)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)  # Show only the top spine
    if orientation is not None:
        pos = "top" if orientation == "horizontal" else "right"
        cbar.ax.spines[pos].set_visible(True)
        xtick_width = ut.plot_gco(option="xtick.major.width")
        cbar.ax.spines[pos].set_linewidth(xtick_width)


# II Main Functions
# Inner plotting function
def _plot_inner_heatmap(df_pos=None, ax=None, figsize=(8, 8),
                        start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                        grid_linewidth=0.01, grid_linecolor=None,
                        border_linewidth=2,
                        facecolor_dark=True, vmin=None, vmax=None,
                        cbar_ax=None, cmap=None, cbar_kws=None,
                        x_shift=0.0,
                        xtick_size=11.0, xtick_width=2.0, xtick_length=None):
    """Plot the inner heatmap with specific settings and styles."""
    # Heatmap arguments
    vmin, vmax = _get_vmin_vmax(df_pos=df_pos, vmin=vmin, vmax=vmax)
    center = 0
    facecolor = "black" if facecolor_dark else "white"
    grid_linecolor = grid_linecolor if grid_linecolor is not None else  "gray" if facecolor_dark else "black"
    # Plot with 0 set to NaN
    pe = PlotElements()
    fig, ax = pe.set_figsize(ax=ax, figsize=figsize)
    data = df_pos.replace(0, np.nan)
    ax = sns.heatmap(data, ax=ax,
                     center=center, vmin=vmin, vmax=vmax,
                     linewidths=grid_linewidth, linecolor=grid_linecolor,
                     cbar_ax=cbar_ax, cbar_kws=cbar_kws, cmap=cmap,
                     yticklabels=True, xticklabels=True)
    # Set default x-ticks
    pp = PlotPartPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    pp.add_xticks(ax=ax, xticks_position="bottom", x_shift=x_shift,
                  xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    # Remove y-ticks
    ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    # Set frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_facecolor(facecolor)
    # Add lines to frame
    args = dict(color=grid_linecolor, linestyle="-", linewidth=border_linewidth)
    if jmd_n_len > 0:
        ax.axvline(jmd_n_len, **args)
    if jmd_c_len > 0:
        ax.axvline(x=jmd_n_len + tmd_len, **args)
    return ax


# Main plotting function
def plot_heatmap_(df_feat=None, df_cat=None,
                  col_cat="subcategory", col_val="mean_dif",
                  ax=None, figsize=(8, 8),
                  start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                  tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                  tmd_color="mediumspringgreen", jmd_color="blue",
                  tmd_seq_color="black", jmd_seq_color="white",
                  seq_size=None,
                  fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                  fontsize_labels=11,
                  add_xticks_pos=False,
                  grid_linewidth=0.01, grid_linecolor=None,
                  border_linewidth=2,
                  facecolor_dark=False, vmin=None, vmax=None,
                  cmap=None, cmap_n_colors=101,
                  cbar_ax=None, cbar_pct=True, cbar_kws=None,
                  dict_color=None, legend_kws=None,
                  xtick_size=11.0, xtick_width=2.0, xtick_length=5.0):
    """Main function to plot heatmap for feature value per categories/subcategories per position."""
    df_feat = _adjust_df_feat(df_feat=df_feat, col_val=col_val, cbar_pct=cbar_pct)

    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_fs = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

    # Get df positions
    value_type = _get_value_type(col_val=col_val)
    pp = PlotPartPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(),
                           col_cat=col_cat, col_val=col_val,
                           value_type=value_type, normalize=False)
    vmin, vmax = _get_vmin_vmax(df_pos=df_pos, vmin=vmin, vmax=vmax)
    # Get colorbar arguments
    cmap = _get_cmap_heatmap(df_pos=df_pos, cmap=cmap, n_colors=cmap_n_colors, facecolor_dark=facecolor_dark,
                             vmin=vmin, vmax=vmax)
    dict_cbar, cbar_kws = _get_cbar_args_heatmap(df_pos=df_pos, cbar_kws=cbar_kws)

    # Plotting
    pe = PlotElements()
    fig, ax = pe.set_figsize(ax=ax, figsize=figsize, force_set=True)
    ax = _plot_inner_heatmap(df_pos=df_pos, ax=ax, figsize=figsize,
                             start=start, **args_len,
                             grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                             border_linewidth=border_linewidth,
                             facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                             cbar_ax=cbar_ax, cmap=cmap, cbar_kws=cbar_kws,
                             x_shift=0.5, **args_xtick)
    # Add colorbar
    _set_cbar_heatmap(ax=ax, vmin=vmin, vmax=vmax,
                      dict_cbar=dict_cbar,
                      cbar_kws=cbar_kws, cbar_pct=cbar_pct,
                      weight="normal", fontsize=fontsize_labels)

    # Add tmd_jmd sequence
    if isinstance(tmd_seq, str):
        ax = pp.add_tmd_jmd_seq(ax=ax, **args_fs,
                                weight_tmd_jmd=weight_tmd_jmd,
                                **args_seq, **args_part_color, **args_seq_color,
                                add_xticks_pos=add_xticks_pos, heatmap=True,
                                x_shift=0.5, xtick_size=xtick_size)
    # Add tmd_jmd bar
    else:
        pp.add_tmd_jmd_bar(ax=ax, **args_part_color)
        pp.add_tmd_jmd_xticks(ax=ax, x_shift=0.5, **args_xtick)
        pp.add_tmd_jmd_text(ax=ax, x_shift=0,
                            fontsize_tmd_jmd=fontsize_tmd_jmd,
                            weight_tmd_jmd=weight_tmd_jmd)

    # Add scale bars
    bar_width = _get_bar_width(fig=fig, len_seq=jmd_n_len+tmd_len+jmd_c_len)
    pe.add_subcat_bars(ax=ax, df_pos=df_pos, df_feat=df_feat,
                       col_cat=col_cat, dict_color=dict_color,
                       bar_width=bar_width, bar_spacing=bar_width*0.75)

    # Add scale legend
    ut.plot_legend_(ax=ax, dict_color=dict_color, **legend_kws)

    # Set current axis to main axis object
    plt.sca(plt.gcf().axes[0])
    ax = plt.gca()
    return ax
