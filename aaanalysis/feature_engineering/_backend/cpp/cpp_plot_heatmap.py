"""
This is a script for the backend of the CPPPlot.heatmap() method.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions


# I Helper Functions
def _get_value_type(col_val="abs_auc"):
    """Get value type corresponding to col_val"""
    dict_val_type = ut.DICT_VALUE_TYPE
    val_type = "sum"if ut.COL_FEAT_IMPACT in col_val else dict_val_type[col_val]
    return val_type

def _infer_vmin_vmax_from_data(df_pos=None, vmin=None, vmax=None):
    """Infer vmin and vmax from data if None"""
    vmin = df_pos.min().min() if vmin is None else vmin
    vmax = df_pos.max().max() if vmax is None else vmax
    return vmin, vmax

# Get cbar cmap
def _get_shap_cmap(n_colors=None, facecolor_dark=False):
    """Generate SHAP specific colormap."""
    n = 20
    cmap_low = sns.light_palette(ut.COLOR_SHAP_NEG, input="hex", reverse=True, n_colors=int(n_colors/2) + n)
    cmap_high = sns.light_palette(ut.COLOR_SHAP_POS, input="hex", n_colors=int(n_colors/2) + n)
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
    return cmap

def _get_diverging_cmap(cmap, n_colors=None, facecolor_dark=False):
    """Generate a diverging colormap."""
    n = 5
    cmap = sns.color_palette(cmap, n_colors=n_colors + n * 2)
    cmap_low, cmap_high = cmap[:int((n_colors + n * 2) / 2)], cmap[int((n_colors + n * 2) / 2):]
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
    return cmap

def _get_cmap_heatmap(df_pos=None, cmap=None, n_colors=100, facecolor_dark=True):
    """Generate a sequential or diverging colormap for heatmap."""
    n_colors = 100 if n_colors is None else n_colors
    if cmap == "SHAP":
        return _get_shap_cmap(n_colors, facecolor_dark)
    if cmap is None:
        cmap = "RdBu_r" if df_pos.min().min() < 0 else "flare"
    if df_pos.min().min() >= 0:
        cmap = sns.color_palette(cmap, n_colors=n_colors)
    else:
        cmap = _get_diverging_cmap(cmap, n_colors=n_colors, facecolor_dark=facecolor_dark)
    return cmap

# Get cbar ticks
def _get_cbar_ticks_heatmap(df_pos=None):
    """Get legend ticks for heatmap"""
    stop_legend = df_pos.values.max()
    if type(stop_legend) is int:
        # Ticks for count datasets
        cbar_ticks = [int(x) for x in np.linspace(1, stop_legend, num=stop_legend)]
    else:
        cbar_ticks = None
    return cbar_ticks


def _get_cbar_args_heatmap(cbar_kws=None, df_pos=None):
    """Parameter to set manually"""
    # Get cbar ticks
    #_label = f"Feature value\n{name_test} - {name_ref}"
    _label = "Feature value\nTEST-REF"
    cbar_kws = dict(use_gridspec=False,
                    orientation="horizontal",
                    #ticksize=tick_fontsize,
                    label=_label,
                    pad=2,
                    panchor=(0, 0))
    cbar_ticks = _get_cbar_ticks_heatmap(df_pos=df_pos)
    width, height = plt.gcf().get_size_inches()
    dict_cbar = {"ticksize": width + 3, "labelsize": width + 6, "labelweight": "medium"}
    cbar_kws_ = {"ticks": cbar_ticks, "shrink": 0.5}
    if cbar_kws is not None:
        # Catch color bar arguments that must be set manually
        for arg in dict_cbar:
            if arg in cbar_kws:
                dict_cbar[arg] = cbar_kws.pop(arg)
        cbar_kws_.update(**cbar_kws)
    return dict_cbar, cbar_kws_


# Get cbar args and set cbar
def get_cbar_args(df_pos=None, cmap=None, cmap_n_colors=None, cbar_kws=None, facecolor_dark=None):
    """"""
    center = 0 if df_pos.min().min() < 0 else None
    cmap = _get_cmap_heatmap(df_pos=df_pos, cmap=cmap, n_colors=cmap_n_colors, facecolor_dark=facecolor_dark)
    dict_cbar, cbar_kws_ = _get_cbar_args_heatmap(cbar_kws=cbar_kws, df_pos=df_pos)
    return center, cmap, dict_cbar, cbar_kws_


def set_cbar_heatmap(ax=None, dict_cbar=None, cbar_kws=None,
                     vmin=None, vmax=None, cbar_pct=True, weight="normal",
                     fontsize=11):
    """"""
    # Set colorbar label size and ticksize
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
    cb = ax.collections[0].colorbar
    if "label" in cbar_kws:
        cb.set_label(label=cbar_kws["label"], weight=weight, size=fontsize)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')



# II Main Functions
# Inner plotting function
def _plot_inner_heatmap(ax=None, figsize=(8, 8), df_pos=None, vmin=None, vmax=None,
                        facecolor_dark=True, grid_on=False, linecolor="black",
                        cbar_ax=None, cmap=None, center=None, cbar_kws=None,
                        x_shift=0.0, xtick_size=11.0, xtick_width=2.0, xtick_length=None, ytick_size=None,
                        tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Show summary static values of feature categories/sub_categories per position as heat map"""
    facecolor = "black" if facecolor_dark else "white"
    linecolor = linecolor if linecolor is not None else "gray" if facecolor_dark else "black"
    # Default arguments for heatmap
    linewidths = 0.01 if grid_on else 0
    # Plot with 0 set to NaN
    pe = PlotElements()
    fig, ax = pe.set_figsize(ax=ax, figsize=figsize)   # figsize is not used as argument in seaborn (but in pandas)
    data = df_pos.replace(0, np.NaN)
    ax = sns.heatmap(data, ax=ax, vmin=vmin, vmax=vmax,
                     cbar_ax=cbar_ax, cbar_kws=cbar_kws, cmap=cmap,  center=center,
                     yticklabels=True, xticklabels=True,
                     linewidths=linewidths, linecolor=linecolor)
    # Set default x ticks (if tmd_jmd not shown)
    pp = PlotPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    pp.add_xticks(ax=ax, xticks_position="bottom", x_shift=x_shift, xtick_size=xtick_size, xtick_width=xtick_width,
                  xtick_length=xtick_length)
    ax.tick_params(axis='y', which='both', length=0, labelsize=ytick_size)
    # Set frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_facecolor(facecolor)
    # Add lines to frame
    # TODO as argument grid_lw
    lw = ut.plot_gco(option="grid.linewidth") * 2
    ax.axvline(jmd_n_len, color=linecolor, linestyle="-", linewidth=lw)
    ax.axvline(x=jmd_n_len + tmd_len, color=linecolor, linestyle="-", linewidth=lw)
    return ax


# Main plotting function
def plot_heatmap(df_feat=None, df_cat=None, shap_plot=False,
                 col_cat="subcategory", col_val="mean_dif",
                 normalize=False,
                 ax=None, figsize=(8, 5),
                 start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                 tmd_color="mediumspringgreen", jmd_color="blue",
                 tmd_seq_color="black", jmd_seq_color="white",
                 seq_size=None, fontsize_tmd_jmd=None, fontsize_labels=11,
                 add_xticks_pos=False,
                 grid_on=True, grid_linecolor=None, grid_linewidth=1,
                 add_tmd_jmd_border=True, border_linewidth=None,
                 add_legend_cat=True, dict_color=None, legend_kws=None,
                 facecolor_dark=None, add_jmd_tmd=True,
                 vmin=None, vmax=None,
                 cmap="RdBu_r", cmap_n_colors=None, cbar_kws=None, cbar_ax=None,  #cbar_ax_pos=(0.5, 0.01, 0.2, 0.015)
                 cbar_pct=True,
                 xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, ytick_size=None):
    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_size = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd) # TODO check if need (where is labelfontsiz)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    # Get df positions
    value_type = _get_value_type(col_val=col_val)
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(), col_cat=col_cat,
                           col_val=col_val, value_type=value_type, normalize=normalize)
    vmin, vmax = _infer_vmin_vmax_from_data(df_pos=df_pos, vmin=vmin, vmax=vmax)
    # Get cbar args
    center, cmap, dict_cbar, cbar_kws_ = get_cbar_args(df_pos=df_pos, cmap=cmap, cmap_n_colors=cmap_n_colors,
                                                       cbar_kws=cbar_kws, facecolor_dark=facecolor_dark)
    # Plotting
    print(dict_cbar)
    print(cbar_kws_)
    pe = PlotElements()
    fig, ax = pe.set_figsize(ax=ax, figsize=figsize, force_set=True)
    #cbar_ax = fig.add_axes(cbar_ax_pos)
    ax = _plot_inner_heatmap(ax=ax, figsize=figsize, df_pos=df_pos, vmin=vmin, vmax=vmax,
                             facecolor_dark=facecolor_dark, grid_on=grid_on, linecolor=grid_linecolor,
                             cbar_ax=cbar_ax, cmap=cmap, center=center, cbar_kws=cbar_kws,
                             x_shift=0.5, **args_xtick, ytick_size=ytick_size,
                             **args_len, start=start)
    # Add tmd_jmd sequence
    if isinstance(tmd_seq, str):
        ax = pp.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                add_xticks_pos=add_xticks_pos,
                                x_shift=0.5, xtick_size=xtick_size)
        # TODO check how to implement in functional style (relates to update_seq)
        #self.ax_seq = ax
    # Add tmd_jmd bar
    elif add_jmd_tmd:
        pp.add_tmd_jmd_bar(ax=ax, **args_part_color)
        pp.add_tmd_jmd_xticks(ax=ax, x_shift=0.5, **args_xtick)
        pp.add_tmd_jmd_text(ax=ax, x_shift=0, fontsize_tmd_jmd=fontsize_tmd_jmd)
    # Add cbar
    # TODO under and with title
    set_cbar_heatmap(ax=ax, vmin=vmin, vmax=vmax,
                     dict_cbar=dict_cbar, cbar_kws=cbar_kws_, cbar_pct=cbar_pct,
                     weight="normal", fontsize=fontsize_labels)
    # Add scale classification
    """
    if add_legend_cat:
        legend_kws = pe.update_legend_kws(legend_kws=legend_kws)
        ut.plot_legend_(ax=ax, dict_color=dict_color, **legend_kws)
    """
    if add_legend_cat:
        ax = pe.add_legend_cat(ax=ax, df_pos=df_pos, df_cat=df_cat, y=col_cat, dict_color=dict_color,
                               legend_kws=legend_kws)
    # Set current axis to main axis object depending on tmd sequence given or not
    plt.sca(plt.gcf().axes[0])
    ax = plt.gca()
    return ax
