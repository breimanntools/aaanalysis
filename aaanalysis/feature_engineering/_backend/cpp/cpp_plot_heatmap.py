"""
This is a script for the backend of the cpp_plot.heatmap method.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions


# I Helper Functions
# cbar helper functions
def _get_center_heatmap(df_pos=None):
    """Get center of heatmap colormap"""
    center = 0 if df_pos.min().min() < 0 else None
    return center


def _get_cmap_heatmap(df_pos=None, cmap=None, n_colors=None, higher_color=None, lower_color=None, facecolor_dark=True):
    """Get sequential or diverging cmap for heatmap"""
    n_colors = 100 if n_colors is None else n_colors
    if cmap == "SHAP":
        n = 20
        cmap_low = sns.light_palette(lower_color, input="hex", reverse=True, n_colors=int(n_colors/2)+n)
        cmap_high = sns.light_palette(higher_color, input="hex", n_colors=int(n_colors/2)+n)
        c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
        cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
        return cmap
    if cmap is None:
        # Use diverging colormap if positive and negative
        if df_pos.min().min() < 0:
            cmap = "RdBu_r"
        # Use sequential colormap if values just positive
        else:
            cmap = "flare"
    if df_pos.min().min() >= 0:
        cmap = sns.color_palette(cmap, n_colors=n_colors)
    else:
        n = 5
        cmap = sns.color_palette(cmap, n_colors=n_colors+n*2)
        cmap_low, cmap_high = cmap[0:int((n_colors+n*2)/2)], cmap[int((n_colors+n*2)/2):]
        c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
        cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
    return cmap


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
    """
    _label = f"Feature value\n{name_test} - {name_ref}"
    cbar_kws = dict(use_gridspec=False,
                    orientation="horizontal",
                    ticksize=tick_fontsize,
                    label=_label,
                    pad=2,
                    panchor=(0, 0))
    """
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


def _infer_vmin_vmax_from_data(df_pos=None, vmin=None, vmax=None):
    """Infer vmin and vmax from data if None"""
    vmin = df_pos.min().min() if vmin is None else vmin
    vmax = df_pos.max().max() if vmax is None else vmax
    return vmin, vmax


# Get cbar args and set cbar
def _get_cbar_arguments(df_pos=None, cmap=None, cmap_n_colors=None, cbar_kws=None, facecolor_dark=None):
    """"""
    center = _get_center_heatmap(df_pos=df_pos)
    cmap = _get_cmap_heatmap(df_pos=df_pos, cmap=cmap, n_colors=cmap_n_colors, higher_color=ut.COLOR_SHAP_POS,
                             lower_color=ut.COLOR_SHAP_NEG, facecolor_dark=facecolor_dark)
    dict_cbar, cbar_kws_ = _get_cbar_args_heatmap(cbar_kws=cbar_kws, df_pos=df_pos)
    return center, cmap, dict_cbar, cbar_kws_


def _set_cbar_heatmap(ax=None, dict_cbar=None, cbar_kws=None,
                      vmin=None, vmax=None, cbar_pct=True, weight="normal",
                      fontsize=11):
    """"""
    # Set colorbar labelsize and ticksize
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
    cb.set_label(label=cbar_kws["label"], weight=weight, size=fontsize)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')


# Add importance map (for feature map)
def _add_importance_map(ax=None, df_feat=None, df_cat=None, start=None, args_len=None, col_cat=None):
    """"""
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat, col_cat=col_cat,
                           col_value=ut.COL_FEAT_IMPORT, value_type="sum",
                           normalize=True)
    _df = pd.melt(df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, "position", ut.COL_FEAT_IMPORT]
    _list_sub_cat = _df[ut.COL_SUBCAT].unique()
    for i, sub_cat in enumerate(_list_sub_cat):
        _dff = _df[_df[ut.COL_SUBCAT] == sub_cat]
        for pos, val in enumerate(_dff[ut.COL_FEAT_IMPORT]):
            _symbol = "■"
            color = "black"
            size = 7 if val >= 1 else (5 if val >= 0.5 else 3)
            _args_symbol = dict(ha="center", va="center", color=color, size=size)
            if val >= 0.2:
                ax.text(pos + 0.5, i + 0.5, _symbol, **_args_symbol)


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
    pe.set_figsize(figsize=figsize)   # figsize is not used as argument in seaborn (but in pandas)
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
    # TODO as argument
    ax.axvline(jmd_n_len, color=linecolor, linestyle="-", linewidth=1.5)
    ax.axvline(x=jmd_n_len + tmd_len, color=linecolor, linestyle="-", linewidth=1.5)
    return ax


# TODO adjust and integrate into _cpp_plot.heatmap
# Outer plotting function
def plot_heatmap(df_feat=None, df_cat=None, col_cat="subcategory", col_value="mean_dif", value_type="mean", normalize=False,
                 figsize=(8, 5), ax=None, dict_color=None,
                 vmin=None, vmax=None, grid_on=True,
                 cmap="RdBu_r", cmap_n_colors=None, cbar_kws=None, cbar_ax=None,  #cbar_ax_pos=(0.5, 0.01, 0.2, 0.015),
                 facecolor_dark=False, add_jmd_tmd=True,
                 tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1,
                 tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None, linecolor=None, add_importance_map=False,
                 tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                 seq_size=None, fontsize_tmd_jmd=None, fontsize_labels=11,
                 xticks_pos=False, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, ytick_size=None,
                 add_legend_cat=True, legend_kws=None, cbar_pct=True):
    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_size = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    # Get df positions
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(), col_cat=col_cat,
                           col_value=col_value, value_type=value_type,
                           normalize=normalize)
    # Get cbar args
    vmin, vmax = _infer_vmin_vmax_from_data(df_pos=df_pos, vmin=vmin, vmax=vmax)
    center, cmap, dict_cbar, cbar_kws_ = _get_cbar_arguments(df_pos=df_pos, cmap=cmap, cmap_n_colors=cmap_n_colors,
                                                             cbar_kws=cbar_kws, facecolor_dark=facecolor_dark)
    # fig = plt.gcf()
    # cbar_ax = fig.add_axes(cbar_ax_pos)
    # Plotting
    ax = _plot_inner_heatmap(ax=ax, figsize=figsize, df_pos=df_pos, vmin=vmin, vmax=vmax,
                             facecolor_dark=facecolor_dark, grid_on=grid_on, linecolor=linecolor,
                             cbar_ax=cbar_ax, cmap=cmap, center=center, cbar_kws=cbar_kws,
                             x_shift=0.5, **args_xtick, ytick_size=ytick_size,
                             **args_len, start=start)
    # Autosize tmd sequence & annotation
    pe = PlotElements()
    opt_size = pe.optimize_label_size(ax=ax, df_pos=df_pos)
    # Add tmd_jmd sequence
    if isinstance(tmd_seq, str):
        ax = pp.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                xticks_pos=xticks_pos,
                                x_shift=0.5, xtick_size=xtick_size)
        # TODO check how to implement in functional style (relates to update_seq)
        #self.ax_seq = ax
    # Add tmd_jmd bar
    elif add_jmd_tmd:
        size = opt_size if fontsize_tmd_jmd is None else fontsize_tmd_jmd
        pp.add_tmd_jmd_bar(ax=ax, **args_part_color)
        pp.add_tmd_jmd_xticks(ax=ax, x_shift=0.5, **args_xtick)
        pp.add_tmd_jmd_text(ax=ax, x_shift=0, fontsize_tmd_jmd=size)
    # Add cbar
    _set_cbar_heatmap(ax=ax, vmin=vmin, vmax=vmax,
                      dict_cbar=dict_cbar, cbar_kws=cbar_kws_, cbar_pct=cbar_pct,
                      weight="normal", fontsize=fontsize_labels)
    # Add scale classification
    if add_legend_cat:
        ax = pe.add_legend_cat(ax=ax, df_pos=df_pos, df_cat=df_cat, y=col_cat, dict_color=dict_color,
                               legend_kws=legend_kws)
    # Add importance map
    if add_importance_map:
        _add_importance_map(ax=ax, df_feat=df_feat, df_cat=df_cat, start=start, args_len=args_len, col_cat=col_cat)
    # Set current axis to main axis object depending on tmd sequence given or not
    plt.sca(plt.gcf().axes[0])
    ax = plt.gca()
    return ax


# Special calling example fomr sub pred
"""
def plot_feat_heatmap(df_rel_nonsub=None, df_feat=None, n=50, ytick_size=16, label_fontsize=15,
                      just_subexpert=True, annotation="Uniprot", name_test="SUBEXP", name_ref="REF",
                      facecolor_dark=True, show_title=False, add_importance_map=False):
    """"""
    cp = CPPPlot(df_rel_nonsub=df_rel_nonsub, just_subexpert=just_subexpert, annotation=annotation)
    # Adjust for plotting profiles
    df_feat = df_feat.sort_values(by=ut.COL_FEAT_IMP, ascending=False).reset_index(drop=True)
    sum_n = round(sum(df_feat[ut.COL_FEAT_IMP].head(n)))
    f = lambda x: x if x != 0 else 0.1
    df_feat[ut.COL_FEAT_IMP] = [f(x) for x in df_feat[ut.COL_FEAT_IMP]]
    ut.plot_settings(change_size=False)
    if n >= 60:
        args_fig = dict(figsize=(10, 10), feat_pad_factor=3.2)
    else:
        args_fig = dict(figsize=(10, 9), feat_pad_factor=3.6)
    title = f"{name_test} vs {name_ref} (top {n} features)" if show_title else None
    args = dict(df_feat=df_feat.copy(), facecolor_dark=facecolor_dark, title=title)
    cp.heatmap(**args, n=n, sum_imp=sum_n, tick_fontsize=ytick_size, label_fontsize=label_fontsize,
               **args_fig, add_importance_map=add_importance_map, name_ref=name_ref, name_test=name_test)
"""