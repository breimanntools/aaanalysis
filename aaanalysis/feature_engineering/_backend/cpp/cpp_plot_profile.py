"""
This is a script for the backend of the cpp_plot.profile method.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import aaanalysis.utils as ut
from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions

# I Helper Functions
def _scale_ylim(df_bars=None, ylim=None, col_imp=None, retrieve_plot=False, scaling_factor=1.1, verbose=True):
    """Check if ylim is valid and appropriate for the given dataframe and column."""
    if ylim is not None:
        max_val = round(max(df_bars[col_imp]), 3)
        min_val = round(min(df_bars[col_imp]), 3)
        min_y, max_y = ylim
        warning_msg = ""
        if min_val < min_y and max_val > max_y:
            warning_msg = f"The provided 'ylim' ({ylim}) is both smaller and larger than the data range ({min_val}, {max_val})."
            ylim = (min_val, max_val * scaling_factor)
        elif min_val < min_y:
            warning_msg = f"The minimum value of 'ylim' ({min_y}) is larger than the minimum data value ({min_val})."
            ylim = (min_val, max_y)
        elif max_val > max_y:
            warning_msg = f"The maximum value of 'ylim' ({max_y}) is smaller than the maximum data value ({max_val})."
            ylim = (min_y, max_val * scaling_factor)
        if warning_msg and verbose:
            warnings.warn(warning_msg, UserWarning)
    else:
        if retrieve_plot:
            ylim = plt.ylim()
            ylim = (ylim[0] * scaling_factor, ylim[1] * scaling_factor)
    return ylim


# I Helper Functions
def _plot_cpp_shap_profile(ax=None, df_pos=None, ylim=None, plot_args=None):
    """"""
    df_bar = df_pos.T
    df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="col_cat")
    df_pos = df_bar[df_bar > 0]
    df_pos = df_pos.sum(axis=1)
    df_neg = df_bar[df_bar < 0]
    df_neg = df_neg.sum(axis=1)
    ax = df_pos.plot(ax=ax, color=ut.COLOR_SHAP_POS, **plot_args)
    ax = df_neg.plot(ax=ax, color=ut.COLOR_SHAP_NEG, **plot_args)
    ylim = _scale_ylim(df_bars=df, col_imp="col_cat", ylim=ylim, retrieve_plot=True)
    plt.ylim(ylim)
    return ax


def _plot_cpp_profile(ax=None, df_pos=None, dict_color=None, add_legend=True, color=None, ylim=None,
                      plot_args=None, legend_kws=None):
    """"""
    pe = PlotElements()
    handles, labels = pe.get_legend_handles_labels(dict_color=dict_color, list_cat=list(df_pos.index))
    df_bar = df_pos.T[labels]
    # TODO df_bar only valid for y = "categories"
    df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="col_cat")
    color = dict_color if add_legend else color
    if not add_legend:
        df_bar = df_bar.sum(axis=1)
    ax = df_bar.plot(ax=ax, color=color, **plot_args)
    ylim = _scale_ylim(df_bars=df, col_imp="col_cat", ylim=ylim)
    plt.ylim(ylim)
    # Set legend TODO check if can be replaced by plot_set_legend
    if add_legend:
        _legend_kws = dict(ncol=2, prop={"size": 10}, loc=2, frameon=True, columnspacing=1, facecolor="white",
                           framealpha=1)
        if legend_kws is not None:
            _legend_kws.update(legend_kws)
            if "fontsize" in _legend_kws:
                fs = _legend_kws["fontsize"]
                _legend_kws.update(dict(prop={"size": fs}))
        plt.legend(handles=handles, labels=labels, **_legend_kws)
    return ax


# II Main Functions
# Inner plotting function
def _plot_profile(df_pos=None, shap_plot=False, ax=None, color="tab:gray",
                  start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                  add_legend=True, legend_kws=None, dict_color=None,
                  edge_color="none", bar_width=0.8,ylim=None,
                  xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                  ytick_size=None, ytick_width=None, ytick_length=None):
    """Inner feature profile plotting function: Plot CPP or CPP-SHAP profile, set ticks and limits"""
    # Constants
    XLIM_ADD = 3 if shap_plot else 1
    seq_len = jmd_n_len + tmd_len + jmd_c_len
    pp = PlotPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    plot_args = dict(kind="bar", stacked=True, rot=0, width=bar_width, edgecolor=edge_color,
                     legend=False, zorder=10)
    # Plot
    if shap_plot:
        ax = _plot_cpp_shap_profile(ax=ax, df_pos=df_pos, plot_args=plot_args, ylim=ylim)
    else:
        ax = _plot_cpp_profile(ax=ax, df_pos=df_pos, plot_args=plot_args, ylim=ylim, dict_color=dict_color,
                               add_legend=add_legend, color=color, legend_kws=legend_kws)
    # Set default x-ticks (if tmd_jmd not shown)
    xticks, xticks_labels = pp.get_xticks_with_labels(step=5)
    ax.tick_params(axis="x", color="black", width=xtick_width, length=xtick_length)
    ax.set_xticks(xticks)
    if xtick_size > 0:
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
    # Set default y-ticks
    # TODO why not shown in main figure if sequence??"?"?
    plt.yticks(size=ytick_size)
    ax.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
    # Add extra flanking space for x-axis
    x_lim = (min(xticks) - XLIM_ADD, max(xticks) + XLIM_ADD)
    ax.set_xlim(x_lim)
    # Add extra flanking space for y-axis
    if ylim is None:
        ymin, ymax = ax.get_ylim()
        y_space = min(0, (ymax - ymin) * 0.25)
        y_lim = (ymin - y_space, ymax + y_space)
        ax.set_ylim(y_lim)
    # Plot baseline
    ax.plot([-0.5, seq_len - 0.5], [0, 0], color="black", linestyle="-")
    sns.despine(top=True, right=True)
    return ax


# Outer plotting function
def plot_profile(figsize=(7, 5), ax=None, df_feat=None, df_cat=None,
                 col_imp="feat_importance", normalize=False,
                 dict_color=None,
                 edge_color="none", bar_width=0.75,
                 tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 start=1, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                 tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                 seq_size=None, fontsize_tmd_jmd=None, fontsize_label=None,
                 xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, add_xticks_pos=False,
                 ytick_size=None, ytick_width=None, ytick_length=None, ylim=None,
                 highlight_tmd_area=True, highlight_alpha=0.15, grid_axis=None,
                 add_legend_cat=True, legend_kws=None, shap_plot=False):
    """Out feature profile plotting function: pre-process df_feat and add general style elements (e.g., TMD area)"""
    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    args_ytick= dict(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
    # Get df positions
    value_type = "sum" if col_imp else "count"
    col_cat = "scale_name" if shap_plot else "category"
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(), col_cat=col_cat,
                           col_value=col_imp, value_type=value_type, normalize=normalize)
    # Plotting
    pe = PlotElements()
    pe.set_figsize(figsize=figsize)
    ax = _plot_profile(df_pos=df_pos, shap_plot=shap_plot, ax=ax,
                       start=start, **args_len,
                       add_legend=add_legend_cat, dict_color=dict_color, legend_kws=legend_kws,
                       edge_color=edge_color, bar_width=bar_width,
                       ylim=ylim, **args_ytick, **args_xtick,)
    # Set default ylabel
    if col_imp is None:
        ylabel = ut.LABEL_FEAT_NUMBER
    else:
        ylabel = ut.LABEL_FEAT_IMPACT_CUM if shap_plot else ut.LABEL_FEAT_IMPORT_CUM
    ax.set_ylabel(ylabel, size=fontsize_label)
    # Add grid
    if grid_axis is not None:
        ax.set_axisbelow(True)  # Grid behind datasets
        ax.grid(which="major", axis=grid_axis, linestyle="-")
    # Add tmd area
    if highlight_tmd_area:
        pp.highlight_tmd_area(ax=ax, x_shift=-0.5, tmd_color=tmd_color, alpha=highlight_alpha)
    # Add tmd_jmd sequence
    if type(tmd_seq) == str:
        pp.add_tmd_jmd_seq(ax=ax, seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                           **args_seq, **args_part_color, **args_seq_color,
                           add_xticks_pos=add_xticks_pos, heatmap=False, x_shift=0,
                           xtick_size=xtick_size)
    # Add tmd_jmd bar
    else:
        pp.add_tmd_jmd_bar(ax=ax, x_shift=-0.5, **args_part_color)
        pp.add_tmd_jmd_xticks(ax=ax, x_shift=0, **args_xtick)
        pp.add_tmd_jmd_text(ax=ax, x_shift=-0.5, fontsize_tmd_jmd=fontsize_tmd_jmd)
    # Set current axis to main axis object depending on tmd sequence given or not
    plt.sca(plt.gcf().axes[0])
    ax = plt.gca()
    return ax
