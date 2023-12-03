"""
This is a script for the backend of the cpp_plot.profile method.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis.utils as ut
from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPositions


# Constants


# I Helper Functions
# Plotting functions
def _plot_cpp_shap_profile(ax=None, df_pos=None, ylim=None, plot_args=None):
    """"""
    df_bar = df_pos.T
    df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="y")
    df_pos = df_bar[df_bar > 0]
    df_pos = df_pos.sum(axis=1)
    df_neg = df_bar[df_bar < 0]
    df_neg = df_neg.sum(axis=1)
    ax = df_pos.plot(ax=ax, color=ut.COLOR_SHAP_POS, **plot_args)
    ax = df_neg.plot(ax=ax, color=ut.COLOR_SHAP_NEG, **plot_args)
    ylim = ut.check_ylim(df=df, col_value="y", ylim=ylim, retrieve_plot=True)
    plt.ylim(ylim)
    return ax


def _plot_cpp_profile(ax=None, df_pos=None, dict_color=None, add_legend=True, color=None, ylim=None,
                      plot_args=None, legend_kws=None):
    """"""
    pe = PlotElements()
    handles, labels = pe.get_legend_handles_labels(dict_color=dict_color, list_cat=list(df_pos.index))
    df_bar = df_pos.T[labels]
    # TODO df_bar only valid for y = "categories"
    df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="y")
    color = dict_color if add_legend else color
    if not add_legend:
        df_bar = df_bar.sum(axis=1)
    ax = df_bar.plot(ax=ax, color=color, **plot_args)
    ylim = ut.check_ylim(df=df, col_value="y", ylim=ylim)
    plt.ylim(ylim)
    # Set legend
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
def _plot_profile(ax=None, df_pos=None, dict_color=None, edge_color="none",
                  bar_width=0.8, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                  ylim=None, color="tab:gray", add_legend=True, legend_kws=None,
                  shap_plot=False, tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Show count of feature categories/sub_categories per position for positive and
    negative features, i.e., feature with positive resp. negative mean_dif. The profile
    is a bar chart with positive and negative counts"""
    # Constants
    XLIM_ADD = 3 if shap_plot else 1
    seq_len = jmd_n_len + tmd_len + jmd_c_len

    pp = PlotPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    plot_args = dict(kind="bar", stacked=True, rot=0, width=bar_width, edgecolor=edge_color, legend=False,
                     zorder=10)
    # Plot
    if shap_plot:
        ax = _plot_cpp_shap_profile(ax=ax, df_pos=df_pos, plot_args=plot_args, ylim=ylim)
    else:
        ax = _plot_cpp_profile(ax=ax, df_pos=df_pos, plot_args=plot_args, ylim=ylim, dict_color=dict_color,
                               add_legend=add_legend, color=color, legend_kws=legend_kws)
    # Set default x ticks (if tmd_jmd not shown)
    xticks, xticks_labels = pp.get_xticks_with_labels(step=5)
    ax.tick_params(axis="x", color="black", width=xtick_width, length=xtick_length)
    ax.set_xticks(xticks)
    if xtick_size > 0:
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
    # Add extra flanking space for x axis
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
                 col_value="mean_dif", value_type="count", normalize=False,
                 dict_color=None,
                 edge_color="none", bar_width=0.75,
                 add_jmd_tmd=True, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 start=1, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                 tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                 seq_size=None, fontsize_tmd_jmd=None, fontsize_label=None,
                 xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, xticks_pos=False,
                 ytick_size=None, ytick_width=None, ytick_length=None, ylim=None,
                 highlight_tmd_area=True, highlight_alpha=0.15, grid_axis=None,
                 add_legend_cat=True, legend_kws=None, shap_plot=False):
    """
    Plot feature profile for given features from 'df_feat'.
    """
    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    #args_size = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    # Get df positions
    y = "scale_name" if shap_plot else "category"   # Column name in df_feat for grouping.
    pp = PlotPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(), y=y,
                           col_value=col_value, value_type=value_type,
                           normalize=normalize)
    # Plotting
    pe = PlotElements()
    pe.set_figsize(figsize=figsize)
    ax = _plot_profile(df_pos=df_pos, ax=ax, ylim=ylim, dict_color=dict_color, edge_color=edge_color,
                       bar_width=bar_width, add_legend=add_legend_cat, legend_kws=legend_kws, shap_plot=shap_plot,
                       **args_xtick, **args_len, start=start)
    # Autosize tmd sequence & annotation
    # TODO chanage to adjust by aa.plot_settings()
    # Set default ylabel
    ylabel = ut.LABEL_FEAT_IMPACT_CUM if shap_plot else ut.LABEL_FEAT_IMPORT_CUM    # TODO adjust
    ax.set_ylabel(ylabel, size=fontsize_label)
    # Adjust y-ticks
    plt.yticks(size=ytick_size)
    plt.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
    # Add grid
    if grid_axis is not None:
        ax.set_axisbelow(True)  # Grid behind datasets
        ax.grid(which="major", axis=grid_axis, linestyle="-")
    # Add tmd area
    if highlight_tmd_area:
        pp.highlight_tmd_area(ax=ax, x_shift=-0.5, tmd_color=tmd_color, alpha=highlight_alpha)
    # Add tmd_jmd sequence
    if type(tmd_seq) == str:
        opt_size = pe.optimize_label_size(ax=ax, df_pos=df_pos, label_term=False)
        seq_size = opt_size if seq_size is None else seq_size
        ax = pp.add_tmd_jmd_seq(ax=ax, seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                                **args_seq,  **args_part_color, **args_seq_color,
                                xticks_pos=xticks_pos, heatmap=False, x_shift=0,
                                xtick_size=xtick_size)
        # TODO check how to implement in functional style (relates to update_seq)
        #self.ax_seq = ax
    # Add tmd_jmd bar
    elif add_jmd_tmd:
        pp.add_tmd_jmd_bar(ax=ax, x_shift=-0.5, **args_part_color)
        pp.add_tmd_jmd_xticks(ax=ax, x_shift=0, **args_xtick)
        pp.add_tmd_jmd_text(ax=ax, x_shift=-0.5, fontsize_tmd_jmd=fontsize_tmd_jmd)
    # Set current axis to main axis object depending on tmd sequence given or not
    plt.sca(plt.gcf().axes[0])
    ax = plt.gca()
    return ax
