"""
This is a script for the backend of the AALogoPlot class.
"""
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns

import aaanalysis.utils as ut


# I Helper Functions
def _add_bit_score_bar(ax_info=None, df_logo_info=None, bar_color="gray", size=None, show_right=True, ylim=None):
    """Add bit score plot on top of motif"""
    ax_info.bar(df_logo_info.index, df_logo_info.values, color=bar_color)
    ax_info.spines["top"].set_visible(False)
    if show_right:
        ax_info.spines["left"].set_visible(False)
        ax_info.yaxis.set_label_position("right")
        ax_info.yaxis.tick_right()
    else:
        ax_info.spines["right"].set_visible(False)
    if ylim is not None:
        ax_info.set_ylim(ylim)
    ax_info.xaxis.set_tick_params(labelbottom=False)
    ax_info.set_ylabel("Bits", size=size)


def _add_p_sites(ax_logo=None, df_logo=None, target_p1_site=None, xtick_size=None):
    """Add PX to PX' as x-axis"""
    x_ticks = list(range(0, len(df_logo)))
    list_p_n_term = [f"P{target_p1_site - i}" for i in range(0, target_p1_site)]
    list_p_c_term = [f"P{i}'" for i in range(1, len(x_ticks) - target_p1_site + 1)]
    x_ticks_labels = list_p_n_term + list_p_c_term
    ax_logo.tick_params(axis="x", length=0, color="black", width=0, bottom=True)
    ax_logo.set_xticks(x_ticks)
    ax_logo.set_xticklabels(x_ticks_labels, fontsize=xtick_size)


def _add_tmd_jmd_label(ax=None, x_shift=0.0, fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                       tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1, height_factor=1.3):
    """Add TMD JMD labels under bars"""
    name_tmd = ut.options["name_tmd"]
    name_jmd_n = ut.options["name_jmd_n"]
    name_jmd_c = ut.options["name_jmd_c"]
    ut.add_tmd_jmd_text(ax=ax, x_shift=x_shift,
                        fontsize_tmd_jmd=fontsize_tmd_jmd, weight_tmd_jmd=weight_tmd_jmd,
                        tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                        name_tmd=name_tmd, name_jmd_n=name_jmd_n, name_jmd_c=name_jmd_c,
                        start=start, height_factor=height_factor)


def _add_name_data(ax=None, name_data=None, name_data_pos=None, fontsize=None, color="black"):
    """Add the name_test to the plot at the specified position."""
    args = dict(transform=ax.transAxes, fontsize=fontsize, color=color, multialignment="center")
    if name_data_pos == "top":
        ax.text(0.5, 1.02, name_data, ha="center", va="bottom", **args)
    elif name_data_pos == "right":
        ax.text(1.02, 0.5, name_data, ha='left', va='center', **args)
    elif name_data_pos == "bottom":
        ax.text(0.5, -0.25, name_data, ha='center', va='top', **args)
    elif name_data_pos == "left":
        ax.text(-0.12, 0.5, name_data, ha='right', va='center', **args)


# II Main Functions
def single_logo_(df_logo=None, df_logo_info=None,
                 info_bar_color="gray", info_bar_ylim=None,
                 target_p1_site=None, figsize=(8, 3.5), height_ratio=(1, 6),
                 fontsize_labels=None, y_label="Counts",
                 name_data=None,  name_data_color="black",
                 name_data_pos="top", name_data_fontsize=None,
                 logo_font_name="Verdana", logo_color_scheme="weblogo_protein",
                 logo_stack_order="big_on_top",
                 logo_width=0.96, logo_vpad=0.05, logo_vsep=0.0,
                 start=1, tmd_len=None, jmd_n_len=10, jmd_c_len=10,
                 tmd_color="mediumspringgreen", jmd_color="blue",
                 fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                 highlight_tmd_area=True, highlight_alpha=0.15,
                 xtick_size=None, xtick_width=2.0, xtick_length=11.0):
    """Plot a single logo with optional annotations and modifications."""
    if df_logo_info is not None:
        _args = dict(nrows=2, gridspec_kw={"height_ratios": height_ratio}, figsize=figsize, sharex=True)
        fig, (ax_info, ax_logo) = plt.subplots(**_args)
    else:
        fig, ax_logo = plt.subplots(figsize=figsize)

    # Plot sequence logo
    if y_label == "Probability [%]":
        df_logo *= 100
    logomaker.Logo(df_logo, ax=ax_logo, figsize=figsize,
                   font_name=logo_font_name, color_scheme=logo_color_scheme,
                   width=logo_width, vpad=logo_vpad, vsep=logo_vsep,
                   stack_order=logo_stack_order)
    if y_label == "Probability [%]":
        y_ticks = list(range(0, 125, 25))
        ax_logo.set_yticks(y_ticks)
    # Adjust plot elements
    args_parts = dict(ax=ax_logo, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    if target_p1_site is not None:
        _add_p_sites(ax_logo=ax_logo, df_logo=df_logo, target_p1_site=target_p1_site, xtick_size=xtick_size)
    else:
        ut.add_tmd_jmd_bar(**args_parts, x_shift=-0.5, jmd_color=jmd_color,
                           tmd_color=tmd_color, bar_height_factor=1.8)
        ut.add_tmd_jmd_xticks(**args_parts, x_shift=0, start=start, xtick_size=xtick_size,
                              xtick_width=xtick_width, xtick_length=xtick_length)
        _add_tmd_jmd_label(**args_parts, x_shift=-0.5, height_factor=2.6, start=start,
                           fontsize_tmd_jmd=fontsize_tmd_jmd, weight_tmd_jmd=weight_tmd_jmd)
    if highlight_tmd_area:
        ut.highlight_tmd_area(**args_parts, x_shift=-0.5, tmd_color=tmd_color,
                              alpha=highlight_alpha, start=start)
    # Adjust labels and formatting
    sns.despine(ax=ax_logo, top=True, bottom=True)
    ax_logo.set_ylabel(y_label, size=fontsize_labels)
    if df_logo_info is not None:
        _add_bit_score_bar(ax_info=ax_info, df_logo_info=df_logo_info, size=fontsize_labels,
                           bar_color=info_bar_color, ylim=info_bar_ylim)
    if name_data is not None:
        fs = ut.plot_gco() + 1 if name_data_fontsize is None else name_data_fontsize
        ax = ax_info if name_data_pos == "top" else ax_logo
        _add_name_data(ax=ax, name_data=name_data, name_data_pos=name_data_pos, color=name_data_color, fontsize=fs)

    # Adjust plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if df_logo_info is not None:
        axes = (ax_logo, ax_info)
    else:
        axes = (ax_logo)
    return fig, axes


# TODO name_data_color - > color_data?
# TODO improve checks if wrong df_logo (e.g., _df)
def multi_logo_(list_df_logo=None, target_p1_site=None, figsize_per_logo=(8, 3),
                fontsize_labels=None, y_label="Counts",
                list_name_data=None, list_name_data_color="black",
                name_data_pos="top", name_data_fontsize=None,
                logo_font_name="Verdana", logo_color_scheme="weblogo_protein",
                logo_stack_order="big_on_top",
                logo_width=0.96, logo_vpad=0.05, logo_vsep=0.0,
                start=1, tmd_len=None, jmd_n_len=10, jmd_c_len=10,
                tmd_color="mediumspringgreen", jmd_color="blue",
                fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                highlight_tmd_area=True, highlight_alpha=0.15,
                xtick_size=None, xtick_width=2.0, xtick_length=11.0):
    """Plot multiple sequence logos stacked vertically with full customization."""
    n_plots = len(list_df_logo)
    figsize = (figsize_per_logo[0], figsize_per_logo[1] * n_plots)
    fig, axes = plt.subplots(nrows=n_plots, figsize=figsize)
    y_max = max([d.T.sum().max() for d in list_df_logo])
    if y_label == "Probability [%]" and y_max < 100:
        y_max *= 100
    for i, df_logo in enumerate(list_df_logo):
        name_data = list_name_data[i] if list_name_data else None
        name_data_color = list_name_data_color[i] if isinstance(list_name_data_color, list) else list_name_data_color
        ax_logo = axes[i]
        # Plot sequence logo
        if y_label == "Probability [%]" and df_logo.T.sum().max() < 100:
            df_logo *= 100
        logomaker.Logo(df_logo, ax=ax_logo, font_name=logo_font_name,
                       color_scheme=logo_color_scheme, width=logo_width,
                       vpad=logo_vpad, vsep=logo_vsep,
                       stack_order=logo_stack_order)
        if y_label == "Probability [%]":
            y_ticks = list(range(0, 125, 25))
            ax_logo.set_yticks(y_ticks)
        # Adjust plot elements
        args_parts = dict(ax=ax_logo, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if target_p1_site is not None:
            _add_p_sites(ax_logo=ax_logo, df_logo=df_logo, target_p1_site=target_p1_site, xtick_size=xtick_size)
        else:
            ut.add_tmd_jmd_bar(**args_parts, x_shift=-0.5,
                               jmd_color=jmd_color, tmd_color=tmd_color,
                               bar_height_factor=1.8*n_plots)
            # Only show x-ticks and TMD, JMD text for bottom subplot
            if i+1 == n_plots:
                ut.add_tmd_jmd_xticks(**args_parts, x_shift=0, start=start, xtick_size=xtick_size,
                                      xtick_width=xtick_width,
                                      xtick_length=xtick_length-n_plots*1.3)
                _add_tmd_jmd_label(**args_parts, x_shift=-0.5, height_factor=2.3*n_plots, start=start,
                                   fontsize_tmd_jmd=fontsize_tmd_jmd, weight_tmd_jmd=weight_tmd_jmd)
            else:
                ax_logo.set_xticks([])
        if highlight_tmd_area:
            ut.highlight_tmd_area(**args_parts, x_shift=-0.5, tmd_color=tmd_color,
                                  alpha=highlight_alpha, start=start, y_max=y_max)
        # Adjust labels and formatting
        sns.despine(ax=ax_logo, top=True, bottom=True)
        ax_logo.set_ylabel(y_label, size=fontsize_labels)
        if name_data:
            fs = ut.plot_gco() + 1 if name_data_fontsize is None else name_data_fontsize
            _add_name_data(ax=ax_logo, name_data=name_data, name_data_pos=name_data_pos,
                           color=name_data_color, fontsize=fs)
    # Same y-axis
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    # Adjust plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.33)
    return fig, axes
