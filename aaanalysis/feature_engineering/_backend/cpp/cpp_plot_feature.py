"""
This is a script for backend of the cpp_plot.feature method.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot import get_color_dif
from ._utils_cpp import get_list_parts, get_df_parts_, get_feature_matrix_, get_amino_acids_

COL_FEAT_VAL = "feature_values"


# I Helper Functions
# Data processing
def _get_df_feat(feature, df_seq, df_scales, accept_gaps):
    list_parts = get_list_parts(features=feature)
    df_parts = get_df_parts_(df_seq=df_seq, list_parts=list_parts)
    X = get_feature_matrix_(features=feature,
                            df_parts=df_parts,
                            df_scales=df_scales,
                            accept_gaps=accept_gaps)
    list_names = df_seq[ut.COL_NAME].to_list()
    df = pd.DataFrame({COL_FEAT_VAL: X.flatten(), ut.COL_NAME: list_names})
    return df


def _get_mean_vals_(df, labels):
    """Get mean values for test and reference group and their difference"""
    mask_test = [x == 0 for x in labels]
    mask_ref = [x == 1 for x in labels]
    mean_test = df[COL_FEAT_VAL][mask_test].mean()
    mean_ref = df[COL_FEAT_VAL][mask_ref].mean()
    mean_dif = round(mean_test - mean_ref, 3)
    return mean_test, mean_ref, mean_dif


def _get_df_show(df, feature, df_seq, names_to_show, show_seq):
    names_to_show = [] if names_to_show is None else names_to_show
    df_show = df[df[ut.COL_NAME].isin(names_to_show)][[COL_FEAT_VAL, ut.COL_NAME, ut.COL_LABEL]]
    names_to_show = df_show[ut.COL_NAME].to_list()
    if show_seq and len(names_to_show) > 0:
        df_seq = df_seq[df_seq[ut.COL_NAME].isin(names_to_show)]
        list_parts = ["jmd_n", "tmd", "jmd_c"]
        df_parts = get_df_parts_(df_seq=df_seq, list_parts=list_parts)
        names_with_seq = []
        for i, (_, row) in enumerate(df_parts.iterrows()):
            jmd_n, tmd, jmd_c = row[list_parts]
            list_aa = get_amino_acids_(features=feature, jmd_n_seq=jmd_n, tmd_seq=tmd, jmd_c_seq=jmd_c)
            name = str(names_to_show[i]) + "\n" + list_aa[0]
            names_with_seq.append(name)
        df_show[ut.COL_NAME] = names_with_seq
    return df_show


# Additional plot information
def _add_mean_dif(ax, mean_test, mean_ref, mean_dif, y_mean_dif, alpha_dif):
    """Add area to highlight mean differences between reference and test group"""
    color_dif = get_color_dif(mean_dif=mean_dif)
    plt.plot([mean_test, mean_ref], [y_mean_dif, y_mean_dif], "-", color=color_dif, linewidth=4)
    args_lines = dict(color="black", markeredgewidth=5)
    plt.plot(mean_ref, y_mean_dif, "|", **args_lines)
    plt.plot(mean_test, y_mean_dif, "|", **args_lines)

    plt.plot([mean_test, mean_test], [0, y_mean_dif], "-", color=color_dif, linewidth=2, alpha=0.25)
    plt.plot([mean_ref, mean_ref], [0, y_mean_dif], "-", color=color_dif, linewidth=2, alpha=0.25)

    bar = mpl.patches.Rectangle((mean_ref, 0), width=mean_dif, height=y_mean_dif, linewidth=1, color=color_dif,
                                zorder=0, clip_on=False, alpha=alpha_dif)

    ax.add_patch(bar)
    plt.plot([mean_test, mean_test], [0, y_mean_dif], "--", color="black", linewidth=2, alpha=1)
    plt.plot([mean_ref, mean_ref], [0, y_mean_dif], "--", color="black", linewidth=2, alpha=1)


def _add_mean_dif_text(mean_test, mean_ref, mean_dif, y_mean_dif, name_test, name_ref,
                       color_test, color_ref, fontsize_name_test, fontsize_name_ref, fontsize_mean_dif):
    """Add text annotations for mean difference"""
    middle = min([mean_test, mean_ref]) + abs(mean_dif) / 2
    str_mean_dif = f"{ut.LABEL_MEAN_DIF}={mean_dif}"
    plt.text(middle, y_mean_dif * 1.05, str_mean_dif, weight="bold", color="black", ha="center", size=fontsize_mean_dif)
    d = 0.015
    x_test = mean_test - d if mean_dif < 0 else mean_test + d
    ha_test = "right" if mean_dif < 0 else "left"
    x_ref = mean_ref - d if mean_dif > 0 else mean_ref + d
    ha_ref = "right" if mean_dif > 0 else "left"
    plt.text(x_test, y_mean_dif, name_test, ha=ha_test, va="center_baseline", color=color_test, weight="bold",
             size=fontsize_name_test)
    plt.text(x_ref, y_mean_dif, name_ref, ha=ha_ref, va="center_baseline", color=color_ref, weight="bold",
             size=fontsize_name_ref)


def _add_names_to_show(ax, df_show, name_test, color_test, color_ref, fontsize_names_to_show, show_seq):
    """Add individual data points with arrows to plot"""
    for i, row in df_show.iterrows():
        val, name, label = row.values
        color = color_test if label == name_test else color_ref
        fontname = ut.FONT_AA if show_seq else None
        ax.annotate(name, xy=(val, 0), xytext=(0, 30), textcoords='offset points',
                    ha="center", va="center", fontname=fontname,
                    xycoords="data", fontsize=fontsize_names_to_show,
                    arrowprops=dict(arrowstyle="->", lw=3, color=color, shrinkA=0, shrinkB=0),
                    color="black")


# II Main Functions
def plot_feature(ax=None, figsize=(5.6, 4.8), feature=str,
                 df_seq=None, labels=None, df_scales=None, accept_gaps=None,
                 name_test="TEST", name_ref="REF", names_to_show=None,
                 show_seq=True,
                 color_test="tab:green", color_ref="tab:gray",
                 histplot=False, alpha_hist=0.1, alpha_dif=0.25,
                 fontsize_mean_dif=15,  fontsize_name_test=13,
                 fontsize_name_ref=13, fontsize_names_to_show=13,
                 ):
    """Plot distributions of feature values for test and reference datasets highlighting their mean difference."""
    # Get data
    df = _get_df_feat(feature, df_seq, df_scales, accept_gaps)
    df[ut.COL_LABEL] = [name_test if x == 1 else name_ref for x in labels]
    mean_test, mean_ref, mean_dif = _get_mean_vals_(df, labels)
    # Get data for individual proteins to show
    df_show = _get_df_show(df, feature, df_seq, names_to_show, show_seq)
    # Plotting
    pe = PlotElements()
    pe.set_figsize(figsize=figsize)
    args = dict(palette=[color_test, color_ref], hue_order=[name_test, name_ref],
                data=df, hue=ut.COL_LABEL,
                x=COL_FEAT_VAL,
                legend=False, ax=ax)
    if histplot:
        sns.histplot(**args)
    else:
        sns.kdeplot(**args, fill=True, alpha=alpha_hist, common_norm=False)
    ax = plt.gca()
    fs = ut.plot_gco(option='font.size')  # Either None or current font size (if not provided, default size is used)
    y_mean_dif = plt.ylim()[1] * 1.05
    plt.ylim(0, y_mean_dif * 1.2)
    plt.xlim(0, 1)
    # Add mean dif
    _add_mean_dif(ax, mean_test, mean_ref, mean_dif, y_mean_dif, alpha_dif)
    _add_mean_dif_text(mean_test, mean_ref, mean_dif, y_mean_dif, name_test, name_ref,
                       color_test, color_ref, fontsize_name_test, fontsize_name_ref, fontsize_mean_dif)
    # Add plot labels
    y_label = ut.LABEL_HIST_COUNT if histplot else ut.LABEL_HIST_DEN
    plt.ylabel(y_label, size=fs)
    plt.xlabel(ut.LABEL_FEAT_VAL, size=fs)
    # Add individual data points
    if len(df_show) != 0:
        _add_names_to_show(ax, df_show, name_test, color_test, color_ref, fontsize_names_to_show, show_seq)
    # Update ticks
    pe.set_x_ticks(ax=ax, fs=fs)
    pe.set_y_ticks(ax=ax, fs=fs)
    # Adjust plot
    sns.despine()
    plt.tight_layout()
    return ax

