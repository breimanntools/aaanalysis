"""
This is a script for CPP plotting class (heat maps, profiles)
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

import aaanalysis.utils as ut

# TODO add to plot_element
# I Helper Functions
def get_color_dif(mean_dif=0):
    return ut.COLOR_FEAT_NEG if mean_dif < 0 else ut.COLOR_FEAT_POS


def add_feature_title(y=None, fontsize_title=None, pad_factor=2.0):
    """"""
    f_space = lambda x: " "*x
    plt.text(0, y, "Scale (subcategory)" + f_space(3), size=fontsize_title, weight="bold", ha="right")
    plt.text(0, y, f_space(3) + "Positions", size=fontsize_title, weight="bold", ha="left")
    plt.text(0, y * pad_factor, "Feature", size=fontsize_title + 1, weight="bold", ha="center")
    plt.text(0, y, "+", size=fontsize_title, weight="bold", ha="center")


def get_optimal_fontsize(ax, labels):
    """Get optimal sequence size for plot"""
    plt.tight_layout()
    fontsize = 30
    f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer())
    while 1 < fontsize:
        for label in labels:
            label.set_fontsize(fontsize)
        l1 = f(labels[0])
        l2 = f(labels[1])
        width_char = l1.x1 - l1.x0
        dist_char = l2.x0 - l1.x1
        if dist_char < width_char/25:
            fontsize -= 0.1
        else:
            break
    return fontsize


def get_new_axis(ax=None, heatmap=False):
    """Get new axis object with same y axis as input ax"""
    if heatmap:
        ax_new = ax.twiny()
    else:
        ymin, ymax = plt.ylim()
        ytick_old = list([round(x, 3) for x in plt.yticks()[0]])
        yticks = [ymin] + [y for y in ytick_old if ymin < y < ymax] + [ymax]
        ax_new = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False, yticks=yticks)
        ax_new.set_ylim(ymin, ymax)
        # Remove last and/or first element from yticks
        if ymax not in ytick_old:
            yticks.pop()
        if ymin not in ytick_old:
            yticks.pop(0)
        ax_new.yaxis.set_ticks(yticks)
        # Hide y-labels of first axis object
        ax.yaxis.set_ticks(yticks, minor=True)
        ax.tick_params(axis="y", colors="white")
    return ax_new


def add_part_seq(ax=None, jmd_n_len=10, jmd_c_len=10, tmd_len=20, y=-0.75, height=0.2,
                 tmd_color="mediumspringgreen", jmd_color="blue", alpha=1.0, start=0.0):
    """Add colored box for sequence parts in figure"""
    list_color = [jmd_color, tmd_color, jmd_color]
    list_length = [jmd_n_len, tmd_len, jmd_c_len]
    # Add jmd_n
    for length, color in zip(list_length, list_color):
        bar = mpl.patches.Rectangle((start, y), width=length, height=height, linewidth=0, color=color, zorder=4,
                                    clip_on=False, alpha=alpha)
        start += length
        ax.add_patch(bar)


# TODO check if needed
def draw_shap_legend(x=None, y=10, offset_text=1, fontsize=13):
    """Draw higher lower element for indicating colors"""
    arrow_dif = y * 0.02
    plt.text(x - offset_text, y, 'higher',
             fontweight='bold',
             fontsize=fontsize, color=ut.COLOR_SHAP_POS,
             horizontalalignment='right')

    plt.text(x + offset_text * 1.1, y, 'lower',
             fontweight='bold',
             fontsize=fontsize, color=ut.COLOR_SHAP_NEG,
             horizontalalignment='left')

    plt.text(x, y - arrow_dif, r'$\leftarrow$',
             fontweight='bold',
             fontsize=fontsize+1, color=ut.COLOR_SHAP_NEG,
             horizontalalignment='center')

    plt.text(x, y + arrow_dif, r'$\rightarrow$',
             fontweight='bold',
             fontsize=fontsize+1, color=ut.COLOR_SHAP_POS,
             horizontalalignment='center')