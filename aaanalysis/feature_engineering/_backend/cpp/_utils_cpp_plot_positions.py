"""
This is a script for ...
"""
import statistics
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis as aa
import aaanalysis.utils as ut

from ._utils_feature import get_positions_, get_df_pos_
from ._utils_cpp_plot import get_optimal_fontsize, get_new_axis


# I Helper Functions
# Positions
def _get_df_pos_sign(df_feat=None, count=True, col_cat="category", col_value=None, value_type="count",
                     start=None, stop=None, normalize_for_pos=False):
    """Get position DataFrame for positive and negative values"""
    kwargs = dict(col_cat=col_cat, col_value=col_value, value_type=value_type,
                  start=start, stop=stop,
                  normalize_for_pos=normalize_for_pos)
    list_df = []
    df_p = df_feat[df_feat[col_value] > 0]
    df_n = df_feat[df_feat[col_value] <= 0]
    if len(df_p) > 0:
        df_positive = get_df_pos_(df_feat=df_p, **kwargs)
        list_df.append(df_positive)
    if len(df_n) > 0:
        df_negative = get_df_pos_(df_feat=df_n, **kwargs)
        if count:
            df_negative = -df_negative
        list_df.append(df_negative)
    df_pos = pd.concat(list_df)
    return df_pos


# TMD JMD bar functions
def _get_bar_height(ax=None, divider=50):
    """"""
    ylim = ax.get_ylim()
    width, height = plt.gcf().get_size_inches()
    bar_height = abs(ylim[0] - ylim[1]) / divider / height*6
    return bar_height


def _get_y(ax=None, bar_height=None, height_factor=1.0, reversed_weight=0):
    """"""
    ylim = ax.get_ylim()
    reversed_y = reversed_weight if ylim[0] > ylim[1] else 1
    y = ylim[0] - (bar_height * height_factor) * reversed_y
    return y


def _add_part_bar(ax=None, start=1.0, len_part=40.0, color="blue"):
    """Get colored bar for tmd and jmd showing sequence parts"""
    bar_height = _get_bar_height(ax=ax)
    y = _get_y(ax=ax, bar_height=bar_height)
    bar = mpl.patches.Rectangle((start, y), width=len_part, height=bar_height, linewidth=0,
                                color=color, zorder=3, clip_on=False)
    ax.add_patch(bar)


def _add_part_text(ax=None, start=1.0, len_part=10.0, fontsize=None, text=None):
    """Get text for tmd and jmd marking sequence parts"""
    bar_height = _get_bar_height(ax=ax)
    y = _get_y(ax=ax, bar_height=bar_height, height_factor=1.3, reversed_weight=-1)
    x = start + len_part / 2    # Middle of part
    ax.text(x, y, text,
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='normal',
            color='black')


def _add_part_seq(ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, x_shift=0.0, seq_size=None,
                  tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white"):
    """Add colored box for sequence parts in figure"""
    tmd_jmd = jmd_n_seq + tmd_seq + jmd_c_seq
    colors = [jmd_color] * len(jmd_n_seq) + [tmd_color] * len(tmd_seq) + [jmd_color] * len(jmd_c_seq)
    dict_seq_color = {tmd_color: tmd_seq_color, jmd_color: jmd_seq_color}
    xticks = range(0, len(tmd_jmd))
    # Set major and minor ticks to enable proper grid lines
    major_xticks = [x for x in xticks if x in ax.get_xticks()]
    minor_xticks = [x for x in xticks if x not in ax.get_xticks()]
    ax.set_xticks([x + x_shift for x in major_xticks], minor=False)
    ax.set_xticks([x + x_shift for x in minor_xticks], minor=True)
    kws_ticks = dict(rotation="horizontal", fontsize=seq_size, fontweight="bold", fontname=ut.FONT_AA,
                     zorder=2)
    ax.set_xticklabels(labels=[tmd_jmd[i] for i in minor_xticks], minor=True, **kws_ticks)
    ax.set_xticklabels(labels=[tmd_jmd[i] for i in major_xticks], minor=False, **kws_ticks)
    ax.tick_params(length=0, which="both")
    # Get labels in order of sequence (separate between minor and major ticks)
    dict_pos_label = dict(zip(minor_xticks, ax.xaxis.get_ticklabels(which="minor")))
    dict_pos_label.update(dict(zip(major_xticks, ax.xaxis.get_ticklabels(which="major"))))
    labels = list(dict(sorted(dict_pos_label.items(), key=lambda item: item[0])).values())
    # Adjust font size to prevent overlap
    if seq_size is None:
        seq_size = get_optimal_fontsize(ax, labels)
    # Set colored box around sequence to indicate TMD, JMD
    lw = plt.gcf().get_size_inches()[0]/5
    for l, c in zip(labels, colors):
        l.set_fontsize(seq_size)
        l.set_bbox(dict(facecolor=c, edgecolor=c, zorder=0.1, alpha=1, clip_on=False, pad=0, linewidth=lw))
        l.set_color(dict_seq_color[c])
    return seq_size


def _add_part_seq_second_ticks(ax2=None, seq_size=11.0, xticks=None, xtick_labels=None,
                               fontsize_tmd_jmd=11, xtick_size=11, x_shift=0.5):
    """Add additional ticks for box of sequence parts"""
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("outward", seq_size + 7))
    ax2.set_xticks([x + x_shift for x in xticks])
    ax2.set_xticklabels(xtick_labels, size=xtick_size, rotation=0, color="black", ha="center", va="top")
    ax2.tick_params(axis="x", color="black", length=0, width=0, bottom=True, pad=0)
    ax2.set_frame_on(False)
    labels = ax2.xaxis.get_ticklabels()
    for l in labels:
        text = l.get_text()
        if "TMD" in text:
            l.set_size(fontsize_tmd_jmd)
            l.set_weight("bold")
        elif "JMD" in text:
            l.set_size(fontsize_tmd_jmd)
            l.set_weight("bold")


# II Main Functions
class PlotPositions:
    """Class for plotting functions for CPP analysis"""
    def __init__(self, tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
        self.jmd_n_len = jmd_n_len
        self.tmd_len = tmd_len
        self.jmd_c_len = jmd_c_len
        self.seq_len = jmd_n_len + tmd_len + jmd_c_len
        self.start = start
        self.stop = start + self.seq_len - 1

    # Helper methods
    def _get_starts(self, x_shift=0):
        """"""
        jmd_n_start = 0 + x_shift
        tmd_start = self.jmd_n_len + x_shift
        jmd_c_start = self.jmd_n_len + self.tmd_len + x_shift
        return jmd_n_start, tmd_start, jmd_c_start

    def _get_middles(self, x_shift=0.0):
        """"""
        jmd_n_middle = int(self.jmd_n_len/2) + x_shift
        tmd_middle = self.jmd_n_len + int(self.tmd_len/2) + x_shift
        jmd_c_middle = self.jmd_n_len + self.tmd_len + int(self.jmd_c_len/2) + x_shift
        return jmd_n_middle, tmd_middle, jmd_c_middle

    def _get_ends(self, x_shift=-1):
        """"""
        jmd_n_end = self.jmd_n_len + x_shift
        tmd_end = self.jmd_n_len + self.tmd_len + x_shift
        jmd_c_end = self.jmd_n_len + self.tmd_len + self.jmd_c_len + x_shift
        return jmd_n_end, tmd_end, jmd_c_end

    # Main methods
    # TODO check df_pos and simplify
    def get_df_pos(self, df_feat=None, df_cat=None, col_cat="category", col_value="mean_dif",
                   value_type="count", normalize=False):
        """Get df_pos with values (e.g., counts or mean auc) for each feature (y) and positions (x)"""
        # TODO simplify
        df_feat = df_feat.copy()
        # Adjust feature positions
        features = df_feat[ut.COL_FEATURE].to_list()
        feat_positions = get_positions_(features=features,
                                        start=self.start, tmd_len=self.tmd_len, jmd_n_len=self.jmd_n_len,
                                        jmd_c_len=self.jmd_c_len)
        df_feat[ut.COL_POSITION] = feat_positions
        # Get dataframe with
        # TODO check if right
        kwargs = dict(col_cat=col_cat, col_value=col_value, value_type=value_type, start=self.start, stop=self.stop)
        if value_type == "count":
            if col_value is None:
                df_pos = get_df_pos_(df_feat=df_feat, **kwargs)
            else:
                df_pos = _get_df_pos_sign(df_feat=df_feat, count=True, **kwargs)
        else:
            df_pos = get_df_pos_(df_feat=df_feat, **kwargs)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        # Sort according to given categories
        list_cat = list(df_cat[col_cat].drop_duplicates())
        list_col = list(df_pos.T)
        sorted_col = [x for x in list_cat if x in list_col]
        df_pos = df_pos.T[sorted_col].T
        return df_pos.round(3)

    # Add tmd jmd bars
    def get_xticks_with_labels(self, step=5):
        """"""
        second_pos = int((self.start+step)/step)*step
        xticks_middle = list(range(second_pos, self.stop, step))
        xticks_labels = [self.start] + xticks_middle + [self.stop]
        xticks = [x-self.start for x in xticks_labels]
        return xticks, xticks_labels

    def add_xticks(self, ax=None, x_shift=0.0, xticks_position="bottom", xtick_size=11.0, xtick_width=2.0,
                   xtick_length=4.0):
        """"""
        # Set default x ticks (if tmd_jmd not shown)
        xticks, xticks_labels = self.get_xticks_with_labels(step=5)
        ax.set_xticks([x + x_shift for x in xticks])
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
        ax.tick_params(axis="x", color="black", length=xtick_length, width=xtick_width)
        ax.xaxis.set_ticks_position(xticks_position)

    def add_tmd_jmd_bar(self, ax=None, x_shift=0, jmd_color="blue", tmd_color="mediumspringgreen"):
        """"""
        jmd_n_start, tmd_start, jmd_c_start = self._get_starts(x_shift=x_shift)
        _add_part_bar(ax=ax, start=jmd_n_start, len_part=self.jmd_n_len, color=jmd_color)
        _add_part_bar(ax=ax, start=tmd_start, len_part=self.tmd_len, color=tmd_color)
        _add_part_bar(ax=ax, start=jmd_c_start, len_part=self.jmd_c_len, color=jmd_color)

    def add_tmd_jmd_text(self, ax=None, x_shift=0, fontsize_tmd_jmd=None):
        """"""
        jmd_n_start, tmd_start, jmd_c_start = self._get_starts(x_shift=x_shift)
        if fontsize_tmd_jmd is None or fontsize_tmd_jmd > 0:
            _add_part_text(ax=ax, start=tmd_start, len_part=self.tmd_len, text="TMD", fontsize=fontsize_tmd_jmd)
            _add_part_text(ax=ax, start=jmd_n_start, text="JMD-N", len_part=self.jmd_n_len, fontsize=fontsize_tmd_jmd)
            _add_part_text(ax=ax, start=jmd_c_start, text="JMD-C", len_part=self.jmd_c_len, fontsize=fontsize_tmd_jmd)

    def add_tmd_jmd_xticks(self, ax=None, x_shift=0, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0):
        """"""
        # Adjust tick tick_length based on figure size
        width, height = plt.gcf().get_size_inches()
        xtick_length += height if xtick_length != 0 else 0
        # Adjust for start counting at position 0
        jmd_n_end, tmd_end, jmd_c_end = self._get_ends(x_shift=-1)
        xticks = [0, jmd_n_end, tmd_end, jmd_c_end]
        ax.set_xticks([x + x_shift for x in xticks])
        ax.set_xticklabels([x + self.start for x in xticks], size=xtick_size, rotation=0)
        ax.tick_params(axis="x", length=xtick_length, color="black", width=xtick_width, bottom=True)

    def highlight_tmd_area(self, ax=None, x_shift=0, tmd_color="mediumspringgreen", alpha=0.2):
        """"""
        jmd_n_start, tmd_start, jmd_c_start = self._get_starts(x_shift=x_shift)
        y_min, y_max = plt.ylim()
        height = abs(y_min) + y_max
        rect = mpl.patches.Rectangle((tmd_start, y_min), width=self.tmd_len, height=height, linewidth=0,
                                     color=tmd_color, zorder=0.1, clip_on=True, alpha=alpha)
        ax.add_patch(rect)

    def add_tmd_jmd_seq(self, ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, xticks_pos=False, heatmap=True,
                        tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                        x_shift=0, xtick_size=11, seq_size=None, fontsize_tmd_jmd=None):
        """"""
        seq_size = _add_part_seq(ax=ax, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq, x_shift=x_shift,
                                 seq_size=seq_size, tmd_color=tmd_color, jmd_color=jmd_color,
                                 tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        fontsize_tmd_jmd = seq_size * 1.1 if fontsize_tmd_jmd is None else fontsize_tmd_jmd
        # Set second axis (with ticks and part annotations)
        jmd_n_end, tmd_end, jmd_c_end = self._get_ends(x_shift=-1)
        jmd_n_middle, tmd_middle, jmd_c_middle = self._get_middles(x_shift=-0.5)
        if not xticks_pos:
            xticks = [jmd_n_middle, tmd_middle, jmd_c_middle]
            xtick_labels = ["JMD-N", "TMD", "JMD-C"]
        else:
            xticks = [0, jmd_n_middle, jmd_n_end, tmd_middle, tmd_end, jmd_c_middle, jmd_c_end]
            xtick_labels = [self.start, "JMD-N", jmd_n_end + self.start, "TMD", tmd_end + self.start, "JMD-C",
                            jmd_c_end + self.start]
        ax2 = get_new_axis(ax=ax, heatmap=heatmap)
        ax2.set_xlim(ax.get_xlim())
        _add_part_seq_second_ticks(ax2=ax2, xticks=xticks, xtick_labels=xtick_labels, x_shift=x_shift, seq_size=seq_size,
                                   fontsize_tmd_jmd=fontsize_tmd_jmd, xtick_size=xtick_size)
        return ax


