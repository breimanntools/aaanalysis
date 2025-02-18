"""
This is a script for the backend PlotPosition utility class for the CPPPlot class.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis.utils as ut
from .utils_feature import get_positions_, get_df_pos_
from ._utils_cpp_plot import get_sorted_list_cat_


# I Helper Functions
# Optimize fontsize
def _get_optimal_fontsize(ax=None, labels=None, max_x_dist=0.1):
    """Optimize font size of sequence characters"""
    min_fontsize, max_fontsize = 1, 60
    th_binary_search = 0.01
    fs_reduction = 0.05
    # Line width for the bounding box
    lw = ax.figure.get_size_inches()[0] / 5
    # Function to compute the bounding box for each label
    f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())

    def set_label_properties(_label, _fontsize):
        _label.set_fontsize(_fontsize)
        _label.set_bbox(dict(facecolor='none', edgecolor='none', zorder=0.1, alpha=1, clip_on=False, pad=0, linewidth=lw))

    # Function to check overlap between bounding boxes
    def check_overlap(_bbox_list):
        for i in range(len(_bbox_list) - 1):
            x_distance = _bbox_list[i + 1].x0 - _bbox_list[i].x1
            if x_distance < max_x_dist:
                return True
        return False

    # Binary search to narrow down the font size range
    while max_fontsize - min_fontsize > th_binary_search:
        fontsize = (min_fontsize + max_fontsize) / 2
        for label in labels:
            set_label_properties(label, fontsize)
        bbox_list = [f(label) for label in labels]
        if check_overlap(bbox_list):
            max_fontsize = fontsize
        else:
            min_fontsize = fontsize

    # High precision step-wise reduction
    optimal_fontsize = max_fontsize
    while optimal_fontsize > min_fontsize:
        for label in labels:
            set_label_properties(label, optimal_fontsize)
        bbox_list = [f(label) for label in labels]
        if not check_overlap(bbox_list):
            break  # If no overlap is detected, this is the optimal size
        optimal_fontsize -= fs_reduction
    return optimal_fontsize


# Positions
def _get_df_pos_sign(df_feat=None, count=True, col_cat="category", col_val=None, value_type="count",
                     start=None, stop=None):
    """Get DataFrame for positive and negative values based on feature importance."""
    kwargs = dict(col_cat=col_cat, col_val=col_val, value_type=value_type,
                  start=start, stop=stop)
    list_df = []
    df_p = df_feat[df_feat[col_val] > 0]
    df_n = df_feat[df_feat[col_val] <= 0]
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


# TMD-JMD bar functions
def _adjust_xticks_labels(xticks=None, xtick_labels=None, add_xtick_pos=True,
                          exists_jmd_n=True, exists_jmd_c=True):
    """Remove JMD-N and/or JMD-C from x-ticks and x-tick labels if not exist"""
    n = 2 if add_xtick_pos else 1
    if not exists_jmd_n:
        # Remove JMD-N related ticks and labels if it does not exist
        xticks = xticks[n:]
        xtick_labels = xtick_labels[n:]
    if not exists_jmd_c:
        # Remove JMD-C related ticks and labels if it does not exist
        xticks = xticks[:-n]
        xtick_labels = xtick_labels[:-n]
    return xticks, xtick_labels


def _add_part_seq(ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, x_shift=0.0, seq_size=None,
                  tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white"):
    """Add colored boxes for TMD and JMD sequences."""
    tmd_jmd = jmd_n_seq + tmd_seq + jmd_c_seq
    colors = [jmd_color] * len(jmd_n_seq) + [tmd_color] * len(tmd_seq) + [jmd_color] * len(jmd_c_seq)
    dict_seq_color = {tmd_color: tmd_seq_color, jmd_color: jmd_seq_color}
    xticks = range(0, len(tmd_jmd))
    # Set major and minor ticks to enable proper grid lines
    major_xticks = [x for x in xticks if x in ax.get_xticks()]
    minor_xticks = [x for x in xticks if x not in ax.get_xticks()]
    ax.set_xticks([x + x_shift for x in major_xticks], minor=False)
    ax.set_xticks([x + x_shift for x in minor_xticks], minor=True)
    kws_ticks = dict(rotation="horizontal", fontsize=seq_size, fontweight="bold",
                     fontname=ut.FONT_AA, zorder=2)
    ax.set_xticklabels(labels=[tmd_jmd[i] for i in minor_xticks], minor=True, **kws_ticks)
    ax.set_xticklabels(labels=[tmd_jmd[i] for i in major_xticks], minor=False, **kws_ticks)
    ax.tick_params(axis="x", length=0, which="both")
    # Get labels in order of sequence (separate between minor and major ticks)
    dict_pos_label = dict(zip(minor_xticks, ax.xaxis.get_ticklabels(which="minor")))
    dict_pos_label.update(dict(zip(major_xticks, ax.xaxis.get_ticklabels(which="major"))))
    labels = list(dict(sorted(dict_pos_label.items(), key=lambda item: item[0])).values())
    # Adjust font size to prevent overlap
    if seq_size is None:
        seq_size = _get_optimal_fontsize(ax=ax, labels=labels)
    # Set colored box around sequence to indicate TMD, JMD
    lw = plt.gcf().get_size_inches()[0]/5
    for l, c in zip(labels, colors):
        l.set_fontsize(seq_size)
        l.set_bbox(dict(facecolor=c, edgecolor=c, zorder=0.1, alpha=1, clip_on=False,
                        pad=0, linewidth=lw))
        l.set_color(dict_seq_color[c])
    return seq_size


def _add_part_seq_second_ticks(ax2=None, seq_size=11.0, xticks=None, xtick_labels=None,
                               fontsize_tmd_jmd=11, weight_tmd_jmd="normal",
                               xtick_size=11, x_shift=0.5, heatmap=False):
    """Add additional ticks for box of sequence parts"""
    name_tmd = ut.options["name_tmd"]
    name_jmd_n = ut.options["name_jmd_n"]
    name_jmd_c = ut.options["name_jmd_c"]
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    y_pos = 5 if heatmap else 7
    ax2.spines["bottom"].set_position(("outward", seq_size+y_pos))
    ax2.set_xticks([x + x_shift for x in xticks])
    ax2.set_xticklabels(xtick_labels, size=xtick_size, rotation=0, color="black", ha="center", va="top")
    ax2.tick_params(axis="x", color="black", length=0, width=0, bottom=True, pad=0)
    ax2.set_frame_on(False)
    labels = ax2.xaxis.get_ticklabels()
    for l in labels:
        text = l.get_text()
        if name_tmd in text:
            l.set_size(fontsize_tmd_jmd)
            l.set_weight(weight_tmd_jmd)
        elif name_jmd_n in text or name_jmd_c in text:
            l.set_size(fontsize_tmd_jmd)
            l.set_weight(weight_tmd_jmd)


def _get_new_axis(ax=None):
    """Get new axis object with same y-axis as input ax"""
    ax_new = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False, yticks=[])
    return ax_new


# II Main Functions
class PlotPartPositions:
    """Class for plotting positional information for CPP analysis"""

    def __init__(self, tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
        """Initialize the plot positions with given lengths and start position."""
        self.jmd_n_len = jmd_n_len
        self.tmd_len = tmd_len
        self.jmd_c_len = jmd_c_len
        self.seq_len = jmd_n_len + tmd_len + jmd_c_len
        self.start = start
        self.stop = start + self.seq_len - 1

    # Helper methods
    def _get_starts(self, x_shift=0):
        """Calculate the starting positions for JMD and TMD."""
        jmd_n_start = 0 + x_shift
        tmd_start = self.jmd_n_len + x_shift
        jmd_c_start = self.jmd_n_len + self.tmd_len + x_shift
        return jmd_n_start, tmd_start, jmd_c_start

    def _get_middles(self, x_shift=0.0):
        """Calculate the middle positions for JMD and TMD."""
        jmd_n_middle = int(self.jmd_n_len/2) + x_shift
        tmd_middle = self.jmd_n_len + int(self.tmd_len/2) + x_shift
        jmd_c_middle = self.jmd_n_len + self.tmd_len + int(self.jmd_c_len/2) + x_shift
        return jmd_n_middle, tmd_middle, jmd_c_middle

    def _get_ends(self, x_shift=-1):
        """Calculate the ending positions for JMD and TMD."""
        jmd_n_end = self.jmd_n_len + x_shift
        tmd_end = self.jmd_n_len + self.tmd_len + x_shift
        jmd_c_end = self.jmd_n_len + self.tmd_len + self.jmd_c_len + x_shift
        return jmd_n_end, tmd_end, jmd_c_end

    # Main methods
    def get_df_pos(self, df_feat=None, df_cat=None,
                   col_cat="category", col_val="mean_dif",
                   value_type="count", normalize=False):
        """Get df_pos with values (e.g., counts or mean auc) for each feature (y) and positions (x)"""
        df_feat = df_feat.copy()
        # Adjust feature positions
        features = df_feat[ut.COL_FEATURE].to_list()
        feat_positions = get_positions_(features=features,
                                        start=self.start,
                                        tmd_len=self.tmd_len,
                                        jmd_n_len=self.jmd_n_len,
                                        jmd_c_len=self.jmd_c_len)
        df_feat[ut.COL_POSITION] = feat_positions
        # Get dataframe with positions
        kwargs = dict(col_cat=col_cat, col_val=col_val, value_type=value_type, start=self.start, stop=self.stop)
        if value_type == "count":
            if col_val is None:
                df_pos = get_df_pos_(df_feat=df_feat, **kwargs)
            else:
                df_pos = _get_df_pos_sign(df_feat=df_feat, count=True, **kwargs)
        else:
            df_pos = get_df_pos_(df_feat=df_feat, **kwargs)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        # Sort according to given categories
        sorted_col = get_sorted_list_cat_(df_cat=df_cat,
                                          list_cat=list(df_pos.T),
                                          col_cat=col_cat)
        df_pos = df_pos.T[sorted_col].T
        return df_pos.round(3)

    # Add TMD-JMD sequence
    @staticmethod
    def get_optimal_fontsize(ax, labels, max_x_dist=0.1):
        """Get sequence fontsize optimized to not overlap"""
        opt_fs = _get_optimal_fontsize(ax=ax, labels=labels, max_x_dist=max_x_dist)
        return round(opt_fs, 2)

    def add_tmd_jmd_seq(self, ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                        tmd_color="mediumspringgreen", jmd_color="blue",
                        tmd_seq_color="black", jmd_seq_color="white",
                        add_xticks_pos=False, heatmap=True,
                        x_shift=0, xtick_size=11, seq_size=None,
                        fontsize_tmd_jmd=None, weight_tmd_jmd="normal"):
        """Add sequences and corresponding x-ticks for TMD and JMD regions."""
        seq_size = _add_part_seq(ax=ax, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq, x_shift=x_shift,
                                 seq_size=seq_size, tmd_color=tmd_color, jmd_color=jmd_color,
                                 tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        fontsize_tmd_jmd = seq_size if fontsize_tmd_jmd is None else fontsize_tmd_jmd
        # Set second axis (with ticks and part annotations)
        name_tmd = ut.options["name_tmd"]
        name_jmd_n = ut.options["name_jmd_n"]
        name_jmd_c = ut.options["name_jmd_c"]
        jmd_n_end, tmd_end, jmd_c_end = self._get_ends(x_shift=-1)
        jmd_n_middle, tmd_middle, jmd_c_middle = self._get_middles(x_shift=-0.5)
        # Get x-ticks and x-tick labels
        if not add_xticks_pos:
            xticks = [jmd_n_middle, tmd_middle, jmd_c_middle]
            xtick_labels = [name_jmd_n, name_tmd, name_jmd_c]
        else:
            xticks = [0, jmd_n_middle, jmd_n_end, tmd_middle, tmd_end, jmd_c_middle, jmd_c_end]
            xtick_labels = [self.start, name_jmd_n, jmd_n_end + self.start, name_tmd, tmd_end + self.start, name_jmd_c,
                            jmd_c_end + self.start]
        # Adjust x-ticks and x-tick labels
        exists_jmd_n = len(jmd_n_seq) > 0
        exists_jmd_c = len(jmd_c_seq) > 0
        xticks, xtick_labels = _adjust_xticks_labels(xticks=xticks, xtick_labels=xtick_labels,
                                                     add_xtick_pos=add_xticks_pos,
                                                     exists_jmd_n=exists_jmd_n, exists_jmd_c=exists_jmd_c)
        # Set x-ticks and x-tick labels
        if fontsize_tmd_jmd > 0:
            ax2 = _get_new_axis(ax=ax)
            ax2.set_xlim(ax.get_xlim())
            _add_part_seq_second_ticks(ax2=ax2, xticks=xticks, xtick_labels=xtick_labels,
                                       x_shift=x_shift, seq_size=seq_size,xtick_size=xtick_size,
                                       fontsize_tmd_jmd=fontsize_tmd_jmd, weight_tmd_jmd=weight_tmd_jmd,
                                       heatmap=heatmap)
        return ax

    # Add TMD-JMD elements
    def add_tmd_jmd_bar(self, ax=None, x_shift=0, jmd_color="blue", tmd_color="mediumspringgreen"):
        """Add colored bars to indicate TMD and JMD regions."""
        ut.add_tmd_jmd_bar(ax=ax,
                           x_shift=x_shift,
                           jmd_color=jmd_color,
                           tmd_color=tmd_color,
                           tmd_len=self.tmd_len,
                           jmd_n_len=self.jmd_n_len,
                           jmd_c_len=self.jmd_c_len,
                           start=self.start)

    def add_tmd_jmd_text(self, ax=None, x_shift=0, fontsize_tmd_jmd=None, weight_tmd_jmd="normal"):
        """Add text labels for TMD and JMD regions."""
        name_tmd = ut.options["name_tmd"]
        name_jmd_n = ut.options["name_jmd_n"]
        name_jmd_c = ut.options["name_jmd_c"]
        ut.add_tmd_jmd_text(ax=ax,
                            x_shift=x_shift,
                            fontsize_tmd_jmd=fontsize_tmd_jmd,
                            weight_tmd_jmd=weight_tmd_jmd,
                            name_tmd=name_tmd,
                            name_jmd_n=name_jmd_n,
                            name_jmd_c=name_jmd_c,
                            tmd_len=self.tmd_len,
                            jmd_n_len=self.jmd_n_len,
                            jmd_c_len=self.jmd_c_len,
                            start=self.start)

    def add_tmd_jmd_xticks(self, ax=None, x_shift=0, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0):
        """Adjust x-ticks for TMD and JMD regions."""
        ut.add_tmd_jmd_xticks(ax=ax,
                              x_shift=x_shift,
                              xtick_size=xtick_size,
                              xtick_width=xtick_width,
                              xtick_length=xtick_length,
                              tmd_len=self.tmd_len,
                              jmd_n_len=self.jmd_n_len,
                              jmd_c_len=self.jmd_c_len,
                              start=self.start)

    def highlight_tmd_area(self, ax=None, x_shift=0, tmd_color="mediumspringgreen", alpha=0.2):
        """Highlight the TMD area in the plot."""
        ut.highlight_tmd_area(ax=ax,
                              x_shift=x_shift,
                              tmd_color=tmd_color,
                              alpha=alpha,
                              tmd_len=self.tmd_len,
                              jmd_n_len=self.jmd_n_len,
                              jmd_c_len=self.jmd_c_len,
                              start=self.start)

    # Add x-ticks
    def get_xticks_with_labels(self, step=5):
        """Generate x-ticks and their labels for the plot."""
        second_pos = int((self.start+step)/step)*step
        xticks_middle = list(range(second_pos, self.stop, step))
        xticks_labels = [self.start] + xticks_middle + [self.stop]
        xticks = [x-self.start for x in xticks_labels]
        return xticks, xticks_labels

    def add_xticks(self, ax=None, x_shift=0.0, xticks_position="bottom",
                   xtick_size=11.0, xtick_width=2.0, xtick_length=4.0):
        """Add x-ticks to the plot."""
        xticks, xticks_labels = self.get_xticks_with_labels(step=5)
        ax.set_xticks([x + x_shift for x in xticks])
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
        ax.tick_params(axis="x", color="black", length=xtick_length, width=xtick_width)
        ax.xaxis.set_ticks_position(xticks_position)
