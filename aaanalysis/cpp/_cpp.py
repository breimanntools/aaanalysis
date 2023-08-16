"""
This is a script for CPP plotting class (heat maps, profiles)
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

import aaanalysis.cpp._utils as ut


# I Helper Functions
def check_positions(df=None):
    """Check if 'positions' in df"""
    if "positions" not in list(df):
        raise ValueError("'positions' must be a column in 'df'")


def check_value_type(value_type=None):
    """Check if value_type 'count', 'mean', or 'sum'"""
    list_value_type = ["count", "mean", "sum", "std"]
    if value_type not in list_value_type:
        raise ValueError(f"'value_type' ('{value_type}') must be one of following {list_value_type}")


def check_val_col(df=None, val_col=None, value_type=None):
    """Check if val_col in df"""
    list_num_columns = [col for col, data_type in zip(list(df), df.dtypes) if data_type == float]
    list_value_type = ["mean", "sum", "std"]
    if val_col is not None and val_col not in list_num_columns:
        raise ValueError("'val_col' ('{}') must be one of following columns with numerical values"
                         " of 'df': {}".format(val_col, list_num_columns))
    if val_col is not None and value_type is None:
        raise ValueError("If 'val_col' is given, 'value_type' must be one of following: {}".format(list_value_type))


# Data processing functions
def _get_df_pos_long(df=None, y="category", val_col=None):
    """Get """
    if val_col is None:
        df_feat = df[["feature", y]].set_index("feature")
    else:
        df_feat = df[["feature", y, val_col]].set_index("feature")
    df_pos_long = df["positions"].str.split(",").apply(pd.Series)
    df_pos_long.index = df["feature"]
    df_pos_long = df_pos_long.stack().reset_index(level=1).drop("level_1", axis=1).rename({0: "positions"}, axis=1)
    df_pos_long = df_pos_long.join(df_feat)
    return df_pos_long


def _get_df_pos(df=None, y="category", value_type="count", val_col=None, start=None, stop=None,
                normalize_for_pos=False):
    """Get df with counts for each combination of column values and positions"""
    check_value_type(value_type=value_type)
    list_y_cat = sorted(set(df[y]))
    if normalize_for_pos:
        df[val_col] = df[val_col] / [len(x.split(",")) for x in df["positions"]]

    # Get df with features for each position
    df_pos_long = _get_df_pos_long(df=df, y=y, val_col=val_col)
    # Get dict with values of categories for each position
    dict_pos_val = {str(p): [] for p in range(start, stop+1)}
    dict_cat_val = {c: 0 for c in list_y_cat}
    for p in dict_pos_val:
        if value_type == "count":
            dict_val = dict(df_pos_long[df_pos_long["positions"] == p][y].value_counts())
        elif value_type == "mean":
            dict_val = dict(df_pos_long[df_pos_long["positions"] == p].groupby(y).mean()[val_col])
        elif value_type == "sum":
            dict_val = dict(df_pos_long[df_pos_long["positions"] == p].groupby(y).sum()[val_col])
        else:
            dict_val = dict(df_pos_long[df_pos_long["positions"] == p].groupby(y).std()[val_col])
        dict_pos_val[p] = {**dict_cat_val, **dict_val}
    # Get df with values (e.g., counts) of each category and each position
    df_pos = pd.DataFrame(dict_pos_val)
    df_pos = df_pos.T[list_y_cat].T     # Filter and order categories
    return df_pos


def _get_df_pos_sign(df=None, count=False, val_col=None, **kwargs):
    """Get position DataFrame for positive and negative values"""
    list_df = []
    df_p = df[df[val_col] > 0]
    df_n = df[df[val_col] <= 0]
    if len(df_p) > 0:
        df_positive = _get_df_pos(df=df_p, val_col=val_col, **kwargs)
        list_df.append(df_positive)
    if len(df_n) > 0:
        df_negative = _get_df_pos(df=df_n, val_col=val_col, **kwargs)
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


def add_part_bar(ax=None, start=1.0, len_part=40.0, color="blue", add_white_bar=True):
    """Get colored bar for tmd and jmd showing sequence parts"""
    bar_height = _get_bar_height(ax=ax)
    y = _get_y(ax=ax, bar_height=bar_height)
    bar = mpl.patches.Rectangle((start, y), width=len_part, height=bar_height, linewidth=0,
                                color=color, zorder=3, clip_on=False)
    ax.add_patch(bar)
    """
    if add_white_bar:
        bar = mpl.patches.Rectangle((start, y), width=len_part, height=bar_height/4, linewidth=0,
                                    color="white", zorder=3, clip_on=False)
        ax.add_patch(bar)
    """


def add_part_text(ax=None, start=1.0, len_part=10.0, fontsize=11, text=None):
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


def add_part_seq(ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, x_shift=0.0, seq_size=None,
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
    kws_ticks = dict(rotation="horizontal", fontsize=seq_size, fontweight="bold", fontname="DejaVu Sans Mono",
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


def add_part_seq_second_ticks(ax2=None, seq_size=11.0, xticks=None, xtick_labels=None,
                              tmd_jmd_fontsize=11, xtick_size=11, x_shift=0.5):
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
            l.set_size(tmd_jmd_fontsize)
            l.set_weight("bold")
        elif "JMD" in text:
            l.set_size(tmd_jmd_fontsize)
            l.set_weight("bold")


# Figure Modifications
def _get_legend_handles_labels(dict_color=None, list_cat=None):
    """Get legend handles from dict_color"""
    dict_leg = {cat: dict_color[cat] for cat in dict_color if cat in list_cat}
    f = lambda l, c: mpl.patches.Patch(color=l, label=c, lw=0)
    handles = [f(l, c) for c, l in dict_leg.items()]
    labels = list(dict_leg.keys())
    return handles, labels


# Heatmap settings functions
def get_cmap_heatmap(df_pos=None, cmap=None, n_colors=None, higher_color=None, lower_color=None, facecolor_dark=True):
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


def get_cbar_ticks_heatmap(df_pos=None):
    """Get legend ticks for heatmap"""
    stop_legend = df_pos.values.max()
    if type(stop_legend) is int:
        # Ticks for count datasets
        cbar_ticks = [int(x) for x in np.linspace(1, stop_legend, num=stop_legend)]
    else:
        cbar_ticks = None
    return cbar_ticks


# TODO refactor cbar (easier handling for combination
def get_cbar_args_heatmap(cbar_kws=None, df_pos=None):
    """Parameter to set manually"""
    # Get cbar ticks
    cbar_ticks = get_cbar_ticks_heatmap(df_pos=df_pos)
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


def infer_vmin_vmax_from_data(df_pos=None, kwargs=None):
    """Infer vmin and vmax from data if None"""
    vmin, vmax = kwargs["vmin"], kwargs["vmax"]
    if vmin is None:
        kwargs["vmin"] = df_pos.min().min()
    if vmax is None:
        kwargs["vmax"] = df_pos.max().max()
    return kwargs


def set_cbar_heatmap(ax=None, dict_cbar=None, cbar_kws_=None, vmin=None, vmax=None, cbar_pct=True):
    """"""
    # Set colorbar labelsize and ticksize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=dict_cbar["ticksize"])
    if "label" in cbar_kws_:
        cbar.set_label(label=cbar_kws_["label"], weight="bold", size=dict_cbar["labelsize"])
    cbar.ax.yaxis.label.set_size(dict_cbar["labelsize"])
    cbar_ticks = [x for x in cbar.get_ticks() if vmin <= x <= vmax]
    cbar.set_ticks(cbar_ticks)
    str_zero = "[0]"
    str_pct = "%" if cbar_pct else ""
    f = lambda x: int(x) if int(x) == float(x) else round(x, 1)
    cbar.set_ticklabels([f"{f(x)}{str_pct}" if float(x) != 0 else str_zero for x in cbar_ticks])


def get_center_heatmap(df_pos=None):
    """Get center of heatmap colormap"""
    if df_pos.min().min() < 0:
        center = 0
    else:
        center = None
    return center


def _get_ytick_pad_heatmap(ax=None):
    """"""
    xmax = ax.get_xlim()[1]
    width, height = plt.gcf().get_size_inches()
    pad = width+9-xmax/10
    return pad


def _get_kws_legend_under_plot(list_cat=None, items_per_col=3, x_adjust=-0.3, y_adjust=-0.03):
    """"""
    width, height = plt.gcf().get_size_inches()
    ncol = len(set(list_cat))
    bbox_y = width / height * y_adjust
    bbox_x = -0.2 + x_adjust
    if ncol > 4:
        ncol = int(np.ceil(ncol/items_per_col))
    kws_legend = dict(ncol=ncol, loc=2, frameon=False, bbox_to_anchor=(bbox_x, bbox_y),
                      columnspacing=0.3, labelspacing=0.05, handletextpad=0.15,
                      facecolor="white",  shadow=False,
                      title="scale category")
    return kws_legend


def _update_kws_legend_under_plot(kws_legend=None, legend_kws=None):
    """"""
    # Set title_fontsize to fontsize if given
    if legend_kws is not None:
        if "title_fontproperties" not in legend_kws:
            if "fontsize" in legend_kws and "title_fontsize" not in legend_kws:
                legend_kws["title_fontproperties"] = dict(weight="bold", size=legend_kws["fontsize"])
        kws_legend.update(**legend_kws)
    return kws_legend


# SHAP plot
def draw_shap_legend(x=None, y=10, offset_text=1, fontsize=13):
    """Draw higher lower element for indicating colors"""
    arrow_dif = y * 0.02
    plt.text(x - offset_text, y, 'higher',
             fontweight='bold',
             fontsize=fontsize, color=ut.COLOR_SHAP_HIGHER,
             horizontalalignment='right')

    plt.text(x + offset_text*1.1, y, 'lower',
             fontweight='bold',
             fontsize=fontsize, color=ut.COLOR_SHAP_LOWER,
             horizontalalignment='left')

    plt.text(x, y-arrow_dif, r'$\leftarrow$',
             fontweight='bold',
             fontsize=fontsize+1, color=ut.COLOR_SHAP_LOWER,
             horizontalalignment='center')

    plt.text(x, y+arrow_dif, r'$\rightarrow$',
             fontweight='bold',
             fontsize=fontsize+1, color=ut.COLOR_SHAP_HIGHER,
             horizontalalignment='center')


# II Main Functions
class CPPPlots:
    """Class for plotting functions for CPP analysis"""
    def __init__(self, tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
        ut.check_non_negative_number(name="tmd_len", val=tmd_len, min_val=1)
        ut.check_non_negative_number(name="jmd_n_len", val=jmd_n_len, min_val=0)
        ut.check_non_negative_number(name="jmd_c_len", val=jmd_c_len, min_val=0)
        ut.check_non_negative_number(name="start", val=start)
        self.jmd_n_len = jmd_n_len
        self.tmd_len = tmd_len
        self.jmd_c_len = jmd_c_len
        self.seq_len = jmd_n_len + tmd_len + jmd_c_len
        self.start = start
        self.stop = start + self.seq_len - 1

    # Constants
    XLIM_ADD = 3
    YLIM_ADD = 1
    HIGHER_COLOR = '#FF0D57'    # (255, 13, 87)
    LOWER_COLOR = '#1E88E5'     # (30, 136, 229)

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

    def _get_xticks_with_labels(self, step=5):
        """"""
        assert step in [5, 10]  # Internal check if step just five or 10
        second_pos = int((self.start+step)/step)*step
        xticks_middle = list(range(second_pos, self.stop, step))
        xticks_labels = [self.start] + xticks_middle + [self.stop]
        xticks = [x-self.start for x in xticks_labels]
        return xticks, xticks_labels

    # Get new axis object
    @staticmethod
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
            ax.tick_params(axis="y",  colors="white")
        return ax_new

    # Autosize labels
    @staticmethod
    def optimize_label_size(ax=None, df_pos=None, label_term=True):
        """Auto scaling of size of sequence characters"""
        max_len_label = max([len(x) for x in df_pos.index])
        n_pos = len(list(df_pos))
        if label_term:
            l = max([(85-n_pos-max_len_label)/6, 0])
        else:
            l = 8
        width, height = plt.gcf().get_size_inches()
        xmax = ax.get_xlim()[1]
        # Formula based on manual optimization (on tmd=13-23, jmd=10)
        size = l - (6 + xmax/10) + width*2.4
        return size

    # Set figsize
    @staticmethod
    def set_figsize(figsize=None):
        """Set figsize of figure only if not part of subplots"""
        if len(plt.gcf().get_axes()) == 0:
            plt.figure(figsize=figsize)

    # Set title
    @staticmethod
    def set_title(title=None, title_kws=None):
        """"""
        if title_kws is None:
            title_kws = {}
        plt.title(title, **title_kws)

    # Add tmd jmd bars
    def add_xticks(self, ax=None, x_shift=0.0, xticks_position="bottom",
                   xtick_size=11.0, xtick_width=2.0, xtick_length=4.0):
        """"""
        # Set default x ticks (if tmd_jmd not shown)
        xticks, xticks_labels = self._get_xticks_with_labels(step=5)
        ax.set_xticks([x + x_shift for x in xticks])
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
        ax.tick_params(axis="x", color="black", length=xtick_length, width=xtick_width)
        ax.xaxis.set_ticks_position(xticks_position)

    def add_tmd_jmd_bar(self, ax=None, x_shift=0, jmd_color="blue", tmd_color="mediumspringgreen", add_white_bar=True):
        """"""
        # Start positions
        jmd_n_start, tmd_start, jmd_c_start = self._get_starts(x_shift=x_shift)
        args = dict(ax=ax, add_white_bar=add_white_bar)
        add_part_bar(start=jmd_n_start, len_part=self.jmd_n_len, color=jmd_color, **args)
        add_part_bar(start=tmd_start, len_part=self.tmd_len, color=tmd_color, **args)
        add_part_bar(start=jmd_c_start, len_part=self.jmd_c_len, color=jmd_color, **args)

    def add_tmd_jmd_text(self, ax=None, x_shift=0, tmd_jmd_fontsize=11):
        """"""
        # Start positions
        jmd_n_start, tmd_start, jmd_c_start = self._get_starts(x_shift=x_shift)
        if tmd_jmd_fontsize > 0:
            add_part_text(ax=ax, start=tmd_start, len_part=self.tmd_len, text="TMD", fontsize=tmd_jmd_fontsize)
            add_part_text(ax=ax, start=jmd_n_start, text="JMD-N", len_part=self.jmd_n_len, fontsize=tmd_jmd_fontsize)
            add_part_text(ax=ax, start=jmd_c_start, text="JMD-C", len_part=self.jmd_c_len, fontsize=tmd_jmd_fontsize)

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

    def add_tmd_jmd_seq(self, ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, xticks_pos=False,
                        heatmap=True, tmd_color="mediumspringgreen", jmd_color="blue",
                        tmd_seq_color="black", jmd_seq_color="white",
                        x_shift=0, xtick_size=11, seq_size=None, tmd_jmd_fontsize=None):
        """"""
        seq_size = add_part_seq(ax=ax, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                x_shift=x_shift, seq_size=seq_size,
                                tmd_color=tmd_color, jmd_color=jmd_color,
                                tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        tmd_jmd_fontsize = seq_size * 1.1 if tmd_jmd_fontsize is None else tmd_jmd_fontsize
        # Set second axis (with ticks and part annotations)
        jmd_n_end, tmd_end, jmd_c_end = self._get_ends(x_shift=-1)
        jmd_n_middle, tmd_middle, jmd_c_middle = self._get_middles(x_shift=-0.5)
        if not xticks_pos:
            xticks = [jmd_n_middle, tmd_middle, jmd_c_middle]
            xtick_labels = ["JMD-N", "TMD", "JMD-C"]
        else:
            xticks = [0, jmd_n_middle, jmd_n_end, tmd_middle, tmd_end, jmd_c_middle, jmd_c_end]
            xtick_labels = [self.start,  "JMD-N", jmd_n_end + self.start, "TMD",
                            tmd_end + self.start, "JMD-C", jmd_c_end + self.start]
        ax2 = self.get_new_axis(ax=ax, heatmap=heatmap)
        ax2.set_xlim(ax.get_xlim())
        add_part_seq_second_ticks(ax2=ax2, xticks=xticks, xtick_labels=xtick_labels, x_shift=x_shift,
                                  seq_size=seq_size, tmd_jmd_fontsize=tmd_jmd_fontsize, xtick_size=xtick_size)
        return ax

    @staticmethod
    def add_legend_cat(ax=None, df_pos=None, df_cat=None, y=None, dict_color=None, legend_kws=None):
        """"""
        # Get list of categories and their counts
        dict_cat = {}
        columns = [ut.COL_CAT, y]
        for i, cat in enumerate(columns):
            for j, sub_cat in enumerate(columns):
                if i < j or i == j == 0:
                    dict_cat = dict(zip(df_cat[sub_cat], df_cat[cat]))
                    dict_cat.update(dict_cat)
        list_cat = [dict_cat[i] for i in df_pos.index]
        dict_counts = {cat: list_cat.count(cat) for cat in sorted(set(list_cat))}
        # Add colored bars indicating super categories
        y = 0
        for cat in dict_counts:
            n = dict_counts[cat]
            width = 0.55
            bar = mpl.patches.Rectangle((-width*1.5, y), width=width, height=n, linewidth=0,
                                        color=dict_color[cat], zorder=4, clip_on=False)
            y += n
            ax.add_patch(bar)
        # Move ytick labels to left
        pad = _get_ytick_pad_heatmap(ax=ax)
        ax.tick_params(axis="y", pad=pad)
        # Set legend
        # TODO check adjust and legend positioning
        kws_legend = _get_kws_legend_under_plot(list_cat=list_cat)
        kws_legend = _update_kws_legend_under_plot(kws_legend=kws_legend, legend_kws=legend_kws)
        handles, labels = _get_legend_handles_labels(dict_color=dict_color, list_cat=sorted(set(list_cat)))
        legend = ax.legend(handles=handles, labels=labels, **kws_legend)
        legend._legend_box.align = "left"
        return ax

    # Data processing methods
    def get_df_pos(self, df=None, y="category", value_type="count", val_col="mean_dif",
                   normalize=False, normalize_for_pos=False):
        """Get df_pos with values (e.g., counts or mean auc)
        for each combination scale categories (y) and positions (x)"""
        ut.check_y_categorical(df=df, y=y)
        check_positions(df=df)
        check_val_col(df=df, val_col=val_col, value_type=value_type)
        kwargs = dict(y=y, val_col=val_col, value_type=value_type,
                      start=self.start, stop=self.stop, normalize_for_pos=normalize_for_pos)
        if value_type != "count":
            df_pos = _get_df_pos(df=df, **kwargs)
        else:
            if val_col is None:
                df_pos = _get_df_pos(df=df, **kwargs)
            else:
                df_pos = _get_df_pos_sign(df=df, count=True, **kwargs)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        return df_pos.round(3)

    # Main plotting methods
    def profile(self, df_pos=None, ax=None, dict_color=None, edge_color="none", bar_width=0.8,
                xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, ylim=None, color="tab:gray",
                add_legend=True, legend_kws=None, shap_plot=False, shap_legend=False, **kwargs):
        """Show count of feature categories/sub_categories per position for positive and
        negative features, i.e., feature with positive resp. negative mean_dif. The profile
        is a bar chart with positive and negative counts"""
        plot_args = dict(kind="bar", stacked=True, rot=0, width=bar_width, edgecolor=edge_color, legend=False,
                         zorder=10)
        plot_args.update(**kwargs)  # Update and add new arguments
        # Plot
        if shap_plot:
            df_bar = df_pos.T
            df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="y")
            df_pos = df_bar[df_bar > 0]
            df_pos = df_pos.sum(axis=1)
            df_neg = df_bar[df_bar < 0]
            df_neg = df_neg.sum(axis=1)
            ax = df_pos.plot(ax=ax, color=ut.COLOR_SHAP_HIGHER, **plot_args)
            ax = df_neg.plot(ax=ax, color=ut.COLOR_SHAP_LOWER, **plot_args)
            ylim = ut.check_ylim(df=df, val_col="y", ylim=ylim, retrieve_plot=True)
            plt.ylim(ylim)
            if add_legend:
                if legend_kws is not None and "fontsize" in legend_kws:
                    fs = legend_kws["fontsize"]
                else:
                    fs = 13
                if shap_legend:
                    draw_shap_legend(x=plt.xlim()[0]+4, y=ylim[0]+1.5,  offset_text=1, fontsize=fs)
        else:
            handles, labels = _get_legend_handles_labels(dict_color=dict_color, list_cat=list(df_pos.index))
            df_bar = df_pos.T[labels]
            df = pd.concat([df_bar[df_bar < 0].sum(axis=1), df_bar[df_bar > 0].sum(axis=1)]).to_frame(name="y")
            color = dict_color if add_legend else color
            if not add_legend:
                df_bar = df_bar.sum(axis=1)
            ax = df_bar.plot(ax=ax, color=color, **plot_args)
            ylim = ut.check_ylim(df=df, val_col="y", ylim=ylim)
            plt.ylim(ylim)
            # Set legend
            if add_legend:
                _legend_kws = dict(ncol=2, prop={"size": 10}, loc=2, frameon=True,
                                   columnspacing=1, facecolor="white", framealpha=1)
                if legend_kws is not None:
                    _legend_kws.update(legend_kws)
                    if "fontsize" in _legend_kws:
                        fs = _legend_kws["fontsize"]
                        _legend_kws.update(dict(prop={"size": fs}))
                plt.legend(handles=handles, labels=labels, **_legend_kws)
        # Set default x ticks (if tmd_jmd not shown)
        xticks, xticks_labels = self._get_xticks_with_labels(step=5)
        ax.tick_params(axis="x", color="black", width=xtick_width, length=xtick_length)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels, rotation=0, size=xtick_size)
        # Add extra flanking space for x axis
        x_lim = (min(xticks) - self.XLIM_ADD, max(xticks) + self.XLIM_ADD)
        ax.set_xlim(x_lim)
        # Add extra flanking space for y axis
        if ylim is None:
            ymin, ymax = ax.get_ylim()
            y_space = min(self.YLIM_ADD, (ymax - ymin)*0.25)
            y_lim = (ymin - y_space, ymax + y_space)
            ax.set_ylim(y_lim)
        # Plot baseline
        ax.plot([-0.5, self.seq_len - 0.5], [0, 0], color="black", linestyle="-")
        return ax

    def heatmap(self, df_pos=None, ax=None, cmap=None, cmap_n_colors=None, cbar_kws=None, grid_on=False, x_shift=0.0,
                xtick_size=11.0, xtick_width=2.0, xtick_length=None, ytick_size=None, facecolor_dark=True,
                cbar_pct=True, linecolor="black", **kwargs):
        """Show summary static values of feature categories/sub_categories per position as heat map"""
        facecolor = "black" if facecolor_dark else "white"
        # Default arguments for heatmap
        cmap = get_cmap_heatmap(df_pos=df_pos, cmap=cmap, n_colors=cmap_n_colors, higher_color=ut.COLOR_SHAP_HIGHER,
                                lower_color=ut.COLOR_SHAP_LOWER, facecolor_dark=facecolor_dark)
        center = get_center_heatmap(df_pos=df_pos)
        dict_cbar, cbar_kws_ = get_cbar_args_heatmap(cbar_kws=cbar_kws, df_pos=df_pos)
        linewidths = 0.01 if grid_on else 0
        kws_plot = dict(ax=ax, cmap=cmap, cbar_kws=cbar_kws_, center=center, linewidths=linewidths, linecolor=linecolor)
        kws_plot.update(**kwargs)  # Update and add new arguments
        # Plot with 0 set to NaN
        ax = sns.heatmap(df_pos.replace(0, np.NaN), yticklabels=True, xticklabels=True, **kws_plot)
        # Set default x ticks (if tmd_jmd not shown)
        self.add_xticks(ax=ax, xticks_position="bottom", x_shift=x_shift, xtick_size=xtick_size,
                        xtick_width=xtick_width, xtick_length=xtick_length)
        # Adjust y ticks
        ax.tick_params(axis='y', which='both', length=0, labelsize=ytick_size)
        # Set colorbar labelsize and ticksize
        kwargs = infer_vmin_vmax_from_data(df_pos=df_pos, kwargs=kwargs)
        set_cbar_heatmap(ax=ax, dict_cbar=dict_cbar, cbar_kws_=cbar_kws_, vmin=kwargs["vmin"], vmax=kwargs["vmax"],
                         cbar_pct=cbar_pct)
        # Set frame
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.set_facecolor(facecolor)
        return ax
