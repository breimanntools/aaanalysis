"""
This is a script for ...
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions


# Figure Modifications


# Heatmap settings functions
# TODO refactor cbar (easier handling for combination
def _get_ytick_pad_heatmap(ax=None):
    """"""
    # TODO check add bars aa.plot
    xmax = ax.get_xlim()[1]
    width, height = plt.gcf().get_size_inches()
    pad = width+8-xmax/10
    return pad


def _get_kws_legend_under_plot(list_cat=None, items_per_col=3, x_adjust=-0.3, y_adjust=-0.03):
    """"""
    width, height = plt.gcf().get_size_inches()
    ncol = len(set(list_cat))
    bbox_y = width / height * y_adjust
    bbox_x = -0.2 + x_adjust
    print(bbox_y, bbox_x)
    # TODO
    if ncol > 4:
        ncol = int(np.ceil(ncol/items_per_col))
    kws_legend = dict(ncol=ncol, loc=2, frameon=False, bbox_to_anchor=(bbox_x, bbox_y),
                      columnspacing=0.3, labelspacing=0.05, handletextpad=0.15,
                      facecolor="white",  shadow=False,
                      title="Scale category")
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


# II Main Functions
class PlotElements:
    """https://matplotlib.org/stable/gallery/showcase/anatomy.html"""

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
            ax.tick_params(axis="y", colors="white")
        return ax_new

    # Get legend items (TODO check if not in plotting utils)
    @staticmethod
    def get_legend_handles_labels(dict_color=None, list_cat=None):
        """Get legend handles from dict_color"""
        dict_leg = {cat: dict_color[cat] for cat in dict_color if cat in list_cat}
        f = lambda l, c: mpl.patches.Patch(color=l, label=c, lw=0)
        handles = [f(l, c) for c, l in dict_leg.items()]
        labels = list(dict_leg.keys())
        return handles, labels

    # Autosize labels
    @staticmethod
    def optimize_label_size(ax=None, df_pos=None, label_term=True):
        """Autoscaling of size of sequence characters"""
        max_len_label = max([len(x) for x in df_pos.index])
        n_pos = len(list(df_pos))
        if label_term:
            l = max([(85 - n_pos - max_len_label) / 6, 0])
        else:
            l = 8
        width, height = plt.gcf().get_size_inches()
        xmax = ax.get_xlim()[1]
        # Formula based on manual optimization (on tmd=13-23, jmd=10)
        size = l - (6 + xmax / 10) + width * 2.4
        size = max(min(15, size), 10)   # Should range between 10 and 15 (can be adjusted)
        return size

    # Set figsize
    @staticmethod
    def set_figsize(figsize=None):
        """Set figsize of figure only if not part of subplots"""
        if len(plt.gcf().get_axes()) == 0:
            plt.figure(figsize=figsize)

    @staticmethod
    def set_title_(title=None, title_kws=None):
        """"""
        if title_kws is None:
            title_kws = {}
        plt.title(title, **title_kws)

    # Set legend
    # TODO for profile or move to heatmap only
    def add_legend_cat(self, ax=None, df_pos=None, df_cat=None, y=None, dict_color=None, legend_kws=None):
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
            bar = mpl.patches.Rectangle((-width * 1.5, y), width=width, height=n, linewidth=0, color=dict_color[cat],
                                        zorder=4, clip_on=False)
            y += n
            ax.add_patch(bar)
        # Move ytick labels to left
        pad = _get_ytick_pad_heatmap(ax=ax)
        ax.tick_params(axis="y", pad=pad)
        # Set legend
        # TODO check adjust and legend positioning
        # TODO simplify
        kws_legend = _get_kws_legend_under_plot(list_cat=list_cat)
        kws_legend = _update_kws_legend_under_plot(kws_legend=kws_legend, legend_kws=legend_kws)
        handles, labels = self.get_legend_handles_labels(dict_color=dict_color, list_cat=sorted(set(list_cat)))
        legend = ax.legend(handles=handles, labels=labels, **kws_legend)
        legend._legend_box.align = "left"
        return ax

    @staticmethod
    def set_y_ticks(ax=None, fs=None):
        """"""
        tick_positions = ax.get_yticks()
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        tick_labels = [round(float(i), 1) for i in tick_positions]
        ax.set_yticklabels(tick_labels, size=fs)

    @staticmethod
    def set_x_ticks(ax=None, fs=None):
        """"""
        tick_positions = ax.get_xticks()
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        tick_labels = [round(float(i), 1) for i in tick_positions]
        ax.set_xticklabels(tick_labels, size=fs)

