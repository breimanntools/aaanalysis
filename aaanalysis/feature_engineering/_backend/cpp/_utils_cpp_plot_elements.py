"""
This is a script for the backend PlotElements utility class for the CPPPlot class.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
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

# Heatmap settings functions
# TODO refactor cbar (easier handling for combination
def _get_ytick_pad_heatmap(ax=None):
    """Calculate padding for y-ticks in a heatmap."""
    xmax = ax.get_xlim()[1]
    width, height = plt.gcf().get_size_inches()
    pad = width+8-xmax/10
    return pad


def _get_kws_legend_under_plot(list_cat=None, items_per_col=3, x_adjust=-0.3, y_adjust=-0.03):
    """Generate keyword arguments for placing a legend under a plot"""
    width, height = plt.gcf().get_size_inches()
    bbox_y = width / height * y_adjust
    bbox_x = -0.2 + x_adjust
    # TODO
    n_cols = len(set(list_cat))
    if n_cols > 4:
        n_cols = int(np.ceil(n_cols/items_per_col))
    kws_legend = dict(n_cols=n_cols, loc=2, frameon=False, bbox_to_anchor=(bbox_x, bbox_y),
                      columnspacing=0.3, labelspacing=0.05, handletextpad=0.15,
                      facecolor="white",  shadow=False,
                      title="Scale category")
    return kws_legend


def _update_kws_legend_under_plot(kws_legend=None, legend_kws=None):
    """Update the keyword arguments for a plot legend with additional settings."""
    if kws_legend is None:
        kws_legend = {}
    # Set title_fontsize to fontsize if given
    if legend_kws is not None:
        if "title_fontproperties" not in legend_kws:
            if "fontsize" in legend_kws and "title_fontsize" not in legend_kws:
                legend_kws["title_fontproperties"] = dict(weight="bold", size=legend_kws["fontsize"])
        kws_legend.update(**legend_kws)
    return kws_legend


# II Main Functions
class PlotElements:
    """Utility class for plot element configurations and enhancements."""

    # Get color dict
    @staticmethod
    def get_color_dif(mean_dif=0):
        """Return color based on the mean difference value."""
        return ut.COLOR_FEAT_NEG if mean_dif < 0 else ut.COLOR_FEAT_POS


    # Get legend items
    @staticmethod
    def get_legend_handles_labels(dict_color=None, list_cat=None):
        """Create legend handles from the provided color dictionary."""
        dict_leg = {cat: dict_color[cat] for cat in dict_color if cat in list_cat}
        f = lambda l, c: mpl.patches.Patch(color=l, label=c, lw=0)
        handles = [f(l, c) for c, l in dict_leg.items()]
        labels = list(dict_leg.keys())
        return handles, labels

    # Set figsize
    @staticmethod
    def set_figsize(ax=None, figsize=None, force_set=False):
        """Set the figure size. Optionally create a figure and an axes object if none exists."""
        if ax:
            fig = ax.figure
        elif not plt.get_fignums():
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        # Set figure size if force_set is True or if there are no axes in the figure
        if force_set or len(fig.get_axes()) == 0:
            fig.set_size_inches(figsize, forward=True)
        return fig, ax


    @staticmethod
    def set_title_(title=None, title_kws=None):
        """Set the title of the plot."""
        if title_kws is None:
            title_kws = {}
        plt.title(title, **title_kws)

    # Set legend
    # TODO simplify (use set legend utils) for profile or move to heatmap only
    def add_legend_cat(self, ax=None, df_pos=None, df_cat=None, y=None, dict_color=None, legend_kws=None):
        """Add a category legend to the plot."""
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
        """Set custom y-tick labels with the specified font size."""
        tick_positions = ax.get_yticks()
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        tick_labels = [round(float(i), 1) for i in tick_positions]
        ax.set_yticklabels(tick_labels, size=fs)

    @staticmethod
    def set_x_ticks(ax=None, fs=None):
        """Set custom x-tick labels with the specified font size."""
        tick_positions = ax.get_xticks()
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(tick_positions))
        tick_labels = [round(float(i), 1) for i in tick_positions]
        ax.set_xticklabels(tick_labels, size=fs)

