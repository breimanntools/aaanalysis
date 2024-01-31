"""
This is a script for the backend PlotElements utility class for the CPPPlot class.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import aaanalysis.utils as ut

# I Helper Functions
DICT_LEGEND_CAT = dict(n_cols=2,
                       loc=9,
                       title=ut.LABEL_SCALE_CAT,
                       weight_title="normal",
                       labelspacing=0.05,
                       handletextpad=0.2)


def _get_colors_for_col_cat(labels=None, dict_color=None, df_feat=None, col_cat=None):
    """Extend color dict for given category column"""
    dict_cat = dict(zip(df_feat[col_cat], df_feat[ut.COL_CAT]))
    colors = [dict_color[dict_cat[c]] for c in labels]
    return colors


# II Main Functions
class PlotElements:
    """Utility class for plot element configurations and enhancements."""

    # Set plot elements
    @staticmethod
    def set_figsize(ax=None, figsize=None, force_set=False):
        """Set the figure size. Optionally create a figure and an axes object if none exists."""
        # DEV: figsize is not used as argument in seaborn (but in pandas)
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

    # Scale classification elements
    @staticmethod
    def update_cat_legend_kws(legend_kws=None, _legend_kws=None):
        """Update legend arguments and set defaults"""
        if _legend_kws is None:
            _legend_kws = DICT_LEGEND_CAT.copy()
        if legend_kws is not None:
            _legend_kws.update(legend_kws)
        return _legend_kws

    @staticmethod
    def add_subcat_bars(ax=None, df_pos=None, df_feat=None, col_cat=None, dict_color=None,
                        bar_width=0.3, bar_spacing=0.15):
        """Add left colored sidebar to indicate category grouping"""
        labels = list(df_pos.index)
        colors = _get_colors_for_col_cat(labels=labels, df_feat=df_feat, col_cat=col_cat, dict_color=dict_color)
        ut.plot_add_bars(ax=ax, labels=labels, colors=colors,
                         bar_width=bar_width, bar_spacing=bar_spacing, label_spacing_factor=2)

