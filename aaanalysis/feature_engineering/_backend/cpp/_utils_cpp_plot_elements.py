"""
This is a script for the backend PlotElements utility class for the CPPPlot class.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import aaanalysis.utils as ut

# I Helper Functions
DICT_LEGEND_CAT = dict(loc=9,
                       weight_title="normal",
                       labelspacing=0.05,
                       handletextpad=0.2)


def _get_colors_for_col_cat(labels=None, dict_color=None, df_feat=None, col_cat=None):
    """Extend color dict for given category column"""
    dict_cat = dict(zip(df_feat[col_cat], df_feat[ut.COL_CAT]))
    colors = [dict_color[dict_cat[c]] for c in labels]
    return colors


def _force_render(fig):
    """Force a render so text ``get_window_extent`` returns true extents.

    Without a render the extents are degenerate (~1px), which would report false
    "no overlap"; a bare draw is enough on the Agg backend used for figures.
    """
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _row_labels_overlap(label_artists, renderer, overlap_frac=0.15):
    """Whether any two vertically-adjacent row labels overlap beyond a fraction.

    Row labels share the same x (right-aligned), so the vertical overlap fraction
    of the smaller box is an accurate stand-in for the area fraction.
    """
    boxes = sorted((t.get_window_extent(renderer) for t in label_artists), key=lambda b: b.y0)
    for lower, upper in zip(boxes[:-1], boxes[1:]):
        overlap = lower.y1 - upper.y0
        if overlap > 0 and overlap / min(lower.height, upper.height) >= overlap_frac:
            return True
    return False


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
    def add_subcat_bars(ax=None, df_pos=None, df_feat=None, col_cat=None, dict_color=None,
                        bar_width=0.3, bar_spacing=0.15, optimize_labels=False):
        """Add left colored sidebar to indicate category grouping.

        ``optimize_labels`` runs the overlap-driven font-shrink fallback. It stays
        off by default (byte-identical to a no-gate render); the frontend enables it
        only when ``auto_font`` is on AND the caller forced a fixed figsize, so the
        figure cannot be grown to fit. In the normal ``auto_font`` path the figure is
        instead rescaled to a constant cell size, which keeps labels full-size.
        """
        labels = list(df_pos.index)
        colors = _get_colors_for_col_cat(labels=labels, df_feat=df_feat, col_cat=col_cat, dict_color=dict_color)
        before = list(ax.texts)
        ut.plot_add_bars(ax=ax, labels=labels, colors=colors,
                         bar_width=bar_width, bar_spacing=bar_spacing, label_spacing_factor=2)
        if optimize_labels:
            row_labels = [t for t in ax.texts if t not in before]
            PlotElements.optimize_subcat_label_fontsize(fig=ax.figure, label_artists=row_labels)

    @staticmethod
    def optimize_subcat_label_fontsize(fig=None, label_artists=None, floor=5.0, fs_step=0.5):
        """Shrink subcategory row-label font just enough to clear vertical overlap.

        The row labels are hand-placed text artists at the current rcParams font
        size; on dense feature maps / heatmaps (many subcategories in a fixed
        figure height) they collide into an unreadable column. This reduces their
        font stepwise until consecutive labels no longer overlap, never dropping
        below the legibility ``floor`` (~5 pt). Sparse maps already fit, so their
        labels are left untouched and their output stays unchanged.
        """
        if not label_artists or len(label_artists) < 2:
            return
        fs = label_artists[0].get_fontsize()
        # Cheap pre-check to avoid an expensive forced render on the common sparse
        # map. Estimate how many rows fit before labels could touch, using a
        # deliberately conservative axes-height fraction (0.6, below the real
        # ~0.75) so we only *skip* when overlap is impossible and always render
        # when it is even plausible. A wrong skip would reintroduce overlap; a
        # wrong render just costs one draw.
        fig_height_in = fig.get_size_inches()[1]
        rows_capacity = (fig_height_in * 0.6 * 72.0) / max(fs * 1.1, 1e-6)
        if len(label_artists) <= rows_capacity:
            return
        renderer = _force_render(fig)
        if not _row_labels_overlap(label_artists, renderer):
            return
        while fs > floor and _row_labels_overlap(label_artists, renderer):
            fs = max(floor, fs - fs_step)
            for t in label_artists:
                t.set_fontsize(fs)
            renderer = _force_render(fig)


    @staticmethod
    def adjust_cat_legend_kws(legend_kws=None, n_cat=None,
                              legend_xy=None, fontsize_labels=None):
        """Optimize legend position and appearance based on the number of categories and provided keywords."""
        n_cols = 2 if legend_kws is None else legend_kws.get("n_cols", 2)
        n_rows = np.floor(n_cat / n_cols)

        # Create legend position [center (x), top (y)]
        legend_xy_default = (-0.1, -0.01)
        _legend_xy = ut.adjust_tuple_elements(tuple_in=legend_xy,
                                              tuple_default=legend_xy_default)
        x, y = _legend_xy
        str_space = "\n" * int((6-n_rows))
        title = f"{str_space}{ut.LABEL_SCALE_CAT}"
        # Prepare legend keywords
        _legend_kws = dict(fontsize=fontsize_labels,
                           fontsize_title=fontsize_labels,
                           n_cols=n_cols,
                           title=title,
                           x=x, y=y,
                           **DICT_LEGEND_CAT)
        if legend_kws is not None:
            _legend_kws.update(legend_kws)
        return _legend_kws

    # Colorbar elements
    @staticmethod
    def adjust_cbar_kws(fig=None, cbar_kws=None, cbar_xywh=None,
                        fontsize_labels=None, label=None):
        """Set colorbar position, appearance, and label with default or provided keywords."""
        width, height = plt.gcf().get_size_inches()
        bar_height = 0.15/height
        bar_bottom = 0.06/height

        # Create cbar positions: [left (x), bottom (y), width, height]
        cbar_xywh_default = (0.5, bar_bottom, 0.2, bar_height)
        _cbar_xywh = ut.adjust_tuple_elements(tuple_in=cbar_xywh,
                                              tuple_default=cbar_xywh_default)

        # Create colorbar axes
        cbar_ax = fig.add_axes(_cbar_xywh)

        # Prepare colorbar keywords
        _cbar_kws = dict(ticksize=fontsize_labels, label=label)
        if cbar_kws is not None:
            _cbar_kws.update(cbar_kws)
        return _cbar_kws, cbar_ax
