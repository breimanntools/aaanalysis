"""
This is a script for the backend of the shared AAPredPlot legend placement.

A single house-style legend for the across-samples prediction figures (``predict_group`` /
``eval``): a frameless, titled legend centered *below* the plot, so every method advertises its
color key the same way instead of each dropping the legend in a different corner.
"""
from typing import Optional, List

import aaanalysis.utils as ut


# I Helper Functions
def _resolve_ncol(n_entries, ncol):
    """Default to a single row (all entries side by side), capped so labels stay legible."""
    if ncol is not None:
        return ncol
    return int(min(max(n_entries, 1), 4))


# II Main Functions
def _offset_below_axes(ax):
    """Axes-fraction y to place a legend just below the axes' bottom decorations (x-tick labels +
    x-label), so multi-line ticks / a wrapped x-label never collide with it. Measured now (before a
    later ``tight_layout``); the ratio of decoration-height to axes-height stays roughly stable."""
    fig = ax.get_figure()
    fig.canvas.draw()
    ax_bb = ax.get_window_extent()
    tb = ax.get_tightbbox(fig.canvas.get_renderer())
    below = max(0.0, (ax_bb.y0 - tb.y0) / ax_bb.height)  # decoration height / axes height
    return -(below + 0.09)


def place_legend_below_(ax=None, fig=None, handles=None, labels=None, title=None, ncol=None,
                        y=None, fontsize="x-small", title_fontsize="x-small"):
    """Draw a frameless, LEFT-aligned legend centered below the axes (or figure), house style.

    Matches the package legend look (frameless, left-aligned, compact): the title weight follows
    ``options['legend_title_bold']`` (default non-bold), and a **single-entry** key gets no title
    (the lone label already names it, as in the paper figures). When ``y`` is not given (single-axes
    case) it is placed just below the axes' x-tick labels and x-label so it never overlaps them.
    Pulls the labeled artists from ``ax`` when ``handles`` / ``labels`` are not given; pass ``fig``
    (with explicit handles/labels) to anchor one legend below a *multi-panel* figure. Returns the
    ``Legend`` or ``None``.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if title is not None and len(labels) <= 1:
        title = None  # one entry: the label is self-explanatory, a title would be redundant
    ncol = _resolve_ncol(len(labels), ncol)
    if y is None:
        y = _offset_below_axes(ax) if fig is None else -0.05
    title_kws = {}
    if title:
        weight = "bold" if ut.check_legend_title_bold() else "normal"
        title_kws["title_fontproperties"] = {"weight": weight, "size": title_fontsize}
    target = fig if fig is not None else ax
    leg = target.legend(handles, labels, title=title, loc="upper center",
                        bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                        fontsize=fontsize, **title_kws)
    leg._legend_box.align = "left"  # left-align the title over the entries (house style)
    return leg
