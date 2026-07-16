"""
This is a script for internal plotting part utility functions used in the backend.
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl


# Helper functions
def _get_bar_height(ax=None, divider=50):
    """Calculate bar height for sequence part visualization."""
    ylim = ax.get_ylim()
    width, height = plt.gcf().get_size_inches()
    bar_height = abs(ylim[0] - ylim[1]) / divider / height*6
    return bar_height


def _get_y(ax=None, bar_height=None, height_factor=1.0, reversed_weight=0):
    """Determine y-coordinate for bar placement."""
    ylim = ax.get_ylim()
    reversed_y = reversed_weight if ylim[0] > ylim[1] else 1
    y = ylim[0] - (bar_height * height_factor) * reversed_y
    return y


def _add_part_bar(ax=None, start=1.0, len_part=40.0, color="blue", bar_height_factor=1, bar_height=None):
    """Add colored bar for TMD and JMD sequence parts.

    ``bar_height`` (in y-data units, i.e. grid rows) pins the bar to a constant thickness independent
    of the figure size — the ``divider``-based default scales with the figure. On a constant-cell grid
    a fixed row count renders at a constant physical height, matching the subcategory cells.
    """
    if bar_height is None:
        bar_height = _get_bar_height(ax=ax) * bar_height_factor
    y = _get_y(ax=ax, bar_height=bar_height)
    bar = mpl.patches.Rectangle((start, y), width=len_part, height=bar_height, linewidth=0,
                                color=color, zorder=3, clip_on=False)
    ax.add_patch(bar)


def _add_part_text(ax=None, text=None, start=1.0, len_part=10.0, fontsize=None,
                   fontweight="normal", height_factor=1.3, bar_height=None):
    """Place text marking for TMD and JMD sequence parts."""
    if bar_height is None:
        bar_height = _get_bar_height(ax=ax)
    y = _get_y(ax=ax, bar_height=bar_height, height_factor=height_factor, reversed_weight=-1)
    x = start + len_part / 2    # Middle of part
    return ax.text(x, y, text,
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontsize=fontsize,
                   fontweight=fontweight,
                   color='black')


# Helper class
class PlotPart:
    """Helper class for adjusting tmd and jmd-c/n sequence length"""

    def __init__(self, tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
        """Initialize the plot positions with given lengths and start position."""
        self.jmd_n_len = jmd_n_len
        self.tmd_len = tmd_len
        self.jmd_c_len = jmd_c_len
        self.seq_len = jmd_n_len + tmd_len + jmd_c_len
        self.start = start
        self.stop = start + self.seq_len - 1

    # Helper methods
    def get_starts(self, x_shift=0):
        """Calculate the starting positions for JMD and TMD."""
        jmd_n_start = 0 + x_shift
        tmd_start = self.jmd_n_len + x_shift
        jmd_c_start = self.jmd_n_len + self.tmd_len + x_shift
        return jmd_n_start, tmd_start, jmd_c_start

    def get_ends(self, x_shift=-1):
        """Calculate the ending positions for JMD and TMD."""
        jmd_n_end = self.jmd_n_len + x_shift
        tmd_end = self.jmd_n_len + self.tmd_len + x_shift
        jmd_c_end = self.jmd_n_len + self.tmd_len + self.jmd_c_len + x_shift
        return jmd_n_end, tmd_end, jmd_c_end


# II Main functions
def add_tmd_jmd_bar(ax=None, x_shift=0, jmd_color="blue", tmd_color="mediumspringgreen",
                    bar_height_factor=1, bar_height=None,
                    tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Add colored bars to indicate TMD and JMD regions."""
    pp = PlotPart(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    jmd_n_start, tmd_start, jmd_c_start = pp.get_starts(x_shift=x_shift)
    args = dict(bar_height_factor=bar_height_factor, bar_height=bar_height)
    _add_part_bar(ax=ax, start=jmd_n_start, len_part=jmd_n_len, color=jmd_color, **args)
    _add_part_bar(ax=ax, start=tmd_start, len_part=tmd_len, color=tmd_color, **args)
    _add_part_bar(ax=ax, start=jmd_c_start, len_part=jmd_c_len, color=jmd_color, **args)


def add_tmd_jmd_text(ax=None, x_shift=0, fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                     name_tmd="TMD", name_jmd_n="JMD-N", name_jmd_c="JMD-C",
                     tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1,
                     height_factor=1.3, bar_height=None):
    """Add text labels for TMD and JMD regions. Returns the created text artists."""
    pp = PlotPart(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    jmd_n_start, tmd_start, jmd_c_start = pp.get_starts(x_shift=x_shift)
    exists_jmd_n = jmd_n_len > 0
    exists_jmd_c = jmd_c_len > 0
    part_texts = []
    if fontsize_tmd_jmd is None or fontsize_tmd_jmd > 0:
        args = dict(ax=ax, fontsize=fontsize_tmd_jmd, fontweight=weight_tmd_jmd,
                    height_factor=height_factor, bar_height=bar_height)
        part_texts.append(_add_part_text(start=tmd_start, len_part=tmd_len, text=name_tmd, **args))
        if exists_jmd_n:
            part_texts.append(_add_part_text(start=jmd_n_start, text=name_jmd_n, len_part=jmd_n_len, **args))
        if exists_jmd_c:
            part_texts.append(_add_part_text(start=jmd_c_start, text=name_jmd_c, len_part=jmd_c_len, **args))
    return part_texts


def align_part_text_below_xticks(ax=None, part_texts=None, gap_pt=6.0):
    """Drop the TMD/JMD part labels to a row below the x-tick labels if the two would collide.

    Each part label is centred on its part span in data coordinates, but its width is fixed in
    points, so the tick labels bounding that span close in on the label as the figure narrows until
    the two overlap. On collision every part label moves down by the same amount, keeping their
    shared baseline, until they clear the lowest tick label. A figure with room to spare never
    collides and is left untouched.

    ``gap_pt`` is deliberately larger than a hairline: a caller that runs ``tight_layout`` after the
    plot reflows the axes to fit the dropped labels, which shortens the cleared gap by a couple of
    points. The margin keeps the two rows apart through that reflow.
    """
    xtick_labels = [t for t in ax.get_xticklabels() if t.get_text() and t.get_visible()]
    if not part_texts or not xtick_labels:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb_ticks = [t.get_window_extent(renderer) for t in xtick_labels]
    bb_parts = [t.get_window_extent(renderer) for t in part_texts]
    if not any(bb_part.overlaps(bb_tick) for bb_part in bb_parts for bb_tick in bb_ticks):
        return
    gap = gap_pt * fig.dpi / 72
    shift = max(b.y1 for b in bb_parts) - (min(b.y0 for b in bb_ticks) - gap)
    if shift <= 0:
        return
    offset = mpl.transforms.ScaledTranslation(0, -shift / fig.dpi, fig.dpi_scale_trans)
    for part_text in part_texts:
        part_text.set_transform(part_text.get_transform() + offset)


def add_tmd_jmd_xticks(ax=None, x_shift=0, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                       tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Adjust x-ticks for TMD and JMD regions."""
    # Remove the xticks and return early
    if xtick_size == 0:
        ax.set_xticks([])
        ax.set_xticklabels([])
        return
    # Adjust tick tick_length based on figure size
    width, height = plt.gcf().get_size_inches()
    xtick_length += height if xtick_length != 0 else 0
    # Adjust for start counting at position 0
    pp = PlotPart(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    jmd_n_end, tmd_end, jmd_c_end = pp.get_ends(x_shift=-1)
    exists_jmd_n = jmd_n_len > 0
    exists_jmd_c = jmd_c_len > 0
    xticks = [0]
    if exists_jmd_n:
        xticks.append(jmd_n_end)
    xticks.append(tmd_end)
    if exists_jmd_c:
        xticks.append(jmd_c_end)
    ax.set_xticks([x + x_shift for x in xticks])
    ax.set_xticklabels([x + pp.start for x in xticks], size=xtick_size, rotation=0)
    ax.tick_params(axis="x", length=xtick_length, color="black", width=xtick_width, bottom=True)


def highlight_tmd_area(ax=None, x_shift=0, tmd_color="mediumspringgreen", alpha=0.2,
                       tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1, y_max=None):
    """Highlight the TMD area in the plot."""
    pp = PlotPart(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    jmd_n_start, tmd_start, jmd_c_start = pp.get_starts(x_shift=x_shift)
    y_min, _y_max = ax.get_ylim()
    y_max = _y_max if y_max is None else y_max
    height = abs(y_min) + y_max
    rect = mpl.patches.Rectangle((tmd_start, y_min), width=tmd_len, height=height, linewidth=0,
                                 color=tmd_color, zorder=0.1, clip_on=True, alpha=alpha)
    ax.add_patch(rect)
