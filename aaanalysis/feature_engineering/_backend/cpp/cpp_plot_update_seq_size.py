"""
This is a script for the backend of the CPPPlot.update_seq_size() method.
"""
import matplotlib.pyplot as plt
from ._utils_cpp_plot_positions import PlotPartPositions, repaint_seq_cell_backgrounds_


# Residue letters (fill mode) are capped to a fraction of the grid cell HEIGHT, so they stay
# proportional to the row labels instead of filling the cell width and dwarfing them. The full-width
# per-residue cells provide the gap-free colour band, so a smaller letter adds no gap. The default is
# length-adaptive ("auto"): full cell height for a normal TMD, stepped down for a short one where a
# few large letters would dominate.
def _auto_seq_frac(n_tmd):
    """Auto residue-size fraction (of the cell height) from the TMD length."""
    if n_tmd > 10:
        return 1.0
    if n_tmd >= 6:
        return 0.9
    return 0.8


# I Helper functions
def _get_sorted_x_tick_labels(ax=None):
    # Get all x-axis tick labels
    labels = ax.xaxis.get_ticklabels(which="both")
    # Compute the positions of the labels and sort them
    f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer())
    tick_positions = [f(l).x0 for l in labels]
    sorted_tick_positions, sorted_labels = zip(*sorted(zip(tick_positions, labels), key=lambda t: t[0]))
    return sorted_labels


# II Main Function
def get_tmd_jmd_seq(ax=None, jmd_n_len=10, jmd_c_len=10):
    """Get tmd_seq, jmd_n_seq, and jmd_c_seq from axes"""
    _labels = _get_sorted_x_tick_labels(ax=ax)
    tmd_jmd_seq = "".join([x.get_text() for x in _labels])
    jmd_n_seq = tmd_jmd_seq[0:jmd_n_len]
    tmd_seq = tmd_jmd_seq[jmd_n_len:-jmd_c_len]
    jmd_c_seq = tmd_jmd_seq[-jmd_c_len:]
    return jmd_n_seq, tmd_seq, jmd_c_seq


def update_seq_size_(ax=None, tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     max_x_dist=0.1,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white", fill=False, req_size=None):
    """Update the font size of the sequence characters so they never overlap.

    The residue letters are always sized with a small inter-character gap (``max_x_dist``), so
    adjacent glyphs never touch or overlap at any grid width. In ``fill`` mode the continuous,
    gap-free colored band is supplied by the full-width per-residue cells painted underneath
    (``_add_seq_cell_backgrounds``), so each letter's own box is kept transparent — never a colored
    glyph box, whose thick border would otherwise merge neighbouring letters on a wide figure.
    ``fill=False`` keeps the legacy colored glyph box (byte-identical to the pre-auto_font path).
    """
    colors = [jmd_color] * len(jmd_n_seq) + [tmd_color] * len(tmd_seq) + [jmd_color] * len(jmd_c_seq)
    dict_seq_color = {tmd_color: tmd_seq_color, jmd_color: jmd_seq_color}
    # Get all x-axis tick labels
    labels = _get_sorted_x_tick_labels(ax=ax)
    # Size letters to keep a real gap between neighbours (never the fill=True "just touch" size,
    # which lets bold wide glyphs overlap once the layout is rescaled); the band comes from the cells.
    pp = PlotPartPositions()
    seq_size = pp.get_optimal_fontsize(ax=ax, labels=labels, max_x_dist=max_x_dist, fill=False)
    if fill:
        # Cap the letters (never larger than the non-overlapping size); the cells give the gap-free
        # band, so smaller letters stay proportional to the labels without exposing any gap.
        # req_size: None -> length-adaptive auto (fraction of the cell height); 0<req_size<=1 -> that
        # fraction of the cell height; req_size>1 -> an absolute font size in points.
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((0.0, 1.0))
        row_pt = abs(p1[1] - p0[1]) / ax.figure.dpi * 72.0
        if req_size is None:
            cap = row_pt * _auto_seq_frac(len(tmd_seq))
        elif req_size <= 1:
            cap = row_pt * req_size
        else:
            cap = req_size
        seq_size = min(seq_size, cap)
    lw = plt.gcf().get_size_inches()[0]/5
    for l, c in zip(labels, colors):
        l.set_fontsize(seq_size)
        box_color = "none" if fill else c
        l.set_bbox(dict(facecolor=box_color, edgecolor=box_color, zorder=0.1, alpha=1,
                        clip_on=False,
                        pad=0, linewidth=lw))
        l.set_color(dict_seq_color[c])
    # In fill mode the color band is the full-width per-residue cells (letters are transparent);
    # repaint them around the just-fitted letters so the band frames them at the final grid size.
    if fill:
        repaint_seq_cell_backgrounds_(ax=ax, labels=labels, colors=colors)
    return ax, seq_size


def update_tmd_jmd_labels(fig=None, seq_size=None, fontsize_tmd_jmd=None, weight_tmd_jmd="bold"):
    """Adjust size and position of TMD-JMD labels"""
    fs_labels = fontsize_tmd_jmd if fontsize_tmd_jmd else seq_size
    if fig is not None and len(fig.axes) > 1:
        ax2 = fig.axes[1]
        ax2.spines["bottom"].set_position(("outward", seq_size + 7))
        ax2.tick_params(axis='x', labelsize=fs_labels)
        for label in ax2.get_xticklabels():
            label.set_weight(weight_tmd_jmd)
