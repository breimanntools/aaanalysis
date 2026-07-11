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
def _get_optimal_fontsize(ax=None, labels=None, max_x_dist=0.1, fill=False):
    """Optimize font size of sequence characters.

    With ``fill=True`` the required inter-character gap is dropped to 0, so the
    font grows until adjacent characters just touch (no whitespace between them)
    while still never overlapping. ``fill=False`` keeps the default spacing.
    """
    if fill:
        max_x_dist = 0.0
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


# Colored sequence cell: fraction of the (heatmap-edge) top margin mirrored below the letters.
# 1.0 = symmetric; smaller keeps the band flush to the grid on top but slim underneath.
_SEQ_CELL_BOTTOM_MARGIN_FRAC = 0.4

# Gap between the heatmap grid and the sequence band. Expressed in grid-CELL (row) units so it
# equals the category-sidebar gap (_SUBCAT_BAR_GAP_CELLS) and scales with the cell like it does.
# The x-tick pad moves the residue letters clear of that gap (points; a touch above the cell gap).
_SEQ_HEATMAP_GAP_CELLS = 0.22
_SEQ_HEATMAP_GAP_PT = 8.0

# Constant gap (points) between the sequence band and the JMD-N/TMD/JMD-C part labels.
_JMD_LABEL_GAP_PT = 3.0

# Upper bound (points) on the auto-sized part-label font. The part labels otherwise
# default to the residue-letter size, which balloons on short/wide grids; the cap keeps
# them readable-but-bounded. An explicit ``fontsize_tmd_jmd`` is honored uncapped.
_JMD_LABEL_MAX_PT = 12.0

# Distance (inches) from the heatmap grid bottom to the colorbar top, matching the look of a
# dense feature map. Holding it constant keeps the colorbar clear of the sequence band on a
# sparse grid (small figure), where the figure-bottom-anchored default would ride up into it.
_CBAR_BELOW_GRID_IN = 1.4

# Fixed colorbar thickness (inches). The backend sizes it as a figure fraction, which collapses
# to a thin line once the figure shrinks; pinning it to a constant inch height keeps a proper
# gradient bar at any figure size (the cheat-sheet look).
_CBAR_HEIGHT_IN = 0.16

# Compact gap (inches) between the sequence/part-label block and the bottom furniture row, and
# between the colorbar and the importance legend beside it.
_FURNITURE_GAP_IN = 0.12
_FURNITURE_ITEM_GAP_IN = 0.25


def _add_seq_cell_backgrounds(ax=None, labels=None, colors=None, x_shift=0.0):
    """Paint a seamless, full-width colored cell behind each residue letter (``fill`` mode).

    The per-letter text bbox is glyph-sized, so narrow residues leave hairline gaps and the
    colored band reads as ragged against the uniform heatmap grid. Draw one rectangle per
    residue instead: full-width (1.0 data unit) and centered on the tick so it lines up exactly
    with the heatmap column, its **top flush against the heatmap edge** and a slim margin below
    the letters, so the band is gap-free without piling excess color under the residues.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    # Letter band in data coordinates (uniform, spanning the tallest letter).
    exts = [l.get_window_extent(renderer).transformed(inv) for l in labels]
    lo = min(e.y0 for e in exts)
    hi = max(e.y1 for e in exts)
    # The sequence band sits just outside the heatmap; snap its inner edge to the nearest axis
    # limit (the heatmap border) so there is no gap between the cells and the grid, then mirror
    # that margin past the far side of the letters so top and bottom margins match.
    center = 0.5 * (lo + hi)
    edge = min(ax.get_ylim(), key=lambda v: abs(v - center))
    near, far = (lo, hi) if abs(lo - edge) <= abs(hi - edge) else (hi, lo)
    d = 1 if far >= edge else -1  # from the heatmap edge toward the letters
    # Leave a gap between the heatmap and the band top (matches the sidebar gap), but never uncover
    # the letters: clamp the band top just above the letters' near edge if the pad is too small.
    gap = min(_SEQ_HEATMAP_GAP_CELLS, max(0.0, abs(near - edge) - 0.05))
    inner = edge + gap * d
    margin = abs(near - edge) * _SEQ_CELL_BOTTOM_MARGIN_FRAC
    y_far = far + margin * (1 if far >= near else -1)
    y0, y1 = sorted((inner, y_far))
    # Center each cell on its residue's own data-x (glyph center = tick), so the band frames the
    # letters regardless of x_shift and can be repainted around any (re-fitted) letter size.
    for e, c in zip(exts, colors):
        xc = 0.5 * (e.x0 + e.x1)
        rect = mpl.patches.Rectangle((xc - 0.5, y0), width=1.0, height=y1 - y0,
                                     facecolor=c, edgecolor="none", linewidth=0,
                                     zorder=0.1, clip_on=False, gid="_seq_cell")
        ax.add_patch(rect)


def repaint_seq_cell_backgrounds_(ax=None, labels=None, colors=None):
    """Re-draw the full-width residue cells to frame the FINAL (re-fitted) letters (``fill`` mode).

    The initial cells are painted at the seed layout; once the constant-cell sizer rescales the
    figure and ``update_seq_size_`` re-fits the residue letters, those seed cells no longer frame
    the letters (the band reads too thin and clips them). Remove the tagged cells and repaint around
    the current letters, so the gap-free band always fully covers the transparent-boxed residues.
    """
    for p in [p for p in ax.patches if p.get_gid() == "_seq_cell"]:
        p.remove()
    _add_seq_cell_backgrounds(ax=ax, labels=labels, colors=colors)


def _add_part_seq(ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None, x_shift=0.0, seq_size=None,
                  tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                  fill=False):
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
    # `pad` opens a small gap between the heatmap grid and the sequence band (like the cheat sheet).
    ax.tick_params(axis="x", length=0, which="both", pad=_SEQ_HEATMAP_GAP_PT)
    # Get labels in order of sequence (separate between minor and major ticks)
    dict_pos_label = dict(zip(minor_xticks, ax.xaxis.get_ticklabels(which="minor")))
    dict_pos_label.update(dict(zip(major_xticks, ax.xaxis.get_ticklabels(which="major"))))
    labels = list(dict(sorted(dict_pos_label.items(), key=lambda item: item[0])).values())
    # Adjust font size to prevent overlap
    if seq_size is None:
        seq_size = _get_optimal_fontsize(ax=ax, labels=labels, fill=fill)
    # Color the sequence to indicate TMD, JMD. With fill=True the glyph-sized text bbox leaves
    # hairline gaps between narrow letters (and reads as ragged against the uniform heatmap grid),
    # so paint a seamless full-width cell per residue instead and keep each letter's own box
    # transparent. fill=False keeps the legacy glyph bbox (byte-identical to the pre-auto_font path).
    lw = plt.gcf().get_size_inches()[0]/5
    for l, c in zip(labels, colors):
        l.set_fontsize(seq_size)
        box_color = "none" if fill else c
        l.set_bbox(dict(facecolor=box_color, edgecolor=box_color, zorder=0.1, alpha=1, clip_on=False,
                        pad=0, linewidth=lw))
        l.set_color(dict_seq_color[c])
    if fill:
        _add_seq_cell_backgrounds(ax=ax, labels=labels, colors=colors, x_shift=x_shift)
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
    # Offset the twin axis below the host (extra gap so the part labels clear the sequence band)
    y_pos = (5 + _JMD_LABEL_GAP_PT) if heatmap else 7
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


def finalize_part_labels_(fig=None, ax=None, gap_pt=_JMD_LABEL_GAP_PT, cap=_JMD_LABEL_MAX_PT,
                          fontsize_tmd_jmd=None, weight_tmd_jmd="normal"):
    """Cap the TMD/JMD part-label font and pin it a constant gap below the sequence band.

    Re-places the JMD-N/TMD/JMD-C part labels AFTER the residue letters are final (so the
    measurement reflects the letters the user sees): the label font is capped at ``cap``
    (unless ``fontsize_tmd_jmd`` is given, which is honored uncapped), and the label row is
    put a constant ``gap_pt`` points below the MEASURED bottom of the sequence letters.
    Both the label size and the label-to-sequence distance otherwise ride the residue-letter
    size, which balloons on short/wide grids; pinning them here keeps them constant across
    sequence length and subcategory count. ``ax`` is the grid/profile axes carrying the
    sequence letters; the part labels live on a sibling twin axes. No-op if either is absent.
    """
    name_tmd = ut.options["name_tmd"]
    name_jmd_n = ut.options["name_jmd_n"]
    name_jmd_c = ut.options["name_jmd_c"]
    names = {name_tmd, name_jmd_n, name_jmd_c}
    part_ax = next((a for a in fig.axes
                    if any(t.get_text() in names for t in a.get_xticklabels())), None)
    if part_ax is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    seq_labels = [t for t in ax.get_xticklabels(which="both") if t.get_text().strip()]
    if not seq_labels:
        return
    seq_size = max(t.get_fontsize() for t in seq_labels)
    band_bottom = min(t.get_window_extent(renderer).y0 for t in seq_labels)
    ax_bottom = ax.get_window_extent(renderer).y0
    band_below_pt = max(0.0, (ax_bottom - band_bottom) / fig.dpi * 72.0)
    size = fontsize_tmd_jmd if fontsize_tmd_jmd is not None else min(seq_size, cap)
    part_ax.spines["bottom"].set_position(("outward", band_below_pt + gap_pt))
    for label in part_ax.get_xticklabels():
        if label.get_text() in names:
            label.set_size(size)
            label.set_weight(weight_tmd_jmd)


def place_colorbar_below_grid_(fig=None, ax=None, gap_in=_CBAR_BELOW_GRID_IN):
    """Re-place the heatmap colorbar a constant distance below the grid (grid-relative).

    The colorbar is otherwise anchored near the figure bottom, which sits below the sequence
    only when the grid fills the figure; on a sparse grid (small figure) that anchor rides up
    into the sequence band. Pinning the colorbar's top ``gap_in`` inches below the grid bottom
    keeps it clear of the sequence at any grid size, and its importance-dot legend (anchored to
    the colorbar axes) follows. No-op if the grid has no attached colorbar.
    """
    if not ax.collections or ax.collections[0].colorbar is None:
        return
    cbar_ax = ax.collections[0].colorbar.ax
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    grid_pos = ax.get_position()
    cbar_pos = cbar_ax.get_position()
    top_in = grid_pos.y0 * h_in - gap_in
    new_bottom = (top_in - cbar_pos.height * h_in) / h_in
    cbar_ax.set_position([cbar_pos.x0, new_bottom, cbar_pos.width, cbar_pos.height])


def align_bottom_furniture_(fig=None, ax=None, gap_in=_FURNITURE_GAP_IN, item_gap_in=_FURNITURE_ITEM_GAP_IN):
    """Lay the scale-category legend, colorbar and importance legend as one COMPACT bottom row.

    Clustered near the grid centre in left / middle / right order with small ``item_gap_in`` gaps,
    a small ``gap_in`` below the sequence/part-label block, tops on one line. The colorbar is a
    fixed-inch-thick, fixed-inch-wide gradient bar (the backend sizes it as a figure fraction, which
    collapses when the figure shrinks). Run after ``grow_to_fit``, in figure coordinates. Missing
    items (e.g. no importance legend on the standalone heatmap) are skipped.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    cat_legend = ax.get_legend()
    cbar = ax.collections[0].colorbar if (ax.collections and ax.collections[0].colorbar) else None
    cbar_ax = cbar.ax if cbar is not None else None
    imp_legend = cbar_ax.get_legend() if cbar_ax is not None else None
    grid = ax.get_position()
    # Lowest grid-attached content = min bottom of the grid axes' own tight bbox (position ticks,
    # sequence letters, row labels; the legend is hidden so it does not feed back) and any sibling
    # twin sharing the grid position (the part-label JMD/TMD axis, whose spine is pushed below).
    if cat_legend is not None:
        cat_legend.set_visible(False)
    fig.canvas.draw()
    low_disp = ax.get_tightbbox(renderer).y0
    for a in fig.axes:
        if a is ax or a is cbar_ax:
            continue
        p = a.get_position()
        if abs(p.x0 - grid.x0) < 0.02 and abs(p.width - grid.width) < 0.02:
            low_disp = min(low_disp, a.get_tightbbox(renderer).y0)
    if cat_legend is not None:
        cat_legend.set_visible(True)
    top_in = low_disp / dpi - gap_in
    top_frac = top_in / h_in
    if cat_legend is not None:  # drop the blank-line title padding (an old positioning hack)
        title = cat_legend.get_title()
        title.set_text(title.get_text().lstrip("\n"))
    # Colorbar: fixed thickness AND width (inches), centred on the grid, top-aligned.
    cbar_left_in = cbar_right_in = None
    if cbar_ax is not None:
        pos = cbar_ax.get_position()
        if imp_legend is not None:
            imp_legend.set_visible(False)
            fig.canvas.draw()
        label_above_in = cbar_ax.get_tightbbox(renderer).y1 / dpi - pos.y1 * h_in
        if imp_legend is not None:
            imp_legend.set_visible(True)
        cbar_w_in = min(max((grid.x1 - grid.x0) * w_in * 0.42, 1.4), 3.0)
        cbar_left_in = 0.5 * (grid.x0 + grid.x1) * w_in - 0.5 * cbar_w_in
        cbar_right_in = cbar_left_in + cbar_w_in
        bottom = (top_in - label_above_in - _CBAR_HEIGHT_IN) / h_in
        cbar_ax.set_position([cbar_left_in / w_in, bottom, cbar_w_in / w_in, _CBAR_HEIGHT_IN / h_in])
    # Category legend just LEFT of the colorbar; importance legend just RIGHT of it.
    if cat_legend is not None:
        right_edge = (cbar_left_in - item_gap_in) if cbar_left_in is not None else (grid.x0 * w_in - item_gap_in)
        cat_legend.set_loc("upper right")
        cat_legend.set_bbox_to_anchor((right_edge / w_in, top_frac), transform=fig.transFigure)
    if imp_legend is not None:
        left_edge = (cbar_right_in + item_gap_in) if cbar_right_in is not None else (grid.x1 * w_in + item_gap_in)
        imp_legend.set_loc("upper left")
        imp_legend.set_bbox_to_anchor((left_edge / w_in, top_frac), transform=fig.transFigure)


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
    def get_optimal_fontsize(ax, labels, max_x_dist=0.1, fill=False):
        """Get sequence fontsize optimized to not overlap (``fill=True`` also removes whitespace)"""
        opt_fs = _get_optimal_fontsize(ax=ax, labels=labels, max_x_dist=max_x_dist, fill=fill)
        return round(opt_fs, 2)

    def add_tmd_jmd_seq(self, ax=None, jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                        tmd_color="mediumspringgreen", jmd_color="blue",
                        tmd_seq_color="black", jmd_seq_color="white",
                        add_xticks_pos=False, heatmap=True,
                        x_shift=0, xtick_size=11, seq_size=None,
                        fontsize_tmd_jmd=None, weight_tmd_jmd="normal", fill=False):
        """Add sequences and corresponding x-ticks for TMD and JMD regions."""
        seq_size = _add_part_seq(ax=ax, jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq, x_shift=x_shift,
                                 seq_size=seq_size, tmd_color=tmd_color, jmd_color=jmd_color,
                                 tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color, fill=fill)
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
        # direction="out" so the tick marks point DOWN (below the TMD-JMD bar),
        # not up into the heatmap cells.
        ax.tick_params(axis="x", color="black", length=xtick_length, width=xtick_width,
                       direction="out")
        ax.xaxis.set_ticks_position(xticks_position)
