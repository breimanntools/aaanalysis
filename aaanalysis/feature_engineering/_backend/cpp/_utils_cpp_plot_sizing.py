"""Constant grid-cell sizing for the CPP composite plots.

The stability guarantee is: hold every grid **cell** at a constant physical size
(inches) regardless of how many subcategory rows or residue-position columns the
grid has. The *figure* shrinks for a small grid and grows for a large one; the
cells — and therefore the fonts, which are measured in points, and the residue
letters, which are fitted to the cell — stay put. This keeps subcategory
row-labels, position ticks and residue letters legible at any data size without
the caller hand-tuning ``plot_settings(font_scale=...)``.

Two-stage mechanism, both operating on an already-laid-out figure (gridspec +
``tight_layout`` + manually-placed colorbar/legends) WITHOUT re-running
``tight_layout`` (which would re-proportion the grid and break the cell target):

1. :func:`fit_cells_by_rescale` measures the heatmap's *figure fraction* and
   rescales the whole figure so ``fraction * figure_inches / n_cells`` equals the
   target cell size. Because the fractions are preserved, the cell hits the target
   exactly — shrinking as well as growing (a sparse grid yields a small figure).
2. :func:`grow_to_fit` then enlarges the figure by any tight-bbox overflow and
   rigid-body translates every axes inward by the same amount, so the point-sized
   furniture (labels, colorbar, legends) that a shrunken figure would clip moves
   back inside the canvas. A translation is size-preserving, so the cells stay
   exactly on target. This is what makes the result clip-free at any grid size.

Fonts, being in points, never scale — they stay at the ``plot_settings`` values.
"""

# I Constants
#: Target on-figure size (inches) of a single heatmap cell — one residue-position
#: column wide, one subcategory row tall. Calibrated so the shipped ``DOM_GSEC``
#: feature map (~36 subcategories x ~40 positions) matches its hand-tuned look; a
#: ~1 : 1.1 width:height keeps cells slightly taller than wide, as in the notebook.
CELL_W_IN = 0.16
CELL_H_IN = 0.19

#: Target cell height (inches) for the STANDALONE heatmap. Kept equal to the feature-map
#: ``CELL_H_IN`` so both grids render with the same cell geometry; label overlap on a dense
#: heatmap is handled by the row-label overlap-shrink, not by a taller cell.
HEATMAP_CELL_H_IN = CELL_H_IN

#: Hard upper bound on either figure dimension. Reaching it means the grid is far
#: larger than anything sensible to render in one figure; we clamp and warn rather
#: than silently shrink the cells (which would reintroduce the overlap auto_font
#: exists to prevent).
SAFETY_CAP_IN = 200.0


# II Main Functions
def fit_cells_by_rescale(fig=None, ax_grid=None, n_rows=None, n_cols=None,
                         cell_w=CELL_W_IN, cell_h=CELL_H_IN):
    """Rescale ``fig`` so ``ax_grid``'s cells render at ``(cell_w, cell_h)`` inches.

    Must be called AFTER the figure is fully laid out (``tight_layout`` + colorbar
    /legends placed) and BEFORE any residue-letter fitting, so letters are fitted
    to the final column width. Does not re-run ``tight_layout``: the axes fractions
    are already correct, so one rescale hits the cell size exactly.

    The figure both SHRINKS (sparse grid -> small figure) and GROWS (dense grid ->
    large figure); there is no lower floor. On a shrunken figure the point-sized
    furniture (colorbar, legends, importance bars) can overflow the smaller canvas;
    :func:`grow_to_fit` is called next to reclaim that overflow without changing the
    cell size. Clamped above at ``SAFETY_CAP_IN``. Returns ``(fig_w, fig_h, capped)``.
    """
    fig.canvas.draw()
    pos = ax_grid.get_position()
    frac_w, frac_h = pos.width, pos.height
    if frac_w <= 0 or frac_h <= 0:
        w_in, h_in = fig.get_size_inches()
        return float(w_in), float(h_in), False
    want_w = n_cols * cell_w / frac_w
    want_h = n_rows * cell_h / frac_h
    capped = want_w > SAFETY_CAP_IN or want_h > SAFETY_CAP_IN
    target_w = min(want_w, SAFETY_CAP_IN)
    target_h = min(want_h, SAFETY_CAP_IN)
    fig.set_size_inches(target_w, target_h)
    return float(target_w), float(target_h), capped


def fit_width_by_rescale(fig=None, ax_grid=None, n_cols=None, cell_w=CELL_W_IN):
    """Rescale ``fig`` WIDTH only so per-position column width stays ``cell_w`` inches.

    For plots with a position x-axis but no subcategory rows (e.g. the profile):
    the width tracks the sequence length (shrinking for a short sequence, growing
    for a long one) while the height and all fonts stay fixed. No lower floor;
    clamped above at ``SAFETY_CAP_IN``. Returns ``(fig_w, capped)``.
    """
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    pos = ax_grid.get_position()
    if pos.width <= 0:
        return float(w_in), False
    want_w = n_cols * cell_w / pos.width
    capped = want_w > SAFETY_CAP_IN
    target_w = min(want_w, SAFETY_CAP_IN)
    fig.set_size_inches(target_w, h_in)
    return float(target_w), capped


def grow_to_fit(fig=None, max_passes=3, tol_in=0.002):
    """Enlarge ``fig`` and shift every axes inward so no content clips the figure edge.

    :func:`fit_cells_by_rescale` sizes the figure so the grid cells hit their target,
    which can leave the point-sized furniture (subcategory labels, colorbar, legends,
    part labels) overflowing a shrunken canvas. Measure the tight-bbox overflow on each
    side, grow the figure by exactly that overflow, and rigid-body translate every axes
    by the same amount: the grid cells keep their physical size (a translation is
    size-preserving, so the cell target is untouched) while the spilling furniture moves
    inside the canvas. Requires every furniture artist to ride an axes (the CPP legends,
    colorbar and part-label twin all do), so one pass is analytically exact; the loop is
    a safety margin. Clamped at ``SAFETY_CAP_IN``. Call LAST, after residue-letter
    fitting. Returns ``fig``.
    """
    for _ in range(max_passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w_in, h_in = (float(v) for v in fig.get_size_inches())
        tb = fig.get_tightbbox(renderer)  # inches, figure lower-left origin
        left = max(0.0, -tb.x0)
        bottom = max(0.0, -tb.y0)
        right = max(0.0, tb.x1 - w_in)
        top = max(0.0, tb.y1 - h_in)
        if max(left, bottom, right, top) < tol_in:
            break
        new_w = min(w_in + left + right, SAFETY_CAP_IN)
        new_h = min(h_in + bottom + top, SAFETY_CAP_IN)
        for ax in fig.axes:
            p = ax.get_position()
            ax.set_position([(p.x0 * w_in + left) / new_w,
                             (p.y0 * h_in + bottom) / new_h,
                             p.width * w_in / new_w,
                             p.height * h_in / new_h])
        fig.set_size_inches(new_w, new_h)
    return fig


def ranking_figheight(n_items=None, per_item_in=0.22, base_in=1.0):
    """Height (inches) for a ranked-bar plot of ``n_items`` bars.

    Encodes the maintainer's manual notebook rule ``figsize=(5, 0.22*n + 1)`` so a
    ranking figure grows linearly with the number of ranked features/items while
    each bar (and its label) keeps a constant height.
    """
    return float(max(base_in, per_item_in * int(n_items or 0) + base_in))
