"""Constant grid-cell sizing for the CPP composite plots under ``auto_font``.

The stability guarantee is: hold every grid **cell** at a constant physical size
(inches) regardless of how many subcategory rows or residue-position columns the
grid has. As the data grows the *figure* grows; the cells — and therefore the
fonts, which are measured in points — stay put. This is what keeps subcategory
row-labels, position ticks and residue letters legible at any data size without
the caller hand-tuning ``plot_settings(font_scale=...)``.

Mechanism (see :func:`fit_cells_by_rescale`): the frontend first lays the figure
out normally (gridspec + ``tight_layout`` + manually-placed colorbar/legends),
which positions every axes in *figure fractions*. We then measure the heatmap's
fractional size and rescale the whole figure so that
``fraction * figure_inches / n_cells`` equals the target cell size. Because we do
NOT re-run ``tight_layout``, the fractions are preserved, so a single rescale is
exact and every sibling axes (importance bars, colorbar, legends) scales with the
heatmap and stays glued. Fonts, being in points, do not scale — they remain at the
``plot_settings`` values.
"""
import numpy as np

# I Constants
#: Target on-figure size (inches) of a single heatmap cell — one residue-position
#: column wide, one subcategory row tall. Calibrated so the shipped ``DOM_GSEC``
#: feature map (~36 subcategories x ~40 positions) matches its hand-tuned look; a
#: ~1 : 1.1 width:height keeps cells slightly taller than wide, as in the notebook.
CELL_W_IN = 0.16
CELL_H_IN = 0.176

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
    are already correct, so one rescale is exact. Returns ``(fig_w, fig_h, capped)``.
    """
    fig.canvas.draw()
    pos = ax_grid.get_position()
    frac_w, frac_h = pos.width, pos.height
    if frac_w <= 0 or frac_h <= 0:
        w_in, h_in = fig.get_size_inches()
        return float(w_in), float(h_in), False
    target_w = n_cols * cell_w / frac_w
    target_h = n_rows * cell_h / frac_h
    capped = target_w > SAFETY_CAP_IN or target_h > SAFETY_CAP_IN
    target_w = min(target_w, SAFETY_CAP_IN)
    target_h = min(target_h, SAFETY_CAP_IN)
    fig.set_size_inches(target_w, target_h)
    return float(target_w), float(target_h), capped


def fit_width_by_rescale(fig=None, ax_grid=None, n_cols=None, cell_w=CELL_W_IN):
    """Rescale ``fig`` WIDTH only so per-position column width stays ``cell_w`` inches.

    For plots with a position x-axis but no subcategory rows (e.g. the profile):
    width grows with the sequence length while the height (and all fonts) stay
    fixed. Returns the new figure width in inches.
    """
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    pos = ax_grid.get_position()
    if pos.width <= 0:
        return float(w_in)
    target_w = min(n_cols * cell_w / pos.width, SAFETY_CAP_IN)
    fig.set_size_inches(target_w, h_in)
    return float(target_w)


def ranking_figheight(n_items=None, per_item_in=0.22, base_in=1.0):
    """Height (inches) for a ranked-bar plot of ``n_items`` bars.

    Encodes the maintainer's manual notebook rule ``figsize=(5, 0.22*n + 1)`` so a
    ranking figure grows linearly with the number of ranked features/items while
    each bar (and its label) keeps a constant height.
    """
    return float(max(base_in, per_item_in * int(n_items or 0) + base_in))
