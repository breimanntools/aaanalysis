"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``plot_eval`` helper: it turns a
:func:`aaanalysis.pipe.find_features` sweep table into a set of **publication-ready** evaluation
figures. The high-dimensional Part x Split x Scale grid is decomposed into a series of clean 2D
``viridis`` heatmaps — the two most-informative axes on each panel, the least-informative axis as the
slice — sharing one color scale, with the selected configuration starred, plus a marginal-impact bar
panel and an ``n_filter`` refinement panel. Each figure is returned separately so it drops straight
into a paper.
"""
from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import aaanalysis.utils as ut


COL_SELECTED = "is_selected"
COL_STAGE = "stage"
# Structural sweep axes (the Part x Split x Scale grid), in fallback inner-priority order; ``n_filter``
# is a refinement sweep shown in its own panel, not a heatmap axis. ``pattern_mode`` is the split-type
# axis (a 1:1 recoding of ``split_types``); the two are never both counted.
LIST_STRUCT_AXES = ["list_parts", "pattern_mode", "n_split_max", "scale"]
LIST_NUMERIC_AXES = ["n_filter", "n_split_max"]
DICT_AXIS_LABEL = {"n_filter": "n_filter", "n_split_max": "n_split_max", "scale": "scale set",
                   "list_parts": "parts", "pattern_mode": "splits (pattern mode)"}
STR_CMAP = "viridis"


# I Helper Functions
def _resolve_score_col(df_eval, metric=None, score_col=None):
    """Pick the score column: explicit ``score_col``, else ``metric``+'_mean', else first ``*_mean``."""
    if score_col is not None:
        return score_col
    if metric is not None:
        return metric + "_mean"
    means = [c for c in df_eval.columns if c.endswith("_mean")]
    if not means:
        raise ValueError(f"'df_eval' columns ({list(df_eval.columns)}) should contain a '<metric>_mean' "
                         f"score column, or pass 'score_col'.")
    return means[0]


def _check_df_eval(df_eval=None, score_col=None):
    """Validate the eval table and that the score column is present and numeric."""
    if not isinstance(df_eval, pd.DataFrame):
        raise ValueError(f"'df_eval' ({type(df_eval).__name__}) should be a pd.DataFrame.")
    if len(df_eval) == 0:
        raise ValueError("'df_eval' (empty) should have at least one configuration row.")
    if score_col not in df_eval.columns:
        raise ValueError(f"'score_col' ({score_col}) should be a column of df_eval "
                         f"({list(df_eval.columns)}).")
    if df_eval[score_col].dropna().empty:
        raise ValueError(f"'df_eval[{score_col!r}]' (all-NaN) should have at least one scored "
                         f"configuration.")


def _sensitivity_view(df_eval):
    """Rows that form the structural sweep grid (the ``sensitivity`` stage if the column is present)."""
    if COL_STAGE in df_eval.columns and (df_eval[COL_STAGE] == "sensitivity").any():
        return df_eval[df_eval[COL_STAGE] == "sensitivity"].reset_index(drop=True)
    return df_eval.reset_index(drop=True)


def _axis_impact(df_eval, col, score_col):
    """Marginal-mean impact (``max - min`` of the per-level mean score) of one axis."""
    marg = df_eval.groupby(col, dropna=False)[score_col].mean()
    return float(marg.max() - marg.min()) if len(marg) else 0.0


def _ranked_axes(df_eval, score_col):
    """Structural axes that vary, ranked by impact (most-informative first)."""
    axes = [c for c in LIST_STRUCT_AXES
            if c in df_eval.columns and df_eval[c].nunique(dropna=False) > 1]
    return sorted(axes, key=lambda a: _axis_impact(df_eval, a, score_col), reverse=True)


def _axis_levels(df_eval, col):
    """Sorted unique levels of an axis (numeric-aware; ``None`` sorts last)."""
    vals = list(df_eval[col].unique())
    if col in LIST_NUMERIC_AXES:
        nums = sorted(v for v in vals if v is not None and not pd.isna(v))
        return nums + ([None] if any(v is None or pd.isna(v) for v in vals) else [])
    return sorted(vals, key=lambda v: "" if v is None else str(v))


def _level_label(col, v):
    """Display label for one axis level."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "all"
    if col in LIST_NUMERIC_AXES:
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v)


def _best_row(df_eval, score_col):
    """Index label of the selected/best configuration (``is_selected`` if present, else argmax)."""
    if COL_SELECTED in df_eval.columns and bool(df_eval[COL_SELECTED].any()):
        return df_eval.index[df_eval[COL_SELECTED].to_numpy().astype(bool)][0]
    return df_eval[score_col].idxmax()


def _row_mask(df_eval, coords):
    """Boolean mask of rows matching every (col -> value) in ``coords`` (NaN/None-safe)."""
    mask = pd.Series(True, index=df_eval.index)
    for col, val in coords.items():
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            mask &= df_eval[col].isna()
        else:
            mask &= (df_eval[col] == val)
    return mask


def _heatmap_grid(df_eval, x_axis, y_axis, score_col, x_levels, y_levels):
    """Mean-score grid over (y_axis x x_axis) for the given pinned level lists (NaN/None-safe)."""
    grid = np.full((len(y_levels), len(x_levels)), np.nan)
    for yi, yv in enumerate(y_levels):
        for xi, xv in enumerate(x_levels):
            mask = _row_mask(df_eval, {x_axis: xv, y_axis: yv})
            if mask.any():
                grid[yi, xi] = float(df_eval.loc[mask, score_col].mean())
    return grid


def _star_cell(coords, x_axis, y_axis, x_levels, y_levels):
    """(xi, yi) cell index of the best configuration on this panel's grid, or None if off-grid."""
    bx, by = coords.get(x_axis), coords.get(y_axis)
    xi = next((i for i, v in enumerate(x_levels) if (pd.isna(v) and pd.isna(bx)) or v == bx), None)
    yi = next((i for i, v in enumerate(y_levels) if (pd.isna(v) and pd.isna(by)) or v == by), None)
    return (xi, yi) if xi is not None and yi is not None else None


def _draw_heatmap(ax, grid, x_levels, y_levels, x_axis, y_axis, vmin, vmax, star=None):
    """Render one viridis heatmap panel; ``star`` is the (xi, yi) cell to mark as the best."""
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=STR_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([_level_label(x_axis, v) for v in x_levels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels([_level_label(y_axis, v) for v in y_levels])
    ax.set_xlabel(DICT_AXIS_LABEL.get(x_axis, x_axis))
    ax.set_ylabel(DICT_AXIS_LABEL.get(y_axis, y_axis))
    if star is not None:
        ax.scatter([star[0]], [star[1]], marker="*", s=240, c="red",
                   edgecolors="white", linewidths=0.8, zorder=5)
    return im


def _product(list_levels):
    """Cartesian product of the per-axis level lists (empty -> a single empty combo)."""
    combos = [()]
    for levels in list_levels:
        combos = [c + (v,) for c in combos for v in levels]
    return combos


def _fig_heatmap(df_grid, x_axis, y_axis, score_col, x_levels, y_levels, vmin, vmax,
                 best_coords, title, figsize):
    """One self-contained heatmap Figure (a single publication panel)."""
    fig, ax = plt.subplots(figsize=figsize or (1.2 * len(x_levels) + 3.0, 0.9 * len(y_levels) + 2.5),
                           constrained_layout=True)
    grid = _heatmap_grid(df_grid, x_axis, y_axis, score_col, x_levels, y_levels)
    star = _star_cell(best_coords, x_axis, y_axis, x_levels, y_levels) if best_coords else None
    im = _draw_heatmap(ax, grid, x_levels, y_levels, x_axis, y_axis, vmin, vmax, star=star)
    fig.colorbar(im, ax=ax, label=score_col)
    ax.set_title(title, fontsize=10)
    return fig


def _fig_marginal(df_grid, axes, score_col, figsize):
    """Marginal-impact bar Figure: how much each swept axis moves the score (max - min)."""
    impacts = {a: _axis_impact(df_grid, a, score_col) for a in axes}
    fig, ax = plt.subplots(figsize=figsize or (0.9 * len(axes) + 3.0, 3.5), constrained_layout=True)
    names = list(impacts)
    ax.bar(range(len(names)), [impacts[a] for a in names], color="0.5", edgecolor="0.2")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([DICT_AXIS_LABEL.get(a, a) for a in names], rotation=30, ha="right")
    ax.set_ylabel(f"impact (max - min {score_col})")
    ax.set_title("Axis impact (sensitivity)", fontsize=10)
    return fig


def _fig_nfilter(df_nf, score_col, std_col, best_coords, figsize):
    """``n_filter`` refinement Figure: score vs n_filter (a line over the dominant-axis winner)."""
    levels = _axis_levels(df_nf, "n_filter")
    means = [float(df_nf.loc[_row_mask(df_nf, {"n_filter": v}), score_col].mean()) for v in levels]
    fig, ax = plt.subplots(figsize=figsize or (6.0, 3.8), constrained_layout=True)
    x = np.arange(len(levels))
    cmap, norm = plt.get_cmap(STR_CMAP), plt.Normalize(min(means), max(means) if max(means) > min(means) else min(means) + 1e-9)
    ax.plot(x, means, color="0.6", zorder=1)
    if std_col is not None and std_col in df_nf.columns:
        stds = [float(df_nf.loc[_row_mask(df_nf, {"n_filter": v}), std_col].mean()) for v in levels]
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="0.7", capsize=3, zorder=1)
    ax.scatter(x, means, c=[cmap(norm(m)) for m in means], s=90, edgecolors="0.3", zorder=2)
    bx = best_coords.get("n_filter") if best_coords else None
    bpos = next((i for i, v in enumerate(levels) if v == bx), None)
    if bpos is not None:
        ax.scatter([bpos], [means[bpos]], marker="*", s=260, c="red", edgecolors="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label("n_filter", v) for v in levels], rotation=45, ha="right")
    ax.set_xlabel("n_filter")
    ax.set_ylabel(score_col)
    ax.set_title("n_filter refinement", fontsize=10)
    return fig


# II Main Functions
def plot_eval(df_eval: pd.DataFrame,
              metric: Optional[str] = None,
              score_col: Optional[str] = None,
              figsize: Optional[Tuple[float, float]] = None,
              ) -> List[Figure]:
    """
    Decompose a :func:`find_features` sweep into a set of publication-ready evaluation figures.

    The high-dimensional structural sweep (Part x Split x Scale) is **decomposed** rather than
    crammed into one plot: the two **most-informative** axes (largest marginal-mean impact on the
    score) become each heatmap's rows and columns, and the least-informative axis becomes the
    **slice** — one clean 2D ``viridis`` heatmap per slice level, all sharing a single color scale so
    panels are directly comparable, with the selected configuration starred. Alongside the heatmaps
    it returns a **marginal-impact** bar panel (how much each lever moves the score) and an
    **``n_filter``** refinement panel. Every figure is returned separately so each can be saved and
    placed individually in a publication.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Per-configuration sweep table — the third return value of :func:`find_features` (with a
        ``stage`` column and one ``<metric>_mean`` column per metric), or any compatible eval table.
        An ``is_selected`` column (if present) marks the winner; otherwise the maximum score is used.
    metric : str, optional
        Metric whose ``<metric>_mean`` column is colored. If ``None``, the first ``*_mean`` column.
    score_col : str, optional
        Explicit score column to color by (overrides ``metric``).
    figsize : tuple of float, optional
        Per-figure size in inches. If ``None``, a size is derived from each panel's grid.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        The evaluation figures (heatmap slices, then the marginal-impact panel, then the ``n_filter``
        panel). Empty when no axis varies (a single configuration, e.g. ``search="fast"``).

    See Also
    --------
    * :func:`find_features` : the CPP AutoML pipeline whose ``df_eval`` this visualizes; it attaches
      these figures to the returned ``ax`` as ``ax.eval`` when ``plot=True``.

    Examples
    --------
    .. include:: examples/aap_plot_eval.rst
    """
    # Validate
    if not isinstance(df_eval, pd.DataFrame):
        raise ValueError(f"'df_eval' ({type(df_eval).__name__}) should be a pd.DataFrame.")
    score_col = _resolve_score_col(df_eval, metric=metric, score_col=score_col)
    _check_df_eval(df_eval=df_eval, score_col=score_col)
    std_col = score_col.replace("_mean", "_std")
    std_col = std_col if std_col in df_eval.columns else None

    # Coordinates of the selected winner (used to star the matching cell across panels).
    best_coords = {}
    if COL_SELECTED in df_eval.columns and bool(df_eval[COL_SELECTED].any()):
        best = df_eval[df_eval[COL_SELECTED].to_numpy().astype(bool)].iloc[0]
        best_coords = {a: best[a] for a in LIST_STRUCT_AXES + ["n_filter"] if a in df_eval.columns}

    df_grid = _sensitivity_view(df_eval)
    axes = _ranked_axes(df_grid, score_col)
    figs: List[Figure] = []
    if axes:
        vmin, vmax = float(df_grid[score_col].min()), float(df_grid[score_col].max())
        if len(axes) == 1:
            # A single varying axis: one heatmap with that axis on x and a dummy single-row y is
            # over-engineered — fold it into the marginal panel only.
            pass
        else:
            x_axis, y_axis = axes[0], axes[1]
            x_levels, y_levels = _axis_levels(df_grid, x_axis), _axis_levels(df_grid, y_axis)
            facet_axes = axes[2:]
            for combo in _product([_axis_levels(df_grid, a) for a in facet_axes]):
                coords = dict(zip(facet_axes, combo))
                sub = df_grid[_row_mask(df_grid, coords)] if facet_axes else df_grid
                title = ", ".join(f"{DICT_AXIS_LABEL.get(a, a)}={_level_label(a, v)}"
                                  for a, v in coords.items()) or "sensitivity grid"
                figs.append(_fig_heatmap(sub, x_axis, y_axis, score_col, x_levels, y_levels,
                                         vmin, vmax, best_coords, title, figsize))
        figs.append(_fig_marginal(df_grid, axes, score_col, figsize))

    # n_filter refinement panel (its own stage).
    if COL_STAGE in df_eval.columns:
        df_nf = df_eval[df_eval[COL_STAGE] == "n_filter"]
        if "n_filter" in df_eval.columns and df_nf["n_filter"].nunique(dropna=False) > 1:
            figs.append(_fig_nfilter(df_nf.reset_index(drop=True), score_col, std_col,
                                     best_coords, figsize))
    return figs
