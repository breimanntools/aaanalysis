"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``plot_eval`` helper: it turns a
:func:`aaanalysis.pipe.find_features` sweep table into a set of **publication-ready** evaluation
figures. The high-dimensional Part x Split x Scale grid is decomposed into a series of clean 2D
``viridis`` heatmaps — the two most-informative axes on each panel, the least-informative axis as the
slice — sharing one color scale, every cell annotated with its score and the selected configuration
boxed, plus a marginal-impact bar panel and an ``n_filter`` refinement panel. Each figure is returned
separately so it drops straight into a paper.
"""
from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import aaanalysis.utils as ut


COL_SELECTED = "is_selected"
COL_STAGE = "stage"
# Structural sweep axes (the Part x Split x Scale x JMD-length grid), in fallback inner-priority
# order; ``n_filter`` is a refinement sweep shown in its own panel, not a heatmap axis.
# ``pattern_mode`` is the split-type axis (a 1:1 recoding of ``split_types``); the two are never both
# counted. ``n_jmd`` is the symmetric JMD length (jmd_n_len = jmd_c_len), a numeric heatmap axis.
LIST_STRUCT_AXES = ["list_parts", "pattern_mode", "n_split_max", "n_jmd", "scale"]
LIST_NUMERIC_AXES = ["n_filter", "n_split_max", "n_jmd"]
DICT_AXIS_LABEL = {"n_filter": "n_filter", "n_split_max": "n_split_max", "scale": "scale set",
                   "list_parts": "parts", "pattern_mode": "splits (pattern mode)",
                   "n_jmd": "JMD length (n_jmd)"}
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


def _metric_label(score_col):
    """Readable metric name for titles / colorbars (``balanced_accuracy_mean`` -> ``CV balanced accuracy``)."""
    name = score_col[:-5] if score_col.endswith("_mean") else score_col
    name = name.replace("_", " ").strip()
    return name if name.lower().startswith("cv ") else f"CV {name}"


def _text_color(value, vmin, vmax, cmap):
    """Black or white annotation text, whichever contrasts with the cell's colormap color."""
    if value is None or np.isnan(value):
        return "0.3"
    span = (vmax - vmin) or 1.0
    r, g, b, _ = cmap((value - vmin) / span)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


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


def _draw_heatmap(ax, grid, x_levels, y_levels, x_axis, y_axis, vmin, vmax, best=None, fmt="{:.2f}"):
    """Render one annotated viridis heatmap panel; ``best`` is the (xi, yi) winner cell (red box)."""
    cmap = plt.get_cmap(STR_CMAP)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([_level_label(x_axis, v) for v in x_levels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels([_level_label(y_axis, v) for v in y_levels])
    ax.set_xlabel(DICT_AXIS_LABEL.get(x_axis, x_axis))
    ax.set_ylabel(DICT_AXIS_LABEL.get(y_axis, y_axis))
    # Annotate every populated cell with its value (auto-contrasting text), like the reference heatmaps.
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            v = grid[yi, xi]
            if not np.isnan(v):
                ax.text(xi, yi, fmt.format(v), ha="center", va="center", fontsize=8,
                        color=_text_color(v, vmin, vmax, cmap))
    # Crisp white cell separators (the published-heatmap look).
    ax.set_xticks(np.arange(-0.5, len(x_levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_levels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    # Mark the winning configuration with a box (not a star), matching the publication style.
    if best is not None:
        ax.add_patch(plt.Rectangle((best[0] - 0.5, best[1] - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=2.5, zorder=5))
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
    fig, ax = plt.subplots(figsize=figsize or (1.15 * len(x_levels) + 3.4, 0.78 * len(y_levels) + 2.4),
                           constrained_layout=True)
    grid = _heatmap_grid(df_grid, x_axis, y_axis, score_col, x_levels, y_levels)
    best = _star_cell(best_coords, x_axis, y_axis, x_levels, y_levels) if best_coords else None
    im = _draw_heatmap(ax, grid, x_levels, y_levels, x_axis, y_axis, vmin, vmax, best=best)
    fig.colorbar(im, ax=ax, label=_metric_label(score_col))
    ax.set_title(title, fontsize=11)
    return fig


def _fig_marginal(df_grid, axes, score_col, figsize):
    """Marginal-impact bar Figure: how much each swept axis moves the score (max - min)."""
    impacts = {a: _axis_impact(df_grid, a, score_col) for a in axes}
    fig, ax = plt.subplots(figsize=figsize or (1.15 * len(axes) + 2.6, 3.8), constrained_layout=True)
    names = list(impacts)
    vals = [impacts[a] for a in names]
    bars = ax.bar(range(len(names)), vals, color="0.6", edgecolor="0.2", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([DICT_AXIS_LABEL.get(a, a) for a in names], rotation=20, ha="right")
    ax.set_ylabel("Impact (max − min)")
    ax.set_ylim(0, (max(vals) * 1.18) if vals else 1.0)
    ax.set_title(f"Axis impact on {_metric_label(score_col)}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _fig_nfilter(df_nf, score_col, std_col, best_coords, figsize):
    """``n_filter`` refinement Figure: score vs n_filter (a line over the dominant-axis winner)."""
    # Drop levels whose score is undefined (all-NaN) so min/max and the color norm stay finite.
    pairs = [(v, float(df_nf.loc[_row_mask(df_nf, {"n_filter": v}), score_col].mean()))
             for v in _axis_levels(df_nf, "n_filter")]
    pairs = [(v, m) for v, m in pairs if not np.isnan(m)]
    levels, means = [v for v, _ in pairs], [m for _, m in pairs]
    fig, ax = plt.subplots(figsize=figsize or (6.0, 3.8), constrained_layout=True)
    x = np.arange(len(levels))
    lo, hi = (min(means), max(means)) if means else (0.0, 1.0)
    cmap, norm = plt.get_cmap(STR_CMAP), plt.Normalize(lo, hi if hi > lo else lo + 1e-9)
    ax.plot(x, means, color="0.6", zorder=1)
    if std_col is not None and std_col in df_nf.columns:
        stds = [float(df_nf.loc[_row_mask(df_nf, {"n_filter": v}), std_col].mean()) for v in levels]
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="0.7", capsize=3, zorder=1)
    ax.scatter(x, means, c=[cmap(norm(m)) for m in means], s=90, edgecolors="0.3", zorder=2)
    bx = best_coords.get("n_filter") if best_coords else None
    bpos = next((i for i, v in enumerate(levels) if (pd.isna(v) and pd.isna(bx)) or v == bx), None)
    if bpos is not None:
        ax.scatter([bpos], [means[bpos]], marker="*", s=300, c="red", edgecolors="white",
                   linewidths=0.8, zorder=4)
        ax.annotate(f"{means[bpos]:.3f}", (bpos, means[bpos]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="red", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label("n_filter", v) for v in levels], rotation=45, ha="right")
    ax.set_xlabel("n_filter")
    ax.set_ylabel(_metric_label(score_col))
    ax.set_title("n_filter refinement", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
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
    panels are directly comparable, every cell annotated with its score and the selected configuration
    boxed. Alongside the heatmaps
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
                if facet_axes and sub.empty:
                    continue   # a sparse/correlated sweep can leave a facet combo unpopulated
                base = (f"{_metric_label(score_col)}: "
                        f"{DICT_AXIS_LABEL.get(y_axis, y_axis)} × {DICT_AXIS_LABEL.get(x_axis, x_axis)}")
                facet = ", ".join(f"{DICT_AXIS_LABEL.get(a, a)}={_level_label(a, v)}"
                                  for a, v in coords.items())
                title = f"{base}\n({facet})" if facet else base
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
