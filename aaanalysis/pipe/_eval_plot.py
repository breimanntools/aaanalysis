"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``plot_eval`` helper: a
``viridis`` evaluation-grid plot of a CPP optimization sweep that adapts to the number of swept
axes. It visualizes the per-configuration cross-validated scores produced by
:func:`aaanalysis.pipe.find_features` (or any ``CPPGrid``-style evaluation table), marking the
selected configuration and, when available, annotating the simplify / RFE refinement deltas.
"""
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import aaanalysis.utils as ut


# Schema of the ``find_features`` sweep table this helper consumes (kept local so the plot stays
# usable standalone on any CPPGrid-style eval table without importing the pipeline module).
COL_SCORE = "cv_bacc_mean"
COL_STD = "cv_bacc_std"
COL_SELECTED = "is_selected"
# Logical sweep axes, in inner-priority order (most natural inner-x first). ``pattern_mode``
# represents the split-type axis (a 1:1 recoding of ``split_types``), so the two are never both
# counted as separate dimensions.
LIST_SWEEP_AXES = ["n_filter", "n_split_max", "n_explain", "list_parts", "pattern_mode"]
# Axes whose values are inherently numeric (ordered as numbers); the rest are categorical.
LIST_NUMERIC_AXES = ["n_filter", "n_split_max", "n_explain"]
# Pretty axis labels.
DICT_AXIS_LABEL = {"n_filter": "n_filter", "n_split_max": "n_split_max",
                   "n_explain": "n_explain (scale breadth)", "list_parts": "parts",
                   "pattern_mode": "splits (pattern mode)"}
STR_CMAP = "viridis"


# I Helper Functions
def _check_df_eval(df_eval=None, score_col=None):
    """Validate the eval table and that the score column is present and numeric."""
    if not isinstance(df_eval, pd.DataFrame):
        raise ValueError(f"'df_eval' ({type(df_eval).__name__}) should be a pd.DataFrame.")
    if len(df_eval) == 0:
        raise ValueError("'df_eval' (empty) should have at least one configuration row.")
    if score_col not in df_eval.columns:
        raise ValueError(f"'score_col' ({score_col}) should be a column of df_eval "
                         f"({list(df_eval.columns)}).")
    if not np.issubdtype(df_eval[score_col].dropna().to_numpy().dtype, np.number):
        raise ValueError(f"'df_eval[{score_col!r}]' should be numeric.")
    if df_eval[score_col].dropna().empty:
        raise ValueError(f"'df_eval[{score_col!r}]' (all-NaN) should have at least one scored "
                         f"configuration.")


def _check_dict_refine(dict_refine=None):
    """Validate the optional refinement summary: scores numeric-or-None, *_kept boolean-ish."""
    ut.check_dict(name="dict_refine", val=dict_refine)
    for key in ("base", "simplify", "rfe"):
        val = dict_refine.get(key)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(f"'dict_refine[{key!r}]' ({type(val).__name__}) should be a number "
                             f"or None.")


def _sweep_axes(df_eval):
    """Return the sweep axes that actually vary (>1 unique value), in inner-priority order."""
    axes = []
    for col in LIST_SWEEP_AXES:
        if col in df_eval.columns and df_eval[col].nunique(dropna=False) > 1:
            axes.append(col)
    return axes


def _display_order(df_eval, axes):
    """Order axes for the heatmap/facets: ``n_filter`` on x (if swept), then by cardinality.

    The two leading axes become the inner heatmap (x, y); any remaining axes are faceted. Putting
    the highest-cardinality axes inner minimizes the number of facet panels (never a silent
    collapse — just a denser, more legible grid).
    """
    by_card = sorted(axes, key=lambda a: df_eval[a].nunique(dropna=False), reverse=True)
    if "n_filter" in axes:
        return ["n_filter"] + [a for a in by_card if a != "n_filter"]
    return by_card


def _axis_levels(df_eval, col):
    """Sorted unique levels of an axis (numeric-aware; ``None`` sorts last as 'all')."""
    vals = list(df_eval[col].unique())
    if col in LIST_NUMERIC_AXES:
        # Numeric axis may carry a None (e.g. n_explain=None means 'all scales'): order numbers
        # ascending, place None last.
        nums = sorted(v for v in vals if v is not None and not pd.isna(v))
        if any(v is None or pd.isna(v) for v in vals):
            return nums + [None]
        return nums
    return sorted(vals, key=lambda v: ("", "") if v is None else (str(v),))


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


def _heatmap_grid(df_eval, x_axis, y_axis, score_col, x_levels=None, y_levels=None):
    """Mean-score grid over (y_axis x x_axis) plus the ordered level lists.

    Pass ``x_levels`` / ``y_levels`` to pin the axes to a shared (e.g. global, for facets) ordering;
    otherwise they are derived from ``df_eval``. The cell lookup is NaN/None-safe via :func:`_row_mask`
    (sweep tables are small — dozens of configurations — so the per-cell scan is not a hot path).
    """
    x_levels = _axis_levels(df_eval, x_axis) if x_levels is None else x_levels
    y_levels = _axis_levels(df_eval, y_axis) if y_levels is None else y_levels
    grid = np.full((len(y_levels), len(x_levels)), np.nan)
    for yi, yv in enumerate(y_levels):
        for xi, xv in enumerate(x_levels):
            mask = _row_mask(df_eval, {x_axis: xv, y_axis: yv})
            if mask.any():
                grid[yi, xi] = float(df_eval.loc[mask, score_col].mean())
    return grid, x_levels, y_levels


def _row_mask(df_eval, coords):
    """Boolean mask of rows matching every (col -> value) in ``coords`` (NaN/None-safe)."""
    mask = pd.Series(True, index=df_eval.index)
    for col, val in coords.items():
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            mask &= df_eval[col].isna()
        else:
            mask &= (df_eval[col] == val)
    return mask


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
        ax.scatter([star[0]], [star[1]], marker="*", s=220, c="red",
                   edgecolors="white", linewidths=0.8, zorder=5)
    return im


def _refine_str(dict_refine):
    """Render the simplify / RFE refinement as a before -> after Δ-score line (req: Δ annotations).

    ``dict_refine`` carries the cross-validated score before refinement and after each kept step:
    ``{"base": float, "simplify": float|None, "simplify_kept": bool, "rfe": float|None,
    "rfe_kept": bool}``. Steps that were skipped or did not improve are reported as no-ops. Returned
    as a string so it can be folded into the suptitle (laid out by ``constrained_layout``).
    """
    base = dict_refine.get("base")
    parts = [f"refine:  base={base:.3f}" if base is not None else "refine:  base=n/a"]
    # Δ is measured against the *running* base — the score going into each step — so a no-op RFE
    # after a kept simplify reads Δ+0.000, not the (already-counted) simplify gain.
    running = base
    for step in ("simplify", "rfe"):
        score = dict_refine.get(step)
        kept = dict_refine.get(f"{step}_kept", False)
        if score is None:
            parts.append(f"{step}: —")
            continue
        delta = score - running if running is not None else float("nan")
        tag = "kept" if kept else "no-op"
        parts.append(f"{step}: {score:.3f} (Δ{delta:+.3f}, {tag})")
        if kept:
            running = score
    return "   |   ".join(parts)


# II Main Functions
def plot_eval(df_eval: pd.DataFrame,
              score_col: str = COL_SCORE,
              std_col: Optional[str] = COL_STD,
              figsize: Optional[Tuple[float, float]] = None,
              title: Optional[str] = None,
              dict_refine: Optional[dict] = None,
              ) -> Optional[Figure]:
    """
    Plot a CPP optimization sweep as a ``viridis`` evaluation grid that adapts to its dimensionality.

    Visualizes the per-configuration cross-validated scores of a :func:`find_features` sweep (its
    ``df_eval``) so the optimization is no longer a black box: which configuration won, and how
    sensitive the score is to each swept lever. The plot form follows the number of axes that
    actually vary in ``df_eval`` (``list_parts``, the split ``pattern_mode``, ``n_split_max``,
    ``n_explain``, ``n_filter``): **0** varying axes (a single configuration, e.g.
    ``optimization="fast"``) draws nothing and returns ``None``; **1** axis draws a line (numeric
    lever) or bar (categorical lever); **2** axes draw a single heatmap; **3 or more** axes draw
    faceted small-multiples — a grid of 2-D heatmaps, one panel per level of the extra axes — never
    collapsing a higher-dimensional sweep silently. Color encodes the cross-validated score
    (the selection metric) and the selected configuration is marked with a star.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Per-configuration sweep table (one row per configuration) such as the third return value of
        :func:`find_features`, or any ``CPPGrid``-style evaluation table. Must contain ``score_col``
        and the swept descriptor columns; an ``is_selected`` boolean column (if present) marks the
        best configuration, otherwise the maximum ``score_col`` is used.
    score_col : str, default="cv_bacc_mean"
        Name of the cross-validated score column to color by (the selection metric).
    std_col : str, optional
        Name of the score standard-deviation column, drawn as error bars on the 1-D line plot.
        Ignored (and may be ``None``) for the heatmap and faceted forms.
    figsize : tuple of float, optional
        Figure size in inches. If ``None``, a size is derived from the layout.
    title : str, optional
        Figure suptitle. If ``None``, a default naming the swept axes is used.
    dict_refine : dict, optional
        Optional simplify / RFE refinement summary annotated as before -> after Δ-score text.
        Recognized keys: ``base``, ``simplify``, ``simplify_kept``, ``rfe``, ``rfe_kept``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The evaluation-grid figure, or ``None`` when no axis varies (a single configuration, so
        there is nothing to compare).

    See Also
    --------
    * :func:`find_features` : the CPP AutoML pipeline whose ``df_eval`` this visualizes; it draws
      this grid as an auxiliary figure when ``plot=True``.
    * :class:`aaanalysis.CPPGrid` : the configuration sweep underlying ``df_eval``.

    Examples
    --------
    .. include:: examples/aap_plot_eval.rst
    """
    # Validate
    _check_df_eval(df_eval=df_eval, score_col=score_col)
    if std_col is not None and not isinstance(std_col, str):
        raise ValueError(f"'std_col' ({type(std_col).__name__}) should be a str or None.")
    if title is not None:
        ut.check_str(name="title", val=title)
    if dict_refine is not None:
        _check_dict_refine(dict_refine=dict_refine)

    # A fresh integer index keeps the best-row scalar lookups well-defined even for a standalone
    # table whose index is non-unique (e.g. concatenated sweeps).
    df_eval = df_eval.reset_index(drop=True)
    axes = _sweep_axes(df_eval)
    if len(axes) == 0:
        return None
    best_idx = _best_row(df_eval, score_col)
    # One global color scale (over the raw scores) shared by every layout, so the same score maps
    # to the same viridis color whether the sweep renders as a line, a heatmap, or facets.
    vmin, vmax = float(df_eval[score_col].min()), float(df_eval[score_col].max())
    suptitle = title if title is not None else (
        "CPP sweep evaluation — " + " x ".join(DICT_AXIS_LABEL.get(a, a) for a in axes))
    if dict_refine is not None:
        suptitle = f"{suptitle}\n{_refine_str(dict_refine)}"

    if len(axes) == 1:
        fig = _plot_1d(df_eval, axes[0], score_col, std_col, best_idx, figsize, vmin, vmax)
    elif len(axes) == 2:
        fig = _plot_heatmap(df_eval, _display_order(df_eval, axes), score_col, best_idx,
                            figsize, vmin, vmax)
    else:
        fig = _plot_facets(df_eval, _display_order(df_eval, axes), score_col, best_idx,
                           figsize, vmin, vmax)
    fig.suptitle(suptitle)
    return fig


def _plot_1d(df_eval, axis, score_col, std_col, best_idx, figsize, vmin, vmax):
    """1-D sweep: a line (numeric lever) or bar (categorical lever) colored by the score."""
    fig, ax = plt.subplots(figsize=figsize or (6.0, 4.0), constrained_layout=True)
    levels = _axis_levels(df_eval, axis)
    means = [float(df_eval.loc[_row_mask(df_eval, {axis: v}), score_col].mean()) for v in levels]
    cmap = plt.get_cmap(STR_CMAP)
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
    colors = [cmap(norm(m)) for m in means]
    x = np.arange(len(levels))
    best_val = df_eval.loc[best_idx, axis]
    best_pos = next((i for i, v in enumerate(levels)
                     if (pd.isna(v) and pd.isna(best_val)) or v == best_val), None)
    if axis in LIST_NUMERIC_AXES and not any(v is None or pd.isna(v) for v in levels):
        ax.plot(x, means, color="0.6", zorder=1)
        if std_col is not None and std_col in df_eval.columns:
            stds = [float(df_eval.loc[_row_mask(df_eval, {axis: v}), std_col].mean()) for v in levels]
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="0.7", capsize=3, zorder=1)
        ax.scatter(x, means, c=colors, s=90, edgecolors="0.3", linewidths=0.6, zorder=2)
    else:
        ax.bar(x, means, color=colors, edgecolor="0.3", linewidth=0.6)
    if best_pos is not None:
        ax.scatter([best_pos], [means[best_pos]], marker="*", s=260, c="red",
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_level_label(axis, v) for v in levels], rotation=45, ha="right")
    ax.set_xlabel(DICT_AXIS_LABEL.get(axis, axis))
    ax.set_ylabel(score_col)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=score_col)
    return fig


def _plot_heatmap(df_eval, axes, score_col, best_idx, figsize, vmin, vmax):
    """2-D sweep: a single viridis heatmap (axes[0] on x, axes[1] on y)."""
    x_axis, y_axis = axes[0], axes[1]
    grid, x_levels, y_levels = _heatmap_grid(df_eval, x_axis, y_axis, score_col)
    fig, ax = plt.subplots(figsize=figsize or (1.4 * len(x_levels) + 3, 1.0 * len(y_levels) + 3),
                           constrained_layout=True)
    star = _star_cell(df_eval, best_idx, x_axis, y_axis, x_levels, y_levels)
    im = _draw_heatmap(ax, grid, x_levels, y_levels, x_axis, y_axis, vmin, vmax, star=star)
    fig.colorbar(im, ax=ax, label=score_col)
    return fig


def _plot_facets(df_eval, axes, score_col, best_idx, figsize, vmin, vmax):
    """3+ -D sweep: faceted heatmaps — inner (axes[0] x axes[1]), one panel per extra-axis combo."""
    x_axis, y_axis = axes[0], axes[1]
    facet_axes = axes[2:]
    facet_levels = [_axis_levels(df_eval, a) for a in facet_axes]
    combos = list(_product(facet_levels))
    ncols = min(len(combos), max(1, math.ceil(math.sqrt(len(combos)))))
    nrows = math.ceil(len(combos) / ncols)
    x_levels_all = _axis_levels(df_eval, x_axis)
    y_levels_all = _axis_levels(df_eval, y_axis)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                            figsize=figsize or (ncols * (1.0 * len(x_levels_all) + 2.5),
                                                nrows * (0.7 * len(y_levels_all) + 2.5)))
    im = None
    for k, combo in enumerate(combos):
        ax = axs[k // ncols][k % ncols]
        coords = dict(zip(facet_axes, combo))
        sub = df_eval[_row_mask(df_eval, coords)]
        # Pin every panel to the global x/y levels so the small-multiples share one aligned grid.
        grid, _, _ = _heatmap_grid(sub, x_axis, y_axis, score_col,
                                   x_levels=x_levels_all, y_levels=y_levels_all)
        star = None
        if best_idx in sub.index:
            star = _star_cell(df_eval, best_idx, x_axis, y_axis, x_levels_all, y_levels_all)
        im = _draw_heatmap(ax, grid, x_levels_all, y_levels_all, x_axis, y_axis, vmin, vmax, star=star)
        ax.set_title(", ".join(f"{DICT_AXIS_LABEL.get(a, a)}={_level_label(a, v)}"
                               for a, v in zip(facet_axes, combo)), fontsize=9)
    for k in range(len(combos), nrows * ncols):
        axs[k // ncols][k % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axs, label=score_col, shrink=0.8)
    return fig


def _star_cell(df_eval, best_idx, x_axis, y_axis, x_levels, y_levels):
    """(xi, yi) cell index of the best configuration, or None if off this panel's grid."""
    bx, by = df_eval.loc[best_idx, x_axis], df_eval.loc[best_idx, y_axis]
    xi = next((i for i, v in enumerate(x_levels)
               if (pd.isna(v) and pd.isna(bx)) or v == bx), None)
    yi = next((i for i, v in enumerate(y_levels)
               if (pd.isna(v) and pd.isna(by)) or v == by), None)
    if xi is None or yi is None:
        return None
    return (xi, yi)


def _product(list_levels):
    """Cartesian product of the per-axis level lists (empty -> a single empty combo)."""
    combos = [()]
    for levels in list_levels:
        combos = [c + (v,) for c in combos for v in levels]
    return combos
