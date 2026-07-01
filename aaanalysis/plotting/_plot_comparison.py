"""
This is a script for the frontend of the plot_comparison function — a grouped
method x condition comparison barplot with value labels and a chance/baseline line.
"""
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from aaanalysis import utils as ut
from ._plot_get_clist import plot_get_clist


# I Helper Functions
def _resolve_order(values: List, order: Optional[List], name: str) -> List:
    """First-appearance order by default; otherwise validate the explicit order covers all values."""
    seen = list(dict.fromkeys(values))
    if order is None:
        return seen
    ut.check_list_like(name=name, val=order)
    missing = set(seen) - set(order)
    if missing:
        raise ValueError(f"'{name}' is missing values present in the data: {sorted(map(str, missing))}")
    # Keep only the groups that actually occur, in the requested order.
    return [g for g in order if g in set(seen)]


def _resolve_colors(group_order: List, colors: Optional[Union[List, Dict]]) -> Dict:
    """Build a {group: color} map: explicit list/dict wins, else the house categorical palette."""
    n = len(group_order)
    if colors is None:
        palette = plot_get_clist(n_colors=max(n, 2))
        return {g: palette[i] for i, g in enumerate(group_order)}
    if isinstance(colors, dict):
        missing = [g for g in group_order if g not in colors]
        if missing:
            raise ValueError(f"'colors' dict is missing colors for groups: {missing}")
        return {g: colors[g] for g in group_order}
    ut.check_list_like(name="colors", val=colors)
    if len(colors) < n:
        raise ValueError(f"'colors' (n={len(colors)}) should provide at least one color "
                         f"per group (n_groups={n}).")
    return {g: colors[i] for i, g in enumerate(group_order)}


def _auto_annotation_fmt(values: np.ndarray) -> str:
    """Pick a value-label format: no decimals for integers, else 2 decimals for small
    (fractional, e.g. AUC in [0, 1]) values and 1 decimal for larger scales."""
    vals = values[~np.isnan(values)]
    if vals.size == 0 or np.allclose(vals, np.round(vals)):
        return "{:.0f}"
    if float(np.max(np.abs(vals))) < 10:
        return "{:.2f}"
    return "{:.1f}"


# II Main Functions
def plot_comparison(df_eval: pd.DataFrame,
                    group: str = "group",
                    condition: str = "condition",
                    value: str = "value",
                    baseline: Optional[Union[int, float]] = 50,
                    baseline_label: Optional[str] = None,
                    annotate: bool = True,
                    annotation_fmt: Optional[str] = None,
                    group_order: Optional[List[str]] = None,
                    condition_order: Optional[List[str]] = None,
                    colors: Optional[Union[List[str], Dict[str, str]]] = None,
                    bar_width: Union[int, float] = 0.8,
                    ax: Optional[Axes] = None,
                    figsize: Tuple[Union[int, float], Union[int, float]] = (7, 4.2),
                    xlabel: Optional[str] = None,
                    ylabel: str = "Score",
                    title: Optional[str] = None,
                    ylim: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                    fontsize_annotations: Union[int, float] = 10,
                    ) -> Tuple[Figure, Axes]:
    """
    Plot a grouped method x condition comparison barplot with value labels and a baseline line.

    Draws the recurring "benchmark result" figure from a tidy eval frame in one call:
    each ``condition`` is a cluster on the x-axis, each ``group`` a colored bar within it
    (auto offsets / widths for *N* groups), with optional per-bar value labels and an
    optional dashed chance / baseline line. Replaces the manual ``x ± w/2`` offsets, the
    per-bar ``ax.text`` annotation loop, and the manual ``axhline`` of a hand-built grouped
    barplot.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    df_eval : pd.DataFrame, shape (n_rows, n_cols)
        Tidy (long-form) evaluation frame with one row per (``group``, ``condition``)
        cell; must contain the ``group``, ``condition``, and ``value`` columns. Repeated
        (``group``, ``condition``) rows are averaged.
    group : str, default="group"
        Column whose distinct values become the colored bars within each cluster (the legend).
    condition : str, default="condition"
        Column whose distinct values become the x-axis clusters.
    value : str, default="value"
        Column with the numeric bar heights (e.g. balanced accuracy in percent).
    baseline : int or float, optional
        y-value of a dashed horizontal chance / baseline line. If ``None``, no line is drawn.
    baseline_label : str, optional
        Text label for the baseline line. If ``None`` and ``baseline`` is set, a label
        ``"chance (<baseline>)"`` is generated; pass ``""`` to draw the line without a label.
    annotate : bool, default=True
        If ``True``, write each bar's value above it.
    annotation_fmt : str, optional
        Python format string for the value labels (e.g. ``"{:.1f}"``). If ``None``, it is
        chosen from the data: no decimals for integer-valued scores, two decimals for
        fractional metrics (e.g. AUC in ``[0, 1]``), one decimal otherwise.
    group_order : list of str, optional
        Order of the groups (bar order within each cluster). Defaults to first-appearance order.
    condition_order : list of str, optional
        Order of the conditions (x-axis clusters). Defaults to first-appearance order.
    colors : list of str or dict, optional
        Bar colors: a list aligned to ``group_order``, or a ``group -> color`` dict.
        Defaults to the house categorical palette (:func:`plot_get_clist`).
    bar_width : int or float, default=0.8
        Total width of each condition cluster (split evenly across the groups). Must be in (0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    figsize : tuple, default=(7, 4.2)
        Figure size when ``ax`` is ``None``.
    xlabel : str, optional
        x-axis label (none by default; the cluster tick labels carry the conditions).
    ylabel : str, default="Score"
        y-axis label.
    title : str, optional
        Axes title.
    ylim : tuple, optional
        y-axis limits ``(bottom, top)``. If ``None``, the top is extended above the tallest
        bar / baseline to leave room for the value labels.
    fontsize_annotations : int or float, default=10
        Font size of the per-bar value labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes with the grouped comparison barplot.

    See Also
    --------
    * :func:`aaanalysis.plot_get_clist` for the house categorical palette.
    * :func:`aaanalysis.plot_rank` for a per-protein rank scatter of a deployed predictor.
    * :func:`aaanalysis.pipe.plot_eval` for a viridis evaluation-grid view of a wider sweep.

    Examples
    --------
    .. include:: examples/plot_comparison.rst
    """
    # Check input
    ut.check_str(name="group", val=group)
    ut.check_str(name="condition", val=condition)
    ut.check_str(name="value", val=value)
    if len({group, condition, value}) < 3:
        raise ValueError(f"'group', 'condition', and 'value' should be three distinct columns, "
                         f"got group={group!r}, condition={condition!r}, value={value!r}.")
    ut.check_df(name="df_eval", df=df_eval, cols_required=[group, condition, value])
    if len(df_eval) == 0:
        raise ValueError("'df_eval' (0 rows) should contain at least one row.")
    ut.check_number_val(name="baseline", val=baseline, accept_none=True, just_int=False)
    ut.check_str(name="baseline_label", val=baseline_label, accept_none=True)
    ut.check_bool(name="annotate", val=annotate)
    ut.check_str(name="annotation_fmt", val=annotation_fmt, accept_none=True)
    ut.check_list_like(name="group_order", val=group_order, accept_none=True)
    ut.check_list_like(name="condition_order", val=condition_order, accept_none=True)
    ut.check_number_range(name="bar_width", val=bar_width, min_val=0, max_val=1, just_int=False)
    if bar_width == 0:
        raise ValueError("'bar_width' should be greater than 0.")
    ut.check_ax(ax=ax, accept_none=True)
    ut.check_figsize(figsize=figsize, accept_none=True)
    ut.check_str(name="xlabel", val=xlabel, accept_none=True)
    ut.check_str(name="ylabel", val=ylabel, accept_none=True)
    ut.check_str(name="title", val=title, accept_none=True)
    ut.check_number_range(name="fontsize_annotations", val=fontsize_annotations, min_val=0,
                          just_int=False)
    if ylim is not None:
        ut.check_lim(name="ylim", val=ylim)

    # Resolve orders + colors
    group_order = _resolve_order(df_eval[group].tolist(), group_order, "group_order")
    condition_order = _resolve_order(df_eval[condition].tolist(), condition_order, "condition_order")
    dict_group_color = _resolve_colors(group_order, colors)

    # Build the (group x condition) value grid (mean over any repeated cells)
    grid = (df_eval.groupby([group, condition])[value].mean()
            .unstack(condition)
            .reindex(index=group_order, columns=condition_order))

    # Draw
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    n_groups = len(group_order)
    n_cond = len(condition_order)
    x = np.arange(n_cond)
    each_w = bar_width / n_groups
    heights_max = float(np.nanmax(grid.values)) if grid.size else 0.0
    if annotation_fmt is None:
        annotation_fmt = _auto_annotation_fmt(grid.values)
    for j, g in enumerate(group_order):
        offset = (j - (n_groups - 1) / 2) * each_w
        heights = grid.loc[g].values.astype(float)
        bars = ax.bar(x + offset, heights, each_w, label=str(g), color=dict_group_color[g])
        if annotate:
            for b, h in zip(bars, heights):
                if np.isnan(h):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01 * max(heights_max, 1),
                        annotation_fmt.format(h), ha="center", va="bottom",
                        fontsize=fontsize_annotations, weight="bold")

    # Baseline / chance line
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="black", lw=1)
        if baseline_label is None:
            baseline_label = f"chance ({baseline:g})"
        if baseline_label != "":
            ax.text(n_cond - 1 + 0.5 * bar_width, baseline + 0.01 * max(heights_max, 1),
                    baseline_label, ha="right", va="bottom",
                    fontsize=max(fontsize_annotations - 1, 1))

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(condition_order)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif heights_max > 0:
        top = heights_max if baseline is None else max(heights_max, baseline)
        ax.set_ylim(top=top * (1.15 if annotate else 1.05))
    # Legend outside the axes (upper right): a grouped barplot fills the plot area, so an
    # inside legend would overlap the tallest bars / their value labels.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    sns.despine(ax=ax)
    # ``ax.figure`` is typed ``Figure | SubFigure | None`` by the matplotlib stubs,
    # but a top-level Axes here always belongs to a real Figure.
    return fig, ax  # pyright: ignore[reportReturnType]
