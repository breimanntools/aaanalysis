"""
Backend for AAPredPlot.eval(kind="comparison"): a grouped method x condition comparison barplot
with per-bar value labels and a chance/baseline line.
"""
from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import aaanalysis.utils as ut
from .aa_pred_plot_legend import place_legend_below_


# I Helper Functions
def _resolve_order(values: List, order: Optional[List], name: str) -> List:
    """First-appearance order by default; otherwise validate the order covers all values."""
    seen = list(dict.fromkeys(values))
    if order is None:
        return seen
    ut.check_list_like(name=name, val=order)
    seen_set = set(seen)
    missing = seen_set - set(order)
    if missing:
        raise ValueError(f"'{name}' is missing values present in the data: {sorted(map(str, missing))}")
    return [g for g in dict.fromkeys(order) if g in seen_set]


def _resolve_colors(group_order: List, colors: Optional[Union[List, Dict]]) -> Dict:
    """Build a {group: color} map: explicit list/dict wins, else the house categorical palette."""
    n = len(group_order)
    if colors is None:
        palette = ut.plot_get_clist_(n_colors=max(n, 2))
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
    """Pick a value-label format from the data scale."""
    vals = values[~np.isnan(values)]
    if vals.size == 0 or np.allclose(vals, np.round(vals)):
        return "{:.0f}"
    if float(np.max(np.abs(vals))) < 10:
        return "{:.2f}"
    return "{:.1f}"


# II Main Functions
def plot_comparison_(df_eval=None, group="group", condition="condition", value="value",
                     baseline=50, baseline_label=None, annotate=True, annotation_fmt=None,
                     group_order=None, condition_order=None, dict_color=None, bar_width=0.8,
                     ax=None, figsize=(7, 4.2), xlabel=None, ylabel="Score", title=None,
                     ylim=None, fontsize_annotations=10, xtick_rotation=0, legend_title=None):
    """Draw the grouped comparison barplot. Returns (fig, ax)."""
    group_order = _resolve_order(df_eval[group].tolist(), group_order, "group_order")
    condition_order = _resolve_order(df_eval[condition].tolist(), condition_order, "condition_order")
    dict_group_color = _resolve_colors(group_order, dict_color)
    grid = (df_eval.groupby([group, condition])[value].mean()
            .unstack(condition)
            .reindex(index=group_order, columns=condition_order))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    n_groups = len(group_order)
    x = np.arange(len(condition_order))
    each_w = bar_width / n_groups
    all_nan = grid.size == 0 or bool(np.all(np.isnan(grid.values)))
    heights_max = 0.0 if all_nan else float(np.nanmax(grid.values))
    label_pad = 0.01 * max(heights_max, 1)
    if annotation_fmt is None:
        annotation_fmt = _auto_annotation_fmt(grid.values)
    for j, g in enumerate(group_order):
        offset = (j - (n_groups - 1) / 2) * each_w
        heights = grid.loc[g].to_numpy(dtype=float)
        bars = ax.bar(x + offset, heights, each_w, label=str(g), color=dict_group_color[g])
        if annotate:
            for b, h in zip(bars, heights):
                if np.isnan(h):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, h + label_pad, annotation_fmt.format(h),
                        ha="center", va="bottom", fontsize=fontsize_annotations, weight="bold")
    if baseline is not None:
        if baseline_label is None:
            baseline_label = f"chance ({baseline:g})"
        ax.axhline(baseline, ls="--", color="black", lw=1,
                   label=baseline_label if baseline_label != "" else "_nolegend_")
    ax.set_xticks(x)
    if xtick_rotation:
        ax.set_xticklabels(condition_order, rotation=xtick_rotation, ha="right", rotation_mode="anchor")
    else:
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
    place_legend_below_(ax=ax, title=legend_title)
    sns.despine(ax=ax)
    return fig, ax
