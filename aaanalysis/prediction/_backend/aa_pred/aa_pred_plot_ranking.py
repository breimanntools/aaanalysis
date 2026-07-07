"""
Backend for AAPredPlot.predict(kind="ranking"): horizontal ranked-candidate bars colored by class,
with optional per-item error bars and confidence cut-off lines.
"""
import numpy as np
from matplotlib import pyplot as plt

import aaanalysis.utils as ut


# I Helper Functions
def ranking_figheight(n_items, per_item_in=0.22, base_in=1.0):
    """Height (inches) for a ranked-bar plot of ``n_items`` bars (notebook rule)."""
    return float(max(base_in, per_item_in * int(n_items or 0) + base_in))


def _resolve_group_colors(groups, colors):
    """Map each distinct group to a color (dict wins; else house palette)."""
    order = list(dict.fromkeys(groups))
    if isinstance(colors, dict):
        missing = [g for g in order if g not in colors]
        if missing:
            raise ValueError(f"'colors' dict is missing colors for groups: {missing}")
        return colors
    palette = ut.plot_get_clist_(n_colors=max(len(order), 2))
    return {g: palette[i] for i, g in enumerate(order)}


# II Main Functions
def plot_ranking_(df_pred=None, col_name="name", col_score="score", col_group=None,
                  col_std=None, colors=None, cutoffs=(50, 80), top_n=None, ascending=False,
                  ax=None, figsize=None, xlabel="Prediction score", title=None):
    """Draw the ranked-candidate horizontal bar plot. Returns (fig, ax)."""
    d = df_pred.sort_values(col_score, ascending=ascending).reset_index(drop=True)
    if top_n is not None:
        d = d.head(top_n)
    n = len(d)
    if figsize is None:
        figsize = (5.0, ranking_figheight(n))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    y = np.arange(n)[::-1]  # highest score on top
    if col_group is not None:
        dict_color = _resolve_group_colors(d[col_group].tolist(), colors)
        bar_colors = [dict_color[g] for g in d[col_group]]
    else:
        bar_colors = ut.plot_get_clist_(n_colors=2)[0]
    xerr = d[col_std].to_numpy(dtype=float) if col_std is not None else None
    ax.barh(y, d[col_score].to_numpy(dtype=float), color=bar_colors,
            xerr=xerr, capsize=2.5, error_kw=dict(lw=1))
    ax.set_yticks(y)
    ax.set_yticklabels(d[col_name].astype(str).tolist(), fontsize=9)
    for c in (cutoffs or []):
        ax.axvline(c, ls="--", color="0.3", lw=1.2)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if col_group is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=dict_color[g]) for g in dict_color]
        ax.legend(handles, list(dict_color), frameon=False, fontsize=9,
                  loc="lower right")
    ax.margins(y=0.01)
    return fig, ax
