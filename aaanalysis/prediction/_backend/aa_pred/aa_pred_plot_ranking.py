"""
Backend for AAPredPlot.predict(kind="ranking"): horizontal ranked-candidate bars colored by class,
with optional per-item error bars and confidence cut-off lines.
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import aaanalysis.utils as ut
from .aa_pred_plot_legend import place_legend_below_


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


def _cutoff_labels(thresholds):
    """Text label for each cut-off line: the usual two-threshold case is the low/high-confidence
    convention (lower value = ``"LC cut-off"``, higher = ``"HC cut-off"``); other counts unlabeled."""
    if len(thresholds) == 2:
        labels = [None, None]
        lo, hi = (0, 1) if thresholds[0] <= thresholds[1] else (1, 0)
        labels[lo], labels[hi] = "LC cut-off", "HC cut-off"
        return labels
    return [None] * len(thresholds)


# II Main Functions
def plot_ranking_(df_pred=None, col_name="name", col_score="score", col_group=None,
                  col_std=None, dict_color=None, thresholds=(50, 80), top_n=None, ascending=False,
                  ax=None, figsize=None, xlabel="Prediction score", title=None,
                  sort="score", group_order=None, legend_title=None, draw_legend=True):
    """Draw the ranked-candidate horizontal bar plot. Returns (fig, ax).

    ``draw_legend=False`` suppresses this panel's own legend so a multi-panel caller can place a
    single combined legend below the whole figure."""
    if col_group is not None and sort == "group":
        # Cluster bars by group (following `group_order`), ranked by score within each group.
        order = list(group_order) if group_order is not None else list(dict.fromkeys(df_pred[col_group].tolist()))
        grank = {g: i for i, g in enumerate(order)}
        d = df_pred.assign(_grank=df_pred[col_group].map(lambda g: grank.get(g, len(order))))
        d = d.sort_values(["_grank", col_score], ascending=[True, ascending])
        d = d.drop(columns="_grank").reset_index(drop=True)
    else:
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
        dict_color = _resolve_group_colors(d[col_group].tolist(), dict_color)
        bar_colors = [dict_color[g] for g in d[col_group]]
    else:
        bar_colors = ut.plot_get_clist_(n_colors=2)[0]
    xerr = d[col_std].to_numpy(dtype=float) if col_std is not None else None
    ax.barh(y, d[col_score].to_numpy(dtype=float), color=bar_colors,
            xerr=xerr, capsize=2.5, error_kw=dict(lw=1))
    ax.set_yticks(y)
    ax.set_yticklabels(d[col_name].astype(str).tolist(), fontsize=9)
    ax.tick_params(axis="y", length=0)  # no y-tick marks; labels sit next to the bars
    ths = list(thresholds or [])
    cut_labels = _cutoff_labels(ths)
    for c, lab in zip(ths, cut_labels):
        ax.axvline(c, ls="--", color="0.3", lw=1.2)
        if lab is not None:  # cut-off label above the line (e.g. "LC cut-off" / "HC cut-off")
            ax.text(c, 1.005, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, clip_on=False)
    ax.set_xlabel(xlabel)
    if title is not None:  # extra pad so the title clears the cut-off labels
        ax.set_title(title, pad=22 if any(cut_labels) else None)
    present = list(dict.fromkeys(d[col_group].tolist())) if col_group is not None else []
    if draw_legend and len(present) >= 2:
        # Legend shows only the groups drawn in this panel, in their draw order (so a per-group
        # panel does not advertise the whole palette); a single-group panel needs no legend.
        handles = [plt.Rectangle((0, 0), 1, 1, color=dict_color[g]) for g in present]
        place_legend_below_(ax=ax, handles=handles, labels=present, title=legend_title, fontsize=9)
    ax.margins(y=0.01)
    ax.set_xlim(left=0)  # bars start flush at the y-axis (drop the default left margin/gap)
    sns.despine(ax=ax)
    return fig, ax
