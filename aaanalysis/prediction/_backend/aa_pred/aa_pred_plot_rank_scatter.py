"""
Backend for AAPredPlot.predict_group(kind="rank_scatter"): a per-protein rank scatter
(max-score-per-protein sorted by score, colored by group, with optional threshold lines).
"""
from typing import Optional, List, Dict
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import aaanalysis.utils as ut
from .aa_pred_plot_legend import place_legend_below_


# I Helper Functions
#: auto_font width sizing for the rank scatter: grow the width with the number of ranked
#: proteins (per-protein inches) so markers do not crowd, floored at the default width and
#: clamped. Height and fonts stay fixed — the auto_font analog for a point plot with no grid.
_RANK_W_PER_PROTEIN_IN = 0.05
_RANK_W_MIN_IN = 7.0
_RANK_W_CAP_IN = 24.0
_RANK_H_IN = 5.0


# Canonical group -> color defaults (overridable via dict_color); leans on the
# locked sample palette so substrate/non-substrate read green/magenta out of the box.
_DEFAULT_GROUP_COLORS = {
    "substrate": ut.COLOR_POS,        # green
    "non-substrate": ut.COLOR_NEG,    # magenta
    "non_substrate": ut.COLOR_NEG,
    "nonsubstrate": ut.COLOR_NEG,
    "hold-out": ut.COLOR_REL_NEG,     # brownish
    "hold_out": ut.COLOR_REL_NEG,
    "holdout": ut.COLOR_REL_NEG,
    "unlabeled": ut.COLOR_UNL,        # gray
}


def _resolve_group_colors(group_order: List[str], dict_color: Optional[Dict[str, str]] = None):
    """Build a {group: color} map: explicit dict_color wins, then canonical defaults,
    then a curated fallback palette for any remaining groups."""
    dict_color = dict(dict_color) if dict_color is not None else {}
    fallback = ut.plot_get_clist_(n_colors=max(len(group_order), 2))
    out, i = {}, 0
    for g in group_order:
        if g in dict_color:
            out[g] = dict_color[g]
        elif str(g).lower() in _DEFAULT_GROUP_COLORS:
            out[g] = _DEFAULT_GROUP_COLORS[str(g).lower()]
        else:
            out[g] = fallback[i % len(fallback)]
            i += 1
    return out


# II Main Functions
def plot_rank_scatter_(df_rank=None, col_score="score", col_group="group", group_order=None,
                       dict_color=None, thresholds=None, ax=None, figsize=None, marker_size=25,
                       xlabel="Protein rank", ylabel="Max score per protein", fontsize_labels=None,
                       legend_title=None):
    """Draw the per-protein rank scatter. Returns (fig, ax)."""
    # Resolve order + colors
    if group_order is None:
        group_order = list(dict.fromkeys(df_rank[col_group].tolist()))
    dict_group_color = _resolve_group_colors(group_order=group_order, dict_color=dict_color)

    # Build the ranking (descending score -> rank 1..N on the x-axis)
    df = df_rank.sort_values(col_score, ascending=False).reset_index(drop=True)
    df["_rank"] = np.arange(1, len(df) + 1)

    # Draw
    if ax is None:
        # Auto-sizing applies only when the caller OMITS figsize (figsize is None). Under
        # auto_font the width grows with the number of ranked proteins so markers do not
        # crowd; height and fonts stay fixed. Off (or explicit figsize) -> the (7, 5) default.
        if figsize is None:
            if ut.check_auto_font():
                width = min(_RANK_W_CAP_IN, max(_RANK_W_MIN_IN, _RANK_W_PER_PROTEIN_IN * len(df)))
                figsize = (width, _RANK_H_IN)
            else:
                figsize = (_RANK_W_MIN_IN, _RANK_H_IN)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for g in group_order:
        sub = df[df[col_group] == g]
        if len(sub) == 0:
            continue
        ax.scatter(sub["_rank"], sub[col_score], s=marker_size,
                   color=dict_group_color[g], label=str(g))
    for t in (thresholds or []):
        ax.axhline(t, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    place_legend_below_(ax=ax, title=legend_title)
    sns.despine(ax=ax)
    return fig, ax
