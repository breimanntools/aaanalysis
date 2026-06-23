"""
This is a script for the frontend of the plot_rank function — a per-protein rank scatter.
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


def _resolve_group_colors(group_order=None, dict_color=None):
    """Build a {group: color} map: explicit dict_color wins, then canonical defaults,
    then a curated fallback palette for any remaining groups."""
    dict_color = dict(dict_color) if dict_color is not None else {}
    fallback = plot_get_clist(n_colors=max(len(group_order), 2))
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
def plot_rank(df_rank: Optional[pd.DataFrame] = None,
              col_score: str = "score",
              col_group: str = "group",
              group_order: Optional[List[str]] = None,
              dict_color: Optional[Dict[str, str]] = None,
              threshold: Optional[Union[int, float, List[Union[int, float]]]] = None,
              ax: Optional[Axes] = None,
              figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
              marker_size: Union[int, float] = 25,
              xlabel: str = "Protein rank",
              ylabel: str = "Max score per protein",
              fontsize_labels: Optional[Union[int, float]] = None,
              ) -> Tuple[Figure, Axes]:
    """
    Plot a per-protein rank scatter: max-score-per-protein sorted by score, colored by group.

    The single most useful sanity check for a deployed per-protein predictor — proteins are
    ranked by their maximum score and colored by membership in groups such as substrate /
    hold-out / non-substrate, with optional threshold lines for the deployment caller.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    df_rank : pd.DataFrame, shape (n_proteins, n_info)
        One row per protein; must contain ``col_score`` (the protein's max score) and
        ``col_group`` (its group label).
    col_score : str, default="score"
        Column with the per-protein score used for ranking (descending) on the y-axis.
    col_group : str, default="group"
        Column with the per-protein group label used for coloring.
    group_order : list of str, optional
        Order in which groups are colored / drawn. Defaults to first-appearance order.
    dict_color : dict, optional
        Mapping ``group -> color`` (overrides the canonical defaults). Canonical group names
        (``substrate``, ``non-substrate``, ``hold-out``) default to the locked sample palette.
    threshold : int, float, or list, optional
        One or more y-values drawn as horizontal threshold lines.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    figsize : tuple, default=(7, 5)
        Figure size when ``ax`` is ``None``.
    marker_size : int or float, default=25
        Scatter marker size.
    xlabel, ylabel : str
        Axis labels.
    fontsize_labels : int or float, optional
        Font size for the axis labels (matplotlib default if ``None``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes with the rank scatter.

    See Also
    --------
    * :func:`aaanalysis.plot_get_clist` for the fallback color palette.
    * :func:`aaanalysis.comp_per_protein_ap` / :func:`aaanalysis.comp_detection_metrics`
      for the numeric companions to this visual check.

    Examples
    --------
    .. include:: examples/plot_rank.rst
    """
    # Check input
    ut.check_str(name="col_score", val=col_score)
    ut.check_str(name="col_group", val=col_group)
    ut.check_df(name="df_rank", df=df_rank, cols_required=[col_score, col_group])
    if len(df_rank) == 0:
        raise ValueError("'df_rank' (0 rows) should contain at least one protein.")
    ut.check_list_like(name="group_order", val=group_order, accept_none=True)
    ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
    ut.check_ax(ax=ax, accept_none=True)
    ut.check_figsize(figsize=figsize, accept_none=True)
    ut.check_number_range(name="marker_size", val=marker_size, min_val=0, just_int=False)
    ut.check_str(name="xlabel", val=xlabel, accept_none=True)
    ut.check_str(name="ylabel", val=ylabel, accept_none=True)
    ut.check_number_range(name="fontsize_labels", val=fontsize_labels, min_val=0,
                          accept_none=True, just_int=False)
    list_thresholds = []
    if threshold is not None:
        list_thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else [threshold]
        for i, t in enumerate(list_thresholds):
            ut.check_number_val(name=f"threshold[{i}]", val=t, just_int=False)

    # Resolve order + colors
    if group_order is None:
        group_order = list(dict.fromkeys(df_rank[col_group].tolist()))
    else:
        missing = set(df_rank[col_group]) - set(group_order)
        if missing:
            raise ValueError(f"'group_order' is missing groups present in 'df_rank': {missing}")
    dict_group_color = _resolve_group_colors(group_order=group_order, dict_color=dict_color)

    # Build the ranking (descending score -> rank 1..N on the x-axis)
    df = df_rank.sort_values(col_score, ascending=False).reset_index(drop=True)
    df["_rank"] = np.arange(1, len(df) + 1)

    # Draw
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for g in group_order:
        sub = df[df[col_group] == g]
        if len(sub) == 0:
            continue
        ax.scatter(sub["_rank"], sub[col_score], s=marker_size,
                   color=dict_group_color[g], label=str(g))
    for t in list_thresholds:
        ax.axhline(t, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax.legend()
    sns.despine(ax=ax)
    return fig, ax
