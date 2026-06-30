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
from matplotlib.patches import Patch

from aaanalysis import utils as ut
from ._plot_get_clist import plot_get_clist

# Default axis labels for the scatter (rank) mode; used to detect "left at default"
# so the additive ranked-candidates (bar) mode can substitute sensible labels.
_DEFAULT_XLABEL = "Protein rank"
_DEFAULT_YLABEL = "Max score per protein"

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


def _resolve_group_colors(group_order: List[str], dict_color: Optional[Dict[str, str]] = None):
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


def _plot_ranked_candidates(ax, df_rank, col_score, col_class, col_std, group_order,
                            dict_group_color, list_thresholds, xlabel, ylabel,
                            fontsize_labels):
    """Draw the ranked-candidates horizontal-bar variant (port of plot_pred3_top_hits):
    named candidates as horizontal bars colored by class, optional per-item error bars,
    and vertical threshold (cutoff) lines."""
    # Sort by class (in group_order) then ascending score, keep the index as candidate names
    order_map = {c: i for i, c in enumerate(group_order)}
    df = df_rank.copy()
    df["_name"] = df.index.astype(str)
    df["_sorter"] = df[col_class].map(order_map)
    df = df.sort_values(["_sorter", col_score], ascending=[False, True]).reset_index(drop=True)
    y_pos = np.arange(len(df))
    colors = [dict_group_color[c] for c in df[col_class]]
    ax.barh(y_pos, df[col_score].to_numpy(), color=colors)
    if col_std is not None:
        ax.errorbar(x=df[col_score].to_numpy(), y=y_pos, xerr=df[col_std].to_numpy(),
                    fmt="none", ecolor="black", capsize=3, capthick=1, elinewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["_name"].tolist())
    ax.tick_params(length=0, axis="y")
    ax.set_ylim(-0.75, len(df))
    ax.set_xlim(left=0)
    for t in list_thresholds:
        ax.axvline(t, color="grey", linestyle="--", linewidth=1)
    # Substitute bar-appropriate axis labels only when the caller kept the scatter defaults
    xl = "Prediction score" if xlabel == _DEFAULT_XLABEL else xlabel
    yl = "" if ylabel == _DEFAULT_YLABEL else ylabel
    ax.set_xlabel(xl, fontsize=fontsize_labels)
    ax.set_ylabel(yl, fontsize=fontsize_labels)
    present = set(df[col_class])
    handles = [Patch(color=dict_group_color[g], label=str(g))
               for g in group_order if g in present]
    ax.legend(handles=handles)
    sns.despine(ax=ax)


# II Main Functions
def plot_rank(df_rank: pd.DataFrame,
              col_score: str = "score",
              col_group: str = "group",
              group_order: Optional[List[str]] = None,
              dict_color: Optional[Dict[str, str]] = None,
              threshold: Optional[Union[int, float, List[Union[int, float]]]] = None,
              col_std: Optional[str] = None,
              col_class: Optional[str] = None,
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

    Passing ``col_class`` switches to an additive **ranked-candidates** variant: named
    candidates (the DataFrame index) are drawn as horizontal bars colored by class, with
    optional per-item error bars (``col_std``) and vertical cutoff lines (``threshold``).
    This reproduces the recurring "top-hits with agreement" figure. The default scatter
    output is unchanged when ``col_class`` is ``None``.

    .. versionadded:: 1.1.0

    .. versionchanged:: 1.1.0
       Added the additive ``col_std`` / ``col_class`` ranked-candidates (error-bar) mode.

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
        One or more cutoff values drawn as threshold lines (horizontal in scatter mode,
        vertical in ranked-candidates mode).
    col_std : str, optional
        Column with a per-item standard deviation. When given (only valid together with
        ``col_class``), symmetric horizontal error bars are drawn on the candidate bars.
    col_class : str, optional
        Column with the per-item class label. When given, the plot switches to the
        ranked-candidates horizontal-bar variant (bars colored by class, candidate names
        taken from the DataFrame index); ``col_group`` is then unused.
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
    ut.check_str(name="col_std", val=col_std, accept_none=True)
    ut.check_str(name="col_class", val=col_class, accept_none=True)
    if col_std is not None and col_class is None:
        raise ValueError("'col_std' (error bars) requires 'col_class' (ranked-candidates mode).")
    # Column that carries the class/group labels (col_group for scatter, col_class for bars)
    col_groups = col_group if col_class is None else col_class
    cols_required = [col_score, col_groups]
    if col_std is not None:
        cols_required.append(col_std)
    ut.check_df(name="df_rank", df=df_rank, cols_required=cols_required)
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
        group_order = list(dict.fromkeys(df_rank[col_groups].tolist()))
    else:
        missing = set(df_rank[col_groups]) - set(group_order)
        if missing:
            raise ValueError(f"'group_order' is missing groups present in 'df_rank': {missing}")
    dict_group_color = _resolve_group_colors(group_order=group_order, dict_color=dict_color)

    # Create / reuse axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if col_class is not None:
        # Additive ranked-candidates (horizontal-bar) mode
        _plot_ranked_candidates(ax=ax, df_rank=df_rank, col_score=col_score, col_class=col_class,
                                col_std=col_std, group_order=group_order,
                                dict_group_color=dict_group_color, list_thresholds=list_thresholds,
                                xlabel=xlabel, ylabel=ylabel, fontsize_labels=fontsize_labels)
        # ``ax.figure`` is typed ``Figure | SubFigure | None`` by the matplotlib stubs,
        # but a top-level Axes here always belongs to a real Figure.
        return fig, ax  # pyright: ignore[reportReturnType]

    # Default per-protein rank scatter (descending score -> rank 1..N on the x-axis)
    df = df_rank.sort_values(col_score, ascending=False).reset_index(drop=True)
    df["_rank"] = np.arange(1, len(df) + 1)
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
    # ``ax.figure`` is typed ``Figure | SubFigure | None`` by the matplotlib stubs,
    # but a top-level Axes here always belongs to a real Figure.
    return fig, ax  # pyright: ignore[reportReturnType]
