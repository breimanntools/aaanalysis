"""
This is a script for the frontend of the (internal) plot_prediction_hist function — a
class-separated histogram of a 0-100 model prediction score.

This symbol is **not** re-exported in ``aaanalysis/__init__.py`` yet; it is reached via the
submodule import ``from aaanalysis.plotting._plot_prediction_hist import plot_prediction_hist``.
"""
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from aaanalysis import utils as ut
from ._plot_rank import _resolve_group_colors

# TODO(#305): re-export plot_prediction_hist in __init__ (CONFIRM-FIRST, maintainer review)


# I Helper Functions
def _check_binrange(name="binrange", val=None):
    """Validate a 2-element numeric (low, high) range with low < high."""
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(f"'{name}' ({val}) should be a (low, high) tuple of two numbers.")
    low, high = val
    for v in (low, high):
        ut.check_number_val(name=name, val=v, just_int=False)
    if not low < high:
        raise ValueError(f"'{name}' ({val}) should satisfy low < high.")


# II Main Functions
def plot_prediction_hist(df_pred: pd.DataFrame,
                         score: str = "score",
                         group: str = "group",
                         group_order: Optional[List[str]] = None,
                         dict_color: Optional[Dict[str, str]] = None,
                         binwidth: Union[int, float] = 5,
                         binrange: Tuple[Union[int, float], Union[int, float]] = (0, 100),
                         stacked: bool = True,
                         kde: bool = False,
                         ax: Optional[Axes] = None,
                         figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
                         xlabel: str = "Prediction score [%]",
                         ylabel: str = "Count",
                         fontsize_labels: Optional[Union[int, float]] = None,
                         legend: bool = True,
                         ) -> Tuple[Figure, Axes]:
    """
    Plot a class-separated histogram of a per-sample prediction score (internal).

    Shows how a model's prediction score (e.g. a substrate-probability in ``[0, 100]`` %)
    distributes across known classes such as substrate / non-substrate / unknown, so the
    class separation of a deployed predictor can be read off at a glance. Scores in
    ``[0, 1]`` are auto-rescaled to a ``[0, 100]`` percent axis.

    This function is **internal**: it is not part of the public ``aaanalysis`` namespace and
    may change without a deprecation cycle.

    Parameters
    ----------
    df_pred : pd.DataFrame, shape (n_samples, n_info)
        One row per sample; must contain ``score`` (the prediction score) and ``group``
        (the class label used to split the histogram).
    score : str, default="score"
        Column with the per-sample prediction score (``[0, 1]`` or ``[0, 100]``).
    group : str, default="group"
        Column with the per-sample class label used to color / separate the bars.
    group_order : list of str, optional
        Order in which classes are colored / stacked. Defaults to first-appearance order.
    dict_color : dict, optional
        Mapping ``group -> color`` (overrides the canonical defaults). Canonical class names
        (``substrate``, ``non-substrate``, ``hold-out``) default to the locked sample palette.
    binwidth : int or float, default=5
        Width of the histogram bins (in score units).
    binrange : tuple, default=(0, 100)
        ``(low, high)`` range over which bins are computed; also used as the x-axis limits.
    stacked : bool, default=True
        If ``True``, class histograms are stacked (``multiple="stack"``); else overlaid
        (``multiple="layer"``).
    kde : bool, default=False
        If ``True``, overlay a kernel-density estimate per class.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    figsize : tuple, default=(7, 5)
        Figure size when ``ax`` is ``None``.
    xlabel, ylabel : str
        Axis labels.
    fontsize_labels : int or float, optional
        Font size for the axis labels (matplotlib default if ``None``).
    legend : bool, default=True
        Whether to draw the class legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes with the class-separated histogram.

    See Also
    --------
    * :func:`aaanalysis.plot_rank` for the companion ranked-candidates view.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from aaanalysis.plotting._plot_prediction_hist import plot_prediction_hist

        df_pred = pd.DataFrame({"score": [95, 80, 60, 40, 20, 5],
                                "group": ["substrate", "substrate", "hold-out",
                                          "non-substrate", "non-substrate", "non-substrate"]})
        fig, ax = plot_prediction_hist(df_pred, score="score", group="group")

    .. note::
       This symbol is internal; an example notebook and a public re-export are tracked as a
       TODO (re-export under :mod:`aaanalysis` is CONFIRM-FIRST, pending maintainer review).
    """
    # Check input
    ut.check_str(name="score", val=score)
    ut.check_str(name="group", val=group)
    ut.check_df(name="df_pred", df=df_pred, cols_required=[score, group])
    if len(df_pred) == 0:
        raise ValueError("'df_pred' (0 rows) should contain at least one sample.")
    ut.check_list_like(name="group_order", val=group_order, accept_none=True)
    ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
    ut.check_number_range(name="binwidth", val=binwidth, min_val=0, exclusive_limits=True, just_int=False)
    _check_binrange(name="binrange", val=binrange)
    ut.check_bool(name="stacked", val=stacked)
    ut.check_bool(name="kde", val=kde)
    ut.check_bool(name="legend", val=legend)
    ut.check_ax(ax=ax, accept_none=True)
    ut.check_figsize(figsize=figsize, accept_none=True)
    ut.check_str(name="xlabel", val=xlabel, accept_none=True)
    ut.check_str(name="ylabel", val=ylabel, accept_none=True)
    ut.check_number_range(name="fontsize_labels", val=fontsize_labels, min_val=0,
                          accept_none=True, just_int=False)

    # Resolve order + colors
    if group_order is None:
        group_order = list(dict.fromkeys(df_pred[group].tolist()))
    else:
        missing = set(df_pred[group]) - set(group_order)
        if missing:
            raise ValueError(f"'group_order' is missing groups present in 'df_pred': {missing}")
    dict_group_color = _resolve_group_colors(group_order=group_order, dict_color=dict_color)

    # Auto-rescale a [0, 1] probability to a [0, 100] percent axis
    df = df_pred.copy()
    scores = pd.to_numeric(df[score], errors="coerce")
    if scores.notna().any() and float(np.nanmax(scores.to_numpy())) <= 1:
        df[score] = scores * 100

    # Draw
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    multiple = "stack" if stacked else "layer"
    sns.histplot(data=df, x=score, hue=group, hue_order=group_order,
                 palette=dict_group_color, binwidth=binwidth, binrange=tuple(binrange),
                 multiple=multiple, kde=kde, legend=legend, ax=ax)
    ax.set_xlim(binrange[0], binrange[1])
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    # Integer y-ticks (counts)
    yticks = [t for t in ax.get_yticks() if float(t).is_integer()]
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([int(t) for t in yticks])
    sns.despine(ax=ax)
    # ``ax.figure`` is typed ``Figure | SubFigure | None`` by the matplotlib stubs,
    # but a top-level Axes here always belongs to a real Figure.
    return fig, ax  # pyright: ignore[reportReturnType]
