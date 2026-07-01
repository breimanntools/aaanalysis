"""
This is a script for the frontend of the plot_eval_heatmap function — a house-preset
annotated evaluation heatmap for a static score grid.
"""
from typing import Optional, Union, Tuple
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from aaanalysis import utils as ut


# I Helper Functions


# II Main Functions
def plot_eval_heatmap(df_eval: pd.DataFrame,
                      xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None,
                      vmin: Union[int, float] = 50,
                      vmax: Union[int, float] = 100,
                      cbar_label: Optional[str] = "Balanced accuracy [%]",
                      xtick_rotation: Union[int, float] = 0,
                      ytick_rotation: Union[int, float] = 0,
                      square: bool = True,
                      figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                      ax: Optional[Axes] = None,
                      ) -> Axes:
    """
    Plot a house-preset annotated evaluation heatmap from a static score grid.

    Renders ``df_eval`` (rows × columns of evaluation scores, e.g. balanced accuracy in
    percent) as a ``viridis`` heatmap with integer annotations, fixed ``[vmin, vmax]``
    color limits, and a labeled colorbar — collapsing the hand-built seaborn block that is
    otherwise copied for every sweep result into a single call. Left/bottom ticks are
    removed for the package look; tick labels are horizontal by default and can be rotated
    (e.g. for long column labels) via ``xtick_rotation`` / ``ytick_rotation``.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    df_eval : pd.DataFrame, shape (n_rows, n_cols)
        Numeric score grid. Each cell holds the score for one row × column configuration
        (the ``index`` becomes the y-axis levels, the ``columns`` the x-axis levels).
    xlabel : str, optional
        Label for the x-axis. If ``None``, seaborn's default (the ``columns`` name, if any)
        is kept.
    ylabel : str, optional
        Label for the y-axis. If ``None``, seaborn's default (the ``index`` name, if any)
        is kept.
    vmin : int or float, default=50
        Lower bound of the color scale (and colorbar).
    vmax : int or float, default=100
        Upper bound of the color scale (and colorbar). Must be greater than ``vmin``.
    cbar_label : str, optional
        Label drawn next to the colorbar. If ``None``, no colorbar label is set.
    xtick_rotation : int or float, default=0
        Rotation (degrees) of the x-axis tick labels. Non-zero values are right-aligned to
        keep long column labels legible without overlap.
    ytick_rotation : int or float, default=0
        Rotation (degrees) of the y-axis tick labels.
    square : bool, default=True
        If ``True``, force each heatmap cell to be equal in width and height (a square grid),
        the convention for an evaluation map. Set ``False`` to let the cells stretch to fill
        the axes/figure.
    figsize : tuple, optional
        Figure ``(width, height)`` used when ``ax`` is ``None``. If ``None``, matplotlib's
        default figure size is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the annotated heatmap.

    See Also
    --------
    * :func:`aaanalysis.pipe.plot_eval` is the adaptive sibling: it inspects a
      ``find_features`` sweep table and lays out one or more panels automatically.
      ``plot_eval_heatmap`` is the simple **static** entry point — hand it a ready-made
      score grid and it applies the house preset, with no axis selection or multi-panel
      logic.

    Examples
    --------
    .. include:: examples/plot_eval_heatmap.rst
    """
    # Check input
    ut.check_df(name="df_eval", df=df_eval, accept_none=False)
    if df_eval.empty:
        raise ValueError(f"'df_eval' (shape {df_eval.shape}) should contain at least one "
                         f"row and one column.")
    non_numeric = df_eval.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"'df_eval' should be all-numeric; non-numeric columns: {non_numeric}.")
    ut.check_str(name="xlabel", val=xlabel, accept_none=True)
    ut.check_str(name="ylabel", val=ylabel, accept_none=True)
    ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
    ut.check_str(name="cbar_label", val=cbar_label, accept_none=True)
    ut.check_number_val(name="xtick_rotation", val=xtick_rotation, just_int=False)
    ut.check_number_val(name="ytick_rotation", val=ytick_rotation, just_int=False)
    ut.check_bool(name="square", val=square)
    ut.check_figsize(figsize=figsize, accept_none=True)
    ut.check_ax(ax=ax, accept_none=True)

    # Draw
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    cbar_kws = dict(label=cbar_label) if cbar_label is not None else None
    sns.heatmap(df_eval, ax=ax, vmin=vmin, vmax=vmax, cmap="viridis", annot=True,
                fmt=".0f", linewidths=0.1, square=square, cbar_kws=cbar_kws)
    ax.tick_params(left=False, bottom=False)
    x_ha = "right" if xtick_rotation % 360 != 0 else "center"
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha=x_ha)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
