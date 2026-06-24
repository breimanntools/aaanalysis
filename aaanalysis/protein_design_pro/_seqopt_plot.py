"""
This is a script for the frontend of the SeqOptPlot class (**[pro]**) for visualizing SeqOpt
multi-objective directed-evolution results: the Pareto-front objective scatter and the
per-generation hypervolume convergence trace.
"""
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import aaanalysis.utils as ut


# I Helper Functions
def check_objective_col(df_pareto=None, name=None, arg=None):
    """Check that an objective column name exists in df_pareto."""
    ut.check_str(name=arg, val=name, accept_none=False)
    if name not in df_pareto.columns:
        raise ValueError(f"'{arg}' ({name}) should be a column of 'df_pareto': "
                         f"{list(df_pareto.columns)}.")


# II Main Functions
class SeqOptPlot:
    """
    Plotting class for :class:`SeqOpt` (Sequence Optimizer) results (**[pro]**) [Breimann24a]_.

    Visualizes the Pareto front produced by :meth:`SeqOpt.run`: a 2-D objective scatter colored
    by non-dominated rank, and the per-generation hypervolume convergence trace.

    Every plotting method returns a ``(fig, ax)`` pair (a thin tuple subclass): unpack as
    ``fig, ax = ...``. For backward compatibility, the returned object also forwards attribute
    access to ``ax``.

    .. versionadded:: 1.0.0

    """
    def __init__(self,
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * :class:`SeqOpt`: the logic class whose Pareto front this visualizes.
        """
        self._verbose = ut.check_verbose(verbose)

    # Main methods
    def pareto_front(self,
                     df_pareto: pd.DataFrame,
                     x: str,
                     y: str,
                     ax: Optional[Axes] = None,
                     figsize: tuple = (6, 5),
                     front_only: bool = False,
                     ):
        """
        Scatter two objectives of a Pareto front, colored by non-dominated rank.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run`.
        x : str
            Objective column for the x-axis.
        y : str
            Objective column for the y-axis.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, default=(6, 5)
            Figure size when ``ax`` is None.
        front_only : bool, default=False
            If ``True``, plot only the first (``rank=0``) front.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the scatter was drawn on.

        Examples
        --------
        .. include:: examples/seqopt_pareto_front.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto", cols_required=[ut.COL_RANK])
        check_objective_col(df_pareto=df_pareto, name=x, arg="x")
        check_objective_col(df_pareto=df_pareto, name=y, arg="y")
        ut.check_bool(name="front_only", val=front_only)
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        df = df_pareto[df_pareto[ut.COL_RANK] == 0] if front_only else df_pareto
        ranks = df[ut.COL_RANK].to_numpy()
        sc = ax.scatter(df[x], df[y], c=ranks, cmap="viridis_r", s=45,
                        edgecolor="white", linewidth=0.5)
        # Connect the first front (sorted by x) to show the trade-off curve.
        front = df_pareto[df_pareto[ut.COL_RANK] == 0].sort_values(x)
        ax.plot(front[x], front[y], color=ut.COLOR_BASE if hasattr(ut, "COLOR_BASE") else "black",
                alpha=0.4, zorder=0)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if len(np.unique(ranks)) > 1:
            cbar = ax.get_figure().colorbar(sc, ax=ax)
            cbar.set_label(ut.COL_RANK)
        return ut.FigAxResult(ax.get_figure(), ax)

    def hypervolume(self,
                    trajectory: ut.ArrayLike1D,
                    ax: Optional[Axes] = None,
                    figsize: tuple = (6, 4),
                    ):
        """
        Plot the per-generation hypervolume convergence trace.

        Parameters
        ----------
        trajectory : array-like, shape (n_gen + 1,)
            Per-generation hypervolume of the front (``SeqOpt.trajectory_`` after a ``run``).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, default=(6, 4)
            Figure size when ``ax`` is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the trace was drawn on.

        Examples
        --------
        .. include:: examples/seqopt_hypervolume.rst
        """
        # Validate
        traj = np.asarray(ut.check_list_like(name="trajectory", val=trajectory), dtype=float)
        if len(traj) == 0:
            raise ValueError("'trajectory' should not be empty.")
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(traj)), traj, marker="o", markersize=3)
        ax.set_xlabel(ut.COL_GENERATION)
        ax.set_ylabel(ut.COL_HYPERVOLUME)
        return ut.FigAxResult(ax.get_figure(), ax)
