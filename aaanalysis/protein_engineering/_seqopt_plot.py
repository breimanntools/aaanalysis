"""
This is a script for the frontend of the SeqOptPlot class for visualizing SeqOpt
multi-objective directed-evolution results: the Pareto-front objective scatter and the
per-generation hypervolume convergence trace.
"""
from typing import Optional, Tuple, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import aaanalysis.utils as ut


# I Helper Functions
def check_objective_col(df_pareto=None, name=None, arg=None):
    """Check that an objective column name exists in df_pareto."""
    ut.check_str(name=arg, val=name, accept_none=False)
    if name not in df_pareto.columns:
        raise ValueError(f"'{arg}' ({name}) should be a column of 'df_pareto': "
                         f"{list(df_pareto.columns)}.")


def mutations_from_label_(label) -> frozenset:
    """Parse a '<from><pos><to>'-joined variant label into a frozenset of (pos, to_aa)."""
    return frozenset((int(m.group(2)), m.group(3))
                     for m in re.finditer(r"([A-Z])(\d+)([A-Z])", str(label)))


def _objective_cols(df_pareto):
    """Objective columns of a df_pareto (all but the fixed base/rank/crowding columns)."""
    fixed = set(ut.COLS_PARETO_BASE) | {ut.COL_RANK, ut.COL_CROWDING}
    return [c for c in df_pareto.columns if c not in fixed]


# II Main Functions
class SeqOptPlot:
    """
    Plotting class for :class:`SeqOpt` (Sequence Optimizer) results [Breimann24a]_.

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
                     z: Optional[str] = None,
                     ax: Optional[Axes] = None,
                     figsize: tuple = (6, 5),
                     front_only: bool = False,
                     cmap: str = "viridis_r",
                     ) -> Tuple[Figure, Axes]:
        """
        Scatter two (or three) objectives of a Pareto front, colored by non-dominated rank.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run`.
        x : str
            Objective column for the x-axis.
        y : str
            Objective column for the y-axis.
        z : str, optional
            Third objective column. When given, a 3-D scatter is drawn (for ``> 3`` objectives
            use :meth:`SeqOptPlot.parallel_coordinates`).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure (3-D when ``z`` is given) is created when ``None``.
        figsize : tuple, default=(6, 5)
            Figure size when ``ax`` is None.
        front_only : bool, default=False
            If ``True``, plot only the first (``rank=0``) front.
        cmap : str, default="viridis_r"
            Matplotlib colormap name for the rank coloring.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the scatter was drawn on.

        Examples
        --------
        .. include:: examples/seqo_pareto_front.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto", cols_required=[ut.COL_RANK])
        check_objective_col(df_pareto=df_pareto, name=x, arg="x")
        check_objective_col(df_pareto=df_pareto, name=y, arg="y")
        if z is not None:
            check_objective_col(df_pareto=df_pareto, name=z, arg="z")
        ut.check_bool(name="front_only", val=front_only)
        # Plot
        df = df_pareto[df_pareto[ut.COL_RANK] == 0] if front_only else df_pareto
        ranks = df[ut.COL_RANK].to_numpy()
        if z is not None:
            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(df[x], df[y], df[z], c=ranks, cmap=cmap, s=40,
                            edgecolor="white", linewidth=0.5)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(df[x], df[y], c=ranks, cmap=cmap, s=45,
                            edgecolor="white", linewidth=0.5)
            # Connect the first front (sorted by x) to show the trade-off curve.
            front = df_pareto[df_pareto[ut.COL_RANK] == 0].sort_values(x)
            ax.plot(front[x], front[y], color="black", alpha=0.4, zorder=0)
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
                    ) -> Tuple[Figure, Axes]:
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
        .. include:: examples/seqo_hypervolume.rst
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

    def convergence(self,
                    history: pd.DataFrame,
                    figsize: tuple = (6, 7),
                    ) -> Tuple[Figure, List[Axes]]:
        """
        Plot per-generation convergence: hypervolume, spread and per-objective best.

        A multi-panel view of the optimization converging across generations — the dominated
        hypervolume rising, the front diversity (spread), and each objective's best front value.

        Parameters
        ----------
        history : pd.DataFrame
            Per-generation history (``SeqOpt.history_`` after a ``run``) with a ``generation``
            column, ``hypervolume``, ``spread`` and one ``best_<objective>`` column per objective.
        figsize : tuple, default=(6, 7)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : numpy.ndarray of matplotlib.axes.Axes
            The panel axes (hypervolume, spread, per-objective best).

        Examples
        --------
        .. include:: examples/seqo_convergence.rst
        """
        # Validate
        ut.check_df(df=history, name="history", cols_required=[ut.COL_GENERATION])
        best_cols = [c for c in history.columns if c.startswith("best_")]
        gen = history[ut.COL_GENERATION]
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        if ut.COL_HYPERVOLUME in history:
            axes[0].plot(gen, history[ut.COL_HYPERVOLUME], marker="o", markersize=3)
        axes[0].set_ylabel(ut.COL_HYPERVOLUME)
        if ut.COL_SPREAD in history:
            axes[1].plot(gen, history[ut.COL_SPREAD], marker="o", markersize=3, color="C1")
        axes[1].set_ylabel(ut.COL_SPREAD)
        # Per-objective best line + (when tracked) the population mean line and min-max band —
        # the classic GA "fitness over generations" view.
        for i, c in enumerate(best_cols):
            name = c[len("best_"):]
            color = f"C{i}"
            axes[2].plot(gen, history[c], marker="o", markersize=3, color=color, label=name)
            mean_c, worst_c = f"mean_{name}", f"worst_{name}"
            if mean_c in history and worst_c in history:
                axes[2].plot(gen, history[mean_c], color=color, alpha=0.6, linewidth=1, ls="--")
                axes[2].fill_between(gen, history[worst_c], history[c], color=color, alpha=0.12)
        axes[2].set_ylabel("objective (best / mean band)")
        axes[2].set_xlabel(ut.COL_GENERATION)
        if best_cols:
            axes[2].legend(fontsize="small")
        return ut.FigAxResult(fig, axes)

    def parallel_coordinates(self,
                             df_pareto: pd.DataFrame,
                             objectives: List[str],
                             ax: Optional[Axes] = None,
                             figsize: tuple = (7, 4),
                             front_only: bool = True,
                             cmap: str = "viridis_r",
                             ) -> Tuple[Figure, Axes]:
        """
        Parallel-coordinates plot of a Pareto front over any number of objectives.

        Each variant is a line across the objective axes (min-max normalized per objective),
        colored by non-dominated rank — the way to read trade-offs for ``> 3`` objectives.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run`.
        objectives : list of str
            Objective columns to place on the parallel axes (in order).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, default=(7, 4)
            Figure size when ``ax`` is None.
        front_only : bool, default=True
            If ``True``, plot only the first (``rank=0``) front.
        cmap : str, default="viridis_r"
            Matplotlib colormap name for the rank coloring.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the lines were drawn on.

        Examples
        --------
        .. include:: examples/seqo_parallel_coordinates.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto", cols_required=[ut.COL_RANK])
        objectives = ut.check_list_like(name="objectives", val=objectives, accept_none=False)
        if len(objectives) < 2:
            raise ValueError(f"'objectives' (n={len(objectives)}) should list at least two columns.")
        for o in objectives:
            check_objective_col(df_pareto=df_pareto, name=o, arg="objectives")
        ut.check_bool(name="front_only", val=front_only)
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        df = df_pareto[df_pareto[ut.COL_RANK] == 0] if front_only else df_pareto
        M = df[objectives].to_numpy(dtype=float)
        lo, hi = M.min(axis=0), M.max(axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        Mn = (M - lo) / span
        xs = np.arange(len(objectives))
        cmap_obj = plt.get_cmap(cmap)
        ranks = df[ut.COL_RANK].to_numpy()
        rmax = max(int(ranks.max()), 1)
        for row, r in zip(Mn, ranks):
            ax.plot(xs, row, color=cmap_obj(r / rmax), alpha=0.6, linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(objectives, rotation=20, ha="right")
        ax.set_ylabel("min-max normalized")
        return ut.FigAxResult(ax.get_figure(), ax)

    def mutation_map(self,
                     df_pareto: pd.DataFrame,
                     ax: Optional[Axes] = None,
                     figsize: tuple = (8, 4),
                     front_only: bool = True,
                     cmap: str = "Reds",
                     ) -> Tuple[Figure, Axes]:
        """
        Heatmap of substitution enrichment across the Pareto front (position x amino acid).

        Each cell counts how often a given substitution (target amino acid at a 1-based position)
        appears among the front's variants — the directed-evolution view of *which* mutations the
        optimization converged on.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run` (its ``variant`` labels are parsed).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, default=(8, 4)
            Figure size when ``ax`` is None.
        front_only : bool, default=True
            If ``True``, count only the first (``rank=0``) front.
        cmap : str, default="Reds"
            Matplotlib colormap name for the enrichment counts.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the heatmap was drawn on.

        Examples
        --------
        .. include:: examples/seqo_mutation_map.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto",
                    cols_required=[ut.COL_RANK, ut.COL_VARIANT])
        ut.check_bool(name="front_only", val=front_only)
        # Count substitutions (parse '<from><pos><to>' tokens) across the selected variants
        df = df_pareto[df_pareto[ut.COL_RANK] == 0] if front_only else df_pareto
        aas = list(ut.LIST_CANONICAL_AA)
        counts: dict = {}
        for label in df[ut.COL_VARIANT]:
            for m in re.finditer(r"([A-Z])(\d+)([A-Z])", str(label)):
                pos, to_aa = int(m.group(2)), m.group(3)
                counts[(pos, to_aa)] = counts.get((pos, to_aa), 0) + 1
        if not counts:
            raise ValueError("'df_pareto' has no mutations to map (front is the wild-type only).")
        positions = sorted({p for p, _ in counts})
        mat = np.zeros((len(aas), len(positions)), dtype=float)
        for (pos, to_aa), c in counts.items():
            mat[aas.index(to_aa), positions.index(pos)] = c
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=90, fontsize="small")
        ax.set_yticks(range(len(aas)))
        ax.set_yticklabels(aas, fontsize="small")
        ax.set_xlabel("position")
        ax.set_ylabel("substituted amino acid")
        cbar = ax.get_figure().colorbar(im, ax=ax)
        cbar.set_label("count on front")
        return ut.FigAxResult(ax.get_figure(), ax)

    def genealogy(self,
                  df_pareto: pd.DataFrame,
                  ax: Optional[Axes] = None,
                  figsize: tuple = (8, 5),
                  front_only: bool = True,
                  cmap: str = "viridis",
                  ) -> Tuple[Figure, Axes]:
        """
        Mutational-lineage tree of the variants, rooted at the wild-type.

        The directed-evolution analogue of a genealogy tree: nodes are variants placed by their
        number of mutations (depth), each linked to the largest lower-order variant whose
        mutation set it extends (or to the wild-type), and colored by the first objective. It
        shows how the designed variants are built up mutation by mutation from the wild-type.

        Parameters
        ----------
        df_pareto : pd.DataFrame
            Output of :meth:`SeqOpt.run`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, default=(8, 5)
            Figure size when ``ax`` is None.
        front_only : bool, default=True
            If ``True``, build the lineage from the first (``rank=0``) front only.
        cmap : str, default="viridis"
            Matplotlib colormap name for the objective coloring.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes the lineage was drawn on.

        Examples
        --------
        .. include:: examples/seqo_genealogy.rst
        """
        # Validate
        ut.check_df(df=df_pareto, name="df_pareto",
                    cols_required=[ut.COL_RANK, ut.COL_VARIANT])
        ut.check_bool(name="front_only", val=front_only)
        obj_cols = _objective_cols(df_pareto)
        if not obj_cols:
            raise ValueError("'df_pareto' should carry at least one objective column.")
        primary = obj_cols[0]
        df = df_pareto[df_pareto[ut.COL_RANK] == 0] if front_only else df_pareto
        # Nodes: wild-type (empty mutation set) + each variant; value = first objective.
        nodes = {frozenset(): float("nan")}
        for label, val in zip(df[ut.COL_VARIANT], df[primary]):
            nodes[mutations_from_label_(label)] = float(val)
        keys = sorted(nodes, key=len)
        # Edge: each variant -> the largest proper subset present (else wild-type).
        edges = []
        for k in keys:
            if len(k) == 0:
                continue
            parent = max((j for j in keys if j < k), key=len, default=frozenset())
            edges.append((parent, k))
        # Layout: x = number of mutations, y = spread within a depth level.
        levels: dict = {}
        for k in keys:
            levels.setdefault(len(k), []).append(k)
        pos = {}
        for depth, ks in levels.items():
            for i, k in enumerate(ks):
                pos[k] = (depth, i - (len(ks) - 1) / 2.0)
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        for a, b in edges:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="grey", alpha=0.5, linewidth=0.8, zorder=0)
        variant_keys = [k for k in keys if len(k) > 0]
        xs = [pos[k][0] for k in variant_keys]
        ys = [pos[k][1] for k in variant_keys]
        vals = [nodes[k] for k in variant_keys]
        sc = ax.scatter(xs, ys, c=vals, cmap=cmap, s=45, edgecolor="white", linewidth=0.5)
        ax.scatter([0], [0], marker="s", s=70, color="black")
        ax.annotate("WT", (0, 0), textcoords="offset points", xytext=(0, 8), ha="center")
        ax.set_xlabel("number of mutations")
        ax.set_yticks([])
        cbar = ax.get_figure().colorbar(sc, ax=ax)
        cbar.set_label(primary)
        return ut.FigAxResult(ax.get_figure(), ax)
