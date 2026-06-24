"""
This is a script for the frontend of the SeqMutPlot class for visualizing per-position
ΔCPP mutation landscapes.
"""
from typing import Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

import aaanalysis.utils as ut


# I Helper Functions
def check_df_scan(df_scan=None) -> None:
    """Check that df_scan is a valid SeqMut.scan output."""
    ut.check_df(df=df_scan, name="df_scan", cols_required=ut.COLS_SEQMUT_SCAN)
    if len(df_scan) == 0:
        raise ValueError("'df_scan' should not be empty.")


def get_entry_subset(df_scan=None, entry=None):
    """Return the rows of df_scan for a single entry (default: the first entry)."""
    if entry is None:
        entry = df_scan[ut.COL_ENTRY].iloc[0]
    sub = df_scan[df_scan[ut.COL_ENTRY] == entry]
    if len(sub) == 0:
        raise ValueError(f"'entry' ({entry}) is not present in 'df_scan'.")
    return sub, entry


# II Main Functions
class SeqMutPlot:
    """
    Plotting class for :class:`SeqMut` (Sequence Mutator) results [Breimann24a]_.

    Visualizes the ΔCPP mutational landscape produced by :meth:`SeqMut.scan`: a per-position
    substitution heatmap and the substitution profile of a single residue.

    Every plotting method returns a ``(fig, ax)`` pair (a thin tuple subclass): unpack as
    ``fig, ax = ...``. For backward compatibility, the returned object also forwards attribute
    access to ``ax``, so legacy ``ax = ...; ax.set_title(...)`` keeps working.

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
        * :class:`SeqMut`: the logic class whose mutational scan this visualizes.
        """
        self._verbose = ut.check_verbose(verbose)

    # Main methods
    def mutation_landscape(self,
                           df_scan: pd.DataFrame,
                           entry: Optional[str] = None,
                           ax: Optional[Axes] = None,
                           figsize: Tuple[Union[int, float], Union[int, float]] = (10, 5),
                           cmap: str = "viridis",
                           ) -> Tuple[Figure, Axes]:
        """
        Plot the per-position ΔCPP mutation landscape for one sequence.

        Rows are the 20 canonical amino acids, columns are scanned positions; each cell is the
        ``delta_cpp`` of mutating that position to that amino acid.

        Parameters
        ----------
        df_scan : pd.DataFrame
            Mutational landscape produced by :meth:`SeqMut.scan`.
        entry : str, optional
            Protein entry to plot. If ``None``, the first entry in ``df_scan`` is used.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(10, 5)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        cmap : str, default='viridis'
            Matplotlib colormap name for the heatmap.

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            Axes object of the mutation landscape.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`SeqMutPlot` for the shared return contract).

        Examples
        --------
        .. include:: examples/seqmut_plot_mutation_landscape.rst
        """
        # Validate
        check_df_scan(df_scan=df_scan)
        # Plot
        sub, entry = get_entry_subset(df_scan=df_scan, entry=entry)
        mat = (sub.pivot_table(index=ut.COL_TO_AA, columns=ut.COL_POS, values=ut.COL_DELTA_CPP)
               .reindex(index=ut.LIST_CANONICAL_AA))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.heatmap(mat, ax=ax, cmap=cmap, cbar_kws={"label": "delta_cpp"})
        ax.set_title(f"Mutation landscape: {entry}")
        ax.set_xlabel("position")
        ax.set_ylabel("to amino acid")
        return ut.FigAxResult(ax.get_figure(), ax)

    def residue_mutation_impact(self,
                                df_scan: pd.DataFrame,
                                entry: Optional[str] = None,
                                *,
                                pos: int,
                                ax: Optional[Axes] = None,
                                figsize: Tuple[Union[int, float], Union[int, float]] = (6, 5),
                                color: Optional[str] = None,
                                ) -> Tuple[Figure, Axes]:
        """
        Plot the substitution impact for a single residue across all substitutions.

        A bar chart of the ``delta_cpp`` of every scanned substitution at one position, showing
        which target residues perturb the CPP profile most at that site.

        Parameters
        ----------
        df_scan : pd.DataFrame
            Mutational landscape produced by :meth:`SeqMut.scan`.
        entry : str, optional
            Protein entry to plot. If ``None``, the first entry in ``df_scan`` is used.
        pos : int
            1-based residue position to plot.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(6, 5)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        color : str, optional
            Bar color. If ``None``, the TMD color is used.

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            Axes object of the residue mutation-impact plot.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`SeqMutPlot` for the shared return contract).

        Examples
        --------
        .. include:: examples/seqmut_plot_residue_mutation_impact.rst
        """
        # Validate
        check_df_scan(df_scan=df_scan)
        ut.check_number_range(name="pos", val=pos, min_val=1, just_int=True)
        # Plot
        sub, entry = get_entry_subset(df_scan=df_scan, entry=entry)
        sub = sub[sub[ut.COL_POS] == pos].sort_values(ut.COL_DELTA_CPP, ascending=False)
        if len(sub) == 0:
            raise ValueError(f"No scanned mutations at 'pos' ({pos}) for entry '{entry}'.")
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = color if color is not None else ut.COLOR_TMD
        ax.bar(sub[ut.COL_TO_AA], sub[ut.COL_DELTA_CPP], color=color)
        from_aa = sub[ut.COL_FROM_AA].iloc[0]
        ax.set_title(f"{entry}: {from_aa}{pos} substitutions")
        ax.set_xlabel("to amino acid")
        ax.set_ylabel("delta_cpp")
        return ut.FigAxResult(ax.get_figure(), ax)
