"""
This is a script for the frontend of the AAMutPlot class for visualizing amino acid
substitution impact across physicochemical scales.
"""
from typing import Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

import aaanalysis.utils as ut


# I Helper Functions
def check_df_impact(df_impact=None) -> None:
    """Check that df_impact is a valid AAMut.run output."""
    ut.check_df(df=df_impact, name="df_impact", cols_required=ut.COLS_AAMUT)
    if len(df_impact) == 0:
        raise ValueError("'df_impact' should not be empty.")


# II Main Functions
class AAMutPlot:
    """
    Plotting class for :class:`AAMut` (Amino Acid Mutator) results [Breimann24a]_.

    Visualizes the per-scale substitution-impact table produced by :meth:`AAMut.run`: a
    pairwise substitution matrix, a per-scale sensitivity ranking, and a per-pair scale
    comparison.

    .. versionadded:: 1.0.0

    """
    def __init__(self,
                 verbose: bool = False,
                 df_scales: Optional[pd.DataFrame] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of amino acid scales. Default from :func:`load_scales`.

        See Also
        --------
        * :class:`AAMut`: the logic class whose substitution impact this visualizes.
        """
        self._verbose = ut.check_verbose(verbose)
        self.df_scales = df_scales

    # Main methods
    def substitution_matrix(self,
                            df_impact: Optional[pd.DataFrame] = None,
                            ax: Optional[Axes] = None,
                            figsize: Tuple[Union[int, float], Union[int, float]] = (7, 6),
                            cmap: str = "viridis",
                            ) -> Axes:
        """
        Plot the 20x20 amino acid substitution-impact matrix.

        Each cell is the mean absolute substitution delta over scales for a ``from_aa`` ->
        ``to_aa`` pair (the overall physicochemical magnitude of that substitution).

        Parameters
        ----------
        df_impact : pd.DataFrame
            Substitution-impact table produced by :meth:`AAMut.run`.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(7, 6)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        cmap : str, default='viridis'
            Matplotlib colormap name for the heatmap.

        Returns
        -------
        ax : Axes
            Axes object of the substitution matrix.

        Examples
        --------
        .. include:: examples/aamut_plot_substitution_matrix.rst
        """
        # Validate
        check_df_impact(df_impact=df_impact)
        # Plot
        mat = (df_impact.pivot_table(index=ut.COL_FROM_AA, columns=ut.COL_TO_AA,
                                     values=ut.COL_ABS_DELTA, aggfunc="mean")
               .reindex(index=ut.LIST_CANONICAL_AA, columns=ut.LIST_CANONICAL_AA))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.heatmap(mat, ax=ax, cmap=cmap, cbar_kws={"label": "mean |delta|"})
        ax.set_xlabel("to amino acid")
        ax.set_ylabel("from amino acid")
        return ax

    def scale_ranking(self,
                      df_impact: Optional[pd.DataFrame] = None,
                      top_n: int = 20,
                      ax: Optional[Axes] = None,
                      figsize: Tuple[Union[int, float], Union[int, float]] = (6, 5),
                      color: Optional[str] = None,
                      ) -> Axes:
        """
        Plot the per-scale ranking of substitution sensitivity.

        A horizontal bar chart of the ``top_n`` scales with the largest mean absolute
        substitution delta, ordered most-sensitive first.

        Parameters
        ----------
        df_impact : pd.DataFrame
            Substitution-impact table produced by :meth:`AAMut.run`.
        top_n : int, default=20
            Number of most sensitive scales to show.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(6, 5)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        color : str, optional
            Bar color. If ``None``, the TMD color is used.

        Returns
        -------
        ax : Axes
            Axes object of the scale-ranking plot.

        Examples
        --------
        .. include:: examples/aamut_plot_scale_ranking.rst
        """
        # Validate
        check_df_impact(df_impact=df_impact)
        ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True)
        # Plot
        ranked = (df_impact.groupby(ut.COL_SCALE_ID, sort=False)[ut.COL_ABS_DELTA]
                  .mean().sort_values(ascending=False).head(top_n))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = color if color is not None else ut.COLOR_TMD
        ax.barh(ranked.index[::-1], ranked.values[::-1], color=color)
        ax.set_xlabel("mean |delta| (substitution sensitivity)")
        ax.set_ylabel("scale")
        return ax

    def aa_comparison(self,
                      df_impact: Optional[pd.DataFrame] = None,
                      from_aa: Optional[str] = None,
                      to_aa: Optional[str] = None,
                      top_n: int = 20,
                      ax: Optional[Axes] = None,
                      figsize: Tuple[Union[int, float], Union[int, float]] = (6, 5),
                      ) -> Axes:
        """
        Plot the per-scale signed delta for one amino acid substitution pair.

        Bars are colored by sign (increase vs decrease) so the physicochemical direction of a
        single ``from_aa`` -> ``to_aa`` substitution is visible across the most-affected scales.

        Parameters
        ----------
        df_impact : pd.DataFrame
            Substitution-impact table produced by :meth:`AAMut.run`.
        from_aa : str
            Substituted-from amino acid (single letter).
        to_aa : str
            Substituted-to amino acid (single letter).
        top_n : int, default=20
            Number of scales with the largest absolute delta to show.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(6, 5)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).

        Returns
        -------
        ax : Axes
            Axes object of the amino acid comparison plot.

        Examples
        --------
        .. include:: examples/aamut_plot_aa_comparison.rst
        """
        # Validate
        check_df_impact(df_impact=df_impact)
        ut.check_str(name="from_aa", val=from_aa)
        ut.check_str(name="to_aa", val=to_aa)
        ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True)
        # Plot
        sub = df_impact[(df_impact[ut.COL_FROM_AA] == from_aa) & (df_impact[ut.COL_TO_AA] == to_aa)]
        if len(sub) == 0:
            raise ValueError(f"No rows in 'df_impact' for {from_aa}->{to_aa}.")
        sub = sub.reindex(sub[ut.COL_ABS_DELTA].sort_values(ascending=False).index).head(top_n)
        colors = [ut.COLOR_FEAT_POS if d > 0 else ut.COLOR_FEAT_NEG for d in sub[ut.COL_DELTA]]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.barh(sub[ut.COL_SCALE_ID][::-1], sub[ut.COL_DELTA][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"delta ({from_aa} -> {to_aa})")
        ax.set_ylabel("scale")
        return ax
