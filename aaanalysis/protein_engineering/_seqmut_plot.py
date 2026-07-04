"""
This is a script for the frontend of the SeqMutPlot class for visualizing per-position
ΔCPP / model prediction-shift mutation landscapes.
"""
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

import aaanalysis.utils as ut


# I Helper Functions
def check_df_scan(df_scan=None) -> None:
    """Check that df_scan is a valid SeqMut.scan output."""
    ut.check_df(df=df_scan, name="df_scan", cols_required=ut.COLS_SEQMUT_SCAN)
    if len(df_scan) == 0:
        raise ValueError("'df_scan' should not be empty.")


def check_df_variant(df_variant=None) -> None:
    """Check that df_variant is a valid SeqMut.combine output."""
    ut.check_df(df=df_variant, name="df_variant", cols_required=ut.COLS_SEQMUT_VARIANT)
    if len(df_variant) == 0:
        raise ValueError("'df_variant' should not be empty.")


def get_entry_subset(df=None, entry=None):
    """Return the rows of a SeqMut output for a single entry (default: the first entry)."""
    if entry is None:
        entry = df[ut.COL_ENTRY].iloc[0]
    sub = df[df[ut.COL_ENTRY] == entry]
    if len(sub) == 0:
        raise ValueError(f"'entry' ({entry}) is not present in the table.")
    return sub, entry


def get_region_color(region=None):
    """Map a sequence part (jmd_n / tmd / jmd_c) to its part color."""
    return ut.COLOR_TMD if region == ut.COL_TMD else ut.COLOR_JMD


def get_diverging_norm(values=None):
    """Return a 0-centered TwoSlopeNorm spanning the (signed) value range."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.max(finite)) if len(finite) else 1.0
    vmin = float(np.min(finite)) if len(finite) else -1.0
    eps = 1e-9
    return TwoSlopeNorm(vmin=min(vmin, -eps), vcenter=0.0, vmax=max(vmax, eps))


# II Main Functions
class SeqMutPlot:
    """
    Plotting class for :class:`SeqMut` (Sequence Mutator) results [Breimann24a]_.

    Visualizes the mutational landscape produced by :meth:`SeqMut.scan` (a per-position
    substitution heatmap, colored by the model prediction shift ``delta_pred`` when a model is
    bound, else by ``delta_cpp``) and the single-residue substitution profile, plus the combined
    designs of :meth:`SeqMut.combine` (a ranked-variant bar and a pairwise epistasis map).

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
                           figsize: Tuple[Union[int, float], Union[int, float]] = (12, 6),
                           cmap: Optional[str] = None,
                           class_names: Optional[Tuple[str, str]] = None,
                           ) -> Tuple[Figure, Axes]:
        """
        Plot the per-position mutation-scan heatmap for one sequence.

        Rows are the 20 canonical amino acids, columns are the scanned positions (labelled by
        the wild-type residue and colored by sequence part — JMD-N / TMD / JMD-C). Each cell is
        the change the substitution induces: the model prediction shift ``delta_pred`` (in
        percentage points, diverging blue-white-red) when :meth:`SeqMut.scan` was run with a
        bound model, otherwise the model-free ``delta_cpp``. With a model, the title reports the
        wild-type prediction score.

        Parameters
        ----------
        df_scan : pd.DataFrame
            Mutational landscape produced by :meth:`SeqMut.scan`.
        entry : str, optional
            Protein entry to plot. If ``None``, the first entry in ``df_scan`` is used.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(12, 6)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        cmap : str, optional
            Matplotlib colormap name. If ``None``, a diverging ``'bwr'`` is used for the signed
            ``delta_pred`` and ``'viridis'`` for the non-negative ``delta_cpp``.
        class_names : tuple of str, optional
            ``(negative, positive)`` class labels. When given (and a model was bound), the
            predicted wild-type class is appended to the title.

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            Axes object of the mutation-scan heatmap.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`SeqMutPlot` for the shared return contract).

        Examples
        --------
        .. include:: examples/seqmut_plot_mutation_landscape.rst
        """
        # Validate
        check_df_scan(df_scan=df_scan)
        # Select value column: model prediction shift when present, else the model-free magnitude
        sub, entry = get_entry_subset(df=df_scan, entry=entry)
        use_pred = ut.COL_DELTA_PRED in sub.columns
        value_col = ut.COL_DELTA_PRED if use_pred else ut.COL_DELTA_CPP
        mat = (sub.pivot_table(index=ut.COL_TO_AA, columns=ut.COL_POS, values=value_col)
               .reindex(index=ut.LIST_CANONICAL_AA))
        positions = list(mat.columns)
        pos_info = sub.drop_duplicates(subset=ut.COL_POS).set_index(ut.COL_POS)
        wt_res = [pos_info.loc[p, ut.COL_FROM_AA] for p in positions]
        regions = [pos_info.loc[p, ut.COL_REGION] for p in positions]
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.get_figure()
        if cmap is None:
            cmap = "bwr" if use_pred else "viridis"
        norm = get_diverging_norm(values=mat.to_numpy(dtype=float)) if use_pred else None
        label = "Δ prediction score [%]" if use_pred else ut.COL_DELTA_CPP
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label, fraction=0.025, pad=0.02)
        ax.set_yticks(range(len(ut.LIST_CANONICAL_AA)))
        ax.set_yticklabels(ut.LIST_CANONICAL_AA)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(wt_res)
        # Color each position label by its sequence part (parts bar).
        for tick, region in zip(ax.get_xticklabels(), regions):
            tick.set_color("white")
            tick.set_fontweight("bold")
            tick.set_bbox(dict(facecolor=get_region_color(region=region), edgecolor="none",
                               pad=1.5))
        # Vertical separators where the sequence part changes.
        for i in range(1, len(regions)):
            if regions[i] != regions[i - 1]:
                ax.axvline(i - 0.5, color="black", linewidth=1.5)
        ax.set_xlabel("Sequence positions")
        ax.set_ylabel("Amino acid substitutions")
        title = f"Mutation Scan for {entry}"
        if use_pred and ut.COL_WT_PRED in sub.columns:
            wt = float(pos_info[ut.COL_WT_PRED].iloc[0])
            wt_std = float(pos_info[ut.COL_WT_PRED_STD].iloc[0])
            anno = f"{wt:.1f} ± {wt_std:.1f}%" if np.isfinite(wt_std) else f"{wt:.1f}%"
            if class_names is not None:
                anno += f", {class_names[1] if wt >= 50 else class_names[0]}"
            title += f" ({anno})"
        ax.set_title(title)
        return ut.FigAxResult(fig, ax)

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

        A bar chart of the per-substitution change at one position — the model prediction shift
        ``delta_pred`` when present, otherwise ``delta_cpp`` — showing which target residues move
        the outcome most at that site.

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
        sub, entry = get_entry_subset(df=df_scan, entry=entry)
        use_pred = ut.COL_DELTA_PRED in sub.columns
        value_col = ut.COL_DELTA_PRED if use_pred else ut.COL_DELTA_CPP
        sub = sub[sub[ut.COL_POS] == pos].sort_values(value_col, ascending=False)
        if len(sub) == 0:
            raise ValueError(f"No scanned mutations at 'pos' ({pos}) for entry '{entry}'.")
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = color if color is not None else ut.COLOR_TMD
        ax.bar(sub[ut.COL_TO_AA], sub[value_col], color=color)
        from_aa = sub[ut.COL_FROM_AA].iloc[0]
        ax.set_title(f"{entry}: {from_aa}{pos} substitutions")
        ax.set_xlabel("to amino acid")
        ax.set_ylabel("Δ prediction score [%]" if use_pred else ut.COL_DELTA_CPP)
        return ut.FigAxResult(ax.get_figure(), ax)

    def variant_impact(self,
                       df_variant: pd.DataFrame,
                       entry: Optional[str] = None,
                       n: Optional[int] = None,
                       ax: Optional[Axes] = None,
                       figsize: Tuple[Union[int, float], Union[int, float]] = (8, 5),
                       ) -> Tuple[Figure, Axes]:
        """
        Plot a ranked bar chart of combined variants by their impact.

        One horizontal bar per combined variant (from :meth:`SeqMut.combine`), its length the
        variant's ``delta_pred`` (model prediction shift) when present, otherwise ``shift_score``;
        bars are colored red/blue by sign. This is the variant-level view of stacking 2-3
        mutations — complementary to the single-mutation :meth:`SeqMutPlot.mutation_landscape`.

        Parameters
        ----------
        df_variant : pd.DataFrame
            Combined-variant table produced by :meth:`SeqMut.combine`.
        entry : str, optional
            Protein entry to plot. If ``None``, all variants are shown.
        n : int, optional
            Plot only the top ``n`` variants by impact. If ``None``, all are shown.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(8, 5)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            Axes object of the ranked-variant bar chart.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`SeqMutPlot` for the shared return contract).

        Examples
        --------
        .. include:: examples/seqmut_plot_variant_impact.rst
        """
        # Validate
        check_df_variant(df_variant=df_variant)
        if n is not None:
            ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
        # Plot
        sub = df_variant
        if entry is not None:
            sub, entry = get_entry_subset(df=df_variant, entry=entry)
        use_pred = ut.COL_DELTA_PRED in sub.columns
        value_col = ut.COL_DELTA_PRED if use_pred else ut.COL_SHIFT_SCORE
        sub = sub.sort_values(value_col, ascending=False)
        if n is not None:
            sub = sub.head(n)
        sub = sub.iloc[::-1]  # largest at top of the horizontal bar chart
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        colors = [ut.COLOR_TMD if v >= 0 else ut.COLOR_JMD for v in sub[value_col]]
        ax.barh(sub[ut.COL_VARIANT].astype(str), sub[value_col], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ prediction score [%]" if use_pred else ut.COL_SHIFT_SCORE)
        ax.set_ylabel("variant")
        ax.set_title("Combined-variant impact")
        return ut.FigAxResult(ax.get_figure(), ax)

    def epistasis(self,
                  df_variant: pd.DataFrame,
                  entry: Optional[str] = None,
                  ax: Optional[Axes] = None,
                  figsize: Tuple[Union[int, float], Union[int, float]] = (7, 6),
                  cmap: str = "bwr",
                  ) -> Tuple[Figure, Axes]:
        """
        Plot the pairwise non-additivity (epistasis) of mutation pairs.

        Built from a :meth:`SeqMut.combine` table holding the single mutations (``n_mut == 1``)
        and their pairwise combinations (``n_mut == 2``) of a chosen mutation set. Off-diagonal
        cell ``(i, j)`` is the epistasis ``ΔP(i+j) - (ΔP(i) + ΔP(j))`` — positive (red) means the
        pair does better than the sum of the singles (synergy), negative (blue) worse
        (antagonism); the diagonal carries the single-mutation effect. Uses ``delta_pred`` when
        present, otherwise ``shift_score``.

        Parameters
        ----------
        df_variant : pd.DataFrame
            Combined-variant table from :meth:`SeqMut.combine` containing the singles and pairs.
        entry : str, optional
            Protein entry to plot. If ``None``, the first entry in ``df_variant`` is used.
        ax : Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new one is created.
        figsize : tuple, default=(7, 6)
            Figure dimensions (width, height) in inches (used when ``ax`` is ``None``).
        cmap : str, default='bwr'
            Diverging matplotlib colormap name (0-centered).

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            Axes object of the epistasis heatmap.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`SeqMutPlot` for the shared return contract).

        Examples
        --------
        .. include:: examples/seqmut_plot_epistasis.rst
        """
        # Validate
        check_df_variant(df_variant=df_variant)
        sub, entry = get_entry_subset(df=df_variant, entry=entry)
        use_pred = ut.COL_DELTA_PRED in sub.columns
        value_col = ut.COL_DELTA_PRED if use_pred else ut.COL_SHIFT_SCORE
        singles = {r[ut.COL_VARIANT]: float(r[value_col]) for _, r in sub.iterrows()
                   if int(r[ut.COL_N_MUT]) == 1}
        if len(singles) < 2:
            raise ValueError("'df_variant' should contain at least two single mutations "
                             "(n_mut == 1) plus their pairwise combinations for an epistasis map.")
        labels = sorted(singles)
        idx = {m: i for i, m in enumerate(labels)}
        mat = np.full((len(labels), len(labels)), np.nan)
        for m, v in singles.items():
            mat[idx[m], idx[m]] = v
        for _, r in sub.iterrows():
            if int(r[ut.COL_N_MUT]) != 2:
                continue
            parts = str(r[ut.COL_VARIANT]).split("+")
            if len(parts) != 2 or parts[0] not in idx or parts[1] not in idx:
                continue
            i, j = idx[parts[0]], idx[parts[1]]
            epi = float(r[value_col]) - (singles[parts[0]] + singles[parts[1]])
            mat[i, j] = mat[j, i] = epi
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.get_figure()
        norm = get_diverging_norm(values=mat)
        im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest")
        fig.colorbar(im, ax=ax, label="epistasis (Δ prediction score [%])" if use_pred
                     else "epistasis (shift_score)", fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_title(f"Pairwise epistasis: {entry}")
        return ut.FigAxResult(fig, ax)
