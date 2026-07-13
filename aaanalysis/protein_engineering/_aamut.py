"""
This is a script for the frontend of the AAMut class for analysing the effect of amino acid
substitutions on physicochemical property scales.
"""
from typing import Optional, List, Union
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool
from ._backend.aamut.aamut import comp_substitution_impact, eval_substitution_impact


# I Helper Functions
def check_df_scales_aa(df_scales=None) -> None:
    """Check that df_scales is a DataFrame indexed by the canonical amino acids."""
    if not isinstance(df_scales, pd.DataFrame):
        raise ValueError(f"'df_scales' ({type(df_scales)}) should be a pandas DataFrame.")
    missing = [aa for aa in ut.LIST_CANONICAL_AA if aa not in df_scales.index]
    if len(missing) > 0:
        raise ValueError(f"'df_scales' index should contain the 20 canonical amino acids; "
                         f"missing: {missing}")
    if df_scales.shape[1] == 0:
        raise ValueError("'df_scales' should contain at least one scale (column).")


def check_aa(name=None, val=None):
    """Check that val (or each element) is a canonical amino acid letter."""
    if val is None:
        return None
    list_aa = [val] if isinstance(val, str) else list(val)
    wrong = [aa for aa in list_aa if aa not in ut.LIST_CANONICAL_AA]
    if len(wrong) > 0:
        raise ValueError(f"'{name}' ({wrong}) should be canonical amino acid(s) from "
                         f"{ut.LIST_CANONICAL_AA}")
    return list_aa


def check_scales_subset(scales=None, df_scales=None):
    """Check that requested scales are columns of df_scales; default to all columns."""
    if scales is None:
        return list(df_scales.columns)
    scales = ut.check_list_like(name="scales", val=scales, accept_str=True)
    wrong = [s for s in scales if s not in df_scales.columns]
    if len(wrong) > 0:
        raise ValueError(f"'scales' ({wrong}) should be columns of 'df_scales'.")
    return scales


# II Main Functions
class AAMut(Tool):
    """
    Amino Acid Mutator (AAMut) class for analyzing the physicochemical impact of amino acid
    substitutions on property scales [Breimann24a]_.

    As a ``Tool``, it implements the ``.run`` / ``.eval`` pipeline contract.

    ``AAMut`` is **CPP-agnostic**: it quantifies how substituting one amino acid for another
    shifts each physicochemical scale value, independent of any sequence or prediction task.
    It is the residue-level building block of the protein-design module — the sequence-level,
    CPP-aware counterpart is :class:`SeqMut`.

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
            DataFrame of amino acid scales (index = canonical amino acids, columns = scale ids).
            Default from :func:`load_scales`.

        See Also
        --------
        * :class:`SeqMut` for the sequence-level, CPP-guided mutation analysis.
        """
        self._verbose = ut.check_verbose(verbose)
        if df_scales is None:
            df_scales = ut.load_default_scales()
        check_df_scales_aa(df_scales=df_scales)
        self.df_scales = df_scales
        self._df_cat = ut.load_default_scales(scale_cat=True)

    # Main method
    def run(self,
            from_aa: Optional[Union[str, List[str]]] = None,
            to_aa: Optional[Union[str, List[str]]] = None,
            scales: Optional[List[str]] = None,
            ) -> pd.DataFrame:
        """
        Compute the signed per-scale impact of amino acid substitutions.

        For every ``from_aa`` -> ``to_aa`` pair and every scale, the impact is the signed
        difference ``delta = df_scales.loc[to_aa, scale] - df_scales.loc[from_aa, scale]``.
        With both ``from_aa`` and ``to_aa`` unset, all 20x19 ordered substitution pairs are
        returned (a long, AAontology-annotated substitution table).

        Parameters
        ----------
        from_aa : str or list of str, optional
            Amino acid(s) to substitute from. If ``None``, all canonical amino acids are used.
        to_aa : str or list of str, optional
            Amino acid(s) to substitute to. If ``None``, all canonical amino acids are used.
        scales : list of str, optional
            Subset of scale ids to evaluate. If ``None``, all scales of ``df_scales`` are used.

        Returns
        -------
        df_impact : pd.DataFrame, shape (n_pairs * n_scales, 7)
            Tidy substitution-impact table with columns ``from_aa``, ``to_aa``, ``scale_id``,
            ``category``, ``subcategory``, ``delta`` (signed), and ``abs_delta`` (magnitude).

        Examples
        --------
        .. include:: examples/aam_run.rst
        """
        # Validate
        list_from = check_aa(name="from_aa", val=from_aa) or ut.LIST_CANONICAL_AA
        list_to = check_aa(name="to_aa", val=to_aa) or ut.LIST_CANONICAL_AA
        list_scales = check_scales_subset(scales=scales, df_scales=self.df_scales)
        # Compute
        df_impact = comp_substitution_impact(df_scales=self.df_scales, df_cat=self._df_cat,
                                             list_from=list_from, list_to=list_to,
                                             list_scales=list_scales)
        if self._verbose:
            ut.print_out(f"AAMut computed {len(df_impact)} substitution-impact values "
                         f"({len(list_from)}x{len(list_to)} pairs, {len(list_scales)} scales).")
        return df_impact

    def eval(self,
             df_impact: pd.DataFrame,
             ) -> pd.DataFrame:
        """
        Evaluate substitution impact by ranking scales on their mean substitution sensitivity.

        Scales with a high mean absolute delta are the physicochemical properties most affected
        by amino acid substitutions; the ranking points to which scales drive a mutation's impact.

        Parameters
        ----------
        df_impact : pd.DataFrame
            Substitution-impact table produced by :meth:`AAMut.run`.

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_scales, 2)
            One row per scale with the mean absolute substitution delta
            (``mean_delta_cpp``), sorted from most to least sensitive.

        Examples
        --------
        .. include:: examples/aam_eval.rst
        """
        # Validate
        ut.check_df(df=df_impact, name="df_impact", cols_required=ut.COLS_AAMUT)
        # Evaluate
        df_eval = eval_substitution_impact(df_impact=df_impact)
        return df_eval
