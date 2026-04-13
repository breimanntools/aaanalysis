"""
This is a script for the frontend of the AAlogo class for computing sequence logo matrices
and conservation scores.
"""
from typing import Optional, Literal
import pandas as pd

import aaanalysis.utils as ut
from ._backend._aalogo.aalogo import get_df_logo_, get_df_logo_info_, get_conservation_


# I Helper Functions
def check_df_logo_info(df_logo_info=None):
    """Check if df_logo_info is a valid pd.Series with index name 'pos'."""
    ut.check_df(name="df_logo_info", df=df_logo_info,
                check_series=True, accept_none=False, accept_nan=False)
    if df_logo_info.index.name != "pos":
        raise ValueError(f"'df_logo_info' index name must be 'pos', got '{df_logo_info.index.name}'.")


def check_match_df_parts_logo_parts(df_parts=None):
    """Check that df_parts contains at least one of the required sequence part columns."""
    list_parts = [x for x in ut.COLS_SEQ_PARTS if x in list(df_parts)]
    if len(list_parts) == 0:
        raise ValueError(f"'df_parts' should contain at least one of the following parts: {ut.COLS_SEQ_PARTS}")
    return list_parts


def check_tmd_len(tmd_len=None, df_parts=None):
    """Check if tmd_len is valid and does not exceed the maximum TMD length in df_parts."""
    ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True, accept_none=True)
    if tmd_len is not None and ut.COL_TMD in list(df_parts):
        max_tmd_len = df_parts[ut.COL_TMD].apply(len).max()
        if tmd_len > max_tmd_len:
            raise ValueError(f"'tmd_len' ({tmd_len}) should not exceed the maximum TMD length "
                             f"in 'df_parts' ({max_tmd_len}).")


def check_pseudocount(pseudocount=None):
    """Check if pseudocount is a non-negative float."""
    ut.check_number_range(name="pseudocount", val=pseudocount, min_val=0,
                          just_int=False, accept_none=False)


def check_characters_to_ignore(characters_to_ignore=None):
    """Check if characters_to_ignore is a string."""
    ut.check_str(name="characters_to_ignore", val=characters_to_ignore, accept_none=False)


# II Main Functions
# TODO adjust examples, add in index sequence analys
class AAlogo:
    """
    Amino Acid Logo (**AAlogo**) class for computing sequence logo matrices and conservation scores.

    Sequence logos visualize the amino acid composition and conservation at each residue position.
    AAlogo computes logo matrices from sequence parts using the
    `logomaker <https://logomaker.readthedocs.io/en/latest/>`_ package.

    """

    def __init__(self,
                 logo_type: Literal["probability", "weight", "counts", "information"] = "probability"
                 ):
        """
        Parameters
        ----------
        logo_type : {'probability', 'weight', 'counts', 'information'}, default='probability'
            Type of sequence logo encoding:

            - ``probability``: Normalized probability distribution of amino acids per position.
            - ``weight``: Weighted (log-odds) representation.
            - ``counts``: Raw amino acid counts per position.
            - ``information``: Information content in bits per position.

        See Also
        --------
        * :class:`AAlogoPlot`: the respective plotting class.
        * `logomaker <https://logomaker.readthedocs.io/en/latest/>`_: the underlying logo computation package.
        """
        list_logo_types = ["probability", "weight", "counts", "information"]
        ut.check_str_options(name="logo_type", val=logo_type, list_str_options=list_logo_types)
        self._logo_type = logo_type

    def get_df_logo(self,
                    df_parts: pd.DataFrame = None,
                    labels: Optional[ut.ArrayLike1D] = None,
                    label_test: int = 1,
                    tmd_len: Optional[int] = None,
                    start_n: bool = True,
                    characters_to_ignore: str = ".-",
                    pseudocount: float = 0.0,
                    ) -> pd.DataFrame:
        """
        Compute a sequence logo matrix for the provided sequence parts.

        For each residue position, the relative frequency (or another encoding) of each amino acid
        is computed across all sequences. If variable-length TMD sequences are provided, they are
        aligned to a uniform length via N- or C-terminal padding before computing the logo.

        Parameters
        ----------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            Sequence parts DataFrame with at least one column from the standard parts:
            ``jmd_n``, ``tmd``, ``jmd_c``. Must be a valid parts DataFrame.
        labels : array-like, shape (n_samples,), optional
            Class labels for samples in ``df_parts``. If provided, only samples with
            ``label_test`` are included in the logo computation.
        label_test : int, default=1
            Class label of the test group to select from ``labels``.
        tmd_len : int, optional
            Fixed length (>=1) to align all TMD sequences. If ``None``, the maximum TMD
            length in ``df_parts`` is used. Only relevant if ``tmd`` column is present.
        start_n : bool, default=True
            Alignment direction for variable-length TMDs:

            - ``True``: Align from N-terminus (C-terminal padding with gaps).
            - ``False``: Align from C-terminus (N-terminal padding with gaps).

        characters_to_ignore : str, default='.-'
            Characters excluded from the logo matrix computation.
        pseudocount : float, default=0.0
            Pseudocount (>=0) added to all amino acid counts to avoid log(0) issues.

        Returns
        -------
        df_logo : pd.DataFrame, shape (n_positions, n_amino_acids)
            Logo matrix with residue positions as rows and amino acids as columns.

        See Also
        --------
        * :meth:`AAlogo.get_df_logo_info`: for per-position information content.
        * `logomaker.alignment_to_matrix <https://logomaker.readthedocs.io/en/latest/>`_:
          the underlying matrix computation function.

        Examples
        --------
        .. include:: examples/aalogo_get_df_logo.rst
        """
        # Check input
        ut.check_df_parts(df_parts=df_parts)
        list_parts = check_match_df_parts_logo_parts(df_parts=df_parts)
        ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
        if labels is not None:
            n_samples = len(df_parts)
            labels = ut.check_labels(labels=labels, len_required=n_samples)
        check_tmd_len(tmd_len=tmd_len, df_parts=df_parts)
        ut.check_bool(name="start_n", val=start_n, accept_none=False)
        check_characters_to_ignore(characters_to_ignore=characters_to_ignore)
        check_pseudocount(pseudocount=pseudocount)
        # Filter by label
        _df_parts = df_parts[list_parts].copy()
        if labels is not None:
            _df_parts = _df_parts[labels == label_test]
        if len(_df_parts) == 0:
            raise ValueError(f"No samples remaining after filtering by 'label_test' ({label_test}). "
                             f"Check that 'labels' contains this value.")
        # Compute logo matrix
        df_logo = get_df_logo_(df_parts=_df_parts, logo_type=self._logo_type,
                               tmd_len=tmd_len, start_n=start_n,
                               characters_to_ignore=characters_to_ignore,
                               pseudocount=pseudocount)
        return df_logo

    def get_df_logo_info(self,
                         df_parts: pd.DataFrame = None,
                         labels: Optional[ut.ArrayLike1D] = None,
                         label_test: int = 1,
                         tmd_len: Optional[int] = None,
                         start_n: bool = True,
                         characters_to_ignore: str = ".-",
                         pseudocount: float = 0.0,
                         ) -> pd.Series:
        """
        Compute per-position information content (in bits) from sequence parts.

        Information content is computed using the information logo type regardless of the
        ``logo_type`` set during initialization. The result reflects sequence conservation:
        higher values indicate stronger conservation at that position.

        Parameters
        ----------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            Sequence parts DataFrame with at least one column from the standard parts:
            ``jmd_n``, ``tmd``, ``jmd_c``.
        labels : array-like, shape (n_samples,), optional
            Class labels for samples in ``df_parts``. If provided, only samples with
            ``label_test`` are included.
        label_test : int, default=1
            Class label of the test group to select from ``labels``.
        tmd_len : int, optional
            Fixed length (>=1) to align all TMD sequences. If ``None``, the maximum TMD
            length in ``df_parts`` is used.
        start_n : bool, default=True
            Alignment direction for variable-length TMDs.
        characters_to_ignore : str, default='.-'
            Characters excluded from the logo matrix computation.
        pseudocount : float, default=0.0
            Pseudocount (>=0) added to all amino acid counts.

        Returns
        -------
        df_logo_info : pd.Series, shape (n_positions,)
            Per-position information content in bits, with index named 'pos'.
            Values range from 0 (no conservation) to ~4.248 (fully conserved).

        See Also
        --------
        * :meth:`AAlogo.get_conservation`: to summarize the per-position scores into a single value.
        * :meth:`AAlogo.get_df_logo`: for the full logo matrix.

        Examples
        --------
        .. include:: examples/aalogo_get_df_logo_info.rst
        """
        # Check input
        ut.check_df_parts(df_parts=df_parts)
        list_parts = check_match_df_parts_logo_parts(df_parts=df_parts)
        ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
        if labels is not None:
            n_samples = len(df_parts)
            labels = ut.check_labels(labels=labels, len_required=n_samples)
        check_tmd_len(tmd_len=tmd_len, df_parts=df_parts)
        ut.check_bool(name="start_n", val=start_n, accept_none=False)
        check_characters_to_ignore(characters_to_ignore=characters_to_ignore)
        check_pseudocount(pseudocount=pseudocount)
        # Filter by label
        _df_parts = df_parts[list_parts].copy()
        if labels is not None:
            _df_parts = _df_parts[labels == label_test]
        if len(_df_parts) == 0:
            raise ValueError(f"No samples remaining after filtering by 'label_test' ({label_test}). "
                             f"Check that 'labels' contains this value.")
        # Compute information content
        df_logo_info = get_df_logo_info_(df_parts=_df_parts, tmd_len=tmd_len, start_n=start_n,
                                         characters_to_ignore=characters_to_ignore,
                                         pseudocount=pseudocount)
        return df_logo_info

    @staticmethod
    def get_conservation(df_logo_info: pd.Series = None,
                         value_type: Literal["min", "mean", "median", "max"] = "mean",
                         ) -> float:
        """
        Summarize per-position information content into a single conservation score.

        Aggregates the per-position information content from :meth:`AALogo.get_df_logo_info`
        into a single scalar value representing overall sequence conservation.

        Parameters
        ----------
        df_logo_info : pd.Series, shape (n_positions,)
            Per-position information content with index name 'pos', as returned by
            :meth:`AAlogo.get_df_logo_info`.
        value_type : {'min', 'mean', 'median', 'max'}, default='mean'
            Aggregation method:

            - ``min``: Minimum conservation across all positions.
            - ``mean``: Average conservation across all positions.
            - ``median``: Median conservation across all positions.
            - ``max``: Maximum conservation at any single position.

        Returns
        -------
        cons_val : float
            Conservation score ranging from 0 (no conservation) to ~4.248 (fully conserved).

        Notes
        -----
        * The maximum theoretical information content per position is log2(20) ≈ 4.248 bits,
          corresponding to a completely conserved amino acid.
        * Use ``value_type='mean'`` for an overall conservation estimate and
          ``value_type='max'`` to identify the most conserved position.

        See Also
        --------
        * :meth:`AAlogo.get_df_logo_info`: to compute the per-position information content.

        Examples
        --------
        .. include:: examples/aalogo_get_conservation.rst
        """
        # Check input
        check_df_logo_info(df_logo_info=df_logo_info)
        list_value_types = ["min", "mean", "median", "max"]
        ut.check_str_options(name="value_type", val=value_type, list_str_options=list_value_types)
        # Compute conservation
        cons_val = get_conservation_(df_logo_info=df_logo_info, value_type=value_type)
        return cons_val
