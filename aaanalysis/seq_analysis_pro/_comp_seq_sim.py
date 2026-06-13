"""
This is a script for the frontend of the comp_seq_sim function to compute sequence similarities.
"""
from typing import Literal, Optional, Union
import pandas as pd

import aaanalysis.utils as ut

from ._backend.comp_seq_sim import comp_seq_sim_, comp_pw_seq_sim_


# Main functions
def comp_seq_sim(seq1: Optional[str] = None,
                 seq2: Optional[str] = None,
                 df_seq: Optional[pd.DataFrame] = None,
                 ) -> Union[float, pd.DataFrame]:
    """
    Compute pairwise similarity between two or more sequences (**[pro]**, requires ``aaanalysis[pro]``).

    The sequence similarity score between two sequences is the alignment score expressed as a percentage of
    the length of the longest sequence (range ``[0, 100]``). The alignment score is obtained using the
    :class:`Bio.Align.PairwiseAligner` from `BioPython <https://biopython.org/>`_ with default settings.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    seq1 : str, optional
        First sequence to align.
    seq2 : str, optional
        Second sequence to align.
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
        DataFrame containing an ``entry`` column with unique protein identifiers
        and a ``sequence`` column with full protein sequences.

    Returns
    -------
    seq_sim : float
        Sequence similarity score as a percentage in ``[0, 100]``, returned when ``seq1`` and ``seq2``
        are provided (pairwise comparison of two strings).
    df_pw_sim : pd.DataFrame, shape (n_samples, n_samples)
        Pairwise similarity matrix with ``entry`` values as both index and columns,
        returned when ``df_seq`` is provided.

    See Also
    --------
    * :class:`Bio.Align.PairwiseAligner` for details on the similarity computation.

    Warnings
    --------
    * This function requires `biopython`, which is automatically installed via `pip install aaanalysis[pro]`.

    Examples
    --------
    .. include:: examples/comp_seq_sim.rst
    """
    # Check input
    if df_seq is None:
        ut.check_str(name="seq1", val=seq1, accept_none=False)
        ut.check_str(name="seq2", val=seq2, accept_none=False)

    else:
        ut.check_df(name="df_seq", df=df_seq, accept_none=False, accept_nan=False,
                    cols_required=[ut.COL_SEQ, ut.COL_ENTRY])
        for entry, seq in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]):
            ut.check_str(name=f"sequence ({entry}", val=seq, accept_none=False)
    if df_seq is None:
        # Compute similarity
        seq_sim = comp_seq_sim_(seq1=seq1, seq2=seq2)
        return seq_sim
    else:
        # Compute pairwise similarities
        df_pw_sim = comp_pw_seq_sim_(df_seq=df_seq)
        return df_pw_sim
