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
    Compute pairwise similarity between two or more sequences.

    The normalized sequence similarity score between two sequences is computed as a fraction of the alignment score
    to the length of the longest sequence. The alignment score is obtained using the :class:`Bio.Align.PairwiseAligner`
    from ´BioPython <https://biopython.org/>´ with default settings.

    Parameters
    ----------
    seq1 : str, optional
        First sequence to align.
    seq2 : str, optional
        Second sequence to align.
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
        DataFrame containing an ``entry`` (unique protein identifier) and an ``sequence`` (protein sequences) column.

    Returns
    -------
    seq_sim : float
        If ``seq1`` and ``seq2`` are provided, returns the sequence similarity between both sequences.
        If ``df_seq`` is provided, returns a DataFrame containing pairwise sequence similarity scores.

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
                    cols_requiered=[ut.COL_SEQ, ut.COL_ENTRY])
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
