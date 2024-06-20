"""
This is a script for ...
"""
import pandas as pd
from Bio.Align import PairwiseAligner
import itertools
import numpy as np

from aaanalysis import utils as ut


# TODO add as proper function
# Main functions
def comp_seq_sim(seq1=None, seq2=None, alignment_mode='global'):
    """
    Compute sequence similarity between two sequences.

    The normalized sequence similarity score between two sequences is computed using the :class:`PairwiseAligner`
    from BioPython with default settings.

    Parameters:
        seq1 (str): First sequence to align.
        seq2 (str): Second sequence to align.
        alignment_mode (str): Type of alignment. 'global' for global alignment, 'local' for local alignment.

    Returns:
        float: Identity score as a fraction of the alignment score to the length of the longest sequence.
    """
    aligner = PairwiseAligner()
    aligner.mode = alignment_mode  # Set alignment mode to global or local
    # Compute the alignment score using default scores
    score = aligner.score(seq1, seq2)
    # Determine the longest sequence length for normalization
    max_length = max(len(seq1), len(seq2))
    # Normalize the score to get the identity as a fraction
    identity = score / max_length * 100
    return identity


def comp_pw_seq_sim(df_seq=None, alignment_mode="global"):
    """Compute pairwise sequence similarity between all sequences from sequence DataFrame"""
    list_seq = df_seq[ut.COL_SEQ].to_list()
    list_ids = df_seq[ut.COL_ENTRY].to_list()
    dict_id_seq = dict(zip(list_ids, list_seq))
    # Initialize an empty DataFrame
    df_pw_sim = pd.DataFrame(index=list_ids, columns=list_ids, dtype=float)
    # Calculate pairwise similarities
    for (id1, id2) in itertools.combinations(list_ids, 2):
        seq1, seq2 = dict_id_seq[id1], dict_id_seq[id2]
        sim_score = comp_seq_sim(seq1=seq1, seq2=seq2, alignment_mode=alignment_mode)
        df_pw_sim.at[id1, id2] = sim_score
        df_pw_sim.at[id2, id1] = sim_score
    # Fill diagonal with 1s for self-similarity
    np.fill_diagonal(df_pw_sim.values, 1)
    df_pw_sim = df_pw_sim.round(4)
    return df_pw_sim


