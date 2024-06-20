"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
from ._utils import pad_sequences

# I Helper Functions


# II Main Functions
def encode_integer(list_seq: List[str] = None,
                   alphabet: str = "ARNDCEQGHILKMFPSTWYV",
                   gap: str = "_",
                   pad_at: Literal["C", "N"] = "C",
                   ) -> np.array:
    """
    Integer-encode a list of protein sequences into a feature matrix, padding shorter sequences
    with gaps represented as zero vectors.

    Parameters:
    ----------
    list_seq : List of str
        List of protein sequences to encode.
    alphabet : str, default='ARNDCEQGHILKMFPSTWYV'
        The alphabet of amino acids used for encoding. The gap character is not part of the alphabet.
    gap : str, default='_'
        The character used to represent gaps in sequences.
    pad_at : Literal['N', 'C'], default='C'
        Specifies where to add the padding:
        'N' for N-terminus (beginning of the sequence),
        'C' for C-terminus (end of the sequence).

    Returns:
    -------
    np.array
        A numpy array where each row represents an encoded sequence, and each column represents a feature.

    """
    # Validate input parameters
    if pad_at not in ['N', 'C']:
        raise ValueError(f"pad_at must be 'N' or 'C', got {pad_at}")

    # Validate if all characters in the sequences are within the given alphabet
    all_chars = set(''.join(list_seq))
    if not all_chars.issubset(set(alphabet + gap)):
        invalid_chars = all_chars - set(alphabet + gap)
        raise ValueError(f"Found invalid amino acid(s) {invalid_chars} not in alphabet.")

    # Map amino acids to integers
    aa_to_int = {aa: idx + 1 for idx, aa in enumerate(alphabet)}
    aa_to_int[gap] = 0

    # Pad sequences
    padded_sequences = pad_sequences(list_seq, pad_at=pad_at)

    # Create integer encoding
    max_length = len(padded_sequences[0])
    feature_matrix = np.zeros((len(padded_sequences), max_length), dtype=int)
    for idx, seq in enumerate(padded_sequences):
        encoded_seq = [aa_to_int[aa] for aa in seq]
        feature_matrix[idx, :] = encoded_seq

    return feature_matrix


