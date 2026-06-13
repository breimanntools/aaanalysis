"""
This is a script for the backend of the SequenceProcessor().encode_one_hot() method.
"""
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
import numpy as np

from ._utils import pad_sequences


# I Helper Functions


# II Main Functions
def encode_one_hot(list_seq=None, alphabet="ACDEFGHIKLMNPQRSTVWY", gap="-", pad_at="C"):
    """
    One-hot-encode a list of protein sequences into a feature matrix with padding shorter sequences
    with gaps represented as zero vectors.
    """
    # Pad sequences
    padded_sequences = pad_sequences(list_seq, pad_at=pad_at, gap=gap)
    # Create feature names
    max_length = len(padded_sequences[0])
    list_features = [f"{i}{aa}" for i in range(1, max_length+1) for aa in alphabet]
    # One-hot-encoding (vectorized; identical output to the per-residue form). A gap — which
    # the frontend guarantees is not in the alphabet — maps to index -1 and yields an all-zero
    # block; every other residue sets a single 1 at its alphabet index. Column order is
    # position-major then alphabet, matching ``list_features`` and the original flatten.
    num_amino_acids = len(alphabet)
    n_seq = len(padded_sequences)
    char_matrix = np.array([list(seq) for seq in padded_sequences])
    aa_to_index = {aa: i for i, aa in enumerate(alphabet)}
    index_matrix = np.full(char_matrix.shape, -1, dtype=int)
    for aa, i in aa_to_index.items():
        index_matrix[char_matrix == aa] = i
    one_hot = np.zeros((n_seq, max_length, num_amino_acids), dtype=int)
    valid = index_matrix >= 0
    rows, positions = np.nonzero(valid)
    one_hot[rows, positions, index_matrix[valid]] = 1
    feature_matrix = one_hot.reshape(n_seq, max_length * num_amino_acids)
    return feature_matrix, list_features
