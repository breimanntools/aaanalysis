"""
This is a script for the backend of the SequenceProcessor().encode_one_hot() method.
"""
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
import numpy as np

from ._utils import pad_sequences


# I Helper Functions
def _one_hot_encode(amino_acid=None, alphabet=None, gap="_"):
    """
    Encodes a single amino acid into a one-hot vector based on a specified alphabet.
    Returns a zero vector for gaps represented as '_'.
    """
    dict_aa_index = {aa: i for i, aa in enumerate(alphabet)}
    vector = np.zeros(len(alphabet), dtype=int)
    if amino_acid != gap:
        vector[dict_aa_index[amino_acid]] = 1
    return vector


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
    # Create one-hot-encoding
    num_amino_acids = len(alphabet)
    feature_matrix = np.zeros((len(padded_sequences), max_length * num_amino_acids), dtype=int)
    args = dict(alphabet=alphabet, gap=gap)
    for idx, seq in enumerate(padded_sequences):
        encoded_seq = [_one_hot_encode(amino_acid=aa, **args) for aa in seq]
        feature_matrix[idx, :] = np.array(encoded_seq).flatten()
    return feature_matrix, list_features

