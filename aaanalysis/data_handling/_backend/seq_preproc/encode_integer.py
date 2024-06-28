"""
This is a script for the backend of the SequenceProcessor().encode_integer() method.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
from ._utils import pad_sequences

# I Helper Functions


# II Main Functions
def encode_integer(list_seq=None, alphabet="ACDEFGHIKLMNPQRSTVWY", gap="-", pad_at="C"):
    """
    Integer-encode a list of protein sequences into a feature matrix, padding shorter sequences
    with gaps represented as zero vectors.
    """
    # Map amino acids to integers
    aa_to_int = {aa: idx + 1 for idx, aa in enumerate(alphabet)}
    aa_to_int[gap] = 0

    # Pad sequences
    padded_sequences = pad_sequences(list_seq, pad_at=pad_at, gap=gap)
    # Create feature names
    max_length = len(padded_sequences[0])
    list_features = [f"P{i}" for i in range(1, max_length+1)]
    # Create integer encoding
    feature_matrix = np.zeros((len(padded_sequences), max_length), dtype=int)
    for idx, seq in enumerate(padded_sequences):
        encoded_seq = [aa_to_int[aa] for aa in seq]
        feature_matrix[idx, :] = encoded_seq
    return feature_matrix, list_features


