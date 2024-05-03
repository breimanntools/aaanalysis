"""
This is a script for creating one-hot-encoding of sequences used as baseline representation.
"""
import pandas as pd
import numpy as np


# I Helper Functions
def _pad_sequences(sequences, pad_at='C'):
    """
    Pads all sequences in the list to the length of the longest sequence by adding gaps.
    """
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        gap_length = max_length - len(seq)
        if pad_at == 'N':
            padded_seq = '_' * gap_length + seq
        else:
            padded_seq = seq + '_' * gap_length
        padded_sequences.append(padded_seq)
    return padded_sequences


def _one_hot_encode(amino_acid=None, alphabet=None, gap="_"):
    """
    Encodes a single amino acid into a one-hot vector based on a specified alphabet.
    Returns a zero vector for gaps represented as '_'.
    """
    index_dict = {aa: i for i, aa in enumerate(alphabet)}
    vector = np.zeros(len(alphabet), dtype=int)
    if amino_acid != gap:
        if amino_acid in index_dict:
            vector[index_dict[amino_acid]] = 1
        else:
            raise ValueError(f"Unrecognized amino acid '{amino_acid}' not in alphabet.")
    return vector


# II Main Functions
# TODO finish, docu, test, example ..
def encode_seq(list_seq=None,
               alphabet='ARNDCEQGHILKMFPSTWYV',
               gap="_",
               pad_at='C'):
    """
    One-hot-encode a list of protein sequences into a feature matrix, padding shorter sequences
    with gaps represented as zero vectors.

    Parameters:
    ----------
    sequences : list of str
        List of protein sequences to encode.
    alphabet : str, default='ARNDCEQGHILKMFPSTWYV'
        The alphabet of amino acids used for encoding. The gap character is not part of the alphabet.
    gap : str, default='_'
        The character used to represent gaps in sequences.
    pad_at : str, default='C'
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
    if not all_chars.issubset(set(alphabet + '_')):
        invalid_chars = all_chars - set(alphabet + '_')
        raise ValueError(f"Found invalid amino acid(s) {invalid_chars} not in alphabet.")

    # Pad sequences
    padded_sequences = _pad_sequences(list_seq, pad_at=pad_at)
    max_length = len(padded_sequences[0])
    num_amino_acids = len(alphabet)
    feature_matrix = np.zeros((len(padded_sequences), max_length * num_amino_acids), dtype=int)
    args = dict(alphabet=alphabet, gap=gap)
    # Create one-hot-encoding
    for idx, seq in enumerate(padded_sequences):
        encoded_seq = [_one_hot_encode(amino_acid=aa, **args) for aa in seq]
        feature_matrix[idx, :] = np.array(encoded_seq).flatten()

    return feature_matrix

