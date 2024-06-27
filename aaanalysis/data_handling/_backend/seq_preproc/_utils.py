"""
This is a script for utility sequence padding.
"""


def pad_sequences(sequences, pad_at='C', gap="_"):
    """
    Pads all sequences in the list to the length of the longest sequence by adding gaps.
    """
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        gap_length = max_length - len(seq)
        if pad_at == 'N':
            padded_seq = gap * gap_length + seq
        else:
            padded_seq = seq + gap * gap_length
        padded_sequences.append(padded_seq)
    return padded_sequences
