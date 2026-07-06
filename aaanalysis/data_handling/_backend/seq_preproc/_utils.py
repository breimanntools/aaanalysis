"""
This is a script for utility sequence padding.
"""


def pad_sequences(sequences, pad_at='C', gap="_", length=None):
    """
    Pads all sequences in the list to a uniform length by adding gaps.

    If ``length`` is ``None`` (default), sequences are padded to the length of the
    longest sequence, preserving the historical behavior. Otherwise, they are padded
    to ``length``; a sequence longer than ``length`` raises a ``ValueError`` because
    padding can only extend (never truncate) a sequence.

    ``pad_at`` places the ``k`` needed gaps at the C-terminus (``'C'``, end), the
    N-terminus (``'N'``, start), or symmetrically (``'both'``): ``floor(k/2)`` gaps at
    the N-terminus and the remainder (``k - floor(k/2)``) at the C-terminus.
    """
    max_length = max(len(seq) for seq in sequences)
    if length is None:
        target_length = max_length
    else:
        if max_length > length:
            raise ValueError(f"'length' ({length}) should be >= the longest sequence length ({max_length})")
        target_length = length
    padded_sequences = []
    for seq in sequences:
        gap_length = target_length - len(seq)
        if pad_at == 'N':
            padded_seq = gap * gap_length + seq
        elif pad_at == 'both':
            n_gap = gap_length // 2
            c_gap = gap_length - n_gap
            padded_seq = gap * n_gap + seq + gap * c_gap
        else:
            padded_seq = seq + gap * gap_length
        padded_sequences.append(padded_seq)
    return padded_sequences
