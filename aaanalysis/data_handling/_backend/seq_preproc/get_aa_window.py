"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np


# I Helper Functions


# II Main Functions
def get_aa_window(seq: str, pos_start: int, pos_stop: int = None, window_size: int = None, gap: str = '-', accept_gap: bool = True) -> str:
    """
    Extracts a window of amino acids from a sequence, padding with gaps if necessary.

    Parameters:
    ----------
    seq : str
        The protein sequence from which to extract the window.
    pos_start : int
        The starting position of the window (1-based index).
    pos_end : int, optional
        The ending position of the window (1-based index). If None, window_size is used.
    window_size : int, optional
        The size of the window to extract. Only used if pos_end is None.
    gap : str, default='-'
        The character used to represent gaps.
    accept_gap : bool, default=True
        Whether to accept gaps in the window. If False, windows containing gaps are rejected.

    Returns:
    -------
    str
        The extracted window of amino acids, padded with gaps if necessary.

    Raises:
    ------
    ValueError:
        If both pos_end and window_size are None, or if the window contains gaps and accept_gap is False.
    """
    if pos_stop is None and window_size is None:
        raise ValueError("Either pos_end or window_size must be specified.")

    if pos_stop is None:
        pos_stop = pos_start + window_size - 1

    # Convert 1-based positions to 0-based indices
    pos_start -= 1
    pos_stop -= 1

    # Calculate the necessary padding if pos_end exceeds sequence length
    seq_length = len(seq)
    if pos_stop >= seq_length:
        seq += gap * (pos_stop - seq_length + 1)

    # Extract the window
    window = seq[pos_start:pos_stop + 1]

    if not accept_gap and gap in window:
        raise ValueError("The window contains gaps and accept_gap is set to False.")

    return window

