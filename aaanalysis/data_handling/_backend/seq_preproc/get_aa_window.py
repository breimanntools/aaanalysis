"""
This is a script for the backend of the SequenceProcessor().get_aa_window() method.
"""
import time
import pandas as pd
import numpy as np


# I Helper Functions


# II Main Functions
def get_aa_window(seq=None, pos_start=None, pos_stop=None, window_size=None, gap='-'):
    """Extracts a window of amino acids from a sequence, padding with gaps if necessary."""
    if pos_stop is None:
        pos_stop = pos_start + window_size - 1
    # Calculate the necessary padding if pos_end exceeds sequence length
    seq_length = len(seq)
    if pos_stop >= seq_length:
        seq += gap * (pos_stop - seq_length + 1)
    # Extract the window
    window = seq[pos_start:pos_stop + 1]
    return window

