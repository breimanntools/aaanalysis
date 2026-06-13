"""
This is a script for the backend of the SequenceProcessor().get_sliding_aa_window() method.
"""
import pandas as pd


# I Helper Functions


# II Main Functions
def get_sliding_aa_window(seq=None, slide_start=0, slide_stop=None, window_size=5, gap='-', index1=False):
    """Extracts sliding list_windows of amino acids from a sequence"""
    if slide_stop is None:
        slide_stop = len(seq) - 1
        if index1:
            slide_stop += 1
    n_windows = slide_stop - window_size - slide_start + 1
    # Inline strided slice: ``get_aa_window`` re-pads the whole string on every
    # call, so slice + pad-on-overrun directly (byte-identical to that helper for
    # the non-negative starts this loop produces).
    seq_length = len(seq)
    list_windows = []
    for start in range(slide_start, slide_start + n_windows + 1):
        stop = start + window_size - 1
        if stop >= seq_length:
            real = seq[start:seq_length] if start < seq_length else ""
            aa_window = real + gap * max(0, (stop + 1) - max(start, seq_length))
        else:
            aa_window = seq[start:stop + 1]
        list_windows.append(aa_window)
    return list_windows
