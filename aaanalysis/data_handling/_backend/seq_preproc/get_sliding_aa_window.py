"""
This is a script for the backend of the SequenceProcessor().get_sliding_aa_window() method.
"""
import pandas as pd

from .get_aa_window import get_aa_window


# I Helper Functions


# II Main Functions
def get_sliding_aa_window(seq=None, slide_start=0, slide_stop=None, window_size=5, gap='-', index1=False):
    """Extracts sliding list_windows of amino acids from a sequence"""
    if slide_stop is None:
        slide_stop = len(seq) - 1
        if index1:
            slide_stop += 1
    n_windows = slide_stop - window_size - slide_start + 1
    list_windows = []
    for start in range(slide_start, slide_start + n_windows + 1):
        # Do not provide index1 again (it will be otherwise two time corrected)
        aa_window = get_aa_window(seq, pos_start=start, window_size=window_size, gap=gap)
        list_windows.append(aa_window)
    return list_windows
