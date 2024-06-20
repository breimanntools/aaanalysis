"""
This is a script for ...
"""
import pandas as pd

from .get_aa_window import get_aa_window

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
def get_sliding_aa_window(seq: str,
                          slide_start: int,
                          slide_stop: int = None,
                          window_size: int = 5,
                          gap: str = '-',
                          accept_gap: bool = True):
    """
    Extracts sliding list_windows of amino acids from a sequence.

    Parameters:
    ----------
    seq : str
        The protein sequence from which to extract the list_windows.
    slide_start : int
        The starting position for sliding window extraction (1-based index).
    slide_end : int, optional
        The ending position for sliding window extraction (1-based index). If None, extract all possible list_windows.
    window_size : int, default=5
        The size of each window to extract.
    gap : str, default='-'
        The character used to represent gaps.
    accept_gap : bool, default=True
        Whether to accept gaps in the list_windows. If False, list_windows containing gaps are rejected.

    Returns:
    -------
    List[str]
        A list of extracted list_windows of amino acids.
    """
    if slide_stop is None:
        slide_stop = len(seq)
        if not accept_gap:
            slide_stop -= window_size
    list_windows = []
    for start in range(slide_start, slide_stop + 1):
        aa_window = get_aa_window(seq, pos_start=start, window_size=window_size, gap=gap, accept_gap=accept_gap)
        list_windows.append(aa_window)
    return list_windows
