"""
This is a script for the frontend of the SequencePreprocessor class,
a supportive class for preprocessing protein sequences.
"""
from typing import Optional, Union, List, Literal
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.seq_preproc.encode_one_hot import encode_one_hot
from ._backend.seq_preproc.encode_integer import encode_integer
from ._backend.seq_preproc.get_aa_window import get_aa_window
from ._backend.seq_preproc.get_sliding_aa_window import get_sliding_aa_window

# I Helper Functions


# II Main Functions
# TODO finish SequencePreprocessor, test, docu
# TODO manage aaanalysis[pro] (add info/warning in docu for every function/module whose dependencies are not installed)
# TODO e.g., seq_filter, comp_seq_sim, SHAP ...
class SequencePreprocessor:
    """
    This class provides methods for preprocessing protein sequences, including encoding and window extraction.
    """

    @staticmethod
    def encode_one_hot(sequence: List[str],
                       alphabet: str = "ARNDCEQGHILKMFPSTWYV",
                       gap: str = "_"
                       ) -> np.ndarray:
        """
        One-hot encodes a list of protein sequences into a feature matrix.

        Parameters
        ----------
        sequence : List[str]
            List of protein sequences to encode.
        alphabet : str, default='ARNDCEQGHILKMFPSTWYV'
            The alphabet of amino acids used for encoding.
        gap : str, default='_'
            The character used to represent gaps in sequences.

        Returns
        -------
        np.ndarray
            A numpy array where each row represents an encoded sequence.
        """
        return encode_one_hot(sequence, alphabet, gap)

    @staticmethod
    def encode_integer(list_seq: List[str], alphabet: str = "ARNDCEQGHILKMFPSTWYV", gap: str = "_", pad_at: Literal["C", "N"] = "C") -> np.ndarray:
        """
        Integer encodes a list of protein sequences into a feature matrix.

        Parameters
        ----------
        list_seq : List[str]
            List of protein sequences to encode.
        alphabet : str, default='ARNDCEQGHILKMFPSTWYV'
            The alphabet of amino acids used for encoding.
        gap : str, default='_'
            The character used to represent gaps in sequences.
        pad_at : Literal['C', 'N'], default='C'
            Specifies where to add the padding.

        Returns
        -------
        np.ndarray
            A numpy array where each row represents an encoded sequence.
        """
        return encode_integer(list_seq, alphabet, gap, pad_at)

    @staticmethod
    def get_aa_window(seq: str, pos_start: int, pos_stop: int = None, window_size: int = None, gap: str = '-', accept_gap: bool = True) -> str:
        """
        Extracts a window of amino acids from a sequence.

        Parameters
        ----------
        seq : str
            The protein sequence from which to extract the window.
        pos_start : int
            The starting position of the window (1-based index).
        pos_stop : int, optional
            The ending position of the window (1-based index). If None, window_size is used.
        window_size : int, optional
            The size of the window to extract. Only used if pos_end is None.
        gap : str, default='-'
            The character used to represent gaps.
        accept_gap : bool, default=True
            Whether to accept gaps in the window.

        Returns
        -------
        str
            The extracted window of amino acids.
        """
        return get_aa_window(seq, pos_start, pos_stop, window_size, gap, accept_gap)

    @staticmethod
    def get_sliding_aa_window(seq: str = None,
                              slide_start: int = 1,
                              slide_stop: int = None,
                              window_size: int = 10,
                              gap: str = '-',
                              accept_gap: bool = False
                              ) -> List[str]:
        """
        Extract sliding windows of amino acids from a sequence.

        Parameters
        ----------
        seq : str
            The protein sequence from which to extract the windows.
        slide_start : int, default=1
            The starting position for sliding window extraction (1-based index).
        slide_stop : int, optional
            The ending position for sliding window extraction (1-based index). If None, extract all possible windows.
        window_size : int, default=10
            The size of each window to extract.
        gap : str, default='-'
            The character used to represent gaps.
        accept_gap : bool, default=False
            Whether to accept gaps in the amino acid windows.

        Returns
        -------
        List[str]
            A list of extracted windows of amino acids.
        """
        # Check input
        ut.check_str(name="seq", val=seq, accept_none=False)
        ut.check_number_val(name="slide_start", val=slide_start, accept_none=False, just_int=True)
        ut.check_number_val(name="slide_stop", val=slide_stop, accept_none=True, just_int=True)

        # Get sliding windows
        list_windows = get_sliding_aa_window(seq, slide_start, slide_stop, window_size, gap, accept_gap)
        return list_windows