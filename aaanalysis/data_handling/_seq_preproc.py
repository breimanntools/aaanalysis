"""
This is a script for the frontend of the SequencePreprocessor class,
a supportive class for preprocessing protein sequences.
"""
from typing import Optional, Union, List, Literal, Tuple
import numpy as np

import aaanalysis.utils as ut
from ._backend.seq_preproc.encode_one_hot import encode_one_hot
from ._backend.seq_preproc.encode_integer import encode_integer
from ._backend.seq_preproc.get_aa_window import get_aa_window
from ._backend.seq_preproc.get_sliding_aa_window import get_sliding_aa_window


# I Helper Functions
def check_gap(gap="_"):
    """Check if string of length one"""
    ut.check_str(name="gap", val=gap, accept_none=False)
    if len(gap) != 1:
        raise ValueError(f"'gap' ('{gap}') should be a single character.")


# Encoding check functions
def check_match_list_seq_alphabet(list_seq=None, alphabet=None, gap="-"):
    """Validate if all characters in the sequences are within the given alphabet"""
    all_chars = set(''.join(list_seq))
    if not all_chars.issubset(set(alphabet + gap)):
        invalid_chars = all_chars - set(alphabet + gap)
        raise ValueError(f"Following amino acid(s) from 'list_seq' are not in 'alphabet': {invalid_chars}")


def check_match_gap_alphabet(gap="_", alphabet=None):
    """Check that gap is not in alphabet"""
    if gap in alphabet:
        raise ValueError(f"'gap' ('{gap}') should not be contained in the 'alphabet' ('{alphabet}')")


# Window size check functions
def adjust_positions(start=None, stop=None, index1=False):
    """Adjust positions depending on indexing mode"""
    if index1:
        start -= 1
        if stop is not None:
            stop -= 1
    return start, stop


def check_match_pos_start_pos_stop(pos_start=None, pos_stop=None):
    """Check if start position smaller than stop position"""
    if pos_stop is not None and pos_start > pos_stop:
        raise ValueError(f"'pos_start' ({pos_start}) should be smaller than 'pos_stop' ({pos_stop})")


def check_match_pos_stop_window_size(pos_stop=None, window_size=None):
    """Check if one is given"""
    if pos_stop is None and window_size is None:
        raise ValueError("Either 'pos_end' or 'window_size' must be specified. Both are 'None'.")
    if pos_stop is not None and window_size is not None:
        raise ValueError(f"Either 'pos_end' ({pos_stop}) or 'window_size' ({window_size}) must be specified."
                         f" Both are given.")


def check_match_seq_pos(seq=None, pos_start=None, pos_stop=None):
    """Check if pos_start matches length of sequence"""
    seq_len = len(seq)
    if pos_start >= seq_len:
        raise ValueError(f"'pos_start' ({pos_start}) must be smaller than the sequence length ({seq_len})")
    if pos_stop is not None and pos_stop >= seq_len:
        raise ValueError(f"'pos_stop' ({pos_stop}) must be smaller than the sequence length ({seq_len})")


def check_match_seq_pos_start_window_size(seq=None, pos_start=None, window_size=None):
    """Check if start position and window size do not extend the sequence length"""
    if window_size is not None:
        seq_len = len(seq)
        pos_stop = pos_start + window_size
        if pos_stop > seq_len:
            raise ValueError(f"'pos_start' ({pos_start}) + 'window_size' ({window_size}) should be >= "
                             f"the sequence length ({seq_len})")


# Sliding window check functions
def check_match_slide_start_slide_stop(slide_start=None, slide_stop=None):
    """Check if start sliding position smaller than stop position"""
    if slide_stop is not None and slide_start > slide_stop:
        raise ValueError(f"'slide_start' ({slide_start}) should be smaller than 'slide_stop' ({slide_stop})")


def check_match_slide_start_slide_stop_window_size(slide_start=None, slide_stop=None, window_size=None):
    """Check if one is given"""
    if slide_stop is not None:
        min_window_size = slide_stop - slide_stop
        if window_size < min_window_size:
            raise ValueError(f"'window_size' ('{window_size}') should be smaller then the distance ({min_window_size})"
                             f" between 'slide_start' ('{slide_start}') and 'slide_stop' ({slide_stop}).")


def check_match_seq_slide(seq=None, slide_start=None, slide_stop=None):
    """Check if slide_start matches length of sequence"""
    seq_len = len(seq)
    if slide_start >= seq_len:
        raise ValueError(f"'slide_start' ({slide_start}) must be smaller than the sequence length ({seq_len})")
    if slide_stop is not None and slide_stop >= seq_len:
        raise ValueError(f"'slide_stop' ({slide_stop}) must be smaller than the sequence length ({seq_len})")


def check_match_seq_slide_start_window_size(seq=None, slide_start=None, window_size=None):
    """Check if start position and window size do not extend the sequence length"""
    seq_len = len(seq)
    slide_stop = slide_start + window_size
    if slide_stop > seq_len:
        raise ValueError(f"'slide_start' ({slide_start}) + 'window_size' ({window_size}) should be >= "
                         f"the sequence length ({seq_len})")


# II Main Functions
class SequencePreprocessor:
    """
    Utility data preprocessing class to encode and represent protein sequences.
    """

    # Sequence encoding
    @staticmethod
    def encode_one_hot(list_seq: Union[List[str], str] = None,
                       alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
                       gap: str = "-",
                       pad_at: Literal["C", "N"] = "C",
                       ) -> Tuple[np.ndarray, List[str]]:
        """
        One-hot-encode a list of protein sequences into a feature matrix.

        Each residue is represented by a binary vector of length equal to the alphabet size.
        For each sequence position, the amino acid is set to 1 in its corresponding position in the vector,
        while all other positions are set to 0. Gaps are represented by zero vectors. Shorter sequences are
        padded with gaps either N- or C-terminally.

        Parameters
        ----------
        list_seq : list of str or str
            List of protein sequences to encode. All characters in each sequence must part of the ``alphabet`` or
            be represented by the ``gap``.
        alphabet : str, default='ACDEFGHIKLMNPQRSTVWY'
            The alphabet of amino acids used for encoding.
        gap : str, default='-'
            The character used to represent gaps within sequences. It should not be included in the ``alphabet``.
        pad_at : str, default='C'
            Specifies where to add the padding:

            - 'N' for N-terminus (beginning of the sequence),
            - 'C' for C-terminus (end of the sequence).

        Returns
        -------
        X: array-like, shape (n_samples, n_residues*n_characters)
            Feature matrix containing one-hot encoded position-wise representation of residues.
        features : list of str
            List of feature names corresponding to each position and amino acid in the encoded matrix.

        Examples
        --------
        .. include:: examples/sp_encode_one_hot.rst
        """
        # Check input
        list_seq = ut.check_list_like(name="list_seq", val=list_seq,
                                      check_all_str_or_convertible=True,
                                      accept_none=False, accept_str=True)
        ut.check_str(name="alphabet", val=alphabet, accept_none=False)
        check_gap(gap=gap)
        ut.check_str_options(name="pad_at", val=pad_at, list_str_options=["N", "C"])
        check_match_gap_alphabet(gap=gap, alphabet=alphabet)
        check_match_list_seq_alphabet(list_seq=list_seq, alphabet=alphabet, gap=gap)
        # Create encoding
        X, features = encode_one_hot(list_seq=list_seq, alphabet=alphabet, gap=gap, pad_at=pad_at)
        return X, features

    @staticmethod
    def encode_integer(list_seq: Union[List[str], str] = None,
                       alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
                       gap: str = "-",
                       pad_at: Literal["C", "N"] = "C",
                       ) -> Tuple[np.ndarray, List[str]]:
        """
        Integer-encode a list of protein sequences into a feature matrix.

        Each amino acid is represented by an integer between 1 and n, where n is the number of characters.
        Gaps are represented by 0. Shorter sequences are padded with gaps either N- or C-terminally.

        Parameters
        ----------
        list_seq : list of str or str
            List of protein sequences to encode. All characters in each sequence must part of the ``alphabet`` or
            be represented by the ``gap``.
        alphabet : str, default='ACDEFGHIKLMNPQRSTVWY'
            The alphabet of amino acids used for encoding.
        gap : str, default='-'
            The character used to represent gaps within sequences. It should not be included in the ``alphabet``.
        pad_at : str, default='C'
            Specifies where to add the padding:

            - 'N' for N-terminus (beginning of the sequence),
            - 'C' for C-terminus (end of the sequence).

        Returns
        -------
        X: array-like, shape (n_samples, n_residues)
            Feature matrix containing one-hot encoded position-wise representation of residues.
        features : list of str
            List of feature names corresponding to each position in the encoded matrix.

        Examples
        --------
        .. include:: examples/sp_encode_integer.rst
        """
        # Check input
        list_seq = ut.check_list_like(name="list_seq", val=list_seq,
                                      check_all_str_or_convertible=True,
                                      accept_none=False, accept_str=True)
        ut.check_str(name="alphabet", val=alphabet, accept_none=False)
        check_gap(gap=gap)
        ut.check_str_options(name="pad_at", val=pad_at, list_str_options=["N", "C"])
        check_match_gap_alphabet(gap=gap, alphabet=alphabet)
        check_match_list_seq_alphabet(list_seq=list_seq, alphabet=alphabet, gap=gap)
        # Create encoding
        X, features = encode_integer(list_seq=list_seq, alphabet=alphabet, gap=gap, pad_at=pad_at)
        return X, features

    @staticmethod
    def get_aa_window(seq: str = None,
                      pos_start: int = 0,
                      pos_stop: Optional[int] = None,
                      window_size: Optional[int] = None,
                      index1: bool = False,
                      gap: str = '-',
                      accept_gap: bool = True,
                      ) -> str:
        """
        Extracts a window of amino acids from a sequence.

        This window starts from a given start position (``pos_start``) and stops either at a defined
        stop position (``pos_stop``) or after a number of residues defined by ``window_size``.

        Parameters
        ----------
        seq : str
            The protein sequence from which to extract the window.
        pos_start : int, default=0
            The starting position (>=0) of the window.
        pos_stop : int, optional
            The ending position (>=``pos_start``) of the window. If ``None``, ``window_size`` is used to determine it.
        window_size : int, optional
            The size of the window (>=1) to extract. Only used if ``pos_stop`` is ``None``.
        index1 : bool, default=False
            Whether position index starts at 1 (if ``True``) or 0 (if ``False``),
            where the first amino acid is at position 1 or 0, respectively.
        gap : str, default='-'
            The character used to represent gaps.
        accept_gap : bool, default=True
            Whether to accept gaps in the window. If ``True``, C-terminally padding is enabled.

        Returns
        -------
        window : str
            The extracted window of amino acids.

        Notes
        -----
        * A ``ValueError`` is raised if both ``pos_stop`` and ``window_size`` are ``None`` or if both are provided.

        Examples
        --------
        .. include:: examples/sp_get_aa_window.rst
        """
        # Check input
        ut.check_str(name="seq", val=seq, accept_none=False)
        ut.check_bool(name="index1", val=index1, accept_none=False)
        min_val_pos = 1 if index1 else 0
        str_add = f"If 'index1' is '{index1}'."
        ut.check_number_range(name="pos_start", val=pos_start, min_val=min_val_pos,
                              accept_none=False, just_int=True, str_add=str_add)
        ut.check_number_range(name="pos_stop", val=pos_stop, min_val=min_val_pos,
                              accept_none=True, just_int=True, str_add=str_add)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, accept_none=True, just_int=True)
        check_gap(gap=gap)
        ut.check_bool(name="accept_gap", val=accept_gap, accept_none=False)
        pos_start, pos_stop = adjust_positions(start=pos_start, stop=pos_stop, index1=index1)
        check_match_pos_start_pos_stop(pos_start=pos_start, pos_stop=pos_stop)
        check_match_pos_stop_window_size(pos_stop=pos_stop, window_size=window_size)
        if not accept_gap:
            check_match_seq_pos(seq=seq, pos_start=pos_start, pos_stop=pos_stop)
            check_match_seq_pos_start_window_size(seq=seq, pos_start=pos_start, window_size=window_size)
        # Get amino acid window
        window = get_aa_window(seq=seq, pos_start=pos_start, pos_stop=pos_stop,
                               window_size=window_size, gap=gap)
        return window

    @staticmethod
    def get_sliding_aa_window(seq: str = None,
                              slide_start: int = 0,
                              slide_stop: Optional[int] = None,
                              window_size: int = 5,
                              index1: bool = False,
                              gap: str = '-',
                              accept_gap: bool = True
                              ) -> List[str]:
        """
        Extract sliding windows of amino acids from a sequence.

        Parameters
        ----------
        seq : str
            The protein sequence from which to extract the windows.
        slide_start : int, default=0
            The starting position (>=0) for sliding window extraction.
        slide_stop : int, optional
            The ending position (>=1) for sliding window extraction. If ``None``, extract all possible windows.
        window_size : int, default=5
            The size of each window (>=1) to extract.
        index1 : bool, default=False
            Whether position index starts at 1 (if ``True``) or 0 (if ``False``),
            where first amino acid is at position 1 or 0, respectively.
        gap : str, default='-'
            The character used to represent gaps.
        accept_gap : bool, default=True
            Whether to accept gaps in the window. If ``True``, C-terminally padding is enabled.

        Returns
        -------
        list_windows : list of str
            A list of extracted windows of amino acids.

        Examples
        --------
        .. include:: examples/sp_get_sliding_aa_window.rst
        """
        # Check input
        ut.check_str(name="seq", val=seq, accept_none=False)
        ut.check_bool(name="index1", val=index1, accept_none=False)
        min_val_pos = 1 if index1 else 0
        str_add = f"If 'index1' is '{index1}'."
        ut.check_number_range(name="slide_start", val=slide_start, min_val=min_val_pos,
                              accept_none=False, just_int=True, str_add=str_add)
        ut.check_number_range(name="slide_stop", val=slide_stop, min_val=min_val_pos,
                              accept_none=True, just_int=True, str_add=str_add)
        ut.check_number_range(name="window_size", val=window_size, min_val=1, accept_none=False, just_int=True)
        check_gap(gap=gap)
        ut.check_bool(name="accept_gap", val=accept_gap, accept_none=False)
        slide_start, slide_stop = adjust_positions(start=slide_start, stop=slide_stop, index1=index1)
        check_match_slide_start_slide_stop(slide_start=slide_start, slide_stop=slide_stop)
        check_match_slide_start_slide_stop_window_size(slide_start=slide_start, slide_stop=slide_stop,
                                                       window_size=window_size)
        if not accept_gap:
            check_match_seq_slide(seq=seq, slide_start=slide_start, slide_stop=slide_stop)
            check_match_seq_slide_start_window_size(seq=seq, slide_start=slide_start, window_size=window_size)
        # Get sliding windows
        list_windows = get_sliding_aa_window(seq=seq, slide_start=slide_start, slide_stop=slide_stop,
                                             window_size=window_size, gap=gap)
        return list_windows
