"""
Script for (backend class) Split objects used to fragment sequence parts into distinct segments or patterns.
"""
import numpy as np
import itertools

import aaanalysis.utils as ut


# Pattern helper functions
def _get_pattern_pos(steps=None, repeat=2, len_max=12):
    """Get all possible positions from steps with number of repeats and maximum length using itertools"""
    # Generate Cartesian product of steps
    list_steps = np.array([s for s in itertools.product(steps, repeat=repeat)])
    # Get all possible pattern positions
    list_pattern_pos = []
    for s in list_steps:
        p = np.cumsum(s)
        max_p = np.max(p)
        min_p = np.min(p)
        if max_p <= len_max:
            for i in range(max_p-len_max, min_p):
                pattern_pos = list(p - i)
                if pattern_pos not in list_pattern_pos:
                    list_pattern_pos.append(pattern_pos)
    return list_pattern_pos


def _get_list_pattern_pos(steps=None, n_min=2, n_max=4, len_max=15):
    """Get list of pattern positions using get_pattern_pos"""
    list_pattern_pos = []
    for n in range(n_min, n_max+1):
        list_pattern_pos.extend(_get_pattern_pos(steps=steps, repeat=n, len_max=len_max))
    list_pattern_pos = sorted(list_pattern_pos)
    return list_pattern_pos


def _get_list_periodic_pattern_pos(len_seq=None, step1=None, step2=None, pos=1):
    """Get list of periodic pattern positions"""
    list_pos = [pos]
    while pos <= len_seq:
        if len(list_pos) % 2 != 0:
            pos += step1
        else:
            pos += step2
        if pos <= len_seq:
            list_pos.append(pos)
    return list_pos


# II Main Functions
class Split:
    """Class for splitting parts into Segments, Patterns, and PeriodicPatterns.
    Counting of amino acid positions for splits start at 1 and N-terminal of the
    protein sequence"""
    def __init__(self, type_str=True):
        self.type_str = type_str
        self.dict_list_periodic_pattern_pos = {}

    def get_list_periodic_pattern_pos(self, seq=None, step1=None, step2=None, start=1):
        args_key = (len(seq), step1, step2, start)
        if args_key not in self.dict_list_periodic_pattern_pos:
            list_pos = _get_list_periodic_pattern_pos(len_seq=len(seq), step1=step1, step2=step2, pos=start)
        else:
            list_pos = self.dict_list_periodic_pattern_pos[args_key]
        return list_pos

    @staticmethod
    def segment(seq=None, i_th=1, n_split=2):
        """Get i-th segment of sequence that is split into n contiguous segments.
        For starting at the C-terminus, the sequence will be reversed.

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        i_th: integer indicating the selected segment
        n_split: integer indicating the number of segments to split the given sequence in

        Returns
        -------
        seq_segment: Segment split for given sequence

        Notes
        -----
        Segments are denoted as 'Segment(i-th,n_split)'
        """
        len_segment = len(seq) / n_split
        start = int(len_segment * (i_th - 1))   # Start at 0 for i_th = 1
        end = int(len_segment * i_th)
        # DEV: If 'IndexError: list index out of range',
        #  check in interface that sequence size matches with features
        #  via 'check_match_features_seq_parts'
        seq_segment = seq[start:end]
        return seq_segment

    def pattern(self, seq=None, terminus="N", list_pos=None):
        """Get sequence pattern consisting of n positions given in list_pos.By default, a pattern starts at
        the N-terminus. For starting at the C-terminus, the sequence will be reversed.

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        terminus: 'N' or 'C' indicating the start side of split
        list_pos: list with integers indicating the positions the Pattern split should consist of

        Returns
        -------
        seq_pattern: Pattern split for given sequence

        Notes
        -----
        Patterns are denoted as 'Pattern(N/C,p1,p2,...,pn)',
            where N or C specifies whether the pattern starts at the N- or C-terminus of the sequence
            and pn denotes the n-th position from list_pos.
        """
        if terminus == "C":
            seq = seq[::-1]
        if self.type_str:
            seq_pattern = "".join([seq[int(i)-1] for i in list_pos])
        else:
            # DEV: If 'IndexError: list index out of range',
            #  check in interface that sequence size matches with features
            #  via 'check_match_features_seq_parts'
            seq_pattern = [seq[int(i)-1] for i in list_pos]
        return seq_pattern

    def periodicpattern(self, seq=None, terminus="N", step1=None, step2=None, start=1):
        """Get a periodic sequence pattern consisting of elements with an alternating step size given by step1, step2
        and starting at the given start position. For starting at the C-terminus, the sequence will be reversed.

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        terminus: 'N' or 'C' indicating the start side of split
        step1: integer indicating odd step sizes
        step2: integer indicating even step sizes
        start: integer indicating start position (starting from 1)

        Returns
        -------
        seq_periodicpattern: PeriodicPattern split for given sequence

        Notes
        -----
        PeriodicPatterns are denoted as 'periodicpattern(N/C,i+step1/step2,start)',
            where N or C specifies whether the pattern starts at the N- or C-terminus of the sequence,
            i+step1/step2 defines the alternating g step sizes, and start gives the start position beginning at 0.

        Giving an example for proteins, for step1=3, step2=4, start=1, the periodic pattern consists of
            alternating 3rd and 4th amino acids within the whole sequence starting at the position one.
            By default, a pattern starts at the N-terminus. For starting at the C-terminus, the sequence must
            be reversed. For IMP substrates, the periodic pattern is representing a face of the
            helical transmembrane domain given for step1, step2 in {3, 4} and start >= step1.
        """
        if terminus == "C":
            seq = seq[::-1]
        list_pos = self.get_list_periodic_pattern_pos(seq=seq, step1=step1, step2=step2, start=start)
        if self.type_str:
            seq_periodicpattern = "".join([seq[i-1] for i in list_pos])
        else:
            # DEV: If 'IndexError: list index out of range',
            #  check in interface that sequence size matches with features
            #  via 'check_match_features_seq_parts'
            seq_periodicpattern = [seq[i-1] for i in list_pos]
        return seq_periodicpattern


class SplitVec:
    """Class for splitting part vectors into Segments, Patterns, and PeriodicPatterns (see Split() doc string)"""
    def __init__(self, type_str=True):
        self.type_str = type_str
        self.dict_list_periodic_pattern_pos = {}

    def get_list_periodic_pattern_pos(self, seq=None, step1=None, step2=None, start=1):
        args_key = (len(seq), step1, step2, start)
        if args_key not in self.dict_list_periodic_pattern_pos:
            list_pos = _get_list_periodic_pattern_pos(len_seq=len(seq), step1=step1, step2=step2, pos=start)
        else:
            list_pos = self.dict_list_periodic_pattern_pos[args_key]
        return list_pos

    @staticmethod
    def segment(seq=None, i_th=1, n_split=2):
        """Get i-th segment of sequence that is split into n contiguous segments."""
        len_segment = len(seq) / n_split
        start = int(len_segment * (i_th - 1))
        end = int(len_segment * i_th)
        seq_segment = seq[start:end]
        return seq_segment

    @staticmethod
    def pattern(seq=None, terminus="N", list_pos=None):
        """Get sequence pattern consisting of n positions given in list_pos"""
        if terminus == "C":
            seq = seq[::-1]
        seq_pattern = [seq[int(i)-1] for i in list_pos]
        return seq_pattern

    def periodicpattern(self, seq=None, terminus="N", step1=None, step2=None, start=1):
        """Get a periodic sequence pattern consisting of elements with an alternating step size given by step1, step2
        and starting at the given start position."""
        if terminus == "C":
            seq = seq[::-1]
        list_pos = self.get_list_periodic_pattern_pos(seq=seq, step1=step1, step2=step2, start=start)
        seq_periodicpattern = [seq[i-1] for i in list_pos]
        return seq_periodicpattern


class SplitRange:
    """Class for creating range of splits for testing sets of multiple features in CPP"""

    def __init__(self, split_type_str=True):
        """"""
        # Dictionary for caching list_pattern_pos results
        self.dict_list_pattern_pos = {}
        if split_type_str:
            self.sp = Split(type_str=split_type_str)
        else:
            self.sp = SplitVec()

    # Helper method
    def get_list_pattern_pos(self, steps=None, n_min=2, n_max=4, len_max=15):
        args_key = tuple(steps) if isinstance(steps, list) else steps
        if args_key not in self.dict_list_pattern_pos:
            list_pattern_pos = _get_list_pattern_pos(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max)
            self.dict_list_pattern_pos[args_key] = list_pattern_pos
        else:
            list_pattern_pos = self.dict_list_pattern_pos[args_key]
        return list_pattern_pos

    # Segment methods
    def segment(self, seq=None, n_split_min=1, n_split_max=15):
        """Get range of all possible Segment splits for given sequences."""
        f = self.sp.segment  # Unbound function for higher performance
        return [f(seq=seq, n_split=n_split, i_th=i_th)
                for n_split in range(n_split_min, n_split_max+1)
                for i_th in range(1, n_split+1)]

    @staticmethod
    def labels_segment(n_split_min=1, n_split_max=15):
        """Get labels for range of Segment splits."""
        return [f"{ut.STR_SEGMENT}({i_th},{n_split})"
                for n_split in range(n_split_min, n_split_max+1)
                for i_th in range(1, n_split+1)]

    # Pattern methods
    def pattern(self, seq=None, steps=None, n_min=2, n_max=4, len_max=15):
        """Get range of all possible Pattern splits for given sequence."""
        list_pattern_pos = self.get_list_pattern_pos(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max)
        f = self.sp.pattern  # Use pre-initialized instance
        return [f(seq=seq, terminus=terminus, list_pos=pattern_pos)
                for terminus in ['N', 'C'] for pattern_pos in list_pattern_pos]

    def labels_pattern(self, steps=None, n_min=2, n_max=4, len_max=15):
        """Get labels for range of Pattern splits."""
        list_pattern_pos = self.get_list_pattern_pos(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max)
        return [f"{ut.STR_PATTERN}({terminus},{','.join(map(str, pattern_pos))})"
                for terminus in ['N', 'C'] for pattern_pos in list_pattern_pos]

    # Periodic pattern methods
    def periodicpattern(self, seq=None, steps=None):
        """Get range of all possible PeriodicPattern splits for given sequence."""
        f = self.sp.periodicpattern  # Use pre-initialized instance
        return [f(seq=seq, terminus=terminus, step1=step1, step2=step2, start=start)
                for terminus in ['N', 'C']
                for step1, step2 in itertools.product(steps, repeat=2)
                for start in range(1, step1+1)]

    @staticmethod
    def labels_periodicpattern(steps=None):
        """Get labels of all possible PeriodicPattern splits."""
        return [f"{ut.STR_PERIODIC_PATTERN}({terminus},i+{step1}/{step2},{start})"
                for terminus in ['N', 'C']
                for step1, step2 in itertools.product(steps, repeat=2)
                for start in range(1, step1+1)]
