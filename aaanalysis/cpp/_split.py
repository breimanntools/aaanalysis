"""
Script with Split objects used to fragment sequence parts into distinct segments or patterns.
"""
import numpy as np
import itertools

import aaanalysis.cpp._utils as ut


# I Helper Functions
# Check functions
def check_seq(seq=None):
    """Check if seq is not None"""
    if seq is None:
        raise ValueError("'seq' should not be None")


def check_steps(steps=None):
    """Check steps and set to default if None"""
    if steps is None:
        steps = [3, 4]
    if type(steps) is not list or len(steps) < 2:
        raise ValueError("'steps' must be a list with more than 2 elements")
    return steps


def check_segment(seq=None, i_th=1, n_split=2):
    """Check arguments for segment split method"""
    check_seq(seq=seq)
    if type(i_th) != int or type(n_split) != int:
        raise ValueError("'i_th' and 'n_split' must be int")
    if len(seq) < n_split:
        error = f"'n_split' ('{n_split}') should not be higher than length of sequence ('{seq}',len={len(seq)})"
        raise ValueError(error)


def check_pattern(seq=None, terminus=None, list_pos=None):
    """Check arguments for pattern split method"""
    check_seq(seq=seq)
    # Check terminus
    if terminus not in ["N", "C"]:
        raise ValueError("'terminus' must be either 'N' or 'C'")
    # Check if minimum one position is specified
    if type(list_pos) is not list:
        raise ValueError("'list_pos' must have type list")
    if len(list_pos) < 0:
        raise ValueError("'list_pos' must contain at least one element")
    # Check if arguments are in order
    if not sorted(list_pos) == list_pos:
        raise ValueError("Pattern position should be given in ascending order")
    if max(list_pos) > len(seq):
        raise ValueError("Maximum pattern position should not exceed sequence length")


def check_periodicpattern(seq=None, terminus=None, step1=None, step2=None, start=1):
    """Check arguments for periodicpattern split method"""
    check_seq(seq=seq)
    if terminus not in ["N", "C"]:
        raise ValueError("'terminus' must be either 'N' or 'C'")
    if type(step1) != int or type(step2) != int or type(start) != int:
        raise ValueError("'step1', 'step2', and 'start' must be type int")


# Pattern helper functions
def get_pattern_pos(steps=None, repeat=2, len_max=12):
    """Get all possible positions from steps with number of repeats and maximum length using
    itertools: https://docs.python.org/3/library/itertools.html"""
    list_steps = itertools.product(steps, repeat=repeat)    # Cartesian product of all step combinations
    list_pos = [np.cumsum(s) for s in list_steps]       # Positions from steps
    # Get all possible pattern positions
    list_pattern_pos = []
    for p in list_pos:
        max_p = max(p)
        min_p = min(p)
        if max_p <= len_max:
            for i in range(max_p-len_max, min_p):
                pattern_pos = list(p - i)
                if pattern_pos not in list_pattern_pos:
                    list_pattern_pos.append(pattern_pos)
    return list_pattern_pos


def get_list_pattern_pos(steps=None, n_min=2, n_max=4, len_max=15):
    """Get list of pattern positions using get_pattern_pos"""
    list_pattern_pos = []
    for n in range(n_min, n_max+1):
        list_pattern_pos.extend(get_pattern_pos(steps=steps, repeat=n, len_max=len_max))
    list_pattern_pos = sorted(list_pattern_pos)
    return list_pattern_pos


# II Main Functions
class Split:
    """Class for splitting parts into Segments, Patterns, and PeriodicPatterns.
    Counting of amino acid positions for splits start at 1 and N-terminal of the
    protein sequence"""
    def __init__(self, type_str=True):
        self.type_str = type_str

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
        check_segment(seq=seq, i_th=i_th, n_split=n_split)
        len_segment = len(seq) / n_split
        start = int(len_segment * (i_th - 1))   # Start at 0 for i_th = 1
        end = int(len_segment * i_th)
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
        list_pos = [int(x) for x in list_pos if x is not None]
        check_pattern(seq=seq, terminus=terminus, list_pos=list_pos)
        if terminus == "C":
            seq = seq[::-1]
        if self.type_str:
            seq_pattern = "".join([seq[i-1] for i in list_pos])
        else:
            seq_pattern = [seq[i-1] for i in list_pos]
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
        PeriodicPatterns are denoted as 'Periodic_Pattern(N/C,i+step1/step2,start)',
            where N or C specifies whether the pattern starts at the N- or C-terminus of the sequence,
            i+step1/step2 defines the alternating g step sizes, and start gives the start position beginning at 0.

        Giving an example for proteins, for step1=3, step2=4, start=1, the periodic pattern consists of
            alternating 3rd and 4th amino acids within the whole sequence starting at the position one.
            By default, a pattern starts at the N-terminus. For starting at the C-terminus, the sequence must
            be reversed. For IMP substrates, the periodic pattern is representing a face of the
            helical transmembrane domain given for step1, step2 in {3, 4} and start >= step1.
        """
        check_periodicpattern(seq=seq, terminus=terminus, step1=step1, step2=step2, start=start)
        if terminus == "C":
            seq = seq[::-1]
        pos = start
        list_pos = [pos]
        while pos <= len(seq):
            if len(list_pos) % 2 != 0:
                pos += step1
            else:
                pos += step2
            if pos <= len(seq):
                list_pos.append(pos)
        if self.type_str:
            seq_periodicpattern = "".join([seq[i-1] for i in list_pos])
        else:
            seq_periodicpattern = [seq[i-1] for i in list_pos]
        return seq_periodicpattern


class SplitRange:
    """Class for creating range of splits for testing sets of multiple features in CPP"""

    # Segment methods
    @staticmethod
    def segment(seq=None, n_split_min=1, n_split_max=15):
        """Get range of all possible Segment splits for given sequences.
        Output matches with SequenceFeature.labels_segment.

        Parameters
        ----------
        seq: seq: sequence (e.g., amino acids sequence for protein)
        n_split_min: integer indicating minimum Segment size
        n_split_max: integer indicating maximum Segment size

        Returns
        -------
        seq_splits: list of sequence Segment splits
        """
        sp = Split()
        f = sp.segment  # Unbound function for higher performance
        seq_splits = []
        for n_split in range(n_split_min, n_split_max+1):
            for i_th in range(1, n_split+1):
                seq_segment = f(seq=seq, n_split=n_split, i_th=i_th)
                seq_splits.append(seq_segment)
        return seq_splits

    @staticmethod
    def labels_segment(n_split_min=1, n_split_max=15):
        """Get labels for range of Segment splits.
        Output matches with SequenceFeature.segment.

        Parameters
        ----------
        n_split_min: integer indicating minimum Segment size
        n_split_max: integer indicating maximum Segment size

        Returns
        -------
        labels: list of labels of Segment splits
        """
        labels = []
        for n_split in range(n_split_min, n_split_max+1):
            for i_th in range(1, n_split+1):
                name = "{}({},{})".format(ut.STR_SEGMENT, i_th, n_split)
                labels.append(name)
        return labels

    # Pattern methods
    @staticmethod
    def pattern(seq=None, steps=None, n_min=2, n_max=4, len_max=15):
        """Get range of all possible Pattern splits for given sequence.
        Output matches with SequenceFeature.labels_pattern.

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        steps: list of integer indicating possible step sizes
        n_min: integer indicating minimum of elements in Pattern split
        n_min: integer indicating maximum of elements in Pattern split
        len_max: maximum of sequence length for splitting

        Returns
        -------
        seq_splits: list of sequence Pattern splits
        """
        steps = check_steps(steps=steps)
        list_pattern_pos = get_list_pattern_pos(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max)
        sp = Split()
        f = sp.pattern  # Unbound function for higher performance
        seq_splits = []
        for terminus in ['N', 'C']:
            for pattern_pos in list_pattern_pos:
                seq_pattern = f(seq=seq, terminus=terminus, list_pos=pattern_pos)
                seq_splits.append(seq_pattern)
        return seq_splits

    @staticmethod
    def labels_pattern(steps=None, n_min=2, n_max=4, len_max=15):
        """Get labels for range of Pattern splits.
        Output matches with SequenceFeature.pattern.

        Parameters
        ----------
        steps: list of integer indicating possible step sizes
        n_min: integer indicating minimum of elements in Pattern split
        n_min: integer indicating maximum of elements in Pattern split
        len_max: maximum of sequence length for splitting

        Returns
        -------
        labels: list of labels of Pattern splits
        """
        steps = check_steps(steps=steps)
        list_pattern_pos = get_list_pattern_pos(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max)
        labels = []
        for terminus in ['N', 'C']:
            for pattern_pos in list_pattern_pos:
                name = "{}({},{})".format(ut.STR_PATTERN, terminus, ",".join(str(x) for x in pattern_pos))
                labels.append(name)
        return labels

    # Periodic pattern methods
    @staticmethod
    def periodicpattern(seq=None, steps=None):
        """Get range of all possible PeriodicPattern splits for given sequence.
        Output matches with SequenceFeature.labels_periodicpattern.

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        steps: list of integer indicating possible step sizes

        Returns
        -------
        seq_splits: list of sequence PeriodicPattern splits"""
        steps = check_steps(steps=steps)
        sp = Split()
        f = sp.periodicpattern  # Unbound function for higher performance
        seq_splits = []
        for terminus in ['N', 'C']:
            for step1, step2 in itertools.product(steps, repeat=2):
                for start in range(1, step1+1):
                    seq_periodicpattern = f(seq=seq, terminus=terminus, step1=step1, step2=step2, start=start)
                    seq_splits.append(seq_periodicpattern)
        return seq_splits

    @staticmethod
    def labels_periodicpattern(steps=None):
        """Get labels of all possible PeriodicPattern splits.
        Output matches with SequenceFeature.periodicpattern.

        Parameters
        ----------
        steps: list of integer indicating possible step sizes

        Returns
        -------
        labels: list of labels of Pattern splits
        """
        steps = check_steps(steps=steps)
        labels = []
        for terminus in ['N', 'C']:
            for step1, step2 in itertools.product(steps, repeat=2):
                for start in range(1, step1+1):
                    name = "{}({},i+{}/{},{})".format(ut.STR_PERIODIC_PATTERN, terminus, step1, step2, start)
                    labels.append(name)
        return labels

