"""
This is a script for the common SequenceFeature and CPP checking functions
"""
import aaanalysis.utils as ut

# Helper functions
# TODO remove from _part
def check_input_part_creation(seq=None, tmd_start=None, tmd_stop=None):
    """Check if input for part creation is given"""
    if None in [seq, tmd_start, tmd_stop]:
        raise ValueError("'seq', 'tmd_start', 'tmd_stop' must be given (should not be None).")


def check_parts_exist(tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None):
    """Check if parts are given"""
    list_parts = [tmd_seq, jmd_n_seq, jmd_c_seq]
    if None in list_parts:
        raise ValueError("'tmd', 'jmd_n', and 'jmd_c' must be given (should not be None)")


# TODO remove from _split
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
        raise ValueError(f"'list_pos' ({list_pos}) must have type list")
    if len(list_pos) < 0:
        raise ValueError(f"'list_pos' ({list_pos}) must contain at least one element")
    # Check if arguments are in order
    if not sorted(list_pos) == list_pos:
        raise ValueError(f"Pattern position ({list_pos})should be given in ascending order")
    if max(list_pos) > len(seq):
        raise ValueError(f"Maximum pattern position ({list_pos}) should not exceed sequence length ({len(seq)})")


def check_periodicpattern(seq=None, terminus=None, step1=None, step2=None, start=1):
    """Check arguments for periodicpattern split method"""
    check_seq(seq=seq)
    if terminus not in ["N", "C"]:
        raise ValueError("'terminus' must be either 'N' or 'C'")
    if type(step1) != int or type(step2) != int or type(start) != int:
        raise ValueError("'step1', 'step2', and 'start' must be type int")


# II Main Functions
def check_split_kws(split_kws=None, accept_none=True):
    """Check if argument dictionary for splits is a valid input"""
    # Split dictionary with data types
    split_kws_types = {ut.STR_SEGMENT: dict(n_split_min=int, n_split_max=int),
                       ut.STR_PATTERN: dict(steps=list, n_min=int, n_max=int, len_max=int),
                       ut.STR_PERIODIC_PATTERN: dict(steps=list)}
    if accept_none and split_kws is None:
        return None     # Skip check
    if not isinstance(split_kws, dict):
        raise ValueError(f"'split_kws' should be type dict (not {split_kws})")
    # Check if split_kws contains wrong split_types
    wrong_split_types = [x for x in split_kws if x not in ut.LIST_SPLIT_TYPES]
    if len(wrong_split_types) > 0:
        error = f"Following keys are invalid: {wrong_split_types}." \
                "\n  'split_kws' should have following structure: {split_kws_types}."
        raise ValueError(error)
    if len(split_kws) == 0:
        raise ValueError("'split_kws' should be not empty")
    # Check if arguments are valid and have valid type
    for split_type in split_kws:
        for arg in split_kws[split_type]:
            if arg not in split_kws_types[split_type]:
                error = f"'{arg}' arg in '{split_type}' of 'split_kws' is invalid." \
                        "\n  'split_kws' should have following structure: {split_kws_types}."
                raise ValueError(error)
            arg_val = split_kws[split_type][arg]
            arg_type = type(arg_val)
            target_arg_type = split_kws_types[split_type][arg]
            if target_arg_type != arg_type:
                error = f"Type of '{arg}':'{arg_val}' ({arg_type}) should be {target_arg_type}"
                raise ValueError(error)
            if arg_type is list:
                wrong_type = [x for x in arg_val if type(x) is not int]
                if len(wrong_type) > 0:
                    error = f"All list elements ({arg_val}) of '{arg}' should have type int."
                    raise ValueError(error)
    # Check Segment
    if ut.STR_SEGMENT in split_kws:
        segment_args = split_kws[ut.STR_SEGMENT]
        if segment_args["n_split_min"] > segment_args["n_split_max"]:
            raise ValueError(f"For '{ut.STR_SEGMENT}', 'n_split_min' should be smaller or equal to 'n_split_max'")
    # Check Pattern
    if ut.STR_PATTERN in split_kws:
        pattern_args = split_kws[ut.STR_PATTERN]
        if pattern_args["n_min"] > pattern_args["n_max"]:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'n_min' should be smaller or equal to 'n_max'")
        if pattern_args["steps"] != sorted(pattern_args["steps"]):
            raise ValueError(f"For '{ut.STR_PATTERN}', 'steps' should be ordered in ascending order.")
        if pattern_args["steps"][0] >= pattern_args["len_max"]:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'len_max' should be greater than the smallest step in 'steps'.")
    # Check PeriodicPattern
    if ut.STR_PERIODIC_PATTERN in split_kws:
        periodicpattern_args = split_kws[ut.STR_PERIODIC_PATTERN]
        if periodicpattern_args["steps"] != sorted(periodicpattern_args["steps"]):
            raise ValueError(f"For '{ut.STR_PERIODIC_PATTERN}', 'steps' should be ordered in ascending order.")
