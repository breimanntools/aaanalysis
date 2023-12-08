"""
This is a script for the common SequenceFeature and CPP checking functions
"""
import aaanalysis.utils as ut

# Helper functions


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
        raise ValueError("'split_kws' should be type dict (not {})".format(type(split_kws)))
    # Check if split_kws contains wrong split_types
    split_types = [ut.STR_SEGMENT, ut.STR_PATTERN, ut.STR_PERIODIC_PATTERN]
    wrong_split_types = [x for x in split_kws if x not in split_types]
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
        periodic_pattern_args = split_kws[ut.STR_PERIODIC_PATTERN]
        if periodic_pattern_args["steps"] != sorted(periodic_pattern_args["steps"]):
            raise ValueError(f"For '{ut.STR_PERIODIC_PATTERN}', 'steps' should be ordered in ascending order.")


