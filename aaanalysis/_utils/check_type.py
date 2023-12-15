"""
Basic utility check functions for type checking
"""
import pandas as pd
import numpy as np


# Type checking functions
def check_number_val(name=None, val=None, accept_none=False, just_int=False):
    """Check if value is a valid integer or float"""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    # Define valid types for integers and floating points
    integer_types = (int, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                     np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    float_types = (float, np.float_, np.float16, np.float32, np.float64)
    valid_types = integer_types if just_int else integer_types + float_types
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        raise ValueError(f"'{name}' should be {type_description}, but got {type(val).__name__}.")


def check_number_range(name=None, val=None, min_val=0, max_val=None, accept_none=False, just_int=None):
    """Check if value of given name is within defined range"""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    # Define valid types for integers and floating points
    integer_types = (int, np.int_, np.intc, np.intp, np.int8, np.int16,
                     np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    float_types = (float, np.float_, np.float16, np.float32, np.float64)
    valid_types = integer_types if just_int else integer_types + float_types

    # Verify the value's type and range
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        raise ValueError(f"'{name}' should be {type_description}, but got {type(val).__name__}.")
    if val < min_val or (max_val is not None and val > max_val):
        range_desc = f"n >= {min_val}" if max_val is None else f"{min_val} <= n <= {max_val}"
        raise ValueError(f"'{name}' should be {type_description} with {range_desc}, but got {val}.")
    return val


def check_str(name=None, val=None, accept_none=False, return_empty_string=False):
    """Check type string"""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return "" if return_empty_string else None
    if not isinstance(val, str):
        raise ValueError(f"'{name}' ('{val}') should be string.")
    return val

# TODO check if used
def check_str_in_list(name=None, val=None, list_options=None, accept_none=False):
    """Check if val is one of the given options."""
    if list_options is None or not list_options:
        raise ValueError("list_options must be provided and not empty.")
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if not isinstance(val, str) or val not in list_options:
        raise ValueError(f"'{name}' ('{val}') should be one of the following: {list_options}")


def check_bool(name=None, val=None):
    """Check if the provided value is a boolean."""
    if not isinstance(val, bool):
        raise ValueError(f"'{name}' ({val}) should be bool.")


def check_dict(name=None, val=None, accept_none=False):
    """Check if the provided value is a dictionary."""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if not isinstance(val, dict):
        raise ValueError(f"'{name}' ({val}) should be a dictionary.")


def check_tuple(name=None, val=None, n=None, check_n_number=True, accept_none=False):
    """Check if the provided value is a tuple, optionally of a certain length and containing only numbers."""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if not isinstance(val, tuple):
        raise ValueError(f"'{name}' ({val}) should be a tuple.")
    if n is not None and len(val) != n:
        raise ValueError(f"'{name}' ({val}) should be a tuple with {n} elements.")
    if n is not None and check_n_number:
        for v in val:
            check_number_val(name=name, val=v, just_int=False, accept_none=False)


def check_list_like(name=None, val=None, accept_none=False, convert=True, accept_str=False, check_all_non_neg_int=False):
    """Check if the value is list-like, optionally converting it to a list, and performing additional checks."""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if not convert:
        if not isinstance(val, list):
            raise ValueError(f"'{name}' (type: {type(val)}) should be a list.")
    elif accept_str and isinstance(val, str):
        return [val]
    else:
        allowed_types = (list, tuple, np.ndarray, pd.Series)
        if not isinstance(val, allowed_types):
            raise ValueError(f"'{name}' (type: {type(val)}) should be one of {allowed_types}.")
        if isinstance(val, np.ndarray) and val.ndim != 1:
            raise ValueError(f"'{name}' is a multi-dimensional numpy array and cannot be considered as a list.")
        val = list(val) if isinstance(val, (np.ndarray, pd.Series)) else val
    if check_all_non_neg_int:
        if any(type(i) != int or i < 0 for i in val):
            raise ValueError(f"'{name}' should only contain non-negative integers.")
    return val


# Check special types
def check_ax(ax=None, accept_none=False):
    """Check if the provided value is a matplotlib Axes instance or None."""
    import matplotlib.axes
    if accept_none and ax is None:
        return None
    if not isinstance(ax, matplotlib.axes.Axes):
        raise ValueError(f"'ax' (type={type(ax)}) should be mpl.axes.Axes or None.")

