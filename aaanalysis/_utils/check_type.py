"""
Basic utility check functions for type checking
"""
import pandas as pd
import numpy as np

from ._utils import add_str
from .utils_types import VALID_INT_TYPES, VALID_INT_FLOAT_TYPES


# Type checking functions
def check_number_val(name=None, val=None, accept_none=False, just_int=False, str_add=None):
    """Check if value is a valid integer or float"""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    # Define valid types for integers and floating points
    valid_types = VALID_INT_TYPES if just_int else VALID_INT_TYPES + VALID_INT_FLOAT_TYPES
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        str_error = add_str(str_error=f"'{name}' should be {type_description}, but got {type(val).__name__}.",
                            str_add=str_add)
        raise ValueError(str_error)


def check_number_range(name=None, val=None, min_val=0, max_val=None, exclusive_limits=False,
                       accept_none=False, just_int=None, str_add=None):
    """Check if value of given name is within defined range"""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    # Define valid types for integers and floating points
    valid_types = VALID_INT_TYPES if just_int else VALID_INT_TYPES + VALID_INT_FLOAT_TYPES
    # Verify the value's type and range
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        str_error = add_str(str_error=f"'{name}' should be {type_description}, but got {type(val).__name__}.",
                            str_add=str_add)
        raise ValueError(str_error)
    # Min and max values are excluded from allowed values
    if exclusive_limits:
        if val <= min_val or (max_val is not None and val >= max_val):
            range_desc = f"n > {min_val}" if max_val is None else f"{min_val} < n < {max_val}"
            str_error = add_str(str_error=f"'{name}' should be {type_description} with {range_desc}, but got {val}.",
                                str_add=str_add)
            raise ValueError(str_error)
    else:
        if val < min_val or (max_val is not None and val > max_val):
            range_desc = f"n >= {min_val}" if max_val is None else f"{min_val} <= n <= {max_val}"
            str_error = add_str(str_error=f"'{name}' should be {type_description} with {range_desc}, but got {val}.",
                                str_add=str_add)
            raise ValueError(str_error)


def check_str(name=None, val=None, accept_none=False, str_add=None):
    """Check type string"""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return
    if not isinstance(val, str):
        str_error = add_str(str_error=f"'{name}' ('{val}') should be string.",
                            str_add=str_add)
        raise ValueError(str_error)


def check_str_options(name=None, val=None, accept_none=False, list_str_options=None):
    """Check if valid string option"""
    if accept_none and val is None:
        return None  # Skip test
    str_add = add_str(str_error=f"'{name}' ({val}) should be one of: {list_str_options}")
    check_str(name=name, val=val, accept_none=accept_none, str_add=str_add)
    if val not in list_str_options:
        raise ValueError(str_add)


def check_bool(name=None, val=None, accept_none=False, str_add=None):
    """Check if the provided value is a boolean."""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if not isinstance(val, bool):
        str_error = add_str(str_error=f"'{name}' ({val}) should be bool.",
                            str_add=str_add)
        raise ValueError(str_error)


def check_dict(name=None, val=None, accept_none=False, str_add=None):
    """Check if the provided value is a dictionary."""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if not isinstance(val, dict):
        str_error = add_str(str_error=f"'{name}' ({val}) should be a dictionary.",
                            str_add=str_add)
        raise ValueError(str_error)


def check_tuple(name=None, val=None, n=None, check_number=True, accept_none=False,
                accept_none_number=False, str_add=None):
    """Check if the provided value is a tuple, optionally of a certain length and containing only numbers."""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if not isinstance(val, tuple):
        str_error = add_str(str_error=f"'{name}' ({val}) should be a tuple.",
                            str_add=str_add)
        raise ValueError(str_error)
    if n is not None and len(val) != n:
        str_error = add_str(str_error=f"'{name}' ({val}) should be a tuple with {n} elements.",
                            str_add=str_add)
        raise ValueError(str_error)
    if n is not None and check_number:
        for v in val:
            check_number_val(name=name, val=v, just_int=False, accept_none=accept_none_number,
                             str_add=str_add)


def check_list_like(name=None, val=None, accept_none=False, convert=True, accept_str=False, min_len=None,
                    check_all_non_neg_int=False, check_all_non_none=True, check_all_str_or_convertible=False,
                    str_add=None):
    """Check if the value is list-like, optionally converting it to a list, and performing additional checks."""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return None
    if not convert:
        if not isinstance(val, list):
            str_error = add_str(str_error=f"'{name}' (type: {type(val)}) should be a list.",
                                str_add=str_add)
            raise ValueError(str_error)
    elif accept_str and isinstance(val, str):
        return [val]
    else:
        allowed_types = (list, tuple, np.ndarray, pd.Series)
        if not isinstance(val, allowed_types):
            str_error = add_str(str_error=f"'{name}' (type: {type(val)}) should be one of {allowed_types}.",
                                str_add=str_add)
            raise ValueError(str_error)
        if isinstance(val, np.ndarray) and val.ndim != 1:
            str_error = add_str(str_error=f"'{name}' is a multi-dimensional numpy array and cannot"
                                          f" be considered as a list.",
                                str_add=str_add)
            raise ValueError(str_error)
        val = list(val) if isinstance(val, (np.ndarray, pd.Series)) else val
    if check_all_non_none:
        n_none = len([x for x in val if x is None])
        if n_none > 0:
            str_error = add_str(str_error=f"'{name}' should not contain 'None' (n={n_none})",
                                str_add=str_add)
            raise ValueError(str_error)
    if check_all_non_neg_int:
        if any(type(i) != int or i < 0 for i in val):
            str_error = add_str(str_error=f"'{name}' should only contain non-negative integers.",
                                str_add=str_add)
            raise ValueError(str_error)
    if check_all_str_or_convertible:
        wrong_elements = [x for x in val if not isinstance(x, (str, int, float, np.number))]
        if len(wrong_elements) > 0:
            str_error = add_str(str_error=f"The following elements in '{name}' are not strings or"
                                          f" reasonably convertible: {wrong_elements}",
                                str_add=str_add)
            raise ValueError(str_error)
        else:
            val = [str(x) for x in val]
    if min_len is not None and len(val) < min_len:
        str_error = add_str(str_error=f"'{name}' should not contain at least {min_len} elements",
                            str_add=str_add)
        raise ValueError(str_error)
    # Check for numpy string objects and convert them to Python strings
    val = [x.item() if hasattr(x, 'item') and isinstance(x, np.str_) else x for x in val]
    return val
