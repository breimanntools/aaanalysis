"""
Basic utility check functions for type checking
"""
import pandas as pd
import numpy as np

from ._utils import add_str, VALID_INT_TYPES, VALID_INT_FLOAT_TYPES


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
    valid_types = VALID_INT_TYPES if just_int else VALID_INT_TYPES + VALID_INT_FLOAT_TYPES
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        raise ValueError(f"'{name}' should be {type_description}, but got {type(val).__name__}.")


def check_number_range(name=None, val=None, min_val=0, max_val=None, exclusive_limits=False,
                       accept_none=False, just_int=None):
    """Check if value of given name is within defined range"""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    # Define valid types for integers and floating points
    valid_types = VALID_INT_TYPES if just_int else VALID_INT_TYPES + VALID_INT_FLOAT_TYPES
    # Verify the value's type and range
    type_description = "an integer" if just_int else "a float or an integer"
    if not isinstance(val, valid_types):
        raise ValueError(f"'{name}' should be {type_description}, but got {type(val).__name__}.")
    # Min and max values are excluded from allowed values
    if exclusive_limits:
        if val <= min_val or (max_val is not None and val >= max_val):
            range_desc = f"n > {min_val}" if max_val is None else f"{min_val} < n < {max_val}"
            raise ValueError(f"'{name}' should be {type_description} with {range_desc}, but got {val}.")
    else:
        if val < min_val or (max_val is not None and val > max_val):
            range_desc = f"n >= {min_val}" if max_val is None else f"{min_val} <= n <= {max_val}"
            raise ValueError(f"'{name}' should be {type_description} with {range_desc}, but got {val}.")
    return val


def check_str(name=None, val=None, accept_none=False, return_empty_string=False, str_add=None):
    """Check type string"""
    if val is None:
        if not accept_none:
            str_error = add_str(str_error=f"'{name}' should not be None.", str_add=str_add)
            raise ValueError(str_error)
        return "" if return_empty_string else None
    if not isinstance(val, str):
        str_error = add_str(str_error= f"'{name}' ('{val}') should be string.", str_add=str_add)
        raise ValueError(str_error)
    return val


def check_bool(name=None, val=None, accept_none=False):
    """Check if the provided value is a boolean."""
    if val is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None.")
        return None
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


def check_list_like(name=None, val=None, accept_none=False, convert=True, accept_str=False, min_len=None,
                    check_all_non_neg_int=False, check_all_non_none=True, check_all_str_or_convertible=False):
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
    if check_all_non_none:
        n_none = len([x for x in val if x is None])
        if n_none > 0:
            raise ValueError(f"'{name}' should not contain 'None' (n={n_none})")
    if check_all_non_neg_int:
        if any(type(i) != int or i < 0 for i in val):
            raise ValueError(f"'{name}' should only contain non-negative integers.")
    if check_all_str_or_convertible:
        wrong_elements = [x for x in val if not isinstance(x, (str, int, float, np.number))]
        if len(wrong_elements) > 0:
            raise ValueError(f"The following elements in '{name}' are not strings or"
                             f" reasonably convertible: {wrong_elements}")
        else:
            val = [str(x) for x in val]
    if min_len is not None and len(val) < min_len:
        raise ValueError(f"'{name}' should not contain at least {min_len} elements")
    return val


# Check special types
def check_ax(ax=None, accept_none=False):
    """Check if the provided value is a matplotlib Axes instance or None."""
    import matplotlib.axes
    if accept_none and ax is None:
        return None
    if not isinstance(ax, matplotlib.axes.Axes):
        raise ValueError(f"'ax' (type={type(ax)}) should be mpl.axes.Axes or None.")


def check_figsize(figsize=None, accept_none=False):
    """Check size of figure"""
    if accept_none and figsize is None:
        return None # skip check
    check_tuple(name="figsize", val=figsize, n=2)
    check_number_range(name="figsize:width", val=figsize[0], min_val=1, just_int=False)
    check_number_range(name="figsize:height", val=figsize[1], min_val=1, just_int=False)
