"""
Basic utility check functions for type checking
"""
import pandas as pd
import numpy as np


# Type checking functions
def check_number_val(name=None, val=None, accept_none=False, just_int=False):
    """Check if value is float"""
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    if accept_none and val is None:
        return None
    valid_types = (int,) if just_int else (float, int)
    type_description = "int" if just_int else "float or int"
    if not isinstance(val, valid_types):
        raise ValueError(f"'{name}' ({val}) should be {type_description}.")


def check_number_range(name=None, val=None, min_val=0, max_val=None, accept_none=False, just_int=None):
    """Check if value of given name is within defined range"""
    if just_int is None:
        raise ValueError("'just_int' must be specified")
    if accept_none and val is None:
        return None
    valid_types = (int,) if just_int else (float, int)
    type_description = "int" if just_int else "float or int n, with"

    # Verify the value's type and range
    if not isinstance(val, valid_types) or val < min_val or (max_val is not None and val > max_val):
        range_desc = f"n>={min_val}" if max_val is None else f"{min_val}<=n<={max_val}"
        error = f"'{name}' ({val}) should be {type_description} {range_desc}. "
        if accept_none:
            error += "None is also accepted."
        raise ValueError(error)


def check_str(name=None, val=None, accept_none=False):
    """Check type string"""
    if accept_none and val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"'{name}' ('{val}') should be string.")


def check_bool(name=None, val=None):
    """Check if the provided value is a boolean."""
    if not isinstance(val, bool):
        raise ValueError(f"'{name}' ({val}) should be bool.")


def check_dict(name=None, val=None, accept_none=False):
    """Check if the provided value is a dictionary."""
    if accept_none and val is None:
        return None
    if not isinstance(val, dict):
        error = f"'{name}' ({val}) should be a dictionary"
        error += " or None." if accept_none else "."
        raise ValueError(error)


def check_tuple(name=None, val=None, n=None, check_n=True, accept_none=False):
    """"""
    if accept_none and val is None:
        return None
    if not isinstance(val, tuple):
        raise ValueError(f"'{name}' ({val}) should be a tuple.")
    if check_n and n is not None and len(val) != n:
        raise ValueError(f"'{name}' ({val}) should be a tuple with {n} elements.")


def check_list_like(name=None, val=None, accept_none=False, convert=True):
    """"""
    if accept_none and val is None:
        return None
    if not convert:
        if not isinstance(val, list):
            raise ValueError(f"'{name}' (type: {type(val)}) should be a list.")
    else:
        allowed_types = (list, tuple, np.ndarray, pd.Series)
        if not isinstance(val, allowed_types):
            raise ValueError(f"'{name}' (type: {type(val)}) should be one of {allowed_types}.")
        if isinstance(val, np.ndarray) and val.ndim != 1:
            raise ValueError(f"'{name}' is a multi-dimensional numpy array and cannot be considered as a list.")
        val = list(val)
    return val

# Check special types
def check_ax(ax=None, accept_none=False):
    """"""
    import matplotlib.axes
    if accept_none and ax is None:
        return None
    if not isinstance(ax, matplotlib.axes.Axes):
        raise ValueError(f"'ax' (type={type(ax)}) should be mpl.axes.Axes or None.")

