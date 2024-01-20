"""
This is a script for plot checking utility functions.
"""
import warnings
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import aaanalysis._utils.check_type as ut_check


# Helper functions
def _is_valid_hex_color(val):
    """Check if a value is a valid hex color."""
    return isinstance(val, str) and re.match(r'^#[0-9A-Fa-f]{6}$', val)

def _is_valid_rgb_tuple(val):
    """Check if a value is a valid RGB tuple."""
    return (isinstance(val, tuple) and len(val) == 3 and
            all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in val))


# Check min and max values
def check_vmin_vmax(vmin=None, vmax=None):
    """Check if vmin and vmax are valid numbers and vmin is less than vmax."""
    ut_check.check_number_val(name="vmin", val=vmin, accept_none=True, just_int=False)
    ut_check.check_number_val(name="vmax", val=vmax, accept_none=True, just_int=False)
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(f"'vmin' ({vmin}) < 'vmax' ({vmax}) not fulfilled.")


def check_lim(name="xlim", val=None, accept_none=True, min_vaL_req=None, max_val_req=None, adjust_lim=False, verbose=True):
    """Validate that lim parameter ('xlim' or 'ylim') is tuple with two numbers, where the first is less than the second."""
    if val is None:
        if accept_none:
            return None  # Skip check
        else:
            raise ValueError(f"'{name}' should not be None")
    ut_check.check_tuple(name=name, val=val, n=2)
    min_val, max_val = val
    ut_check.check_number_val(name=f"{name}:min", val=min_val, just_int=False)
    ut_check.check_number_val(name=f"{name}:max", val=max_val, just_int=False)
    if min_val >= max_val:
        raise ValueError(f"'{name}:min' ({min_val}) should be < '{name}:max' ({max_val}).")
    str_error_warn_min = f"'{name}:min' ({min_val}) should be <= '{min_vaL_req}'."
    str_error_warn_max = f"'{name}:max' ({max_val}) should be >= '{max_val_req}'."
    if not adjust_lim:
        if min_val < min_vaL_req:
            raise ValueError(str_error_warn_min)
        if max_val > max_val_req:
            raise ValueError(str_error_warn_max)
    else:
        if min_val < min_vaL_req:
           if verbose:
               warnings.warn(str_error_warn_min)
           return None
        if max_val > max_val_req:
            if verbose:
                warnings.warn(str_error_warn_max)
            return None



def check_dict_xlims(dict_xlims=None, n_ax=None):
    """Validate the structure and content of dict_xlims to ensure it contains the correct keys and value formats."""
    if n_ax is None:
        # DEV: Developer warning
        raise ValueError("'n_ax' must be specified")
    if dict_xlims is None:
        return
    ut_check.check_dict(name="dict_xlims", val=dict_xlims)
    wrong_keys = [x for x in list(dict_xlims) if x not in range(n_ax)]
    if len(wrong_keys) > 0:
        raise ValueError(
            f"'dict_xlims' contains invalid keys: {wrong_keys}. Valid keys are axis indices from 0 to {n_ax - 1}.")
    for key in dict_xlims:
        check_lim(name="xlim", val=dict_xlims[key])


# Check colors
def check_color(name=None, val=None, accept_none=False):
    """Check if the provided value is a valid color for matplotlib."""
    base_colors = list(mcolors.BASE_COLORS.keys())
    tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
    css4_colors = list(mcolors.CSS4_COLORS.keys())
    all_colors = base_colors + tableau_colors + css4_colors
    if accept_none:
        all_colors.append("none")
    # Check if valid hex or RGB tuple
    if _is_valid_hex_color(val) or _is_valid_rgb_tuple(val):
        return
    elif val not in all_colors:
        error = f"'{name}' ('{val}') is not a valid color. Chose from following: {all_colors}"
        raise ValueError(error)


def check_list_colors(name=None, val=None, accept_none=False, min_n=None, max_n=None):
    """Check if color list is valid"""
    if accept_none and val is None:
        return None # Skip check
    val = ut_check.check_list_like(name=name, val=val, accept_none=accept_none, accept_str=True)
    for l in val:
        check_color(name=name, val=l, accept_none=accept_none)
    if min_n is not None and len(val) < min_n:
        raise ValueError(f"'{name}' should contain at least {min_n} colors")
    if max_n is not None and len(val) > max_n:
        raise ValueError(f"'{name}' should contain no more than {max_n} colors")


def check_dict_color(name="dict_color", val=None, accept_none=False, min_n=None, max_n=None):
    """Check if colors in dict_color are valid"""
    if accept_none and val is None:
        return None # Skip check
    ut_check.check_dict(name=name, val=val, accept_none=accept_none)
    for key in val:
        check_color(name=name, val=val[key], accept_none=accept_none)
    if min_n is not None and len(val) < min_n:
        raise ValueError(f"'{name}' should contain at least {min_n} colors")
    if max_n is not None and len(val) > max_n:
        raise ValueError(f"'{name}' should contain no more than {max_n} colors")


def check_cmap(name=None, val=None, accept_none=False):
    """Check if cmap is a valid colormap for matplotlib."""
    valid_cmaps = plt.colormaps()
    if accept_none and val is None:
        pass
    elif val not in valid_cmaps:
        error = f"'{name}' ('{val}') is not a valid cmap. Chose from following: {valid_cmaps}"
        raise ValueError(error)


def check_palette(name=None, val=None, accept_none=False):
    """Check if the provided value is a valid color palette."""
    if isinstance(val, str):
        check_cmap(name=name, val=val, accept_none=accept_none)
    elif isinstance(val, list):
        for v in val:
            check_color(name=name, val=v, accept_none=accept_none)
