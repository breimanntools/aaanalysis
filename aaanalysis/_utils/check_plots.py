"""
This is a script for plot checking utility functions.
"""
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ._utils import add_str
import aaanalysis._utils.check_type as check_type
import numpy as np


# Helper functions
def _is_valid_hex_color(val):
    """Check if a value is a valid hex color."""
    return isinstance(val, str) and re.match(r'^#[0-9A-Fa-f]{6}$', val)


def _is_valid_rgb_tuple(val):
    """Check if a value is a valid RGB tuple."""
    return (isinstance(val, tuple) and len(val) == 3 and
            all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in val))


# Check figure
def check_fig(fig=None, accept_none=False, str_add=None):
    """Check if the provided value is a matplotlib Figure instance or None."""
    import matplotlib.figure
    if accept_none and fig is None:
        return None
    if not isinstance(fig, matplotlib.figure.Figure):
        str_error = f"'fig' (type={type(fig)}) should be mpl.figure.Figure or None."
        if str_add:
            str_error += " " + str_add
        raise ValueError(str_error)


def check_ax(ax=None, accept_none=False, str_add=None, return_first=False):
    """Check if the provided value is a matplotlib Axes instance or None."""
    import matplotlib.axes
    if accept_none and ax is None:
        return None
    if not isinstance(ax, matplotlib.axes.Axes):
        str_error = add_str(str_error=f"'ax' (type={type(ax)}) should be mpl.axes.Axes or None.",
                            str_add=str_add)
        raise ValueError(str_error)
    if return_first:
        if isinstance(ax, (list, tuple, np.ndarray)) and len(ax) > 0:
            ax = ax[0]
    return ax


def check_figsize(figsize=None, accept_none=False, str_add=None):
    """Check size of figure"""
    if accept_none and figsize is None:
        return None # skip check
    check_type.check_tuple(name="figsize", val=figsize, n=2, str_add=str_add)
    args = dict(min_val=1, just_int=False, str_add=str_add)
    check_type.check_number_range(name="figsize:width", val=figsize[0], **args)
    check_type.check_number_range(name="figsize:height", val=figsize[1], **args)


def check_fontsize_args(**kwargs):
    """Check fontsize parameters"""
    args_fs = {}
    for name, val in kwargs.items():
        check_type.check_number_range(name=name, val=val, min_val=0, accept_none=True, just_int=False)
        args_fs[name] = val
    return args_fs


# Check min and max values
def check_vmin_vmax(vmin=None, vmax=None, str_add=None):
    """Check if vmin and vmax are valid numbers and vmin is less than vmax."""
    args = dict(accept_none=True, just_int=False, str_add=str_add)
    check_type.check_number_val(name="vmin", val=vmin, **args)
    check_type.check_number_val(name="vmax", val=vmax, **args)
    if vmin is not None and vmax is not None:
        if vmin >= vmax:
            str_error = add_str(str_error=f"'vmin' ({vmin}) < 'vmax' ({vmax}) not fulfilled.",
                                str_add=str_add)
            raise ValueError(str_error)


def check_lim(name="xlim", val=None, accept_none=True, str_add=None):
    """Validate that lim parameter ('xlim' or 'ylim') is tuple with two numbers, where the first
     is less than the second."""
    if val is None:
        if accept_none:
            return None  # Skip check
        else:
            raise ValueError(f"'{name}' should not be None")
    check_type.check_tuple(name=name, val=val, n=2)
    min_val, max_val = val
    args = dict(just_int=False, str_add=str_add)
    check_type.check_number_val(name=f"{name}:min", val=min_val, **args)
    check_type.check_number_val(name=f"{name}:max", val=max_val, **args)
    if min_val >= max_val:
        str_error = add_str(str_error=f"'{name}:min' ({min_val}) should be < '{name}:max' ({max_val}).",
                            str_add=str_add)
        raise ValueError(str_error)


def check_dict_xlims(dict_xlims=None, n_ax=None, str_add=None):
    """Validate the structure and content of dict_xlims to ensure it contains the correct keys and value formats."""
    if n_ax is None:
        # DEV: Developer warning
        raise ValueError("'n_ax' must be specified")
    if dict_xlims is None:
        return
    check_type.check_dict(name="dict_xlims", val=dict_xlims, str_add=str_add)
    wrong_keys = [x for x in list(dict_xlims) if x not in range(n_ax)]
    if len(wrong_keys) > 0:
        str_error = add_str(str_error= f"'dict_xlims' contains invalid keys: {wrong_keys}. "
                                       f"Valid keys are axis indices from 0 to {n_ax - 1}.",
                            str_add=str_add)
        raise ValueError(str_error)
    for key in dict_xlims:
        check_lim(name="xlim", val=dict_xlims[key], str_add=str_add)


# Check colors
def check_color(name=None, val=None, accept_none=False, str_add=None):
    """Check if the provided value is a valid color for matplotlib."""
    if val is None:
        if accept_none:
            return None # Skip test
        else:
            raise ValueError(f"'{name}' should not be None")
    base_colors = list(mcolors.BASE_COLORS.keys())
    tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
    css4_colors = list(mcolors.CSS4_COLORS.keys())
    all_colors = base_colors + tableau_colors + css4_colors
    if _is_valid_hex_color(val) or _is_valid_rgb_tuple(val):
        return
    elif val not in all_colors:
        str_error = add_str(str_error=f"'{name}' ('{val}') is not a valid color. Chose from following: {all_colors}",
                            str_add=str_add)
        raise ValueError(str_error)


def check_list_colors(name=None, val=None, accept_none=False, min_n=None, max_n=None, str_add=None):
    """Check if color list is valid"""
    if accept_none and val is None:
        return None # Skip check
    val = check_type.check_list_like(name=name, val=val, accept_none=accept_none, accept_str=True, str_add=str_add)
    for l in val:
        check_color(name=name, val=l, accept_none=accept_none, str_add=str_add)
    if min_n is not None and len(val) < min_n:
        str_error = add_str(str_error=f"'{name}' should contain at least {min_n} colors",
                            str_add=str_add)
        raise ValueError(str_error)
    if max_n is not None and len(val) > max_n:
        str_error = add_str(str_error=f"'{name}' should contain no more than {max_n} colors",
                            str_add=str_add)
        raise ValueError(str_error)


def check_dict_color(name="dict_color", val=None, accept_none=False, min_n=None, max_n=None, str_add=None):
    """Check if colors in dict_color are valid"""
    if accept_none and val is None:
        return None # Skip check
    check_type.check_dict(name=name, val=val, accept_none=accept_none)
    for key in val:
        check_color(name=name, val=val[key], accept_none=accept_none)
    if min_n is not None and len(val) < min_n:
        str_error = add_str(str_error=f"'{name}' should contain at least {min_n} colors",
                            str_add=str_add)
        raise ValueError(str_error)
    if max_n is not None and len(val) > max_n:
        str_error = add_str(str_error=f"'{name}' should contain no more than {max_n} colors",
                            str_add=str_add)
        raise ValueError(str_error)


def check_cmap(name=None, val=None, accept_none=False, str_add=None):
    """Check if cmap is a valid colormap for matplotlib."""
    valid_cmaps = plt.colormaps()
    if accept_none and val is None:
        pass
    elif val not in valid_cmaps:
        str_error = add_str(str_error=f"'{name}' ('{val}') is not a valid cmap. Chose from following: {valid_cmaps}",
                            str_add=str_add)
        raise ValueError(str_error)


def check_palette(name=None, val=None, accept_none=False, str_add=None):
    """Check if the provided value is a valid color palette."""
    if isinstance(val, str):
        check_cmap(name=name, val=val, accept_none=accept_none, str_add=str_add)
    elif isinstance(val, list):
        for v in val:
            check_color(name=name, val=v, accept_none=accept_none, str_add=str_add)
