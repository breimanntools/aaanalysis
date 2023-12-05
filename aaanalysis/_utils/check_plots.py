"""
This is a script for plot checking utility functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import aaanalysis._utils.check_type as ut_check
# Helper functions


# Check min and max values
def check_vmin_vmax(vmin=None, vmax=None):
    """Check if number of cmap colors is valid with given value range"""
    ut_check.check_number_val(name="vmin", val=vmin, accept_none=True, just_int=False)
    ut_check.check_number_val(name="vmax", val=vmax, accept_none=True, just_int=False)
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(f"'vmin' ({vmin}) < 'vmax' ({vmax}) not fulfilled.")


def check_color(name=None, val=None, accept_none=False):
    """Check if color valid for matplotlib"""
    base_colors = list(mcolors.BASE_COLORS.keys())
    tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
    css4_colors = list(mcolors.CSS4_COLORS.keys())
    all_colors = base_colors + tableau_colors + css4_colors
    if accept_none:
        all_colors.append("none")
    if val not in all_colors:
        error = f"'{name}' ('{val}') is not a valid color. Chose from following: {all_colors}"
        raise ValueError(error)


def check_cmap(name=None, val=None, accept_none=False):
    """Check if cmap is valid for matplotlib"""
    valid_cmaps = plt.colormaps()
    if accept_none and val is None:
        pass
    elif val not in valid_cmaps:
        error = f"'{name}' ('{val}') is not a valid cmap. Chose from following: {valid_cmaps}"
        raise ValueError(error)

# CPP plots
def check_ylim(df=None, ylim=None, col_value=None, retrieve_plot=False, scaling_factor=1.1):
    """"""
    if ylim is not None:
        ut_check.check_tuple(name="ylim", val=ylim, n=2)
        ut_check.check_number_val(name="ylim:min", val=ylim[0], just_int=False)
        ut_check.check_number_val(name="ylim:max", val=ylim[1], just_int=False)
        max_val = round(max(df[col_value]), 3)
        max_y = ylim[1]
        if max_val >= max_y:
            error = "Maximum of 'ylim' ({}) must be higher than maximum" \
                    " value of given datasets ({}).".format(max_y, max_val)
            raise ValueError(error)
    else:
        if retrieve_plot:
            ylim = plt.ylim()
            ylim = (ylim[0] * scaling_factor, ylim[1] * scaling_factor)
    return ylim