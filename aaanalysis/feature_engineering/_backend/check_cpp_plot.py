"""
This is a script for general CPP Plot check functions.
"""
import time
import pandas as pd
import numpy as np
import warnings

import aaanalysis.utils as ut


# Check for plotting methods
def check_args_xtick(xtick_size=None, xtick_width=None, xtick_length=None):
    """Check if x tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=0)
    ut.check_number_range(name="xtick_size", val=xtick_size, **args)
    ut.check_number_range(name="xtick_width", val=xtick_width, **args)
    ut.check_number_range(name="xtick_length", val=xtick_length, **args)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    return args_xtick


def check_args_ytick(ytick_size=None, ytick_width=None, ytick_length=None):
    """Check if y tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=0)
    ut.check_number_range(name="ytick_size", val=ytick_size, **args)
    ut.check_number_range(name="ytick_width", val=ytick_width, **args)
    ut.check_number_range(name="ytick_length", val=ytick_length, **args)
    args_ytick = dict(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
    return args_ytick


# Check colors
def check_part_color(tmd_color=None, jmd_color=None):
    """Check if part colors valid"""
    ut.check_color(name="tmd_color", val=tmd_color)
    ut.check_color(name="jmd_color", val=jmd_color)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    return args_part_color


def check_seq_color(tmd_seq_color=None, jmd_seq_color=None):
    """Check sequence colors"""
    ut.check_color(name="tmd_seq_color", val=tmd_seq_color, accept_none=True)
    ut.check_color(name="jmd_seq_color", val=jmd_seq_color, accept_none=True)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    return args_seq_color


def check_match_dict_color_df(dict_color=None, df=None, name_df="df_feat"):
    """Check if color dictionary is matching to DataFrame with categories"""
    ut.check_df(name=name_df, df=df, cols_requiered=ut.COL_CAT)
    list_cats = list(sorted(set(df[ut.COL_CAT])))
    if dict_color is None:
        dict_color = ut.DICT_COLOR_CAT
    str_add = f"'dict_color' should be a dictionary with colors for: {list_cats}"
    ut.check_dict_color(val=dict_color, str_add=str_add)
    list_cat_not_in_dict_cat = [x for x in list_cats if x not in dict_color]
    if len(list_cat_not_in_dict_cat) > 0:
        error = f"'dict_color' not complete! Following categories are missing from '{name_df}': {list_cat_not_in_dict_cat}"
        raise ValueError(error)
    for key in list_cats:
        color = dict_color[key]
        ut.check_color(name=key, val=color)
    # Filter colors
    _dict_color = {cat: dict_color[cat] for cat in list_cats}
    return _dict_color
