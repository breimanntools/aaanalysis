"""
This is a script for general CPP Plot check functions.
"""
import time
import pandas as pd
import numpy as np
import warnings

import aaanalysis.utils as ut


# Check input data
# TODO check if needed
def check_value_type(value_type=None, count_in=True):
    """Check if value type is valid"""
    list_value_type = ["count", "sum", "mean"]
    if count_in:
        list_value_type.append("count")
    if value_type not in list_value_type:
        raise ValueError(f"'value_type' ('{value_type}') should be on of following: {list_value_type}")

# TODO check if needed
def check_y_categorical(df=None, y=None):
    """Check if the y column in the dataframe is categorical."""
    list_cat_columns = [col for col, data_type in zip(list(df), df.dtypes)
                        if data_type != float and "position" not in col]# and col != "feature"]
    if y not in list_cat_columns:
        raise ValueError(f"'y' ({y}) should be one of following columns with categorical values "
                         f"of 'df': {list_cat_columns}")


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


# Check sequence size
def check_args_size(seq_size=None, fontsize_tmd_jmd=None):
    """Check if sequence size parameters match"""
    ut.check_number_range(name="seq_size", val=seq_size, min_val=0, accept_none=True, just_int=False)
    ut.check_number_range(name="fontsize_tmd_jmd", val=fontsize_tmd_jmd, min_val=0, accept_none=True, just_int=False)
    args_size = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
    return args_size


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
