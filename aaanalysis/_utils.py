#! /usr/bin/python3
"""
Config with folder structure
"""
import os
import platform
from sklearn.utils import check_array, check_consistent_length

# Helper Function
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


# Folder
SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, 'data')
URL_DATA = "https://github.com/breimanntools/aaanalysis/tree/master/aaanalysis/data/"

# Default data for protein analysis
STR_SCALES = "scales"   # Min-max normalized scales (from AAontology)
STR_SCALES_RAW = "scales_raw"   # Ras scales (from AAontology)
STR_SCALES_PC = "scales_pc"     # AAclust pc-based scales (pc: principal component)
STR_SCALE_CAT = "scale_classification"  # AAontology
STR_TOP60 = "top60"    # AAclustTop60
STR_TOP60_EVAL = "top60_eval"  # AAclustTop60 evaluation

# Column names
COL_SCALE_ID = "scale_id"
COL_SEQ = "sequence"
COL_CAT = "category"
COL_SUBCAT = "subcategory"


# General check functions
def check_non_negative_number(name=None, val=None, min_val=0, max_val=None, accept_none=False,
                              just_int=True):
    """Check if value of given name variable is non-negative integer"""
    check_types = [int] if just_int else [float, int]
    str_check = "non-negative int" if just_int else "non-negative float or int"
    add_str = f"n>={min_val}" if max_val is None else f"{min_val}<=n<={max_val}"
    if accept_none:
        add_str += " or None"
    error = f"'{name}' ({val}) should be {str_check} n, with " + add_str
    if accept_none and val is None:
        return None
    if type(val) not in check_types:
        raise ValueError(error)
    if val < min_val:
        raise ValueError(error)
    if max_val is not None and val > max_val:
        raise ValueError(error)


def check_float(name=None, val=None, accept_none=False, just_float=True):
    """Check if value is float"""
    if accept_none and val is None:
        return None
    if type(val) not in [float, int]:
        error = f"'{name}' ({val}) should be float"
        if not just_float:
            error += " or int."
        else:
            error += "."
        raise ValueError(error)


def check_str(name=None, val=None, accept_none=False):
    """"""
    if accept_none and val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"'{name}' ('{val}') should be string.")


def check_bool(name=None, val=None):
    """"""
    if type(val) != bool:
        raise ValueError(f"'{name}' ({val}) should be bool.")


def check_dict(name=None, val=None, accept_none=False):
    """"""
    error = f"'{name}' ('{val}') should be a dictionary"
    if accept_none:
        error += " or None."
    else:
        error += "."
    if accept_none and val is None:
        return None
    if not isinstance(val, dict):
        raise ValueError(error)


def check_tuple(name=None, val=None, n=None):
    """"""
    error = f"'{name}' ('{val}') should be a tuple"
    if n is not None:
        error += f" with {n} elements."
    else:
        error += "."
    if not isinstance(val, tuple):
        raise ValueError(error)
    if n is not None and len(val) != n:
        raise ValueError(error)


# Data check functions
# TODO update
def check_feat_matrix(X=None, names=None, labels=None):
    """Check if X and y match (y can be labels or names). Otherwise, transpose X or give error."""
    # TODO type check
    X = check_array(X)
    if labels is not None:
        check_consistent_length(X, labels)
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError(f"Shape of X ({n_samples}, {n_features}) indicates empty feature matrix.")
    if names is None:
        return X, names
    else:
        if n_samples != len(names):
            X = X.transpose()
        if X.shape[0] != len(names):
            error = f"Shape of X ({n_samples}, {n_features}) does not match with number of labels in y ({len(names)})."
            raise ValueError(error)
        return X, names

"""
def check_feat_matrix(X=None, y=None):
    #Check if X (feature matrix) and y (class labels) are not None and match
    if X is None:
        raise ValueError("'X' should not be None")
    check_array(X)    # Default checking function from sklearn

    if len(y) != X.shape[0]:
        raise ValueError(f"'y' (labels) does not match to 'X' (feature matrix)")
"""


# Plotting & print functions
def print_red(input_str, **args):
    """Prints the given string in red text."""
    print(f"\033[91m{input_str}\033[0m", **args)
