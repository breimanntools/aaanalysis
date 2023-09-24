"""
Utility check functions
"""
from sklearn.utils import check_array, check_consistent_length
import pandas as pd


# Type checking functions
def check_non_negative_number(name=None, val=None, min_val=0, max_val=None, accept_none=False,
                              just_int=None):
    """Check if value of given name variable is non-negative integer"""
    if just_int is None:
        raise ValueError("'just_int' must be specified")
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


# Array checking functions
def check_feat_matrix(X=None, names=None, labels=None):
    """Transpose matrix and check if X and y match (y can be labels or names). Transpose back otherwise """
    X = check_array(X).transpose()
    if labels is not None:
        check_consistent_length(X, labels)
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError(f"Shape of 'X' ({n_samples}, {n_features}) indicates empty feature matrix.")
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


# df checking functions
def check_col_in_df(df=None, name_df=None, col=None, col_type=None, accept_nan=False, error_if_exists=False):
    """
    Check if the column exists in the DataFrame, if the values have the correct type, and if NaNs are allowed.
    """
    # Check if the column already exists and raise error if error_if_exists is True
    if error_if_exists and (col in df.columns):
        raise ValueError(f"Column '{col}' already exists in '{name_df}'")

    # Check if the column exists in the DataFrame
    if col not in df.columns:
        raise ValueError(f"'{col}' must be a column in '{name_df}': {list(df.columns)}")

    # Make col_type a list if it is not already
    if col_type is not None and not isinstance(col_type, list):
        col_type = [col_type]

    # Check if the types match
    if col_type is not None:
        wrong_types = [x for x in df[col] if not any([isinstance(x, t) for t in col_type])]

        # Remove NaNs from the list of wrong types if they are accepted
        if accept_nan:
            wrong_types = [x for x in wrong_types if not pd.isna(x)]

        if len(wrong_types) > 0:
            raise ValueError(f"Values in '{col}' should be of type(s) {col_type}, "
                             f"but the following values do not match: {wrong_types}")

    # Check if NaNs are present when they are not accepted
    if not accept_nan:
        if df[col].isna().sum() > 0:
            raise ValueError(f"NaN values are not allowed in '{col}'.")




