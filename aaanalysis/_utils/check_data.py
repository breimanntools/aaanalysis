"""
This is a script for data checking utility functions.
"""
import pandas as pd
import numpy as np
from sklearn.utils import check_array
import aaanalysis._utils.check_type as check_type

# Helper functions
def check_array_like(name=None, val=None, dtype=None, ensure_2d=False, allow_nan=False):
    """
    Check if the provided value matches the specified dtype.
    If dtype is None, checks for general array-likeness.
    If dtype is 'int', 'float', or 'any', checks for specific types.
    """
    if name is None:
        raise ValueError(f"'{name}' should not be None.")
    # Utilize Scikit-learn's check_array for robust checking
    if dtype == 'int':
        expected_dtype = 'int'
    elif dtype == 'float':
        expected_dtype = 'float64'
    elif dtype == 'any' or dtype is None:
        expected_dtype = None
    else:
        raise ValueError(f"'dtype' ({dtype}) not recognized.")
    try:
        val = check_array(val, dtype=expected_dtype, ensure_2d=ensure_2d, force_all_finite=not allow_nan)
    except Exception as e:
        raise ValueError(f"'{name}' should be array-like with {dtype} values."
                         f"\nscikit message:\n\t{e}")
    return val


# Check feature matrix and labels
def check_X(X, min_n_samples=3, min_n_features=2, ensure_2d=True, allow_nan=False):
    """Check the feature matrix X is valid."""
    X = check_array_like(name="X", val=X, dtype="float", ensure_2d=ensure_2d, allow_nan=allow_nan)
    if np.isinf(X).any():
        raise ValueError(f"'X' should not contain infinite values")
    n_samples, n_features = X.shape
    if n_samples < min_n_samples:
        raise ValueError(f"n_samples ({n_samples} in 'X') should be >= {min_n_samples}."
                         f"\nX = {X}")
    if n_features < min_n_features:
        raise ValueError(f"n_features ({n_features} in 'X') should be >= {min_n_features}."
                         f"\nX = {X}")
    return X


def check_X_unique_samples(X, min_n_unique_samples=3):
    """Check if the matrix X has a sufficient number of unique samples."""
    n_unique_samples = len(set(map(tuple, X)))
    if n_unique_samples == 1:
        raise ValueError("Feature matrix 'X' should not have all identical samples.")
    if n_unique_samples < min_n_unique_samples:
        raise ValueError(f"n_unique_samples ({n_unique_samples}) should be >= {min_n_unique_samples}."
                         f"\nX = {X}")
    return X

def check_labels(labels=None, vals_requiered=None, len_requiered=None):
    """"""
    if labels is None:
        raise ValueError(f"'labels' should not be None.")
    # Convert labels to a numpy array if it's not already
    labels = np.asarray(labels)
    unique_labels = set(labels)
    if len(unique_labels) == 1:
       raise ValueError(f"'labels' should contain more than one different value ({unique_labels}).")
    wrong_types = [l for l in unique_labels if not np.issubdtype(type(l), np.integer)]
    if wrong_types:
        raise ValueError(f"Labels in 'labels' should be type int, but contain: {set(map(type, wrong_types))}")
    if vals_requiered is not None:
        missing_vals = [x for x in vals_requiered if x not in labels]
        if len(missing_vals) > 0:
            raise ValueError(f"'labels' ({unique_labels}) does not contain requiered value: {missing_vals}")
    if len_requiered is not None and len(labels) != len_requiered:
        raise ValueError(f"'labels' (n={len(labels)}) should contain {len_requiered} values.")
    return labels


def check_match_X_labels(X=None, X_name="X", labels=None, labels_name="labels"):
    """"""
    n_samples, n_features = X.shape
    if n_samples != len(labels):
        raise ValueError(f"n_samples does not match for '{X_name}' ({len(X)}) and '{labels_name}' ({len(labels)}).")

# Check sets
def check_superset_subset(subset=None, superset=None, name_subset=None, name_superset=None):
    """"""
    wrong_elements = [x for x in subset if x not in superset]
    if len(wrong_elements) != 0:
        raise ValueError(f"'{name_superset}' does not contain the following elements of '{name_subset}': {wrong_elements}")


# df checking functions
def check_df(name="df", df=None, accept_none=False, accept_nan=True, check_all_positive=False,
             cols_requiered=None, cols_forbidden=None, cols_nan_check=None):
    """"""
    # Check DataFrame and values
    if df is None:
        if not accept_none:
            raise ValueError(f"'{name}' should not be None")
        else:
            return None
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"'{name}' ({type(df)}) should be DataFrame")
    if not accept_nan and df.isna().any().any():
        raise ValueError(f"'{name}' contains NaN values, which are not allowed")
    if check_all_positive:
        numeric_df = df.select_dtypes(include=['float', 'int'])
        if numeric_df.min().min() <= 0:
            raise ValueError(f"'{name}' should not contain non-positive values.")

    # Check columns
    args = dict(accept_str=True, accept_none=True)
    cols_requiered = check_type.check_list_like(name='cols_requiered', val=cols_requiered, **args)
    cols_forbidden = check_type.check_list_like(name='cols_forbidden', val=cols_forbidden, **args)
    cols_nan_check = check_type.check_list_like(name='cols_nan_check', val=cols_nan_check, **args)
    if cols_requiered is not None:
        missing_cols = [col for col in cols_requiered if col not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"'{name}' is missing required columns: {missing_cols}")
    if cols_forbidden is not None:
        forbidden_cols = [col for col in cols_forbidden if col in df.columns]
        if len(forbidden_cols) > 0:
            raise ValueError(f"'{name}' is contains forbidden columns: {forbidden_cols}")
    if cols_nan_check is not None:
        if df[cols_nan_check].isna().sum().sum() > 0:
            raise ValueError(f"NaN values are not allowed in '{cols_nan_check}'.")
