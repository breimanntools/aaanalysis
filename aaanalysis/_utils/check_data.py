"""
This is a script for data checking utility functions.
"""
import pandas as pd
import numpy as np
from sklearn.utils import check_array

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

def check_labels(labels=None):
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
