"""
This is a script for ...
"""
import pandas as pd
import numpy as np
# Write wrapper around scikit checkers
from sklearn.utils import check_array

# Array checking functions
def check_array_like(name=None, val=None, dtype=None, accept_none=False,
                     ensure_2d=False, allow_nan=False):
    """
    Check if the provided value matches the specified dtype.
    If dtype is None, checks for general array-likeness.
    If dtype is 'int', 'float', or 'any', checks for specific types.
    """
    if accept_none and val is None:
        return None

    # Convert DataFrame and Series to np.ndarray
    if isinstance(val, (pd.DataFrame, pd.Series)):
        val = val.values

    # Utilize Scikit-learn's check_array for robust checking
    if dtype == 'int':
        expected_dtype = 'int'
    elif dtype == 'float':
        expected_dtype = 'float64'
    elif dtype == 'any':
        expected_dtype = None
    else:
        raise ValueError(f"'dtype' ({dtype}) not recognized.")
    try:
        val = check_array(val, dtype=expected_dtype, ensure_2d=ensure_2d, force_all_finite=not allow_nan)
    except Exception as e:
        raise ValueError(f"'{name}' should be array-like with {dtype} values."
                         f"\nscikit message:\n\t{e}")

    return val


def check_feat_matrix(X=None, y=None, name_y="labels",
                      ensure_2d=True, allow_nan=False):
    """Check feature matrix valid and matches with y if (if provided)"""
    if X is None:
        raise ValueError("Feature matrix 'X' should not be None")
    try:
        X = check_array(X, dtype="float64", ensure_2d=ensure_2d, force_all_finite=not allow_nan)
    except Exception as e:
        raise ValueError(f"Feature matrix 'X'  be array-like with float values."
                         f"\nscikit message:\n\t{e}")
    if y is None:
        return X
    n_samples, n_features = X.shape
    if n_samples != len(y):
        raise ValueError(f"Number of samples does not match for 'X' ({len(n_samples)}) and '{name_y}' ({y}.")
    if n_samples == 0 or n_features == 0:
        raise ValueError(f"Shape of 'X' ({n_samples}, {n_features}) indicates empty feature matrix.")
    return X

# TODO check these functions if used
"""
def check_feat_matrix(X=None, y=None, labels=None):
    X = check_array(X).transpose()
    if labels is not None:
        check_consistent_length(X, labels)
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError(f"Shape of 'X' ({n_samples}, {n_features}) indicates empty feature matrix.")
    if y is None:
        return X, y
    else:
        if n_samples != len(y):
            X = X.transpose()
        if X.shape[0] != len(y):
            error = f"Shape of X ({n_samples}, {n_features}) does not match with number of labels in y ({len(y)})."
            raise ValueError(error)
        return X, y
"""
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
