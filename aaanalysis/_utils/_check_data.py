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

# TODO separation of concerns
def check_feat_matrix(X=None, y=None, y_name="labels", accept_none_y=True,
                      ensure_2d=True, allow_nan=False, min_n_unique_samples=3, min_n_features=2):
    """Check feature matrix valid and matches with y if (if provided)"""
    # Check if X is None
    if X is None:
        raise ValueError("Feature matrix 'X' should not be None.")
    if not accept_none_y and y is None:
        raise ValueError(f"'{y_name}' ({y}) should not be None.")
    # Use check_array from scikit to convert
    try:
        X = check_array(X, dtype="float64", ensure_2d=ensure_2d, force_all_finite=not allow_nan)
    except Exception as e:
        raise ValueError(f"Feature matrix 'X' should be array-like with float values."
                         f"\nscikit message:\n\t{e}")

    # Check X values (not Nan, inf or None)
    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError("Feature matrix 'X' should not contain NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Feature matrix 'X' should not contain infinite values.")
    if X.dtype == object:
        if np.any([elem is None for row in X for elem in row]):
            raise ValueError("Feature matrix 'X' should not contain None.")

    # Check all identical samples
    if len(set(map(tuple, X))) == 1:
        raise ValueError("Feature matrix 'X' should not have all identical samples.")

    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    if y is not None and n_samples != len(y):
        raise ValueError(f"Number of samples does not match for 'X' ({n_samples}) and '{y_name}' ({y}.")

    if n_samples == 0 or n_features == 0:
        raise ValueError(f"Shape of 'X' ({n_samples}, {n_features}) indicates empty feature matrix."
                         f"\nX = {X}")
    if n_unique_samples < min_n_unique_samples or n_samples < min_n_unique_samples:
        raise ValueError(f"Number of unique samples ({n_unique_samples}) should be at least {min_n_unique_samples}."
                         f"\nX = {X}")
    if n_features < min_n_features:
        raise ValueError(f"'n_features' ({n_features}) should be at least {min_n_features}."
                         f"\nX = {X}")
    return X


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
