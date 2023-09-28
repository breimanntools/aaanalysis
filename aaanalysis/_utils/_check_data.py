"""
This is a script for ...
"""
from sklearn.utils import check_array, check_consistent_length
import pandas as pd

# TODO check these functions if used
# Array checking functions
def check_feat_matrix(X=None, y=None, labels=None):
    """Transpose matrix and check if X and y match (y can be labels or names). Transpose back otherwise """
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
