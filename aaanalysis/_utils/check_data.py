"""
This is a script for data checking utility functions.
"""
import pandas as pd
import numpy as np
from sklearn.utils import check_array
import aaanalysis._utils.check_type as check_type

# Helper functions
def _convert_2d(val=None, name=None):
    """
    Convert array-like data to 2D array. Handles lists of arrays, lists of lists, and 1D lists.
    """
    str_error = f"'{name}' should be a 2D list or 2D array with rows having the same number of columns."
    if isinstance(val, list):
        # Check if List with arrays and return if yes
        if all(isinstance(i, np.ndarray) for i in val):
            try:
                val = np.asarray(val)
            except ValueError:
                raise ValueError(str_error)
        # Convert 1D list to 2D list
        elif all(not isinstance(i, list) for i in val):
            try:
                val = np.asarray([val])
            except ValueError:
                raise ValueError(str_error)
        # For nested lists, ensure they are 2D (list of lists with equal lengths)
        else:
            try:
                val = np.array(val)  # Convert nested list to numpy array
                if val.ndim != 2:
                    raise ValueError
            except ValueError:
                raise ValueError(str_error)
    elif hasattr(val, 'ndim') and val.ndim == 1:
        try:
            val = np.asarray([val])
        except ValueError:
            raise ValueError(str_error)
    return val


# Check array like
def check_array_like(name=None, val=None, dtype=None, ensure_2d=False, allow_nan=False,
                     convert_2d=False, accept_none=False):
    """Check if the provided value is array-like and matches the specified dtype."""
    if val is None:
        if accept_none:
            return None # skip tests
        else:
            raise ValueError(f"'{name}' should not be None.")
    # Type checking
    if dtype == 'int':
        expected_dtype = 'int'
    elif dtype == 'float':
        expected_dtype = 'float64'
    elif dtype == 'any' or dtype is None:
        expected_dtype = None
    else:
        raise ValueError(f"'dtype' ({dtype}) not recognized.")
    # Convert a 1D list or array to a 2D array
    if convert_2d:
        val = _convert_2d(val=val, name=name)
    # Utilize Scikit-learn's check_array for robust checking
    try:
        val = check_array(val, dtype=expected_dtype, ensure_2d=ensure_2d, force_all_finite=not allow_nan)
    except Exception as e:
        dtype = "any type" if dtype is None else dtype
        raise ValueError(f"'{name}' should be array-like with '{dtype}' values."
                         f"\nscikit message:\n\t{e}")
    return val


# Check feature matrix and labels
def check_X(X, X_name="X", min_n_samples=3, min_n_features=2, ensure_2d=True, allow_nan=False, accept_none=False):
    """Check the feature matrix X is valid."""
    if X is None:
        if not accept_none:
            raise ValueError(f"'{X_name}' should not be None")
        else:
            return None
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



def check_labels(labels=None, name="labels", vals_requiered=None, len_requiered=None, allow_other_vals=True,
                 len_per_group_requiered=None):
    """Check the provided labels against various criteria like type, required values, and length."""
    if labels is None:
        raise ValueError(f"'{name}' should not be None.")
    # Convert labels to a numpy array if it's not already
    labels = np.asarray(labels)
    # Ensure labels is at least 1-dimensional
    if labels.ndim == 0:
        labels = np.array([labels.item()])  # Convert 0-d array to 1-d array
    unique_labels = set(labels)
    if len(unique_labels) == 1:
       raise ValueError(f"'{name}' should contain more than one different value ({unique_labels}).")
    integer_types = (int, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
    wrong_types = [l for l in unique_labels if not isinstance(l, integer_types)]
    if wrong_types:
        raise ValueError(f"Labels in '{name}' should be type int, but contain: {set(map(type, wrong_types))}")
    if vals_requiered is not None:
        missing_vals = [x for x in vals_requiered if x not in labels]
        if len(missing_vals) > 0:
            raise ValueError(f"'{name}' ({unique_labels}) does not contain requiered values: {missing_vals}")
        if not allow_other_vals:
            wrong_vals = [x for x in labels if x not in vals_requiered]
            if len(wrong_vals) > 0:
                raise ValueError(f"'{name}' ({unique_labels}) does contain wrong values: {wrong_vals}")
    if len_requiered is not None and len(labels) != len_requiered:
        raise ValueError(f"'{name}' (n={len(labels)}) should contain {len_requiered} values.")
    # Check for minimum length per group
    if len_per_group_requiered is not None:
        label_counts = {label: np.sum(labels == label) for label in unique_labels}
        underrepresented_labels = {label: count for label, count in label_counts.items() if
                                   count < len_per_group_requiered}
        if underrepresented_labels:
            raise ValueError(
                f"Each label should have at least {len_per_group_requiered} occurrences. "
                f"Underrepresented labels: {underrepresented_labels}")
    return labels


def check_match_X_labels(X=None, X_name="X", labels=None, labels_name="labels", check_variability_for_kld=False):
    """Check if the number of samples in X matches the number of labels."""
    n_samples, n_features = X.shape
    if n_samples != len(labels):
        raise ValueError(f"n_samples does not match for '{X_name}' ({len(X)}) and '{labels_name}' ({len(labels)}).")
    if check_variability_for_kld:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            group_X = X[labels == label]
            if not np.all(np.var(group_X, axis=0) != 0):
                raise ValueError(f"Variance in 'X' for label '{label}' from '{labels_name}' is too low to compute KDL")


def check_match_X_list_labels(X=None, list_labels=None, comp_kld=False, vals_requiered=None):
    """Check if each label set is matching with X"""
    for i, labels in enumerate(list_labels):
        check_labels(labels=labels, vals_requiered=vals_requiered)
        check_match_X_labels(X=X, labels=labels, labels_name=f"list_labels (set {i+1})",
                             check_variability_for_kld=comp_kld)


def check_match_list_labels_names_datasets(list_labels=None, names_datasets=None):
    """Check if length of list_labels and names match"""
    if names_datasets is None:
        return None # Skip check
    if len(list_labels) != len(names_datasets):
        raise ValueError(f"Length of 'list_labels' ({len(list_labels)}) and 'names' ({len(names_datasets)} does not match) ")



# Check sets
def check_superset_subset(subset=None, superset=None, name_subset=None, name_superset=None):
    """Check if all elements of the subset are contained in the superset."""
    wrong_elements = [x for x in subset if x not in superset]
    if len(wrong_elements) != 0:
        raise ValueError(f"'{name_superset}' does not contain the following elements of '{name_subset}': {wrong_elements}")


# df checking functions
def check_df(name="df", df=None, accept_none=False, accept_nan=True, check_all_positive=False,
             cols_requiered=None, cols_forbidden=None, cols_nan_check=None):
    """Check if the provided DataFrame meets various criteria such as NaN values, required/forbidden columns, etc."""
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
