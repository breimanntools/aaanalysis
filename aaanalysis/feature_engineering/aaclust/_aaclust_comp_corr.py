"""
This is a script for the AAclust comp_correlation method.
"""
import pandas as pd
import numpy as np

# Settings

# I Helper Functions
def _sort_X_labels_names(X, labels=None, names=None):
    """"""
    sorted_order = np.argsort(labels)
    labels = [labels[i] for i in sorted_order]
    X = X[sorted_order]
    if names:
        names = [names[i] for i in sorted_order]
    return X, labels, names

def _get_df_corr(X=None, X_ref=None):
    """"""
    # Temporary labels to avoid any confusion with potential duplicates
    X_labels = range(len(X))
    X_ref_labels = range(len(X), len(X) + len(X_ref))
    combined = np.vstack((X, X_ref))
    df_corr_full = pd.DataFrame(combined.T).corr()
    # Select only the rows corresponding to X and columns corresponding to X_ref
    df_corr = df_corr_full.loc[X_labels, X_ref_labels]
    return df_corr


# II Main Functions
def compute_correlation(X, X_ref=None, labels=None, labels_ref=None, names=None, names_ref=None):
    """Computes Pearson correlation of given data with reference data."""
    # Sort based on labels
    X, labels, names = _sort_X_labels_names(X, labels=labels, names=names)
    if X_ref is not None:
        X_ref, labels_ref, names_ref = _sort_X_labels_names(X_ref, labels=labels_ref, names=names_ref)
    # Compute correlations
    if X_ref is None:
        df_corr = pd.DataFrame(X.T).corr()
    else:
        df_corr = _get_df_corr(X=X, X_ref=X_ref)
    # Replace indexes and columns with names or labels
    df_corr.index = names if names else labels
    if X_ref is None:
        df_corr.columns = names if names else labels
    else:
        df_corr.columns = names_ref if names_ref else labels_ref
    return df_corr
