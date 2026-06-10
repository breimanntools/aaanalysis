"""
This is a script for the backend of the SequenceFeature.prune_by_variance() method.
"""
import numpy as np


# II Main Functions
def filter_variance_(X, threshold=0.0):
    """Filter features whose column variance is at or below ``threshold``."""
    X = np.asarray(X, dtype=float)
    # Population variance per feature column (ddof=0).
    variances = np.var(X, axis=0)
    # A genuinely constant column can yield a tiny float epsilon (not exactly 0) when its
    # value is not exactly representable; snap zero-range columns to 0 so a threshold of 0
    # removes exactly the constant features.
    variances = np.where(np.ptp(X, axis=0) == 0, 0.0, variances)
    is_selected = variances > threshold
    return is_selected
