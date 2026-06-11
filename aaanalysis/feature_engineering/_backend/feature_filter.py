"""
This is a script for the shared backend of the model-free feature filters used by
both :meth:`NumericalFeature.filter_correlation` and the :class:`SequenceFeature`
pruning methods (``prune_by_variance`` / ``prune_by_correlation``). Kept at the
common ``_backend`` level so neither frontend reaches into the other's dedicated
backend package.
"""
import numpy as np


# II Main Functions
def filter_correlation_(X, max_cor=0.7):
    """Filter features based on Pearson correlation"""
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)
    # Get number of features
    n_features = X.shape[1]
    # Initialize the mask to select features
    is_selected = np.ones(n_features, dtype=bool)
    # Iterate over the upper triangle of the correlation matrix
    for i in range(n_features):
        if is_selected[i]:
            for j in range(i + 1, n_features):
                if is_selected[j] and abs(corr_matrix[i, j]) > max_cor:
                    is_selected[j] = False
    return is_selected


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
