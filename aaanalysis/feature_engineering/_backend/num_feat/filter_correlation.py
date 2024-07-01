"""
This is a script for the backend of the NumericalFeature.filter_correlation() method.
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
