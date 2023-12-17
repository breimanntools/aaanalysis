"""
This is a script for utility functions for statistical measures.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_adjusted(X=None, y=None):
    """Get adjusted Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    comparing, for each feature, groups (given by y (labels)) by feature values in X (feature matrix).
    """
    auc = np.apply_along_axis((lambda x: roc_auc_score(y, x) - 0.5), 0, X)
    auc = np.round(auc, 3)
    return auc
