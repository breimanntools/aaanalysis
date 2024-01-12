"""This is a script for the backend of the TreeModel.fit() method."""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import aaanalysis.utils as ut

# II Main methods
# 1. Step: Recursive feature elimination (RFE)
def recursive_feature_elimination(X, labels=None, step=None, n_feat_max=50, n_feat_min=25, n_cv=5, scoring="f1",
                                  random_state=None, verbose=None, i=None, n_rounds=None):
    """Perform Recursive Feature Elimination to select the best feature set."""
    rf = RandomForestClassifier(random_state=random_state)
    n_features = X.shape[1]
    selected_features = np.ones(n_features, dtype=bool)
    n_total = len(selected_features)
    best_score = 0
    is_selected = selected_features.copy()
    while n_features > n_feat_min:
        if verbose:
            pct_progress = abs(1 - (n_features - n_feat_min) / (n_total - n_feat_min))
            ut.print_progress(i=i+pct_progress, n=n_rounds)
        rf.fit(X[:, selected_features], labels)
        importances = rf.feature_importances_
        if step is None:
            # Remove all features with the minimum importance
            min_importance = np.min(importances)
            features_to_remove = (importances == min_importance)
        else:
            # Remove a fixed number of least important features
            indices_to_remove = np.argsort(importances)[:step]
            features_to_remove = np.zeros_like(importances, dtype=bool)
            features_to_remove[indices_to_remove] = True

        selected_features[np.where(selected_features)[0][features_to_remove]] = False
        n_features -= np.sum(features_to_remove)
        # Evaluate the current feature set
        if n_features <= n_feat_max:
            current_score = np.mean(cross_val_score(rf, X[:, selected_features], labels, scoring=scoring, cv=n_cv))
            if current_score > best_score:
                best_score = current_score
                is_selected = selected_features.copy()
    return is_selected


# 2. Step: Computation of feature importance
def compute_feature_importance(X, labels=None, is_selected=None, list_model_classes=None,
                               list_model_kwargs=None):
    """Compute the average feature importance across multiple tree-based models."""
    selected_indices = np.where(is_selected)[0]
    X_selected = X[:, selected_indices]
    importances = np.zeros((len(list_model_classes), len(is_selected)))
    list_models = []
    for i, model_class in enumerate(list_model_classes):
        model_kwargs = list_model_kwargs[i] if list_model_kwargs is not None else {}
        model = model_class(**model_kwargs)
        model.fit(X_selected, labels)
        list_models.append(model)
        # Update importances only for selected features
        importances[i, selected_indices] = model.feature_importances_
    # Compute the average feature importance across all models
    avg_importance = np.mean(importances, axis=0)
    return avg_importance, list_models
