"""This is a script for the backend of the TreeModel.fit() method."""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import aaanalysis.utils as ut


# II Main methods
# 1. Step: Recursive feature elimination (RFE)
def _recursive_feature_elimination(X, labels=None, step=None, n_feat_max=50, n_feat_min=25, n_cv=5, scoring="f1",
                                   random_state=None, verbose=None, i=None, n_rounds=None, is_preselected=None):
    """Perform Recursive Feature Elimination to select the best feature set."""
    rf = RandomForestClassifier(random_state=random_state)
    selected_features = is_preselected.copy()
    n_total = sum(selected_features)
    n_features = n_total
    best_score = 0
    is_selected = selected_features.copy()
    while n_features > n_feat_min:
        if verbose:
            pct_progress = abs(1 - (n_features - n_feat_min) / (n_total - n_feat_min))
            ut.print_progress(i=i+pct_progress, n_total=n_rounds)
        rf.fit(X[:, selected_features], labels)
        importances = rf.feature_importances_
        if step is None:
            # Remove all features with the minimum importance
            min_importance = np.min(importances)
            features_to_remove = (importances == min_importance)
        else:
            # Remove a fixed number of the least important features (a maximum of n_features -1)
            _step = min(n_features-1, step)
            indices_to_remove = np.argsort(importances)[:_step]
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
def _compute_feature_importance(X, labels=None, is_selected=None, list_model_classes=None,
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


# Complete fit method
@ut.catch_undefined_metric_warning()
def fit_tree_based_models(X=None, labels=None,
                          list_model_classes=None, list_model_kwargs=None, is_preselected=None,
                          n_rounds=None, use_rfe=True, n_cv=5, n_feat_min=10, n_feat_max=50, metric="accuracy", step=1,
                          verbose=True, random_state=None):
    """Fit tree-based models"""
    n_features = X.shape[1]
    feature_importance = np.empty(shape=(n_rounds, n_features))
    is_selected_rounds = np.ones(shape=(n_rounds, n_features), dtype=bool)
    list_models_rounds = []
    if verbose and use_rfe:
        str_n_sets = "1 set" if n_rounds == 1 else f"{n_rounds} sets"
        start_message = (f"Tree Model starts recursive feature elimination "
                         f"to obtain {str_n_sets} of {n_feat_min} to {n_feat_max} features.")
        ut.print_start_progress(start_message=start_message)
    for i in range(n_rounds):
        if is_preselected is not None:
            is_selected = is_preselected.astype(bool)
        else:
            is_selected = np.ones(n_features, dtype=bool)
        # 1. Step: Recursive feature elimination
        if use_rfe:
            args = dict(labels=labels, n_feat_min=n_feat_min, n_feat_max=n_feat_max, n_cv=n_cv, scoring=metric,
                        step=step, random_state=random_state, verbose=verbose, i=i, n_rounds=n_rounds,
                        is_preselected=is_selected)
            is_selected = _recursive_feature_elimination(X, **args)
        is_selected_rounds[i, :] = is_selected
        # 2. Step: Compute feature importance
        avg_importance, list_models = _compute_feature_importance(X, labels=labels, is_selected=is_selected,
                                                                  list_model_classes=list_model_classes,
                                                                  list_model_kwargs=list_model_kwargs)
        feature_importance[i, :] = avg_importance
        list_models_rounds.append(list_models)
    if verbose and use_rfe:
        end_message = "Tree Model finished recursive feature elimination and saves results."
        ut.print_end_progress(end_message=end_message)
    return feature_importance, is_selected_rounds, list_models_rounds
