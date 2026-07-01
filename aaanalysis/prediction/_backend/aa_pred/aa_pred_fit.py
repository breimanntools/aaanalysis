"""
This is a script for the backend of the AAPred class: fitting models and obtaining prediction scores.
"""
import numpy as np

import aaanalysis.utils as ut


# I Helper Functions
def _positive_proba(model, X, label_pos):
    """Return the positive-class probability column for a fitted model."""
    proba = np.asarray(model.predict_proba(X))
    classes = list(model.classes_)
    idx_pos = classes.index(label_pos)
    return proba[:, idx_pos]


# II Main Functions
def fit_models(X, labels, list_model_classes=None, list_model_kwargs=None):
    """Fit every model on the full data and return the list of fitted estimators."""
    list_models = []
    for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs):
        model = model_class(**model_kwargs)
        model.fit(X, labels)
        list_models.append(model)
    return list_models


def predict_proba_models(X, list_models=None, label_pos=1):
    """Average positive-class probability across fitted models.

    Returns the mean positive-class score per sample and its std across models
    (0 for a single model).
    """
    probas = np.vstack([_positive_proba(model, X, label_pos=label_pos) for model in list_models])
    pred = probas.mean(axis=0)
    pred_std = probas.std(axis=0) if len(list_models) > 1 else np.zeros(len(pred))
    return pred, pred_std
