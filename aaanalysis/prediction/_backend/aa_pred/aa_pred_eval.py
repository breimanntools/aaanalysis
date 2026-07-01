"""
This is a script for the backend of the AAPred.eval method: model x metric x principle scoring.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer

import aaanalysis.utils as ut


# I Helper Functions
def _score_cv(model_class, model_kwargs, X, labels, metric, n_cv, random_state):
    """Cross-validated score (mean, std) for one model x metric."""
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    model = model_class(**model_kwargs)
    scores = cross_val_score(model, X, labels, cv=cv, scoring=metric)
    return float(np.mean(scores)), float(np.std(scores))


def _score_holdout(fitted_model, X_holdout, labels_holdout, metric):
    """Held-out score for one already-fitted model x metric (std is NaN: single estimate)."""
    scorer = get_scorer(metric)
    return float(scorer(fitted_model, X_holdout, labels_holdout)), float("nan")


# II Main Functions
def eval_models(X, labels, list_model_classes=None, list_model_kwargs=None,
                list_models=None, metrics=None, n_cv=5, random_state=None,
                X_holdout=None, labels_holdout=None):
    """Score every model x metric by cross-validation and (optionally) on a held-out set.

    Returns a long-format ``df_eval`` with one row per (model, metric, principle).
    """
    rows = []
    for i, (model_class, model_kwargs) in enumerate(zip(list_model_classes, list_model_kwargs)):
        model_name = model_class.__name__
        for metric in metrics:
            score, score_std = _score_cv(model_class=model_class, model_kwargs=model_kwargs,
                                         X=X, labels=labels, metric=metric,
                                         n_cv=n_cv, random_state=random_state)
            rows.append([model_name, metric, ut.STR_PRINCIPLE_CV, score, score_std])
            if X_holdout is not None:
                score, score_std = _score_holdout(fitted_model=list_models[i], X_holdout=X_holdout,
                                                  labels_holdout=labels_holdout, metric=metric)
                rows.append([model_name, metric, ut.STR_PRINCIPLE_HOLDOUT, score, score_std])
    df_eval = pd.DataFrame(rows, columns=ut.COLS_EVAL_PRED)
    return df_eval
