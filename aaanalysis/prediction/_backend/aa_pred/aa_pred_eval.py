"""
This is a script for the backend of the AAPred.eval method: model x metric x principle scoring.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer

import aaanalysis.utils as ut


# I Helper Functions
def _score_cv(estimator, X, labels, metric, n_cv, random_state):
    """Cross-validated score (mean, std) for one estimator x metric."""
    from sklearn.base import clone
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clone(estimator), X, labels, cv=cv, scoring=metric)
    return float(np.mean(scores)), float(np.std(scores))


def _score_holdout(fitted_model, X_holdout, labels_holdout, metric):
    """Held-out score for one already-fitted model x metric (std is NaN: single estimate)."""
    scorer = get_scorer(metric)
    return float(scorer(fitted_model, X_holdout, labels_holdout)), float("nan")


def _eval_one(X, labels, list_estimators, list_models, metrics, n_cv, random_state,
              X_holdout, labels_holdout):
    """Rows [model, metric, principle, score, score_std] for one feature matrix ``X``."""
    rows = []
    for i, estimator in enumerate(list_estimators):
        model_name = type(estimator).__name__
        for metric in metrics:
            score, score_std = _score_cv(estimator=estimator, X=X, labels=labels, metric=metric,
                                         n_cv=n_cv, random_state=random_state)
            rows.append([model_name, metric, ut.STR_PRINCIPLE_CV, score, score_std])
            if X_holdout is not None:
                score, score_std = _score_holdout(fitted_model=list_models[i], X_holdout=X_holdout,
                                                  labels_holdout=labels_holdout, metric=metric)
                rows.append([model_name, metric, ut.STR_PRINCIPLE_HOLDOUT, score, score_std])
    return rows


# II Main Functions
def eval_models(X, labels, list_estimators=None, list_models=None, metrics=None, n_cv=5,
                random_state=None, X_holdout=None, labels_holdout=None, dict_X_baseline=None):
    """Score every model x metric by cross-validation and (optionally) on a held-out set.

    Returns a long-format ``df_eval`` with one row per (model, metric, principle). When
    ``dict_X_baseline`` (``{kind: X_baseline}``) is given, the bound-feature rows are tagged
    ``'cpp'`` and each baseline matrix is scored (cross-validation only, same models/folds) and
    appended, tagged by its kind, under a leading ``features`` column.
    """
    rows = _eval_one(X, labels, list_estimators, list_models, metrics, n_cv, random_state,
                     X_holdout, labels_holdout)
    if not dict_X_baseline:
        return pd.DataFrame(rows, columns=ut.COLS_EVAL_PRED)
    # Baseline-comparison mode: tag the bound-feature rows, then append CV rows per baseline.
    tagged = [[ut.STR_FEATURES_CPP] + row for row in rows]
    for kind, X_baseline in dict_X_baseline.items():
        base_rows = _eval_one(X_baseline, labels, list_estimators, None, metrics, n_cv,
                              random_state, None, None)
        tagged += [[kind] + row for row in base_rows]
    return pd.DataFrame(tagged, columns=ut.COLS_EVAL_PRED_FEATURES)
