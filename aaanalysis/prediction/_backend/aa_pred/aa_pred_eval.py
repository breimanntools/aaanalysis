"""
This is a script for the backend of the AAPred.eval method: model x metric x principle scoring.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (get_scorer, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)

import aaanalysis.utils as ut


# Metric name -> (score function, needs_proba). The pooled principle scores each metric once on
# the held-out predictions, so it applies the bare metric function (not a per-estimator scorer):
# five metrics score hard class labels, roc_auc scores the positive-class probability. The names
# match ``ut.LIST_METRICS_PRED``, already validated by the frontend before reaching the backend.
METRIC_SCORE_FUNCS = {
    "accuracy": (accuracy_score, False),
    "balanced_accuracy": (balanced_accuracy_score, False),
    "precision": (precision_score, False),
    "recall": (recall_score, False),
    "f1": (f1_score, False),
    "roc_auc": (roc_auc_score, True),
}


# I Helper Functions
def _score_cv(estimator, X, labels, metric, n_cv, random_state):
    """Cross-validated score (mean, std) for one estimator x metric."""
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clone(estimator), X, labels, cv=cv, scoring=metric)
    return float(np.mean(scores)), float(np.std(scores))


def _score_cv_pooled(estimator, X, labels, metrics, cv):
    """Pooled out-of-fold scores for one estimator across ``metrics`` under a custom splitter.

    Runs ``cross_val_predict`` once over ``cv`` (again for probabilities only if a metric needs
    them) and applies each metric a single time on the pooled held-out predictions, reproducing
    ``metric(labels, cross_val_predict(estimator, X, labels, cv=cv))``. Returns ``{metric:
    (score, nan)}`` — the std is NaN because a pooled score is a single estimate, not a fold
    distribution.
    """
    labels = np.asarray(labels)
    pred_label = cross_val_predict(clone(estimator), X, labels, cv=cv)
    pred_proba = None
    if any(METRIC_SCORE_FUNCS[m][1] for m in metrics):
        # Positive class = the greater label (sklearn's binary convention for roc_auc), whose
        # probability is the last column of the class-sorted predict_proba output.
        proba = cross_val_predict(clone(estimator), X, labels, cv=cv, method="predict_proba")
        pred_proba = proba[:, -1]
    dict_scores = {}
    for metric in metrics:
        score_func, needs_proba = METRIC_SCORE_FUNCS[metric]
        y_pred = pred_proba if needs_proba else pred_label
        dict_scores[metric] = (float(score_func(labels, y_pred)), float("nan"))
    return dict_scores


def _score_holdout(fitted_model, X_holdout, labels_holdout, metric):
    """Held-out score for one already-fitted model x metric (std is NaN: single estimate)."""
    scorer = get_scorer(metric)
    return float(scorer(fitted_model, X_holdout, labels_holdout)), float("nan")


def _eval_one(X, labels, list_estimators, list_models, metrics, n_cv, random_state,
              X_holdout, labels_holdout, cv=None):
    """Rows [model, metric, principle, score, score_std] for one feature matrix ``X``.

    With ``cv=None`` (default) the cross-validation rows are per-fold ``StratifiedKFold`` scores
    tagged ``'cv'`` (byte-identical to before). With a custom splitter ``cv`` they are pooled
    out-of-fold scores tagged ``'cv_pooled'``. The per-metric row order (cv/pooled row, then the
    optional holdout row) is unchanged either way.
    """
    rows = []
    for i, estimator in enumerate(list_estimators):
        model_name = type(estimator).__name__
        dict_pooled = None
        if cv is not None:
            dict_pooled = _score_cv_pooled(estimator=estimator, X=X, labels=labels,
                                           metrics=metrics, cv=cv)
        for metric in metrics:
            if cv is None:
                score, score_std = _score_cv(estimator=estimator, X=X, labels=labels, metric=metric,
                                             n_cv=n_cv, random_state=random_state)
                rows.append([model_name, metric, ut.STR_PRINCIPLE_CV, score, score_std])
            else:
                score, score_std = dict_pooled[metric]
                rows.append([model_name, metric, ut.STR_PRINCIPLE_CV_POOLED, score, score_std])
            if X_holdout is not None:
                score, score_std = _score_holdout(fitted_model=list_models[i], X_holdout=X_holdout,
                                                  labels_holdout=labels_holdout, metric=metric)
                rows.append([model_name, metric, ut.STR_PRINCIPLE_HOLDOUT, score, score_std])
    return rows


# II Main Functions
def eval_models(X, labels, list_estimators=None, list_models=None, metrics=None, n_cv=5,
                random_state=None, X_holdout=None, labels_holdout=None, dict_X_baseline=None,
                cv=None):
    """Score every model x metric by cross-validation and (optionally) on a held-out set.

    Returns a long-format ``df_eval`` with one row per (model, metric, principle). With a custom
    splitter ``cv`` the cross-validation rows are pooled out-of-fold scores (``'cv_pooled'``);
    otherwise they are per-fold ``StratifiedKFold`` scores (``'cv'``). When ``dict_X_baseline``
    (``{kind: X_baseline}``) is given, the bound-feature rows are tagged ``'cpp'`` and each
    baseline matrix is scored (cross-validation only, same models/folds/splitter) and appended,
    tagged by its kind, under a leading ``features`` column.
    """
    rows = _eval_one(X, labels, list_estimators, list_models, metrics, n_cv, random_state,
                     X_holdout, labels_holdout, cv=cv)
    if not dict_X_baseline:
        return pd.DataFrame(rows, columns=ut.COLS_EVAL_PRED)
    # Baseline-comparison mode: tag the bound-feature rows, then append CV rows per baseline.
    tagged = [[ut.STR_FEATURES_CPP] + row for row in rows]
    for kind, X_baseline in dict_X_baseline.items():
        base_rows = _eval_one(X_baseline, labels, list_estimators, None, metrics, n_cv,
                              random_state, None, None, cv=cv)
        tagged += [[kind] + row for row in base_rows]
    return pd.DataFrame(tagged, columns=ut.COLS_EVAL_PRED_FEATURES)
