"""
This is a script for the backend of the ModelEvaluator class: repeated cross-validation scoring,
bootstrap confidence intervals, and paired model comparison.
"""
import itertools
from functools import partial
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, matthews_corrcoef)
from scipy.stats import wilcoxon

import aaanalysis.utils as ut


# Metric name -> (score function, needs_proba). Hard-label metrics score the ``predict`` output;
# ``roc_auc`` scores the positive-class probability. The names match ``ut.LIST_METRICS_MODELEVAL``,
# already validated by the frontend before reaching the backend. ``accuracy``, ``balanced_accuracy``
# and ``mcc`` are label-value agnostic; ``precision``/``recall``/``f1`` follow scikit-learn's binary
# ``pos_label=1`` convention, ``roc_auc`` its greater-label-is-positive convention.
METRIC_SCORE_FUNCS = {
    "accuracy": (accuracy_score, False),
    "balanced_accuracy": (balanced_accuracy_score, False),
    "precision": (partial(precision_score, zero_division=0), False),
    "recall": (partial(recall_score, zero_division=0), False),
    "f1": (partial(f1_score, zero_division=0), False),
    "mcc": (matthews_corrcoef, False),
    "roc_auc": (roc_auc_score, True),
}


# I Helper Functions
def _score_predictions(labels_true, pred_label, pred_proba, metrics):
    """Score one held-out fold's predictions across ``metrics``."""
    scores = {}
    for metric in metrics:
        score_func, needs_proba = METRIC_SCORE_FUNCS[metric]
        y_pred = pred_proba if needs_proba else pred_label
        scores[metric] = float(score_func(labels_true, y_pred))
    return scores


def _ordered_scores(df_scores, name, metric):
    """Fold scores for one model x metric in a stable (round, fold) order (for paired alignment)."""
    mask = (df_scores[ut.COL_MODEL] == name) & (df_scores[ut.COL_METRIC] == metric)
    df = df_scores[mask].sort_values([ut.COL_ROUND, ut.COL_FOLD])
    return df[ut.COL_SCORE].to_numpy(dtype=float)


def _paired_pvalue(diffs):
    """Two-sided Wilcoxon signed-rank p-value on the paired per-fold differences.

    Identical models (all differences zero) carry no evidence of a difference, so ``p=1.0``;
    a degenerate sample for which the test is undefined yields ``NaN``.
    """
    diffs = np.asarray(diffs, dtype=float)
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        _, p_value = wilcoxon(diffs)
    except ValueError:
        return float("nan")
    return float(p_value)


# II Main Functions
@ut.catch_undefined_metric_warning()
def comp_fold_scores(X, labels, list_estimators=None, list_model_names=None, metrics=None,
                     n_cv=5, n_rounds=1, random_state=None):
    """One score per (round, fold, model, metric) from repeated stratified cross-validation.

    Every model is scored on the **same** fold splits within a round (shared train/test indices),
    so per-fold scores are paired across models and a paired comparison is valid. Each round
    reshuffles with a distinct seed derived from ``random_state`` (``random_state + round``), so
    multi-round aggregation is reproducible. Returns a long-format DataFrame with columns
    ``round``, ``fold``, ``model``, ``metric``, ``score`` (one row per fold score).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    needs_proba = any(METRIC_SCORE_FUNCS[m][1] for m in metrics)
    rows = []
    for r in range(n_rounds):
        seed = None if random_state is None else random_state + r
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            for name, estimator in zip(list_model_names, list_estimators):
                est = clone(estimator)
                est.fit(X_train, y_train)
                pred_label = est.predict(X_test)
                pred_proba = est.predict_proba(X_test)[:, -1] if needs_proba else None
                scores = _score_predictions(y_test, pred_label, pred_proba, metrics)
                for metric in metrics:
                    rows.append([r, fold, name, metric, scores[metric]])
    return pd.DataFrame(rows, columns=[ut.COL_ROUND, ut.COL_FOLD, ut.COL_MODEL,
                                       ut.COL_METRIC, ut.COL_SCORE])


def aggregate_scores(df_scores, list_model_names=None, metrics=None, ci=0.95, ci_seed=None):
    """Aggregate per-fold scores into one row per (model, metric).

    ``score`` is the mean and ``score_std`` the (population) std over the ``n_cv * n_rounds`` fold
    scores; ``ci_low``/``ci_high`` are a percentile bootstrap CI of the mean (``NaN`` when
    ``ci is None``); ``n_scores`` is the number of fold scores aggregated.
    """
    rows = []
    for name in list_model_names:
        for metric in metrics:
            values = _ordered_scores(df_scores, name, metric)
            mean = float(np.mean(values))
            std = float(np.std(values))
            if ci is None:
                ci_low = ci_high = float("nan")
            else:
                _, ci_low, ci_high = ut.bootstrap_ci_(values=values, n_rounds=1000, ci=ci, seed=ci_seed)
            rows.append([name, metric, mean, std, float(ci_low), float(ci_high), int(len(values))])
    return pd.DataFrame(rows, columns=ut.COLS_EVAL_MODELEVAL)


def compare_models(df_scores, list_model_names=None, metric="mcc", ci=0.95, ci_seed=None):
    """Paired comparison of every model pair on a single ``metric`` over the shared folds.

    For each ordered pair ``(a, b)`` the per-fold paired difference ``d = score_a - score_b`` (same
    fold) gives the signed ``delta`` (mean), ``delta_std``, a percentile bootstrap CI on ``d``, and
    a two-sided Wilcoxon signed-rank ``p_value``. Returns one row per model pair.
    """
    rows = []
    for name_a, name_b in itertools.combinations(list_model_names, 2):
        scores_a = _ordered_scores(df_scores, name_a, metric)
        scores_b = _ordered_scores(df_scores, name_b, metric)
        diffs = scores_a - scores_b
        delta = float(np.mean(diffs))
        delta_std = float(np.std(diffs))
        if ci is None:
            ci_low = ci_high = float("nan")
        else:
            _, ci_low, ci_high = ut.bootstrap_ci_(values=diffs, n_rounds=1000, ci=ci, seed=ci_seed)
        p_value = _paired_pvalue(diffs)
        rows.append([name_a, name_b, metric, delta, delta_std, float(ci_low), float(ci_high), p_value])
    return pd.DataFrame(rows, columns=ut.COLS_COMPARE_MODELEVAL)
