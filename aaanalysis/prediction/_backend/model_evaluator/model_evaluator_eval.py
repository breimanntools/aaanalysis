"""
This is a script for the backend of the ModelEvaluator class: repeated cross-validation scoring,
bootstrap confidence intervals, paired model comparison, and learning curves.
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


def _seed_estimator(estimator, random_state):
    """Seed a fresh clone with ``random_state`` only where the estimator left it unset.

    Keeps per-call ``random_state`` reproducible even when the constructor seed was ``None`` (the
    estimators would otherwise stay unseeded), while respecting a seed the user pinned on a passed
    estimator instance.
    """
    est = clone(estimator)
    if random_state is not None:
        params = est.get_params(deep=False)
        if "random_state" in params and params["random_state"] is None:
            est.set_params(random_state=random_state)
    return est


def _ordered_scores(df_scores, dict_group):
    """Fold scores of one group in a stable (round, fold) order (for paired alignment).

    ``dict_group`` maps each grouping column to the value selecting the group (``model`` x
    ``metric`` for the evaluation and comparison tables, plus ``train_size`` for the curve).
    """
    mask = np.ones(len(df_scores), dtype=bool)
    for col, val in dict_group.items():
        mask &= (df_scores[col] == val).to_numpy()
    df = df_scores[mask].sort_values([ut.COL_ROUND, ut.COL_FOLD])
    return df[ut.COL_SCORE].to_numpy(dtype=float)


def _fit_and_score(estimator, random_state, X_train, labels_train, X_test, labels_test,
                   metrics, needs_proba):
    """Fit a seeded clone on one training (sub)set and score it on the untouched held-out fold."""
    est = _seed_estimator(estimator, random_state)
    est.fit(X_train, labels_train)
    pred_label = est.predict(X_test)
    pred_proba = est.predict_proba(X_test)[:, -1] if needs_proba else None
    return _score_predictions(labels_test, pred_label, pred_proba, metrics)


def _stratified_subset(labels_train, class_orders, size):
    """Positions of a stratified subset of ``size`` samples within one training fold.

    Class counts follow the class proportions of the training fold (largest-remainder rounding)
    with at least one sample per class, and each class contributes the first samples of its fixed
    random order (``class_orders``). For binary labels the counts grow monotonically with ``size``,
    so the subsets of one fold are nested. Positions are returned sorted, so the full training
    fold keeps the exact sample order used by :func:`comp_fold_scores`.
    """
    classes = sorted(class_orders)
    n_train = len(labels_train)
    exact = np.array([size * len(class_orders[c]) / n_train for c in classes], dtype=float)
    counts = np.floor(exact).astype(int)
    remainder = size - int(counts.sum())
    if remainder > 0:
        # Stable tie-break on the class order, so the allocation is deterministic.
        for i in np.argsort(-(exact - counts), kind="stable")[:remainder]:
            counts[i] += 1
    # Guarantee at least one sample per class (every estimator needs both classes to fit).
    for i in range(len(classes)):
        if counts[i] == 0:
            counts[i] = 1
            counts[int(np.argmax(counts))] -= 1
    pos = np.concatenate([class_orders[c][:k] for c, k in zip(classes, counts)])
    return np.sort(pos)


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
                scores = _fit_and_score(estimator, random_state, X_train, y_train, X_test, y_test,
                                        metrics, needs_proba)
                for metric in metrics:
                    rows.append([r, fold, name, metric, scores[metric]])
    return pd.DataFrame(rows, columns=[ut.COL_ROUND, ut.COL_FOLD, ut.COL_MODEL,
                                       ut.COL_METRIC, ut.COL_SCORE])


def aggregate_grouped(df_scores, group_cols=None, list_groups=None, ci=0.95, ci_seed=None):
    """Aggregate per-fold scores into one row per group (the single aggregation used everywhere).

    ``group_cols`` are the grouping columns and ``list_groups`` their value tuples in output order:
    ``(model, metric)`` for :func:`aggregate_scores` and ``(model, train_size, metric)`` for
    :func:`comp_learning_curve`, so both paths share one mean / std / bootstrap implementation.
    ``score`` is the mean and ``score_std`` the (population) std over the fold scores of the group;
    ``ci_low``/``ci_high`` are a percentile bootstrap CI of the mean (``NaN`` when ``ci is None``);
    ``n_scores`` is the number of fold scores aggregated.
    """
    rows = []
    for group in list_groups:
        values = _ordered_scores(df_scores, dict(zip(group_cols, group)))
        mean = float(np.mean(values))
        std = float(np.std(values))
        if ci is None:
            ci_low = ci_high = float("nan")
        else:
            # One bootstrap for the whole class: this is the shared helper that the public
            # 'comp_bootstrap_ci' wraps, so run, eval and the learning curve report the same CI.
            _, ci_low, ci_high = ut.bootstrap_ci_(values=values, n_rounds=1000, ci=ci, seed=ci_seed)
        rows.append([*group, mean, std, float(ci_low), float(ci_high), int(len(values))])
    columns = list(group_cols) + [ut.COL_SCORE, ut.COL_SCORE_STD, ut.COL_CI_LOW, ut.COL_CI_HIGH,
                                  ut.COL_N_SCORES]
    return pd.DataFrame(rows, columns=columns)


def aggregate_scores(df_scores, list_model_names=None, metrics=None, ci=0.95, ci_seed=None):
    """Aggregate per-fold scores into one row per (model, metric) (columns ``COLS_EVAL_MODELEVAL``)."""
    list_groups = [(name, metric) for name in list_model_names for metric in metrics]
    return aggregate_grouped(df_scores, group_cols=[ut.COL_MODEL, ut.COL_METRIC],
                             list_groups=list_groups, ci=ci, ci_seed=ci_seed)


def compare_models(df_scores, list_model_names=None, metric="mcc", ci=0.95, ci_seed=None):
    """Paired comparison of every model pair on a single ``metric`` over the shared folds.

    For each ordered pair ``(a, b)`` the per-fold paired difference ``d = score_a - score_b`` (same
    fold) gives the signed ``delta`` (mean), ``delta_std``, a percentile bootstrap CI on ``d``, and
    a two-sided Wilcoxon signed-rank ``p_value``. Returns one row per model pair.
    """
    rows = []
    for name_a, name_b in itertools.combinations(list_model_names, 2):
        scores_a = _ordered_scores(df_scores, {ut.COL_MODEL: name_a, ut.COL_METRIC: metric})
        scores_b = _ordered_scores(df_scores, {ut.COL_MODEL: name_b, ut.COL_METRIC: metric})
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


@ut.catch_undefined_metric_warning()
def comp_learning_curve(X, labels, list_estimators=None, list_model_names=None, metrics=None,
                        train_sizes=None, train_fracs=None, n_cv=5, n_rounds=1, ci=None,
                        random_state=None):
    """Learning curve: cross-validated scores per (model, training size, metric).

    Uses the same repeated stratified folds as :func:`comp_fold_scores` (``random_state + round``).
    Within each training fold, every model is fitted on a stratified subset of the training fold
    and scored on the **full, untouched test fold**, so the test set never changes with the
    training size and never enters training. ``train_sizes`` are the curve-point labels resolved by
    the frontend. With ``train_fracs`` (one fraction per label) each fold resolves its own subset
    size from its own training-fold size, so the fraction ``1.0`` is the fold's complete training
    set and reproduces :func:`comp_fold_scores` exactly even for unequal folds; without them the
    labels are absolute counts used as given. The per-fold scores are aggregated by the same
    :func:`aggregate_grouped` helper as :func:`aggregate_scores`, into one row per (model, training
    size, metric).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    needs_proba = any(METRIC_SCORE_FUNCS[m][1] for m in metrics)
    rows = []
    for r in range(n_rounds):
        seed = None if random_state is None else random_state + r
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
        rng = np.random.default_rng(seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
            y_train_fold = labels[train_idx]
            n_train_fold = len(train_idx)
            class_orders = {c: rng.permutation(np.flatnonzero(y_train_fold == c))
                            for c in np.unique(y_train_fold)}
            X_test, y_test = X[test_idx], labels[test_idx]
            for i, size in enumerate(train_sizes):
                if train_fracs is not None:
                    # Resolve the fraction against THIS fold (floored at one sample per class), so
                    # a larger fold never silently drops samples and 1.0 keeps its whole fold.
                    size = min(n_train_fold, max(2, int(np.floor(train_fracs[i] * n_train_fold))))
                sub_idx = train_idx[_stratified_subset(y_train_fold, class_orders, size)]
                # Derived invariant: a training subset must never touch the held-out fold.
                if np.intersect1d(sub_idx, test_idx).size != 0:
                    raise RuntimeError("Learning-curve training subset overlaps the test fold.")
                X_train, y_train = X[sub_idx], labels[sub_idx]
                for name, estimator in zip(list_model_names, list_estimators):
                    scores = _fit_and_score(estimator, random_state, X_train, y_train, X_test,
                                            y_test, metrics, needs_proba)
                    for metric in metrics:
                        # The row carries the curve-point label, not the fold-resolved size, so
                        # the folds of one curve point aggregate together.
                        rows.append([r, fold, name, int(train_sizes[i]), metric, scores[metric]])
    df_scores = pd.DataFrame(rows, columns=[ut.COL_ROUND, ut.COL_FOLD, ut.COL_MODEL,
                                            ut.COL_TRAIN_SIZE, ut.COL_METRIC, ut.COL_SCORE])
    list_groups = [(name, int(size), metric) for name in list_model_names
                   for size in train_sizes for metric in metrics]
    return aggregate_grouped(df_scores, group_cols=[ut.COL_MODEL, ut.COL_TRAIN_SIZE, ut.COL_METRIC],
                             list_groups=list_groups, ci=ci, ci_seed=random_state)
