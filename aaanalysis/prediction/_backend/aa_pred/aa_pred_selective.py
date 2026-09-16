"""
This is a script for the backend of the AAPred.eval_selective method: risk-coverage scoring.
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut

from .aa_pred_eval import METRIC_SCORE_FUNCS


# Metrics that are undefined on a single-class subset. Retaining only the most confident samples
# can leave one class behind, and these metrics then have no meaning (sklearn warns and returns a
# placeholder), so the selective table reports NaN instead of a misleading number.
LIST_METRICS_NEED_BOTH_CLASSES = ["balanced_accuracy", "roc_auc"]
# Metrics whose denominator can be empty on a retained subset (no predicted/true positive). They
# are scored with zero_division=0, so an empty positive set is a score of 0 rather than a warning.
DICT_METRIC_KWARGS = {"precision": {"zero_division": 0},
                      "recall": {"zero_division": 0},
                      "f1": {"zero_division": 0}}


# I Helper Functions
def _score_subset(labels, scores, metric, label_pos, label_neg):
    """Score one retained subset with one metric, or NaN where the metric is undefined.

    Hard-label metrics threshold the positive-class probability at 0.5 (the scores are
    probabilities, so 0.5 is the decision boundary); ``roc_auc`` scores the probability itself.
    """
    score_func, needs_proba = METRIC_SCORE_FUNCS[metric]
    if metric in LIST_METRICS_NEED_BOTH_CLASSES and len(np.unique(labels)) < 2:
        return float("nan")
    if needs_proba:
        labels = labels == label_pos
        y_pred = scores
    else:
        y_pred = np.where(scores >= 0.5, label_pos, label_neg)
    kwargs = DICT_METRIC_KWARGS.get(metric, {})
    if metric in {"precision", "recall", "f1"}:
        kwargs = {**kwargs, "pos_label": label_pos}
    return float(score_func(labels, y_pred, **kwargs))


def _comp_area(coverages, scores):
    """Mean height of the coverage-performance curve (trapezoidal area / coverage span).

    Normalizing by the span makes the number comparable across coverage grids and puts it on the
    same scale as the metric itself: a flat curve at 0.8 has an area of 0.8. It is NaN when the
    curve has fewer than two points, when the span is zero, or when any point is NaN (an
    undefined metric leaves a hole the area cannot bridge).
    """
    if len(coverages) < 2 or np.any(np.isnan(scores)):
        return float("nan")
    span = float(coverages[-1]) - float(coverages[0])
    if span <= 0:
        return float("nan")
    area = 0.0
    for i in range(len(coverages) - 1):
        width = float(coverages[i + 1]) - float(coverages[i])
        area += width * (float(scores[i]) + float(scores[i + 1])) / 2
    return area / span


# II Main Functions
def eval_selective_scores(labels, scores, confidence, metrics=None, coverages=None,
                          label_pos=1, label_neg=0):
    """Score every metric at every coverage level of a confidence-ranked sample set.

    Samples are ranked by ``confidence`` (most confident first, ties keeping their original
    order), and each coverage level retains the leading ``ceil(coverage * n_samples)`` of them,
    so a level is always non-empty and the highest level (1.0) retains every sample. Returns the
    long-format ``df_eval_selective`` with one row per (metric, coverage) plus the per-metric
    area under that metric's coverage-performance curve, repeated on each of its rows.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    n_samples = len(labels)
    # Stable descending sort: equally confident samples keep their input order, so the retained
    # subset of a tie-heavy ranking is reproducible rather than sort-implementation dependent.
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for metric in metrics:
        list_scores = []
        list_coverages = []
        list_rows = []
        for coverage in coverages:
            n_retained = int(np.ceil(float(coverage) * n_samples))
            n_retained = max(1, min(n_samples, n_retained))
            retained_coverage = n_retained / n_samples
            idx = order[:n_retained]
            score = _score_subset(labels=labels[idx], scores=scores[idx], metric=metric,
                                  label_pos=label_pos, label_neg=label_neg)
            list_scores.append(score)
            list_coverages.append(retained_coverage)
            list_rows.append([metric, retained_coverage, n_retained, score])
        area = _comp_area(coverages=list_coverages, scores=list_scores)
        rows += [row + [area] for row in list_rows]
    return pd.DataFrame(rows, columns=ut.COLS_EVAL_SELECTIVE)
