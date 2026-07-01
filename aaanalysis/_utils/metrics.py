"""
This is a script for utility functions for statistical measures.
"""
import numpy as np
from scipy.stats import entropy, gaussian_kde
from collections import OrderedDict
from scipy.spatial import distance
from joblib import Parallel, delayed
import os
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict, check_cv
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from ..config import resolve_n_jobs

DTYPE = np.float64


# AUC adjusted
def _compute_auc_sorted(X, labels):
    """Compute AUC for a subset of features using the same ranking approach as roc_auc_score."""
    n_samples, n_features = X.shape
    auc_values = np.empty(n_features, dtype=np.float64)
    for j in range(n_features):
        # Rank the feature values, handling ties properly
        ranks = rankdata(X[:, j])  # Average ranking for ties
        pos = np.sum(labels)
        neg = n_samples - pos
        if pos == 0 or neg == 0:
            auc_values[j] = 0.5  # Undefined AUC when only one class is present
            continue
        # Compute AUC using the Mann-Whitney U statistic
        rank_sum_pos = np.sum(ranks[labels == 1])
        auc_values[j] = (rank_sum_pos - (pos * (pos + 1) / 2)) / (pos * neg)
    return np.round(auc_values - 0.5, 3)


def auc_adjusted_(X=None, labels=None, label_test=1, n_jobs=None):
    """Get adjusted ROC AUC with pre-sorting and parallel computation."""
    # Get binary labels and precompute ranks for all features
    labels_binary = np.array([int(y == label_test) for y in labels], dtype=DTYPE)
    ranked_X = np.apply_along_axis(rankdata, 0, X)

    n_jobs = resolve_n_jobs(n_jobs=n_jobs, n_work=X.shape[1])

    if n_jobs == 1:
        return _compute_auc_sorted(ranked_X, labels_binary)

    feature_chunks = np.array_split(np.arange(X.shape[1]), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_compute_auc_sorted)(ranked_X[:, chunk], labels_binary)
                           for chunk in feature_chunks)
    auc_values = np.concatenate(results)
    return auc_values


# Bayesian Information Criterion for clusters
def _cluster_center(X):
    """Compute cluster center (i.e., arithmetical mean over all data points/observations of a cluster)"""
    return X.mean(axis=0)[np.newaxis, :]


def _compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    labels_centers = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in labels_centers]
    centers = np.concatenate([_cluster_center(X[mask]) for mask in list_masks]).round(3)
    labels_centers = np.array(labels_centers)
    return centers, labels_centers


def bic_score_(X, labels=None):
    """Computes the Bayesian Information Criterion (BIC) metric for given clusters."""
    epsilon = 1e-10  # prevent division by zero

    # Check if labels match to number of clusters
    n_classes = len(set(labels))
    n_samples, n_features = X.shape
    if n_classes >= n_samples:
        raise ValueError(f"Number of classes in 'labels' ({n_classes}) must be smaller than n_samples ({n_samples})")
    if n_features == 0:
        raise ValueError(f"'n_features' should not be 0")

    # Map labels to increasing order starting with 0
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    labels = inverse
    centers, center_labels = _compute_centers(X, labels=labels)
    size_clusters = np.bincount(labels)

    # Compute variance over all clusters
    list_masks = [labels == label for label in center_labels]
    sum_squared_dist = sum([sum(distance.cdist(X[mask], [center], 'euclidean') ** 2) for mask, center in zip(list_masks, centers)])

    # Compute between-cluster variance
    denominator = max((n_samples - n_classes) * n_features, epsilon)
    bet_clu_var = max((1.0 / denominator) * sum_squared_dist, epsilon)

    # Compute BIC components
    const_term = 0.5 * n_classes * np.log(n_samples) * (n_features + 1)

    log_size_clusters = np.log(size_clusters + epsilon)
    log_n_samples = np.log(n_samples + epsilon)
    log_bcv = np.log(2 * np.pi * bet_clu_var)

    bic_components = size_clusters * (log_size_clusters - log_n_samples) - 0.5 * size_clusters * n_features * log_bcv - 0.5 * (size_clusters - 1) * n_features
    bic = np.sum(bic_components) - const_term
    return bic


# Kullback-Leibler Divergence
def _comp_kld_for_feature(args):
    """Compute KLD for a single feature."""
    x1, x2 = args
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)
    xmin = min(x1.min(), x2.min())
    xmax = max(x1.max(), x2.max())
    x = np.linspace(xmin, xmax, 1000)
    density1 = kde1(x)
    density2 = kde2(x)
    return entropy(density1, density2)


def _comp_kld_chunk(X1_chunk, X2_chunk):
    """Compute KLD for a contiguous block of features (one parallel unit)."""
    return np.array([_comp_kld_for_feature((X1_chunk[:, i], X2_chunk[:, i]))
                     for i in range(X1_chunk.shape[1])])


def kullback_leibler_divergence_(X=None, labels=None, label_test=0, label_ref=1, n_jobs=1):
    """Calculate the average Kullback-Leibler Divergence (KLD) for each feature.

    Per-feature KLDs are independent, so chunking across workers yields the same
    values in the same order (identical to the serial path). ``n_jobs=1`` keeps the
    original in-process loop; ``None`` defers to the optimized worker count.
    """
    mask_test = np.asarray([x == label_test for x in labels])
    mask_ref = np.asarray([x == label_ref for x in labels])
    X1 = X[mask_ref]
    X2 = X[mask_test]
    n_jobs = resolve_n_jobs(n_jobs=n_jobs, n_work=X.shape[1])
    if n_jobs == 1:
        # Prepare arguments for each feature and compute KLD for each feature
        args = [(X1[:, i], X2[:, i]) for i in range(X.shape[1])]
        return np.array([_comp_kld_for_feature(arg) for arg in args])
    feature_chunks = np.array_split(np.arange(X.shape[1]), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_comp_kld_chunk)(X1[:, chunk], X2[:, chunk])
                           for chunk in feature_chunks)
    return np.concatenate(results)


# Per-protein site-localization metrics (windowed protease / PTM prediction)
def _avg_precision_at_positions(scores, pos_idx, tolerance=0):
    """Average precision for one protein's per-residue ``scores`` where the
    0-based indices in ``pos_idx`` are positive sites.

    NaN scores are dropped before ranking. With ``tolerance > 0`` a predicted
    rank counts as a hit if it lies within ``tolerance`` residues of any
    positive (off-by-one positional jitter). Returns ``np.nan`` when the protein
    has no positives or no finite scores."""
    scores = np.asarray(scores, dtype=DTYPE)
    finite = ~np.isnan(scores)
    if not finite.any():
        return np.nan
    pos_set = set(int(p) for p in pos_idx)
    if not pos_set:
        return np.nan
    idx_finite = np.flatnonzero(finite)
    # Rank finite positions by descending score (stable for ties).
    order = idx_finite[np.argsort(-scores[idx_finite], kind="stable")]
    n_pos = len(pos_set)
    hits = 0
    sum_prec = 0.0
    matched = set()
    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        # A position is a true hit if within +/- tolerance of an unmatched positive.
        target = None
        for d in range(-tolerance, tolerance + 1):
            cand = idx + d
            if cand in pos_set and cand not in matched:
                target = cand
                break
        if target is not None:
            matched.add(target)
            hits += 1
            sum_prec += hits / rank
    if hits == 0:
        return 0.0
    return sum_prec / n_pos


def per_protein_ap_(list_scores=None, list_positions=None, tolerance=0):
    """Per-protein average precision over a list of (scores, positive-positions).

    ``list_positions`` holds 0-based positive indices per protein. Returns a 1D
    array of per-protein AP (``np.nan`` for proteins with no positives / no
    finite scores), so callers can ``np.nanmean`` for the dataset score."""
    out = []
    for scores, positions in zip(list_scores, list_positions):
        out.append(_avg_precision_at_positions(scores, positions, tolerance=tolerance))
    return np.asarray(out, dtype=DTYPE)


# Detection metrics at a fixed score threshold (pooled over proteins)
def detection_metrics_(list_scores=None, list_positions=None, threshold=0.5, tolerance=0):
    """Pool TP/FP/FN/TN across proteins at a fixed ``threshold`` and return
    ``dict(recall, precision, f1, mcc, tp, fp, fn, tn)``.

    A residue scoring ``>= threshold`` is a positive call. With ``tolerance>0``
    a call within ``tolerance`` of a true site counts as TP and that site is
    consumed (so each true site is credited at most once). NaN scores are
    ignored (neither called nor counted as residues)."""
    tp = fp = fn = tn = 0
    for scores, positions in zip(list_scores, list_positions):
        scores = np.asarray(scores, dtype=DTYPE)
        finite = ~np.isnan(scores)
        pos_set = set(int(p) for p in positions)
        called = set(int(i) for i in np.flatnonzero(finite & (scores >= threshold)))
        matched_sites = set()
        matched_calls = set()
        for c in called:
            for d in range(-tolerance, tolerance + 1):
                cand = c + d
                if cand in pos_set and cand not in matched_sites:
                    matched_sites.add(cand)
                    matched_calls.add(c)
                    break
        tp += len(matched_sites)
        fp += len(called - matched_calls)
        fn += len(pos_set - matched_sites)
        # Negatives = finite residues that are neither a positive site nor a call.
        n_finite = int(finite.sum())
        tn += n_finite - len(called) - len(pos_set - matched_sites)
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision) and not np.isnan(recall)
          else (0.0 if (tp + fp + fn) else np.nan))
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else np.nan
    return {"recall": recall, "precision": precision, "f1": f1, "mcc": mcc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# Bootstrap confidence interval over a per-protein metric vector
def bootstrap_ci_(values=None, n_rounds=1000, ci=0.95, seed=None):
    """Percentile bootstrap CI of the mean of ``values`` (NaN-aware).

    Resamples proteins with replacement ``n_rounds`` times; returns
    ``(mean, low, high)`` for the central ``ci`` interval. Deterministic given
    ``seed``."""
    values = np.asarray(values, dtype=DTYPE)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_rounds, dtype=DTYPE)
    for i in range(n_rounds):
        means[i] = np.mean(values[rng.integers(0, n, size=n)])
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(np.mean(values)), float(low), float(high)


# Peak-preserving score smoothing (NaN-aware, pure numpy)
def _triangular_kernel(window):
    """Normalized triangular weights of length ``2*window+1``."""
    ramp = np.arange(1, window + 1, dtype=DTYPE)
    w = np.concatenate([ramp, [window + 1.0], ramp[::-1]])
    return w / w.sum()


def _gaussian_kernel(window, sigma):
    """Normalized Gaussian weights of length ``2*window+1``."""
    x = np.arange(-window, window + 1, dtype=DTYPE)
    w = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


def smooth_scores_(scores=None, method="triangular", window=2, sigma=None,
                   peak_preserving=True):
    """Smooth a 1D per-residue ``scores`` vector with a triangular or Gaussian
    kernel, NaN-aware, optionally peak-preserving.

    NaNs are ignored in the weighted average (weights renormalized over finite
    neighbours); positions with no finite neighbour stay NaN. When
    ``peak_preserving`` the output is ``max(smoothed, raw)`` elementwise so true
    peaks are never attenuated below their original height."""
    scores = np.asarray(scores, dtype=DTYPE)
    n = scores.size
    if method == "triangular":
        kernel = _triangular_kernel(window)
    else:
        kernel = _gaussian_kernel(window, sigma if sigma is not None else max(window / 2.0, 1e-6))
    finite = ~np.isnan(scores)
    filled = np.where(finite, scores, 0.0)
    out = np.full(n, np.nan, dtype=DTYPE)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        k_lo = lo - (i - window)
        k_hi = k_lo + (hi - lo)
        w = kernel[k_lo:k_hi] * finite[lo:hi]
        wsum = w.sum()
        if wsum > 0:
            out[i] = np.dot(w, filled[lo:hi]) / wsum
    if peak_preserving:
        out = np.where(np.isnan(out), scores,
                       np.where(np.isnan(scores), out, np.maximum(out, scores)))
    return out


# Feature-set evaluation (model + CV + metric scorer; incl. PU mask-known-positives CV)
# Registry of supported (y_true, y_pred) -> score classification metrics. Every metric
# is bounded in [0, 1], so the ``* 100`` scaling in ``eval_features_`` yields a genuine
# percentage in [0, 100] (this reproduces the notebook's
# ``balanced_accuracy_score(y, y_pred) * 100`` recipe). Signed-range metrics (e.g.
# Matthews correlation coefficient, [-1, 1]) are intentionally excluded so the reported
# score keeps a single percentage semantics.
DICT_METRICS_EVAL = {
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}
LIST_METRICS_EVAL = list(DICT_METRICS_EVAL)


def get_default_eval_model_():
    """Return the default feature-set evaluator: a linear-kernel SVM (matches the
    γ-secretase notebook's ``SVC(kernel='linear')`` benchmark estimator)."""
    return SVC(kernel="linear")


def _masked_cv_score(X, y, model=None, cv=None, f_score=None, mask=None):
    """PU mask-known-positives CV: known positives stay in every training fold but
    are never scored as test points (port of ``cv_leave_one_out_masked``).

    Generalized to any CV splitter: for each fold the masked test indices are dropped
    from scoring but folded back into the training set, so the masked known positives
    still inform every scored fold (for leave-one-out this is a no-op — a masked
    single-sample test fold is skipped outright — so the default path is unchanged)."""
    mask = np.asarray(mask, dtype=bool)
    y_true, y_pred = [], []
    for train_index, test_index in cv.split(X, y):
        keep = test_index[~mask[test_index]]
        if len(keep) == 0:
            continue
        # Masked known positives in this test fold are never scored but stay in training.
        masked_in_test = test_index[mask[test_index]]
        train_full = np.concatenate([train_index, masked_in_test]).astype(int)
        est = clone(model)
        est.fit(X[train_full], y[train_full])
        preds = est.predict(X[keep])
        y_true.extend(np.asarray(y)[keep].tolist())
        y_pred.extend(np.asarray(preds).tolist())
    if len(y_true) == 0:
        # Defensive net: the frontend already rejects an all-masked input, so this only
        # triggers for a custom splitter whose test folds never cover an unmasked sample.
        raise ValueError("No unmasked test points remain to score; check the 'cv' splitter "
                         "and 'mask_known_pos' combination.")
    return f_score(np.array(y_true), np.array(y_pred))


def eval_features_(X=None, labels=None, model=None, cv=None, metric="balanced_accuracy",
                   mask_known_pos=None, random_state=None):
    """Score a feature set by cross-validated classification performance.

    Reproduces the notebook recipe when ``model``/``cv`` default to a linear SVM and
    leave-one-out: ``balanced_accuracy_score(y, cross_val_predict(...)) * 100``. When
    ``mask_known_pos`` is given, the PU mask-known-positives CV variant is used."""
    X = np.asarray(X)
    y = np.asarray(labels)
    f_score = DICT_METRICS_EVAL[metric]
    model = get_default_eval_model_() if model is None else clone(model)
    # Honor the random_state contract for stochastic estimators (e.g. RandomForest)
    if random_state is not None and "random_state" in model.get_params():
        model.set_params(random_state=random_state)
    # Normalize cv to a splitter object: None -> leave-one-out, int -> stratified k-fold
    # (matching cross_val_predict's own int handling), splitter -> used as-is. This makes
    # the masked path work for an int cv too (it needs a real ``.split``).
    cv = LeaveOneOut() if cv is None else check_cv(cv, y, classifier=True)
    if mask_known_pos is None:
        y_pred = cross_val_predict(model, X, y, cv=cv)
        score = f_score(y, y_pred)
    else:
        score = _masked_cv_score(X, y, model=model, cv=cv, f_score=f_score, mask=mask_known_pos)
    return float(score) * 100
