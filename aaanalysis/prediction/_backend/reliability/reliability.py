"""
This is a script for the backend of the ReliabilityModel class (prediction-reliability measures).

Pure, side-effect-free numeric helpers on numpy arrays / fitted scikit-learn estimators:
uncertainty aggregation, applicability-domain (out-of-distribution) distances, calibrated-score
sharpness, and marginal split-conformal prediction sets. The frontend orchestrates them.
"""
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


# --- uncertainty (epistemic stability) ------------------------------------------------------
def positive_proba(model, X, label_pos=1):
    """Positive-class probability column of a fitted classifier."""
    proba = np.asarray(model.predict_proba(X))
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(label_pos) if label_pos in classes else (proba.shape[1] - 1)
    return proba[:, idx]


def proba_members_ensemble(list_models, X, label_pos=1):
    """Stack per-member positive-class probabilities, shape (n_members, n_samples)."""
    return np.vstack([positive_proba(m, X, label_pos=label_pos) for m in list_models])


def proba_members_bootstrap(estimator, X_train, y_train, X, n_bootstrap=20, label_pos=1,
                            random_state=0):
    """Fit ``n_bootstrap`` resampled clones and stack their positive-class probabilities for ``X``."""
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    rows = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_train[idx])) < 2:           # keep both classes in the resample
            continue
        est = clone(estimator)
        _set_seed(est, int(rng.integers(0, 2**31 - 1)))
        est.fit(X_train[idx], y_train[idx])
        rows.append(positive_proba(est, X, label_pos=label_pos))
    if not rows:
        rows = [np.full(len(X), np.nan)]
    return np.vstack(rows)


def comp_uncertainty(proba_members, ci=90.0):
    """Mean, std, and central-``ci``% interval of per-member probabilities, per sample."""
    P = np.asarray(proba_members, dtype=float)
    mean = P.mean(axis=0)
    std = P.std(axis=0) if P.shape[0] > 1 else np.zeros(P.shape[1])
    lo = np.percentile(P, (100 - ci) / 2, axis=0)
    hi = np.percentile(P, 100 - (100 - ci) / 2, axis=0)
    return mean, std, lo, hi


# --- applicability domain (out-of-distribution) ---------------------------------------------
def comp_applicability_domain(X_train, X_new, k=5, percentile=95.0, ridge=1e-6):
    """Distance-based applicability domain (QSAR-style): kNN distance, Mahalanobis, leverage.

    Distances are computed in the standardized training space. ``in_domain`` uses the
    ``percentile`` of the training kNN-distance distribution as the domain boundary, and
    ``ood_score`` is the new point's kNN distance relative to that boundary (1.0 = on it).
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xnew = scaler.transform(X_train), scaler.transform(X_new)
    n_train, n_feat = Xtr.shape

    kq = max(1, min(k, n_train - 1))
    nn = NearestNeighbors(n_neighbors=kq).fit(Xtr)
    d_self, _ = NearestNeighbors(n_neighbors=min(kq + 1, n_train)).fit(Xtr).kneighbors(Xtr)
    train_knn = d_self[:, 1:].mean(axis=1) if d_self.shape[1] > 1 else d_self[:, 0]
    thr = float(np.percentile(train_knn, percentile))
    d_new, _ = nn.kneighbors(Xnew)
    knn = d_new.mean(axis=1)
    ood_score = knn / thr if thr > 0 else knn
    in_domain = knn <= thr if thr > 0 else np.ones(len(Xnew), dtype=bool)

    mu = Xtr.mean(axis=0)
    cov = np.atleast_2d(np.cov(Xtr, rowvar=False))
    inv_cov = np.linalg.pinv(cov + ridge * np.eye(n_feat))
    diff = Xnew - mu
    maha = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0, None))

    Xc = Xtr - mu
    gram_inv = np.linalg.pinv(Xc.T @ Xc + ridge * np.eye(n_feat))
    leverage = np.einsum("ij,jk,ik->i", diff, gram_inv, diff)

    return dict(ood_score=ood_score, in_domain=in_domain, knn=knn, maha=maha, leverage=leverage)


# --- calibrated-score sharpness (aleatoric) -------------------------------------------------
def comp_sharpness(p):
    """Aleatoric sharpness of a probability: decisiveness ``margin`` and binary ``entropy``."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    margin = np.abs(p - 0.5) * 2.0
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return margin, entropy


# --- distribution-free validity (marginal split-conformal) ----------------------------------
def split_conformal(estimator, X_train, y_train, X_new, alpha=0.1, label_pos=1,
                    cal_size=0.3, random_state=0):
    """Marginal split-conformal prediction sets for a binary classifier.

    Refits a clone on a proper-train split, calibrates nonconformity ``1 - p(true)`` on the
    held-out split, and returns, per new sample, which classes fall in the ``1 - alpha`` set:
    ``'pos'``/``'neg'`` (confident singleton), ``'both'`` (ambiguous), ``'none'`` (abstain).
    Returns ``None`` if the estimator cannot be cloned/refit.
    """
    try:
        Xtr, Xcal, ytr, ycal = train_test_split(
            X_train, y_train, test_size=cal_size, random_state=random_state, stratify=y_train)
        est = clone(estimator)
        _set_seed(est, random_state)
        est.fit(Xtr, ytr)
    except (ValueError, TypeError):
        return None
    p_cal = positive_proba(est, Xcal, label_pos=label_pos)
    label_neg = next(c for c in getattr(est, "classes_", [0, 1]) if c != label_pos)
    nonconf_cal = np.where(np.asarray(ycal) == label_pos, 1 - p_cal, p_cal)
    n = len(nonconf_cal)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(nonconf_cal, level, method="higher"))
    p_new = positive_proba(est, X_new, label_pos=label_pos)
    inc_pos, inc_neg = (1 - p_new) <= q, p_new <= q
    out = np.where(inc_pos & inc_neg, "both",
                   np.where(inc_pos, "pos", np.where(inc_neg, "neg", "none")))
    return out.astype(object)


def _set_seed(estimator, seed):
    """Set ``random_state`` on an estimator when it supports it (reproducibility)."""
    try:
        if "random_state" in estimator.get_params(deep=False):
            estimator.set_params(random_state=seed)
    except (ValueError, AttributeError):
        pass
    return estimator
