"""
This is a script for the backend of the ReliabilityModel class (prediction-reliability measures).

Pure, side-effect-free helpers on numpy arrays / fitted scikit-learn estimators. Each measure is
split into a **fit** step (learn a reference from the training data — called once) and an **apply**
step (score new samples — called per prediction): uncertainty, applicability-domain
(out-of-distribution) distances, calibrated-score sharpness, and marginal split-conformal sets.
"""
import numpy as np
from scipy.stats import norm
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


def proba_members(list_models, X, label_pos=1):
    """Stack per-member positive-class probabilities, shape (n_members, n_samples)."""
    return np.vstack([positive_proba(m, X, label_pos=label_pos) for m in list_models])


def fit_bootstrap_models(estimator, X_train, y_train, n_bootstrap=20, random_state=0):
    """Fit ``n_bootstrap`` resampled clones (each resample keeps both classes)."""
    rng = np.random.default_rng(random_state)
    n, models = len(X_train), []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_train[idx])) < 2:
            continue
        est = _set_seed(clone(estimator), int(rng.integers(0, 2**31 - 1)))
        models.append(est.fit(X_train[idx], y_train[idx]))
    return models


def comp_uncertainty(proba_arr, ci=90.0):
    """Std and a ``mean +/- z*std`` (Wald) ``ci``% interval of per-member probabilities, per sample.

    The interval is centred on the mean (the reported ``score``) and clipped to ``[0, 1]``, so the
    score is always inside it — unlike an empirical-percentile band, which need not contain the mean
    of a skewed member distribution.
    """
    P = np.asarray(proba_arr, dtype=float)
    mean = P.mean(axis=0)
    std = P.std(axis=0) if P.shape[0] > 1 else np.zeros(P.shape[1])
    z = float(norm.ppf(0.5 + ci / 200.0))
    return std, np.clip(mean - z * std, 0.0, 1.0), np.clip(mean + z * std, 0.0, 1.0)


# --- applicability domain (out-of-distribution) ---------------------------------------------
def fit_applicability_domain(X_train, k=5, percentile=95.0, ridge=1e-6):
    """Learn the applicability-domain reference (standardized kNN, Mahalanobis, leverage)."""
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    n_train, n_feat = Xtr.shape
    kq = max(1, min(k, n_train - 1))
    nn = NearestNeighbors(n_neighbors=kq).fit(Xtr)
    d_self, _ = NearestNeighbors(n_neighbors=min(kq + 1, n_train)).fit(Xtr).kneighbors(Xtr)
    train_knn = d_self[:, 1:].mean(axis=1) if d_self.shape[1] > 1 else d_self[:, 0]
    mu = Xtr.mean(axis=0)
    cov = np.atleast_2d(np.cov(Xtr, rowvar=False))
    Xc = Xtr - mu
    # Mahalanobis / leverage are only meaningful with more samples than features; otherwise the
    # (pseudo-inverse) covariance is rank-deficient and they collapse to a constant -> report NaN.
    return dict(scaler=scaler, nn=nn, thr=float(np.percentile(train_knn, percentile)), mu=mu,
                degenerate=(n_feat >= n_train),
                inv_cov=np.linalg.pinv(cov + ridge * np.eye(n_feat)),
                gram_inv=np.linalg.pinv(Xc.T @ Xc + ridge * np.eye(n_feat)))


def apply_applicability_domain(state, X_new):
    """Score new samples against a fitted applicability-domain reference."""
    Xnew = state["scaler"].transform(X_new)
    knn = state["nn"].kneighbors(Xnew)[0].mean(axis=1)
    thr = state["thr"]
    diff = Xnew - state["mu"]
    if state.get("degenerate"):
        maha = leverage = np.full(len(Xnew), np.nan)
    else:
        maha = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, state["inv_cov"], diff), 0, None))
        leverage = np.einsum("ij,jk,ik->i", diff, state["gram_inv"], diff)
    return dict(ood_score=(knn / thr if thr > 0 else knn),
                in_domain=(knn <= thr if thr > 0 else np.ones(len(Xnew), dtype=bool)),
                knn=knn, maha=maha, leverage=leverage)


# --- calibrated-score sharpness (aleatoric) -------------------------------------------------
def comp_sharpness(p):
    """Aleatoric sharpness of a probability: decisiveness ``margin`` and binary ``entropy``."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    margin = np.abs(p - 0.5) * 2.0
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return margin, entropy


# --- distribution-free validity (marginal split-conformal) ----------------------------------
def fit_conformal(estimator, X_train, y_train, alpha=0.1, label_pos=1, cal_size=0.3,
                  random_state=0):
    """Fit a marginal split-conformal reference: a clone on a proper-train split + the
    ``1 - alpha`` nonconformity quantile from the held-out calibration split.

    Returns a state dict, or ``None`` if the estimator cannot be split/refit.
    """
    try:
        Xtr, Xcal, ytr, ycal = train_test_split(
            X_train, y_train, test_size=cal_size, random_state=random_state, stratify=y_train)
        est = _set_seed(clone(estimator), random_state).fit(Xtr, ytr)
    except (ValueError, TypeError):
        return None
    p_cal = positive_proba(est, Xcal, label_pos=label_pos)
    nonconf = np.where(np.asarray(ycal) == label_pos, 1 - p_cal, p_cal)
    n = len(nonconf)
    # Canonical split-conformal level; numpy's "higher" convention is ~1 index conservative on a
    # small calibration split, so coverage is >= 1 - alpha (it over-covers, on the safe side).
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return dict(model=est, q=float(np.quantile(nonconf, level, method="higher")), label_pos=label_pos)


def apply_conformal(state, X_new):
    """Map new samples to conformal sets: 'pos' | 'neg' | 'both' (ambiguous) | 'none' (abstain)."""
    p = positive_proba(state["model"], X_new, label_pos=state["label_pos"])
    inc_pos, inc_neg = (1 - p) <= state["q"], p <= state["q"]
    return np.where(inc_pos & inc_neg, "both",
                    np.where(inc_pos, "pos", np.where(inc_neg, "neg", "none"))).astype(object)


def _set_seed(estimator, seed):
    """Set ``random_state`` on an estimator when it supports it (reproducibility)."""
    try:
        if "random_state" in estimator.get_params(deep=False):
            estimator.set_params(random_state=seed)
    except (ValueError, AttributeError):
        pass
    return estimator
