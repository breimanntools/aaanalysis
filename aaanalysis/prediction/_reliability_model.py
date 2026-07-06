"""
This is a script for the frontend of the ReliabilityModel class for prediction-reliability measures.
"""
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper

from ._backend.reliability.reliability import (
    positive_proba, proba_members_ensemble, proba_members_bootstrap, comp_uncertainty,
    comp_applicability_domain, comp_sharpness, split_conformal)


# I Helper Functions
def _resolve_members(model=None):
    """Return a list of fitted member estimators if ``model`` is an ensemble, else ``None``."""
    if isinstance(model, (list, tuple)):
        return list(model)
    list_models = getattr(model, "list_models_", None)   # e.g. a fitted AAPred
    if list_models:
        return list(list_models)
    return None


def _is_fitted(estimator):
    """A classifier is fitted once it exposes ``classes_``."""
    return hasattr(estimator, "classes_")


def check_match_model_X_labels(model=None, members=None):
    """A passed ensemble must be non-empty."""
    if isinstance(model, (list, tuple)) and len(model) == 0:
        raise ValueError("'model' is an empty list; provide at least one estimator.")


# II Main Class
class ReliabilityModel(Wrapper):
    """
    Prediction-reliability model: quantify **how much to trust** a per-sample prediction.

    A prediction score is not the same as its trustworthiness: a model can be confident about a
    ``0.55`` score (a genuinely ambiguous, in-distribution case) yet worthless about a ``1.0``
    score on an input unlike anything it was trained on. ``ReliabilityModel`` reports the score
    **together with** the two orthogonal signals that decide trust — following the aleatoric vs.
    epistemic distinction ([Huellermeier21]_):

    - **Stability / uncertainty** (epistemic): spread of the score across an ensemble or
      bootstrap resamples, as ``score_std`` and a confidence interval.
    - **Applicability domain** (epistemic / out-of-distribution): distance of the input to the
      training set (k-NN distance, Mahalanobis, leverage), as ``ood_score`` and an ``in_domain``
      flag — the QSAR applicability-domain idea ([Sahigara12]_), since features here are a
      descriptor space.
    - **Calibrated sharpness** (aleatoric): ``margin`` / ``entropy`` of a calibrated probability
      — genuine class ambiguity, meaningful only once the score is calibrated.
    - **Distribution-free validity**: a marginal split-conformal prediction set that can
      **abstain** ([Angelopoulos23]_).

    It wraps an already-fitted predictor (an :class:`AAPred`, a :class:`~aaanalysis.TreeModel`, or
    any scikit-learn classifier) plus its training data, and adds nothing but reliability — the
    prediction itself stays with the model. A prediction is trustworthy when it is
    **in-domain, stable, and a confident conformal singleton** (``reliable``).

    Notes
    -----
    * References:

      .. [Huellermeier21] Huellermeier & Waegeman (2021),
         *Aleatoric and epistemic uncertainty in machine learning*, Machine Learning 110:457-506.
      .. [Sahigara12] Sahigara et al. (2012),
         *Comparison of Different Approaches to Define the Applicability Domain of QSAR Models*,
         Molecules 17:4791-4810.
      .. [Angelopoulos23] Angelopoulos & Bates (2023),
         *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
         Quantification*, Foundations and Trends in Machine Learning.

    See Also
    --------
    * :class:`AAPred` for fitting and deploying the prediction models this class assesses.
    """

    def __init__(self,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            Seed for the bootstrap, calibration split, and conformal split, for reproducibility.
        """
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        self._verbose = verbose
        self._random_state = random_state
        # Fitted attributes
        self.model_: Optional[object] = None
        self.label_pos_: Optional[int] = None
        self.label_neg_: Optional[int] = None
        # Internal fitted state
        self._members = None
        self._base = None
        self._calibrator = None
        self._X_train = None
        self._y_train = None
        self._k = 5
        self._ad_percentile = 95.0
        self._ci = 90.0
        self._n_bootstrap = 20
        self._alpha = 0.1

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D,
            model: Optional[Union[object, List]] = None,
            label_pos: int = 1,
            k: int = 5,
            ad_percentile: float = 95.0,
            ci: float = 90.0,
            n_bootstrap: int = 20,
            calibrate: bool = True,
            calibration_method: str = "isotonic",
            conformal_alpha: float = 0.1,
            ) -> "ReliabilityModel":
        """
        Fit the reliability reference from a (fitted or default) model and its training data.

        Records the training feature distribution (applicability domain), the ensemble/bootstrap
        source of uncertainty, an optional probability calibrator, and the split-conformal
        calibration — everything :meth:`predict` needs to score new samples for trust.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix the model was fitted on (the applicability-domain reference).
        labels : array-like, shape (n_samples,)
            Binary training labels.
        model : estimator, list of estimators, AAPred, or None
            A fitted scikit-learn classifier (``predict_proba``), a **list** of fitted estimators
            (ensemble; uncertainty = their disagreement), a fitted :class:`AAPred`, or ``None`` to
            fit a default :class:`~sklearn.ensemble.RandomForestClassifier`.
        label_pos : int, default=1
            Positive-class label whose probability is scored.
        k : int, default=5
            Number of nearest training neighbors for the applicability-domain distance.
        ad_percentile : float, default=95.0
            Training kNN-distance percentile used as the ``in_domain`` boundary.
        ci : float, default=90.0
            Central width (percent) of the reported score confidence interval.
        n_bootstrap : int, default=20
            Bootstrap resamples for uncertainty when ``model`` is a single estimator (not an
            ensemble). ``0`` disables the bootstrap (``score_std`` = 0).
        calibrate : bool, default=True
            Fit a probability calibrator (needed for meaningful ``margin`` / ``entropy``).
        calibration_method : str, default="isotonic"
            ``"isotonic"`` or ``"sigmoid"`` (Platt), passed to
            :class:`~sklearn.calibration.CalibratedClassifierCV`.
        conformal_alpha : float, default=0.1
            Miscoverage level of the split-conformal set (``1 - alpha`` coverage).

        Returns
        -------
        ReliabilityModel
            The fitted instance.

        Examples
        --------
        .. include:: examples/rm_fit.rst
        """
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="label_pos", val=label_pos, min_val=0, just_int=True)
        ut.check_number_range(name="k", val=k, min_val=1, just_int=True)
        ut.check_number_range(name="ad_percentile", val=ad_percentile, min_val=1, max_val=100,
                              just_int=False)
        ut.check_number_range(name="ci", val=ci, min_val=1, max_val=99, just_int=False)
        ut.check_number_range(name="n_bootstrap", val=n_bootstrap, min_val=0, just_int=True)
        ut.check_bool(name="calibrate", val=calibrate)
        ut.check_str(name="calibration_method", val=calibration_method)
        if calibration_method not in ("isotonic", "sigmoid"):
            raise ValueError("'calibration_method' must be 'isotonic' or 'sigmoid'.")
        ut.check_number_range(name="conformal_alpha", val=conformal_alpha, min_val=0, max_val=1,
                              just_int=False, accept_none=False)
        labels = np.asarray(labels)
        if label_pos not in set(labels.tolist()):
            raise ValueError(f"'label_pos' ({label_pos}) is not present in 'labels'.")
        members = _resolve_members(model)
        check_match_model_X_labels(model=model, members=members)

        self._X_train, self._y_train = np.asarray(X), labels
        self.label_pos_ = label_pos
        self.label_neg_ = int(next(v for v in np.unique(labels) if v != label_pos))
        self._k, self._ad_percentile, self._ci = k, ad_percentile, ci
        self._n_bootstrap, self._alpha = n_bootstrap, conformal_alpha

        # Resolve the model / ensemble and a cloneable base estimator
        if members is not None:
            self._members = [m if _is_fitted(m) else clone(m).fit(X, labels) for m in members]
            base = clone(self._members[0]) if _can_clone(self._members[0]) else None
            self.model_ = self._members
        else:
            if model is None:
                base = RandomForestClassifier(random_state=self._random_state)
            else:
                base = model
            if not _is_fitted(base):
                base = clone(base) if _can_clone(base) else base
                base.fit(X, labels)
            self._members = None
            self.model_ = base
        self._base = base if (base is not None and _can_clone(base)) else None

        # Calibration (fit a calibrated clone on the training data)
        self._calibrator = None
        if calibrate and self._base is not None:
            n_min = int(np.min(np.bincount(labels.astype(int)))) if labels.dtype.kind in "iu" else 2
            cv = max(2, min(3, n_min))
            try:
                self._calibrator = CalibratedClassifierCV(
                    clone(self._base), method=calibration_method, cv=cv).fit(X, labels)
            except (ValueError, TypeError):
                self._calibrator = None
        return self

    def predict(self,
                X: ut.ArrayLike2D,
                ) -> pd.DataFrame:
        """
        Score new samples for reliability (one row per sample).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to assess (same feature space as the training ``X``).

        Returns
        -------
        df_rel : pd.DataFrame
            One row per sample with columns: ``score``, ``score_std``, ``ci_low``, ``ci_high``,
            ``ood_score``, ``in_domain``, ``ad_knn_dist``, ``ad_mahalanobis``, ``ad_leverage``,
            ``score_calibrated``, ``margin``, ``entropy``, ``conformal_set``, ``reliable``.

        Examples
        --------
        .. include:: examples/rm_predict.rst
        """
        if self._X_train is None:
            raise RuntimeError("Call 'fit' before 'predict'.")
        X = ut.check_X(X=X, min_n_samples=1)
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(f"'X' has {X.shape[1]} features but the model was fit on "
                             f"{self._X_train.shape[1]}.")
        lp = self.label_pos_

        # Uncertainty: ensemble disagreement or bootstrap
        if self._members is not None:
            proba_members = proba_members_ensemble(self._members, X, label_pos=lp)
        elif self._base is not None and self._n_bootstrap > 0:
            proba_members = proba_members_bootstrap(
                clone(self._base), self._X_train, self._y_train, X, n_bootstrap=self._n_bootstrap,
                label_pos=lp, random_state=self._random_state)
        else:
            proba_members = positive_proba(self.model_, X, label_pos=lp)[None, :]
        score, std, lo, hi = comp_uncertainty(proba_members, ci=self._ci)

        # Applicability domain
        ad = comp_applicability_domain(self._X_train, X, k=self._k, percentile=self._ad_percentile)

        # Calibrated sharpness
        if self._calibrator is not None:
            score_cal = positive_proba(self._calibrator, X, label_pos=lp)
        else:
            score_cal = np.full(len(X), np.nan)
        margin, entropy = comp_sharpness(np.where(np.isnan(score_cal), score, score_cal))

        # Distribution-free validity (marginal split-conformal)
        conf = None
        if self._base is not None:
            conf = split_conformal(clone(self._base), self._X_train, self._y_train, X,
                                   alpha=self._alpha, label_pos=lp, random_state=self._random_state)
        if conf is None:
            conf = np.array([ut.STR_CONF_NONE] * len(X), dtype=object)
            singleton = margin >= 0.5                          # fallback when conformal unavailable
        else:
            singleton = np.isin(conf, [ut.STR_CONF_POS, ut.STR_CONF_NEG])
        reliable = ad["in_domain"] & singleton

        return pd.DataFrame({
            ut.COL_SCORE: score, ut.COL_SCORE_STD: std, ut.COL_CI_LOW: lo, ut.COL_CI_HIGH: hi,
            ut.COL_OOD_SCORE: ad["ood_score"], ut.COL_IN_DOMAIN: ad["in_domain"],
            ut.COL_AD_KNN: ad["knn"], ut.COL_AD_MAHALANOBIS: ad["maha"],
            ut.COL_AD_LEVERAGE: ad["leverage"], ut.COL_SCORE_CAL: score_cal,
            ut.COL_MARGIN: margin, ut.COL_ENTROPY: entropy, ut.COL_CONFORMAL_SET: conf,
            ut.COL_RELIABLE: reliable})

    def eval(self,
             X: Optional[ut.ArrayLike2D] = None,
             labels: Optional[ut.ArrayLike1D] = None,
             n_bins: int = 5,
             ) -> pd.DataFrame:
        """
        Reliability diagnostics: calibration curve, empirical conformal coverage, in-domain rate.

        Parameters
        ----------
        X : array-like, optional
            Evaluation features; the training ``X`` is used if ``None``.
        labels : array-like, optional
            Evaluation labels; the training labels are used if ``None``.
        n_bins : int, default=5
            Number of equal-width score bins for the calibration curve.

        Returns
        -------
        df_eval : pd.DataFrame
            Per-bin rows (``bin``, ``mean_score``, ``empirical_pos``, ``n``) plus a summary row
            with the empirical conformal coverage and the in-domain fraction.

        Examples
        --------
        .. include:: examples/rm_eval.rst
        """
        if self._X_train is None:
            raise RuntimeError("Call 'fit' before 'eval'.")
        if X is None:
            X, labels = self._X_train, self._y_train
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="n_bins", val=n_bins, min_val=2, just_int=True)
        y = (np.asarray(labels) == self.label_pos_).astype(int)
        df = self.predict(X)
        s = df[ut.COL_SCORE].to_numpy()
        edges = np.linspace(0, 1, n_bins + 1)
        rows = []
        for b in range(n_bins):
            m = (s >= edges[b]) & (s <= edges[b + 1] if b == n_bins - 1 else s < edges[b + 1])
            rows.append({"bin": f"{edges[b]:.2f}-{edges[b+1]:.2f}",
                         "mean_score": float(np.mean(s[m])) if m.any() else np.nan,
                         "empirical_pos": float(np.mean(y[m])) if m.any() else np.nan,
                         "n": int(m.sum())})
        covered = np.isin(df[ut.COL_CONFORMAL_SET].to_numpy(),
                          [ut.STR_CONF_POS, ut.STR_CONF_BOTH]) & (y == 1)
        covered |= np.isin(df[ut.COL_CONFORMAL_SET].to_numpy(),
                           [ut.STR_CONF_NEG, ut.STR_CONF_BOTH]) & (y == 0)
        rows.append({"bin": "summary",
                     "mean_score": float(np.mean(df[ut.COL_IN_DOMAIN])),   # in-domain fraction
                     "empirical_pos": float(np.mean(covered)),             # conformal coverage
                     "n": len(X)})
        return pd.DataFrame(rows)


def _can_clone(estimator):
    """Whether an estimator can be cloned (scikit-learn API)."""
    try:
        clone(estimator)
        return True
    except (TypeError, RuntimeError):
        return False
