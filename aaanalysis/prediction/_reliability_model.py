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
    positive_proba, proba_members, fit_bootstrap_models, comp_uncertainty,
    fit_applicability_domain, apply_applicability_domain, comp_sharpness,
    fit_conformal, apply_conformal)


# I Helper Functions
def _resolve_members(model=None):
    """Return a list of member estimators if ``model`` is an ensemble, else ``None``.

    An unfitted :class:`AAPred` (``list_models_`` is ``None``) is rejected with a clear error.
    """
    if isinstance(model, (list, tuple)):
        return list(model)
    if hasattr(model, "list_models_"):
        if not model.list_models_:
            raise ValueError("The passed AAPred is not fitted; call its 'fit' before passing it.")
        return list(model.list_models_)
    return None


def _is_fitted(estimator):
    """A classifier is fitted once it exposes ``classes_``."""
    return hasattr(estimator, "classes_")


def _can_clone(estimator):
    """Whether an estimator can be cloned (scikit-learn API)."""
    try:
        clone(estimator)
        return True
    except (TypeError, RuntimeError):
        return False


def check_model(model=None, members=None):
    """A passed ensemble must be non-empty; a single model must support ``predict_proba``."""
    if isinstance(model, (list, tuple)) and len(model) == 0:
        raise ValueError("'model' is an empty list; provide at least one estimator.")
    if members is not None:
        for m in members:
            if not hasattr(m, "predict_proba"):
                raise ValueError("Every estimator in 'model' must implement 'predict_proba'.")
    elif model is not None and not hasattr(model, "predict_proba"):
        raise ValueError("'model' must implement 'predict_proba' (or pass a list / AAPred / None).")


# II Main Class
class ReliabilityModel(Wrapper):
    """
    Assess **how much to trust** each prediction — the reliability of a score, not the score itself.

    A high score is not the same as a trustworthy one: a model can be right to call a case a
    ``0.55`` toss-up, and badly wrong to call an input it has never seen a ``1.0``. Reporting only
    the score hides this. ``ReliabilityModel`` returns the score **together with** the answer to
    **three plain questions** — the categories that decide trust:

    1. **Has the model seen anything like this before?** If the input is unlike the training data,
       the model is guessing, and the score cannot be trusted no matter how high it is. *(the
       "applicability domain" — ``ood_score`` / ``in_domain``.)*
    2. **Do repeated models agree?** If an ensemble, or the same model refit on resampled data,
       disagree, the score is shaky. *(stability — ``score_std``, ``ci_low`` / ``ci_high``.)*
    3. **Is the case clear-cut, or a genuine toss-up?** Even a familiar, agreed-on case can be a
       real 50/50; a well-calibrated score near ``0.5`` means "honestly borderline," not "broken."
       *(decisiveness — ``margin`` / ``entropy``; the conformal set can also **abstain**.)*

    So the two classic failures separate cleanly:

    .. list-table::
       :header-rows: 1
       :widths: 32 22 22 38

       * - case
         - seen before?
         - clear-cut?
         - verdict
       * - confident about ``0.55``
         - yes
         - no (real toss-up)
         - trust it *as* "borderline"
       * - worthless about ``1.0``
         - no (out-of-distribution)
         - --
         - do **not** trust, at any score

    The headline flag ``reliable`` = **familiar and decisive** (``in_domain`` and a confident
    conformal singleton). It wraps an already-fitted binary predictor (an :class:`AAPred`, a
    :class:`~aaanalysis.TreeModel`, or any scikit-learn classifier) plus its training data and adds
    nothing but reliability — the prediction itself stays with the model.

    In uncertainty-quantification terms, questions 1-2 are **epistemic** uncertainty (the model's
    reducible lack of knowledge) and question 3 is **aleatoric** uncertainty (irreducible ambiguity
    in the data) [Huellermeier21]_. The applicability domain follows the QSAR idea [Sahigara12]_
    (features here are a descriptor space), calibration follows [Guo17]_ (a raw ``predict_proba`` is
    not a confidence until calibrated), and the conformal set follows [Angelopoulos23]_.

    .. versionadded:: 1.1.0

    Notes
    -----
    * **Scope.** Binary classification only. ``ad_mahalanobis`` / ``ad_leverage`` are auxiliary
      diagnostics that need more training samples than features (they are ``NaN`` otherwise); the
      ``in_domain`` decision uses the robust k-NN distance and is always valid.
    * **``score`` is the member mean** — the ensemble average, or the bootstrap ("bagged") average
      for a single model — so it is always the centre of ``[ci_low, ci_high]``. Set ``n_bootstrap=0``
      to report a single model's own probability instead (then ``score_std`` is 0).
    * **``reliable`` is conformal-based** (``in_domain`` and a confident conformal singleton),
      whereas ``margin`` / ``entropy`` are a separate calibrated-sharpness readout — they can
      disagree on a borderline case.
    * **Calibration** affects ``score_calibrated`` / ``margin`` / ``entropy`` only; ``score`` (and
      :meth:`ReliabilityModelPlot.reliability_diagram`) stay on the reported model score. For a
      passed ensemble, the calibrator and conformal reference are built from its first member.
    * **Reproducibility.** The bootstrap, calibration split, and conformal split are stochastic —
      set ``random_state`` for identical output across fits.
    * All fitted-state attributes carry a trailing underscore and are set by :meth:`fit`.

    See Also
    --------
    * :class:`AAPred` for fitting and deploying the prediction models this class assesses.
    * :class:`ReliabilityModelPlot` for visualizing the outputs.
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
        # Internal fitted state
        self._ad_state = None
        self._members = None
        self._calibrator = None
        self._conf_state = None
        self._ci = 90.0

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

        Learns, **once**, everything :meth:`predict` needs: the applicability-domain reference,
        the ensemble / bootstrap source of uncertainty, an optional probability calibrator, and
        the split-conformal calibration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix the model was fitted on (the applicability-domain reference).
        labels : array-like, shape (n_samples,)
            Binary training labels (exactly two classes).
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
            ensemble); ``score`` is then the bagged mean over the resamples (see Notes). ``0``
            disables the bootstrap and reports the model's own probability (``score_std`` = 0).
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

        Raises
        ------
        ValueError
            If ``labels`` are not binary, ``label_pos`` is absent from ``labels``, ``model`` is an
            empty list or lacks ``predict_proba``, a passed :class:`AAPred` is not fitted, or a
            numeric parameter is out of range.

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
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError(f"ReliabilityModel supports binary labels only; got {classes.size} "
                             f"classes {classes.tolist()}.")
        if label_pos not in set(classes.tolist()):
            raise ValueError(f"'label_pos' ({label_pos}) is not present in 'labels'.")
        members = _resolve_members(model)
        check_model(model=model, members=members)

        self.label_pos_ = label_pos
        self._ci = ci
        self._X_train, self._y_train = np.asarray(X), labels

        # Applicability-domain reference (fit once)
        self._ad_state = fit_applicability_domain(np.asarray(X), k=k, percentile=ad_percentile)

        # Resolve the member models. ``score`` is their mean and the interval is their spread, so
        # both come from the SAME set — the score is always the centre of its own interval.
        if members is not None:
            members = [m if _is_fitted(m) else clone(m).fit(X, labels) for m in members]
            base = clone(members[0]) if _can_clone(members[0]) else None
            self.model_ = members
            self._members = members                           # ensemble: mean = score, spread = uncertainty
        else:
            base = RandomForestClassifier(random_state=self._random_state) if model is None else model
            if not _is_fitted(base):
                base = clone(base) if _can_clone(base) else base
                base.fit(X, labels)
            self.model_ = base
            boot = (fit_bootstrap_models(clone(base), np.asarray(X), labels, n_bootstrap=n_bootstrap,
                                         random_state=self._random_state)
                    if (n_bootstrap and _can_clone(base)) else [])
            self._members = boot or [base]                    # score = bagged mean; spread = bootstrap
        cloneable_base = base if (base is not None and _can_clone(base)) else None

        # Calibration (fit a calibrated clone on the training data)
        self._calibrator = None
        if calibrate and cloneable_base is not None:
            n_min = int(np.min(np.bincount((labels == label_pos).astype(int))))
            cv = max(2, min(3, n_min))
            try:
                self._calibrator = CalibratedClassifierCV(
                    clone(cloneable_base), method=calibration_method, cv=cv).fit(X, labels)
            except (ValueError, TypeError):
                self._calibrator = None

        # Split-conformal reference (fit once)
        self._conf_state = (fit_conformal(
            clone(cloneable_base), np.asarray(X), labels, alpha=conformal_alpha,
            label_pos=label_pos, random_state=self._random_state)
            if cloneable_base is not None else None)
        if self._verbose:
            ut.print_out(f"ReliabilityModel fitted (in-domain kNN threshold="
                         f"{self._ad_state['thr']:.3f}; uncertainty from {len(self._members)} "
                         f"member(s); calibrated={self._calibrator is not None}; "
                         f"conformal={self._conf_state is not None}).")
        return self

    def predict(self,
                X: ut.ArrayLike2D,
                ) -> pd.DataFrame:
        """
        Score new samples for reliability (one row per sample).

        Applies the references learned by :meth:`fit` (applicability domain, ensemble / bootstrap,
        calibrator, conformal calibration) — no model is refitted here, so repeated calls are cheap
        and deterministic. Each column maps to one axis of the mental model: ``score_std`` /
        ``ci_*`` (stability), ``ood_score`` / ``in_domain`` / ``ad_*`` (applicability domain),
        ``margin`` / ``entropy`` (calibrated ambiguity), ``conformal_set`` (validity), and
        ``reliable`` (the headline flag).

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

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than the training data.

        Examples
        --------
        .. include:: examples/rm_predict.rst
        """
        if self._ad_state is None:
            raise RuntimeError("Call 'fit' before 'predict'.")
        X = ut.check_X(X=X, min_n_samples=1)
        n_feat_train = self._ad_state["mu"].shape[0]
        if X.shape[1] != n_feat_train:
            raise ValueError(f"'X' has {X.shape[1]} features but the model was fit on "
                             f"{n_feat_train}.")
        lp = self.label_pos_

        proba = proba_members(self._members, X, label_pos=lp)
        score, (std, lo, hi) = proba.mean(axis=0), comp_uncertainty(proba, ci=self._ci)
        ad = apply_applicability_domain(self._ad_state, X)

        if self._calibrator is not None:
            score_cal = positive_proba(self._calibrator, X, label_pos=lp)
        else:
            score_cal = np.full(len(X), np.nan)
        margin, entropy = comp_sharpness(np.where(np.isnan(score_cal), score, score_cal))

        if self._conf_state is not None:
            conf = apply_conformal(self._conf_state, X)
            singleton = np.isin(conf, [ut.STR_CONF_POS, ut.STR_CONF_NEG])
        else:
            conf = np.array([ut.STR_CONF_NONE] * len(X), dtype=object)
            singleton = margin >= 0.5                          # fallback when conformal unavailable
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

        Aggregates :meth:`predict` over a labeled evaluation set into a compact table — per-bin
        predicted-vs-empirical positive rate (how well calibrated the score is) plus a summary row
        with the fraction in the applicability domain and the empirical coverage of the conformal
        sets (which should track ``1 - conformal_alpha``).

        Parameters
        ----------
        X : array-like, optional
            Evaluation features; the training ``X`` is used if ``None`` (a held-out labeled set
            gives an honest estimate — calibration and conformal were fit on the training data).
        labels : array-like, optional
            Evaluation labels; the training labels are used if ``None``.
        n_bins : int, default=5
            Number of equal-width score bins for the calibration curve.

        Returns
        -------
        df_eval : pd.DataFrame
            Per-bin rows (``bin``, ``mean_score``, ``empirical_pos``, ``n``) plus a summary row
            with the in-domain fraction (``mean_score``) and the empirical conformal coverage
            (``empirical_pos``).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        Examples
        --------
        .. include:: examples/rm_eval.rst
        """
        if self._ad_state is None:
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
        sets = df[ut.COL_CONFORMAL_SET].to_numpy()
        covered = (np.isin(sets, [ut.STR_CONF_POS, ut.STR_CONF_BOTH]) & (y == 1)) | \
                  (np.isin(sets, [ut.STR_CONF_NEG, ut.STR_CONF_BOTH]) & (y == 0))
        rows.append({"bin": "summary", "mean_score": float(np.mean(df[ut.COL_IN_DOMAIN])),
                     "empirical_pos": float(np.mean(covered)), "n": len(X)})
        return pd.DataFrame(rows)
