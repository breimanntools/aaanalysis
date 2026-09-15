"""
This is a script for the frontend of the ReliabilityModel class for prediction-reliability measures.
"""
from typing import Optional, List, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper

from ._backend.reliability.reliability import (
    positive_proba, proba_members, fit_bootstrap_models, comp_uncertainty,
    fit_applicability_domain, apply_applicability_domain, comp_ad_status, comp_sharpness,
    fit_conformal, apply_conformal, comp_calibration_bins, comp_brier, comp_ece)


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


def check_finite(name: str, val: float):
    """Check that a numeric value is finite.

    ``NaN`` and ``+/-inf`` are floats that pass *every* range comparison silently (``nan < 0``
    and ``nan > 1`` are both ``False``), so they slip through a plain range check and only
    surface much later as ``NaN`` outputs. Non-numeric values are left to the range check,
    which reports the type.
    """
    if isinstance(val, (float, np.floating)) and not np.isfinite(val):
        raise ValueError(f"'{name}' ({val}) should be a finite float or an integer.")


def check_ad_borderline(ad_borderline: float):
    """Check that ``ad_borderline`` is a finite, non-negative number (not a bool)."""
    if isinstance(ad_borderline, bool):
        raise ValueError(f"'ad_borderline' ({ad_borderline}) should be a finite number >= 0 "
                         f"(a float or an integer), but got bool.")
    ut.check_number_range(name="ad_borderline", val=ad_borderline, min_val=0, just_int=False)
    if not np.isfinite(ad_borderline):
        raise ValueError(f"'ad_borderline' ({ad_borderline}) should be a finite number >= 0.")


def check_ci(ci: float):
    """Check that ``ci`` is a fraction in (0, 1), with a hint when a percent is passed."""
    check_finite(name="ci", val=ci)
    ut.check_number_range(name="ci", val=ci, min_val=0, just_int=False)
    if 1 < ci <= 100:
        raise ValueError(f"'ci' ({ci}) should be a fraction in (0, 1), e.g. 0.90 for a 90% "
                         f"interval, not a percent.")
    ut.check_number_range(name="ci", val=ci, min_val=0.0, max_val=1.0, just_int=False,
                          exclusive_limits=True)


def _reason_no_calibrator(calibrate_requested=False, calibration_error=None):
    """Plain-language reason why a fitted model carries no probability calibrator."""
    if not calibrate_requested:
        return "it was fitted with 'calibrate=False'"
    if calibration_error is not None:
        return (f"it was fitted with 'calibrate=True', but the calibrator could not be fitted "
                f"({calibration_error})")
    return "no calibrator was fitted"


# II Main Functions
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
    conformal singleton, or ``margin >= 0.5`` when no conformal reference is available). It wraps
    an already-fitted binary predictor (an :class:`AAPred`, a :class:`~aaanalysis.TreeModel`, or
    any scikit-learn classifier) plus its training data and adds nothing but reliability — the
    prediction itself stays with the model.

    In uncertainty-quantification terms, questions 1-2 are **epistemic** uncertainty (the model's
    reducible lack of knowledge) and question 3 is **aleatoric** uncertainty (irreducible ambiguity
    in the data) [Huellermeier21]_. The applicability domain follows the QSAR idea [Sahigara12]_
    (features here are a descriptor space), calibration follows [Guo17]_ (a raw ``predict_proba`` is
    not a confidence until calibrated), and the conformal set follows [Angelopoulos23]_.

    .. warning::

        **Experimental.** This class is part of the new v1.1.0 prediction layer and is under active
        development; its API (signatures, defaults, return objects) may change between minor releases
        without the usual deprecation cycle. Pin a version if you depend on the current behaviour.

    .. versionadded:: 1.1.0

    Notes
    -----
    * **Scope.** Binary classification only. ``ad_mahalanobis`` / ``ad_leverage`` are auxiliary
      diagnostics that need more training samples than features (they are ``NaN`` otherwise); the
      ``in_domain`` decision uses the robust k-NN distance and is always valid.
    * **``score`` is the member mean** — the ensemble average, or the bootstrap ("bagged") average
      for a single model — so it is always the centre of ``[ci_low, ci_high]``. Set ``n_bootstrap=0``
      to report a single model's own probability instead (then ``score_std`` is 0).
    * **``score_std`` is a per-sample spread across members.** Here it is the standard deviation of
      one sample's probability across the ensemble members (or bootstrap resamples). The
      same-named column in :meth:`AAPred.eval` and :meth:`ModelEvaluator.run` means something
      different: the standard deviation of a metric across cross-validation folds.
    * **``reliable`` is conformal-based** (``in_domain`` and a confident conformal singleton)
      when a conformal reference is available; otherwise it uses ``in_domain`` and
      ``margin >= 0.5``. ``margin`` / ``entropy`` can therefore disagree with it on a borderline
      case.
    * **Calibration** affects ``score_calibrated`` / ``margin`` / ``entropy`` only; when no
      calibrator is available, the latter two use ``score``. ``score`` stays the reported model
      score. Raw scoring is the **default** everywhere: :meth:`eval` bins
      ``score`` and :meth:`ReliabilityModelPlot.reliability_diagram` draws that curve, whereas
      ``eval(use_calibrated=True)`` returns the calibrated table, from which the same method draws
      the calibrated curve (both can share one ``ax``). If ``calibrate=True`` but no calibrator can
      be fitted (e.g. a class with a single member, or a model that cannot be cloned), :meth:`fit`
      warns, ``score_calibrated`` is ``NaN``, and ``eval(use_calibrated=True)`` raises naming that
      reason. For a passed ensemble, the calibrator and conformal reference are built from its
      first member.
    * **Reproducibility.** The bootstrap, calibration split, and conformal split are stochastic —
      set ``random_state`` for identical output across fits.
    * **Banded domain verdict.** ``ad_status`` refines the bool ``in_domain`` into ``inside``
      (``ood_score <= 1``), ``borderline`` (``1 < ood_score <= 1 + ad_borderline``), ``outside``
      (above the band), and ``unknown`` (the training reference has no usable spread, so
      ``ood_score`` is ``NaN``). ``in_domain`` always equals ``ad_status == "inside"``.
    * All fitted-state attributes carry a trailing underscore and are set by :meth:`fit`.

    Attributes
    ----------
    model_ : estimator or list of estimators
        The assessed model (the fitted single estimator or the list of ensemble members).
    label_pos_ : int
        Positive-class label whose probability is scored.
    ad_threshold_ : float
        Applicability-domain boundary: the ``ad_percentile``-th percentile of the training
        samples' mean distance to their ``k`` nearest other training samples (standardized
        feature space). ``ood_score`` is ``ad_knn / ad_threshold_``, so ``ood_score == 1`` is the
        boundary. A non-positive value marks a degenerate reference (``ad_status='unknown'``).

        .. versionadded:: 1.2.0
    ad_method_ : str
        Decision rule behind ``ood_score`` / ``in_domain`` / ``ad_status``; always ``"knn"``.

        .. versionadded:: 1.2.0

    See Also
    --------
    * :class:`AAPred` for fitting and deploying the prediction models this class assesses.
    * :class:`ReliabilityModelPlot` for visualizing the outputs.
    """

    def __init__(self,
                 *, verbose: bool = True,
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
        self.ad_threshold_: Optional[float] = None
        self.ad_method_: Optional[str] = None
        # Internal fitted state
        self._ad_state = None
        self._members = None
        self._calibrator = None
        self._calibrate_requested = False
        self._calibration_error: Optional[str] = None
        self._conf_state = None
        self._ci = 0.90
        self._ad_borderline = 0.1

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D,
            *, model: Optional[Union[object, List]] = None,
            label_pos: int = 1,
            k: int = 5,
            ad_percentile: float = 95.0,
            ad_borderline: float = 0.1,
            ci: float = 0.90,
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
            Training kNN-distance percentile used as the ``in_domain`` boundary (stored as
            ``ad_threshold_``).
        ad_borderline : float, default=0.1
            Width of the ``borderline`` band just outside the domain boundary, relative to it:
            a sample with ``1 < ood_score <= 1 + ad_borderline`` gets ``ad_status='borderline'``,
            above that ``'outside'``. ``0`` disables the band. Must be a **finite non-negative**
            number (a negative value, ``inf``, ``NaN``, or a bool raises).

            .. versionadded:: 1.2.0
        ci : float, default=0.90
            Central width of the reported score confidence interval, as a fraction in ``(0, 1)``
            (e.g. ``0.90`` for a 90% interval), matching :meth:`ModelEvaluator.run`.

            .. versionchanged:: 1.2.0
               Now a fraction in ``(0, 1)`` (default ``0.90``) instead of a percent (``90.0``),
               matching :meth:`ModelEvaluator.run` and ``comp_bootstrap_ci``. A percent value
               raises a ``ValueError`` with a hint; the interval itself is unchanged.
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
            numeric parameter is out of range or not finite (``NaN`` / ``inf``), including a
            negative ``ad_borderline``.

        Warnings
        --------
        UserWarning
            If ``calibrate=True`` but no calibrator can be fitted, because a class holds fewer
            members than the internal cross-validation needs or the model cannot be cloned.
            ``score_calibrated`` is then ``NaN`` and :meth:`eval` with ``use_calibrated=True``
            raises, naming that reason.

            .. versionadded:: 1.2.0

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
        check_finite(name="ad_percentile", val=ad_percentile)
        ut.check_number_range(name="ad_percentile", val=ad_percentile, min_val=1, max_val=100,
                              just_int=False)
        check_ad_borderline(ad_borderline=ad_borderline)
        check_ci(ci=ci)
        ut.check_number_range(name="n_bootstrap", val=n_bootstrap, min_val=0, just_int=True)
        ut.check_bool(name="calibrate", val=calibrate)
        ut.check_str(name="calibration_method", val=calibration_method)
        if calibration_method not in ("isotonic", "sigmoid"):
            raise ValueError("'calibration_method' must be 'isotonic' or 'sigmoid'.")
        check_finite(name="conformal_alpha", val=conformal_alpha)
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
        self._ad_borderline = float(ad_borderline)
        self.ad_threshold_ = float(self._ad_state["thr"])
        self.ad_method_ = ut.STR_AD_METHOD_KNN

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

        # Calibration (fit a calibrated clone on the training data). A failure is recorded and
        # reported instead of swallowed, so 'eval(use_calibrated=True)' can name the real reason
        # rather than blame a 'calibrate=False' that was never passed.
        self._calibrator = None
        self._calibrate_requested = calibrate
        self._calibration_error = None
        if calibrate:
            if cloneable_base is None:
                self._calibration_error = ("the model cannot be cloned, and the calibrator is "
                                           "fitted on a clone")
            else:
                n_min = int(np.min(np.bincount((labels == label_pos).astype(int))))
                cv = max(2, min(3, n_min))
                try:
                    self._calibrator = CalibratedClassifierCV(
                        clone(cloneable_base), method=calibration_method, cv=cv).fit(X, labels)
                except (ValueError, TypeError) as e:
                    self._calibration_error = (f"{cv}-fold cross-validated calibration failed "
                                               f"({type(e).__name__}: {e})")
            if self._calibration_error is not None:
                warnings.warn(f"'calibrate' (True) could not be applied: "
                              f"{self._calibration_error}. 'score_calibrated' is NaN and "
                              f"'eval(use_calibrated=True)' raises. Provide more samples per "
                              f"class (or a cloneable model), or fit with 'calibrate=False'.",
                              UserWarning)

        # Split-conformal reference (fit once)
        self._conf_state = (fit_conformal(
            clone(cloneable_base), np.asarray(X), labels, alpha=conformal_alpha,
            label_pos=label_pos, random_state=self._random_state)
            if cloneable_base is not None else None)
        if self._verbose:
            str_cal = (f"calibrated={self._calibrator is not None}"
                       if self._calibration_error is None else
                       f"calibrated=False (unavailable: {self._calibration_error})")
            ut.print_out(f"ReliabilityModel fitted (in-domain kNN threshold="
                         f"{self._ad_state['thr']:.3f}; uncertainty from {len(self._members)} "
                         f"member(s); {str_cal}; "
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
        ``ci_*`` (stability), ``ood_score`` / ``in_domain`` / ``ad_*`` (applicability domain,
        banded by ``ad_status``),
        ``margin`` / ``entropy`` (score ambiguity), ``conformal_set`` (validity), and
        ``reliable`` (the headline flag).

        .. versionchanged:: 1.2.0
           The applicability-domain column ``ad_knn_dist`` is named ``ad_knn``, matching its
           ``ad_mahalanobis`` / ``ad_leverage`` siblings, and two columns are appended at the
           end of the table: ``ad_status`` (the banded domain verdict) and ``ad_nearest_train``
           (the closest training row). Every previously returned column keeps its name,
           position, and values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to assess (same feature space as the training ``X``).

        Returns
        -------
        df_rel : pd.DataFrame
            One row per sample with columns: ``score``, ``score_std``, ``ci_low``, ``ci_high``,
            ``ood_score``, ``in_domain``, ``ad_knn``, ``ad_mahalanobis``, ``ad_leverage``,
            ``score_calibrated``, ``margin``, ``entropy``, ``conformal_set``, ``reliable``,
            ``ad_status``, ``ad_nearest_train``.

            * ``ad_status`` (str, never null): ``'inside'`` (``ood_score <= 1``),
              ``'borderline'`` (``1 < ood_score <= 1 + ad_borderline``), ``'outside'`` (above the
              band), or ``'unknown'`` (degenerate training reference, ``ood_score`` is ``NaN``).
              ``in_domain`` equals ``ad_status == 'inside'`` on every row.
            * ``ad_nearest_train`` (int): 0-based row index, into the ``X`` passed to :meth:`fit`,
              of the closest training sample.

            .. versionadded:: 1.2.0
               The ``ad_status`` and ``ad_nearest_train`` columns (appended at the end).

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
            ut.COL_RELIABLE: reliable,
            ut.COL_AD_STATUS: comp_ad_status(ad["ood_score"], ad["in_domain"],
                                             borderline=self._ad_borderline),
            ut.COL_AD_NEAREST_TRAIN: ad["nearest"]})

    def eval(self,
             *, X: Optional[ut.ArrayLike2D] = None,
             labels: Optional[ut.ArrayLike1D] = None,
             n_bins: int = 5,
             use_calibrated: bool = False,
             add_metrics: bool = False,
             ) -> pd.DataFrame:
        """
        Reliability diagnostics: calibration curve, empirical conformal coverage, in-domain rate.

        Aggregates :meth:`predict` over a labeled evaluation set into a compact table — per-bin
        predicted-vs-empirical positive rate (how well calibrated the score is) plus a summary row
        with the fraction in the applicability domain and the empirical coverage of the conformal
        sets (which should track ``1 - conformal_alpha``). Optionally, the calibrated score is
        binned instead of the raw one, and the Brier score and expected calibration error (ECE)
        [Guo17]_ are added as scalar rows, so two calibrations can be compared by number.

        .. versionchanged:: 1.2.0
           The per-bin sample-count column ``n`` is named ``n_samples``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Evaluation features; the training ``X`` is used when ``X`` is ``None`` (a held-out labeled set gives an honest estimate —
            calibration and conformal were fit on the training data).
        labels : array-like, shape (n_samples,), optional
            Evaluation labels, matching ``X`` in length; the training labels are used when both
            ``X`` and ``labels`` are ``None``. Labels given with ``X=None`` are scored against
            the training features and must match them in length. Passing ``X`` without
            ``labels`` raises.
        n_bins : int, default=5
            Number of equal-width score bins for the calibration curve (and the ECE).
        use_calibrated : bool, default=False
            If ``True``, score the calibrated column (``score_calibrated``) instead of the raw
            ``score``: the bins, Brier score, and ECE then describe the calibrated probability.
            Requires an available calibrator, i.e. a model fitted with ``calibrate=True`` whose
            calibration did not fail (:meth:`fit` warns when it does).

            .. versionadded:: 1.2.0
        add_metrics : bool, default=False
            If ``True``, append two scalar rows for the scored column: the Brier score
            (``bin='brier'``, mean squared difference between score and label) and the ECE
            (``bin='ece'``, the ``n_samples``-weighted mean of ``|mean_score - empirical_pos|``
            over the ``n_bins`` bins). Lower is better for both.

            .. versionadded:: 1.2.0

        Returns
        -------
        df_eval : pd.DataFrame
            Per-bin rows (``bin``, ``mean_score``, ``empirical_pos``, ``n_samples``) plus a
            summary row (``bin='summary'``) with the in-domain fraction (``mean_score``), the
            empirical conformal coverage (``empirical_pos``), and the number of evaluated samples
            (``n_samples``). With ``add_metrics=True``, a ``'brier'`` and an ``'ece'`` row follow,
            each holding its value in ``mean_score`` (``empirical_pos`` is ``NaN``, ``n_samples``
            is the number of evaluated samples).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``X`` is given without ``labels``, ``X`` or ``labels`` is invalid, their lengths
            differ, ``labels`` holds a value not observed during :meth:`fit`, ``n_bins`` is not an integer >= 2, ``use_calibrated`` or
            ``add_metrics`` is not a bool, or ``use_calibrated=True`` while no probability
            calibrator is available: either the model was fitted with ``calibrate=False``, or its
            calibration failed (the message names which).

        Notes
        -----
        * With both parameters left at their defaults, the output is the raw-score table of
          earlier versions, unchanged.
        * Scoring the calibrated column on the training data flatters the calibrator (it was fit
          there); pass a held-out ``X`` / ``labels`` to judge whether calibration helps.

        Examples
        --------
        .. include:: examples/rm_eval.rst
        """
        if self._ad_state is None:
            raise RuntimeError("Call 'fit' before 'eval'.")
        if X is None and labels is None:
            X, labels = self._X_train, self._y_train
        elif X is None:
            # Explicit labels with default features: score the training matrix against the
            # supplied labelling. The length check and the observed-label check below reject a
            # labelling that cannot belong to this training set.
            X = self._X_train
        elif labels is None:
            raise ValueError("'labels' (None) should be the evaluation labels matching 'X'; "
                             "only leaving BOTH 'X' and 'labels' as None evaluates on the "
                             "training data.")
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        train_classes = set(np.unique(self._y_train).tolist())
        unknown_labels = sorted(set(np.unique(labels).tolist()) - train_classes)
        if unknown_labels:
            raise ValueError(f"'labels' ({unknown_labels}) should contain only labels observed "
                             f"during 'fit' ({sorted(train_classes)}).")
        ut.check_number_range(name="n_bins", val=n_bins, min_val=2, just_int=True)
        ut.check_bool(name="use_calibrated", val=use_calibrated)
        ut.check_bool(name="add_metrics", val=add_metrics)
        if use_calibrated and self._calibrator is None:
            reason = _reason_no_calibrator(calibrate_requested=self._calibrate_requested,
                                           calibration_error=self._calibration_error)
            raise ValueError(f"'use_calibrated' ({use_calibrated}) should be False for this "
                             f"model: it has no probability calibrator because {reason}.")
        y = (np.asarray(labels) == self.label_pos_).astype(int)
        df = self.predict(X)
        s = df[ut.COL_SCORE_CAL if use_calibrated else ut.COL_SCORE].to_numpy()
        rows = comp_calibration_bins(s, y, n_bins=n_bins)
        sets = df[ut.COL_CONFORMAL_SET].to_numpy()
        covered = (np.isin(sets, [ut.STR_CONF_POS, ut.STR_CONF_BOTH]) & (y == 1)) | \
                  (np.isin(sets, [ut.STR_CONF_NEG, ut.STR_CONF_BOTH]) & (y == 0))
        rows.append([ut.STR_BIN_SUMMARY, float(np.mean(df[ut.COL_IN_DOMAIN])),
                     float(np.mean(covered)), len(X)])
        if add_metrics:
            rows.append([ut.STR_BIN_BRIER, comp_brier(s, y), np.nan, len(X)])
            ece = comp_ece(rows[:n_bins], n_samples=len(X))
            rows.append([ut.STR_BIN_ECE, ece, np.nan, len(X)])
        return pd.DataFrame(rows, columns=ut.COLS_EVAL_RELIABILITY)
