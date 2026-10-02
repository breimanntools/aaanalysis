"""
This is a script for the frontend of the ReliabilityModel class for prediction-reliability measures.
"""
from typing import Optional, List, Union, Literal, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper
from aaanalysis.feature_engineering._sequence_feature import SequenceFeature

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


def check_ad_borderline(ad_borderline: float):
    """Check that ``ad_borderline`` is a finite, non-negative number (not a bool)."""
    if isinstance(ad_borderline, bool):
        raise ValueError(f"'ad_borderline' ({ad_borderline}) should be a finite number >= 0 "
                         f"(a float or an integer), but got bool.")
    ut.check_number_range(name="ad_borderline", val=ad_borderline, min_val=0, just_int=False)


def check_ci(ci: float):
    """Check that ``ci`` is a fraction in (0, 1), with a hint when a percent is passed."""
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


def get_list_features(features: Union[ut.ArrayLike1D, pd.DataFrame]) -> list:
    """Return the feature ids as a plain list, accepting a ``df_feat`` or an array-like."""
    if isinstance(features, pd.DataFrame):
        ut.check_df(name="features", df=features, cols_required=[ut.COL_FEATURE])
        return list(features[ut.COL_FEATURE])
    return ut.check_list_like(name="features", val=features, accept_str=True, min_len=1)


def get_parts_from_features(features: list) -> list:
    """Return the sorted lower-case sequence parts referenced by a list of feature ids."""
    parts = set()
    for feat_id in features:
        ut.check_str(name="feature id", val=feat_id)
        if feat_id.count("-") != 2:
            raise ValueError(f"'features' entry ('{feat_id}') should follow the "
                             f"'PART-SPLIT-SCALE' grammar (e.g. 'TMD-Segment(1,1)-LINS010101').")
        parts.add(ut.split_feat_id(feat_id=feat_id)[0].lower())
    return sorted(parts)


def check_df_seq_pos_based(df_seq: pd.DataFrame) -> None:
    """Check that ``df_seq`` is position-based (sequence + TMD coordinates) with unique entries."""
    ut.check_df_seq(df_seq=df_seq)
    missing = [c for c in ut.COLS_SEQ_POS if c not in df_seq.columns]
    if len(missing) > 0:
        raise ValueError(f"'df_seq' should be in the position-based format with columns "
                         f"{ut.COLS_SEQ_POS}; missing: {missing}. The wild-type sequences and "
                         f"their TMD coordinates are needed to rebuild the candidate parts.")
    entries = list(df_seq[ut.COL_ENTRY])
    if len(set(entries)) != len(entries):
        raise ValueError("'df_seq' should contain unique 'entry' values (one row per wild-type).")


def build_df_seq_candidates(df_cand: pd.DataFrame, df_seq: pd.DataFrame,
                            col_seq: str) -> pd.DataFrame:
    """Build a position-based ``df_seq`` for the candidates, reusing wild-type TMD coordinates.

    Each candidate gets a unique synthetic entry id, so duplicate candidate rows (the same
    variant of the same wild-type) never collapse and the output stays row-aligned.
    """
    seq_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
    start_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_START]))
    stop_by_entry = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_STOP]))
    list_entries, list_seq, list_start, list_stop = [], [], [], []
    for i, (entry, seq) in enumerate(zip(df_cand[ut.COL_ENTRY], df_cand[col_seq])):
        if entry not in seq_by_entry:
            raise ValueError(f"'df_cand' entry ('{entry}') is not in 'df_seq'. Available "
                             f"entries: {ut.preview_options(seq_by_entry)}.")
        ut.check_str(name=f"'{col_seq}' (entry: '{entry}')", val=seq, accept_none=False)
        list_entries.append(f"{entry}__{i}")
        list_seq.append(seq)
        list_start.append(start_by_entry[entry])
        list_stop.append(stop_by_entry[entry])
    return pd.DataFrame({ut.COL_ENTRY: list_entries, ut.COL_SEQ: list_seq,
                         ut.COL_TMD_START: list_start, ut.COL_TMD_STOP: list_stop})


# II Main Functions
class ReliabilityModel(Wrapper):
    """
    Assess **how much to trust** each prediction: the reliability of a score, not the score itself.

    A high score is not the same as a trustworthy one. A model can be right to call a case a
    ``0.55`` toss-up, and badly wrong to call a protein it has never seen a ``1.0``. Alongside the
    score, this class answers the three questions that decide trust:

    1. **Has the model seen anything like this before?** An input unlike the training data leaves
       the model guessing, however high its score (applicability domain: ``ood_score``,
       ``in_domain``, ``ad_status``).
    2. **Do repeated models agree?** A score the ensemble members, or refits on resampled data,
       disagree about is shaky (stability: ``score_std``, ``ci_low`` / ``ci_high``).
    3. **Is the case clear-cut, or a genuine toss-up?** A well-calibrated score near ``0.5`` means
       "honestly borderline", not "broken" (decisiveness: ``margin``, ``entropy``, and a conformal
       set that may abstain).

    The headline flag ``reliable`` combines the first and the last: familiar **and** decisive. The
    class wraps an already-fitted binary predictor (an :class:`AAPred`, a
    :class:`~aaanalysis.TreeModel`, or any scikit-learn classifier) together with its training
    data; the prediction itself stays with the model.

    Questions 1-2 concern epistemic uncertainty, question 3 aleatoric uncertainty
    [Huellermeier21]_. The applicability domain follows the QSAR idea [Sahigara12]_, calibration
    follows [Guo17]_, and the conformal set follows [Angelopoulos23]_.

    .. warning::

        **Experimental.** This class is part of the new v1.1.0 prediction layer and is under active
        development; its API (signatures, defaults, return objects) may change between minor releases
        without the usual deprecation cycle. Pin a version if you depend on the current behaviour.

    .. versionadded:: 1.1.0

    Notes
    -----
    * Binary classification only. ``ad_mahalanobis`` and ``ad_leverage`` are auxiliary diagnostics
      that need more training samples than features and are ``NaN`` otherwise; the ``in_domain``
      decision rests on the k-nearest-neighbor distance and is always valid.
    * ``score`` is the mean over the ensemble members, or over the bootstrap resamples of a single
      model, so it is always the centre of ``[ci_low, ci_high]``. ``score_std`` is one sample's
      spread across those members. The same-named column of :meth:`AAPred.eval` and
      :meth:`ModelEvaluator.run` means something else: the spread of a metric across folds.
    * Calibration affects ``score_calibrated``, ``margin``, and ``entropy`` only. ``score`` stays
      the raw model score, and it is what :meth:`eval` and
      :meth:`ReliabilityModelPlot.reliability_diagram` report unless the calibrated score is asked
      for.
    * The bootstrap, calibration split, and conformal split are stochastic; set ``random_state``
      for identical output across fits.

    Attributes
    ----------
    model_ : estimator or list of estimators
        The assessed model (the fitted single estimator or the list of ensemble members).
    label_pos_ : int
        Positive-class label whose probability is scored.
    ad_threshold_ : float
        Applicability-domain boundary: the ``ad_percentile``-th percentile of the training samples'
        distance to their ``k`` nearest neighbors, so ``ood_score == 1`` sits on the boundary.

        .. versionadded:: 1.2.0
    ad_method_ : str
        Decision rule behind ``ood_score`` and ``in_domain``; always ``"knn"``.

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
            Seed for the bootstrap, calibration split, and conformal split. If ``None``, those
            stochastic steps use a fresh random state on each fit.

        Raises
        ------
        ValueError
            If ``verbose`` is not a bool or ``random_state`` is not an integer or ``None``.
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
            *, model: Optional[Union[object, List, Tuple]] = None,
            label_pos: int = 1,
            k: int = 5,
            ad_percentile: float = 95.0,
            ad_borderline: float = 0.1,
            ci: float = 0.90,
            n_bootstrap: int = 20,
            calibrate: bool = True,
            calibration_method: Literal["isotonic", "sigmoid"] = "isotonic",
            conformal_alpha: float = 0.1,
            ) -> "ReliabilityModel":
        """
        Fit the reliability reference from a (fitted or default) model and its training data.

        Learns, once, everything :meth:`predict` needs: the applicability-domain reference, the
        ensemble or bootstrap source of uncertainty, an optional probability calibrator, and the
        split-conformal calibration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix with at least three rows and two features; it must not contain
            only identical rows. It is the applicability-domain reference and the data used when
            fitting a default model.
        labels : array-like of int, shape (n_samples,)
            Integer training labels matching ``X`` in length. Exactly two classes are required;
            choosing ``label_pos`` determines which class is scored as positive.
        model : estimator, list or tuple of estimators, AAPred, or None, default=None
            A scikit-learn classifier with ``predict_proba`` (an unfitted estimator is fitted on
            ``X`` / ``labels``), a list or tuple of such estimators (ensemble; uncertainty = their
            disagreement), a fitted :class:`AAPred`, or ``None``. If ``None``, fit a default
            :class:`~sklearn.ensemble.RandomForestClassifier`.
        label_pos : int, default=1
            Non-negative positive-class label whose probability is scored. It must occur in
            ``labels``.
        k : int, default=5
            Positive number of nearest training neighbors used for the applicability-domain
            distance; values above the available training neighbors are capped.
        ad_percentile : float, default=95.0
            Training kNN-distance percentile used as the ``in_domain`` boundary (stored as
            ``ad_threshold_``). Must be finite and between 1 and 100, inclusive.
        ad_borderline : float, default=0.1
            Width of the ``borderline`` band just outside the domain boundary, relative to it:
            a sample with ``1 < ood_score <= 1 + ad_borderline`` gets ``ad_status='borderline'``,
            above that ``'outside'``. ``0`` disables the band. Must be a **finite non-negative**
            number (a negative value, ``inf``, ``NaN``, or a bool raises).

            .. versionadded:: 1.2.0
        ci : float, default=0.90
            Central width of the reported score confidence interval, as a fraction in ``(0, 1)``
            (e.g. ``0.90`` for a 90% interval), matching :meth:`ModelEvaluator.run`.

            .. versionchanged:: 1.2.0 A fraction, no longer a percent (``90.0``).
        n_bootstrap : int, default=20
            Bootstrap resamples for uncertainty when ``model`` is a single estimator (not an
            ensemble); ``score`` is then the bagged mean over the resamples (see Notes). ``0``
            disables the bootstrap and reports the model's own probability (``score_std`` = 0).
            Must be a non-negative integer.
        calibrate : bool, default=True
            If ``True``, fit a probability calibrator for ``score_calibrated``, ``margin``, and
            ``entropy``; for an ensemble it is built from the first member. If fitting fails, emit
            a ``UserWarning`` and leave ``score_calibrated`` as ``NaN``; if ``False``,
            ``score_calibrated`` is ``NaN`` and sharpness uses ``score``.
        calibration_method : {'isotonic', 'sigmoid'}, default='isotonic'
            Calibration method passed to :class:`~sklearn.calibration.CalibratedClassifierCV`.

            - ``'isotonic'``: fit a non-parametric monotonic calibration curve.
            - ``'sigmoid'``: fit Platt's sigmoid calibration curve.
        conformal_alpha : float, default=0.1
            Finite miscoverage level of the split-conformal set (``1 - alpha`` coverage), from 0
            through 1 inclusive.

        Returns
        -------
        self : ReliabilityModel
            The fitted instance.

        Raises
        ------
        ValueError
            If ``X`` / ``labels`` are invalid or have different lengths, ``labels`` are not binary,
            ``label_pos`` is absent from ``labels``, ``model`` is an empty list or lacks
            ``predict_proba``, a passed :class:`AAPred` is not fitted, ``calibration_method`` is
            not ``'isotonic'`` or ``'sigmoid'``, or a numeric parameter is out of range or not
            finite (``NaN`` / ``inf``), including a negative ``ad_borderline``.

        Warnings
        --------
        UserWarning
            If ``calibrate=True`` but no calibrator can be fitted, for example because a class
            holds fewer members than the internal cross-validation needs or the model cannot be
            cloned. ``score_calibrated`` is then ``NaN`` and :meth:`eval` with
            ``use_calibrated=True`` raises, naming that reason.

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
        check_ad_borderline(ad_borderline=ad_borderline)
        check_ci(ci=ci)
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
        self._ad_borderline = float(ad_borderline)
        # fit_applicability_domain stores a float under "thr"; the dict's value type widens.
        self.ad_threshold_ = float(self._ad_state["thr"])  # pyright: ignore[reportArgumentType]
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

        Applies the references learned by :meth:`fit`, without refitting any model, so repeated
        calls are cheap and deterministic. Each column belongs to one of the three trust
        questions: stability (``score_std``, ``ci_low`` / ``ci_high``), applicability domain
        (``ood_score``, ``in_domain``, ``ad_*``), and decisiveness (``margin``, ``entropy``,
        ``conformal_set``), with ``reliable`` as the headline flag.

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

            * ``score`` is the mean positive-class probability across model members; ``score_std``
              and ``ci_low`` / ``ci_high`` are its across-member spread and confidence interval.
            * ``ood_score`` is the k-nearest-neighbor distance divided by ``ad_threshold_`` when
              that threshold is positive (otherwise ``NaN``); ``ad_knn`` is the raw mean distance.
              ``ad_mahalanobis`` and ``ad_leverage`` are auxiliary distance diagnostics (both are
              ``NaN`` when features equal or outnumber samples).
            * ``score_calibrated`` is the calibrated positive-class probability (``NaN`` when no
              calibrator is available); ``margin`` and ``entropy`` quantify its decisiveness.
            * ``conformal_set`` is the split-conformal set (``'pos'``, ``'neg'``, ``'both'``, or
              ``'none'``). ``reliable`` requires an in-domain singleton, or an in-domain margin
              of at least ``0.5`` when no conformal reference is available.
            * ``ad_status`` (str, never null): ``'inside'`` (``ood_score <= 1``),
              ``'borderline'`` (``1 < ood_score <= 1 + ad_borderline``), ``'outside'`` (above the
              band), or ``'unknown'`` (degenerate training reference, ``ood_score`` is ``NaN``).
              ``in_domain`` equals ``ad_status == 'inside'`` on every row.
            * ``ad_nearest_train`` (int): 0-based row index, into the ``X`` passed to :meth:`fit`,
              of the closest training sample.

            .. versionadded:: 1.2.0 ``ad_status`` and ``ad_nearest_train``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``X`` is invalid or has a different number of features than the training data.

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

    def predict_candidates(self,
                           df_cand: pd.DataFrame,
                           df_seq: pd.DataFrame,
                           features: Union[ut.ArrayLike1D, pd.DataFrame],
                           *, df_scales: Optional[pd.DataFrame] = None,
                           col_seq: str = ut.COL_SEQ_MUT,
                           jmd_n_len: int = 10,
                           jmd_c_len: int = 10,
                           n_jobs: Optional[int] = 1,
                           ) -> pd.DataFrame:
        """
        Score a set of design candidates for reliability, rebuilding their feature matrix first.

        A designed variant sits, by construction, away from the sequences the model was trained
        on, which is exactly what makes its score hard to trust. The applicability-domain columns
        (``ood_score``, ``ad_status``, ``ad_nearest_train``) say which candidates the model still
        has ground for, and ``reliable`` is the headline verdict.

        The design tier (:meth:`SeqMut.mutate`, :meth:`SeqMut.combine`, :meth:`SeqOpt.run`) returns
        sequences, while :meth:`predict` takes a feature matrix. This method bridges the two: it
        rebuilds the matrix with :meth:`SequenceFeature.feature_matrix`, reusing the wild-type TMD
        coordinates from ``df_seq``, and scores it with :meth:`predict`.

        .. versionadded:: 1.2.0

        Parameters
        ----------
        df_cand : pd.DataFrame, shape (n_candidates, n_cand_info)
            Candidate table from the design tier: an ``entry`` column naming the wild-type each
            candidate derives from, plus the candidate sequence in the ``col_seq`` column.
            :meth:`SeqMut.mutate`, :meth:`SeqMut.combine` and :meth:`SeqOpt.run` all emit this
            shape (``entry`` + ``sequence_mut``); every other column is ignored and kept out of
            the result.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, in the
            **position-based** format (``sequence``, ``tmd_start``, ``tmd_stop``). See
            :meth:`SequenceFeature.get_df_parts` for the full ``df_seq`` format specification.
            It supplies the wild-type TMD coordinates reused for every candidate of an ``entry``.
        features : array-like, shape (n_features,) or pd.DataFrame
            Feature ids (``'PART-SPLIT-SCALE'``), or a ``df_feat`` whose ``'feature'`` column
            holds them. Pass the same feature set, in the same order, that produced the training
            ``X`` of :meth:`fit`; a different number of features raises.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of amino acid scales (index = amino acids, columns = scale ids). Pass the
            scales used to build the training ``X``. If ``None``, the default scale set from
            :func:`load_scales` is used.
        col_seq : str, default='sequence_mut'
            Name of the ``df_cand`` column carrying the candidate sequence. ``'sequence_mut'`` is
            the column :meth:`SeqMut.mutate`, :meth:`SeqMut.combine` and :meth:`SeqOpt.run`
            emit; pass ``'sequence'`` to score wild-type sequences with the same call.
        jmd_n_len : int, default=10
            Length of JMD-N in number of amino acids (a non-negative integer). Use the value the
            training matrix was built with, or the candidate parts will not match it.
        jmd_c_len : int, default=10
            Length of JMD-C in number of amino acids (a non-negative integer). Use the value the
            training matrix was built with, or the candidate parts will not match it.
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used to build the candidate feature matrix. If ``None``,
            the number is optimized automatically; if ``-1``, all available cores are used. It
            changes speed only, never the returned values.

        Returns
        -------
        df_rel : pd.DataFrame, shape (n_candidates, 16)
            The :meth:`predict` table for the rebuilt candidate matrix: one row per ``df_cand``
            row, in ``df_cand`` order and carrying its index, with exactly the columns
            :meth:`predict` returns (``score`` ... ``reliable``, ``ad_status``,
            ``ad_nearest_train``). Attach it to the candidates with ``df_cand.join(df_rel)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``df_cand`` is not a DataFrame, is empty, or lacks an ``entry`` / ``col_seq``
            column; if a candidate ``entry`` is absent from ``df_seq`` or its sequence is not a
            string; if ``df_seq`` is not position-based or its entries are not unique; if
            ``features`` is empty or a feature id does not follow the ``'PART-SPLIT-SCALE'``
            grammar; if ``col_seq`` is not a string, ``jmd_n_len`` / ``jmd_c_len`` are not
            non-negative integers, or ``n_jobs`` is invalid; or if the rebuilt matrix has a
            different number of features than the training ``X``.

        Notes
        -----
        * **Substitutions only.** A substituted candidate keeps the wild-type length, so the
          wild-type ``tmd_start`` / ``tmd_stop`` still locate its TMD. Insertions or deletions
          shift those coordinates: score such candidates by passing a ``df_seq`` whose
          coordinates are already corrected for them.
        * Duplicate candidate rows are kept and scored independently, so the output stays
          row-aligned with ``df_cand``.

        Examples
        --------
        .. include:: examples/rm_predict_candidates.rst
        """
        if self._ad_state is None:
            raise RuntimeError("Call 'fit' before 'predict_candidates'.")
        ut.check_str(name="col_seq", val=col_seq)
        ut.check_df(name="df_cand", df=df_cand, cols_required=[ut.COL_ENTRY, col_seq])
        if len(df_cand) == 0:
            raise ValueError("'df_cand' should contain at least one candidate row.")
        check_df_seq_pos_based(df_seq=df_seq)
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        list_features = get_list_features(features=features)
        list_parts = get_parts_from_features(features=list_features)
        df_seq_cand = build_df_seq_candidates(df_cand=df_cand, df_seq=df_seq, col_seq=col_seq)
        # Rebuild the candidate matrix through the same builder that made the training X,
        # then delegate to predict (feature-count mismatch is caught there).
        sf = SequenceFeature(verbose=False)
        X = sf.feature_matrix(features=list_features, df_seq=df_seq_cand, df_scales=df_scales,
                              n_jobs=n_jobs,
                              df_parts_kws=dict(list_parts=list_parts, jmd_n_len=jmd_n_len,
                                                jmd_c_len=jmd_c_len))
        df_rel = self.predict(np.asarray(X, dtype=float))
        df_rel.index = df_cand.index
        if self._verbose:
            n_in = int(df_rel[ut.COL_IN_DOMAIN].sum())
            ut.print_out(f"ReliabilityModel scored {len(df_rel)} candidate(s) "
                         f"({n_in} inside the applicability domain).")
        return df_rel

    def eval(self,
             *, X: Optional[ut.ArrayLike2D] = None,
             labels: Optional[ut.ArrayLike1D] = None,
             n_bins: int = 5,
             use_calibrated: bool = False,
             add_metrics: bool = False,
             ) -> pd.DataFrame:
        """
        Reliability diagnostics: calibration curve, empirical conformal coverage, in-domain rate.

        Aggregates :meth:`predict` over a labeled evaluation set: the predicted against the
        empirical positive rate per bin (how well calibrated the score is), plus a summary row with
        the fraction inside the applicability domain and the empirical coverage of the conformal
        sets, which should track ``1 - conformal_alpha``. The Brier score and the expected
        calibration error (ECE) [Guo17]_ can be added as scalar rows to compare two calibrations
        by number.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Evaluation features. If ``X`` and ``labels`` are both ``None``, use the training
            features; passing a held-out labeled set gives an honest estimate because calibration
            and conformal were fit on the training data.
        labels : array-like of int, shape (n_samples,), optional
            Evaluation labels, matching ``X`` in length; the training labels are used when both
            ``X`` and ``labels`` are ``None``. Labels given with ``X=None`` are scored against
            the training features and must match them in length. Passing ``X`` without
            ``labels`` raises.
        n_bins : int, default=5
            Number of equal-width score bins for the calibration curve (and the ECE). Must be an
            integer of at least 2.
        use_calibrated : bool, default=False
            If ``True``, score the calibrated column (``score_calibrated``) instead of the raw
            ``score``: the bins, Brier score, and ECE then describe the calibrated probability.
            If ``False``, score the raw model probability. If ``True``, an available calibrator is
            required: fit with ``calibrate=True`` and ensure calibration did not fail
            (:meth:`fit` warns when it does).

            .. versionadded:: 1.2.0
        add_metrics : bool, default=False
            If ``True``, append two scalar rows for the scored column: the Brier score
            (``bin='brier'``, mean squared difference between score and label) and the ECE
            (``bin='ece'``, the ``n_samples``-weighted mean of ``|mean_score - empirical_pos|``
            over the ``n_bins`` bins). Lower is better for both.
            If ``False``, return only the per-bin and summary rows.

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
            differ, ``labels`` holds a value not observed during :meth:`fit`, ``n_bins`` is not an
            integer of at least 2, ``use_calibrated`` or ``add_metrics`` is not a bool, or
            ``use_calibrated=True`` while no probability calibrator is available: either the model
            was fitted with ``calibrate=False``, or its calibration failed (the message names
            which).

        Notes
        -----
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
