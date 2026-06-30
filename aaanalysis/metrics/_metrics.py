"""
This is a script for special statistical measures not available or adjusted from popular python data analysis packages
such as scikit-learn, scipy, or statsmodels.

Developer note: Measures are implemented in aanalysis.utils.metrics to access them within the aanalysis package.
"""
import numpy as np
from typing import Optional, Literal, Union, Any

from aaanalysis.utils import (auc_adjusted_, kullback_leibler_divergence_, bic_score_,
                              per_protein_ap_, detection_metrics_, bootstrap_ci_,
                              smooth_scores_, eval_features_, LIST_METRICS_EVAL)
import aaanalysis.utils as ut


# Helper functions
def _check_n_classes_n_samples(X: np.ndarray, labels: np.ndarray):
    """Check matching X and labels"""
    n_classes = len(set(labels))
    n_samples, n_features = X.shape
    if n_classes >= n_samples:
        raise ValueError(f"Number of classes in 'labels' ({n_classes}) must be smaller than n_samples ({n_samples})")
    if n_features == 0:
        raise ValueError(f"'n_features' should not be 0")


# Adjusted Area Under the Curve (AUC*)
def comp_auc_adjusted(X: ut.ArrayLike2D,
                      labels: ut.ArrayLike1D,
                      label_test: int = 1,
                      label_ref: int = 0,
                      n_jobs: Optional[int] = None
                      ) -> ut.ArrayLike1D:
    """
    Compute an adjusted Area Under the Curve (AUC) [-0.5, 0.5] assessing the similarity between two groups.

    Introduced in [Breimann25]_, this adjusted AUC (denoted 'AUC*') is computed for each feature in the
    dataset ``X``, comparing two groups specified by the labels. It is based on the non-parametric measure of the
    difference between two groups. The adjustment of AUC subtracts 0.5, so it ranges between -0.5 and 0.5.
    An AUC* of 0 indicates an equal distribution between the two groups. This measure is useful for ranking features
    based on their ability to distinguish between the two groups.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. 'Rows' typically correspond to proteins and 'columns' to features.
    labels : array-like, shape (n_samples,)
        Dataset labels of samples in X. Should contain only two different integer label values,
        representing test and reference group (typically, 1 and 0).
    label_test : int, default=1,
        Class label of test group in ``labels``.
    label_ref : int, default=0,
        Class label of reference group in ``labels``.
    n_jobs : int, None, or -1, default=None
        Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
        If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

    Returns
    -------
    auc : array-like, shape (n_features,)
        Array with AUC* values for each feature, ranging from [-0.5, 0.5].
        A value of 0 indicates equal distributions between the two groups for that feature.

    Examples
    --------
    .. include:: examples/comp_auc_adjusted.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X, min_n_unique_samples=1)
    ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
    ut.check_number_val(name="label_ref", val=label_ref, just_int=True, accept_none=False)
    labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref], allow_other_vals=False)
    ut.check_match_X_labels(X=X, labels=labels)
    n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
    # Compute adjusted AUC
    auc = auc_adjusted_(X=X, labels=labels, label_test=label_test, n_jobs=n_jobs)
    return auc


# BIC score
def comp_bic_score(X: ut.ArrayLike2D,
                   labels: ut.ArrayLike1D
                   ) -> float:
    """
    Compute an adjusted Bayesian Information Criterion (BIC) (-∞, ∞) for assessing clustering quality.

    Described in [Breimann24b]_, this adjusted BIC is computed for a given set of clusters in the dataset ``X``.
    The BIC is a clustering model selection criterion that balances the model complexity against the
    likelihood of the data distribution. Unlike the traditional BIC where lower values are better, this adjusted BIC,
    is modified to align with other clustering evaluation measures like the
    Silhouette coefficient and the Calinski-Harabasz score. In this adjusted version, higher values indicate
    better clustering.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. 'Rows' typically correspond to proteins and 'columns' to features.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample. Each label corresponds to a cluster.

    Returns
    -------
    bic : float
        The adjusted Bayesian Information Criterion value. Higher values indicate better clustering quality.

    Notes
    -----
    *  An `epsilon` value (1e-10) is utilized to prevent division by zero in the computation.

    See Also
    --------
    * The Silhouette coefficient [-1, 1] can be computed by :func:`sklearn.metrics.silhouette_score`.
    * The Calinski Harabasz score [0, ∞] can be obtained using :func:`sklearn.metrics.calinski_harabasz_score`.
    * Clustering evaluation can be performed using :meth:`AAclust.eval`.

    Examples
    --------
    .. include:: examples/comp_bic_score.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X)
    labels = ut.check_labels(labels=labels)
    ut.check_match_X_labels(X=X, labels=labels)
    _check_n_classes_n_samples(X=X, labels=labels)
    # Compute bic
    bic = bic_score_(X, labels=labels)
    return bic


# Kullback-Leibler Divergence
def comp_kld(X: ut.ArrayLike2D,
             labels: ut.ArrayLike1D,
             label_test: int = 1,
             label_ref: int = 0
             ) -> ut.ArrayLike1D:
    """
    Compute the Kullback-Leibler Divergence (KLD) [0, ∞) for assessing the similarity between two groups.

    The KLD is calculated for each feature in ``X``, comparing the distributions between two subgroups specified
    by ``label_test`` and ``label_ref`` in labels. Generally, the KLD measures how one probability distribution
    diverges from a second, expected probability distribution. Higher KLD values indicate more divergence. The observed
    upper limit lies around 200 indicating complete divergence of two non-overlapping distributions.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
    labels : array-like, shape (n_samples,)
        Labels for each sample in ``X``. Should contain only integer label values and at least 2 per class.
    label_test : int, default=1,
        Class label of test group in ``labels``.
    label_ref : int, default=0,
        Class label of reference group in ``labels``.

    Returns
    -------
    kld : array-like, shape (n_features,)
        Array of Kullback-Leibler Divergence values for each feature in ``X``. Each value represents
        the divergence of the test group distribution from the reference group distribution for that feature.

    Notes
    -----
    * For valid KLD calculations, the input matrix `X` must meet certain conditions:

      - Ensure adequate variability of features in ``X`` to avoid computational problems like singular
        covariance matrices in Gaussian Kernel Density Estimation (KDE).
      - Avoid rows in ``X`` lying in a lower-dimensional subspace; consider dimensionality reduction if necessary.

    See Also
    --------
    * :func:`scipy.stats.gaussian_kde` function representing a kernel-density estimate using Gaussian kernels.
      It is used for estimating the probability density function of a random variable (i.e., feature in ``X``),
      which is a crucial step in the computation of Kullback-Leibler Divergence (KLD).
    * :func:`scipy.stats.entropy` function for computing the Shannon entropy. In the context of KLD,
      it is used to measure the divergence between two probability distributions, typically derived
      from kernel-density estimates of different data groups.

    Examples
    --------
    .. include:: examples/comp_kld.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X, min_n_unique_samples=3)
    ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
    ut.check_number_val(name="label_ref", val=label_ref, just_int=True, accept_none=False)
    labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                             n_per_group_required=2, allow_other_vals=False)
    ut.check_match_X_labels(X=X, labels=labels, check_variability=True)
    # Compute tge Kullback-Leibler divergence
    try:
        kld = kullback_leibler_divergence_(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
    except Exception as e:
        raise ValueError(f"Following error occurred during the computation of Kullback-Leibler Divergence: {e}")
    return kld


# Helper functions (site-localization metrics)
def _check_list_scores_positions(list_scores=None, list_positions=None) -> "tuple[list, list]":
    """Validate the (scores, positions) per-protein lists used by the
    site-localization metrics."""
    list_scores = ut.check_list_like(name="list_scores", val=list_scores)
    list_positions = ut.check_list_like(name="list_positions", val=list_positions)
    if len(list_scores) != len(list_positions):
        raise ValueError(f"'list_scores' (n={len(list_scores)}) and 'list_positions' "
                         f"(n={len(list_positions)}) should have the same length.")
    if len(list_scores) == 0:
        raise ValueError("'list_scores' should not be empty.")
    return list_scores, list_positions


# Helper functions (feature-set evaluation)
def _check_model(model=None):
    """Validate that ``model`` is an sklearn-style classifier (``fit`` + ``predict``)."""
    if model is None:
        return None
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise ValueError(f"'model' ('{model}') must be an sklearn-style estimator "
                         f"with 'fit' and 'predict' methods, or None for the default linear SVM.")


def _check_cv(cv=None):
    """Validate that ``cv`` is None, an int (>=2), or a CV splitter with a ``split`` method."""
    if cv is None:
        return None
    if isinstance(cv, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"'cv' ('{cv}') must be None, an int >= 2, or a CV splitter, not a bool.")
    if isinstance(cv, int):
        ut.check_number_range(name="cv", val=cv, min_val=2, just_int=True)
        return None
    if not hasattr(cv, "split"):
        raise ValueError(f"'cv' ('{cv}') must be None, an int >= 2, or a CV splitter with a "
                         f"'split' method (e.g. LeaveOneOut, StratifiedKFold).")


def _check_mask_known_pos(mask_known_pos=None, n_samples=None):
    """Validate the PU mask-known-positives boolean array."""
    if mask_known_pos is None:
        return None
    mask = ut.check_array_like(name="mask_known_pos", val=mask_known_pos)
    mask = np.asarray(mask)
    if mask.ndim != 1 or len(mask) != n_samples:
        raise ValueError(f"'mask_known_pos' should be a 1D boolean array of length n_samples "
                         f"({n_samples}); got shape {mask.shape}.")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("'mask_known_pos' should contain only boolean (0/1) values.")


# Feature-set evaluation (model + CV + metric scorer; incl. PU mask-known-positives CV)
def eval_features(X: ut.ArrayLike2D,
                  labels: ut.ArrayLike1D,
                  model: Optional[Any] = None,
                  cv: Optional[Union[int, Any]] = None,
                  metric: str = "balanced_accuracy",
                  mask_known_pos: Optional[ut.ArrayLike1D] = None,
                  random_state: Optional[int] = None,
                  ) -> float:
    """
    Score a feature set by cross-validated classification performance.

    Benchmarking a feature set is the recurring spine of sequence-based protein
    prediction: train a classifier on the feature matrix ``X`` and score its
    cross-validated agreement with ``labels``. The default reproduces the linear-SVM,
    leave-one-out, balanced-accuracy recipe used throughout the γ-secretase analysis
    (``balanced_accuracy_score(y, cross_val_predict(SVC(kernel='linear'), X, y,
    cv=LeaveOneOut())) * 100``), while any scikit-learn estimator, CV splitter, and
    classification metric can be swapped in without leaving the API.

    Setting ``mask_known_pos`` selects the Positive-Unlabeled (PU) mask-known-positives
    CV variant: the masked known positives still inform every training fold but are
    never scored as test points, so the reported score reflects only the held-out,
    non-trivial samples.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. 'Rows' typically correspond to proteins and 'columns' to features.
    labels : array-like, shape (n_samples,)
        Class labels for each sample in ``X`` (typically 1 for the test/positive class
        and 0 for the reference/negative class).
    model : object, optional
        A scikit-learn-style classifier exposing ``fit`` and ``predict``. If ``None``,
        a linear-kernel Support Vector Machine (``SVC(kernel='linear')``) is used.
    cv : int or cross-validation splitter, optional
        Cross-validation strategy. An ``int`` (``>= 2``) selects k-fold CV; a splitter
        object (e.g. :class:`sklearn.model_selection.StratifiedKFold`) is used directly.
        If ``None``, leave-one-out cross-validation is used.
    metric : str, default='balanced_accuracy'
        Classification metric name. One of ``'balanced_accuracy'``, ``'accuracy'``,
        ``'f1'``, ``'precision'``, ``'recall'``, ``'matthews_corrcoef'``.
    mask_known_pos : array-like of bool, shape (n_samples,), optional
        PU mask of known positives. Masked samples are kept in every training fold but
        excluded from scoring. If ``None``, all samples are scored.
    random_state : int, optional
        Random seed forwarded to ``model`` when it accepts a ``random_state``
        parameter, ensuring reproducible scores for stochastic estimators.

    Returns
    -------
    score : float
        Cross-validated ``metric`` score scaled to a percentage (the bounded metric
        value multiplied by 100).

    Notes
    -----
    * The default (linear SVM + leave-one-out + balanced accuracy) reproduces the
      γ-secretase benchmark numbers within numerical tolerance on the same feature
      matrix.
    * Given a fixed ``random_state`` (and a deterministic ``cv``), repeated calls
      return an identical score.

    See Also
    --------
    * :func:`comp_auc_adjusted` for a per-feature class-separation score.
    * :class:`sklearn.model_selection.cross_val_predict` underlies the default path.

    Examples
    --------
    .. include:: examples/eval_features.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
    labels = ut.check_labels(labels=labels)
    ut.check_match_X_labels(X=X, labels=labels)
    _check_model(model=model)
    _check_cv(cv=cv)
    ut.check_str_options(name="metric", val=metric, list_str_options=LIST_METRICS_EVAL)
    _check_mask_known_pos(mask_known_pos=mask_known_pos, n_samples=X.shape[0])
    random_state = ut.check_random_state(random_state=random_state)
    # Compute cross-validated feature-set score
    score = eval_features_(X=X, labels=labels, model=model, cv=cv, metric=metric,
                           mask_known_pos=mask_known_pos, random_state=random_state)
    return score


# Per-protein average precision (site localization)
def comp_per_protein_ap(list_scores: list,
                        list_positions: list,
                        tolerance: int = 0,
                        ) -> ut.ArrayLike1D:
    """
    Compute per-protein average precision (AP) for windowed site prediction.

    The canonical site-localization metric in protease / Post-Translational Modification
    (PTM) prediction: for each protein, residues are ranked by score and AP is computed
    against the known positive sites. ``tolerance`` allows off-by-``k`` positional
    jitter — a ranked residue within ``tolerance`` of an unmatched positive counts as a hit.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    list_scores : list of array-like
        Per-protein per-residue score vectors. ``NaN`` scores are ignored.
    list_positions : list of array-like
        Per-protein 0-based indices of positive sites (empty if none).
    tolerance : int, default=0
        Positional tolerance (in residues) for counting a hit.

    Returns
    -------
    ap : array-like, shape (n_proteins,)
        Per-protein AP. ``np.nan`` for proteins with no positives or no finite
        scores; take ``np.nanmean`` for the dataset-level score.

    See Also
    --------
    * :func:`comp_detection_metrics` for fixed-threshold detection scores.

    Examples
    --------
    .. include:: examples/comp_per_protein_ap.rst
    """
    # Check input
    list_scores, list_positions = _check_list_scores_positions(
        list_scores=list_scores, list_positions=list_positions)
    ut.check_number_range(name="tolerance", val=tolerance, min_val=0, just_int=True)
    # Compute per-protein AP
    return per_protein_ap_(list_scores=list_scores, list_positions=list_positions,
                           tolerance=tolerance)


# Detection metrics at a fixed threshold
def comp_detection_metrics(list_scores: list,
                           list_positions: list,
                           threshold: float = 0.5,
                           tolerance: int = 0,
                           ) -> dict:
    """
    Compute pooled detection metrics at a fixed score threshold.

    Answers "is the true site actually called?" (distinct from ranking): residues
    scoring ``>= threshold`` are positive calls, pooled across proteins into true
    positives, false positives, false negatives, and true negatives (TP/FP/FN/TN), then
    reduced to recall / precision / F1 / Matthews Correlation Coefficient (MCC).
    ``tolerance`` credits a call within ``tolerance`` residues of a true site (each
    site at most once).

    .. versionadded:: 1.1.0

    Parameters
    ----------
    list_scores : list of array-like
        Per-protein per-residue score vectors. ``NaN`` scores are ignored.
    list_positions : list of array-like
        Per-protein 0-based indices of positive sites.
    threshold : float, default=0.5
        Score threshold for a positive call.
    tolerance : int, default=0
        Positional tolerance (in residues) for counting a TP.

    Returns
    -------
    metrics : dict
        Keys ``recall``, ``precision``, ``f1``, ``mcc`` (floats) and ``tp``,
        ``fp``, ``fn``, ``tn`` (ints).

    See Also
    --------
    * :func:`comp_per_protein_ap` for the ranking-based site-localization score.

    Examples
    --------
    .. include:: examples/comp_detection_metrics.rst
    """
    # Check input
    list_scores, list_positions = _check_list_scores_positions(
        list_scores=list_scores, list_positions=list_positions)
    ut.check_number_val(name="threshold", val=threshold, accept_none=False, just_int=False)
    ut.check_number_range(name="tolerance", val=tolerance, min_val=0, just_int=True)
    # Compute detection metrics
    return detection_metrics_(list_scores=list_scores, list_positions=list_positions,
                              threshold=threshold, tolerance=tolerance)


# Bootstrap confidence interval
def comp_bootstrap_ci(values: ut.ArrayLike1D,
                      n_rounds: int = 1000,
                      ci: float = 0.95,
                      seed: Optional[int] = None,
                      ) -> dict:
    """
    Compute a percentile bootstrap Confidence Interval (CI) of the mean.

    Standard small-N uncertainty quantification over a per-protein metric vector
    (e.g. the output of :func:`comp_per_protein_ap`). Resamples with replacement;
    ``NaN`` values are dropped first. Deterministic given ``seed``.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    values : array-like, shape (n_proteins,)
        Per-protein metric values.
    n_rounds : int, default=1000
        Number of bootstrap resamples.
    ci : float, default=0.95
        Central confidence level in ``(0, 1)``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys ``'mean'`` (mean of the finite ``values``),
        ``'ci_low'`` (lower bound of the ``ci`` interval), and ``'ci_high'``
        (upper bound of the ``ci`` interval).

    See Also
    --------
    * :func:`comp_per_protein_ap` for the per-protein metric vector this summarizes.

    Examples
    --------
    .. include:: examples/comp_bootstrap_ci.rst
    """
    # Check input
    values = ut.check_array_like(name="values", val=values, allow_nan=True)
    ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
    ut.check_number_range(name="ci", val=ci, min_val=0.0, max_val=1.0,
                          just_int=False, exclusive_limits=True)
    ut.check_number_range(name="seed", val=seed, min_val=0, accept_none=True, just_int=True)
    # Compute bootstrap CI
    mean, low, high = bootstrap_ci_(values=values, n_rounds=n_rounds, ci=ci, seed=seed)
    return {"mean": mean, "ci_low": low, "ci_high": high}


# Peak-preserving score smoothing
def comp_smooth_scores(scores: ut.ArrayLike1D,
                       method: Literal["triangular", "gaussian"] = "triangular",
                       window: int = 2,
                       sigma: Optional[float] = None,
                       peak_preserving: bool = True,
                       ) -> ut.ArrayLike1D:
    """
    Smooth a per-residue score vector with a NaN-aware, peak-preserving kernel.

    Off-by-one positional jitter is universal in windowed protease / Post-Translational
    Modification (PTM) prediction; smoothing the per-residue score makes nearby high scores
    reinforce a site. The peak-preserving form takes ``max(smoothed, raw)`` so a
    true peak is never attenuated below its original height. Pure-numpy, no SciPy.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    scores : array-like, shape (n_residues,)
        Per-residue score vector. ``NaN`` positions are ignored in the weighted
        average and renormalized over finite neighbours.
    method : str, default='triangular'
        Smoothing kernel: ``'triangular'`` or ``'gaussian'``.
    window : int, default=2
        Half-width of the kernel (covers ``+/- window`` residues).
    sigma : float, optional
        Gaussian standard deviation; defaults to ``window / 2`` when ``None``.
    peak_preserving : bool, default=True
        If ``True``, return ``max(smoothed, raw)`` elementwise.

    Returns
    -------
    smoothed : array-like, shape (n_residues,)
        Smoothed score vector, same length as ``scores``.

    See Also
    --------
    * :func:`plot_rank` for visualizing per-protein score tracks.

    Examples
    --------
    .. include:: examples/comp_smooth_scores.rst
    """
    # Check input
    scores = ut.check_array_like(name="scores", val=scores, allow_nan=True)
    ut.check_str_options(name="method", val=method, list_str_options=["triangular", "gaussian"])
    ut.check_number_range(name="window", val=window, min_val=1, just_int=True)
    ut.check_number_range(name="sigma", val=sigma, min_val=0.0, accept_none=True,
                          just_int=False, exclusive_limits=True)
    ut.check_bool(name="peak_preserving", val=peak_preserving)
    # Compute smoothed scores
    return smooth_scores_(scores=scores, method=method, window=window, sigma=sigma,
                          peak_preserving=peak_preserving)
