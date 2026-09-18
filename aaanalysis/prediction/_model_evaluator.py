"""
This is a script for the frontend of the ModelEvaluator class for rigorous cross-validated
evaluation and paired comparison of prediction models.
"""
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

from ._backend.model_evaluator.model_evaluator_eval import (comp_fold_scores, aggregate_scores,
                                                            compare_models, comp_learning_curve)


# I Helper Functions
def _set_random_state_if_supported(estimator=None, random_state=None, only_if_unset=False):
    """Inject ``random_state`` into an estimator that supports it (for reproducibility)."""
    params = estimator.get_params(deep=False)
    if "random_state" in params and (not only_if_unset or params["random_state"] is None):
        estimator.set_params(random_state=random_state)
    return estimator


def _make_unique_names(names=None):
    """De-duplicate model labels so every row stays uniquely referable.

    A repeated name gets a 1-based suffix, and the suffixed candidate is bumped until it does not
    collide with any original name or an already-assigned label (so inputs like ``["rf", "rf",
    "rf_1"]`` never emit two identical labels).
    """
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    originals = set(names)
    seen, used, out = {}, set(), []
    for name in names:
        if counts[name] == 1:
            label = name
        else:
            seen[name] = seen.get(name, 0) + 1
            k = seen[name]
            label = f"{name}_{k}"
            while label in originals or label in used:
                k += 1
                label = f"{name}_{k}"
            seen[name] = k
        used.add(label)
        out.append(label)
    return out


def _resolve_models(models=None, random_state=None):
    """Resolve ``models`` (registry names, estimator instances, and/or classes) into estimators.

    Returns ``(list_estimators, list_model_names)`` where the names are de-duplicated labels used
    as the ``model`` column of the result tables.
    """
    if not isinstance(models, list):
        models = [models]
    if len(models) == 0:
        raise ValueError("'models' must contain at least one model name or estimator.")
    list_estimators, raw_names = [], []
    for m in models:
        if isinstance(m, str):
            est = ut.get_cv_model_(name=m, random_state=random_state)
            raw_names.append(m)
        elif isinstance(m, type):
            est = _set_random_state_if_supported(m(), random_state, only_if_unset=False)
            raw_names.append(m.__name__)
        else:  # a configured estimator instance: clone + seed only if the user left it unset
            est = _set_random_state_if_supported(clone(m), random_state, only_if_unset=True)
            raw_names.append(type(m).__name__)
        ut.check_mode_class(model_class=type(est))
        list_estimators.append(est)
    return list_estimators, _make_unique_names(raw_names)


def check_metrics(metrics=None):
    """Check that evaluation metrics are supported ModelEvaluator metric names."""
    metrics = ut.check_list_like(name="metrics", val=metrics, accept_str=True, accept_none=False, min_len=1)
    wrong = [m for m in metrics if m not in ut.LIST_METRICS_MODELEVAL]
    if len(wrong) != 0:
        raise ValueError(f"'metrics' ({wrong}) should each be one of: {ut.LIST_METRICS_MODELEVAL}")
    return metrics


def check_n_cv(n_cv=None, labels=None):
    """Check that n_cv is a valid integer not exceeding the smallest class count."""
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    _, counts = np.unique(labels, return_counts=True)
    min_class_count = int(min(counts))
    if n_cv > min_class_count:
        raise ValueError(f"'n_cv' ({n_cv}) should not be greater than the smallest class count ({min_class_count}).")


def check_binary_labels(labels=None):
    """Check that labels are exactly the two classes 0 and 1 (ModelEvaluator scores binary classifiers).

    Requiring ``{0, 1}`` (not just any two classes) keeps every metric consistent on which class is
    positive: the hard-label metrics (precision/recall/f1) use scikit-learn's ``pos_label=1`` and
    ``roc_auc`` treats the greater label as positive, so an arbitrary two-class set (e.g. ``{1, 2}``)
    would make them silently disagree or raise.
    """
    classes = sorted(int(c) for c in np.unique(labels))
    if classes != [0, 1]:
        raise ValueError(f"'labels' should contain exactly the two classes 0 and 1 for "
                         f"'ModelEvaluator', got {classes}.")
    return classes


def check_estimators_proba(list_estimators=None, metrics=None):
    """Require ``predict_proba`` only when a probability metric (e.g. ``roc_auc``) is requested."""
    proba_metrics = [m for m in metrics if m in ut.LIST_METRICS_PRED_PROBA]
    if not proba_metrics:
        return
    missing = sorted({type(e).__name__ for e in list_estimators if not hasattr(e, "predict_proba")})
    if missing:
        raise ValueError(f"Estimators {missing} do not implement 'predict_proba', required by the "
                         f"probability metric(s) {proba_metrics}. Drop them or configure the estimator "
                         f"(e.g. 'SVC(probability=True)').")


def check_train_sizes(train_sizes, labels, n_cv: int) -> Tuple[List[int], Optional[List[float]]]:
    """Check ``train_sizes`` and resolve it into sorted training-subset sizes (and their fractions).

    Sizes are either all fractions in ``(0, 1]`` or all integers >= 2. Absolute counts are used as
    given in every fold, so they must be distinct and fit into the smallest training fold of
    stratified ``n_cv``-fold cross-validation. Fractions are kept as fractions and resolved by the
    backend **within each training fold** (rounded down, raised to at least 2 samples, one per
    class), so ``1.0`` is every fold's complete training set even when the folds differ in size.
    Each curve point is labelled by its size in the smallest training fold; fractions collapsing
    onto the same label are de-duplicated (the smallest of them is kept), so a default grid still
    works on small data. Returns ``(sizes, fracs)``, where ``fracs`` is ``None`` for absolute
    counts and otherwise holds one fraction per label, in the same order.
    """
    train_sizes = ut.check_list_like(name="train_sizes", val=train_sizes, accept_none=False, min_len=2)
    is_number = [isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, (bool, np.bool_))
                 for s in train_sizes]
    if not all(is_number):
        raise ValueError(f"'train_sizes' ({train_sizes}) should be numbers (fractions or sample counts).")
    all_int = all(isinstance(s, (int, np.integer)) for s in train_sizes)
    all_float = all(isinstance(s, (float, np.floating)) for s in train_sizes)
    if not (all_int or all_float):
        raise ValueError(f"'train_sizes' ({train_sizes}) should be either all fractions (float) or "
                         f"all absolute sample counts (int), not a mix.")
    # Smallest training-fold size (fold sizes do not depend on shuffling).
    labels = np.asarray(labels)
    cv = StratifiedKFold(n_splits=n_cv)
    n_train_min = min(len(train_idx) for train_idx, _ in cv.split(np.zeros((len(labels), 1)), labels))
    if all_float:
        wrong = [s for s in train_sizes if not 0 < float(s) <= 1]
        if len(wrong) != 0:
            raise ValueError(f"'train_sizes' ({train_sizes}) should be fractions in (0, 1], got {wrong}.")
        # Label each fraction by its size in the smallest training fold: floored at 2 samples (one
        # per class) and capped at that fold, so a grid stays usable on small data. The fraction
        # itself is carried through, because each fold resolves it against its own training set.
        dict_size_frac = {}
        for frac in sorted(float(s) for s in train_sizes):
            size = min(n_train_min, max(2, int(np.floor(frac * n_train_min))))
            dict_size_frac.setdefault(size, frac)
        sizes = sorted(dict_size_frac)
        if len(sizes) < 2:
            raise ValueError(f"'train_sizes' ({train_sizes}) should be fractions resolving to at least "
                             f"2 distinct sizes, but resolve to {sizes} for the smallest training fold "
                             f"({n_train_min} samples for n_cv={n_cv}).")
        return sizes, [dict_size_frac[size] for size in sizes]
    sizes = [int(s) for s in train_sizes]
    too_small = [s for s in sizes if s < 2]
    too_large = [s for s in sizes if s > n_train_min]
    if len(too_small) != 0:
        raise ValueError(f"'train_sizes' ({train_sizes}) should be sample counts of at least 2 "
                         f"(one per class), got {too_small}.")
    if len(too_large) != 0:
        raise ValueError(f"'train_sizes' ({train_sizes}) should be at most the smallest training "
                         f"fold size ({n_train_min} samples for n_cv={n_cv}), got {too_large}.")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"'train_sizes' ({train_sizes}) should be distinct sample counts, got "
                         f"duplicates in {sizes}.")
    return sorted(sizes), None


def check_is_run(df_scores=None):
    """Check that ModelEvaluator.run has been called before eval."""
    if df_scores is None:
        raise ValueError("'ModelEvaluator' has not been run; call 'ModelEvaluator.run' first.")


# II Main Functions
class ModelEvaluator(Tool):
    """
    ModelEvaluator: rigorous cross-validated evaluation and paired comparison of models (Tool).

    Turns a feature matrix ``X`` and ``labels`` into an honest performance table rather than a
    single optimistic hold-out number. :meth:`run` scores one or more scikit-learn models by
    repeated stratified cross-validation, as a mean and standard deviation over ``n_cv * n_rounds``
    folds with a bootstrap confidence interval. :meth:`eval` compares two or more models on the
    same folds, reporting a signed ``delta`` (e.g. ΔMCC) and a Wilcoxon signed-rank test.
    :meth:`learning_curve` repeats the cross-validation on growing subsets of each training fold,
    showing whether a task is still sampling-limited (the score keeps rising with more data) or has
    saturated (a different representation or model is needed).

    Where :class:`AAPred` deploys a fitted model and :class:`TreeModel` ranks features,
    ``ModelEvaluator`` answers how well a model generalizes, and whether model A is really better
    than model B.

    .. warning::

        **Experimental.** This class is part of the evolving prediction layer; its API (signatures,
        defaults, return objects) may change between minor releases without the usual deprecation
        cycle. Pin a version if you depend on the current behaviour.

    .. versionadded:: 1.1.0

    Notes
    -----
    * The per-fold scores held on ``df_scores_`` are what :meth:`eval` compares, so a comparison
      never re-runs cross-validation.

    See Also
    --------
    * :class:`AAPred` for evaluating and deploying prediction models on raw sequences.
    * :class:`ModelEvaluatorPlot` for visualizing the evaluation and comparison tables.
    * :func:`comp_bootstrap_ci` for the percentile bootstrap confidence interval reused here.
    """

    def __init__(self,
                 *, models: Optional[Union[str, BaseEstimator, List]] = None,
                 list_metrics: Optional[List[str]] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        models : str, estimator, list, or None, default=None
            Models to evaluate, given as registry name strings (e.g. ``"svm"``, ``"rf"``; see
            ``aaanalysis.utils.LIST_PRED_MODELS``) and/or configured scikit-learn estimator
            instances, in any mix. If ``None``, evaluates a single ``"rf"``
            (:class:`RandomForestClassifier`).
            Pass two or more to enable :meth:`eval` (paired comparison). Each model must implement
            ``predict``; ``predict_proba`` is required only for probability metrics (e.g.
            ``roc_auc``).
        list_metrics : list of {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc'} or None, default=None
            Default metrics used by :meth:`run` and :meth:`learning_curve` when their ``metrics``
            argument is ``None``. If ``None``, uses ``["accuracy", "balanced_accuracy", "mcc"]``.
        verbose : bool, default=True
            If ``True``, enables progress output; if ``False``, suppresses it.
        random_state : int or None, default=None
            Seed used for fold shuffling, bootstrap resampling, and supported estimators. If a
            non-negative integer, these stochastic operations are reproducible; if ``None``, they
            are random. ``aaanalysis.options["random_state"]`` overrides this value unless it is
            ``"off"``.

        Examples
        --------
        .. include:: examples/me.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Resolve models into configured estimator instances (cloned before each fit).
        if models is None:
            models = [ut.MODEL_RF]
        list_estimators, list_model_names = _resolve_models(models=models, random_state=random_state)
        if list_metrics is None:
            list_metrics = ["accuracy", "balanced_accuracy", "mcc"]
        else:
            list_metrics = check_metrics(metrics=list_metrics)
        self._verbose = verbose
        self._random_state = random_state
        self._list_estimators = list_estimators
        self._list_model_names = list_model_names
        self._list_metrics = list_metrics
        # Computed state (set by run)
        self._metrics_run = None
        self.df_scores_: Optional[pd.DataFrame] = None
        self.df_eval_: Optional[pd.DataFrame] = None

    def _resolve_seed(self, random_state=None):
        """Per-call ``random_state`` override, falling back to the constructor seed."""
        if random_state is None:
            return self._random_state
        return ut.check_random_state(random_state=random_state)

    def run(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D,
            *, n_cv: int = 5,
            n_rounds: int = 1,
            metrics: Optional[List[str]] = None,
            ci: Optional[float] = 0.95,
            random_state: Optional[int] = None,
            ) -> pd.DataFrame:
        """
        Evaluate every model by repeated stratified cross-validation with bootstrap CIs.

        Runs ``n_rounds`` repeats of stratified ``n_cv``-fold cross-validation (each repeat
        reshuffled with a distinct seed derived from ``random_state``), scoring every model on the
        same folds. The per-fold scores are aggregated per (model, metric) into a mean, a
        population std, a percentile bootstrap confidence interval of the mean, and the fold count.
        The raw per-fold scores are stored on ``df_scores_`` for :meth:`eval`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        labels : array-like, shape (n_samples,)
            Binary class labels for the samples in ``X``.
        n_cv : int, default=5
            Number of stratified cross-validation folds per round (must not exceed the smallest
            class count).
        n_rounds : int, default=1
            Number of cross-validation repeats (multi-seed aggregation). The total number of fold
            scores per (model, metric) is ``n_cv * n_rounds``.
        metrics : list of str, optional
            Performance metrics to compute. Defaults to ``list_metrics`` from the constructor.
        ci : float, optional
            Central confidence level in ``(0, 1)`` for the percentile bootstrap CI of the mean.
            If ``None``, the ``ci_low`` / ``ci_high`` columns are ``NaN``. Default is ``0.95``.
        random_state : int, optional
            Per-call seed overriding the constructor's ``random_state`` for this evaluation.

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_models * n_metrics, 7)
            Long-format evaluation table with columns ``model``, ``metric``, ``score`` (mean over
            folds), ``score_std`` (population std over folds), ``ci_low`` / ``ci_high`` (bootstrap
            CI of the mean, ``NaN`` when ``ci`` is ``None``), and ``n_scores`` (fold count).

        Examples
        --------
        .. include:: examples/me_run.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        check_binary_labels(labels=labels)
        check_n_cv(n_cv=n_cv, labels=labels)
        ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
        metrics = check_metrics(metrics=metrics) if metrics is not None else self._list_metrics
        check_estimators_proba(list_estimators=self._list_estimators, metrics=metrics)
        if ci is not None:
            ut.check_number_range(name="ci", val=ci, min_val=0.0, max_val=1.0,
                                  just_int=False, exclusive_limits=True)
        random_state = self._resolve_seed(random_state=random_state)
        # Evaluate
        df_scores = comp_fold_scores(X, labels, list_estimators=self._list_estimators,
                                     list_model_names=self._list_model_names, metrics=metrics,
                                     n_cv=n_cv, n_rounds=n_rounds, random_state=random_state)
        df_eval = aggregate_scores(df_scores, list_model_names=self._list_model_names,
                                   metrics=metrics, ci=ci, ci_seed=random_state)
        self.df_scores_ = df_scores
        self.df_eval_ = df_eval
        self._metrics_run = metrics
        return df_eval

    def eval(self,
             *, metric: Optional[str] = None,
             ci: Optional[float] = 0.95,
             random_state: Optional[int] = None,
             ) -> pd.DataFrame:
        """
        Compare the evaluated models pairwise on a single metric over the shared folds.

        Uses the per-fold scores from the last :meth:`run` (so no cross-validation is repeated).
        For each model pair the per-fold paired difference ``delta = score_a - score_b`` (same
        fold) gives a signed mean effect, its std, a percentile bootstrap CI, and a two-sided
        Wilcoxon signed-rank ``p_value``. Requires at least two models and a prior :meth:`run`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        metric : str, optional
            Metric to compare on; must be one of the metrics computed by the last :meth:`run`.
            Defaults to ``"mcc"`` when it was computed, otherwise the first metric of that run.
        ci : float, optional
            Central confidence level in ``(0, 1)`` for the bootstrap CI of the paired differences.
            If ``None``, ``ci_low`` / ``ci_high`` are ``NaN``. Default is ``0.95``.
        random_state : int, optional
            Per-call seed overriding the constructor's ``random_state`` for the bootstrap CI.

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_pairs, 8)
            One row per model pair with columns ``model_a``, ``model_b``, ``metric``, ``delta``
            (signed mean of ``score_a - score_b``), ``delta_std``, ``ci_low`` / ``ci_high``
            (bootstrap CI of the paired differences, ``NaN`` when ``ci`` is ``None``), and
            ``p_value`` (two-sided Wilcoxon signed-rank test on the paired differences).

        Examples
        --------
        .. include:: examples/me_eval.rst
        """
        # Check input
        check_is_run(df_scores=self.df_scores_)
        if len(self._list_model_names) < 2:
            raise ValueError(f"'ModelEvaluator.eval' needs at least two models to compare, got "
                             f"{self._list_model_names}. Pass two or more models to the constructor.")
        # Default to mcc when the last run computed it, else its first metric (so eval() never
        # crashes just because run(metrics=...) excluded mcc).
        if metric is None:
            metric = "mcc" if "mcc" in self._metrics_run else self._metrics_run[0]
        ut.check_str_options(name="metric", val=metric, list_str_options=self._metrics_run)
        if ci is not None:
            ut.check_number_range(name="ci", val=ci, min_val=0.0, max_val=1.0,
                                  just_int=False, exclusive_limits=True)
        random_state = self._resolve_seed(random_state=random_state)
        # Compare
        return compare_models(self.df_scores_, list_model_names=self._list_model_names,
                              metric=metric, ci=ci, ci_seed=random_state)

    def learning_curve(self,
                       X: ut.ArrayLike2D,
                       labels: ut.ArrayLike1D,
                       *, train_sizes: Optional[ut.ArrayLike1D] = None,
                       n_cv: int = 5,
                       n_rounds: int = 1,
                       metrics: Optional[Union[str, List[str]]] = None,
                       ci: Optional[float] = 0.95,
                       random_state: Optional[int] = None,
                       ) -> pd.DataFrame:
        """
        Evaluate every model at increasing training sizes to show whether a task is sampling-limited.

        A curve still rising at the largest training size says that more data will help; a curve
        that has flattened says that a different representation or model is needed instead. The
        decision is left to the user.

        Every model is fitted on stratified subsets of each training fold and scored on the full
        test fold, which is never subsampled and never trained on, and the per-fold scores are
        aggregated as in :meth:`run`. The fraction ``1.0`` therefore reproduces the scores of
        :meth:`run` whenever both calls resolve to the same ``random_state``, ``n_cv``,
        ``n_rounds``, and ``metrics``.

        .. versionadded:: 1.2.0

        Parameters
        ----------
        X : array-like of float, shape (n_samples, n_features)
            Finite feature matrix with at least three samples and two features; rows are the
            samples evaluated by every training-size curve.
        labels : array-like of int, shape (n_samples,)
            Binary class labels aligned with ``X``. Values must be exactly 0 and 1; 1 is the
            positive class for ``precision``, ``recall``, ``f1``, and ``roc_auc``.
        train_sizes : array-like of float or int, optional
            Training-subset sizes (at least two), either all fractions in ``(0, 1]`` or all distinct
            absolute sample counts (int >= 2). A fraction is taken of each training fold, so ``1.0``
            is that fold's complete training set and the curve point is labelled by its size in the
            smallest fold; an absolute count is used as given and must fit into the smallest fold.
            If ``None``, uses ``[0.2, 0.4, 0.6, 0.8, 1.0]``, which on small data can collapse to
            fewer, though never fewer than two, distinct sizes.
        n_cv : int, default=5
            Number of stratified cross-validation folds per round. Must be at least 2 and not
            exceed the smallest class count; increasing it changes the train/test split size and
            the number of scores aggregated.
        n_rounds : int, default=1
            Number of cross-validation repeats. Increasing it changes the shuffled splits and
            yields ``n_cv * n_rounds`` fold scores per (model, training size, metric).
        metrics : {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc'} or list of str, optional
            Performance metric(s) to compute. If ``None``, uses ``list_metrics`` from the
            constructor; ``roc_auc`` requires every model to implement ``predict_proba``.
        ci : float or None, default=0.95
            Central confidence level in ``(0, 1)`` for the percentile bootstrap CI of the mean.
            If ``None``, skips bootstrap CIs and sets the ``ci_low`` / ``ci_high`` columns to
            ``NaN``.
        random_state : int or None, default=None
            Per-call seed overriding the constructor's ``random_state`` for the folds, the training
            subsets, and the bootstrap CI. A non-negative integer makes those operations and
            estimators that support ``random_state`` reproducible. If ``None``, the constructor's
            ``random_state`` is used (and stochastic processes are truly random when that is
            ``None`` as well). ``aaanalysis.options["random_state"]`` overrides both unless it is
            ``"off"``.

        Returns
        -------
        df_curve : pd.DataFrame, shape (n_models * n_train_sizes * n_metrics, 8)
            Long-format learning-curve table with columns ``model``, ``train_size`` (number of
            training samples, ascending), ``metric``, ``score`` (mean over folds), ``score_std``
            (population std over folds), ``ci_low`` / ``ci_high`` (bootstrap CI of the mean,
            ``NaN`` when ``ci`` is ``None``), and ``n_scores`` (fold count).

        Raises
        ------
        ValueError
            If ``X`` or ``labels`` are invalid or mismatched, ``labels`` are not exactly the two
            classes 0 and 1, ``n_cv`` exceeds the smallest class count, ``train_sizes`` mix
            fractions and counts, resolve to fewer than two distinct sizes, or exceed the smallest
            training fold, a metric is unknown, a probability metric is requested for a model
            without ``predict_proba``, or a numeric parameter is out of range.
        RuntimeError
            If an internally constructed training subset overlaps its held-out test fold.

        Notes
        -----
        * Unlike :meth:`run`, this method does not change ``df_scores_`` or ``df_eval_``, so a
          subsequent :meth:`eval` still compares the models of the last :meth:`run`.

        See Also
        --------
        * :meth:`ModelEvaluatorPlot.learning_curve` for plotting the curve with its CI band.

        Examples
        --------
        .. include:: examples/me_learning_curve.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        check_binary_labels(labels=labels)
        check_n_cv(n_cv=n_cv, labels=labels)
        if train_sizes is None:
            train_sizes = ut.LIST_TRAIN_SIZES_MODELEVAL
        train_sizes, train_fracs = check_train_sizes(train_sizes=train_sizes, labels=labels, n_cv=n_cv)
        ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
        metrics = check_metrics(metrics=metrics) if metrics is not None else self._list_metrics
        check_estimators_proba(list_estimators=self._list_estimators, metrics=metrics)
        if ci is not None:
            ut.check_number_range(name="ci", val=ci, min_val=0.0, max_val=1.0,
                                  just_int=False, exclusive_limits=True)
        random_state = self._resolve_seed(random_state=random_state)
        # Evaluate
        df_curve = comp_learning_curve(X, labels, list_estimators=self._list_estimators,
                                       list_model_names=self._list_model_names, metrics=metrics,
                                       train_sizes=train_sizes, train_fracs=train_fracs,
                                       n_cv=n_cv, n_rounds=n_rounds, ci=ci,
                                       random_state=random_state)
        return df_curve
