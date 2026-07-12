"""
This is a script for the frontend of the AAPred class for evaluating and deploying prediction models.
"""
from typing import Optional, Dict, List, Tuple, Type, Union
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper

from ._backend.aa_pred.aa_pred_fit import fit_models, predict_proba_models, predict_proba_oof
from ._backend.aa_pred.aa_pred_eval import eval_models


# I Helper Functions
def _set_random_state_if_supported(estimator=None, random_state=None, only_if_unset=False):
    """Inject ``random_state`` into an estimator that supports it (for reproducibility).

    ``only_if_unset`` protects an explicit seed the user already set on a passed instance;
    a value is written only when the estimator's ``random_state`` is currently ``None``.
    """
    params = estimator.get_params(deep=False)
    if "random_state" in params and (not only_if_unset or params["random_state"] is None):
        estimator.set_params(random_state=random_state)
    return estimator


def check_metrics(metrics=None):
    """Check that evaluation metrics are valid scikit-learn scorer names."""
    metrics = ut.check_list_like(name="metrics", val=metrics, accept_str=True, accept_none=False, min_len=1)
    wrong = [m for m in metrics if m not in ut.LIST_METRICS_PRED]
    if len(wrong) != 0:
        raise ValueError(f"'metrics' ({wrong}) should each be one of: {ut.LIST_METRICS_PRED}")
    return metrics


def check_n_cv(n_cv=None, labels=None):
    """Check that n_cv is a valid integer not exceeding the smallest class count."""
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    _, counts = np.unique(labels, return_counts=True)
    min_class_count = int(min(counts))
    if n_cv > min_class_count:
        raise ValueError(f"'n_cv' ({n_cv}) should not be greater than the smallest class count ({min_class_count}).")


def check_cv(cv=None):
    """Check that a custom cross-validation splitter exposes a callable ``split`` method.

    Unlike the integer ``n_cv`` path, a splitter is trusted to define its own folds, so it is
    not capped at the smallest class count — this is what enables ``LeaveOneOut`` on small,
    imbalanced sets.
    """
    if not callable(getattr(cv, "split", None)):
        raise ValueError(f"'cv' ({cv}) should be a scikit-learn cross-validation splitter "
                         f"exposing a 'split' method (e.g. 'LeaveOneOut()', 'StratifiedKFold(...)').")


def check_is_fitted(list_models=None):
    """Check that AAPred.fit has been called."""
    if list_models is None:
        raise ValueError("'AAPred' is not fitted; call 'AAPred.fit' first.")


def check_binary_labels(labels=None):
    """Check that labels define exactly two classes (AAPred is a binary predictor)."""
    classes = list(np.unique(labels))
    if len(classes) != 2:
        raise ValueError(f"'labels' should define exactly two classes for 'AAPred', got {classes}.")
    return classes


def check_featurizer(df_feat=None):
    """Check that a feature definition is bound so raw sequences can be featurized."""
    if df_feat is None:
        raise ValueError("'AAPred' has no bound 'df_feat'; pass 'df_feat=...' to the constructor to "
                         "enable sequence-level prediction (predict at level 'sequence'/'domain'/'window').")


def featurize_seq(df_feat=None, df_scales=None, df_seq=None, list_parts=None, **parts_kwargs):
    """Featurize a ``df_seq`` into the CPP feature matrix ``X`` bound to the model.

    Uses the single-call ``feature_matrix(df_seq=, df_parts_kws=)`` path (which builds
    ``df_parts`` internally) instead of the manual get_df_parts + feature_matrix pair.
    """
    from aaanalysis.feature_engineering import SequenceFeature
    sf = SequenceFeature()
    df_parts_kws = dict(parts_kwargs)
    if list_parts is not None:
        df_parts_kws["list_parts"] = list_parts
    X = sf.feature_matrix(features=df_feat, df_seq=df_seq,
                          df_parts_kws=df_parts_kws or None, df_scales=df_scales)
    return np.asarray(X)


def check_baseline(baseline=None):
    """Resolve the ``baseline`` selector into an ordered, de-duplicated list of kinds (or None).

    ``None``/``False`` -> no baseline; ``True`` -> the scale-composition default; a str or list
    of strs -> those kinds (each validated against ``LIST_BASELINE_KINDS``).
    """
    if baseline is None or baseline is False:
        return None
    if baseline is True:
        return [ut.STR_BASELINE_SCALE]
    if isinstance(baseline, str):
        baseline = [baseline]
    baseline = ut.check_list_like(name="baseline", val=baseline, accept_str=True, min_len=1)
    wrong = [b for b in baseline if b not in ut.LIST_BASELINE_KINDS]
    if len(wrong) != 0:
        raise ValueError(f"'baseline' ({wrong}) should each be one of: {ut.LIST_BASELINE_KINDS}")
    return list(dict.fromkeys(baseline))


def build_baseline_matrices(df_seq=None, list_kinds=None, df_scales=None, list_parts=None):
    """Build the ``{kind: X}`` baseline feature matrices from raw sequences (SequenceFeature).

    Each baseline is a non-positional, fixed-length sequence descriptor row-aligned with
    ``df_seq``; the caller guarantees ``df_seq`` matches the ``X`` / ``labels`` samples.
    A featurizer yields an all-``NaN`` row for a sequence with no scored/canonical residue in
    the selected span; such a matrix cannot be cross-validated, so raise a clear error naming
    the offending baseline rather than letting it reach ``cross_val_score``.
    """
    from aaanalysis.feature_engineering import SequenceFeature
    sf = SequenceFeature()
    dict_X_baseline = {}
    for kind in list_kinds:
        if kind == ut.STR_BASELINE_SCALE:
            X_baseline = sf.scale_composition(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts)
        elif kind == ut.STR_BASELINE_AAC:
            X_baseline = sf.aa_composition(df_seq=df_seq, list_parts=list_parts)
        else:  # ut.STR_BASELINE_DPC
            X_baseline = sf.dipeptide_composition(df_seq=df_seq, list_parts=list_parts)
        X_baseline = np.asarray(X_baseline)
        if np.isnan(X_baseline).any():
            n_bad = int(np.isnan(X_baseline).any(axis=1).sum())
            raise ValueError(f"'baseline' ({kind}) produced {n_bad} all-invalid row(s) "
                             f"(sequence with no scored residue in the selected span); "
                             f"drop or fix those 'df_seq' entries before evaluating.")
        dict_X_baseline[kind] = X_baseline
    return dict_X_baseline


# II Main Functions
class AAPred(Wrapper):
    """
    AAPred: evaluate and deploy sequence-based prediction models (Wrapper) [Breimann25]_.

    A thin, opinionated wrapper that closes the gap left by feature engineering: given a
    feature matrix ``X`` and ``labels``, it **evaluates** one or more scikit-learn model
    classes across metrics by cross-validation and an optional held-out set (:meth:`eval`),
    and **deploys** them by fitting on all data and exposing prediction scores
    (:meth:`fit` / :meth:`predict` / :meth:`eval`).

    Unlike :class:`CPPGrid`, which optimizes the *feature space* and scores configurations
    by feature separation, ``AAPred`` takes a *fixed* feature set and trains models that are
    kept for deployment. It intentionally does **not** perform hyperparameter optimization —
    pass configured estimators and it evaluates and deploys them.

    .. versionadded:: 1.1.0

    Notes
    -----
    * All fitted-state attributes carry a trailing underscore and are set by :meth:`fit`.

    See Also
    --------
    * :class:`TreeModel` for tree-ensemble Monte-Carlo feature importance and selection.
    * :class:`AAPredPlot` for visualizing evaluation and prediction results.
    * :class:`CPPGrid` for optimizing the CPP feature space (upstream of this class).
    """

    def __init__(self,
                 models: Optional[Union[str, "BaseEstimator", List]] = None,
                 list_model_classes: Optional[List[Type[Union[ClassifierMixin, BaseEstimator]]]] = None,
                 list_model_kwargs: Optional[List[Dict]] = None,
                 list_metrics: Optional[List[str]] = None,
                 df_feat: Optional[pd.DataFrame] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        models : str, estimator, or list, optional
            The models to evaluate and deploy, given as registry name strings (e.g.
            ``"svm"``, ``"rf"``; see ``aaanalysis.utils.LIST_PRED_MODELS``) and/or configured
            scikit-learn estimator instances, in any mix. This is the recommended way to
            select models; it is mutually exclusive with ``list_model_classes``. Each model
            must implement ``predict_proba``.
        list_model_classes : list of Type[ClassifierMixin or BaseEstimator], default=[RandomForestClassifier]
            Model classes to evaluate and deploy (legacy alternative to ``models``). Each
            must implement ``predict_proba``.
        list_model_kwargs : list of dict, optional
            Keyword arguments for each model in ``list_model_classes`` (same length).
        list_metrics : list of str, default=["accuracy", "balanced_accuracy", "f1", "roc_auc"]
            Default performance metrics used by :meth:`eval` when ``metrics`` is not given.
            Each should be one of ``accuracy``, ``balanced_accuracy``, ``precision``,
            ``recall``, ``f1``, ``roc_auc``.
        df_feat : pd.DataFrame, shape (n_features, n_feature_info), optional
            CPP feature DataFrame (with a ``feature`` column) bound to the model. When given, the
            feature matrix ``X`` is computed internally from a ``df_seq`` by the sequence-level
            :meth:`predict` method (``level='sequence'``/``'domain'``/``'window'``).
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            Amino acid scales used for internal featurization. Defaults to the bundled AAontology
            scales when ``None``.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of
            stochastic processes are consistent, enabling reproducibility. If ``None``,
            stochastic processes will be truly random.

        Examples
        --------
        .. include:: examples/aap.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Resolve models into configured estimator INSTANCES (``_list_estimators``), which
        # fit/eval clone before use. `models` (registry name strings and/or configured
        # sklearn estimator instances/classes) is the primary API; storing the instance and
        # cloning it (rather than round-tripping through get_params + type) is what makes
        # meta-ensembles (voting/stacking) and **kwargs estimators (xgboost) work and keeps
        # a passed instance's own configuration + random_state intact.
        if models is not None:
            if list_model_classes is not None or list_model_kwargs is not None:
                raise ValueError("Pass either 'models' or 'list_model_classes'/'list_model_kwargs', not both.")
            if not isinstance(models, list):
                models = [models]
            if len(models) == 0:
                raise ValueError("'models' must contain at least one model name or estimator.")
            list_estimators = []
            for m in models:
                if isinstance(m, str):
                    est = ut.get_cv_model_(name=m, random_state=random_state)
                elif isinstance(m, type):
                    est = _set_random_state_if_supported(m(), random_state, only_if_unset=False)
                else:  # a configured estimator instance: clone + seed only if the user left it unset
                    est = _set_random_state_if_supported(clone(m), random_state, only_if_unset=True)
                ut.check_mode_class(model_class=type(est))  # ensure predict_proba
                list_estimators.append(est)
            list_model_classes = [type(e) for e in list_estimators]
            _list_model_kwargs = [{} for _ in list_estimators]  # introspection placeholder
        else:
            # Legacy path: class + kwargs, validated, then instantiated into estimators.
            if list_model_classes is None:
                list_model_classes = [RandomForestClassifier]
            elif not isinstance(list_model_classes, list):
                list_model_classes = [list_model_classes]
            list_model_classes = ut.check_list_like(name="list_model_classes", val=list_model_classes,
                                                    accept_none=False, min_len=1)
            list_model_kwargs = ut.check_list_like(name="list_model_kwargs", val=list_model_kwargs, accept_none=True)
            if list_model_kwargs is None:
                list_model_kwargs = [{} for _ in list_model_classes]
            ut.check_match_list_model_classes_kwargs(list_model_classes=list_model_classes,
                                                     list_model_kwargs=list_model_kwargs)
            _list_model_kwargs = []
            for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs):
                ut.check_mode_class(model_class=model_class)
                model_kwargs = ut.check_model_kwargs(model_class=model_class, model_kwargs=model_kwargs,
                                                     method_to_check="predict_proba", random_state=random_state)
                _list_model_kwargs.append(model_kwargs)
            list_estimators = [cls(**kw) for cls, kw in zip(list_model_classes, _list_model_kwargs)]
        # Metric parameters
        if list_metrics is None:
            list_metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
        list_metrics = check_metrics(metrics=list_metrics)
        # Featurizer parameters
        if df_feat is not None:
            df_feat = ut.check_df_feat(df_feat=df_feat)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._list_model_classes = list_model_classes
        self._list_model_kwargs = _list_model_kwargs
        self._list_estimators = list_estimators
        self._list_metrics = list_metrics
        self._df_feat = df_feat
        self._df_scales = df_scales
        # Output attributes (set during fitting)
        self.list_models_: Optional[List[Union[ClassifierMixin, BaseEstimator]]] = None
        self.label_pos_: Optional[int] = None
        self.label_neg_: Optional[int] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D,
            label_pos: int = 1,
            optimize_hyperparams: bool = False,
            param_grids: Optional[Union[Dict, List[Dict]]] = None,
            n_cv: int = 5,
            ) -> "AAPred":
        """
        Fit every model on the full dataset for deployment.

        Each model class from the constructor is instantiated and fit on all of ``X`` / ``labels``;
        the fitted estimators are kept in ``list_models_`` and reused by :meth:`predict` and
        :meth:`eval`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.
        labels : array-like, shape (n_samples,)
            Class labels for samples in ``X`` (typically ``1`` for the positive class and
            ``0`` for the negative class).
        label_pos : int, default=1
            Label of the positive class whose probability :meth:`predict` scores.
        optimize_hyperparams : bool, default=False
            If ``True``, each model is tuned by ``GridSearchCV`` (``n_cv`` folds) over its
            ``param_grids`` entry, or a built-in default grid when none is given; the best
            estimator is kept. If ``False``, models are fit with their given parameters.
        param_grids : dict or list of dict, optional
            Hyperparameter grid(s) for the optimization. A single dict is applied to every
            model; a list must have one grid per model. Used only when
            ``optimize_hyperparams=True``.
        n_cv : int, default=5
            Number of stratified cross-validation folds used by the hyperparameter search.

        Returns
        -------
        AAPred
            The fitted ``AAPred`` instance (``self``).

        Examples
        --------
        .. include:: examples/aap_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        classes = check_binary_labels(labels=labels)
        ut.check_number_val(name="label_pos", val=label_pos, just_int=True)
        if label_pos not in classes:
            raise ValueError(f"'label_pos' ({label_pos}) should be one of the labels: {classes}")
        ut.check_bool(name="optimize_hyperparams", val=optimize_hyperparams)
        # A single dict applies to every model; a list must match the number of models.
        if isinstance(param_grids, dict):
            list_param_grids = [param_grids for _ in self._list_model_classes]
        elif isinstance(param_grids, list):
            if len(param_grids) != len(self._list_model_classes):
                raise ValueError(f"'param_grids' list length ({len(param_grids)}) should match "
                                 f"the number of models ({len(self._list_model_classes)}).")
            list_param_grids = param_grids
        else:
            list_param_grids = None
        if optimize_hyperparams:
            check_n_cv(n_cv=n_cv, labels=labels)
        # Fit models (optionally tuning hyperparameters via GridSearchCV)
        self.list_models_ = fit_models(X=X, labels=labels,
                                       list_estimators=self._list_estimators,
                                       list_param_grids=list_param_grids,
                                       optimize_hyperparams=optimize_hyperparams,
                                       n_cv=n_cv, random_state=self._random_state)
        self.label_pos_ = label_pos
        self.label_neg_ = int([c for c in classes if c != label_pos][0])
        return self

    def eval(self,
             X: ut.ArrayLike2D,
             labels: ut.ArrayLike1D,
             X_holdout: Optional[ut.ArrayLike2D] = None,
             labels_holdout: Optional[ut.ArrayLike1D] = None,
             metrics: Optional[List[str]] = None,
             n_cv: int = 5,
             cv: Optional[BaseCrossValidator] = None,
             df_seq: Optional[pd.DataFrame] = None,
             baseline: Optional[Union[bool, str, List[str]]] = None,
             list_parts: Optional[Union[str, List[str]]] = None,
             ) -> pd.DataFrame:
        """
        Evaluate every model across metrics by cross-validation and an optional held-out set.

        Up to three evaluation principles are reported: ``cv`` (stratified k-fold
        cross-validation on ``X``, scored per fold then averaged) and, when a held-out set is
        provided, ``holdout`` (models fit on ``X`` and scored on ``X_holdout``). The result is a
        long-format table with one row per (model, metric, principle).

        Pass ``cv`` to cross-validate with an **arbitrary scikit-learn splitter** (e.g.
        ``LeaveOneOut()``) instead of the integer ``n_cv`` folds. The splitter is not capped at
        the smallest class count, and its rows are scored by the ``cv_pooled`` principle: every
        held-out prediction is pooled and each metric is applied **once** on that pooled vector
        (reproducing ``metric(labels, cross_val_predict(estimator, X, labels, cv=cv))``), rather
        than averaging a per-fold score. This is the correct principle for ``LeaveOneOut`` on a
        small, imbalanced set, where a single-sample test fold makes per-fold averaging degenerate.

        Set ``baseline`` to compare the bound features against simple, non-positional
        **baseline featurizers** (amino-acid / dipeptide / scale composition) built internally
        from ``df_seq``: each baseline is cross-validated with the same models and folds and its
        rows are appended, so the whole "CPP vs baseline" comparison comes from one call. This
        quantifies how much the positional CPP features add over a plain composition encoding.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix used for cross-validation (and, for the holdout principle, training).
        labels : array-like, shape (n_samples,)
            Class labels for samples in ``X``.
        X_holdout : array-like, shape (n_holdout, n_features), optional
            Held-out feature matrix. If given, the ``holdout`` principle is added.
        labels_holdout : array-like, shape (n_holdout,), optional
            Class labels for ``X_holdout``. Required if ``X_holdout`` is given.
        metrics : list of str, optional
            Performance metrics to compute. Defaults to ``list_metrics`` from the constructor.
        n_cv : int, default=5
            Number of stratified cross-validation folds (must not exceed the smallest class
            count). Ignored when ``cv`` is given.
        cv : cross-validation splitter, optional
            A scikit-learn cross-validation splitter exposing a ``split`` method (e.g.
            ``LeaveOneOut()``, ``StratifiedKFold(n_splits=10)``). When given, it replaces the
            integer ``n_cv`` folds and the cross-validation rows are scored by the pooled
            ``cv_pooled`` principle (each metric applied once on the pooled out-of-fold
            predictions, so ``score_std`` is ``NaN``). Not capped at the smallest class count.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence
            information (the same input accepted by :meth:`SequenceFeature.get_df_parts`). Must be
            **row-aligned with** ``X`` / ``labels`` (row *i* is the same sample) and built with the
            **same part geometry** as ``X`` for a fair comparison — the alignment cannot be verified
            from the opaque ``X``, only ``len(df_seq) == len(labels)`` is checked. Required when
            ``baseline`` is set, ignored otherwise.
        baseline : bool, str, or list of str, optional
            Baseline featurizer(s) to cross-validate alongside the bound features. ``True`` uses
            the scale-composition baseline (``'scale'``); a str or list selects among
            ``'scale'`` (:meth:`SequenceFeature.scale_composition`), ``'aac'``
            (:meth:`SequenceFeature.aa_composition`), and ``'dpc'``
            (:meth:`SequenceFeature.dipeptide_composition`). ``None`` (default) adds no baseline.
            Baselines average over ``list_parts`` with :meth:`SequenceFeature.get_df_parts`' default
            JMD lengths; match the geometry used to build ``X`` so the comparison is not confounded.
        list_parts : str or list of str, optional
            Sequence parts averaged into each baseline (passed to the featurizers). Defaults to
            the whole ``tmd_jmd`` span. Used only when ``baseline`` is set.

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_rows, 5) — or (n_rows, 6) in baseline mode
            Long-format evaluation table with columns ``model``, ``metric``, ``principle``,
            ``score``, and ``score_std``. The ``principle`` is ``cv`` for the default per-fold
            cross-validation, ``cv_pooled`` when a ``cv`` splitter is passed, and ``holdout`` for
            the held-out set; ``score_std`` is ``NaN`` for the ``holdout`` and ``cv_pooled``
            principles (each is a single estimate). When ``baseline`` is given, a leading
            ``features`` column is added (``'cpp'`` for the bound-feature rows, the baseline kind
            for each baseline's cross-validation rows); with ``baseline=None`` the table is
            unchanged (5 columns).

        Examples
        --------
        .. include:: examples/aap_eval.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        check_binary_labels(labels=labels)
        metrics = check_metrics(metrics=metrics) if metrics is not None else self._list_metrics
        # A custom splitter defines its own folds (pooled scoring), so it bypasses the
        # smallest-class-count cap that the integer n_cv path enforces.
        if cv is not None:
            check_cv(cv=cv)
        else:
            check_n_cv(n_cv=n_cv, labels=labels)
        list_models = None
        if X_holdout is not None:
            X_holdout = ut.check_X(X=X_holdout, min_n_samples=1)
            labels_holdout = ut.check_labels(labels=labels_holdout)
            ut.check_match_X_labels(X=X_holdout, labels=labels_holdout)
            if X_holdout.shape[1] != X.shape[1]:
                raise ValueError(f"'X_holdout' n_features ({X_holdout.shape[1]}) should match "
                                 f"'X' n_features ({X.shape[1]}).")
            list_models = fit_models(X=X, labels=labels, list_estimators=self._list_estimators)
        elif labels_holdout is not None:
            raise ValueError("'labels_holdout' was given without 'X_holdout'.")
        # Resolve the optional baseline-featurizer comparison (built internally from df_seq)
        list_kinds = check_baseline(baseline=baseline)
        dict_X_baseline = None
        if list_kinds is not None:
            ut.check_df_seq(df_seq=df_seq)
            if len(df_seq) != len(labels):
                raise ValueError(f"'df_seq' n_samples ({len(df_seq)}) should match "
                                 f"'labels' n_samples ({len(labels)}).")
            dict_X_baseline = build_baseline_matrices(df_seq=df_seq, list_kinds=list_kinds,
                                                      df_scales=self._df_scales, list_parts=list_parts)
        # Evaluate
        df_eval = eval_models(X=X, labels=labels,
                              list_estimators=self._list_estimators,
                              list_models=list_models, metrics=metrics, n_cv=n_cv,
                              random_state=self._random_state,
                              X_holdout=X_holdout, labels_holdout=labels_holdout,
                              dict_X_baseline=dict_X_baseline, cv=cv)
        return df_eval

    def predict_oof(self,
                    X: ut.ArrayLike2D,
                    labels: ut.ArrayLike1D,
                    label_pos: int = 1,
                    n_cv: int = 5,
                    ) -> pd.DataFrame:
        """
        Score the training set with cross-validated out-of-fold per-sample probabilities.

        Where :meth:`predict` deploys models fit on all data to score *new* proteins, and
        :meth:`eval` reports *aggregate* cross-validated metrics, ``predict_oof`` returns the
        *per-sample* out-of-fold score for the **training** data: every sample is scored by models
        fit on the folds that exclude it (stratified k-fold cross-validation), so the scores are
        honest and free of the optimistic in-sample bias that scoring the training proteins with
        :meth:`predict` would incur. Each configured model is cross-validated independently and the
        per-model out-of-fold scores are averaged, matching the ``score`` / ``score_std`` shape of
        :meth:`predict` (mean over the ensemble, std across models).

        Like :meth:`eval`, this cross-validates the models given at construction and does **not**
        require a prior :meth:`fit`; it never touches the deployment models in ``list_models_``.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.
        labels : array-like, shape (n_samples,)
            Class labels for samples in ``X`` (typically ``1`` for the positive class and
            ``0`` for the negative class).
        label_pos : int, default=1
            Label of the positive class whose out-of-fold probability is scored.
        n_cv : int, default=5
            Number of stratified cross-validation folds (must not exceed the smallest class count).

        Returns
        -------
        df_pred : pd.DataFrame, shape (n_samples, 2)
            Per-sample out-of-fold predictions, row-aligned with ``X``, with columns ``score``
            (mean positive-class probability over the model ensemble) and ``score_std`` (std across
            models; ``0`` for a single model).

        See Also
        --------
        * :meth:`AAPred.predict` for scoring new proteins with the fitted deployment ensemble.
        * :meth:`AAPred.eval` for aggregate cross-validated model metrics (``df_eval``).

        Examples
        --------
        .. include:: examples/aap_predict_oof.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        classes = check_binary_labels(labels=labels)
        ut.check_number_val(name="label_pos", val=label_pos, just_int=True)
        if label_pos not in classes:
            raise ValueError(f"'label_pos' ({label_pos}) should be one of the labels: {classes}")
        check_n_cv(n_cv=n_cv, labels=labels)
        # Out-of-fold scoring: clone the constructor's estimators and cross-validate (no prior fit)
        pred, pred_std = predict_proba_oof(X=X, labels=labels,
                                           list_estimators=self._list_estimators,
                                           n_cv=n_cv, random_state=self._random_state,
                                           label_pos=label_pos)
        df_pred = pd.DataFrame({ut.COL_SCORE: pred, ut.COL_SCORE_STD: pred_std})
        return df_pred

    def predict(self,
                df_seq: pd.DataFrame,
                level: str = "sequence",
                threshold: Optional[Union[int, float]] = None,
                list_parts: Optional[List[str]] = None,
                window: int = 3,
                tmd_len: Optional[int] = None,
                step: int = 1,
                jmd_n_len: int = 10,
                jmd_c_len: int = 10,
                ) -> pd.DataFrame:
        """
        Predict from raw sequences at a chosen level: whole protein, domain, or residue window.

        One predictor for all three granularities, selected with ``level``:

        * ``'sequence'`` — one score per protein (sequence level).
        * ``'domain'`` — a boundary-sensitivity scan: the TMD boundaries are shifted by every
          offset in ``[-window, +window]`` and each shifted definition is featurized and scored.
        * ``'window'`` — a per-residue profile: a length-``tmd_len`` window is slid along each
          sequence (spaced by ``step``) and scored at every valid position.

        Each level featurizes with the bound ``df_feat`` and averages the fitted-model ensemble.
        When ``threshold`` is given, a ``predicted_label`` column is added (score at or above the
        threshold -> ``label_pos`` from :meth:`fit`, else the negative class); with ``threshold=None``
        (default) only the scores are returned.

        .. note::
           Requires a ``df_feat`` bound at construction and a prior :meth:`fit`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_proteins, n_seq_info)
            DataFrame containing an ``entry`` column of unique protein identifiers and a
            ``sequence`` column; ``level='domain'`` additionally needs ``tmd_start`` / ``tmd_stop``.
        level : str, default="sequence"
            Prediction granularity: ``'sequence'``, ``'domain'``, or ``'window'``.
        threshold : int or float, optional
            If given (in ``[0, 1]``), add a ``predicted_label`` column (score ``>= threshold`` ->
            ``label_pos``, else the negative class). ``None`` (default) returns scores only.
        list_parts : list of str, optional
            Sequence parts to build for featurization. Defaults to the standard CPP parts.
        window : int, default=3
            (``level='domain'`` only) Maximum absolute boundary shift, in residues, scanned per side.
        tmd_len : int, optional
            (``level='window'`` only, required there) Length in residues of the window anchored at
            each position.
        step : int, default=1
            (``level='window'`` only) Spacing between consecutive anchor positions.
        jmd_n_len : int, default=10
            (``level='window'`` only) N-terminal flank required around each window.
        jmd_c_len : int, default=10
            (``level='window'`` only) C-terminal flank required around each window.

        Returns
        -------
        df_pred : pd.DataFrame
            Long-format predictions; columns depend on ``level``: ``entry`` / ``score`` /
            ``score_std`` (``'sequence'``), ``entry`` / ``offset`` / ``score`` / ``is_best`` (``'domain'``),
            ``entry`` / ``position`` / ``score`` / ``score_std`` (``'window'``). Plus
            ``predicted_label`` when ``threshold`` is given.

        See Also
        --------
        * :meth:`AAPred.eval` for cross-validated model evaluation (``df_eval``).

        Examples
        --------
        .. include:: examples/aap_predict.rst
        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        check_featurizer(df_feat=self._df_feat)
        ut.check_df_seq(df_seq=df_seq)
        if level not in ("sequence", "domain", "window"):
            raise ValueError(f"'level' ('{level}') must be 'sequence', 'domain', or 'window'.")
        if threshold is not None:
            ut.check_number_range(name="threshold", val=threshold, min_val=0, max_val=1, just_int=False)
        # Dispatch to the level-specific predictor
        if level == "sequence":
            df_pred = self._predict_seq(df_seq=df_seq, list_parts=list_parts)
        elif level == "domain":
            df_pred = self._predict_domain(df_seq=df_seq, window=window, list_parts=list_parts)
        else:
            if tmd_len is None:
                raise ValueError("'tmd_len' is required for level='window'.")
            df_pred = self._predict_window(df_seq=df_seq, tmd_len=tmd_len, step=step,
                                           jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=list_parts)
        # Optional class labels from the score
        if threshold is not None:
            df_pred[ut.COL_PRED_LABEL] = np.where(df_pred[ut.COL_SCORE] >= threshold,
                                                  self.label_pos_, self.label_neg_)
        return df_pred

    def _predict_X(self, X):
        """Averaged positive-class score for a feature matrix (fitted-model ensemble)."""
        return predict_proba_models(X=X, list_models=self.list_models_, label_pos=self.label_pos_)

    def _predict_seq(self,
                    df_seq: pd.DataFrame,
                    list_parts: Optional[List[str]] = None,
                    ) -> pd.DataFrame:
        """
        Predict one score per protein (sequence level) from raw sequences.

        The feature matrix is computed internally from ``df_seq`` using the bound ``df_feat``, so
        the caller does not build ``X`` by hand.

        .. note::
           Requires a ``df_feat`` bound at construction and a prior :meth:`fit`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_proteins, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and the
            boundary information consumed by :meth:`SequenceFeature.get_df_parts` (e.g.
            ``tmd_start`` / ``tmd_stop``).
        list_parts : list of str, optional
            Sequence parts to build for featurization. Defaults to the standard CPP parts.

        Returns
        -------
        df_pred : pd.DataFrame, shape (n_proteins, 3)
            One row per protein with columns ``entry``, ``score`` (positive-class score), and
            ``score_std`` (std across models).

        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        check_featurizer(df_feat=self._df_feat)
        ut.check_df_seq(df_seq=df_seq)
        # Featurize + predict
        X = featurize_seq(df_feat=self._df_feat, df_scales=self._df_scales, df_seq=df_seq, list_parts=list_parts)
        pred, pred_std = self._predict_X(X)
        df_pred = pd.DataFrame({ut.COL_ENTRY: df_seq[ut.COL_ENTRY].to_numpy(),
                                ut.COL_SCORE: pred, ut.COL_SCORE_STD: pred_std})
        return df_pred

    def _predict_domain(self,
                       df_seq: pd.DataFrame,
                       window: int = 3,
                       list_parts: Optional[List[str]] = None,
                       ) -> pd.DataFrame:
        """
        Predict a domain score with a boundary-sensitivity scan.

        The domain boundaries (``tmd_start`` / ``tmd_stop`` in ``df_seq``, user-adjustable) are
        shifted by every offset in ``[-window, +window]``; each shifted definition is featurized
        and scored, so the caller sees how the score depends on the exact boundary and which
        definition scores highest.

        .. note::
           Requires a ``df_feat`` bound at construction and a prior :meth:`fit`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_proteins, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers, plus
            ``sequence``, ``tmd_start`` and ``tmd_stop`` columns.
        window : int, default=3
            Maximum absolute boundary shift (in residues) scanned on each side.
        list_parts : list of str, optional
            Sequence parts to build for featurization. Defaults to the standard CPP parts.

        Returns
        -------
        df_domain : pd.DataFrame, shape (n_rows, 4)
            Long-format table with columns ``entry``, ``offset`` (boundary shift), ``score``, and
            ``is_best`` (``True`` for the highest-scoring offset per protein). Offsets whose shifted
            boundary falls outside the sequence are omitted.

        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        check_featurizer(df_feat=self._df_feat)
        ut.check_df_seq(df_seq=df_seq)
        missing = [c for c in [ut.COL_SEQ, ut.COL_TMD_START, ut.COL_TMD_STOP] if c not in df_seq.columns]
        if missing:
            raise ValueError(f"'df_seq' should contain columns {missing} for predict(level='domain').")
        ut.check_number_range(name="window", val=window, min_val=0, just_int=True)
        # Position-based frame (drop precomputed part columns so shifted boundaries recompute)
        cols = [ut.COL_ENTRY, ut.COL_SEQ, ut.COL_TMD_START, ut.COL_TMD_STOP]
        base = df_seq[cols].copy().reset_index(drop=True)
        seq_len = base[ut.COL_SEQ].str.len().to_numpy()
        rows = []
        for offset in range(-window, window + 1):
            d = base.copy()
            d[ut.COL_TMD_START] = d[ut.COL_TMD_START] + offset
            d[ut.COL_TMD_STOP] = d[ut.COL_TMD_STOP] + offset
            valid = (d[ut.COL_TMD_START] >= 1) & (d[ut.COL_TMD_STOP] <= seq_len)
            d = d[valid]
            if len(d) == 0:
                continue
            X = featurize_seq(df_feat=self._df_feat, df_scales=self._df_scales, df_seq=d, list_parts=list_parts)
            pred, _ = self._predict_X(X)
            for entry, score in zip(d[ut.COL_ENTRY].to_numpy(), pred):
                rows.append([entry, offset, float(score)])
        df_domain = pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_OFFSET, ut.COL_SCORE])
        idx_best = df_domain.groupby(ut.COL_ENTRY)[ut.COL_SCORE].idxmax()
        df_domain["is_best"] = df_domain.index.isin(idx_best)
        return df_domain

    def _predict_window(self,
                       df_seq: pd.DataFrame,
                       tmd_len: int,
                       step: int = 1,
                       jmd_n_len: int = 10,
                       jmd_c_len: int = 10,
                       list_parts: Optional[List[str]] = None,
                       ) -> pd.DataFrame:
        """
        Predict a per-residue score profile by sliding a fixed-length window along each sequence.

        A length-``tmd_len`` domain is anchored at every valid position (spaced by ``step``) and
        scored, yielding one score per position — the input to a per-residue prediction profile.

        .. note::
           Requires a ``df_feat`` bound at construction and a prior :meth:`fit`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_proteins, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and a
            ``sequence`` column.
        tmd_len : int
            Length (in residues) of the domain window anchored at each position.
        step : int, default=1
            Spacing between consecutive anchor positions.
        jmd_n_len : int, default=10
            Length of the N-terminal flanking region required around each window.
        jmd_c_len : int, default=10
            Length of the C-terminal flanking region required around each window.
        list_parts : list of str, optional
            Sequence parts to build for featurization. Defaults to the standard CPP parts.

        Returns
        -------
        df_window : pd.DataFrame, shape (n_rows, 4)
            Long-format table with columns ``entry``, ``position`` (1-based anchor), ``score``, and
            ``score_std``. Positions without enough flanking residues for the full window are omitted.

        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        check_featurizer(df_feat=self._df_feat)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(f"'df_seq' should contain a '{ut.COL_SEQ}' column for predict(level='window').")
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        ut.check_number_range(name="step", val=step, min_val=1, just_int=True)
        ut.check_number_range(name="jmd_n_len", val=jmd_n_len, min_val=0, just_int=True)
        ut.check_number_range(name="jmd_c_len", val=jmd_c_len, min_val=0, just_int=True)
        half_left, half_right = ut.get_window_offsets(tmd_len)
        rows = []
        for entry, seq in zip(df_seq[ut.COL_ENTRY].to_numpy(), df_seq[ut.COL_SEQ].to_numpy()):
            lo = half_left + jmd_n_len + 1
            hi = len(seq) - half_right + 1 - jmd_c_len
            positions = list(range(lo, hi + 1, step))
            if len(positions) == 0:
                continue
            d = pd.DataFrame({ut.COL_ENTRY: [f"{entry}__{p}" for p in positions],
                              ut.COL_SEQ: [seq] * len(positions), ut.COL_POS: positions})
            X = featurize_seq(df_feat=self._df_feat, df_scales=self._df_scales, df_seq=d,
                              list_parts=list_parts, tmd_len=tmd_len,
                              jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            pred, pred_std = self._predict_X(X)
            for p, score, score_std in zip(positions, pred, pred_std):
                rows.append([entry, p, float(score), float(score_std)])
        df_window = pd.DataFrame(rows, columns=[ut.COL_ENTRY, ut.COL_RESIDUE_POS, ut.COL_SCORE, ut.COL_SCORE_STD])
        return df_window
