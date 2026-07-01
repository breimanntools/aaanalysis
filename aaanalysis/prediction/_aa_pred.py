"""
This is a script for the frontend of the AAPred class for evaluating and deploying prediction models.
"""
from typing import Optional, Dict, List, Tuple, Type, Union
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper

from ._backend.aa_pred.aa_pred_fit import fit_models, predict_proba_models
from ._backend.aa_pred.aa_pred_eval import eval_models


# I Helper Functions
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


# II Main Functions
class AAPred(Wrapper):
    """
    AAPred: evaluate and deploy sequence-based prediction models (Wrapper) [Breimann25]_.

    A thin, opinionated wrapper that closes the gap left by feature engineering: given a
    feature matrix ``X`` and ``labels``, it **evaluates** one or more scikit-learn model
    classes across metrics by cross-validation and an optional held-out set (:meth:`eval`),
    and **deploys** them by fitting on all data and exposing prediction scores
    (:meth:`fit` / :meth:`predict_proba` / :meth:`predict`).

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
                 list_model_classes: Optional[List[Type[Union[ClassifierMixin, BaseEstimator]]]] = None,
                 list_model_kwargs: Optional[List[Dict]] = None,
                 list_metrics: Optional[List[str]] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        list_model_classes : list of Type[ClassifierMixin or BaseEstimator], default=[RandomForestClassifier]
            Model classes to evaluate and deploy. Each must implement ``predict_proba``.
        list_model_kwargs : list of dict, optional
            Keyword arguments for each model in ``list_model_classes`` (same length).
        list_metrics : list of str, default=["accuracy", "balanced_accuracy", "f1", "roc_auc"]
            Default performance metrics used by :meth:`eval` when ``metrics`` is not given.
            Each should be one of ``accuracy``, ``balanced_accuracy``, ``precision``,
            ``recall``, ``f1``, ``roc_auc``.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of
            stochastic processes are consistent, enabling reproducibility. If ``None``,
            stochastic processes will be truly random.

        Examples
        --------
        .. include:: examples/aapred.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Model parameters
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
        # Metric parameters
        if list_metrics is None:
            list_metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
        list_metrics = check_metrics(metrics=list_metrics)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._list_model_classes = list_model_classes
        self._list_model_kwargs = _list_model_kwargs
        self._list_metrics = list_metrics
        # Output attributes (set during fitting)
        self.list_models_: Optional[List[Union[ClassifierMixin, BaseEstimator]]] = None
        self.label_pos_: Optional[int] = None
        self.label_neg_: Optional[int] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D,
            label_pos: int = 1,
            ) -> "AAPred":
        """
        Fit every model on the full dataset for deployment.

        Each model class from the constructor is instantiated and fit on all of ``X`` / ``labels``;
        the fitted estimators are kept in ``list_models_`` and reused by :meth:`predict_proba` /
        :meth:`predict`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.
        labels : array-like, shape (n_samples,)
            Class labels for samples in ``X`` (typically ``1`` for the positive class and
            ``0`` for the negative class).
        label_pos : int, default=1
            Label of the positive class whose probability :meth:`predict_proba` returns.

        Returns
        -------
        AAPred
            The fitted ``AAPred`` instance (``self``).

        Examples
        --------
        .. include:: examples/aapred_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        classes = check_binary_labels(labels=labels)
        ut.check_number_val(name="label_pos", val=label_pos, just_int=True)
        if label_pos not in classes:
            raise ValueError(f"'label_pos' ({label_pos}) should be one of the labels: {classes}")
        # Fit models
        self.list_models_ = fit_models(X=X, labels=labels,
                                       list_model_classes=self._list_model_classes,
                                       list_model_kwargs=self._list_model_kwargs)
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
             ) -> pd.DataFrame:
        """
        Evaluate every model across metrics by cross-validation and an optional held-out set.

        Two evaluation principles are reported: ``cv`` (stratified k-fold cross-validation on
        ``X``) and, when a held-out set is provided, ``holdout`` (models fit on ``X`` and scored
        on ``X_holdout``). The result is a long-format table with one row per
        (model, metric, principle).

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
            Number of stratified cross-validation folds (must not exceed the smallest class count).

        Returns
        -------
        df_eval : pd.DataFrame, shape (n_rows, 5)
            Long-format evaluation table with columns ``model``, ``metric``, ``principle``,
            ``score``, and ``score_std`` (``score_std`` is ``NaN`` for the ``holdout`` principle).

        Examples
        --------
        .. include:: examples/aapred_eval.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        check_binary_labels(labels=labels)
        metrics = check_metrics(metrics=metrics) if metrics is not None else self._list_metrics
        check_n_cv(n_cv=n_cv, labels=labels)
        list_models = None
        if X_holdout is not None:
            X_holdout = ut.check_X(X=X_holdout, min_n_samples=1)
            labels_holdout = ut.check_labels(labels=labels_holdout)
            ut.check_match_X_labels(X=X_holdout, labels=labels_holdout)
            if X_holdout.shape[1] != X.shape[1]:
                raise ValueError(f"'X_holdout' n_features ({X_holdout.shape[1]}) should match "
                                 f"'X' n_features ({X.shape[1]}).")
            list_models = fit_models(X=X, labels=labels,
                                     list_model_classes=self._list_model_classes,
                                     list_model_kwargs=self._list_model_kwargs)
        elif labels_holdout is not None:
            raise ValueError("'labels_holdout' was given without 'X_holdout'.")
        # Evaluate
        df_eval = eval_models(X=X, labels=labels,
                              list_model_classes=self._list_model_classes,
                              list_model_kwargs=self._list_model_kwargs,
                              list_models=list_models, metrics=metrics, n_cv=n_cv,
                              random_state=self._random_state,
                              X_holdout=X_holdout, labels_holdout=labels_holdout)
        return df_eval

    def predict_proba(self,
                      X: ut.ArrayLike2D,
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtain positive-class prediction scores for samples in ``X``.

        Scores are averaged across all fitted models from ``list_models_``.

        .. note::
           :meth:`AAPred.fit` must be called before using this method.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.

        Returns
        -------
        pred : array-like, shape (n_samples,)
            Average positive-class prediction score for each sample.
        pred_std : array-like, shape (n_samples,)
            Standard deviation of the score across models (``0`` for a single model).

        Examples
        --------
        .. include:: examples/aapred_predict_proba.rst
        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        X = ut.check_X(X=X, min_n_samples=1)
        # Predict
        pred, pred_std = predict_proba_models(X=X, list_models=self.list_models_, label_pos=self.label_pos_)
        return pred, pred_std

    def predict(self,
                X: ut.ArrayLike2D,
                threshold: Union[int, float] = 0.5,
                ) -> np.ndarray:
        """
        Predict positive-class membership by thresholding the prediction score.

        The averaged positive-class score from :meth:`predict_proba` is compared against
        ``threshold``: samples at or above it are labeled ``label_pos`` (from :meth:`fit`), the rest ``0``.

        .. note::
           :meth:`AAPred.fit` must be called before using this method.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.
        threshold : int or float, default=0.5
            Score at or above which a sample is predicted positive (``label_pos`` from :meth:`fit`).

        Returns
        -------
        pred_labels : array-like, shape (n_samples,)
            Predicted labels (``label_pos`` where the score is ``>= threshold``, else the
            negative class label seen during :meth:`fit`).

        Examples
        --------
        .. include:: examples/aapred_predict.rst
        """
        # Check input
        check_is_fitted(list_models=self.list_models_)
        X = ut.check_X(X=X, min_n_samples=1)
        ut.check_number_range(name="threshold", val=threshold, min_val=0, max_val=1, just_int=False)
        # Predict
        pred, _ = predict_proba_models(X=X, list_models=self.list_models_, label_pos=self.label_pos_)
        pred_labels = np.where(pred >= threshold, self.label_pos_, self.label_neg_)
        return pred_labels
