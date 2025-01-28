"""
This is a script for the frontend of the TreeModel class used to obtain Mote Carlo estimates of feature importance.
"""
from typing import Optional, Dict, List, Tuple, Type, Union, Callable
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
import numpy as np

import aaanalysis.utils as ut

from ._backend.check_models import (check_match_labels_X,
                                    check_match_X_is_selected)
from ._backend.tree_model.tree_model_fit import fit_tree_based_models
from ._backend.tree_model.tree_model_predict_proba import monte_carlo_predict_proba
from ._backend.tree_model.tree_model_eval import eval_feature_selections


# I Helper Functions
def check_is_preselected(is_preselected=None):
    """Check if preselected is valid"""
    is_preselected = ut.check_array_like(name="is_preselected_feature", val=is_preselected, accept_none=True,
                                         expected_dim=1, dtype="bool")
    if is_preselected is not None and sum(is_preselected) == 0:
        raise ValueError(f"Number of preselected features should not be 0 in 'is_preselected': {is_preselected}")
    return is_preselected


def check_n_feat_min_n_feat_max(n_feat_min=None, n_feat_max=None):
    """Check if vmin and vmax are valid numbers and vmin is less than vmax."""
    ut.check_number_range(name="n_feat_min", val=n_feat_min, min_val=1, just_int=True, accept_none=False)
    ut.check_number_range(name="n_feat_max", val=n_feat_max, min_val=1, just_int=True, accept_none=False)
    if n_feat_min > n_feat_max:
        raise ValueError(f"'n_feat_min' ({n_feat_min}) >= 'n_feat_max' ({n_feat_max}) not fulfilled.")


def check_n_cv(n_cv=None, labels=None):
    """Check if n_cv is valid"""
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    # Check if smaller than smaller class
    unique, counts = np.unique(labels, return_counts=True)
    min_class_count = min(counts)
    # Ensure n_cv is not greater than the smallest class count
    if n_cv > min_class_count:
        raise ValueError(f"'n_cv' should not be greater than the smallest class count ({min_class_count}), got {n_cv}.")


def check_metrics(name="list_eval_sores", metrics=None):
    """Check if evaluation metrics are valid"""
    valid_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    if metrics is None:
        raise ValueError(f"'{name}' should not be None. Chose from these: {valid_metrics}")
    metrics = ut.check_list_like(name=name, val=metrics, accept_str=True)
    wrong_metrics = [x for x in metrics if x not in valid_metrics]
    if len(wrong_metrics) != 0:
        raise ValueError(f"Wrong 'metrics' ({wrong_metrics}). Chose from these: {valid_metrics}")


def check_list_is_selected(list_is_selected=None, X=None, convert_1d_to_2d=False):
    """Check if list_is_feature is list containing bool 2D matrix matching to X"""
    if not isinstance(list_is_selected, list):
        raise ValueError("'list_is_feature' must be a list with 2D boolean array.")
    list_is_selected = list_is_selected.copy()
    n_features = X.shape[1]
    str_error = f"Each element in 'list_is_feature' must be a 2D boolean array each with 'n_features'={n_features} elements."
    for index, is_feature_round in enumerate(list_is_selected):
        if convert_1d_to_2d and isinstance(is_feature_round, np.ndarray):
            if is_feature_round.ndim != 1:
                raise ValueError(f"Element at index {index} should be 1D boolean array if 'convert_1d_to_2d=True'")
            else:
                if is_feature_round.dtype != bool:
                    raise ValueError(f"Element at index {index} is not a boolean array.")
                if is_feature_round.size != n_features:
                    raise ValueError(f"1D array at index {index} does not match 'n_features'={n_features}.")
            list_is_selected[index] = is_feature_round.reshape(1, -1)  # Convert and update the array in the list

        is_feature_round = list_is_selected[index]
        if not isinstance(is_feature_round, np.ndarray):
            raise ValueError(f"Element at index {index} in 'list_is_feature' is not a numpy array.\n {str_error}")
        if is_feature_round.dtype != bool:
            raise ValueError(f"Element at index {index} in 'list_is_feature' is not a boolean array.\n {str_error}")
        if is_feature_round.ndim != 2:
            raise ValueError(f"Element at index {index} in 'list_is_feature' is not a 2D array.\n {str_error}")
        if is_feature_round.shape[1] != n_features:
            raise ValueError(
                f"Element at index {index} in 'list_is_feature' does not match the number of features in 'X'. "
                f"Expected {n_features} features, got {is_feature_round.shape[1]}.\n {str_error}")
    return list_is_selected


def check_match_df_feat_importance_arrays(df_feat=None, feat_importance=None, feat_importance_std=None, drop=False):
    """Check if df_feat matches with importance values"""
    if feat_importance is None:
        raise ValueError(f"'feat_importance' is None. Please fit TreeModel before adding feature importance.")
    if feat_importance_std is None:
        raise ValueError(f"'feat_importance_std' is None. Please fit TreeModel before adding feature importance.")
    n_feat = len(df_feat)
    n_feat_imp = len(feat_importance)
    n_feat_imp_std = len(feat_importance_std)
    if n_feat != n_feat_imp:
        raise ValueError(f"Mismatch of number of features in 'df_feat' (n={n_feat} and in 'feat_importance' (n={n_feat_imp})")
    if n_feat != n_feat_imp_std:
        raise ValueError(f"Mismatch of number of features in 'df_feat' (n={n_feat} and in 'feat_importance_std' (n={n_feat_imp})")
    if not drop:
        # Check if columns already exist
        if ut.COL_FEAT_IMPORT in list(df_feat):
            raise ValueError(f"'{ut.COL_FEAT_IMPORT}' already in 'df_feat' columns. To override, set 'drop=True'.")
        if ut.COL_FEAT_IMPORT_STD in list(df_feat):
            raise ValueError(f"'{ut.COL_FEAT_IMPORT_STD}' already in 'df_feat' columns. To override, set 'drop=True'.")


# II Main Functions
class TreeModel:
    """
    Tree Model class: A wrapper for tree-based models to obtain Monte Carlo estimates of feature
    importance and predictions [Breimann25a]_.

    Monte Carlo estimates are derived by averaging feature importance or prediction probabilities across various
    tree-based models and training rounds, enhancing the robustness and reproducibility of these estimates.
    Additionally, the class supports feature selection through recursive feature elimination (RFE) and offers
    comprehensive evaluation of feature selections.

    Attributes
    ----------
    list_models_ : Nested list with objects, shape (n_rounds, n_models)
        List with fitted tree-based models for every round after calling the ``fit`` method.
    feat_importance : array-like, shape (n_features)
        An array containing importance of each feature averaged across all rounds
        and trained models from `list_model_classes`.
    feat_importance_std : array-like, shape (n_features)
        An array containing standard deviation for feature importance across all rounds
        and trained models from `list_model_classes`. Same order as ``feature_importance``.
    is_selected_ : array-like, shape (n_rounds, n_features)
        2D array indicating features being selected by recursive features selection (True) or not (False) for each round.
        Same order as ``feature_importance``.
    """
    def __init__(self,
                 list_model_classes: List[Type[Union[ClassifierMixin, BaseEstimator]]] = None,
                 list_model_kwargs: Optional[List[Dict]] = None,
                 is_preselected: Optional[ut.ArrayLike1D] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        list_model_classes : list of Type[ClassifierMixin or BaseEstimator], default=[RandomForestClassifier, ExtraTreesClassifier]
            A list of tree-based model classes to be used for feature importance analysis.
        list_model_kwargs : list of dict, optional
            A list of dictionaries containing keyword arguments for each model in `list_model_classes`.
        is_preselected : array-like, shape (n_features)
            Boolean array indicating features being preselected before applying recursive features selection.
            ``True`` indicates that a feature is preselected and ``False`` that it is not.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All attributes are set during fitting via the :meth:`TreeModel.fit` method and can be directly accessed.

        See Also
        --------
        * :class:`sklearn.ensemble.RandomForestClassifier` for random forest model.
        * :class:`sklearn.ensemble.ExtraTreesClassifier` for extra trees model.

        Warnings
        --------
        * This class belongs to the explainable AI module requiring `SHAP`,
          which is automatically installed via `pip install aaanalysis[pro]`.

        Examples
        --------
        .. include:: examples/tree_model.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        is_preselected = check_is_preselected(is_preselected=is_preselected)
        # Model parameters
        if list_model_classes is None:
            list_model_classes = [RandomForestClassifier, ExtraTreesClassifier]
        elif not isinstance(list_model_classes, list):
            list_model_classes = [list_model_classes]   # Single models are possible as well (but not recommender)
        list_model_classes = ut.check_list_like(name="list_model_classes", val=list_model_classes,
                                                accept_none=False, min_len=1)
        list_model_kwargs = ut.check_list_like(name="list_model_kwargs", val=list_model_kwargs, accept_none=True)
        if list_model_kwargs is None:
            list_model_kwargs = [{} for _ in list_model_classes]
        ut.check_match_list_model_classes_kwargs(list_model_classes=list_model_classes, list_model_kwargs=list_model_kwargs)
        _list_model_kwargs = []
        for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs):
            ut.check_mode_class(model_class=model_class)
            model_kwargs = ut.check_model_kwargs(model_class=model_class,
                                                 model_kwargs=model_kwargs,
                                                 attribute_to_check="feature_importances_",
                                                 random_state=random_state)
            _list_model_kwargs.append(model_kwargs)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._list_model_classes = list_model_classes
        self._list_model_kwargs = _list_model_kwargs
        self._is_preselected = is_preselected
        # Output parameters (set during model fitting)
        self.feat_importance: Optional[ut.ArrayLike1D] = None
        self.feat_importance_std: Optional[ut.ArrayLike1D] = None
        self.is_selected_: Optional[ut.ArrayLike2D] = None
        self.list_models_: Optional[List[List[Union[ClassifierMixin, BaseEstimator]]]] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            n_rounds: int = 5,
            use_rfe: bool = False,
            n_cv: int = 5,
            n_feat_min: int = 25,
            n_feat_max: int = 75,
            metric: str = "accuracy",
            step: Optional[int] = None,
            ) -> "TreeModel":
        """
        Fit tree-based models and compute average feature importance [Breimann25a]_.

        The feature importance is calculated across all models and rounds. In each round, the set of features can
        optionally be prefiltered using Recursive Feature Elimination (RFE) with a default RandomForestClassifier,
        if ``use_rfe`` is set to True. This RFE process iteratively reduces the number of features to enhance model
        performance, guided by the metric specified in ``metric``. The reduction continues until reaching
        ``n_feat_min``, with an upper limit of ``n_feat_max`` features considered.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples)
            Class labels for samples in ``X`` (typically, 1=positive, 0=negative).
        n_rounds : int, default=5
            The number of rounds (>=1) to fit the model.
        use_rfe : bool, default=False
            If ``True``, recursive feature elimination (RFE) is used with random forest model for feature selection.
        n_cv : int, default=5
            The number of cross-validation folds for RFE, must be > 1 and ≤ the smallest class's sample count.
        n_feat_min : int, default=25
            The minimum number of features to select each round for RFE. Should be 0 < ``n_feat_min`` <= ``n_feat_max``.
        n_feat_max : int, default=75
            The maximum number of features to select each round for RFE. Should be >= ``n_feat_min``.
        metric : str, default="accuracy"
            The name of the scoring function to use for cross-validation for RFE.
            Valid metrics are: {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        step : int, optional
            Number of features to remove per RFE iteration. If ``None``, removes all features with the lowest importance
            scores each iteration, offering a faster but less precise approach. Should be < ``n_features``.
            Ideally, ``step`` should be much smaller than ``n_features``, with 1-5% of `n_features` being a recommended range.

        Returns
        -------
        TreeModel
            The fitted TreeModel instance.

        See Also
        --------
        * [Breimann25a]_ describes recursive feature elimination algorithm and feature importance aggregation.
        * :class:`sklearn.ensemble.RandomForestClassifier` for the random forest model used with default settings
          for recursive feature elimination.
        * :class:`sklearn.feature_selection.RFECV` for similar cross-validation based recursive feature elimination
          algorithm. This one does not provide an upper limit for the number of features to select.
        * :func:`sklearn.model_selection.cross_validate` for details on cross-validation
        * Sckit-learn `cross-validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_ documentation.
        * Sckit-learn `classification metrics and scorings <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.

        Examples
        --------
        .. include:: examples/tm_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        n_samp, n_feat = X.shape
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        labels = check_match_labels_X(labels=labels, X=X)
        ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
        ut.check_bool(name="use_rfe", val=use_rfe)
        check_n_cv(n_cv=n_cv, labels=labels)
        check_n_feat_min_n_feat_max(n_feat_min=n_feat_min, n_feat_max=n_feat_max)
        ut.check_str(name="metric", val=metric)
        check_metrics(name="metric", metrics=metric)
        ut.check_number_range(name="step", val=step, min_val=1, max_val=n_feat-1, accept_none=True, just_int=True)
        # Fit tree-based models and save feature importance
        args = dict(X=X, labels=labels,
                    list_model_classes=self._list_model_classes,
                    list_model_kwargs=self._list_model_kwargs,
                    is_preselected=self._is_preselected,
                    n_rounds=n_rounds, use_rfe=use_rfe, n_cv=n_cv,
                    n_feat_min=n_feat_min, n_feat_max=n_feat_max, metric=metric, step=step,
                    verbose=self._verbose, random_state=self._random_state)
        feature_importance, is_selected_rounds, list_models_rounds = fit_tree_based_models(**args)
        # Save results in output parameters
        self.feat_importance = np.mean(feature_importance, axis=0).round(5) * 100
        self.feat_importance_std = np.std(feature_importance, axis=0).round(5) * 100
        self.is_selected_ = is_selected_rounds
        self.list_models_ = list_models_rounds
        return self

    def eval(self,
             X: ut.ArrayLike2D,
             labels: ut.ArrayLike1D = None,
             list_is_selected: List[ut.ArrayLike2D] = None,
             convert_1d_to_2d: bool = False,
             names_feature_selections: Optional[List[str]] = None,
             n_cv: int = 5,
             list_metrics: Union[str, List[str]] = None,
             ) -> pd.DataFrame:
        """
        Evaluate the prediction performance for different feature selections.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples)
            Class labels for samples in ``X`` (typically, 1=positive, 0=negative).
        list_is_selected : array-like, shape (n_feature_sets, n_round, n_features)
            List of 2D boolean arrays with shape (n_rounds, n_features) indicating different feature selections.
        convert_1d_to_2d : bool, default=False
            If ``True``, convert all boolean arrays in `list_is_selected` from 1D to 2D arrays with a single row.
        names_feature_selections : list of str, optional
            List of dataset names corresponding to ``list_is_feature``.
        n_cv : int, default=5
            The number of cross-validation folds for RFE, must be > 1 and ≤ the smallest class's sample count.
        list_metrics : str or list of str, default=['accuracy', 'precision', 'recall', 'f1']
            List of scoring metrics to use for evaluation. Only metrics for binary classification are allowed.
            Valid metrics are: {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for feature subsets obtained by recursive feature selection give with ``list_is_feature``.

        Notes
        -----
        * :func:`sklearn.metrics.balanced_accuracy_score` is recommended if datasets are unbalanced.

        Warnings
        --------
        * ``list_metrics``: 'precision' and 'f1' metrics may trigger 'UndefinedMetricWarning' in imbalanced or
          small datasets due to division by zero if no positive predictions can be maid.

        See Also
        --------
        * Sckit-learn `classification metrics and scorings <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.

        Examples
        --------
        .. include:: examples/tm_eval.rst
        """
        # Check input
        X = ut.check_X(X=X, min_n_unique_features=2)
        n_samples, n_feat = X.shape
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        labels = check_match_labels_X(labels=labels, X=X)
        list_is_selected = ut.check_list_like(name="list_is_selected", val=list_is_selected)
        ut.check_bool(name="convert_1d_to_2d", val=convert_1d_to_2d)
        list_is_selected = check_list_is_selected(list_is_selected=list_is_selected, X=X,
                                                  convert_1d_to_2d=convert_1d_to_2d)
        names_feature_selections = ut.check_list_like(name="name_feature_selections", val=names_feature_selections, accept_none=True,
                                                      accept_str=True, check_all_str_or_convertible=True)
        if list_metrics is None:
            list_metrics = ["accuracy", "precision", "recall", "f1"]
        check_metrics(name="list_metrics", metrics=list_metrics)
        check_n_cv(n_cv=n_cv, labels=labels)
        # Perform evaluation
        df_eval = eval_feature_selections(X, labels=labels,
                                          list_is_selected=list_is_selected,
                                          names_feature_selections=names_feature_selections,
                                          n_cv=n_cv,
                                          list_metrics=list_metrics,
                                          list_model_classes=self._list_model_classes,
                                          list_model_kwargs=self._list_model_kwargs,
                                          verbose=self._verbose)

        return df_eval

    def predict_proba(self,
                      X: ut.ArrayLike2D,
                      ) -> Tuple[ut.ArrayLike1D, ut.ArrayLike1D]:
        """
        Obtain Monte Carlo estimate of class prediction probabilities for the positive class in `X`.

        Predictions are performed using all tree-based models from the `list_model_classes` attribute and
        feature selections from the `is_selected` attribute.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Rows typically correspond to samples and columns to features.

        Returns
        -------
        pred : array-like, shape (n_samples)
            Array with the average prediction score for the positive class for each sample.
        pred_std : array-like, shape (n_samples)
            Array with the standard deviation of prediction scores for the positive class for each sample.

        See Also
        --------
        * :class:`sklearn.ensemble.RandomForestClassifier` for information on tree-based models.
        * :meth:`TreeModel.fit` for fitting the TreeModel object.

        Examples
        --------
        .. include:: examples/tm_predict_proba.rst
        """
        # Check input
        X = ut.check_X(X=X)
        check_match_X_is_selected(X=X, is_selected=self.is_selected_)
        # Predictions
        pred, pred_std = monte_carlo_predict_proba(X=X,
                                                   list_models=self.list_models_,
                                                   is_selected=self.is_selected_)
        return pred, pred_std

    def add_feat_importance(self,
                            df_feat: pd.DataFrame = None,
                            drop: bool = False
                            ) -> pd.DataFrame:
        """
        Include feature importance and its standard deviation to feature DataFrame.

        Feature importance is included as ``feat_importance`` column and the standard deviation of
        the feature importance as ``feat_importance_std`` column.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        drop : bool, default=False
            If ``True``, allow dropping of already existing ``feat_importance`` and ``feat_importance_std`` columns
            from ``df_feat`` before inserting.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info+2)
            Feature DataFrame including ``feat_importance`` and ``feat_importance_std`` columns.

        See Also
        --------
        * :meth:`CPP.run` for details on CPP statistical measures of feature DataFrame.

        Examples
        --------
        .. include:: examples/tm_add_feat_importance.rst
         """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_bool(name="drop", val=drop)
        check_match_df_feat_importance_arrays(df_feat=df_feat,
                                              feat_importance=self.feat_importance,
                                              feat_importance_std=self.feat_importance_std,
                                              drop=drop)
        # Add feature importance and its standard deviation to the DataFrame
        df_feat = df_feat.copy()
        if drop:
            columns = [x for x in list(df_feat) if x not in [ut.COL_FEAT_IMPORT, ut.COL_FEAT_IMPORT_STD]]
            df_feat = df_feat[columns]
        args = dict(allow_duplicates=False)
        df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT, value=self.feat_importance, **args)
        df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT_STD, value=self.feat_importance_std, **args)
        return df_feat
