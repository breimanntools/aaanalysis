"""
This is a script for the frontend of the TreeModel class used to obtain feature importance reproducibly.
"""
from typing import Optional, Dict, List, Tuple, Type, Union, Callable
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np

import aaanalysis.utils as ut

from .backend.tree_model.tree_model_fit import recursive_feature_elimination, compute_feature_importance
from .backend.tree_model.tree_model_eval import eval_feature_selections


# I Helper Functions
def check_match_list_model_classes_kwargs(list_model_classes=None, list_model_kwargs=None):
    """Check length match of list_model_classes and list_model_kwargs"""
    n_models = len(list_model_classes)
    n_args = len(list_model_kwargs)
    if n_models != n_args:
        raise ValueError(f"Length of 'list_model_kwargs' (n={n_args}) should match to 'list_model_classes' (n{n_models}")


def check_match_labels_X(labels=None, X=None):
    """Check if labels binary classification task labels"""
    n_samples = X.shape[0]
    labels = ut.check_labels(labels=labels, len_requiered=n_samples)
    unique_labels = set(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"'labels' should contain 2 unique labels ({unique_labels})")
    label_test = list(unique_labels)[0]
    labels = np.asarray([1 if x == label_test else 0 for x in labels])
    return labels


def check_n_feat_min_n_feat_max(n_feat_min=None, n_feat_max=None):
    """Check if vmin and vmax are valid numbers and vmin is less than vmax."""
    ut.check_number_range(name="n_feat_min", val=n_feat_min, min_val=1, just_int=True, accept_none=False)
    ut.check_number_range(name="n_feat_max", val=n_feat_min, min_val=2, just_int=True, accept_none=False)
    if n_feat_min >= n_feat_max:
        raise ValueError(f"'n_feat_min' ({n_feat_min}) < 'n_feat_max' ({n_feat_max}) not fulfilled.")


def check_metrics(name="list_eval_sores", metrics=None):
    """Check if evaluation metrics are valid"""
    valid_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    if metrics is None:
        raise ValueError(f"'{name}' should not be None. Chose from these: {valid_metrics}")
    metrics = ut.check_list_like(name=name, val=metrics, accept_str=True)
    wrong_metrics = [x for x in metrics if x not in valid_metrics]
    if len(wrong_metrics) != 0:
        raise ValueError(f"Wrong 'metrics' ({wrong_metrics}). Chose from these: {valid_metrics}")


def check_list_is_feature(list_is_feature=None, X=None):
    """Check if list_is_feature is list containing bool 2D matrix matching to X"""
    if not isinstance(list_is_feature, list):
        raise ValueError("'list_is_feature' must be a list with 2D boolean matrix.")
    if X is not None:
        n_features = X.shape[1]
        for is_feature_rounds in list_is_feature:
            if not all(isinstance(is_feature, np.ndarray) and is_feature.dtype == bool for is_feature in is_feature_rounds):
                raise ValueError("Each element in 'list_is_feature' must be a 2D boolean array.")
            for is_feature in is_feature_rounds:
                _n_features = is_feature.shape[0]
                if _n_features != n_features:
                    raise ValueError(f"Number of feature does not match for 'list_is_feature' (n={_n_features}) and 'X' (n={n_features}.")

def check_match_df_feat_importance_arrays(df_feat=None, feat_importance=None, feat_importance_std=None, drop=False):
    """Check if df_feat matches with importance values"""
    n_feat = len(df_feat)
    n_feat_imp = len(feat_importance)
    n_feat_imp_std = len(feat_importance_std)
    if n_feat != n_feat_imp:
        raise ValueError(f"Mismatch of number of features in 'df_feat' (n={n_feat} and in 'feat_importance' (n={n_feat_imp})")
    if n_feat != n_feat_imp_std:
        raise ValueError(f"Mismatch of number of features in 'df_feat' (n={n_feat} and in 'feat_importance_std' (n={n_feat_imp})")
    if drop:
        # Check if columns already exist
        if ut.COL_FEAT_IMPORT in list(df_feat):
            raise ValueError(f"'{ut.COL_FEAT_IMPORT}' already in 'df_feat' columns: {list(df_feat)}")
        if ut.COL_FEAT_IMPORT_STD in list(df_feat):
            raise ValueError(f"'{ut.COL_FEAT_IMPORT_STD}' already in 'df_feat' columns: {list(df_feat)}")


# II Main Functions
class TreeModel:
    """
    UNDER CONSTRUCTION - Tree Model class: A wrapper for tree-based prediction models to obtain feature importance.

    Attributes
    ----------
    feat_importance : array-like, shape (n_features)
        An arrays containing importance of each feature averaged across all rounds
        and trained models from `list_model_classes`.
    feat_importance_std : array-like, shape (n_features)
        An arrays containing standard deviation for feature importance across all rounds
        and trained models from `list_model_classes`. Same order as ``feature_importance``.
    is_feature_ : array-like, shape (n_rounds, n_features)
        2D array indicating features being selected by recursive features selection  (1) or not (0) for each round.
        Same order as ``feature_importance``.

    Notes
    -----
    * All attributes are set during `.fit()` and can be directly accessed.

    See Also
    --------
    * :class:`sklearn.ensemble.RandomForestClassifier` for Random Forest model.
    * :class:`sklearn.ensemble.ExtraTreesClassifier` for Extra Trees model.
    * :class:`sklearn.ensemble.GradientBoostingClassifier` for Gradient Boosting model.
    """
    def __init__(self,
                 list_model_classes : List[Type[Union[ClassifierMixin, BaseEstimator]]] = None,
                 list_model_kwargs: Optional[List[Dict]] = None,
                 verbose: Optional[bool] = None,
                 random_state: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        list_model_classes : list of Type[ClassifierMixin or BaseEstimator], default=[RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
            A list of tree-based model classes to be used for feature importance analysis.
        list_model_kwargs : list of dict, optional
            A list of dictionaries containing keyword arguments for each model in `list_model_classes`.
        verbose : bool, optional
            If ``True``, verbose outputs are enabled. Global ``verbose`` setting is used if ´´None``.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Examples
        --------
        .. include:: examples/tree_model.rest
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Model parameters
        if list_model_classes is None:
            list_model_classes = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
        elif not isinstance(list_model_classes, list):
            list_model_classes = [list_model_classes]   # Single models are possible as well (but not recommender)
        list_model_classes = ut.check_list_like(name="list_model_classes", val=list_model_classes, accept_none=False)
        ut.check_list_like(name="list_model_kwargs", val=list_model_kwargs, accept_none=True)
        if list_model_kwargs is None:
            list_model_kwargs = [{}] * len(list_model_classes)
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
        # Output parameters (set during model fitting)
        self.feat_importance : Optional[ut.ArrayLike1D] = None
        self.feat_importance_std : Optional[ut.ArrayLike1D] = None
        self.is_feature_ : Optional[ut.ArrayLike2D] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            n_rounds: int = 10,
            use_rfe: bool = True,
            n_cv: int = 5,
            n_feat_min: int = 10,
            n_feat_max: int = 50,
            metric: str = "accuracy",
            step : Optional[int] = None,
            ) -> "TreeModel":
        """
        Fit tree-based models ``n_rounds`` time and compute average feature importance [Breimann24c]_.

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
            Dataset labels of samples in ``X``. Should be either 1 (positive) or 0 (negative).
        n_rounds : int, default=10
            The number of rounds (>=1) to fit the model.
        use_rfe : bool, default=True
            Whether to use recursive feature elimination (RFE) with random forest model for feature selection.
        n_feat_min : int, default=10
            The minimum number of features to select each round for RFE. Should be < ``n_feat_max``.
        n_feat_max : int, default=50
            The maximum number of features to select each round for RFE. Should be > ``n_fat_min``.
        metric : str, default="accuracy"
            The name of the scoring function to use for cross-validation for RFE.
        n_cv : int, default=5
            Determines the number of folds (>1) for cross-validation for RFE.
        step : int, optional
            Number of features to remove at each iteration for RFE. If ``None``, all features that have the minimum
            importance score are removed at each iteration, which is a faster but less controlled strategy.

        Returns
        -------
        TreeModel
            The fitted TreeModel instance.

        See Also
        --------
        * [Breimann24c]_ describes recursive feature elimination algorithm and feature importance aggregation.
        * :class:`sklearn.ensemble.RandomForestClassifier` for the random forest model used with default settings
          for recursive feature elimination.
        * :class:`sklearn.feature_selection.RFECV` for similar cross-validation based recursive feature elimination
          algorithm. This one does not provide an upper limit for the number of features to select.
        * :func:`sklearn.model_selection.cross_validate` for details on cross-validation
        * Sckit-learn `cross-validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_ documentation.
        * Sckit-learn `classification metrics and scorings <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.

        Examples
        --------
        .. include:: tree_model_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        labels = check_match_labels_X(labels=labels, X=X)
        ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
        ut.check_bool(name="use_rfe", val=use_rfe)
        ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
        check_n_feat_min_n_feat_max(n_feat_min=n_feat_min, n_feat_max=n_feat_max)
        ut.check_str(name="metric", val=metric)
        check_metrics(name="metric", metrics=metric)
        ut.check_number_range(name="step", val=step, min_val=1, accept_none=True, just_int=True)
        # Obtain tree-based feature importance
        n_features = X.shape[1]
        feature_importance = np.empty(shape=(n_rounds, n_features))
        is_feature_rounds = np.ones(shape=(n_rounds, n_features), dtype=bool)
        # TODO add progress
        for i in range(n_rounds):
            is_feature = np.ones(n_features, dtype=bool)
            # 1. Step: Recursive feature elimination
            if use_rfe:
                args = dict(labels=labels, n_feat_min=n_feat_min, n_feat_max=n_feat_max,
                            n_cv=n_cv, scoring=metric, step=step, random_state=self._random_state)
                is_feature = recursive_feature_elimination(X, **args)
            # 2. Step: Compute feature importance
            feature_importance[i, :] = compute_feature_importance(X, labels=labels, is_feature=is_feature,
                                                                  list_model_classes=self._list_model_classes,
                                                                  list_model_kwargs=self._list_model_kwargs)
            is_feature_rounds[i, :] = is_feature
        # Save results in output parameters
        self.feat_importance = np.mean(feature_importance, axis=0).round(5) * 100
        self.feat_importance_std = np.std(feature_importance, axis=0).round(5) * 100
        self.is_feature_ = is_feature_rounds
        return self

    def eval(self,
             X : ut.ArrayLike2D,
             labels: ut.ArrayLike1D = None,
             list_is_feature: List[ut.ArrayLike2D] = None,
             names_feature_selections: Optional[List[str]] = None,
             list_metrics: Union[str, List[str]] = None,
             n_cv: int = 5,
             ) -> pd.DataFrame:
        """
        Evaluate the prediction performance for different feature selections.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples)
            Dataset labels of samples in ``X``. Should be either 1 (positive) or 0 (negative).
        list_is_feature : array-like, shape (n_feature_sets, n_round, n_features)
            List of boolean arrays with shape (n_rounds, n_features) indicating different feature selections.
        names_feature_selections : list of str, optional
            List of dataset names corresponding to ``list_is_feature``.
        n_cv : int, default=5
            Number of folds for cross-validation.
        list_metrics : str or list of str, default=['accuracy', 'f1', 'precision', 'recall']
            List of scoring metrics to use for evaluation. Only metrics for binary classification are allowed.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for feature subsets obtained by recursive feature selection give with ``list_is_feature``.

        Notes
        -----
        * :func:'sklearn.metrics.balanced_accuracy_score' is recommended if datasets are unbalanced.

        See Also
        --------
        * Sckit-learn `classification metrics and scorings <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.

        Examples
        --------
        .. include:: examples/tree_model_eval.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        labels = check_match_labels_X(labels=labels, X=X)
        check_list_is_feature(list_is_feature=list_is_feature, X=X)
        names_feature_selections = ut.check_list_like(name="name_feature_selections", val=names_feature_selections, accept_none=True,
                                                      accept_str=True, check_all_str_or_convertible=True)
        if list_metrics is None:
            list_metrics = ["accuracy", "f1", "precision", "recall"]
        check_metrics(name="list_metrics", metrics=list_metrics)
        ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
        # Perform evaluation
        df_eval = eval_feature_selections(X, labels=labels,
                                          list_is_feature=list_is_feature,
                                          names_feature_selections=names_feature_selections,
                                          n_cv=n_cv,
                                          list_metrics=list_metrics,
                                          list_model_classes=self._list_model_classes,
                                          list_model_kwargs=self._list_model_kwargs)

        return df_eval

    def add_feat_importance(self,
                            df_feat : pd.DataFrame = None,
                            drop : bool = False
                            ) -> pd.DataFrame:
        """
        Include feature importance and its standard deviation to feature DataFrame.

        Feature importance is included as ``feat_importance`` column and the standard deviation of
        the feature importance as ``feat_importance_std`` column.

        Parameters
        ----------
        df_feat : DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        drop : bool, default=False
            If ``True``, allow dropping of already existing ``feat_importance`` and ``feat_importance_std`` columns
            from ``df_feat`` before inserting.

        Returns
        -------
        df_feat : DataFrame, shape (n_features, n_feature_info+2)
            Feature DataFrame including ``feat_importance`` and ``feat_importance_std`` columns.

        See Also
        --------
        * :meth:`CPP.run` for details on CPP statistical measures of feature DataFrame.

        Examples
        --------
        .. include:: examples/tree_model_add_feat_importance.rst
         """
        # Check input
        ut.check_df_feat(df_feat=df_feat)
        check_match_df_feat_importance_arrays(df_feat=df_feat,
                                              feat_importance=self.feat_importance,
                                              feat_importance_std=self.feat_importance_std,
                                              drop=drop)
        # Add feature importance and its standard deviation to the DataFrame
        df_feat = df_feat.copy()
        if drop:
            columns = [x for x in list(df_feat) if x not in [ut.COL_FEAT_IMPORT, ut.COL_FEAT_IMPORT_STD]]
            df_feat = df_feat[columns]
        df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT, value=self.feat_importance,
                       allow_duplicates=False)
        df_feat.insert(loc=len(df_feat.columns), column=ut.COL_FEAT_IMPORT_STD, value=self.feat_importance_std,
                       allow_duplicates=False)
        return df_feat
