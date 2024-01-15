"""
This is a script for the processing SHAP values, primarily for the combination of SHAP with CPP.
SHAP models are not included due to instability of SHAP package development. ShapModel should solely work with
SHAP value matrix.
"""
from typing import Optional, Dict, List, Tuple, Type, Union, Callable
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

import aaanalysis.utils as ut
from .backend.check_models import (check_match_list_model_classes_kwargs)
from .backend.shap_explainer.shap_explainer_fit import monte_carlo_shap_estimation
from .backend.shap_explainer.shap_feat import (comp_shap_feature_importance,
                                               insert_shap_feature_importance,
                                               comp_shap_feature_impact,
                                               insert_shap_feature_impact)


# I Helper Functions
def check_pos(pos=None, n_samples=None):
    """Check if pos is int or list of ints"""
    if pos is None:
        # Create position list for all samples
        pos = list(range(n_samples))
        return pos
    if isinstance(pos, list) or isinstance(pos, np.ndarray):
        ut.check_list_like(name="pos", val=pos, check_all_non_neg_int=True)
    else:
        ut.check_number_range(name=f"pos", val=pos, min_val=0, max_val=n_samples, just_int=True)


def check_name(name=None):
    """Check if name is str ror list of str"""
    if name is None:
        return None     # Skip test
    if isinstance(name, list):
        for i in name:
            ut.check_str(name=f"name: {i}", val=name)
    else:
        ut.check_str(name=f"name", val=name)


def check_match_pos_name(pos=None, name=None):
    """Check if length of pos and name matches"""
    if name is not None:
        if isinstance(pos, list) and isinstance(name, list):
            if len(pos) != len(name):
                raise ValueError("Length of 'pos' and 'name' must be equal.")
        elif isinstance(pos, int) and isinstance(name, str):
            pos = [pos]  # Convert to list for consistency
            name = [name]
        else:
            raise ValueError("Type mismatch: 'pos' should be int if 'name' is str, and list if 'name' is list.")
    else:
        # Generate default names if name is None
        if isinstance(pos, list):
            name = [f"Protein{i + 1}" for i in range(len(pos))]
        else:
            name = ["Protein1"]
    return pos, name


def check_match_pos_group_average(pos=None, group_average=False):
    """Check if group_average only True if pos is list"""
    if group_average:
        ut.check_list_like(name=pos, val=pos)
        if len(pos) == 1:
            raise ValueError


def check_match_df_feat_shap_values(df_feat=None, shap_values=None, drop=False, shap_feat_importance=False):
    """Check if df_feat matches with importance values"""
    if shap_values is None:
        raise ValueError(f"'shap_values' is None. Please fit ShapExplainer before adding feature impact.")
    n_feat = len(df_feat)
    n_samples, n_feat_imp = shap_values.shape
    if n_feat != n_feat_imp:
        raise ValueError(f"Mismatch of number of features in 'df_feat' (n={n_feat} and in 'shap_values' (n={n_feat_imp})")
    if not drop:
        # Check if columns already exist
        if shap_feat_importance:
            if ut.COL_FEAT_IMPORT in list(df_feat):
                raise ValueError(f"'{ut.COL_FEAT_IMPORT}' already in 'df_feat' columns. To override, set 'drop=True'.")
        else:
            existing_impact_cols = [x for x in list(df_feat) if ut.COL_FEAT_IMPACT in x]
            if len(existing_impact_cols) > 0:
                raise ValueError(f"Some '{ut.COL_FEAT_IMPACT}' columns exist already in 'df_feat'. "
                                 f"To override, set 'drop=True'.\n These columns comprise: {existing_impact_cols}")


# II Main Functions
class ShapExplainer:
    """
    SHAP Explainer class: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimate
    for feature impact and importance.

    `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ is an explainable Artificial Intelligence (AI) framework.
            - SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
        - SHAP values represent a feature's responsibility for a change in the model output.

    Attributes
    ----------


    """
    def __init__(self,
                 explainer_class: Callable = None,
                 explainer_kwargs: Optional[dict] = None,
                 list_model_classes: List[Type[Union[ClassifierMixin, BaseEstimator]]] = None,
                 list_model_kwargs: Optional[List[dict]] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        explainer_class : model, default=None
            The `SHAP TreeExplainer model <https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html>_`
        list_model_classes : list of Type[ClassifierMixin or BaseEstimator], default=[RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
            A list of tree-based model classes to be used for feature importance analysis.
        list_model_kwargs : list of dict, optional
            A list of dictionaries containing keyword arguments for each model in `list_model_classes`.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All attributes are set during fitting via the :meth:`TreeModel.fit` method and can be directly accessed.
        * The `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ package should be installed by the user
          to provide its ``TreeExplainer`` model.
                Notes


        See Also
        --------
        * `SHAP TreeExplainer model <https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html>_` for
          details on tree-based explainer models in the SHAP package.
        * :class:`sklearn.ensemble.RandomForestClassifier` for Random Forest model.
        * :class:`sklearn.ensemble.ExtraTreesClassifier` for Extra Trees model.
        * :class:`sklearn.ensemble.GradientBoostingClassifier` for Gradient Boosting model.

        Examples
        --------
        .. include:: examples/shap_explainer.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Tree Explainer parameters
        ut.check_mode_class(model_class=explainer_class)
        if explainer_kwargs is None:
            explainer_kwargs = dict(model_output="probability")
        ut.check_model_kwargs(model_class=explainer_class, model_kwargs=explainer_kwargs,
                              param_to_check="model_output")
        # Model parameters
        if list_model_classes is None:
            list_model_classes = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
        elif not isinstance(list_model_classes, list):
            list_model_classes = [list_model_classes]  # Single models are possible as well (but not recommender)
        list_model_classes = ut.check_list_like(name="list_model_classes", val=list_model_classes, accept_none=False,
                                                min_len=1)
        ut.check_list_like(name="list_model_kwargs", val=list_model_kwargs, accept_none=True)
        if list_model_kwargs is None:
            list_model_kwargs = [{} for _ in list_model_classes]
        check_match_list_model_classes_kwargs(list_model_classes=list_model_classes,
                                              list_model_kwargs=list_model_kwargs)
        _list_model_kwargs = []
        for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs):
            ut.check_mode_class(model_class=model_class)
            model_kwargs = ut.check_model_kwargs(model_class=model_class, model_kwargs=model_kwargs,
                                                 attribute_to_check="feature_importances_", random_state=random_state)
            _list_model_kwargs.append(model_kwargs)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._explainer_class = explainer_class
        self._explainer_kwargs = explainer_kwargs
        self._list_model_classes = list_model_classes
        self._list_model_kwargs = _list_model_kwargs
        # Output parameters (set during model fitting)
        self.shap_values_: Optional[ut.ArrayLike2D] = None
        self.exp_value_: Optional[int] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            is_selected: ut.ArrayLike2D = None,
            fuzzy_labeling: bool = False,
            n_rounds: int = 5,
            ):
        """
        Obtain SHAP values aggregated across tree-based models and training rounds

        Examples
        --------
        .. include:: examples/se_fit.rst
        """
        # Check input
        # Compute shap values
        shap_values, exp_val = monte_carlo_shap_estimation(X, labels=labels,
                                                           list_model_classes=self._list_model_classes,
                                                           list_model_kwargs=self._list_model_kwargs,
                                                           explainer_class=self._explainer_class,
                                                           explainer_kwargs=self._explainer_kwargs,
                                                           is_selected=is_selected,
                                                           fuzzy_labeling=fuzzy_labeling)
        self.shap_values_ = shap_values
        self.exp_value_ = exp_val
        return self

    def eval(self, shap_values=None, is_selected=None):
        """
        Evaluate convergence of Monte Carlo estimates of SHAP values depending on number of rounds
        """


    def add_feat_impact(self,
                        df_feat: pd.DataFrame = None,
                        drop: bool = False,
                        pos: Union[int, List[int], None] = None,
                        names: Optional[Union[str, List[str]]] = None,
                        normalize: bool = True,
                        group_average: bool = False,
                        shap_feat_importance: bool = False,
                        ):
        """
        Compute SHAP feature impact (or importance) from SHAP values and add to DataFrame.

        The different scenarios for computing the feature impact are possible:

            a) For a single sample, returning its feature impact.
            b) For multiple samples, returning each sample's feature impact.
            c) For a group of samples, returning the group average feature impact.

        The respective feature impact column(s) is/are included as ``feat_impact+name(s)``.
        The shap explainer-based feature importance column is included as ``feat_importance``.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        drop : bool, default=False
            If ``True``, allow dropping of already existing ``feat_impact`` and ``feat_impact_std`` columns
            from ``df_feat`` before inserting.
        pos : int, list of int, or None
            Position index/indices for the sample(s) in ``shap_values_``.
            If ``None``, the impact for each sample will be returned.
        names: str or list of str, optional
            Name of the sample or group. Used for naming the new columns in `df_feat`.
        normalize : bool, default=True
            Whether to normalize the feature impact to percentage.
        group_average : bool, default=False
            Whether to compute the average of samples given by ``pos``.
        shap_feat_importance : bool, default=False
            If ``True``, the feature importance (i.e., absolute average shap values) will be included in ``df_feat``
            instead of the feature impact.

        Returns
        -------
        df_feat: DataFrame, shape (n_features, n_feature_info+n)
            Feature DataFrame including feature impact. If feature impact for multiple samples is computed,
            n=number of samples; n=1, otherwise.

        Notes
        -----
        - SHAP values represent a feature's responsibility for a change in the model output.
        - Missing values are accepted in SHAP values.

        Examples
        --------
        .. include:: examples/sm_add_feat_impact.rst
        """
        # Check input
        n_samples, n_features = self.shap_values_.shape
        ut.check_df_feat(df_feat=df_feat)
        ut.check_bool(name="drop", val=drop)
        pos = check_pos(pos=pos, n_samples=n_samples)
        check_name(name=names)
        pos, names = check_match_pos_name(pos=pos, name=names)
        ut.check_bool(name="normalize", val=normalize)
        ut.check_bool(name="group_average", val=group_average)
        ut.check_bool(name="shap_feat_importance", val=shap_feat_importance)
        check_match_pos_group_average(pos=pos, group_average=group_average)
        check_match_df_feat_shap_values(df_feat=df_feat, drop=drop, shap_values=self.shap_values_,
                                        shap_feat_importance=shap_feat_importance)
        # Compute feature importance
        if shap_feat_importance:
            feat_importance = comp_shap_feature_importance(shap_values=self.shap_values_, normalize=normalize)
            df_feat = insert_shap_feature_importance(df_feat=df_feat, feat_importance=feat_importance, drop=drop)
        # Compute feature impact
        feat_impact = comp_shap_feature_impact(self.shap_values_, pos=pos,
                                               group_average=group_average,
                                               normalize=normalize)
        df_feat = insert_shap_feature_impact(df_feat=df_feat, feat_impact=feat_impact,
                                             pos=pos, names=names, drop=drop)
        return df_feat
