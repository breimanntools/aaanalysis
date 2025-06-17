"""
This is a script for the frontend of the ShapModel class used to obtain Mote Carlo estimates of feature impact.
"""
from typing import Optional, List, Type, Union, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import warnings
import shap

import aaanalysis.utils as ut
from ._backend.check_models import (check_match_labels_X,
                                    check_match_X_is_selected)
from ._backend.shap_model.shap_model_fit import monte_carlo_shap_estimation
from ._backend.shap_model.sm_add_feat_impact import (comp_shap_feature_importance,
                                                     insert_shap_feature_importance,
                                                     comp_shap_feature_impact,
                                                     insert_shap_feature_impact)
from ._backend.shap_model.sm_add_sample_mean_dif import add_sample_mean_dif_


# I Helper Functions
# Check init
def check_shap_model(explainer_class=None, explainer_kwargs=None):
    """Check if explainer class is a valid shap explainer"""
    ut.check_mode_class(model_class=explainer_class)
    ut.check_dict(name="explainer_kwargs", val=explainer_kwargs, accept_none=True)
    list_valid_explainers = [shap.TreeExplainer, shap.LinearExplainer, shap.KernelExplainer,
                             shap.DeepExplainer, shap.GradientExplainer]
    names_valid_explainers = [x.__name__ for x in list_valid_explainers]
    if not (isinstance(explainer_class, type) and explainer_class in list_valid_explainers):
        raise ValueError(f"'{explainer_class.__name__}' is not a valid 'explainer_class'. "
                         f"Chose from the following: {names_valid_explainers}")
    # Check if explainer has shap_values method
    explainer_kwargs = ut.check_model_kwargs(model_class=explainer_class,
                                             model_kwargs=explainer_kwargs,
                                             name_model_class="explainer_class",
                                             method_to_check="shap_values")
    return explainer_kwargs


def check_match_class_explainer_and_models(explainer_class=None, explainer_kwargs=None, list_model_classes=None):
    """Check if each model in list_model_class is compatible with the shap explainer class"""
    dummy_data = np.array([[0, 1], [1, 0]])  # Minimal dummy data to initialize explainers
    dummy_label = [0, 1]
    for model_class in list_model_classes:
        # Check model compatability
        try:
            # Fit the dummy model
            model = model_class().fit(dummy_data, dummy_label)
            # Determine the correct input for the explainer
            if explainer_class.__name__ in ['KernelExplainer', 'OtherExplainerNeedingPredict']:
                model_input = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            else:
                model_input = model
            # Attempt to create the explainer with the appropriate input
            explainer = explainer_class(model_input, dummy_data, **{})
        except Exception as e:
            str_error = (f"The SHAP explainer '{explainer_class.__name__}' is not compatible with "
                         f"the model '{model_class.__name__}'.\nSHAP message:\n\t{e}")
            raise ValueError(str_error)
        if explainer_kwargs is not None:
            try:
                explainer = explainer_class(model_input, dummy_data, **explainer_kwargs)
            except Exception as e:
                str_error = (f"The SHAP explainer '{explainer_class.__name__}' has invalid 'explainer_kwargs': {explainer_class}"
                             f"\nSHAP message:\n\t{e}")
                raise ValueError(str_error)


# Check functions for fit method
def check_shap_values(shap_values=None):
    """Check if shap values are properly set"""
    if shap_values is None:
        raise ValueError("'shape_values' are None. Use 'ShapModel().fit()' to compute them.")
    _ = ut.check_array_like(name="shap_values", val=shap_values, dtype="numeric")


def check_match_labels_X_fuzzy_labeling(labels=None, X=None, fuzzy_labeling=False):
    """Check if labels are binary classification task labels or apply to fuzzy_labeling [0-1]"""
    if not fuzzy_labeling:
        labels = check_match_labels_X(labels=labels, X=X)
        return labels
    n_samples = X.shape[0]
    # Accept float if fuzzy_labeling is True
    labels = ut.check_labels(labels=labels, len_requiered=n_samples, accept_float=fuzzy_labeling)
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"'labels' should contain at least 2 unique labels ({unique_labels})")
    # The higher label is considered as the positive (test) class
    wrong_labels = [x for x in labels if not 0 <= x <= 1]
    if len(wrong_labels) != 0:
        raise ValueError(f"'labels' should contain only values between 0 and 1. Wrong labels are: {wrong_labels}")
    return labels


def check_is_selected(is_selected=None, n_feat=None):
    """Check is_selected and set if None"""

    if is_selected is None:
        is_selected = np.ones((1, n_feat), dtype=bool)
    else:
        is_selected = ut.check_array_like(name="is_selected_feature", val=is_selected, accept_none=False)
        is_selected = is_selected.astype(bool)
        is_selected = ut.check_array_like(name="is_selected_feature", val=is_selected, accept_none=False,
                                          expected_dim=2, dtype="bool")
    return is_selected


def check_match_labels_fuzzy_labeling(labels=None, fuzzy_labeling=False, verbose=True):
    """Check if only on label is fuzzy labeled and that the remaining sample balanced (best training scenario)"""
    if not fuzzy_labeling:
        return  # Skip check if fuzzy labeling is not enabled
    # Check for one fuzzy label and balance among other labels
    n_fuzzy_labels = len([label for label in labels if label not in [0, 1]])
    if n_fuzzy_labels != 1 and verbose:
        warnings.warn(f"Optimal training with fuzzy labeling requires exactly one fuzzy label, but {n_fuzzy_labels} were given.")
    non_fuzzy_labels = [label for label in labels if label in [0, 1]]
    n_pos = non_fuzzy_labels.count(1)
    n_neg = non_fuzzy_labels.count(0)
    if n_pos != n_neg and verbose:
        warnings.warn(f"For optimal training, non-fuzzy labels should be balanced (equal number of 0s and 1s). "
                      f"\n The 'labels' contain {n_pos} positive (1) and {n_neg} negative (0) samples.")


def check_match_labels_target_class_labels(label_target_class=None, labels=None):
    """Check if class index matches to classes (unique labels) in labels"""
    label_classes = sorted(list(dict.fromkeys([int(x) for x in labels if x == int(x)])))
    if label_target_class not in label_classes:
        raise ValueError(f"'label_target_class' ({label_target_class}) should be from 'labels' classes: {label_classes}")


def check_match_n_background_data_X(n_background_data=None, X=None):
    """Check if n_background_data not >= n_samples"""
    if n_background_data is None:
        return None # Skip test
    n_samples, n_features = X.shape
    if n_background_data >= n_samples:
        raise ValueError(f"'n_background_data' ({n_background_data}) must be < 'n_samples' ({n_samples}) from 'X'.")


# Check function for add_feat_impact method
def check_sample_positions(sample_positions=None, n_samples=None):
    """Check if sample_positions is int or list of ints"""
    str_add = "'sample_position' should be integer or list of integers indicating sample indices."
    if sample_positions is None:
        # Create sample_positions list for all samples
        sample_positions = list(range(n_samples))
    args = dict(min_val=0, max_val=n_samples-1, just_int=True, str_add=str_add)
    if isinstance(sample_positions, list) or isinstance(sample_positions, np.ndarray):
        sample_positions = ut.check_list_like(name="sample_positions", val=sample_positions,
                                              check_all_non_neg_int=True, str_add=str_add)
        for i in sample_positions:
            ut.check_number_range(name=f"sample_positions: {i}", val=i, **args)
    else:
        ut.check_number_range(name=f"sample_positions", val=sample_positions, **args)
    return sample_positions


def check_names(names=None):
    """Check if name is str ror list of str"""
    if names is None:
        return None     # Skip test
    str_add = f"'name' should be string or list of strings, but following was given: {names}"
    if isinstance(names, list):
        for i in names:
            ut.check_str(name=f"names: {i}", val=i, str_add=str_add)
        duplicated_names = list(set([x for x in names if names.count(x) > 1]))
        if len(duplicated_names) > 0:
            raise ValueError(f"'names' should not contain duplicated names: {duplicated_names}")
    else:
        ut.check_str(name=f"names", val=names, str_add=str_add)


def check_match_sample_positions_names(sample_positions=None, names=None, group_average=False):
    """Check if length of sample_positions and names matches"""
    # Group scenario
    if group_average:
        if not isinstance(sample_positions, list):
            raise ValueError(f"For 'group_average', 'sample_positions' ({sample_positions}) must be a list of integers.")
        if names is None:
            names = "Group"  # Default group names
        elif isinstance(names, str):
            names = names  # Ensure names is a list
        else:
            raise ValueError(f"For 'group_average', 'names' ({names}) must be a single string or None.")
    # Single sample or multiple samples scenarios
    else:
        if names is not None:
            if isinstance(sample_positions, int) and isinstance(names, str):
                sample_positions = [sample_positions]  # Convert to list for consistency
                names = [names]
            elif isinstance(sample_positions, list) and isinstance(names, list):
                if len(sample_positions) != len(names):
                    raise ValueError(f"Length of 'sample_positions' (n={len(sample_positions)}) and 'names' (n={len(names)}) must be equal for multiple samples.")
            else:
                raise ValueError("Mismatch: 'sample_positions' should be int and 'names' str for a single sample, and both lists for multiple samples."
                                 f"\n 'sample_positions': {sample_positions}. \n 'names': {names}")
        # Generate default names if names is None
        else:
            if isinstance(sample_positions, list):
                names = [f"Protein{p}" for p in sample_positions]
            else:
                names = [f"Protein{sample_positions}"]
                sample_positions = [sample_positions]
    return sample_positions, names


def check_match_sample_positions_group_average(sample_positions=None, group_average=False):
    """Check if group_average only True if sample_positions is list"""
    if group_average:
        sample_positions = ut.check_list_like(name="sample_positions", val=sample_positions)
        if len(sample_positions) == 1:
            raise ValueError


def check_match_df_feat_shap_values(df_feat=None, shap_values=None, drop=False, shap_feat_importance=False):
    """Check if df_feat matches with importance values"""
    if shap_values is None:
        raise ValueError(f"'shap_values' is None. Please fit ShapModel before adding feature impact.")
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
class ShapModel:
    """
    SHAP Model class: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimates
    for feature impact [Breimann25a]_.

    `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ is an explainable Artificial Intelligence (AI) framework
    and game-theoretic approach to explain the output of any machine learning model using SHAP values. These SHAP values
    represent a feature's responsibility for a change in the model output to increase of decrease a sample prediction
    score due to the positive or negative impact of its features, respectively.

    Attributes
    ----------
    shap_values : array-like, shape(n_samples, n_features)
        2D array with Monte Carlo estimates of SHAP values obtained by SHAP explainer models averaged across all rounds,
        feature selections, and trained models from `list_model_classes`.
    exp_value : int
        Expected value for explaining the model output obtained by SHAP explainer model  averaged across all rounds,
        feature selections, and trained models from `list_model_classes`. Typically, 0.5 for binary classification
        and balanced dataset.

    """
    def __init__(self,
                 explainer_class: Callable = shap.TreeExplainer,
                 explainer_kwargs: Optional[dict] = None,
                 list_model_classes: List[Type[Union[BaseEstimator]]] = None,
                 list_model_kwargs: Optional[List[dict]] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        explainer_class : model, default=TreeExplainer
            The `SHAP Explainer model <https://shap.readthedocs.io/en/latest/api.html#explainers>`_.
            Must be one of the following: :class:`shap.TreeExplainer`, :class:`shap.LinearExplainer`,
            :class:`shap.KernelExplainer`, :class:`shap.DeepExplainer`, :class:`shap.GradientExplainer`.
        explainer_kwargs : dict, default={model_output='probability'}
            Keyword arguments for the explainer class model.
        list_model_classes : list of Type[BaseEstimator], default=[RandomForestClassifier, ExtraTreesClassifier]
            A list of prediction model classes used to obtain SHAP values.
        list_model_kwargs : list of dict, optional
            A list of dictionaries containing keyword arguments for each model in `list_model_classes`.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All attributes are set during fitting via the :meth:`ShapModel.fit` method and can be directly accessed.
        * The Explainer models should be provided from the `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_
          package
        * SHAP model fitting messages appear in red and are not controlled by ``verbose``, unlike AAanalysis progress
          messages in blue.
        * The selection of the SHAP explainer must align with the machine learning models used.
          Following explainer model types are allowed:

          - :class:`shap.TreeExplainer`: Ideal for tree-based models (by default, random forests and extra trees; further
            recommended are XGBoost and CatBoost). Efficient in computing SHAP values by leveraging the tree structure.
          - :class:`shap.LinearExplainer`: Suited for linear models (e.g., logistic regression, linear regression).
            Computes SHAP values directly from model coefficients.
          - :class:`shap.KernelExplainer`: Model-agnostic, works with any model type. Uses weighted linear regression to
            approximate SHAP values. Versatile but less computationally efficient, which can be increased by a background dataset.
          - :class:`shap.DeepExplainer`: Designed for deep learning models (e.g., models from TensorFlow, Keras).
            Approximates SHAP values by analyzing neuron groups, suitable for complex networks.
          - :class:`shap.GradientExplainer`: Also for deep learning, but uses expected gradients.
            Effective for models with differentiable components.

          Proper explainer choice is key for accurate model explanations.

        * By default, :class:`shap.TreeExplainer` is used with random forest, extra trees, and gradient boosting models.

        See Also
        --------
        * :class:`sklearn.ensemble.RandomForestClassifier` for random forest model.
        * :class:`sklearn.ensemble.ExtraTreesClassifier` for extra trees model.
        * :meth:`ShapModel.add_feat_impact` for details on feature impact and SHAP value-based feature importance.

        Warnings
        --------
        * This class requires `SHAP`, which is automatically installed via `pip install aaanalysis[pro]`.

        Examples
        --------
        .. include:: examples/shap_model.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Check SHAP explainer model
        if explainer_class is None:
            explainer_class = shap.TreeExplainer
            explainer_kwargs = explainer_kwargs or dict(model_output="probability")
        explainer_kwargs = check_shap_model(explainer_class=explainer_class, explainer_kwargs=explainer_kwargs)
        # Check model parameters
        if list_model_classes is None:
            list_model_classes = [RandomForestClassifier, ExtraTreesClassifier]
        elif not isinstance(list_model_classes, list):
            # If single model is used (not recommended)
            list_model_classes = [list_model_classes]
        list_model_classes = ut.check_list_like(name="list_model_classes",
                                                val=list_model_classes,
                                                accept_none=False,
                                                min_len=1)
        list_model_kwargs = ut.check_list_like(name="list_model_kwargs", val=list_model_kwargs, accept_none=True)
        if list_model_kwargs is None:
            list_model_kwargs = [{} for _ in list_model_classes]
        # Check matching of model parameters
        ut.check_match_list_model_classes_kwargs(list_model_classes=list_model_classes,
                                                 list_model_kwargs=list_model_kwargs)
        _list_model_kwargs = []
        for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs):
            ut.check_mode_class(model_class=model_class)
            model_kwargs = ut.check_model_kwargs(model_class=model_class,
                                                 model_kwargs=model_kwargs,
                                                 random_state=random_state)
            _list_model_kwargs.append(model_kwargs)
        check_match_class_explainer_and_models(explainer_class=explainer_class,
                                               explainer_kwargs=explainer_kwargs,
                                               list_model_classes=list_model_classes)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._explainer_class = explainer_class
        self._explainer_kwargs = explainer_kwargs
        self._list_model_classes = list_model_classes
        self._list_model_kwargs = _list_model_kwargs
        # Output parameters (set during model fitting)
        self.shap_values: Optional[ut.ArrayLike2D] = None
        self.exp_value: Optional[int] = None

    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            label_target_class: int = 1,
            n_rounds: int = 5,
            is_selected: ut.ArrayLike2D = None,
            fuzzy_labeling: bool = False,
            n_background_data: Optional[int] = None,
            ) -> "ShapModel":
        """
        Obtain SHAP values aggregated across prediction models and training rounds.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples)
            Class labels for samples in ``X`` (typically, 1=positive, 0=negative).
        label_target_class : int, default=1
            The label of the class for which SHAP values are computed in a classification tasks.
            For binary classification, '0' represents the negative class and '1' the positive class.
        n_rounds : int, default=5
            The number of rounds (>=1) to fit the models and obtain the SHAP values by explainer.
        is_selected : array-like, shape (n_selection_round, n_features)
            2D boolean arrays indicating different feature selections.
        fuzzy_labeling : bool, default=False
            If ``True``, fuzzy labeling is applied to approximate SHAP values for samples with uncertain/partial
            memberships (e.g., between >0 and <1 for binary classification scenarios).
        n_background_data : None or int, optional
            The number samples (< 'n_samples') in the background dataset used for the `KernelExplainer`` to reduce
            computation time. The dataset is obtained by k-means clustering. If ``None``, the full dataset 'X' is used.

        Returns
        -------
        ShapModel
            The fitted ShapModel model instance.

        Notes
        -----
        **Fuzzy Labeling**

        * Aim: Compute SHAP value for datasets with uncertain or ambiguous labels. Especially useful to explain newly
          predicted samples, where class label is set to the respective prediction probability.
        * Approach: Uses probabilistic labels to represent degrees of membership.
        * Idea: Adjusts label thresholds dynamically in Monte Carlo estimation to better represent label uncertainties.
        * Background: Inspired by fuzzy logic, replacing binary true/false with degrees of truth.

        See Also
        --------
        * [Breimann25a]_ introduces fuzzy labeling to compute Monte Carlo estimates of SHAP values
          for samples with not clearly defined class membership.

        Examples
        --------
        .. include:: examples/sm_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        n_samples, n_feat = X.shape
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        ut.check_bool(name="fuzzy_labeling", val=fuzzy_labeling)
        labels = check_match_labels_X_fuzzy_labeling(labels=labels, X=X, fuzzy_labeling=fuzzy_labeling)
        check_match_labels_fuzzy_labeling(labels=labels, fuzzy_labeling=fuzzy_labeling, verbose=self._verbose)
        ut.check_number_range(name="label_target_class", val=label_target_class, min_val=0, just_int=True, accept_none=True)
        check_match_labels_target_class_labels(label_target_class=label_target_class, labels=labels)
        is_selected = check_is_selected(is_selected=is_selected, n_feat=n_feat)
        check_match_X_is_selected(X=X, is_selected=is_selected)
        ut.check_number_range(name="n_rounds", val=n_rounds, min_val=1, just_int=True)
        ut.check_number_range(name="n_background_data", val=n_background_data, min_val=1, just_int=True, accept_none=True)
        check_match_n_background_data_X(n_background_data=n_background_data, X=X)
        # Compute SHAP values
        shap_values, exp_val = monte_carlo_shap_estimation(X, labels=labels,
                                                           list_model_classes=self._list_model_classes,
                                                           list_model_kwargs=self._list_model_kwargs,
                                                           explainer_class=self._explainer_class,
                                                           explainer_kwargs=self._explainer_kwargs,
                                                           is_selected=is_selected,
                                                           fuzzy_labeling=fuzzy_labeling,
                                                           n_rounds=n_rounds,
                                                           verbose=self._verbose,
                                                           label_target_class=label_target_class,
                                                           n_background_data=n_background_data)
        self.shap_values = shap_values
        self.exp_value = exp_val
        return self

    def eval(self, shap_values=None, is_selected=None):
        """
        UNDER CONSTRUCTION - Evaluate convergence of Monte Carlo estimates of SHAP values depending on number of rounds
        """

    def add_feat_impact(self,
                        df_feat: pd.DataFrame = None,
                        drop: bool = False,
                        sample_positions: Union[int, List[int], None] = None,
                        names: Optional[Union[str, List[str]]] = None,
                        normalize: bool = True,
                        group_average: bool = False,
                        shap_feat_importance: bool = False,
                        ) -> pd.DataFrame:
        """
        Compute SHAP feature impact (or importance) from SHAP values and add to the feature DataFrame.

        Three different scenarios for computing the feature impact are possible:

            a) **Single sample**: Computes the feature impact for a selected sample.
            b) **Multiple samples**: Computes the feature impact for multiple samples (all by default).
            c) **Group of samples**: Computes the average feature impact for a group of samples (+ standard deviation).

        The calculated feature impacts are added to ``df_feat`` as new columns named ``feat_impact_'name(s)'``,
        corresponding to each sample or group. Additionally, the SHAP value-based feature importance can be included
        as ``feat_importance`` column.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        drop : bool, default=False
            If ``True``, allow dropping of already existing feature impact and feature importance columns from
            ``df_feat`` before inserting.
        sample_positions : int, list of int, or None
            Position index/indices for the sample(s) in ``shap_values``.
            If ``None``, the impact for each sample will be returned.
        names: str or list of str, optional
            Unique name(s) used for the feature impact columns. When provided, they should align with
            ``sample_positions`` as follows:

            - **Single sample**: ``name`` as string and ``sample_positions`` as integer.
            - **Multiple samples**: ``name`` as list of string and ``sample_positions`` as corresponding list of integers.
            - **Group**: ``name`` as string and ``sample_positions`` as list of integers, each indicating a group sample.

            If ``sample_positions`` is ``None`` (all samples are considered), ``name`` must be list with names for each sample.
        normalize : bool, default=True
            If ``True``, normalize the feature impact to percentage.
        group_average : bool, default=False
            If ``True``, compute the average of samples given by ``sample_positions``.
        shap_feat_importance : bool, default=False
            If ``True``, include feature importance (i.e., absolute average SHAP values) instead of impact to ``df_feat``.

        Returns
        -------
        df_feat: pd.DataFrame, shape (n_features, n_feature_info+n)
            Feature DataFrame including feature impact. If the feature impact is computed for multiple samples,
            n=number of samples; n=1, otherwise.

        Notes
        -----
        **Feature impact (sample-level)**:
        The feature impact quantifies the positive or negative contribution of a feature to increase or decrease the model
        output for a specific sample (typically, prediction score). For each sample, the impact of an individual feature
        is represented by its corresponding SHAP value. These values are normalized such that the sum of their absolute
        values equals 100%.

        **Feature impact (group-level)**:
        The feature impact calculated for individual samples can be averaged to determine the feature impact for a group.
        This reflects how features influence the model's output on average within that group.

        **Feature importance (SHAP value-based)**:
        The average of the feature impact across all samples is termed as shap value-based 'feature importance'.
        This quantifies the overall contribution of each feature across the entire dataset.

        Warnings
        --------
        * If ``group_average=True``, warning when the standard deviation of a feature's impact significantly exceeds
          its mean impact, this may indicate an unreliable grouping.

        Examples
        --------
        .. include:: examples/sm_add_feat_impact.rst
        """
        # Check input
        check_shap_values(shap_values=self.shap_values)
        n_samples, n_features = self.shap_values.shape
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_bool(name="drop", val=drop)
        check_names(names=names)
        ut.check_bool(name="group_average", val=group_average)
        sample_positions = check_sample_positions(sample_positions=sample_positions, n_samples=n_samples)
        sample_positions, names = check_match_sample_positions_names(sample_positions=sample_positions, names=names,
                                                                     group_average=group_average)
        ut.check_bool(name="normalize", val=normalize)
        ut.check_bool(name="shap_feat_importance", val=shap_feat_importance)
        check_match_sample_positions_group_average(sample_positions=sample_positions, group_average=group_average)
        check_match_df_feat_shap_values(df_feat=df_feat, drop=drop, shap_values=self.shap_values,
                                        shap_feat_importance=shap_feat_importance)
        # Compute feature importance
        if shap_feat_importance:
            feat_importance = comp_shap_feature_importance(shap_values=self.shap_values,
                                                           normalize=normalize)
            df_feat = insert_shap_feature_importance(df_feat=df_feat,
                                                     feat_importance=feat_importance,
                                                     drop=drop)
        # Compute feature impact
        else:
            feat_impact = comp_shap_feature_impact(self.shap_values,
                                                   normalize=normalize,
                                                   sample_positions=sample_positions,
                                                   verbose=self._verbose,
                                                   group_average=group_average)
            df_feat = insert_shap_feature_impact(df_feat=df_feat,
                                                 feat_impact=feat_impact,
                                                 names=names,
                                                 drop=drop,
                                                 group_average=group_average)

        return df_feat

    @staticmethod
    def add_sample_mean_dif(X: ut.ArrayLike2D,
                            labels: ut.ArrayLike1D = None,
                            label_ref: int = 0,
                            df_feat: pd.DataFrame = None,
                            drop: bool = False,
                            sample_positions: Union[int, List[int], None] = None,
                            names: Optional[Union[str, List[str]]] = None,
                            group_average: bool = False,
                            ) -> pd.DataFrame:
        """
        Compute the feature value difference between selected samples and a reference group average.

        Three different scenarios for computing the difference with the reference group average (MEAN_REF) are possible:

            a) **Single sample**: Computes the difference for a selected sample and MEAN_REF.
            b) **Multiple samples**: Computes differences for multiple selected samples (all by default) individually against MEAN_REF.
            c) **Group of samples**: Computes the difference between the average of a selected group of samples and MEAN_REF.

        The calculated differences are added to ``df_feat`` as new columns named ``mean_dif_'name(s)'``,
        corresponding to each sample or group.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples)
            Class labels for samples in ``X`` (typically, 1=positive, 0=negative).
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        drop : bool, default=False
            If ``True``, allow dropping of already existing sample specific mean difference columns from
            ``df_feat`` before inserting.
        sample_positions : int, list of int, or None
            Position index/indices for the sample(s) in ``shap_values``.
            If ``None``, the impact for each sample will be returned.
        names: str or list of str, optional
            Unique name(s) used for the feature value differences columns. When provided, they should align with
            ``sample_positions`` as follows:

            - **Single sample**: ``name`` as string and ``sample_positions`` as integer.
            - **Multiple samples**: ``name`` as list of string and ``sample_positions`` as corresponding list of integers.
            - **Group**: ``name`` as string and ``sample_positions`` as list of integers, each indicating a group sample.

            If ``sample_positions`` is ``None`` (all samples are considered), ``name`` must be list with names for each sample.
        group_average : bool, default=False
            If ``True``, compute the average of samples given by ``sample_positions``.

        Returns
        -------
        df_feat: pd.DataFrame, shape (n_features, n_feature_info+n)
            Feature DataFrame including feature value difference. If the feature value difference is computed for multiple
            samples, n=number of samples; n=1, otherwise.

        Examples
        --------
        .. include:: examples/sm_add_sample_mean_dif.rst
        """
        # Check input
        X = ut.check_X(X=X)
        n_samples, n_feat = X.shape
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_ref],
                                 len_requiered=n_samples, allow_other_vals=True,
                                 accept_float=True) # Accept fuzzy labeling by default
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_bool(name="drop", val=drop)
        check_names(names=names)
        ut.check_bool(name="group_average", val=group_average)
        sample_positions = check_sample_positions(sample_positions=sample_positions, n_samples=n_samples)
        sample_positions, names = check_match_sample_positions_names(sample_positions=sample_positions, names=names,
                                                                     group_average=group_average)
        # Compute differences
        df_feat = add_sample_mean_dif_(X, labels=labels, label_ref=label_ref,
                                       df_feat=df_feat, drop=drop,
                                       sample_positions=sample_positions, names=names,
                                       group_average=group_average)
        return df_feat
