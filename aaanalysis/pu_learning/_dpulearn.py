"""
This is a script for the frontend of the dPULearn class, used for deterministic Positive-Unlabeled (PU) Learning.
"""
from typing import Optional, Literal, Dict, Union, List, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import aaanalysis.utils as ut
from ._backend.dpulearn.dpul_fit import (get_neg_via_distance, get_neg_via_pca)
from ._backend.dpulearn.dpul_eval import eval_identified_negatives
from ._backend.dpulearn.dpul_compare_sets_neg import compare_sets_negatives_

# Settings
LIST_METRICS = ['euclidean', 'manhattan', 'cosine']


# I Helper Functions
# Check functions
def check_metric(metric=None):
    """Validate provided metric"""
    if metric is not None and metric not in LIST_METRICS:
        raise ValueError(f"'metric' ({metric}) should be None or one of following: {LIST_METRICS}")


def check_n_unl_to_neg(labels=None, n_unl_to_neg=None, label_unl=None):
    """Validate that there are enough unlabeled samples in the dataset."""
    n_unl = np.sum(labels == label_unl)
    if n_unl < n_unl_to_neg:
        raise ValueError(f"Number of unlabeled labels ({n_unl}) must be higher than 'n_unl_to_neg' ({n_unl_to_neg})")


def check_n_components(n_components=1):
    """Check if n_components valid for sklearn PCA object"""
    try:
        # Check number of PCs
        if type(n_components) is int:
            ut.check_number_range(name="n_components", val=n_components, min_val=1, just_int=True)
        # Check percentage of covered explained variance
        else:
            ut.check_number_range(name="n_components", val=n_components, min_val=0, max_val=1.0, just_int=False,
                                  exclusive_limits=True)
    except ValueError:
        raise ValueError(f"'n_components' ({n_components}) should be either "
                         f"\n  an integer >= 1 (number of principal components) or"
                         f"\n  a float with 0.0 < 'n_components' < 1.0 (percentage of covered variance)")


def check_match_X_n_components(X=None, n_components=1):
    """Check if n_components matches to dimensions of X"""
    n_samples, n_features = X.shape
    if min(n_features, n_samples) <= n_components:
        raise ValueError(f"'n_components' ({n_components}) should be < min(n_features, n_samples) from 'X' ({n_features})")


def check_match_list_labels_df_seq(list_labels=None, df_seq=None):
    """Check if length of labels in list_labels and df_seq matches"""
    if df_seq is None:
        return None # Skip check
    n_samples = len(list_labels[0])
    if n_samples != len(df_seq):
        raise ValueError(f"Number of samples (n={n_samples}) in 'list_labels' does not match with "
                         f"samples in 'df_seq' (n={len(df_seq)})")


def check_match_X_X_neg(X=None, X_neg=None):
    """Check if number of features matches in both feature matrices"""
    if X_neg is None:
        return # Skip test
    n_features = X.shape[1]
    n_features_neg =  X_neg.shape[1]
    if n_features != n_features_neg:
        raise ValueError(f"'n_features' does not match between 'X' (n={n_features}) and 'X_neg' (n={n_features_neg})")


# II Main Functions
class dPULearn:
    """
    Deterministic Positive-Unlabeled Learning (**dPULearn**) class for identifying reliable negatives from unlabeled data [Breimann25a]_.

    dPULearn offers a deterministic approach to Positive-Unlabeled (PU) learning, featuring two distinct
    identification approaches:

    - **PCA-based identification**: This is the primary method where Principal Component Analysis (PCA) is utilized
      to reduce the dimensionality of the feature space. Based on the most informative principal components (PCs),
      the model iteratively identifies reliable negatives (labeled by 0) from the set of unlabeled samples (2).
      These reliable negatives are those that are most distant from the positive samples (1) in the feature space.
    - **Distance-based identification**: As a simple alternative, reliable negatives can also be identified using
      similarity measures like ``euclidean``, ``manhattan``, or ``cosine`` distance.

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        New dataset labels of samples in ``X`` with identified negative samples labeled by 0.
    df_pu_ : pd.DataFrame, shape (n_samples, pca_features)
        A DataFrame with the PCA-transformed features of 'X' containing the following groups of columns:

        - 'selection_via': Column indicating how reliable negatives were identified (either giving the distance metric
           or the i-th PC based on which the respective sample was selected).
        - 'PCi': Value columns for the i-th principal component (PC).
        - 'PCi_abs_dif': Absolute difference columns for each PC, representing the absolute deviation of each sample
          from the mean of positives.

        For distance-based identification, 'PCi' columns are replaced with the results for the selected metric.


    """
    def __init__(self,
                 model_kwargs: Optional[dict] = None,
                 verbose: bool = True,
                 random_state: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        model_kwargs : dict, optional
            Additional keyword arguments for Principal Component Analysis (PCA) model.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All attributes are set during fitting via the :meth:`dPULearn.fit` method and can be directly accessed.
        * For a detailed discussion on Positive-Unlabeled (PU) learning, its challenges, and evaluation strategies,
          refer to the PU Learning section in the Usage Principles documentation: `usage_principles/pu_learning`.

        See Also
        --------
        * :class:`dPULearnPlot`: the respective plotting class.
        * :func:`sklearn.decomposition.PCA` for details on principal component analysis.
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Model parameters
        model_kwargs = ut.check_model_kwargs(model_class=PCA,
                                             model_kwargs=model_kwargs,
                                             random_state=random_state)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._model_kwargs = model_kwargs
        # Output parameters (will be set during model fitting)
        self.labels_ = None
        self.df_pu_ = None

    # Main method
    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            n_unl_to_neg: int = None,
            metric: Optional[Literal["euclidean", "manhattan", "cosine"]] = None,
            n_components: Union[float, int] = 0.80,
            ) -> "dPULearn":
        """
        Fit the dPULearn model to identify reliable negative samples (labeled by 0) from unlabeled samples (2)
        based on the distance to positive samples (1).

        Use the ``dPUlearn.labels_`` attribute to retrieve the output labels of samples in ``X``
        including identified negatives.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples,)
            Dataset labels of samples in ``X``. Should be either 1 (positive) or 2 (unlabeled).
        n_unl_to_neg : int, default=1
            Number of negative samples (0) to be reliably identified from unlabeled samples (2).
            Should be < n unlabeled samples.
        metric : str or None, optional
            The distance metric to use. If ``None``, PCA-based identification is performed. For distance-based
            identification one of the following measures can be selected:

            - ``euclidean``: Euclidean distance (minimum)
            - ``manhattan``: Manhattan distance (minimum)
            - ``cosine``: Cosine distance (minimum)

        n_components : int or float, default=0.80
            Number of principal components (a) or the percentage of total variance to be covered (b) when PCA is applied.

            - In case (a): it should be an integer >= 1.
            - In case (b): it should be a float with  0.0 < ``n_components`` < 1.0.

        Returns
        -------
        dPULearn
            The fitted instance of the dPULearn class, allowing direct attribute access.

        Notes
        -----
        * If a distance metric is specified, dPUlearn performs distance-based instead of PCA-based identification.
        * When selecting a distance metric for distance-based identification, consider the dimensionality of the
          feature space, determined by the ratio of the number of features (n_features) to the number of samples
          (n_samples) in `X`. In a low-dimensional space, there are fewer features than samples (n_features < n_samples),
          whereas a high-dimensional space has significantly more features than samples (n_features >> n_samples).
          The choice of metric depends on the specific application, with the following general guidelines:

          - ``euclidean``: Effective in low-dimensional spaces or when direct distances are meaningful.
          - ``manhattan``: Useful when differences along individual dimensions are important, or in the presence of outliers.
          - ``cosine``: Recommended for high-dimensional spaces (e.g., n_features >> n_samples), as it evaluates
            the direction of feature vectors between data points rather than the magnitude of their differences.

        Warnings
        --------
        * When setting ``n_components`` as a percentage of total variance (i.e., a float between 0.0 and 1.0),
          caution is needed if the explained variance per principal component (PC) is low. Selecting too many PCs
          with low explained variance may introduce noise and lead to the selection of outliers rather than true negatives.
        * To mitigate this, users can alternatively set ``n_components`` as an integer (≥1) to explicitly limit
          the number of PCs used.

        See Also
        --------
        * See `scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise>`_
          for details the three different pairwise distance measures.
        * See [Hastie09]_ for a detailed explanation on feature space and high-dimensional problems.

        Examples
        --------
        .. include:: examples/dpul_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        labels = ut.check_labels(labels=labels, vals_requiered=[1, 2], allow_other_vals=False)
        ut.check_number_range(name="n_unl_to_neg", val=n_unl_to_neg, min_val=1, just_int=True)
        check_n_unl_to_neg(labels=labels, n_unl_to_neg=n_unl_to_neg, label_unl=2)
        check_metric(metric=metric)
        check_n_components(n_components=n_components)
        ut.check_match_X_labels(X=X, labels=labels)
        check_match_X_n_components(X=X, n_components=n_components)
        # Compute average distance for threshold-based filtering (Yang et al., 2012, 2014; Nan et al. 2017)
        args = dict(X=X, labels=labels, n_unl_to_neg=n_unl_to_neg, label_neg=0)
        if metric is not None:
            new_labels, df_pu = get_neg_via_distance(**args, metric=metric)
        # Identify most far away negatives in PCA compressed feature space
        else:
            new_labels, df_pu = get_neg_via_pca(**args, n_components=n_components, **self._model_kwargs)
        # Set new labels
        self.labels_ = np.asarray(new_labels)
        self.df_pu_ = df_pu
        return self

    @staticmethod
    def eval(X: ut.ArrayLike2D,
             list_labels: ut.ArrayLike2D = None,
             names_datasets: Optional[List[str]] = None,
             X_neg: Optional[ut.ArrayLike2D] = None,
             comp_kld: bool = False,
             n_jobs: Optional[int] = None
             ) -> pd.DataFrame:
        """
        Evaluates the quality of different sets of identified negatives.

        The quality is assessed regarding two quality groups:

        - **Homogeneity** within the reliably identified negatives (0)
        - **Dissimilarity** between the reliably identified negatives and the groups
          of positive samples ('pos'), unlabeled samples ('unl'), and a ground-truth negative
          ('neg') sample group if provided by ``X_neg``

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        list_labels : array-like, shape (n_datasets, n_samples)
            List of arrays with dataset labels for samples in ``X`` obtained by the :meth:`dPULearn.fit` method.
            Label values should be either 0 (identified negative), 1 (positive) or 2 (unlabeled).
        names_datasets : list, optional
            List of dataset names corresponding to ``list_labels``.
        X_neg : array-like, shape (n_samples_neg, n_features), optional
            Feature matrix where `n_samples_neg` is the number ground-truth negative samples
            and `n_features` is the number of features. Features must correspond to ``X``.
        comp_kld : bool, default=False
            Whether to compute Kullback-Leibler Divergence (KLD) to assess the distribution alignment between
            identified negatives and other data groups. Disable (``False``) if ``X`` is sparse or has low co-variance.
        n_jobs : int, None, or -1, default=None
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for each set of identified negatives from ``list_labels``. For each set, statistical
            measures were averaged across all features.

        Notes
        -----
        ``df_eval`` includes the following columns:

        - 'name': Name of the dataset if ``names`` is provided (typically named by identification approach).
        - 'n_rel_neg': Number of identified negatives.
        - 'avg_std': Average standard deviation (STD) assessing homogeneity of identified negatives.
          Lower values indicate greater homogeneity.
        - 'avg_iqr': Average interquartile range (IQR) assessing homogeneity of identified negatives.
          Lower values suggest greater homogeneity.
        - 'avg_abs_auc_DATASET': Average absolute area under the curve (AUC) assessing the dissimilarity between the
          set of identified negatives with other groups (positives, unlabeled, ground-truth negatives).
          Separate columns are provided for each comparison. Higher values indicate greater dissimilarity.
        - 'avg_kld_DATASET': Average Kullback-Leibler Divergence (KLD) assessing the dissimilarity of distributions
          between the set of identified negatives and the other groups. Higher values indicate greater dissimilarity.
          These columns are omitted if ``kld`` is set to ``False``.

        See Also
        --------
        * :meth:`dPULearnPlot.eval`: the respective plotting method.
        * :ref:`usage_principles_pu_learning` for details on different evaluation strategies.

        Examples
        --------
        .. include:: examples/dpul_eval.rst
        """
        # Check input
        X= ut.check_X(X=X)
        X_neg = ut.check_X(X=X_neg, X_name="X_neg", accept_none=True, min_n_samples=2)
        ut.check_bool(name="comp_kld", val=comp_kld)
        list_labels = ut.check_array_like(name="list_labels", val=list_labels, ensure_2d=True, convert_2d=True)
        names_datasets = ut.check_list_like(name="names_datasets", val=names_datasets, accept_none=True, accept_str=True,
                                            check_all_str_or_convertible=True)
        ut.check_match_X_list_labels(X=X, list_labels=list_labels, check_variability=comp_kld, vals_requiered=[0])
        ut.check_match_list_labels_names_datasets(list_labels=list_labels, names_datasets=names_datasets)
        check_match_X_X_neg(X=X, X_neg=X_neg)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # Evaluation for homogeneity within negatives and alignment of distribution with other datasets
        df_eval = eval_identified_negatives(X=X, list_labels=list_labels, names_datasets=names_datasets,
                                            X_neg=X_neg, comp_kld=comp_kld, n_jobs=n_jobs)
        return df_eval

    @staticmethod
    def compare_sets_negatives(list_labels: ut.ArrayLike1D = None,
                               names_datasets: Optional[List[str]] = None,
                               df_seq: Optional[pd.DataFrame] = None,
                               remove_non_neg : bool = True,
                               return_upset_data: bool = False
                               ) -> pd.DataFrame:
        """
        Create DataFrame for comparing sets of identified negatives.

        Optionally, data format can be created for Upset Plots, which are useful for visualizing the intersection
        and unique elements across these sets.

        Parameters
        ----------
        list_labels : array-like, shape (n_datasets,)
            List of dataset labels for samples in ``X`` obtained by the :meth:`dPULearn.fit` method.
            Label values should be either 0 (identified negative), 1 (positive) or 2 (unlabeled). Must contain 0.
        names_datasets : list, optional
            List of dataset names corresponding to ``list_labels``.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
            DataFrame with sequence information for entries corresponding to 'labels' of ``list_labels``.
        remove_non_neg : bool, default=True
            If ``True``, all rows are removed that do not contain identified negatives in any provided dataset.
        return_upset_data : bool, default=False
            Whether to return a DataFrame for Upset Plot (if ``True``) or for a general comparison of sets of negatives.

        Returns
        -------
        pd.DataFrame or pd.Series
            - If ``return_upset_data=False`` (default):
              Returns a pd.DataFrame (`df_neg_comp`) that combines ``df_seq`` (if provided) with a comparison of the
              negative sets for a general analysis.
            - If ``return_upset_data=True``:
              Returns a pd.Series DataFrame (`upset_data`) formatted for generating  Upset Plots, containing group
              size information for the intersection and unique elements across the label sets.

        See Also
        --------
        * :meth:`dPULearn.fit` for details on how labels are generated.
        * :meth:`SequenceFeature.get_df_parts` for details on format of ``df_seq``.
        * Upset Plot documentation: :func:`upsetplot.plot`.

        Examples
        --------
        .. include:: examples/dpul_compare_sets_negatives.rst
        """
        # Check input
        list_labels = ut.check_array_like(name="list_labels", val=list_labels,
                                          ensure_2d=True, convert_2d=True)
        names_datasets = ut.check_list_like(name="names_datasets", val=names_datasets, accept_none=True,
                                            accept_str=True, check_all_str_or_convertible=True)
        ut.check_df_seq(df_seq=df_seq, accept_none=True)
        ut.check_bool(name="return_upset_data", val=return_upset_data)
        ut.check_match_list_labels_names_datasets(list_labels=list_labels, names_datasets=names_datasets)
        check_match_list_labels_df_seq(list_labels=list_labels, df_seq=df_seq)
        # Comparison of identified sets of negatives
        args = dict(list_labels=list_labels, names=names_datasets,
                    df_seq=df_seq, return_upset_data=return_upset_data, remove_non_neg=remove_non_neg)
        if return_upset_data:
            upset_data = compare_sets_negatives_(**args)
            return upset_data
        df_neg_comp = compare_sets_negatives_(**args)
        return df_neg_comp
