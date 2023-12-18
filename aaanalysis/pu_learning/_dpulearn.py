"""
This is a script for the frontend of the dPULearn class, used for deterministic Positive-Unlabeled (PU) Learning.
"""
from typing import Optional, Literal, Dict, Union, List, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import math

import aaanalysis.utils as ut
from ._backend.dpulearn.dpulearn_eval import (eval_homogeneity,
                                              eval_distribution_alignment,
                                              eval_distribution_alignment_X_neg)

# Settings
LIST_METRICS = ['euclidean', 'manhattan', 'cosine']
COL_SELECTION_VIA = "selection_via"


# I Helper Functions
# Check functions
def _check_metric(metric=None):
    """Validate provided metric"""
    if metric is not None and metric not in LIST_METRICS:
        raise ValueError(f"'metric' ({metric}) should be None or one of following: {LIST_METRICS}")


def _check_n_unl_to_neg(labels=None, n_unl_to_neg=None, label_unl=None):
    """Validate that there are enough unlabeled samples in the dataset."""
    n_unl = np.sum(labels == label_unl)
    if n_unl < n_unl_to_neg:
        raise ValueError(f"Number of unlabeled labels ({n_unl}) must be higher than 'n_unl_to_neg' ({n_unl_to_neg})")


def _check_n_components(n_components=1):
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


def _check_match_X_n_components(X=None, n_components=1):
    """Check if n_components matches to dimensions of X"""
    n_samples, n_features = X.shape
    if min(n_features, n_samples) <= n_components:
        raise ValueError(f"'n_components' ({n_components}) should be < min(n_features, n_samples) from 'X' ({n_features})")


# II Main Functions
# TODO into backend
def _get_neg_via_distance(X=None, labels=None, metric="euclidean", n_unl_to_neg=None,
                          label_neg=0, label_pos=1):
    """Identify distant samples from positive mean as reliable negatives based on a specified distance metric."""
    col_dif = f'{metric}_dif'
    col_dif_abs = f"{metric}_abs_dif"
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    # Compute the distances to the average value of the positive datapoints
    dif_to_pos_mean = pairwise_distances(X[mask_pos], X, metric=metric).mean(axis=0)
    abs_dif_to_pos_mean = np.abs(dif_to_pos_mean)
    # Create a DataFrame with the distances
    df_pu = pd.DataFrame({col_dif: dif_to_pos_mean, col_dif_abs: abs_dif_to_pos_mean})
    # Select negatives based on largest average distance to positives
    top_indices = df_pu[mask_unl].sort_values(by=col_dif_abs).tail(n_unl_to_neg).index
    new_labels = labels.copy()
    new_labels[top_indices] = label_neg
    # Adjust df distance
    df_pu = df_pu.round(4)
    df_pu.insert(0, COL_SELECTION_VIA, [metric if l == 0 else None for l in new_labels])
    return new_labels, df_pu


def _get_neg_via_pca(X=None, labels=None, n_components=0.8, n_unl_to_neg=None,
                     label_neg=0, label_pos=1, **pca_kwargs):
    """Identify distant samples from positive mean as reliable negatives in PCA-compressed feature spaces."""
    # Principal component analysis
    pca = PCA(n_components=n_components, **pca_kwargs)
    pca.fit(X.T)
    list_exp_var = pca.explained_variance_ratio_
    columns_pca = [f"PC{n+1} ({round(exp_var*100, 1)}%)" for n, exp_var in enumerate(list_exp_var)]
    # Determine number of negatives based on explained variance
    _list_n_neg = [math.ceil(n_unl_to_neg * x / sum(list_exp_var)) for x in list_exp_var]
    _list_n_cumsum = np.cumsum(np.array(_list_n_neg))
    list_n_neg = [n for n, cs in zip(_list_n_neg, _list_n_cumsum) if cs <= n_unl_to_neg]
    if sum(list_n_neg) != n_unl_to_neg:
        list_n_neg.append(n_unl_to_neg - sum(list_n_neg))
    columns_pca = columns_pca[:len(list_n_neg)]
    # Create df_pu based on PCA components
    df_pu = pd.DataFrame(pca.components_.T[:, :len(columns_pca)], columns=columns_pca)
    # Get mean of positive data for each component
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    pc_means = df_pu[mask_pos].mean(axis=0)
    # Select negatives based on absolute difference to mean of positives for each component
    df_pu.insert(0, COL_SELECTION_VIA, None)  # New column to store the PC information
    new_labels = labels.copy()
    for col_pc, mean_pc, n in zip(columns_pca, pc_means, list_n_neg):
        col_abs_dif = f"{col_pc}_abs_dif"
        # Calculate absolute difference to the mean for each sample in the component
        df_pu[col_abs_dif] = np.abs(df_pu[col_pc] - mean_pc)
        # Sort and take top n indices
        top_indices = df_pu[mask_unl].sort_values(by=col_abs_dif).tail(n).index
        # Update labels and masks
        new_labels[top_indices] = label_neg
        mask_unl[top_indices] = False
        # Record the PC by which the negatives are selected
        df_pu.loc[top_indices, COL_SELECTION_VIA] = col_pc.split(' ')[0]
    # Adjust df
    cols = [x for x in list(df_pu) if x != COL_SELECTION_VIA]
    df_pu[cols] = df_pu[cols].round(4)
    return new_labels, df_pu


# TODO refactor, test, document
class dPULearn:
    """
    Deterministic Positive-Unlabeled (dPULearn) model introduced in [Breimann24c]_.

    The dPULearn model offers a deterministic approach to Positive-Unlabeled (PU) learning, featuring two distinct
    identification approaches:

    - **PCA-based identification**: This is the primary method where Principal Component Analysis (PCA) is utilized
      to reduce the dimensionality of the feature space. Based on the most informative principal components (PCs),
      the model iteratively identifies reliable negatives (0) from the set of unlabeled samples (2). These reliable
      negatives are those that are most distant from the positive samples (1) in the feature space.
    - **Distance-based identification**: As a simple alternative, reliable negatives can also be identified using
      similarity measures like ``euclidean``, ``manhattan``, or ``cosine`` distance.

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        New dataset labels of samples in ``X`` with identified negative samples labeled by 0.
    df_pu_ : pd.DataFrame
        A DataFrame with the PCA-transformed features of 'X' containing the following groups of columns:

        - 'selection_via': Column indicating how reliable negatives were identified (either giving the distance metric
           or the i-th PC based on which the respective sample was selected).
        - 'PCi': Value columns for the i-th principal component (PC).
        - 'PCi_abs_dif': Absolute difference columns for each PC, representing the absolute deviation of each sample
          from the mean of positives.

        For distance-based identification, 'PCi' columns are replaced with the results for the selected metric.

    Notes
    -----
    * The method is inspired by deterministic PU learning techniques and follows an information-theoretic PU learning approach.

    See Also
    --------
    * For a detailed discussion on Positive-Unlabeled (PU) learning, its challenges, and evaluation strategies,
      refer to the PU Learning section in the Usage Principles documentation: `usage_principles/pu_learning`.

    See Also
    --------
    * See :func:`sklearn.decomposition.PCA`.
    """
    def __init__(self,
                 verbose: Optional[bool] = None,
                 pca_kwargs: Optional[dict] = None,
                 ):
        """
        Parameters
        ----------
        verbose : bool, optional
            Enable verbose output.
        pca_kwargs : dict, optional
            Additional keyword arguments for Principal Component Analysis (PCA) model.
        """
        self._verbose = ut.check_verbose(verbose)
        if pca_kwargs is None:
            pca_kwargs = dict()
        ut.check_model_kwargs(model_class=PCA, model_kwargs=pca_kwargs, param_to_check=None)
        self.pca_kwargs = pca_kwargs
        # Arguments for distance-based identification
        # Output parameters (will be set during model fitting)
        self.labels_ = None
        self.df_pu_ = None

    # Main method
    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            n_unl_to_neg: int = 1,
            metric: Optional[Literal["euclidean", "manhattan", "cosine"]] = None,
            n_components: Union[float, int] = 0.80,
            ) -> "dPULearn":
        """
        Fit the dPULearn model to identify reliable negative samples (0) from unlabeled samples (2)
        based on the distance to positive samples (1).

        Use the ``dPUlearn.labels_`` attribute to retrieve the output labels of samples in ``X``
        with identified negative samples labeled by 0.

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

        See Also
        --------
        * See `scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise>`_
          for details the three different pairwise distance measures.
        * See [Hastie09]_ for a detailed explanation on feature space and high-dimensional problems.

        Examples
        --------
        .. include:: examples/dpulearn_fit.rst
        """
        # Check input
        ut.check_X(X=X)
        ut.check_labels(labels=labels, vals_requiered=[1, 2], allow_other_vals=False)
        ut.check_number_range(name="n_unl_to_neg", val=n_unl_to_neg, min_val=1, just_int=True)
        _check_n_unl_to_neg(labels=labels, n_unl_to_neg=n_unl_to_neg, label_unl=2)
        _check_metric(metric=metric)
        _check_n_components(n_components=n_components)
        ut.check_match_X_labels(X=X, labels=labels)
        _check_match_X_n_components(X=X, n_components=n_components)
        # Compute average distance for threshold-based filtering (Yang et al., 2012, 2014; Nan et al. 2017)
        args = dict(X=X, labels=labels, n_unl_to_neg=n_unl_to_neg, label_neg=0)
        if metric is not None:
            new_labels, df_pu = _get_neg_via_distance(**args, metric=metric)
        # Identify most far away negatives in PCA compressed feature space
        else:
            new_labels, df_pu = _get_neg_via_pca(**args, n_components=n_components, **self.pca_kwargs)
        # Set new labels
        self.labels_ = np.asarray(new_labels)
        self.df_pu_ = df_pu
        return self

    # TODO add bool for disabeling KDE
    @staticmethod
    def eval(X: ut.ArrayLike2D,
             list_labels: ut.ArrayLike1D = None,
             list_names: Optional[List[str]] = None,
             X_neg: Optional[ut.ArrayLike2D] = None
             ) -> pd.DataFrame:
        """Compare different dPULearn results regarding the homogeneity within the reliably identified negatives (0),
        and the differences with the groups of positives (1), unlabeled samples (2), and a ground-truth negative group
        if provided by ``X_neg``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        list_labels : array-like, shape (n_datasets,)
            List of dataset labels for samples in ``X`` obtained by the :meth:`dPULearn.fit` method.
            Label values should be either 0 (identified negative), 1 (positive) or 2 (unlabeled).
        list_names : list, optional
            List of dataset names corresponding to ``list_labels``.
        X_neg : array-like, shape (n_samples, n_features), optional
            Feature matrix where `n_samples` is the number of samples and `n_features` is the number of features.
            Samples should be ground-truth negative samples, and features must correspond to ``X``.

        Returns
        -------
        df_eval : DataFrame
            Evaluation results for each DataFrame from ``list_df_pu`` comprising the following columns:
            - '':
            - ''


        See Also
        --------
        * See `usage_principles/pu_learning` for details on different evaluation strategies,
        """
        # Check input
        # Compute
        unl_in = False
        list_evals = []
        for labels in list_labels:
            # Evaluate homogeneity
            n_rel_neg = sum(labels == 0)
            avg_std, avg_iqr = eval_homogeneity(X=X, labels=labels)
            # Evaluate distribution alignment with positives
            avg_auc_abs, avg_kld = eval_distribution_alignment(X=X, labels=labels, label_test=0, label_ref=1)
            list_eval = [n_rel_neg, avg_std, avg_iqr, avg_auc_abs, avg_kld]
            # Evaluate distribution alignment with unlabeled
            if 2 in labels:
                avg_auc_abs, avg_kld = eval_distribution_alignment(X=X, labels=labels, label_test=0, label_ref=2)
                list_eval += [avg_auc_abs, avg_kld]
                unl_in = True
            # Evaluate distribution alignment with ground-truth negatives, if provided
            if X_neg is not None:
                avg_auc_abs, avg_kld = eval_distribution_alignment_X_neg(X=X, labels=labels, X_neg=X_neg)
                list_eval += [avg_auc_abs, avg_kld]
            list_evals.append(list_eval)
        # Define column names based on the evaluations performed
        cols_eval = ["n_rel_neg", "avg_std", "avg_iqr", "avg_abs_auc_pos", "avg_kld_pos"]
        if unl_in:
            cols_eval += ["avg_abs_auc_unl", "avg_kld_unl"]
        # Add additional columns for ground-truth negatives if X_neg is provided
        if X_neg is not None:
            cols_eval += ["avg_abs_auc_neg", "avg_kld_neg"]
        # Create the DataFrame
        df_eval = pd.DataFrame(list_evals, columns=cols_eval)
        df_eval[cols_eval] = df_eval.round(4)
        if list_labels is not None:
            df_eval.insert(0,"name", list_names)
        return df_eval
