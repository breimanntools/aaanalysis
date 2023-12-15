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


# Settings
LIST_METRICS = ['euclidean', 'manhattan', 'cosine']


# I Helper Functions
# Check functions
def _check_metric(metric=None):
    """Validate provided metric"""
    if metric is not None and metric not in LIST_METRICS:
        raise ValueError(f"'metric' ({metric}) should be None or one of following: {LIST_METRICS}")


def _check_n_unl_to_neg(labels=None, n_unl_to_neg=None, label_unl=None):
    """Validate that there are enough unlabeled samples in the dataset."""
    n_unl = sum([x == label_unl for x in labels])
    if n_unl < n_unl_to_neg:
        raise ValueError(f"Number of unlabeled labels ({n_unl}) must be higher than 'n_unl_to_neg' ({n_unl_to_neg})")


# II Main Functions
def _get_neg_via_distance(X=None, labels=None, metric="euclidean", n_unl_to_neg=None,
                          label_neg=0, label_pos=1):
    """Identify distant samples from positive mean as reliable negatives based on a specified distance metric."""
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    # Compute the average distances to the positive datapoints
    avg_dist = pairwise_distances(X[mask_pos], X, metric=metric).mean(axis=0)
    # Select negatives based on largest average distance to positives
    top_indices = np.argsort(avg_dist[mask_unl])[::-1][:n_unl_to_neg]
    new_labels = labels.copy()
    new_labels[top_indices] = label_neg
    return new_labels


def _get_neg_via_pca(X=None, labels=None, n_components=0.8, n_unl_to_neg=None,
                     label_neg=0, label_pos=1, **pca_kwargs):
    """Identify distant samples from positive mean as reliable negatives in PCA-compressed feature spaces."""
    col_selected = ["Selected_by_PC"]
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

    # Create df_pc based on PCA components
    df_pc = pd.DataFrame(pca.components_.T[:, :len(columns_pca)], columns=columns_pca)
    df_pc[col_selected] = None  # New column to store the PC information
    # Get mean of positive data for each component
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    pc_means = df_pc[mask_pos].mean(axis=0)
    # Select negatives based on absolute difference to mean of positives for each component
    new_labels = labels.copy()
    for col_pc, mean_pc, n in zip(columns_pca, pc_means, list_n_neg):
        col_dif = f"{col_pc}_abs_dif"
        # Calculate absolute difference to the mean for each sample in the component
        df_pc[col_dif] = np.abs(df_pc[col_pc] - mean_pc)
        # Sort and take top n indices
        top_indices = df_pc[mask_unl].sort_values(by=col_dif).tail(n).index
        # Update labels and masks
        new_labels[top_indices] = label_neg
        mask_unl[top_indices] = False
        # Record the PC by which the negatives are selected
        df_pc.loc[top_indices, col_selected] = col_pc.split(' ')[0]
    # Adjust df
    cols = [x for x in list(df_pc) if x != col_selected]
    df_pc[cols] = df_pc[cols].round(4)
    return new_labels, df_pc


# TODO refactor, test, document
class dPULearn:
    """
    Deterministic Positive-Unlabeled (dPULearn) model.

    dPULearn offers a deterministic approach for Positive-Unlabeled (PU) learning. The model primarily employs
    Principal Component Analysis (PCA) to reduce the dimensionality of the feature space. Based on the most
    informative principal components (PCs), it then iteratively identifies reliable negatives (0) from the set of
    unlabeled samples (2). These reliable negatives are those that are most distant from the positive samples (1) in
    the feature space. Alternatively, reliable negatives can also be identified using distance metrics like
    Euclidean, Manhattan, or Cosine distance if specified.

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        New dataset labels of samples in ``X`` with identified negative samples labeled by 0.

    Notes
    -----
    * The method is inspired by deterministic PU learning techniques and follows an information-theoretic PU learning approach.

    See Also
    --------
    * See :func:`sklearn.decomposition.PCA`.
    """
    def __init__(self,
                 verbose: Optional[bool] = None,
                 pca_kwargs: Optional[dict] = None,
                 metric: Optional[Literal["euclidean", "manhattan", "cosine", "None"]] = None
                 ):
        """
        Parameters
        ----------
        verbose
            Enable verbose output.
        pca_kwargs
            Additional keyword arguments for Principal Component Analysis (PCA) model.
        metric
            The distance metric to use. If ``None``, PCA-based identification is used. Distance-based identification
            is performed if metric is one of the following: {'euclidean', 'manhattan', 'cosine'}

        Notes
        -----
        * If ``metric`` is specified, distance-based identification of reliable negatives is performed.
          Otherwise, PCA-based identification is used.
        * Cosine metric is recommended in high-dimensional spaces.

        """
        self._verbose = ut.check_verbose(verbose)
        if pca_kwargs is None:
            pca_kwargs = dict()
        self.pca_kwargs = pca_kwargs
        # Arguments for distance-based identification
        _check_metric(metric=metric)
        self.metric = metric
        # Output parameters (will be set during model fitting)
        self.labels_ = None

    # Main method
    def fit(self,
            X: ut.ArrayLike2D,
            labels: ut.ArrayLike1D = None,
            n_unl_to_neg: int = 1,
            n_components: Union[float, int] = 0.80,
            ) -> pd.DataFrame:
        """
        Fit the dPULearn model to identify reliable negative samples (0) from unlabeled samples (2)
        based on the distance to positive samples (1).

        Use the ``dPUlearn.labels_`` attribute to retrieve the output labels of samples in ``X``
        with identified negative samples labeled by 0.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix where `n_samples` is the number of samples and `n_features` is the number of features.
        labels : array-like, shape (n_samples,)
            Dataset labels of samples in ``X``. Should be either 1 (positive) or 2 (unlabeled).
        n_unl_to_neg : int, default=1
            Number of negative samples (0) to be reliably identified from unlabeled samples (2).
        n_components
            Number of principal components or the percentage of total variance to be covered when PCA is applied.

        Returns
        -------
        df_pu : pd.DataFrame
            A DataFrame with the PCA-transformed features of 'X'. It includes additional columns:

            - 'Selected_by_PC': Indicates the principal component by which an unlabeled sample is identified as a negative.
            - '_abs_dif': Absolute difference columns for each principal component, representing the
              deviation of each sample from the mean of positives.

        Notes
        -----
        * If a distance metric is specified during class initialization, distance-based identification is
          used instead of PCA.

        Examples
        --------
        .. include:: examples/dpulearn_fit.rst
        """
        ut.check_X(X=X)
        ut.check_labels(labels=labels, vals_requiered=[1, 2], allow_other_vals=False)
        ut.check_number_range(name="n_unl_to_neg", val=n_unl_to_neg, min_val=1, just_int=True)
        ut.check_number_range(name="n_components", val=n_components, min_val=0, just_int=False)
        _check_n_unl_to_neg(labels=labels, n_unl_to_neg=n_unl_to_neg, label_unl=2)
        ut.check_match_X_labels(X=X, labels=labels)
        # Compute average distance for threshold-based filtering (Yang et al., 2012, 2014; Nan et al. 2017)
        args = dict(X=X, labels=labels, n_unl_to_neg=n_unl_to_neg, label_neg=0)
        if self.metric is not None:
            new_labels, df_pu = _get_neg_via_distance(**args, metric=self.metric)
        # Identify most far away negatives in PCA compressed feature space
        else:
            new_labels, df_pu = _get_neg_via_pca(**args, n_components=n_components, **self.pca_kwargs)
        # Set new labels
        self.labels_ = new_labels
        return df_pu

    def eval(self):
        """"""  # TODO add evaluation function
        # TODO check overlap, check homogeneity of samples compared to other sample
