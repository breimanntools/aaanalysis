"""
This is a script for special statistical measures not available or adjusted from popular python data analysis packages
such as scikit-learn, scipy, or statsmodels.

Developer note: Measures are implemented in aanalysis.utils.metrics to access them within the aanalysis package.
"""
import numpy as np
from typing import Optional

from aaanalysis.utils import auc_adjusted_, kullback_leibler_divergence_, bic_score_
import aaanalysis.utils as ut


# Helper functions
def _check_n_classes_n_samples(X=None, labels=None):
    """Check matching X and labels"""
    n_classes = len(set(labels))
    n_samples, n_features = X.shape
    if n_classes >= n_samples:
        raise ValueError(f"Number of classes in 'labels' ({n_classes}) must be smaller than n_samples ({n_samples})")
    if n_features == 0:
        raise ValueError(f"'n_features' should not be 0")


# Adjusted Area Under the Curve (AUC*)
def comp_auc_adjusted(X: ut.ArrayLike2D = None,
                      labels: ut.ArrayLike1D = None,
                      label_test: int = 1,
                      label_ref: int = 0,
                      n_jobs: Optional[int] = None
                      ) -> ut.ArrayLike1D:
    """
    Compute an adjusted Area Under the Curve (AUC) [-0.5, 0.5] assessing the similarity between two groups.

    Introduced in [Breimann25a]_, this adjusted AUC (denoted 'AUC*') is computed for each feature in the
    dataset ``X``, comparing two groups specified by the labels. It is based on the non-parametric measure of the
    difference between two groups. The adjustment of AUC subtracts 0.5, so it ranges between -0.5 and 0.5.
    An AUC* of 0 indicates an equal distribution between the two groups. This measure is useful for ranking features
    based on their ability to distinguish between the two groups.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. 'Rows' typically correspond to proteins and 'columns' to features.
    labels : array-like, shape (n_samples,)
        Dataset labels of samples in X. Should contain only two different integer label values,
        representing test and reference group (typically, 1 and 0).
    label_test : int, default=1,
        Class label of test group in ``labels``.
    label_ref : int, default=0,
        Class label of reference group in ``labels``.
    n_jobs : int, None, or -1, default=None
        Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
        If ``-1``, the number is set to all available cores.

    Returns
    -------
    auc : array-like, shape (n_features,)
        Array with AUC* values for each feature, ranging from [-0.5, 0.5].
        A value of 0 indicates equal distributions between the two groups for that feature.

    Examples
    --------
    .. include:: examples/comp_auc_adjusted.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X, min_n_unique_samples=1)
    ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
    ut.check_number_val(name="label_ref", val=label_ref, just_int=True, accept_none=False)
    labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref], allow_other_vals=False)
    ut.check_match_X_labels(X=X, labels=labels)
    n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
    # Compute adjusted AUC
    auc = auc_adjusted_(X=X, labels=labels, label_test=label_test, n_jobs=n_jobs)
    return auc


# BIC score
def comp_bic_score(X: ut.ArrayLike2D = None,
                   labels: ut.ArrayLike1D = None
                   ) -> float:
    """
    Compute an adjusted Bayesian Information Criterion (BIC) (-∞, ∞) for assessing clustering quality.

    Described in [Breimann24b], this adjusted BIC is computed for a given set of clusters in the dataset ``X``.
    The BIC is a clustering model selection criterion that balances the model complexity against the
    likelihood of the data distribution. Unlike the traditional BIC where lower values are better, this adjusted BIC,
    is modified to align with other clustering evaluation measures like the
    Silhouette coefficient and the Calinski-Harabasz score. In this adjusted version, higher values indicate
    better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. 'Rows' typically correspond to proteins and 'columns' to features.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample. Each label corresponds to a cluster.

    Returns
    -------
    bic : float
        The Bayesian Information Criterion value. A lower BIC value indicates a better model fit to the data.

    Notes
    -----
    *  An `epsilon` value (1e-10) is utilized to prevent division by zero in the computation.

    See Also
    --------
    * The Silhouette coefficient [-1, 1] can be computed by :func:`sklearn.metrics.silhouette_score`.
    * The Calinski Harabasz score [0, ∞] can be obtained using :func:`sklearn.metrics.calinski_harabasz_score`.
    * Clustering evaluation can be performed using :meth:`AAclust.eval`.

    Examples
    --------
    .. include:: examples/comp_bic_score.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X)
    labels = ut.check_labels(labels=labels)
    ut.check_match_X_labels(X=X, labels=labels)
    _check_n_classes_n_samples(X=X, labels=labels)
    # Compute bic
    bic = bic_score_(X, labels=labels)
    return bic


# Kullback-Leibler Divergence
def comp_kld(X: ut.ArrayLike2D = None,
             labels: ut.ArrayLike1D =None,
             label_test: int = 1,
             label_ref: int = 0
             ) -> ut.ArrayLike1D:
    """
    Compute the Kullback-Leibler Divergence (KLD) [0, ∞) for assessing the similarity between two groups.

    The KLD is calculated for each feature in ``X``, comparing the distributions between two subgroups specified
    by ``label_test`` and ``label_ref`` in labels. Generally, the KLD measures how one probability distribution
    diverges from a second, expected probability distribution. Higher KLD values indicate more divergence. The observed
    upper limit lies around 200 indicating complete divergence of two non-overlapping distributions.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
    labels : array-like, shape (n_samples,)
        Labels for each sample in ``X``. Should contain only integer label values and at least 2 per class.
    label_test : int, default=1,
        Class label of test group in ``labels``.
    label_ref : int, default=0,
        Class label of reference group in ``labels``.

    Returns
    -------
    kld : array-like, shape (n_features,)
        Array of Kullback-Leibler Divergence values for each feature in ``X``. Each value represents
        the divergence of the test group distribution from the reference group distribution for that feature.

    Notes
    -----
    * For valid KLD calculations, the input matrix `X` must meet certain conditions:

      - Ensure adequate variability of features in ``X`` to avoid computational problems like singular
        covariance matrices in Gaussian KDE.
      - Avoid rows in ``X`` lying in a lower-dimensional subspace; consider dimensionality reduction if necessary.

    See Also
    --------
    * :func:`scipy.stats.gaussian_kde` function representing a kernel-density estimate using Gaussian kernels.
      It is used for estimating the probability density function of a random variable (i.e., feature in ``X``),
      which is a crucial step in the computation of Kullback-Leibler Divergence (KLD).
    * :func:`scipy.stats.entropy` function for computing the Shannon entropy. In the context of KLD,
      it is used to measure the divergence between two probability distributions, typically derived
      from kernel-density estimates of different data groups.

    Examples
    --------
    .. include:: examples/comp_kld.rst
    """
    # Check input
    X = ut.check_X(X=X, min_n_features=1)
    ut.check_X_unique_samples(X=X, min_n_unique_samples=3)
    ut.check_number_val(name="label_test", val=label_test, just_int=True, accept_none=False)
    ut.check_number_val(name="label_ref", val=label_ref, just_int=True, accept_none=False)
    labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                             n_per_group_requiered=2, allow_other_vals=False)
    ut.check_match_X_labels(X=X, labels=labels, check_variability=True)
    # Compute tge Kullback-Leibler divergence
    try:
        kld = kullback_leibler_divergence_(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
    except Exception as e:
        raise ValueError(f"Following error occurred during the computation of Kullback-Leibler Divergence: {e}")
    return kld
