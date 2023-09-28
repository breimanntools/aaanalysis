"""
This is a script for the AAclust clustering wrapper method.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Callable, Dict, Union, List, Tuple
import inspect

import aaanalysis.utils as ut
from aaanalysis.aaclust._aaclust import estimate_lower_bound_n_clusters, optimize_n_clusters, merge_clusters
from aaanalysis.aaclust._aaclust_statics import compute_centers, compute_medoids, compute_correlation, name_clusters


# I Helper Functions
# Check functions
def check_model(model=None, model_kwargs=None, except_None=False):
    """"""
    if except_None:
        return model_kwargs
    list_model_args = list(inspect.signature(model).parameters.keys())
    if "n_clusters" not in list_model_args:
        error = f"'n_clusters' should be argument in given clustering 'model' ({model})."
        raise ValueError(error)
    model_kwargs = {x: model_kwargs[x] for x in model_kwargs if x in list_model_args}
    return model_kwargs


def check_merge_metric(merge_metric=None):
    """"""
    if merge_metric is not None and merge_metric not in ut.LIST_METRICS:
        error = f"'merge_metric' should be None or one of following: {ut.LIST_METRICS}"
        raise ValueError(error)


def check_match_feat_matrix_n_clusters(X=None, n_clusters=None):
    """"""
    n_samples, n_features = X.shape
    if n_clusters is not None and n_samples <= n_clusters:
        raise ValueError(f"'X' must contain more samples ({n_samples}) then 'n_clusters' ({n_clusters})")

# II Main Functions
# TODO check, interface, testing, simplifying (Remove functions if not needed)
class AAclust:
    """
    A k-optimized clustering wrapper for selecting redundancy-reduced sets of numerical scales.

    AAclust is designed primarily for amino acid scales but can be used for any set of numerical indices,
    introduced in [Breimann23a]_. It uses clustering models like from the `scikit-learn clustering model
    <https://scikit-learn.org/stable/modules/clustering.html>`_ that require a pre-defined number of clusters (k),
    set by their ``n_clusters`` parameter. AAclust optimizes the value of k by utilizing Pearson correlation
    and then selects a representative sample ('medoid')vfor each cluster closest to the center,
    resulting in a redundancy-reduced sample set.

    Parameters
    ----------
    model
        A clustering model with ``n_clusters`` parameter.
    model_kwargs
        Keyword arguments to pass to the selected clustering model.
    verbose
        If ``True``, verbose outputs are enabled.

    Attributes
    ----------
    n_clusters : int
        Number of clusters obtained by AAclust.
    labels_ : array-like, shape (n_samples, )
        Cluster labels in the order of samples in ``X``.
    centers_ : array-like, shape (n_clusters, n_features)
        Average scale values corresponding to each cluster.
    center_labels_ : array-like, shape (n_clusters, )
        Cluster labels for each cluster center.
    medoids_ : array-like, shape (n_clusters, n_features)
        Representative samples, one for each cluster.
    medoid_labels_ :  array-like, shape (n_clusters, )
        Cluster labels for each medoid.
    is_medoid_ : array-like, shape (n_samples, )
        Array indicating samples being medoids (1) or not (0). Same order as ``labels_``.
    medoid_names_ : list
        Names of the medoids. Set if ``names`` is provided to ``fit``.

    Notes
    -----
    All attributes are set during ``.fit`` and can be directly accessed.

    """
    def __init__(self,
                 model: Optional[Callable] = None,
                 model_kwargs: Optional[Dict] = None,
                 verbose: bool = False):
        # Model parameters
        if model is None:
            model = KMeans
            # Set to avoid FutureWarning
            model_kwargs = model_kwargs or dict(n_init="auto")
        model_kwargs = check_model(model=model, model_kwargs=model_kwargs)
        self.model = model
        self._model_kwargs = model_kwargs
        self._verbose = ut.check_verbose(verbose)
        # Output parameters (set during model fitting)
        self.n_clusters: Optional[int] = None
        self.labels_: Optional[ut.ArrayLikeInt] = None
        self.centers_: Optional[ut.ArrayLikeFloat] = None
        self.center_labels_: Optional[ut.ArrayLikeInt] = None
        self.medoids_: Optional[ut.ArrayLikeFloat] = None
        self.medoid_labels_: Optional[ut.ArrayLikeInt] = None
        self.is_medoid_: Optional[ut.ArrayLikeBool] = None
        self.medoid_names_: Optional[List[str]] = None

    def fit(self,
            X: ut.ArrayLikeFloat,
            names: Optional[List[str]] = None,
            on_center: bool = True,
            min_th: float = 0,
            merge_metric: Union[str, None] = "euclidean",
            n_clusters: Optional[int] = None
            ) -> "AAclust":
        """
        Applies AAclust algorithm to feature matrix (``X``).

        AAclust determines the optimal number of clusters, k, without pre-specification. It partitions data (``X``) into
        clusters by maximizing the within-cluster Pearson correlation beyond the ``min_th`` threshold. The quality of
        clustering is either based on the minimum Pearson correlation of all members ('min_cor all') or between
        the cluster center and its members ('min_cor center'), governed by `on_center`. See details in [Breimann23a]_.

        The AAclust algorithm has three steps:

        1. Estimate the lower bound of k.
        2. Refine k using the chosen quality metric.
        3. Optionally, merge smaller clusters, as directed by `merge_metric`.

        A representative scale (medoid) closest to each cluster center is chosen for redundancy reduction.

        Parameters
        ----------
        X
            Feature matrix. Shape: (n_samples, n_features).
        names
            Sample names. If provided, sets :attr:`aanalysis.AAclust.medoid_names_` attribute.
        min_th
            Pearson correlation threshold for clustering (between 0 and 1).
        on_center
            If ``True``, ``min_th`` is applied to the cluster center. Otherwise, to all cluster members.
        merge_metric
            Metric used as similarity measure for optional cluster merging:

             - ``None``: No merging is performed.
             - ``euclidean``: Euclidean distance.
             - ``pearson``: Pearson correlation.

        n_clusters
            Pre-defined number of clusters. If provided, AAclust uses this instead of optimizing k.

        Returns
        -------
        self : AAclust
            The fitted instance of the AAclust class, allowing direct attribute access.

        Notes
        -----
        Set all attributes within the :class:`aanalysis.AAclust` class.

        """
        # Check input
        ut.check_list(name="names", val=names, accept_none=True)
        ut.check_number_range(name="mint_th", val=min_th, min_val=0, max_val=1, just_int=False, accept_none=False)
        ut.check_number_range(name="n_clusters", val=n_clusters, min_val=0, just_int=True, accept_none=True)
        check_merge_metric(merge_metric=merge_metric)
        ut.check_bool(name="on_center", val=on_center)
        X = ut.check_feat_matrix(X=X, y=names, name_y="names")
        check_match_feat_matrix_n_clusters(X=X, n_clusters=n_clusters)

        args = dict(model=self.model, model_kwargs=self._model_kwargs, min_th=min_th, on_center=on_center,
                    verbose=self._verbose)

        # Clustering using given clustering models
        if n_clusters is not None:
            labels = self.model(n_clusters=n_clusters, **self._model_kwargs).fit(X).labels_.tolist()

        # Clustering using AAclust algorithm
        else:
            # Step 1.: Estimation of lower bound of k (number of clusters)
            n_clusters_lb = estimate_lower_bound_n_clusters(X, **args)
            # Step 2. Optimization of k by recursive clustering
            n_clusters = optimize_n_clusters(X, n_clusters=n_clusters_lb, **args)
            labels = self.model(n_clusters=n_clusters, **self._model_kwargs).fit(X).labels_.tolist()
            # Step 3. Cluster merging (optional)
            if merge_metric is not None:
                labels = merge_clusters(X, labels=labels, min_th=min_th, on_center=on_center,
                                        metric=merge_metric, verbose=self._verbose)

        # Obtain cluster centers and medoids
        medoids, medoid_labels, medoid_ind = compute_medoids(X, labels=labels)
        centers, center_labels = compute_centers(X, labels=labels)

        # Save results in output parameters
        self.n_clusters = len(set(labels))
        self.labels_ = np.array(labels)
        self.centers_ = centers
        self.center_labels_ = center_labels
        self.medoids_ = medoids     # Representative scales
        self.medoid_labels_ = medoid_labels
        self.is_medoid_ = np.array([i in medoid_ind for i in range(0, len(labels))])
        if names is not None:
            self.medoid_names_ =  [names[i] for i in medoid_ind]
        return self


    def evaluate(self):
        """Evaluate one or more results"""
        # TODO add evaluation function

    @staticmethod
    def name_clusters(X: ut.ArrayLikeFloat,
                      labels: ut.ArrayLikeInt = None,
                      names: List[str] = None
                      ) -> List[str]:
        """
        Assigns names to clusters based on scale names and their frequency.

        This method renames clusters based on the names of the scales in each cluster, with priority given to the
        most frequent scales. If the name is already used or does not exist, it defaults to 'name_unclassified'.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        labels
            Cluster labels for each sample in X.
        names
            List of scale names corresponding to each sample.

        Returns
        -------
        cluster_names : List[str]
            A list of renamed clusters based on scale names.
        """
        # Check input
        ut.check_array_like(name="labels", val=labels, dtype="int")
        ut.check_list(name='names', val=names)
        ut.check_feat_matrix(X=X)
        # Get cluster names
        cluster_names = name_clusters(X, labels=labels, names=names)
        return cluster_names

    @staticmethod
    def compute_centers(X: ut.ArrayLikeFloat,
                        labels: Optional[ut.ArrayLikeInt] = None
                        ) -> Tuple[ut.ArrayLikeFloat, ut.ArrayLikeInt]:
        """
        Computes the center of each cluster based on the given labels.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        labels
            Cluster labels for each sample in X.

        Returns
        -------
        centers
            The computed center for each cluster.
        center_labels
            The labels associated with each computed center.
        """
        centers, center_labels = compute_centers(X, labels=labels)
        return centers, center_labels

    @staticmethod
    def compute_medoids(X: ut.ArrayLikeFloat,
                        labels: Optional[ut.ArrayLikeInt] = None
                        ) -> Tuple[ut.ArrayLikeFloat, ut.ArrayLikeInt]:
        """
        Computes the medoid of each cluster based on the given labels.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        labels
            Cluster labels for each sample in X.

        Returns
        -------
        medoids
            The medoid for each cluster.
        medoid_labels
            The labels corresponding to each medoid.
        medoid_ind
            Indexes of medoids within the original data.
        """
        medoids, medoid_labels, _ = compute_medoids(X, labels=labels)
        return medoids, medoid_labels

    @staticmethod
    def compute_correlation(X: ut.ArrayLikeFloat,
                            X_ref: ut.ArrayLikeFloat,
                            labels: Optional[ut.ArrayLikeInt] = None,
                            labels_ref: Optional[ut.ArrayLikeInt] = None,
                            n: int = 3,
                            positive: bool = True,
                            on_center: bool = False
                            ) -> List[str]:
        """
        Computes the Pearson correlation of given data with reference data.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        X_ref
            Reference feature matrix.
        labels
            Cluster labels for the test data.
        labels_ref
            Cluster labels for the reference data.
        n
            Number of top centers to consider based on correlation strength.
        positive
            If True, considers positive correlations. Else, negative correlations.
        on_center
            If True, correlation is computed with cluster centers. Otherwise, with all cluster members.

        Returns
        -------
        list_top_center_name_corr : List[str]
            Names and correlations of centers having the strongest (positive/negative) correlation with test data samples.
        """
        # Check input
        X, labels = ut.check_feat_matrix(X=X, y=labels)
        X_ref, labels_ref = ut.check_feat_matrix(X=X_ref, y=labels_ref)
        list_top_center_name_corr = compute_correlation(X, X_ref, labels=labels, labels_ref=labels_ref,
                                                        n=n, positive=positive, on_center=on_center)
        return list_top_center_name_corr

    @staticmethod
    def compute_coverage(names=None, names_ref=None):
        """Computes coverage of """

