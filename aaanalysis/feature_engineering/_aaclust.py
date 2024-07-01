"""
This is a script for the frontend of the AAclust class, a clustering wrapper object to obtain redundancy-reduced
scale subsets.
"""
from typing import Optional, Dict, List, Tuple, Type, Literal
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

from aaanalysis.template_classes import Wrapper
import aaanalysis.utils as ut

from ._backend.check_feature import check_df_cat

from ._backend.check_aaclust import check_metric
from ._backend.aaclust.aaclust_fit import estimate_lower_bound_n_clusters, optimize_n_clusters, merge_clusters
from ._backend.aaclust.aaclust_eval import evaluate_clustering
from ._backend.aaclust.aaclust_methods import (compute_centers, compute_medoids, name_clusters,
                                               compute_correlation)


# I Helper Functions
# Check parameter matching functions
def check_match_X_names(X=None, names=None, accept_none=True):
    """Verify that the number of samples in 'X' matches the length of 'names'."""
    if accept_none and names is None:
        return
    n_samples, n_features = X.shape
    if n_samples != len(names):
        raise ValueError(f"n_samples does not match for 'X' ({len(X)}) and 'names' ({len(names)}).")


def check_match_X_n_clusters(X=None, n_clusters=None, accept_none=True):
    """Ensure the number of samples and unique samples in 'X' are greater than or equal to 'n_clusters'."""
    if accept_none and n_clusters is None:
        return
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    if n_samples < n_clusters:
        raise ValueError(f"n_samples={n_samples} (in 'X') should be >= 'n_clusters' ({n_clusters})")
    if n_unique_samples < n_clusters:
        raise ValueError(f"'n_clusters' ({n_clusters}) should be >= n_unique_samples={n_unique_samples} (in 'X').")


def check_X_X_ref(X=None, X_ref=None):
    """Check that the number of features in 'X' matches the number of features in 'X_ref'."""
    n_samples, n_features = X.shape
    n_samples_ref, n_features_ref = X_ref.shape
    if n_features != n_features_ref:
        raise ValueError(f"n_features does not match for 'X' ({n_features}) and 'X_ref' ({n_features_ref}).")


def check_labels_cor(labels=None, labels_name="labels"):
    """Validate that 'labels' contains only integer values and is not None."""
    if labels is None:
        raise ValueError(f"'{labels_name}' should not be None.")
    # Convert labels to a numpy array if it's not already
    labels = np.asarray(labels)
    unique_labels = set(labels)
    wrong_types = [l for l in unique_labels if not np.issubdtype(type(l), np.integer)]
    if wrong_types:
        raise ValueError(f"Labels in '{labels_name}' should be type int, but contain: {set(map(type, wrong_types))}")
    return labels


# Post check functions
def post_check_n_clusters(n_clusters_actual=None, n_clusters=None):
    """Check if n_clusters set properly"""
    if n_clusters is not None and n_clusters_actual < n_clusters:
        warnings.warn(f"'n_clusters' was reduced from {n_clusters} to {n_clusters_actual} "
                      f"during AAclust algorithm.", ConvergenceWarning)


# Matching functions for filter_coverage
def check_match_X_scale_ids(X=None, scale_ids=None, accept_none=True):
    """Verify that the number of samples in 'X' matches the length of 'names'."""
    if accept_none and scale_ids is None:
        return
    n_samples, n_features = X.shape
    if n_samples != len(scale_ids):
        raise ValueError(f"n_samples does not match for 'X' ({len(X)}) and 'scale_ids' ({len(scale_ids)}).")


def check_match_scale_ids_names(scale_ids=None, names=None, df_cat=None, col_name=None):
    """Check scale elements corresponding to scale_ids are superset of names"""
    all_scale_names = df_cat[df_cat[ut.COL_SCALE_ID].isin(scale_ids)][col_name].to_list()
    ut.check_superset_subset(name_superset=f"df_cat ('{col_name}')",
                             superset=all_scale_names,
                             name_subset="names",
                             subset=names)
    ut.check_superset_subset(name_subset=f"df_cat ('{col_name}')",
                             subset=all_scale_names,
                             name_superset="names",
                             superset=names)


# II Main Functions
class AAclust(Wrapper):
    """
    Amino Acid clustering (**AAclust**) class: A k-optimized clustering wrapper for selecting redundancy-reduced sets
    of numerical scales [Breimann24a]_.

    AAclust uses clustering models that require a pre-defined number of clusters (k, set by ``n_clusters``),
    such as k-means or other `scikit-learn clustering models <https://scikit-learn.org/stable/modules/clustering.html>`_.
    It optimizes the value of k by utilizing Pearson correlation and then selects a representative sample ('medoid')
    for each cluster closest to the center, resulting in a redundancy-reduced sample set.

    Attributes
    ----------
    model : object
        The fitted clustering model object after calling the ``fit`` method.
    n_clusters : int
        Number of clusters obtained by AAclust.
    labels_ : array-like, shape (n_samples)
        Cluster labels in the order of samples in ``X``.
    centers_ : array-like, shape (n_clusters, n_features)
        Average scale values corresponding to each cluster.
    labels_centers_ : array-like, shape (n_clusters)
        Cluster labels for each cluster center.
    medoids_ : array-like, shape (n_clusters, n_features)
        Representative samples, one for each cluster.
    labels_medoids_ :  array-like, shape (n_clusters)
        Cluster labels for each medoid.
    is_medoid_ : array-like, shape (n_samples)
        Array indicating samples being medoids (1) or not (0). Same order as ``labels_``.
    medoid_names_ : list
        Names of the medoids. Set if ``names`` is provided to ``.fit``.
    """
    def __init__(self,
                 model_class: Type[ClusterMixin] = KMeans,
                 model_kwargs: Optional[Dict] = None,
                 verbose: bool = True,
                 random_state: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        model_class : Type[ClusterMixin], default=KMeans
            A clustering model class with ``n_clusters`` parameter.
        model_kwargs : dict, optional
            Keyword arguments to pass to the selected clustering model.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All attributes are set during fitting via the :meth:`AAclust.fit` method  and can be directly accessed.
        * AAclust is designed primarily for amino acid scales but can be used for any set of numerical indices.

        See Also
        --------
        * :class:`AAclustPlot`: the respective plotting class.
        * Scikit-learn `clustering model classes <https://scikit-learn.org/stable/modules/clustering.html>`_.

        Examples
        --------
        .. include:: examples/aaclust.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Model parameters
        ut.check_mode_class(model_class=model_class)
        model_kwargs = ut.check_model_kwargs(model_class=model_class,
                                             model_kwargs=model_kwargs,
                                             param_to_check="n_clusters",
                                             random_state=random_state)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        # Output parameters (set during model fitting)
        self.model: Optional[ClusterMixin] = None
        self.n_clusters: Optional[int] = None
        self.labels_: Optional[ut.ArrayLike1D] = None
        self.centers_: Optional[ut.ArrayLike1D] = None
        self.labels_centers_: Optional[ut.ArrayLike1D] = None
        self.medoids_: Optional[ut.ArrayLike1D] = None
        self.labels_medoids_: Optional[ut.ArrayLike1D] = None
        self.is_medoid_: Optional[ut.ArrayLike1D] = None
        self.medoid_names_: Optional[List[str]] = None

    def fit(self,
            X: ut.ArrayLike2D,
            n_clusters: Optional[int] = None,
            on_center: bool = True,
            min_th: float = 0.3,
            merge: bool = True,
            metric: str = "euclidean",
            names: Optional[List[str]] = None
            ) -> "AAclust":
        """
        Applies AAclust algorithm to feature matrix (``X``).

        Introduced in [Breimann24a]_, AAclust determines the optimal number of clusters, k, without pre-specification.
        It partitions data (``X``) into clusters by maximizing the within-cluster Pearson correlation beyond the
        ``min_th`` threshold. The quality of clustering is either based on the minimum Pearson correlation of all
        members (``on_center=False``) or between the cluster center and its members (``on_center=True``),
        using either the ``min_cor_all`` or ``min_cor_center`` correlation measures, respectively.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        n_clusters : int, optional
            Pre-defined number of clusters. If provided, k is not optimized. Must be 0 > n_clusters > n_samples.
        min_th : float, default=0.3
            Pearson correlation threshold for clustering optimization (between 0 and 1).
        on_center : bool, default=True
            If ``True``, ``min_th`` is applied to the cluster center. Otherwise, to all cluster members.
        merge : bool, default=True
            If ``True``, the optional merging step is performed.
        metric : {'correlation', 'euclidean', 'manhattan', 'cosine'}, default='euclidean'
            Similarity measure used for optional cluster merging and obtaining medoids:

             - ``correlation``: Pearson correlation (maximum)
             - ``euclidean``: Euclidean distance (minimum)
             - ``manhattan``: Manhattan distance (minimum)
             - ``cosine``: Cosine distance (minimum)

        names : list of str, optional
            List of sample names. If provided, sets :attr:`AAclust.medoid_names_` attribute.

        Returns
        -------
        AAclust
            The fitted instance of the AAclust class, allowing direct attribute access.

        Notes
        -----
        * The **AAclust** algorithm consists of three main steps:

            1. Estimate the lower bound of k.
            2. Refine k (recursively) using the chosen quality measure.
            3. Optionally, merge smaller clusters as directed by the merge ``metric``.

        * **AAclust** provides two correlation-based quality measure to optimize ``n_clusters``:

            - ``min_cor_center``: Minimum Pearson correlation between the cluster center and all cluster members.
            - ``min_cor_all``: Minium pairwise Pearson correlation among all cluster members.

        * A representative scale (medoid) closest to each cluster center is selected for redundancy reduction.

        See Also
        --------
        * :func:`sklearn.metrics.pairwise_distances` were used as distances for merging.

        Warnings
        --------
        * All RuntimeWarnings during the AAclust algorithm are caught and bundled into one RuntimeWarning.

        Examples
        --------
        .. include:: examples/aaclust_fit.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        names = ut.check_list_like(name="names", val=names, accept_none=True)
        ut.check_number_range(name="min_th", val=min_th, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="n_clusters", val=n_clusters, min_val=1, just_int=True, accept_none=True)
        check_metric(metric=metric)
        ut.check_bool(name="on_center", val=on_center)
        check_match_X_n_clusters(X=X, n_clusters=n_clusters, accept_none=True)
        check_match_X_names(X=X, names=names, accept_none=True)
        args = dict(model=self._model_class, model_kwargs=self._model_kwargs, min_th=min_th, on_center=on_center)
        # Clustering using given clustering models
        if n_clusters is not None:
            self.model = self._model_class(n_clusters=n_clusters, **self._model_kwargs)
            labels = self.model.fit(X).labels_.tolist()

        # Clustering using AAclust algorithm
        else:
            # 1. Step: Estimation of lower bound of k (number of clusters)
            if self._verbose:
                ut.print_out("1. Estimation of lower bound of k (number of clusters)")
            n_clusters_lb = estimate_lower_bound_n_clusters(X, **args)
            # 2. Step: Optimization of k by recursive clustering
            if self._verbose:
                objective_fct = "min_cor_center" if on_center else "min_cor_all"
                ut.print_out(f"2. Optimization of k by recursive clustering ({objective_fct}, min_th={min_th}, k={n_clusters_lb})")
            n_clusters = optimize_n_clusters(X, n_clusters=n_clusters_lb, **args)
            self.model = self._model_class(n_clusters=n_clusters, **self._model_kwargs)
            labels = self.model.fit(X).labels_.tolist()
            # 3. Step: Cluster merging (optional)
            if metric is not None:
                labels = merge_clusters(X, labels=labels, min_th=min_th, on_center=on_center, metric=metric)
                n_clusters = len(set(labels))
                if self._verbose:
                    ut.print_out(f"3. Cluster merging (k={n_clusters})")

        # Obtain cluster centers and medoids
        medoids, medoid_labels, medoid_ind = compute_medoids(X, labels=labels)
        centers, center_labels = compute_centers(X, labels=labels)

        # Save results in output parameters
        post_check_n_clusters(n_clusters_actual=len(set(labels)), n_clusters=n_clusters)
        self.n_clusters = len(set(labels))
        self.labels_ = np.array(labels)
        self.centers_ = centers
        self.labels_centers_ = center_labels
        self.medoids_ = medoids     # Representative scales
        self.labels_medoids_ = medoid_labels
        self.is_medoid_ = np.array([i in medoid_ind for i in range(0, len(labels))])
        if names is not None:
            self.medoid_names_ = [names[i] for i in medoid_ind]
        return self

    def eval(self,
             X: ut.ArrayLike2D,
             list_labels: ut.ArrayLike2D = None,
             names_datasets: Optional[List[str]] = None,
             ) -> pd.DataFrame:
        """
        Evaluates the quality of different clustering results.

        The following established clustering measures are used:

        - ``BIC`` (Bayesian Information Criterion): Reflects the goodness of fit for the clustering while accounting for
          the number of clusters and parameters. The BIC value can range from negative infinity to positive infinity.
          A higher BIC indicates superior clustering quality.
        - ``CH`` (Calinski-Harabasz Index): Represents the ratio of between-cluster dispersion mean to the within-cluster dispersion.
          The CH value ranges from 0 to positive infinity. A higher CH score suggests better-defined clustering.
        - ``SC`` (Silhouette Coefficient): Evaluates the proximity of each data point in one cluster to the points in the neighboring clusters.
          The SC score lies between -1 and 1. A value closer to 1 implies better clustering.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        list_labels : array-like, shape (n_datasets, n_samples)
            List of arrays with cluster labels for samples in ``X`` obtained by the :meth:`AAclust.fit` method.
            Unique label values indicate clusters.
        names_datasets : list, optional
            List of dataset names corresponding to ``list_labels``.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for each set of clustering labels from ``list_labels``.

        Notes
        -----
        ``df_eval`` includes the following columns:

            - 'names': Names (string) of evaluated datasets.
            - 'n_clusters': Number (integer) of clusters, equal to number of medoids.
            - 'BIC': BIC value (float) for clustering (-inf to inf).
            - 'CH': CH value (float) for clustering (0 to inf).
            - 'SC': SC value (float) for clustering (-1 to 1).

        BIC was adapted form this `StackExchange discussion <https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans>`_
        and modified to align with the SC and CH score so that higher values signify better clustering,
        contrary to conventional BIC implementation favoring lower values. See [Breimann24a]_.

        See Also
        --------
        * :meth:`AAclustPlot.eval`: the respective plotting method.
        * :func:`sklearn.metrics.silhouette_score`: a commonly used clustering quality measures.
        * :func:`sklearn.metrics.calinski_harabasz_score`: a commonly used clustering quality measures.

        Examples
        --------
        .. include:: examples/aaclust_eval.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        list_labels = ut.check_array_like(name="list_labels", val=list_labels, ensure_2d=True, convert_2d=True)
        ut.check_match_X_list_labels(X=X, list_labels=list_labels)
        names_datasets = ut.check_list_like(name="names_datasets", val=names_datasets, accept_none=True, accept_str=True,
                                            check_all_str_or_convertible=True)
        ut.check_match_list_labels_names_datasets(list_labels=list_labels, names_datasets=names_datasets)
        # Get number of clusters (number of medoids) and evaluation measures
        df_eval, warn_ch, warn_sc = evaluate_clustering(X, list_labels=list_labels, names_datasets=names_datasets)
        if warn_ch and self._verbose:
            warnings.warn("CH was set to 0 because sklearn.metric.calinski_harabasz_score returned NaN.", RuntimeWarning)
        if warn_sc and self._verbose:
            warnings.warn("SC was set to -1 because sklearn.metric.silhouette_score returned NaN.", RuntimeWarning)
        return df_eval

    @staticmethod
    def name_clusters(X: ut.ArrayLike2D,
                      labels: ut.ArrayLike1D = None,
                      names: List[str] = None,
                      shorten_names : bool = True,
                      ) -> List[str]:
        """
        Assigns names to clusters based on the frequency of names.

        Names with higher frequency are prioritized. If a name is already assigned to a cluster,
        or the cluster contains one sample, its name is set to 'Unclassified'.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``.
        names : list of str
            List of sample names corresponding to ``X``.
        shorten_names : bool, default=True
            If ``True``, shorten version of the names will be used.

        Returns
        -------
        cluster_names : list of str
            A list of renamed clusters based on names.

        Examples
        --------
        .. include:: examples/aaclust_name_clusters.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        names = ut.check_list_like(name="names", val=names, accept_none=False)
        ut.check_bool(name="shorten_names", val=shorten_names)
        ut.check_match_X_labels(X=X, labels=labels)
        check_match_X_names(X=X, names=names, accept_none=False)
        # Get cluster names
        cluster_names = name_clusters(X, labels=labels, names=names, shorten_names=shorten_names)
        return cluster_names

    @staticmethod
    def comp_centers(X: ut.ArrayLike2D,
                     labels: ut.ArrayLike1D = None
                     ) -> Tuple[ut.ArrayLike1D, ut.ArrayLike1D]:
        """
        Computes the center of each cluster based on the given labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``.

        Returns
        -------
        centers : array-like, shape (n_clusters,)
            The computed center for each cluster.
        labels_centers : array-like, shape (n_clusters,)
            The labels associated with each computed center.

        Examples
        --------
        .. include:: examples/aaclust_comp_centers.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        # Get cluster centers
        centers, labels_centers = compute_centers(X, labels=labels)
        return centers, labels_centers

    @staticmethod
    def comp_medoids(X: ut.ArrayLike2D,
                     labels: ut.ArrayLike1D = None,
                     metric: str = "correlation"
                     ) -> Tuple[ut.ArrayLike1D, ut.ArrayLike1D]:
        """
        Computes the medoid of each cluster based on the given labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``.
        metric : {'correlation', 'euclidean', 'manhattan', 'cosine'}, default='correlation'
            Similarity measure used to obtain medoids:

             - ``correlation``: Pearson correlation (maximum)
             - ``euclidean``: Euclidean distance (minimum)
             - ``manhattan``: Manhattan distance (minimum)
             - ``cosine``: Cosine distance (minimum)

        Returns
        -------
        medoids : array-like, shape (n_clusters,)
            The medoid for each cluster.
        labels_medoids : array-like, shape (n_clusters,)
            The labels corresponding to each medoid.

        Examples
        --------
        .. include:: examples/aaclust_comp_medoids.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        check_metric(metric=metric)
        # Get cluster medoids
        medoids, labels_medoids, _ = compute_medoids(X, labels=labels, metric=metric)
        return medoids, labels_medoids

    @staticmethod
    def comp_correlation(X: ut.ArrayLike2D,
                         labels: ut.ArrayLike1D = None,
                         X_ref: Optional[ut.ArrayLike2D] = None,
                         labels_ref: Optional[ut.ArrayLike1D] = None,
                         names: Optional[List[str]] = None,
                         names_ref: Optional[List[str]] = None
                         ) -> Tuple[pd.DataFrame, ut.ArrayLike1D]:
        """
        Computes the Pearson correlation of given data with reference data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``.
        X_ref : array-like, shape (n_samples, n_features)
            Feature matrix of reference data. If given, samples of ``X`` are compared with samples of ``X_ref``.
        labels_ref  : array-like, shape (n_samples_ref,)
            Cluster labels for each sample in ``X_ref``.
        names : list of str, optional
            List of sample names corresponding to ``X``.
        names_ref : list of str, optional
            List of sample names corresponding to ``X_ref``.

        Returns
        -------
        df_corr : pd.DataFrame, shape (n_samples, n_samples or n_samples_ref)
            DataFrame with correlation either for each pair in ``X`` of shape (n_samples, n_samples) or
            for each pair between ``X`` and ``X_ref`` of shape (n_samples, n_samples_ref).
        labels_sorted: array-like, shape (n_samples_ref,)
            Cluster labels for each sample and sorted as in `df_corr`.

        Notes
        -----
        * Rows will be sorted in ascending order of ``labels``.
        * Columns will be sorted in ascending order of ``labels`` or ``labels_ref`` if given.
        * Labels are replaced by respective names if given.

        See Also
        --------
        * :meth:`pandas.DataFrame.corr` used to compute the correlation.

        Examples
        --------
        .. include:: examples/aaclust_comp_correlation.rst
        """
        # Check input
        X = ut.check_X(X=X, min_n_samples=2)
        ut.check_X_unique_samples(X=X, min_n_unique_samples=2)
        labels = check_labels_cor(labels=labels, labels_name="labels")
        ut.check_match_X_labels(X=X, labels=labels)
        check_match_X_names(X=X, names=names, accept_none=True)
        if X_ref is not None:
            X_ref = ut.check_X(X=X_ref, min_n_samples=1)
            labels_ref = check_labels_cor(labels=labels_ref, labels_name="labels_ref")
            ut.check_match_X_labels(X=X_ref, labels=labels_ref)
            check_match_X_names(X=X_ref, names=names_ref, accept_none=True)
            check_X_X_ref(X=X, X_ref=X_ref)
        # Get correlations
        df_corr, labels_sorted = compute_correlation(X, X_ref=X_ref,
                                                     labels=labels, labels_ref=labels_ref,
                                                     names=names, names_ref=names_ref)
        return df_corr, labels_sorted

    @staticmethod
    def comp_coverage(names: List[str] = None,
                      names_ref: List[str] = None
                      ) -> float:
        """
        Computes the percentage of unique names from ``names`` that are present in ``names_ref``.

        This method helps in understanding the coverage of a particular set of names (subset)
        within a reference set of names (universal set). Each name from both ``names`` and ``names_ref``
        are considered only once, regardless of repetition.

        Parameters
        ----------
        names, list of str
            List of sample names. Should be subset of ``names_ref``.
        names_ref, list of str
            List of reference sample names. Should be superset of ``names``.

        Returns
        -------
        coverage : float
            Percentage of unique names from ``names`` that are found in ``names_ref``.

        Examples
        --------
        .. include:: examples/aaclust_comp_coverage.rst
        """
        names = ut.check_list_like(name="names", val=names, accept_none=False)
        names_ref = ut.check_list_like(name="names_ref", val=names_ref, accept_none=False)
        ut.check_superset_subset(subset=names, name_subset="names",
                                 superset=names_ref, name_superset="names_ref")
        # Compute coverage
        n_unique_intersect = len(set(names).intersection(set(names_ref)))
        n_unique_ref = len(set(names_ref))
        coverage = round(n_unique_intersect/n_unique_ref*100, 2)
        return coverage

    def filter_coverage(self,
                        X: ut.ArrayLike2D,
                        scale_ids: List[str] = None,
                        names_ref: List[str] = None,
                        min_coverage: int = 100,
                        df_cat: pd.DataFrame = None,
                        col_name: Literal['category', 'subcategory', 'scale_name'] = "subcategory"
                        ) -> List[str]:
        """
        Select a redundancy-reduced set of numerical scales with defined subcategory coverage.

        This method reduces the number of numerical scales in the feature matrix ``X``, while
        ensuring that the selected scales cover a minimum percentage (``min_coverage``) of subcategories.

        The process involves clustering the scales in ``X`` and selecting one scale per cluster. The initial number of
        clusters is determined by the number of unique subcategories in ``names_ref``. The number of clusters is
        increased step-wise until the overlap (coverage) between the unique elements in ``names_ref`` and the
        subcategories of the selected scales meets a defined threshold (``min_coverage``).

        Parameters
        ----------
        X : array-like, shape (n_scales, n_features)
            Feature matrix. `Rows` correspond to scales and `columns` to amino acids.
        scale_ids : list of str
            List of scale IDs corresponding to the rows in ``X``.
        names_ref : list of str
            List of reference sample names ('subcategories') representing the desired subcategories for coverage.
            Must contain the same unique elements as the unique subcategories associated with ``scale_ids``
        min_coverage : int, default=100
            Minimum coverage percentage of unique subcategories to be achieved by the selected clusters.
        df_cat : pd.DataFrame, optional
            DataFrame containing the categorical information for each scale. Should include columns ``scale_ids`` and
            the specified ``col_name``. Requiered columns are 'scale_id', 'category', 'subcategory', and 'scale_name'.
        col_name : {'category', 'subcategory', 'scale_name'}, default='subcategory'
             Column name in ``df_cat`` that contains the subcategory information (alternatively, category or scale name).

        Returns
        -------
        selected_scale_ids : list of str
            List of selected scale ids that meet the minimum coverage criteria.

        See Also
        --------
        * :meth:`AAclust.fit`: The clustering function used in every round for scale selection.
        * :meth:`AAclust.comp_coverage`: The function used to compute the subcategory coverage.

        Examples
        --------
        .. include:: examples/aaclust_filter_coverage.rst
        """
        # Check input
        X = ut.check_X(X=X, min_n_samples=2)
        scale_ids = ut.check_list_like(name="scale_ids", val=scale_ids, accept_none=False)
        names_ref = ut.check_list_like(name="names_ref", val=names_ref, accept_none=False)
        ut.check_number_range(name="min_coverage", val=min_coverage, just_int=True,
                              min_val=10, max_val=100, accept_none=False)
        check_df_cat(df_cat=df_cat, accept_none=False)
        ut.check_str_options(name="col_name", val=col_name,
                             list_str_options=[ut.COL_CAT, ut.COL_SUBCAT, ut.COL_SCALE_NAME])
        check_match_X_scale_ids(X=X, scale_ids=scale_ids, accept_none=False)
        check_match_scale_ids_names(scale_ids=scale_ids, names=names_ref, df_cat=df_cat, col_name=col_name)
        # Set number of unique names as number of initial clusters
        n = len(set(names_ref))
        n_samples = len(scale_ids)
        coverage = 0
        selected_scale_ids = []
        while coverage < min_coverage:
            selected_scale_ids = self.fit(X, names=scale_ids, n_clusters=n).medoid_names_
            selected_names = df_cat[df_cat[ut.COL_SCALE_ID].isin(selected_scale_ids)][col_name].to_list()
            coverage = self.comp_coverage(names=selected_names, names_ref=names_ref)
            # Heuristic step size to improve filtering speed
            step_size = int((min_coverage - coverage)/10) + 1
            n += step_size
            if n > n_samples:
                n = n_samples
        return selected_scale_ids
