"""
This is a script for the frontend of the dPULearn class, used for deterministic Positive-Unlabeled (PU) Learning.
"""
from typing import Optional, Literal, Dict, Union, List, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import aaanalysis.utils as ut
from aaanalysis.template_classes import Wrapper
from ._backend.dpulearn.dpul_fit import (get_neg_via_distance, get_neg_via_pca)
from ._backend.dpulearn.dpul_eval import eval_identified_negatives
from ._backend.dpulearn.dpul_compare_sets_neg import compare_sets_negatives_

# Settings
LIST_METRICS = ['euclidean', 'manhattan', 'cosine']


# I Helper Functions
# Check functions
def check_metric(metric=None) -> None:
    """Validate provided metric"""
    if metric is not None and metric not in LIST_METRICS:
        raise ValueError(f"'metric' ({metric}) should be None or one of following: {LIST_METRICS}")


def check_n_to_identify(labels=None, n_to_identify=None, label_unl=None) -> None:
    """Validate that there are enough unlabeled samples to identify the requested negatives."""
    n_unl = np.sum(labels == label_unl)
    if n_unl < n_to_identify:
        raise ValueError(f"Number of unlabeled samples ({n_unl}) must be higher than the number of "
                         f"negatives to identify from them ({n_to_identify}).")


def check_match_labels_markers(label_pos=None, label_unl=None, label_neg=None) -> None:
    """Validate the positive/unlabeled/negative marker values used to encode the input labels."""
    # Markers may be any integer (e.g. -1), they only need to be distinct and match the labels.
    ut.check_number_val(name="label_pos", val=label_pos, just_int=True)
    ut.check_number_val(name="label_unl", val=label_unl, just_int=True)
    ut.check_number_val(name="label_neg", val=label_neg, just_int=True, accept_none=True)
    markers = {"label_pos": label_pos, "label_unl": label_unl}
    if label_neg is not None:
        markers["label_neg"] = label_neg
    if len(set(markers.values())) != len(markers):
        raise ValueError(f"Label markers must be distinct, but got {markers}.")


def check_match_labels_markers_present(labels=None, label_pos=None, label_unl=None, label_neg=None) -> None:
    """Ensure 'labels' contains the required markers and no values outside the marker set."""
    allowed = {label_pos, label_unl} | ({label_neg} if label_neg is not None else set())
    present = set(np.asarray(labels).tolist())
    extra = present - allowed
    if extra:
        markers = f"label_pos={label_pos}, label_unl={label_unl}" + (
            f", label_neg={label_neg}" if label_neg is not None else "")
        raise ValueError(f"'labels' ({present}) contains values {sorted(extra)} that are none of "
                         f"the markers ({markers}). Set label_neg= to mark pre-labeled negatives.")


def normalize_pu_labels(labels=None, label_pos=None, label_unl=None, label_neg=None):
    """Map the user's encoding onto the internal 1 (positive) / 2 (unlabeled) / 0 (negative) contract."""
    labels = np.asarray(labels)
    normalized = labels.copy()
    normalized[labels == label_pos] = 1
    normalized[labels == label_unl] = 2
    if label_neg is not None:
        normalized[labels == label_neg] = 0
    return normalized


def resolve_n_to_identify(n_neg=None, n_unl_to_neg=None, n_pre_neg=0):
    """Resolve how many negatives to identify from the unlabeled pool.

    Exactly one of ``n_neg`` (total negatives wanted) or ``n_unl_to_neg`` (negatives to take
    directly from the unlabeled pool) must be given.
    """
    if (n_neg is None) == (n_unl_to_neg is None):
        raise ValueError("Specify exactly one of 'n_neg' (the TOTAL number of negatives wanted) "
                         "or 'n_unl_to_neg' (the number of negatives to identify directly from "
                         "the unlabeled pool).")
    if n_unl_to_neg is not None:
        ut.check_number_range(name="n_unl_to_neg", val=n_unl_to_neg, min_val=1, just_int=True)
        return n_unl_to_neg
    ut.check_number_range(name="n_neg", val=n_neg, min_val=1, just_int=True)
    n_to_identify = n_neg - n_pre_neg
    if n_to_identify < 1:
        raise ValueError(f"'n_neg' ({n_neg}) is the TOTAL number of negatives wanted, but 'labels' "
                         f"already contains {n_pre_neg} pre-labeled negative(s); it must exceed that "
                         f"by at least 1 (or use 'n_unl_to_neg' to set the count directly).")
    return n_to_identify


def check_n_components(n_components=1) -> None:
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


def check_match_X_n_components(X=None, n_components=1) -> None:
    """Check if n_components matches to dimensions of X"""
    n_samples, n_features = X.shape
    if min(n_features, n_samples) <= n_components:
        raise ValueError(f"'n_components' ({n_components}) should be < min(n_features, n_samples) from 'X' ({n_features})")


def check_match_list_labels_df_seq(list_labels=None, df_seq=None) -> None:
    """Check if length of labels in list_labels and df_seq matches"""
    if df_seq is None:
        return None # Skip check
    n_samples = len(list_labels[0])
    if n_samples != len(df_seq):
        raise ValueError(f"Number of samples (n={n_samples}) in 'list_labels' does not match with "
                         f"samples in 'df_seq' (n={len(df_seq)})")


def check_match_X_X_neg(X=None, X_neg=None) -> None:
    """Check if number of features matches in both feature matrices"""
    if X_neg is None:
        return # Skip test
    n_features = X.shape[1]
    n_features_neg =  X_neg.shape[1]
    if n_features != n_features_neg:
        raise ValueError(f"'n_features' does not match between 'X' (n={n_features}) and 'X_neg' (n={n_features_neg})")


def check_match_X_pos_X_unlabelled(X_pos=None, X_unlabelled=None) -> None:
    """Check that positive and unlabeled feature matrices share the same feature dimension."""
    n_features_pos = X_pos.shape[1]
    n_features_unl = X_unlabelled.shape[1]
    if n_features_pos != n_features_unl:
        raise ValueError(f"'n_features' does not match between 'X_pos' (n={n_features_pos}) and "
                         f"'X_unlabelled' (n={n_features_unl})")


# II Main Functions
class dPULearn(Wrapper):
    """
    Deterministic Positive-Unlabeled Learning (**dPULearn**) class for identifying reliable negatives from unlabeled data [Breimann25]_.

    As a ``Wrapper``, it implements the ``.fit`` / ``.eval`` model contract.

    dPULearn offers a deterministic approach to Positive-Unlabeled (PU) learning, featuring two distinct
    identification approaches:

    * **PCA-based identification**: This is the primary method where Principal Component Analysis (PCA) is utilized
      to reduce the dimensionality of the feature space. Based on the most informative principal components (PCs),
      the model iteratively identifies reliable negatives (labeled by 0) from the set of unlabeled samples (2).
      These reliable negatives are those that are most distant from the positive samples (1) in the feature space.
    * **Distance-based identification**: As a simple alternative, reliable negatives can also be identified using
      similarity measures like ``euclidean``, ``manhattan``, or ``cosine`` distance.

    .. versionadded:: 0.1.0

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        New dataset labels of samples in ``X`` with identified negative samples labeled by 0.
    df_pu_ : pd.DataFrame, shape (n_samples, pca_features)
        A DataFrame with the PCA-transformed features of 'X' containing the following groups of columns:

        * 'selection_via': Column indicating how reliable negatives were identified (either giving the distance metric
           or the i-th PC based on which the respective sample was selected).
        * 'PCi': Value columns for the i-th principal component (PC).
        * 'PCi_abs_dif': Absolute difference columns for each PC, representing the absolute deviation of each sample
          from the mean of positives.

        For distance-based identification, 'PCi' columns are replaced with the results for the selected metric.


    """
    def __init__(self,
                 model_kwargs: Optional[dict] = None,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
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
            labels: ut.ArrayLike1D,
            label_pos: int = 1,
            label_unl: int = 2,
            label_neg: Optional[int] = None,
            n_neg: Optional[int] = None,
            n_unl_to_neg: Optional[int] = None,
            metric: Optional[Literal["euclidean", "manhattan", "cosine"]] = None,
            n_components: Union[float, int] = 0.80,
            ) -> "dPULearn":
        """
        Fit the dPULearn model to identify reliable negative samples (labeled by 0) from unlabeled samples (2)
        based on the distance to positive samples (1).

        Only unlabeled samples are candidates for reclassification; any pre-labeled negatives provided via
        ``label_neg`` are kept as negatives and are never re-selected. Specify the count in one of two ways
        (exactly one): ``n_neg`` as the **total** number of negatives wanted (dPULearn identifies ``n_neg``
        minus the pre-labeled negatives), or ``n_unl_to_neg`` to set the number identified **directly from
        the unlabeled pool**.

        Use the ``dPULearn.labels_`` attribute to retrieve the output labels of samples in ``X``
        including identified negatives. Output labels always use the package convention
        (1 = positive, 0 = reliable negative, 2 = remaining unlabeled), regardless of the input markers.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        labels : array-like, shape (n_samples,)
            Dataset labels of samples in ``X``. Must contain the positive marker (``label_pos``) and the
            unlabeled marker (``label_unl``); pre-labeled negatives (``label_neg``) are optional. By
            default positives are ``1`` and unlabeled are ``2``; set ``label_unl=0`` to pass the standard
            ``{0, 1}`` encoding directly (``0`` = unlabeled, ``1`` = positive).
        label_pos : int, default=1
            Value marking positive samples in ``labels``. Must be present.
        label_unl : int, default=2
            Value marking unlabeled samples in ``labels`` (the candidate pool). Must be present. Set
            ``label_unl=0`` to pass the standard ``{0, 1}`` encoding (``0`` = unlabeled, ``1`` =
            positive) without re-encoding.
        label_neg : int or None, default=None
            Value marking pre-labeled (already known) negatives in ``labels``. When given, those
            samples are kept as negatives and never re-selected, and dPULearn only identifies the
            remaining (``n_neg`` minus pre-labeled) negatives from the unlabeled pool. ``None`` means
            ``labels`` contains no pre-labeled negatives. Must differ from ``label_pos`` / ``label_unl``.
        n_neg : int, optional
            **Total** number of negatives (0) wanted in the output: any pre-labeled negatives
            (``label_neg``) plus the newly identified reliable negatives add up to ``n_neg``. So
            dPULearn identifies ``n_neg`` minus the pre-labeled negatives (with no pre-labeled
            negatives it identifies exactly ``n_neg``). It must exceed the number of pre-labeled
            negatives. Provide **exactly one** of ``n_neg`` or ``n_unl_to_neg``.
        n_unl_to_neg : int, optional
            Number of reliable negatives to identify **directly from the unlabeled pool** — direct
            control over how many unlabeled samples are reclassified, independent of any pre-labeled
            negatives (final negatives = pre-labeled + ``n_unl_to_neg``). Provide **exactly one** of
            ``n_neg`` or ``n_unl_to_neg``. With no pre-labeled negatives the two are equivalent.
        metric : str or None, optional
            The distance metric to use. If ``None``, Principal Component Analysis (PCA)-based
            identification is performed. For distance-based identification one of the following
            measures can be selected:

            * ``euclidean``: Euclidean distance (minimum)
            * ``manhattan``: Manhattan distance (minimum)
            * ``cosine``: Cosine distance (minimum)

        n_components : int or float, default=0.80
            Number of principal components (a) or the percentage of total variance to be covered (b) when PCA is applied.

            * In case (a): it should be an integer >= 1.
            * In case (b): it should be a float with  0.0 < ``n_components`` < 1.0.

        Returns
        -------
        dPULearn
            The fitted instance of the dPULearn class, allowing direct attribute access.

        Notes
        -----
        * If a distance metric is specified, dPULearn performs distance-based instead of PCA-based identification.
        * When selecting a distance metric for distance-based identification, consider the dimensionality of the
          feature space, determined by the ratio of the number of features (n_features) to the number of samples
          (n_samples) in `X`. In a low-dimensional space, there are fewer features than samples (n_features < n_samples),
          whereas a high-dimensional space has significantly more features than samples (n_features >> n_samples).
          The choice of metric depends on the specific application, with the following general guidelines:

          * ``euclidean``: Effective in low-dimensional spaces or when direct distances are meaningful.
          * ``manhattan``: Useful when differences along individual dimensions are important, or in the presence of outliers.
          * ``cosine``: Recommended for high-dimensional spaces (e.g., n_features >> n_samples), as it evaluates
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
        check_match_labels_markers(label_pos=label_pos, label_unl=label_unl, label_neg=label_neg)
        labels = ut.check_labels(
            labels=labels, vals_required=[label_pos, label_unl],
            allow_other_vals=(label_neg is not None),
            str_add=f"dPULearn expects positives as {label_pos} ('label_pos') and unlabeled as "
                    f"{label_unl} ('label_unl'); pre-labeled negatives are optional via 'label_neg'. "
                    f"Set label_unl=0 to pass a standard {{0, 1}} encoding.",
        )
        check_match_labels_markers_present(labels=labels, label_pos=label_pos,
                                           label_unl=label_unl, label_neg=label_neg)
        # Normalize the user encoding to the internal 1 (positive) / 2 (unlabeled) / 0 (negative) contract
        labels = normalize_pu_labels(labels=labels, label_pos=label_pos,
                                     label_unl=label_unl, label_neg=label_neg)
        n_pre_neg = int(np.sum(labels == 0))
        # Resolve how many negatives to identify from the unlabeled pool (n_neg total vs n_unl_to_neg direct)
        n_to_identify = resolve_n_to_identify(n_neg=n_neg, n_unl_to_neg=n_unl_to_neg, n_pre_neg=n_pre_neg)
        check_n_to_identify(labels=labels, n_to_identify=n_to_identify, label_unl=2)
        check_metric(metric=metric)
        check_n_components(n_components=n_components)
        ut.check_match_X_labels(X=X, labels=labels)
        check_match_X_n_components(X=X, n_components=n_components)
        # Compute average distance for threshold-based filtering (Yang et al., 2012, 2014; Nan et al. 2017)
        args = dict(X=X, labels=labels, n_to_identify=n_to_identify,
                    label_neg=0, label_pos=1, label_unl=2)
        if metric is not None:
            new_labels, df_pu = get_neg_via_distance(**args, metric=metric)
        # Identify most far away negatives in PCA compressed feature space
        else:
            new_labels, df_pu = get_neg_via_pca(**args, n_components=n_components, **self._model_kwargs)
        # Set new labels
        self.labels_ = np.asarray(new_labels)
        self.df_pu_ = df_pu
        return self

    def mine_negatives(self,
                       X_pos: ut.ArrayLike2D,
                       X_unlabelled: ut.ArrayLike2D,
                       n_neg: Optional[int] = None,
                       n_unl_to_neg: Optional[int] = None,
                       metric: Optional[Literal["euclidean", "manhattan", "cosine"]] = None,
                       n_components: Union[float, int] = 0.80,
                       ) -> np.ndarray:
        """
        Mine reliable negatives from an unlabeled pool given the positives, in one call.

        Convenience wrapper around :meth:`dPULearn.fit` for the common positive/unlabeled
        setup: instead of stacking ``X_pos`` and ``X_unlabelled`` by hand, building a
        ``1`` (positive) / ``2`` (unlabeled) label vector, fitting, and slicing the mined
        rows back out by index, pass the two feature matrices separately and receive a
        **boolean mask over the rows of** ``X_unlabelled`` flagging the identified reliable
        negatives. The mask equals ``labels_[len(X_pos):] == 0`` from the manual stacking
        path exactly.

        After the call the instance is fitted: :attr:`dPULearn.labels_` (over the stacked
        ``X_pos`` then ``X_unlabelled``) and :attr:`dPULearn.df_pu_` are set, so the
        :class:`dPULearnPlot` methods work as usual.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X_pos : array-like, shape (n_pos, n_features)
            Feature matrix of the positive samples.
        X_unlabelled : array-like, shape (n_unl, n_features)
            Feature matrix of the unlabeled samples (the candidate pool). Must have the
            same number of features as ``X_pos``.
        n_neg : int, optional
            Total number of reliable negatives to identify from the unlabeled pool.
            Provide **exactly one** of ``n_neg`` or ``n_unl_to_neg`` (with no pre-labeled
            negatives the two are equivalent).
        n_unl_to_neg : int, optional
            Number of reliable negatives to identify directly from the unlabeled pool.
            Provide **exactly one** of ``n_neg`` or ``n_unl_to_neg``.
        metric : str or None, optional
            Distance metric for distance-based identification (``euclidean``,
            ``manhattan``, ``cosine``). If ``None``, PCA-based identification is performed.
        n_components : int or float, default=0.80
            Number of principal components (int >= 1) or fraction of variance covered
            (float in ``(0.0, 1.0)``) when PCA is applied.

        Returns
        -------
        mask_neg : array-like, shape (n_unl,)
            Boolean mask over the rows of ``X_unlabelled``: ``True`` marks an identified
            reliable negative. ``X_unlabelled[mask_neg]`` are the mined negatives.

        Notes
        -----
        * This is purely additive sugar: it stacks the inputs and calls
          :meth:`dPULearn.fit` with ``label_pos=1`` / ``label_unl=2`` internally, so the
          identification result is identical to the manual path.

        See Also
        --------
        * :meth:`dPULearn.fit`: the underlying fit on a stacked matrix and label vector.
        * :func:`get_labels`: derive a binary label vector from a sequence DataFrame.

        Examples
        --------
        .. include:: examples/dpul_mine_negatives.rst
        """
        # Check input (the >=3 sample floor applies to the stacked matrix, enforced by
        # 'fit' below, so per-matrix validation only coerces + checks the feature dimension;
        # this keeps mine_negatives accepting exactly what the manual stacking path accepts)
        X_pos = ut.check_X(X=X_pos, X_name="X_pos", min_n_samples=1)
        X_unlabelled = ut.check_X(X=X_unlabelled, X_name="X_unlabelled", min_n_samples=1)
        check_match_X_pos_X_unlabelled(X_pos=X_pos, X_unlabelled=X_unlabelled)
        # Stack positives over the unlabeled pool and fit with the package PU markers
        n_pos = X_pos.shape[0]
        X = np.vstack([X_pos, X_unlabelled])
        labels = np.array([1] * n_pos + [2] * X_unlabelled.shape[0])
        self.fit(X=X, labels=labels, label_pos=1, label_unl=2,
                 n_neg=n_neg, n_unl_to_neg=n_unl_to_neg,
                 metric=metric, n_components=n_components)
        # Slice the mined reliable negatives (label 0) back out of the unlabeled block
        mask_neg = np.asarray(self.labels_)[n_pos:] == 0
        return mask_neg

    @staticmethod
    def eval(X: ut.ArrayLike2D,
             list_labels: ut.ArrayLike2D,
             names_datasets: Optional[List[str]] = None,
             X_neg: Optional[ut.ArrayLike2D] = None,
             comp_kld: bool = False,
             n_jobs: Optional[int] = None
             ) -> pd.DataFrame:
        """
        Evaluates the quality of different sets of identified negatives.

        The quality is assessed regarding two quality groups:

        * **Homogeneity** within the reliably identified negatives (0)
        * **Dissimilarity** between the reliably identified negatives and the groups
          of positive samples ('pos'), unlabeled samples ('unl'), and a ground-truth negative
          ('neg') sample group if provided by ``X_neg``

        .. versionadded:: 0.1.0

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
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for each set of identified negatives from ``list_labels``. For each set, statistical
            measures were averaged across all features.

        Notes
        -----
        ``df_eval`` includes the following columns:

        * 'name': Name of the dataset if ``names_datasets`` is provided (typically named by identification approach).
        * 'n_rel_neg': Number of identified negatives.
        * 'avg_STD': Average standard deviation (STD) assessing homogeneity of identified negatives.
          Lower values indicate greater homogeneity.
        * 'avg_IQR': Average interquartile range (IQR) assessing homogeneity of identified negatives.
          Lower values suggest greater homogeneity.
        * 'avg_abs_AUC_pos' / 'avg_abs_AUC_unl' / 'avg_abs_AUC_neg': Average absolute area under the curve (AUC)
          assessing the dissimilarity between the set of identified negatives and each other group (positives,
          unlabeled, ground-truth negatives). Higher values indicate greater dissimilarity.
        * 'avg_KLD_pos' / 'avg_KLD_unl' / 'avg_KLD_neg': Average Kullback-Leibler Divergence (KLD) assessing the
          dissimilarity of distributions between the set of identified negatives and each other group. Higher
          values indicate greater dissimilarity. These columns are omitted if ``comp_kld`` is set to ``False``.

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
        ut.check_match_X_list_labels(X=X, list_labels=list_labels, check_variability=comp_kld, vals_required=[0])
        ut.check_match_list_labels_names_datasets(list_labels=list_labels, names_datasets=names_datasets)
        check_match_X_X_neg(X=X, X_neg=X_neg)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # Evaluation for homogeneity within negatives and alignment of distribution with other datasets
        df_eval = eval_identified_negatives(X=X, list_labels=list_labels, names_datasets=names_datasets,
                                            X_neg=X_neg, comp_kld=comp_kld, n_jobs=n_jobs)
        return df_eval

    @staticmethod
    def compare_sets_negatives(list_labels: ut.ArrayLike1D,
                               names_datasets: Optional[List[str]] = None,
                               df_seq: Optional[pd.DataFrame] = None,
                               remove_non_neg : bool = True,
                               return_upset_data: bool = False
                               ) -> pd.DataFrame:
        """
        Create DataFrame for comparing sets of identified negatives.

        Optionally, data format can be created for Upset Plots, which are useful for visualizing the intersection
        and unique elements across these sets.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        list_labels : array-like, shape (n_datasets,)
            List of dataset labels for samples in ``X`` obtained by the :meth:`dPULearn.fit` method.
            Label values should be either 0 (identified negative), 1 (positive) or 2 (unlabeled). Must contain 0.
        names_datasets : list, optional
            List of dataset names corresponding to ``list_labels``.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences, for the entries
            corresponding to the ``labels`` of ``list_labels``.
        remove_non_neg : bool, default=True
            If ``True``, all rows are removed that do not contain identified negatives in any provided dataset.
        return_upset_data : bool, default=False
            Whether to return a DataFrame for Upset Plot (if ``True``) or for a general comparison of sets of negatives.

        Returns
        -------
        pd.DataFrame or pd.Series
            * If ``return_upset_data=False`` (default):
              Returns a pd.DataFrame (`df_neg_comp`) that combines ``df_seq`` (if provided) with a comparison of the
              negative sets for a general analysis.
            * If ``return_upset_data=True``:
              Returns a pd.Series (`upset_data`) formatted for generating Upset Plots, containing group
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
