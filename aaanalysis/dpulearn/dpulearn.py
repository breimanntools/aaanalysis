"""
This is a script for deterministic Positive-Unlabeled (PU) Learning (dPULearn) class
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import math
import warnings
import aaanalysis._utils as ut

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe

LIST_METRICS = ['euclidean', 'manhattan', 'cosine']

# TODO better example in fit
# TODO more check functions, improve with testing

# I Helper Functions
# Check functions
def _check_metric(metric=None):
    """"""
    if metric is not None and metric not in LIST_METRICS:
        raise ValueError(f"'metric' ({metric}) should be None or one of following: {LIST_METRICS}")


def _check_df_seq(df_seq=None, col_class="class"):
    """"""
    if df_seq is not None:
        if col_class not in df_seq:
            columns = list(df_seq)
            raise ValueError(f"'col_class' ({col_class}) must be a column in 'df_seq': {columns}")
        if not df_seq.index.is_unique:
            df_seq = df_seq.reset_index(drop=True)
            warnings.warn("'df_seq' index was not unique. The index has been reset.", UserWarning)
    return df_seq


def _check_labels(labels=None, verbose=False, label_pos=None):
    # Check if labels is an array or list
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError(f"'labels' should be a list or a NumPy array, not {type(labels)}")

    # Check if labels contain integers
    if not all(isinstance(label, int) for label in labels):
        raise ValueError("All elements in 'labels' should be integers")
    # Check if label_pos in labels
    if label_pos not in labels:
        str_error = f"'label_pos' ('{label_pos}', default=1) should be in 'labels' with ({list(np.unique(labels))})"
        raise ValueError(str_error)
    # Check if integers start with 0
    min_label = min(labels)
    if min_label != 0 and verbose:
        warnings.warn(f"The smallest label is {min_label}, typically should start with 0")

    # Check if integers are consecutive
    unique_labels = sorted(set(labels))
    if any(unique_labels[i] - unique_labels[i - 1] != 1 for i in range(1, len(unique_labels))):
        if verbose:
            warnings.warn("Labels are not consecutive integers")
    if isinstance(labels, list):
        labels = np.array(labels)
    return labels


def _check_n_neg(labels=None, n_neg=None, label_pos=None, label_neg=None):
    """"""
    ut.check_non_negative_number(name='n_neg', val=n_neg, min_val=1)
    if sum([x == label_neg for x in labels]) > 0:
        raise ValueError(f"'labels' should not contain labels for negatives ({label_neg})")
    n_pos = sum([x == label_pos for x in labels])
    n_unl = sum([x != label_pos for x in labels])
    if n_pos < n_neg:
        raise ValueError(f"Number of positive labels ({n_pos}) should higher than 'n_neg' ({n_neg})")
    if n_unl < n_neg:
        raise ValueError(f"Number of unlabeled labels ({n_unl}) should higher than 'n_neg' ({n_neg})")


# Pre-processing helper functions
def _get_label_neg(labels=None):
    """"""
    label_neg = 0 if 0 not in labels else max(labels) + 1
    return label_neg


# II Main Functions
def _get_neg_via_distance(X=None, labels=None, metric="euclidean", n_neg=None,
                          df_seq=None, col_class=None,
                          label_neg=0, label_pos=1, name_neg=None):
    """Identify distant samples from positive mean as reliable negatives based on a specified distance metric.

    Parameters:
    - X: np.ndarray, The input feature matrix of shape (n_samples, n_features).
    - labels: np.ndarray, Class labels for each sample.
    - metric: str, Distance metric ('euclidean', 'manhattan', etc.).
    - n_neg: int, Total number of negatives to identify.
    - df_seq: pd.DataFrame, Dataframe to store distance values.
    - col_class: str, Column name in df_seq to store class information.
    - label_neg, label_pos: int/str, Labels for the negative and positive classes.
    - name_neg: str, Prefix for naming identified negatives.

    Returns:
    - new_labels: np.ndarray, Updated array of labels.
    - df_seq: pd.DataFrame, Dataframe with updated class information and distances.
    """
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    # Compute the average distances to the positive data points
    avg_dist = pairwise_distances(X[mask_pos], X, metric=metric).mean(axis=0)
    # Select negatives based on largest average distance to positives
    top_indices = np.argsort(avg_dist[mask_unl])[::-1][:n_neg]
    new_labels = labels.copy()
    new_labels[top_indices] = label_neg
    # Update classes in df_seq and add average distance to positives
    if df_seq is not None:
        df_seq[metric] = avg_dist
        df_seq.loc[top_indices, col_class] = name_neg
    return new_labels, df_seq


def _get_neg_via_pca(X=None, labels=None, n_components=0.8, n_neg=None,
                     df_seq=None, col_class=None,
                     label_neg=0, label_pos=1, name_neg=None, **pca_kwargs):
    """Identify distant samples from positive mean as reliable negatives in PCA-compressed feature spaces.

    Parameters:
    - X: np.ndarray, The input feature matrix of shape (n_samples, n_features).
    - labels: np.ndarray, Class labels for each sample.
    - n_components: float/int, Number of principal components or the ratio of total explained variance.
    - n_neg: int, Total number of negatives to identify.
    - df_seq: pd.DataFrame, Dataframe to store PCA values.
    - col_class: str, Column name in df_seq to store class information.
    - label_neg, label_pos: int/str, Labels for the negative and positive classes.
    - name_neg: str, Prefix for naming identified negatives.
    - pca_kwargs: dict, Additional keyword arguments for PCA.

    Returns:
    - new_labels: np.ndarray, Updated array of labels.
    - df_seq: pd.DataFrame, Dataframe with updated class information.
    """
    # Principal component analysis
    pca = PCA(n_components=n_components, **pca_kwargs)
    pca.fit(X.T)
    list_exp_var = pca.explained_variance_ratio_
    _columns_pca = [f"PC{n+1} ({round(exp_var*100, 1)}%)" for n, exp_var in zip(range(len(list_exp_var)), list_exp_var)]

    # Number of negatives based on explained variance
    _list_n_neg = [math.ceil(n_neg * x / sum(list_exp_var)) for x in list_exp_var]
    _list_n_cumsum = np.cumsum(np.array(_list_n_neg))
    list_n_neg = [n for n, cs in zip(_list_n_neg, _list_n_cumsum) if cs <= n_neg]
    if sum(list_n_neg) != n_neg:
        list_n_neg.append(n_neg - sum(list_n_neg))
    columns_pca = _columns_pca[0:len(list_n_neg)]
    df_seq[columns_pca] = pca.components_.T[:, 0:len(columns_pca)]

    # Get mean of positive data for each component
    mask_pos = labels == label_pos
    mask_unl = labels != label_pos
    pc_means = df_seq[mask_pos][columns_pca].mean(axis=0)

    # Select negatives based on absolute difference to mean of positives for each component
    new_labels = labels.copy()
    _df = df_seq.copy()
    for col_pc, mean_pc, n in zip(columns_pca, pc_means, list_n_neg):
        name_reg_pc = f"{name_neg}_{col_pc.split(' ')[0]}"
        col_dif = f"{col_pc}_abs_dif"

        # Calculate absolute difference to the mean for each sample in the component
        _df[col_dif] = np.abs(df_seq[col_pc] - mean_pc)

        # Sort and take top n indices
        top_indices = _df.loc[mask_unl].sort_values(by=col_dif).tail(n).index

        # Update labels and masks
        new_labels[top_indices] = label_neg
        mask_unl[top_indices] = False

        # Update classes in df_seq
        if df_seq is not None:
            df_seq.loc[top_indices, col_class] = name_reg_pc
    return new_labels, df_seq


class dPULearn:
    """
    Deterministic Positive-Unlabeled (dPULearn) model.

    dPULearn offers a deterministic approach for Positive-Unlabeled (PU) learning. The model primarily employs
    Principal Component Analysis (PCA) to reduce the dimensionality of the feature space. Based on the most
    informative principal components (PCs), it then iteratively identifies reliable negatives from the set of
    unlabeled samples. These reliable negatives are those that are most distant from the positive samples in
    the feature space. Alternatively, reliable negatives can also be identified using distance metrics like
    Euclidean, Manhattan, or Cosine distance if specified.

    Parameters
    ----------
    verbose : bool, default=False
        Enable verbose output.
    n_components : float or int, default=0.80
        Number of components to cover a maximum percentage of total variance when PCA is applied.
    pca_kwargs : dict, default=None
        Additional keyword arguments to pass to PCA.
    metric : {'euclidean', 'manhattan', 'cosine'} or None, default=None
        The distance metric to use. If None, PCA-based identification is used.
        If a metric is specified, distance-based identification is performed.

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        Labels of each data point.

    Notes
    -----
    - The method is inspired by deterministic PU learning techniques and follows
        an information-theoretic PU learning approach.
    - If `metric` is specified, distance-based identification of reliable negatives is performed.
        Otherwise, PCA-based identification is used.
    - Cosine metric is recommended in high-dimensional spaces.

    """
    def __init__(self, verbose=False, n_components=0.80, pca_kwargs=None, metric=None):
        self.verbose = verbose
        # Arguments for Principal Component Analysis (PCA)-based identification
        self.n_components = n_components
        if pca_kwargs is None:
            pca_kwargs = dict()
        self.pca_kwargs = pca_kwargs
        # Arguments for distance-based identification
        _check_metric(metric=metric)
        self.metric = metric
        # Output parameters (will be set during model fitting)
        self.labels_ = None

    # Main method
    def fit(self, X, labels=None, n_neg=0, label_pos=1, name_neg="REL_NEG", df_seq=None, col_class="class"):
        """
        Fit the dPULearn model to identify reliable negative samples
        from the provided feature matrix and labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix where `n_samples` is the number of samples and `n_features` is the number of features.
        labels : array-like, shape (n_samples,), default=None
            Array of labels; positive samples should be indicated by `label_pos`.
        n_neg : int, default=0
            Number of negative samples to identify.
        label_pos : int or str, default=1
            Label indicating positive samples in the `labels` array.
        name_neg : str, default="REL_NEG"
            Name to assign to the newly identified negative samples.
        df_seq : DataFrame, default=None, optional
            DataFrame containing sequences; will be updated with new negative samples.
        col_class : str, default="class"
            Column name in `df_seq` where the class labels are stored.

        Returns
        -------
        df_seq : DataFrame
            DataFrame with the newly identified reliable negatives. Will be None if not provided.

        Notes
        -----
        Distance-based identification is used if `metric` is specified during class initialization.

        Examples
        --------
        Create small example data for dPUlearn containg positive ('pos', 1) and unlabeled ('unl', 2) data

        >>> import aaanalysis as aa
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = np.array([[0.2, 0.1], [0.3, 0.2], [0.2, 0.3], [0.5, 0.7]])
        >>> labels = np.array([1, 2, 2, 2])
        >>> df_seq = pd.DataFrame({
        ...     'sequence': ['ATGC', 'GCTA', 'ACTG', 'TACG'],
        ...     'class': ['pos', 'unl', 'unl', 'unl']})

        Use dPULearn in default mode (PC-based identification) and modify df_seq automatically

        >>> dpul = aa.dPULearn()
        >>> n_neg = 2
        >>> df_seq = dpul.fit(X=X, df_seq=df_seq, labels=labels, n_neg=n_neg)
        >>> labels = dpul.labels_   # Updated labels

        """
        ut.check_feat_matrix(X=X, labels=labels)
        df_seq = _check_df_seq(df_seq=df_seq, col_class=col_class)
        labels = _check_labels(labels=labels, verbose=self.verbose, label_pos=label_pos)
        label_neg = _get_label_neg(labels=labels)
        _check_n_neg(labels=labels, n_neg=n_neg, label_neg=label_neg, label_pos=label_pos)
        # Compute average distance for threshold-based filtering (Yang et al., 2012, 2014; Nan et al. 2017)
        args = dict(X=X, labels=labels, n_neg=n_neg,
                    df_seq=df_seq, col_class=col_class,
                    label_neg=label_neg, label_pos=label_pos, name_neg=name_neg)
        if self.metric is not None:
            new_labels, df_seq = _get_neg_via_distance(**args, metric=self.metric)
        # Identify most far away negatives in PCA compressed feature space
        else:
            new_labels, df_seq = _get_neg_via_pca(**args, n_components=self.n_components, **self.pca_kwargs)
        # Set new labels
        self.labels_ = new_labels
        return df_seq

