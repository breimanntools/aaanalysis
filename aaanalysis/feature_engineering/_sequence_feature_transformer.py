"""
This is a script for the frontend of the SequenceFeatureTransformer class: a scikit-learn
transformer that runs leak-free CPP feature selection inside a Pipeline / cross_val_score.
"""
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import aaanalysis.utils as ut
from ._sequence_feature import SequenceFeature
from ._cpp import CPP
from aaanalysis.data_handling import load_scales


# I Helper Functions
def _check_supervised_labels(labels=None, label_test=1, label_ref=0):
    """Check that labels are usable by CPP's test-vs-reference selection."""
    labels = ut.check_labels(labels=labels)
    classes = set(int(c) for c in np.unique(labels))
    for name, val in (("label_test", label_test), ("label_ref", label_ref)):
        if val not in classes:
            raise ValueError(f"'{name}' ({val}) is not present in the labels {sorted(classes)}.")
    if label_test == label_ref:
        raise ValueError(f"'label_test' and 'label_ref' should differ (both {label_test}).")
    return labels


# II Main Functions
class SequenceFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    SequenceFeatureTransformer: leak-free CPP feature selection as a scikit-learn transformer.

    Wraps the ``get_df_parts`` -> :meth:`CPP.run` -> ``feature_matrix`` chain (see
    :meth:`SequenceFeature.feature_matrix`) behind the scikit-learn ``fit`` / ``transform`` API so
    CPP feature **selection happens on the training fold only**: :meth:`fit` selects from ``(X, y)``
    and stores them; :meth:`transform` applies the **same** features to produce the numeric matrix
    ``X``. Dropped inside a :class:`sklearn.pipeline.Pipeline` (or
    :func:`sklearn.model_selection.cross_val_score`), the test fold never influences which features
    are chosen — the honest, leak-free counterpart of selecting on the full labeled set.

    ``X`` is a ``df_seq`` DataFrame (sequence information, one row per protein) or a pre-built
    ``df_parts``; ``y`` are the class labels. Selection is binary (one ``label_test`` vs one
    ``label_ref``), configured by the constructor's CPP levers.

    .. warning::

        **Experimental.** Part of the evolving ``aaanalysis.pipe`` ergonomics layer; its API may
        change between minor releases without the usual deprecation cycle.

    .. versionadded:: 1.1.0

    Notes
    -----
    * Follows the scikit-learn estimator contract: the constructor only stores parameters (so the
      estimator is cloneable), all validation happens in :meth:`fit`, and learned state carries a
      trailing underscore (``features_``, ``df_feat_``).

    See Also
    --------
    * :class:`CPP` for the underlying feature selection.
    * :meth:`SequenceFeature.feature_matrix` for the parts-to-matrix step applied by ``transform``.
    """

    def __init__(self,
                 split_kws: Optional[dict] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 n_filter: int = 100,
                 label_test: int = 1,
                 label_ref: int = 0,
                 max_overlap: float = 0.5,
                 max_cor: float = 0.5,
                 simplify: bool = False,
                 n_jobs: Optional[int] = 1,
                 random_state: Optional[int] = None,
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
        split_kws : dict, optional
            CPP split configuration (see :meth:`SequenceFeature.get_split_kws`). ``None`` uses the
            CPP default (Segment + Pattern + PeriodicPattern).
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            Amino-acid scales used for feature selection and the feature matrix. ``None`` uses the
            bundled AAontology scales (:func:`load_scales`).
        n_filter : int, default=100
            Number of features CPP selects (passed to :meth:`CPP.run`).
        label_test : int, default=1
            Class label of the test/positive group for CPP selection.
        label_ref : int, default=0
            Class label of the reference/negative group for CPP selection.
        max_overlap : float, default=0.5
            Maximum feature overlap allowed during CPP selection.
        max_cor : float, default=0.5
            Maximum feature correlation allowed during CPP selection.
        simplify : bool, default=False
            If ``True``, refine the selected feature set with :meth:`CPP.simplify` after selection.
        n_jobs : int, optional
            Number of CPU cores for the CPP run and the feature-matrix build. ``None`` uses the
            optimized number; default ``1``.
        random_state : int, optional
            Seed for the CPP run, for reproducibility.
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.

        Examples
        --------
        .. include:: examples/sft.rst
        """
        self.split_kws = split_kws
        self.df_scales = df_scales
        self.n_filter = n_filter
        self.label_test = label_test
        self.label_ref = label_ref
        self.max_overlap = max_overlap
        self.max_cor = max_cor
        self.simplify = simplify
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # fit needs y (CPP selection is supervised), and X is a sequence/parts DataFrame rather
        # than a validated numeric array, so opt out of scikit-learn's numeric-array validation.
        tags.target_tags.required = True
        tags.no_validation = True
        return tags

    def _to_df_parts(self, X):
        """Resolve ``X`` to a ``df_parts``: build it from a ``df_seq`` or accept one directly.

        The index is reset first (row order preserved) so a fold subset handed in by scikit-learn's
        cross-validation — whose index is non-contiguous — does not trip the numeric-index warning.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("'SequenceFeatureTransformer' expects a 'df_seq' (or 'df_parts') "
                             f"DataFrame as X, got {type(X).__name__}.")
        X = X.reset_index(drop=True)
        if ut.COL_SEQ in X.columns:
            return SequenceFeature(verbose=False).get_df_parts(df_seq=X)
        return X

    def fit(self, X, y=None):
        """
        Select CPP features from ``(X, y)`` (the training fold) and store them.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_seq_info) or (n_samples, n_parts)
            A ``df_seq`` (sequence information) or a pre-built ``df_parts``.
        y : array-like, shape (n_samples,)
            Binary class labels (must contain ``label_test`` and ``label_ref``). Required.

        Returns
        -------
        self : SequenceFeatureTransformer
            The fitted transformer.

        Examples
        --------
        .. include:: examples/sft_fit.rst
        """
        # Check input
        if y is None:
            raise ValueError("'SequenceFeatureTransformer.fit' requires labels 'y' — CPP feature "
                             "selection is supervised.")
        ut.check_number_range(name="n_filter", val=self.n_filter, min_val=1, just_int=True)
        ut.check_bool(name="simplify", val=self.simplify)
        ut.check_number_range(name="max_overlap", val=self.max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_cor", val=self.max_cor, min_val=0.0, max_val=1.0, just_int=False)
        df_parts = self._to_df_parts(X)
        labels = _check_supervised_labels(labels=y, label_test=self.label_test, label_ref=self.label_ref)
        if len(labels) != len(df_parts):
            raise ValueError(f"'y' n_samples ({len(labels)}) should match X n_samples ({len(df_parts)}).")
        # Select features (CPP run on this fold only)
        df_scales = self.df_scales if self.df_scales is not None else load_scales(name="scales")
        cpp = CPP(df_parts=df_parts, split_kws=self.split_kws, df_scales=df_scales,
                  random_state=self.random_state, verbose=False)
        df_feat = cpp.run(labels=labels, label_test=self.label_test, label_ref=self.label_ref,
                          n_filter=self.n_filter, max_cor=self.max_cor, max_overlap=self.max_overlap,
                          n_jobs=self.n_jobs)
        if self.simplify and df_feat is not None and len(df_feat) > 1:
            df_feat = cpp.simplify(df_feat=df_feat, labels=labels,
                                   label_test=self.label_test, label_ref=self.label_ref)
        self.df_feat_ = df_feat.reset_index(drop=True)
        self.features_ = self.df_feat_[ut.COL_FEATURE].tolist()
        self._df_scales_ = df_scales
        return self

    def transform(self, X):
        """
        Build the CPP feature matrix ``X`` from the features selected in :meth:`fit`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_seq_info) or (n_samples, n_parts)
            A ``df_seq`` or ``df_parts`` (same kind as passed to :meth:`fit`).

        Returns
        -------
        X_out : np.ndarray, shape (n_samples, n_selected_features)
            The numeric feature matrix for the features selected in :meth:`fit` (same column order
            as ``features_``).

        Examples
        --------
        .. include:: examples/sft_transform.rst
        """
        check_is_fitted(self, "features_")
        df_parts = self._to_df_parts(X)
        X_out = SequenceFeature(verbose=False).feature_matrix(
            features=self.features_, df_parts=df_parts, df_scales=self._df_scales_, n_jobs=self.n_jobs)
        return np.asarray(X_out)

    def get_feature_names_out(self, input_features=None):
        """
        Output feature names — the CPP feature ids selected in :meth:`fit` (one per output column).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        input_features : array-like of str, optional
            Ignored (the output names are the selected CPP feature ids, not a function of the input
            columns); present for scikit-learn ``get_feature_names_out`` compatibility and to enable
            ``set_output(transform="pandas")``.

        Returns
        -------
        feature_names_out : np.ndarray of str, shape (n_selected_features,)
            The selected CPP feature ids, in the column order of :meth:`transform`'s output.

        Examples
        --------
        .. include:: examples/sft_get_feature_names_out.rst
        """
        check_is_fitted(self, "features_")
        return np.asarray(self.features_, dtype=object)
