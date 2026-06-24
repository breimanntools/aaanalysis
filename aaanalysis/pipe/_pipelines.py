"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) golden pipelines:
thin, stateless one-call wrappers over the existing AAanalysis primitives.
"""
from typing import Optional, List, Type, Tuple, Union
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature
from aaanalysis.explainable_ai import TreeModel


# I Helper Functions


# II Main Functions
def predict_samples(df_feat: pd.DataFrame,
                    df_parts: pd.DataFrame,
                    labels: ut.ArrayLike1D,
                    list_model_classes: Optional[List[Type[Union[ClassifierMixin, BaseEstimator]]]] = None,
                    n_cv: int = 5,
                    random_state: Optional[int] = None,
                    n_jobs: Optional[int] = None,
                    verbose: bool = False,
                    ) -> Tuple[TreeModel, None, pd.DataFrame]:
    """
    Predict from a feature set in one call: build the feature matrix, fit tree models, and evaluate.

    A thin, stateless facade over the explicit primitive path. It rebuilds the feature matrix ``X``
    from the feature identifiers in ``df_feat`` (via :meth:`SequenceFeature.feature_matrix`), fits a
    :class:`TreeModel`, and returns the fitted model together with a cross-validated evaluation. The
    defaults are byte-identical to writing the three calls by hand.

    Parameters
    ----------
    df_feat : pd.DataFrame, shape (n_features, n_feature_info)
        Feature DataFrame with a ``feature`` column of feature identifiers (e.g. from :meth:`CPP.run`
        or :func:`load_features`).
    df_parts : pd.DataFrame, shape (n_samples, n_parts)
        Sequence parts DataFrame (from :meth:`SequenceFeature.get_df_parts`), row-aligned to ``labels``.
    labels : array-like, shape (n_samples,)
        Class labels for the samples (typically, 1=positive, 0=negative).
    list_model_classes : list of Type, optional
        Tree-based model classes passed to :class:`TreeModel`. If ``None``, the ``TreeModel`` default is used.
    n_cv : int, default=5
        Number of cross-validation folds for the evaluation, must be > 1 and ≤ the smallest class count.
    random_state : int, optional
        The seed used by the random number generator. If a positive integer, results of stochastic
        processes are reproducible.
    n_jobs : int, optional
        Number of CPU cores (>=1) for building the feature matrix. If ``None``, the optimized number is used.
    verbose : bool, default=False
        If ``True``, verbose progress information is printed.

    Returns
    -------
    model : TreeModel
        The fitted :class:`TreeModel` instance.
    figs : None
        Always ``None`` — prediction draws no plot by default (the slot reserved for a future
        evaluation plot keeps the uniform ``(results, figs, evals)`` pipeline return shape).
    df_eval : pd.DataFrame
        Cross-validated evaluation results for the fitted feature selection.

    See Also
    --------
    * :meth:`SequenceFeature.feature_matrix` for building ``X`` from feature identifiers.
    * :class:`TreeModel` for the underlying tree-based prediction and evaluation.

    Examples
    --------
    .. include:: examples/aap_predict_samples.rst
    """
    # Validate (thin facade: the wrapped primitives validate the rest)
    df_feat = ut.check_df_feat(df_feat=df_feat)
    ut.check_df_parts(df_parts=df_parts)
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True, accept_none=True)
    ut.check_bool(name="verbose", val=verbose)
    # Build the feature matrix from the feature identifiers in df_feat
    sf = SequenceFeature()
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts, n_jobs=n_jobs)
    # Fit tree-based models and cross-validated evaluation (byte-identical to the manual chain)
    tm = TreeModel(list_model_classes=list_model_classes, random_state=random_state, verbose=verbose)
    tm.fit(X, labels=labels, n_cv=n_cv)
    df_eval = tm.eval(X, labels=labels, list_is_selected=[tm.is_selected_], n_cv=n_cv)
    # Uniform (results, figs, evals) pipeline return triple; figs=None (no plot drawn)
    return tm, None, df_eval
