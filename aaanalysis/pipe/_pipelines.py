"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``predict_samples`` golden
pipeline: a thin, stateless multi-model comparison harness. It cross-validates every
(feature set x model) combination over one ``df_seq`` and returns the fitted predictors together
with a tidy comparison table — a convenience facade over the explicit
``feature_matrix`` -> estimator ``fit`` / ``cross_validate`` chain.
"""
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature
from aaanalysis.prediction import AAPredPlot

# Comparison metrics scored in one cross_validate pass (sklearn scoring keys); balanced_accuracy is
# the headline (imbalance-aware, the paper's bACC). Each becomes a <metric>_mean / <metric>_std
# column pair in df_eval, mirroring the find_features sweep table.
_METRICS = ["balanced_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc"]
# String shortcuts for the model vocabulary (mirrors the find_features presets + extra_trees), so a
# user can write models=["rf", "svm"] instead of importing the estimator classes.
_STR_MODELS = {
    ut.MODEL_SVM: lambda rs: SVC(class_weight="balanced", probability=True, random_state=rs),
    ut.MODEL_RF: lambda rs: RandomForestClassifier(random_state=rs),
    ut.MODEL_LOG_REG: lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
    "extra_trees": lambda rs: ExtraTreesClassifier(random_state=rs),
}
ModelLike = Union[str, BaseEstimator]
ModelsArg = Union[ModelLike, List[ModelLike], Dict[str, ModelLike]]
FeatArg = Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]]


# I Helper Functions
def _default_models(random_state=None):
    """The core-sklearn default comparison set (tree / kernel / linear families; no heavy deps)."""
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(random_state=random_state),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
    }


def _inject_random_state(est, random_state=None):
    """Clone ``est`` and set ``random_state`` only where it is exposed and the user left it unset."""
    est = clone(est)
    params = est.get_params()
    if random_state is not None and params.get("random_state", "missing") is None:
        est.set_params(random_state=random_state)
    return est


def _dedup_name(name, taken):
    """Return ``name`` (or ``name_2`` / ``name_3`` ...) so it does not collide with ``taken``."""
    base, k, out = name, 2, name
    while out in taken:
        out, k = f"{base}_{k}", k + 1
    return out


def _resolve_named_models(models=None, random_state=None):
    """Normalize ``models`` into an ordered ``{name: estimator}`` dict.

    Accepts ``None`` (the core default set), a single str/estimator, a list of str/estimator
    (names taken from the estimator class, de-duplicated), or a ``{name: str|estimator}`` dict.
    """
    if models is None:
        return _default_models(random_state=random_state)
    if isinstance(models, dict):
        items = list(models.items())
    elif isinstance(models, (list, tuple)):
        items = [(None, m) for m in models]
    else:
        items = [(None, models)]
    if len(items) == 0:
        raise ValueError("'models' should contain at least one model (got an empty collection).")
    out: Dict[str, BaseEstimator] = {}
    for name, m in items:
        if isinstance(m, str):
            if m not in _STR_MODELS:
                raise ValueError(f"'models' string ('{m}') should be one of {sorted(_STR_MODELS)}.")
            est, name = _STR_MODELS[m](random_state), (name or m)
        elif isinstance(m, BaseEstimator):
            est, name = _inject_random_state(m, random_state=random_state), (name or type(m).__name__)
        else:
            raise ValueError(f"'models' item ({m}) should be an sklearn estimator or a model-name string.")
        out[_dedup_name(str(name), out)] = est
    return out


def _resolve_named_feat(list_df_feat=None):
    """Normalize ``list_df_feat`` into an ordered ``{name: df_feat}`` dict (each validated)."""
    if isinstance(list_df_feat, dict):
        items = list(list_df_feat.items())
    elif isinstance(list_df_feat, (list, tuple)):
        items = [(f"feat{i + 1}", df) for i, df in enumerate(list_df_feat)]
    else:
        items = [("features", list_df_feat)]
    if len(items) == 0:
        raise ValueError("'list_df_feat' should contain at least one feature DataFrame.")
    out: Dict[str, pd.DataFrame] = {}
    for name, df in items:
        out[_dedup_name(str(name), out)] = ut.check_df_feat(df_feat=df)
    return out


def _cv_row(est, X, labels, n_cv=5):
    """Cross-validate one estimator and return ``{metric: (mean, std)}`` over all metrics."""
    res = cross_validate(clone(est), X, y=labels, cv=n_cv, scoring=list(_METRICS))
    return {m: (float(np.mean(res["test_" + m])), float(np.std(res["test_" + m]))) for m in _METRICS}


def _comparison_long(df_eval):
    """Reshape the wide ``predict_samples`` table into the long ``AAPredPlot.eval(kind='eval')`` form.

    Emits one row per ``(model, metric)`` with the ``COLS_EVAL_PRED`` columns. When more than one
    feature set is compared, the model label is prefixed with the feature-set name so each
    ``(feature_set, model)`` cell is a distinct hued bar.
    """
    metrics = [c[:-len("_mean")] for c in df_eval.columns if c.endswith("_mean")]
    multi = df_eval["feature_set"].nunique() > 1
    rows = []
    for _, r in df_eval.iterrows():
        model = f"{r['feature_set']} · {r['model']}" if multi else r["model"]
        for m in metrics:
            rows.append({ut.COL_MODEL: model, ut.COL_METRIC: m, ut.COL_PRINCIPLE: ut.STR_PRINCIPLE_CV,
                         ut.COL_SCORE: r[m + "_mean"], ut.COL_SCORE_STD: r[m + "_std"]})
    return pd.DataFrame(rows)


# II Main Functions
def predict_samples(list_df_feat: FeatArg,
                    df_seq: pd.DataFrame,
                    labels: ut.ArrayLike1D,
                    models: Optional[ModelsArg] = None,
                    n_cv: int = 5,
                    plot: bool = True,
                    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                    dict_color: Optional[Dict[str, str]] = None,
                    baseline: Optional[Union[int, float]] = None,
                    random_state: Optional[int] = None,
                    n_jobs: Optional[int] = None,
                    verbose: bool = False,
                    ) -> Tuple[Dict[Tuple[str, str], BaseEstimator], Optional[Axes], pd.DataFrame]:
    """
    Train and compare predictors across feature sets and models in one call.

    A thin, stateless facade over the explicit primitive path. For every combination of a feature
    set (in ``list_df_feat``) and a model (in ``models``), it rebuilds the feature matrix ``X`` from
    the feature identifiers (via :meth:`SequenceFeature.feature_matrix`), cross-validates the model,
    and refits it on all samples to give a deployable predictor. The fitted predictors are returned
    as a dictionary together with a tidy cross-validated comparison table.

    Parameters
    ----------
    list_df_feat : pd.DataFrame or list or dict
        One or several feature sets. A single feature DataFrame (with a ``feature`` column of feature
        identifiers, e.g. from :meth:`CPP.run` or :func:`load_features`), a list of such DataFrames,
        or a ``{name: df_feat}`` dict. Lists / single frames are auto-named (``feat1`` ...).
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        Sequence DataFrame, row-aligned to ``labels``. The sequence parts are derived from it once
        and shared across all feature sets.
    labels : array-like, shape (n_samples,)
        Class labels for the samples (typically, 1=positive/test, 0=negative/reference).
    models : str, estimator, list, or dict, optional
        The models to train. A scikit-learn estimator instance, a model-name string (one of
        ``'rf'``, ``'svm'``, ``'log_reg'``, ``'extra_trees'``), a list of either (names taken from
        the estimator class), or a ``{name: estimator}`` dict. If ``None``, a core default set is
        compared: random forest, extra trees, SVM, and logistic regression.
    n_cv : int, default=5
        Number of cross-validation folds, must be > 1 and <= the smallest class count.
    plot : bool, default=True
        If ``True``, draw the model comparison bar plot (hue = model, one bar group per metric,
        with cross-validation ``std`` error bars) from the comparison table and return its ``Axes``.

        .. versionadded:: 1.1.0
    figsize : tuple, optional
        Figure size of the comparison plot; a per-kind default is used when ``None``.

        .. versionadded:: 1.1.0
    dict_color : dict, optional
        Mapping ``model -> color`` for the comparison-plot bars; defaults to the house palette.

        .. versionadded:: 1.1.0
    baseline : int or float, optional
        y-value of a dashed chance line on the comparison plot (e.g. ``0.5``); none when ``None``.

        .. versionadded:: 1.1.0
    random_state : int, optional
        The seed used by the random number generator. If a positive integer, results of stochastic
        processes are reproducible. Injected into each model only where the estimator exposes a
        ``random_state`` parameter and the user left it unset.
    n_jobs : int, optional
        Number of CPU cores (>=1) for building the feature matrix. If ``None``, the optimized number
        is used.
    verbose : bool, default=False
        If ``True``, verbose progress information is printed.

    Returns
    -------
    predictors : dict
        The fitted predictors, keyed by ``(feature_set, model)``. Each value is the corresponding
        scikit-learn estimator refit on all samples.
    fig_ax : matplotlib.axes.Axes or None
        The comparison-plot ``Axes`` when ``plot=True``, else ``None`` (keeping the uniform
        ``(results, figs, evals)`` pipeline return shape).
    df_eval : pd.DataFrame, shape (n_feature_sets * n_models, n_eval_info)
        Comparison table, one row per ``(feature_set, model)``: the descriptors, ``n_features``, one
        ``<metric>_mean`` / ``<metric>_std`` column pair per metric (balanced_accuracy, accuracy,
        f1, precision, recall, roc_auc), ``is_shap_ready`` (the predictor exposes feature
        importances, so it can feed :func:`explain_features`), and ``is_best`` (the single highest
        ``balanced_accuracy_mean`` row).

    See Also
    --------
    * :meth:`SequenceFeature.feature_matrix` for building ``X`` from feature identifiers.
    * :func:`find_features` for producing a ``df_feat`` feature set to predict from.
    * :func:`explain_features` for SHAP-explaining a SHAP-ready predictor's feature set.

    Examples
    --------
    .. include:: examples/aap_predict_samples.rst
    """
    # Validate (thin facade: the wrapped primitives validate the rest)
    ut.check_df_seq(df_seq=df_seq)
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    ut.check_bool(name="plot", val=plot)
    ut.check_figsize(figsize=figsize, accept_none=True)
    ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
    ut.check_number_val(name="baseline", val=baseline, accept_none=True)
    ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True, accept_none=True)
    ut.check_bool(name="verbose", val=verbose)
    dict_df_feat = _resolve_named_feat(list_df_feat=list_df_feat)
    dict_models = _resolve_named_models(models=models, random_state=random_state)
    # Build the shared sequence parts once, then validate labels against the row count
    sf = SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    ut.check_labels(labels=labels, len_required=len(df_parts))
    # Cross-validate and refit every (feature set x model) cell
    predictors: Dict[Tuple[str, str], BaseEstimator] = {}
    rows = []
    for feat_name, df_feat in dict_df_feat.items():
        X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts, n_jobs=n_jobs)
        for model_name, est in dict_models.items():
            scores = _cv_row(est, X, labels, n_cv=n_cv)
            predictor = clone(est).fit(X, labels)
            predictors[(feat_name, model_name)] = predictor
            row = {"feature_set": feat_name, "model": model_name, "n_features": int(X.shape[1])}
            for m in _METRICS:
                row[m + "_mean"], row[m + "_std"] = scores[m]
            row["is_shap_ready"] = hasattr(predictor, "feature_importances_")
            rows.append(row)
            if verbose:
                ut.print_out(f"  {feat_name} x {model_name}: "
                             f"bACC={row['balanced_accuracy_mean']:.3f}")
    df_eval = pd.DataFrame(rows)
    best = int(df_eval["balanced_accuracy_mean"].to_numpy().argmax())
    df_eval["is_best"] = [i == best for i in range(len(df_eval))]
    # Uniform (results, figs, evals) pipeline return triple; draw the model comparison when plot=True
    ax = None
    if plot:
        _, ax = AAPredPlot().eval(_comparison_long(df_eval), kind="eval", figsize=figsize,
                                  dict_color=dict_color, baseline=baseline)
    return predictors, ax, df_eval
