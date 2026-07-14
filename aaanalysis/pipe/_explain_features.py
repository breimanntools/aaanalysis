"""
This is a script for the frontend of the ``aaanalysis.pipe`` (ap) ``explain_features`` golden
pipeline (*pro*): a one-call SHAP explanation of an existing feature set. It rebuilds the feature
matrix from ``df_feat`` + ``df_seq``, fits a :class:`ShapModel`, attaches per-sample SHAP impact to
``df_feat``, and draws the SHAP-coloured feature map. Defaults are byte-identical to the explicit
``feature_matrix`` -> :class:`ShapModel` -> ``feature_map(shap_plot=True)`` chain.
"""
from typing import Optional, List, Type, Tuple, Union
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature, CPPPlot
from aaanalysis.explainable_ai_pro import ShapModel
from aaanalysis.data_handling import load_scales


# I Helper Functions
def _resolve_model_classes(list_model_classes):
    """Resolve the model classes to ShapModel's default ensemble, so the sample-selection probe
    fits the same models that produce the SHAP values."""
    if list_model_classes is None:
        return [RandomForestClassifier, ExtraTreesClassifier]
    if not isinstance(list_model_classes, list):
        return [list_model_classes]
    return list_model_classes


def _most_confident_target_sample(X, labels, list_model_classes, label_target_class, random_state):
    """Return the row position of the ``label_target_class`` sample the models predict most
    confidently.

    Each model in the ensemble is fit on the full matrix; their target-class probabilities are
    averaged and the highest-scoring sample *of the target class* is chosen — the most
    representative correct prediction to explain. Deterministic given ``random_state``.
    """
    labels = np.asarray(labels)
    target_rows = np.where(labels == label_target_class)[0]
    if len(target_rows) == 0:
        raise ValueError(f"'labels' contains no sample of 'label_target_class' ({label_target_class}); "
                         f"pass an explicit 'samples' or a present target class.")
    model_classes = _resolve_model_classes(list_model_classes)
    proba = np.zeros(X.shape[0], dtype=float)
    for model_class in model_classes:
        model = model_class()
        if "random_state" in model.get_params():
            model.set_params(random_state=random_state)
        model.fit(X, labels)
        classes = list(model.classes_)
        if label_target_class not in classes:
            raise ValueError(f"'label_target_class' ({label_target_class}) is not among the fitted "
                             f"class labels {classes}.")
        proba += model.predict_proba(X)[:, classes.index(label_target_class)]
    proba /= len(model_classes)
    return int(target_rows[int(np.argmax(proba[target_rows]))])


def _normalize_samples_names(samples, df_seq):
    """Normalize ``samples`` to parallel lists of (samples, column-name) for ``add_feat_impact``.

    A string is an ``entry`` name (used verbatim); an integer is a row position (named from the
    ``entry`` column when present, else ``sample<pos>``)."""
    samples_list = samples if isinstance(samples, list) else [samples]
    names = []
    for s in samples_list:
        if isinstance(s, str):
            names.append(s)
        elif ut.COL_ENTRY in df_seq.columns:
            names.append(str(df_seq[ut.COL_ENTRY].iloc[int(s)]))
        else:
            names.append(f"sample{int(s)}")
    return samples_list, names


# II Main Functions
def explain_features(df_feat: pd.DataFrame,
                     df_seq: pd.DataFrame,
                     labels: ut.ArrayLike1D,
                     list_model_classes: Optional[List[Type[BaseEstimator]]] = None,
                     label_target_class: int = 1,
                     samples: Union[int, str, List[int], List[str], None] = None,
                     add_sample_mean_dif: bool = False,
                     label_ref: int = 0,
                     name_test: str = "TEST",
                     name_ref: str = "REF",
                     plot: bool = True,
                     random_state: Optional[int] = None,
                     n_jobs: Optional[int] = None,
                     verbose: bool = False,
                     ) -> Tuple[pd.DataFrame, Optional[Axes], None]:
    """
    Explain a feature set in one call: compute per-sample SHAP impact and draw the SHAP feature map.

    A thin, stateless *pro* facade over the explicit primitive path. It rebuilds the feature matrix
    ``X`` from the feature identifiers in ``df_feat`` (via :meth:`SequenceFeature.get_df_parts` +
    :meth:`SequenceFeature.feature_matrix`), fits a :class:`ShapModel`, attaches the per-sample SHAP
    feature impact to ``df_feat`` (via :meth:`ShapModel.add_feat_impact`), and draws the
    SHAP-coloured feature map (:meth:`CPPPlot.feature_map` with ``shap_plot=True``). The defaults are
    byte-identical to writing those calls by hand.

    By default a single sample is explained: the ``label_target_class`` sample the models predict
    most confidently — the most representative correct prediction. Pass ``samples`` (an ``entry``
    name, a row position, or a list) to explain chosen sample(s) instead; the feature map then
    colours by the first requested sample's impact and the impacts of all of them are added to
    ``df_feat``.

    .. warning::

        **Experimental.** This ``aaanalysis.pipe`` (``ap``) golden pipeline is under active
        development; its API (signatures, defaults, return objects) may change between minor releases
        without the usual deprecation cycle. Pin a version if you depend on the current behaviour.

    Parameters
    ----------
    df_feat : pd.DataFrame, shape (n_features, n_feature_info)
        Feature DataFrame with a ``feature`` column of feature identifiers (e.g. from :meth:`CPP.run`,
        :func:`aaanalysis.pipe.find_features`, or :func:`load_features`).
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        Sequence DataFrame with the sequence/parts information, row-aligned to ``labels``. The feature
        matrix is rebuilt from it via :meth:`SequenceFeature.get_df_parts`.
    labels : array-like, shape (n_samples,)
        Class labels for the samples (typically, 1=positive, 0=negative).
    list_model_classes : list of Type[BaseEstimator], optional
        Prediction model classes passed to :class:`ShapModel`. If ``None``, the ``ShapModel`` default
        (``[RandomForestClassifier, ExtraTreesClassifier]``) is used.
    label_target_class : int, default=1
        The class label for which SHAP values are computed and the sample is auto-selected.
    samples : int, str, list of int, list of str, or None
        Sample(s) to explain, given as row position(s) in the feature matrix or ``entry`` name(s)
        from ``df_seq``. If ``None``, the most confidently predicted ``label_target_class`` sample is
        selected automatically.
    add_sample_mean_dif : bool, default=False
        If ``True``, also enrich the returned ``df_feat`` with per-sample **mean-difference** columns
        ``mean_dif_'name'`` (each explained sample's feature value minus the ``label_ref`` group
        average) alongside the SHAP ``feat_impact_'name'`` columns, for the same sample(s) and names.
        This is the per-sample contrast a sample-level CPP-SHAP map/ranking is coloured by (via
        :meth:`ShapModel.add_sample_mean_dif`); compute stays separate from plotting. Default
        ``False`` leaves the returned columns unchanged.
    label_ref : int, default=0
        Reference-group label whose per-feature average each sample is contrasted against for the
        ``mean_dif_'name'`` columns. Used only when ``add_sample_mean_dif=True``.
    name_test : str, default="TEST"
        Name of the test (positive) group, shown on the feature map.
    name_ref : str, default="REF"
        Name of the reference (negative) group, shown on the feature map.
    plot : bool, default=True
        If ``True``, draw the SHAP-coloured feature map and return its ``Axes``; if ``False``, skip
        the plot and return ``None`` in the figure slot.
    random_state : int, optional
        The seed used by the random number generator. If a positive integer, results of stochastic
        processes (the SHAP estimation and sample selection) are reproducible.
    n_jobs : int, optional
        Number of CPU cores (>=1) for building the feature matrix. If ``None``, the optimized number is used.
    verbose : bool, default=False
        If ``True``, verbose progress information is printed.

    Returns
    -------
    df_feat_shap : pd.DataFrame, shape (n_features, n_feature_info+n)
        ``df_feat`` with the per-sample SHAP feature impact added as ``feat_impact_'name'`` column(s),
        plus per-sample ``mean_dif_'name'`` column(s) when ``add_sample_mean_dif=True``.
    ax : matplotlib.axes.Axes or None
        The Axes of the SHAP-coloured feature map, or ``None`` if ``plot=False``.
    evals : None
        Always ``None`` — explanation does no evaluation (keeps the uniform ``(results, figs, evals)``
        pipeline return shape).

    See Also
    --------
    * :class:`ShapModel` for the underlying Monte Carlo SHAP estimation and feature impact.
    * :meth:`CPPPlot.feature_map` for the SHAP-coloured feature map (``shap_plot=True``).
    * :func:`aaanalysis.pipe.find_features` for obtaining ``df_feat`` (the feature discovery step).

    Warnings
    --------
    * This pipeline requires `SHAP`, which is automatically installed via ``pip install aaanalysis[pro]``.

    Examples
    --------
    .. include:: examples/ap_explain_features.rst
    """
    # Validate (thin facade: the wrapped primitives validate labels, samples, and the rest)
    df_feat = ut.check_df_feat(df_feat=df_feat)
    ut.check_df_seq(df_seq=df_seq)
    ut.check_number_range(name="label_target_class", val=label_target_class, min_val=0, just_int=True)
    ut.check_bool(name="add_sample_mean_dif", val=add_sample_mean_dif)
    ut.check_number_range(name="label_ref", val=label_ref, min_val=0, just_int=True)
    ut.check_str(name="name_test", val=name_test)
    ut.check_str(name="name_ref", val=name_ref)
    ut.check_bool(name="plot", val=plot)
    ut.check_number_range(name="random_state", val=random_state, min_val=0, accept_none=True, just_int=True)
    ut.check_bool(name="verbose", val=verbose)
    # Build the feature matrix from the feature identifiers in df_feat (byte-identical to the manual chain)
    sf = SequenceFeature(verbose=verbose)
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts, n_jobs=n_jobs)
    # Fit the SHAP model and obtain Monte Carlo SHAP-value estimates
    sm = ShapModel(list_model_classes=list_model_classes, random_state=random_state, verbose=verbose)
    sm.fit(X, labels=labels, label_target_class=label_target_class)
    # Resolve the sample(s) to explain (auto-select the most confident target-class sample by default)
    if samples is None:
        samples = _most_confident_target_sample(X, labels=labels, list_model_classes=list_model_classes,
                                                 label_target_class=label_target_class,
                                                 random_state=random_state)
    samples_list, names = _normalize_samples_names(samples, df_seq=df_seq)
    # Attach the per-sample SHAP feature impact (scalars for a single sample, lists for several)
    if len(samples_list) == 1:
        df_feat = sm.add_feat_impact(df_feat=df_feat, samples=samples_list[0], names=names[0], df_seq=df_seq)
    else:
        df_feat = sm.add_feat_impact(df_feat=df_feat, samples=samples_list, names=names, df_seq=df_seq)
    # Optionally enrich with the per-sample mean-difference columns (sample minus label_ref average),
    # matched to the same sample(s) / name(s) as the SHAP impact — compute only, no extra plot.
    if add_sample_mean_dif:
        if len(samples_list) == 1:
            df_feat = sm.add_sample_mean_dif(X, labels=labels, label_ref=label_ref, df_feat=df_feat,
                                             samples=samples_list[0], names=names[0], df_seq=df_seq)
        else:
            df_feat = sm.add_sample_mean_dif(X, labels=labels, label_ref=label_ref, df_feat=df_feat,
                                             samples=samples_list, names=names, df_seq=df_seq)
    # Draw the SHAP-coloured feature map for the first requested sample's impact
    ax = None
    if plot:
        col_imp = f"{ut.COL_FEAT_IMPACT}_{names[0]}"
        _, ax = CPPPlot(df_scales=load_scales(name="scales"), verbose=verbose).feature_map(
            df_feat=df_feat, shap_plot=True, col_imp=col_imp, col_val=col_imp,
            name_test=name_test, name_ref=name_ref)
    # Uniform (results, figs, evals) pipeline return triple; evals=None (explanation does no evaluation)
    return df_feat, ax, None
