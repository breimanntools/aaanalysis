"""
This is a script for the backend built-in per-site predictor of ``CPPStructurePlot.explore``.

It turns a fixed feature set (``df_feat``) plus a labeled training population (``df_seq`` +
``labels``) into a ``(sequence, p1) -> df_feat`` callable that, for one P1 anchor, computes the
query window's feature values for the **fixed** feature set (never ``CPP.run`` discovery), predicts
its probability with a fit-once estimator, and attaches the per-site SHAP impact by refitting a
default :class:`ShapModel` with the window soft-labeled at that probability (fuzzy interpolate,
``n_rounds=1`` — the two-fit estimate). The frontend ``explore`` validates and delegates here.
"""
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature
from aaanalysis.explainable_ai_pro import ShapModel

# Entry name for the query window appended to the training population for the per-site fuzzy refit.
# Underscored + unlikely to collide with a real accession; used as the fuzzy-label / sample key.
_QUERY_ENTRY = "__QUERY__"

# String shortcuts for the prediction-estimator vocabulary, mirroring the aap.find_features /
# predict_samples presets (the ut.MODEL_* constants) so the user can write model="rf" instead of
# importing the class. Each lambda takes the random_state. The ShapModel that computes the impact
# always uses its OWN defaults (TreeExplainer + RF/ExtraTrees), independent of this choice.
_STR_MODELS = {
    ut.MODEL_SVM: lambda rs: SVC(class_weight="balanced", probability=True, random_state=rs),
    ut.MODEL_RF: lambda rs: RandomForestClassifier(random_state=rs),
    ut.MODEL_LOG_REG: lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
    "extra_trees": lambda rs: ExtraTreesClassifier(random_state=rs),
}


# I Helper Functions
def resolve_estimators(model, random_state=None):
    """Resolve ``model`` (name, estimator, or list of either) to a list of unfitted estimators.

    Names resolve through ``_STR_MODELS`` (the ``ut.MODEL_*`` vocabulary); a scikit-learn estimator
    is cloned (with ``random_state`` injected where exposed and unset); a list mixes both and its
    per-estimator probabilities are averaged per site.
    """
    items = model if isinstance(model, list) else [model]
    if len(items) == 0:
        raise ValueError("'model' should not be an empty list")
    estimators = []
    for item in items:
        if isinstance(item, str):
            if item not in _STR_MODELS:
                raise ValueError(f"'model' name ({item}) should be one of {sorted(_STR_MODELS)} "
                                 f"or a scikit-learn estimator")
            estimators.append(_STR_MODELS[item](random_state))
        elif isinstance(item, BaseEstimator):
            est = clone(item)
            params = est.get_params()
            if random_state is not None and params.get("random_state", "missing") is None:
                est.set_params(random_state=random_state)
            estimators.append(est)
        else:
            raise ValueError(f"'model' ({item}) should be a name {sorted(_STR_MODELS)}, a "
                             f"scikit-learn estimator, or a list of those")
    return estimators


def _target_proba(estimators, x_query, label_target_class):
    """Average the ``label_target_class`` probability across the fit-once estimators for one row."""
    proba = 0.0
    for est in estimators:
        classes = list(est.classes_)
        if label_target_class not in classes:
            raise ValueError(f"'label_target_class' ({label_target_class}) is not among the fitted "
                             f"class labels {classes}.")
        proba += float(est.predict_proba(x_query)[0, classes.index(label_target_class)])
    return proba / len(estimators)


# II Main Functions
def build_builtin_predictor(df_feat: pd.DataFrame,
                            df_seq: pd.DataFrame,
                            labels: ut.ArrayLike1D,
                            tmd_len: int,
                            jmd_n_len: int,
                            jmd_c_len: int,
                            model: Union[str, BaseEstimator, List] = ut.MODEL_RF,
                            col_imp: str = ut.COL_FEAT_IMPACT,
                            df_scales: Optional[pd.DataFrame] = None,
                            label_target_class: int = 1,
                            random_state: Optional[int] = None,
                            n_jobs: Optional[int] = None,
                            verbose: bool = False):
    """Build a ``(sequence, p1) -> df_feat`` predictor wiring CPP features + a default ShapModel.

    The training feature matrix and the prediction estimator(s) are built **once**; per call only
    the query window's feature vector is computed (fixed feature set, anchor mode) and a default
    :class:`ShapModel` is refit (fuzzy interpolate, ``n_rounds=1``) with the window soft-labeled at
    its predicted probability. The returned ``df_feat`` carries the per-site impact in ``col_imp``
    and the predicted ``label_target_class`` probability in ``df.attrs['proba']``.
    """
    labels = np.asarray(labels)
    features = df_feat[ut.COL_FEATURE]
    # Parts actually referenced by the fixed feature set (the 'PART' of each PART-SPLIT-SCALE id),
    # so get_df_parts builds exactly the part columns feature_matrix needs for any df_feat.
    list_parts = sorted({str(f).split("-")[0].lower() for f in features})
    sf = SequenceFeature(verbose=False)
    # Training feature matrix from the fixed feature set (built once; byte-identical to CPP's kernel)
    df_parts_train = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts,
                                     jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    X_train = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_train,
                                           df_scales=df_scales, n_jobs=n_jobs))
    # Fit the prediction estimator(s) once for an instant per-site predict_proba
    estimators = resolve_estimators(model, random_state=random_state)
    for est in estimators:
        est.fit(X_train, labels)
    # The training entries reused for the fuzzy refit (the query is appended as one extra row)
    df_seq_train = df_seq.reset_index(drop=True).copy()

    def predictor(sequence, p1):
        # Query window with p1 as the first TMD residue, so the TMD spans [p1, p1+tmd_len-1].
        # This matches interactive()'s default site_to_start (start = p1 - jmd_n_len), keeping the
        # predicted feature geometry aligned with the residues the render paints. Position-based
        # df_seq (tmd_start/tmd_stop, 1-based inclusive) per SequenceFeature.get_df_parts.
        df_seq_q = pd.DataFrame({ut.COL_ENTRY: [_QUERY_ENTRY], ut.COL_SEQ: [sequence],
                                 ut.COL_TMD_START: [int(p1)],
                                 ut.COL_TMD_STOP: [int(p1) + tmd_len - 1]})
        df_parts_q = sf.get_df_parts(df_seq=df_seq_q, list_parts=list_parts,
                                     jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        x_q = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts_q,
                                           df_scales=df_scales, n_jobs=1))
        proba = _target_proba(estimators, x_q, label_target_class)
        # Per-site SHAP impact: append the query soft-labeled at its probability and refit (2 fits)
        X_ext = np.vstack([X_train, x_q])
        labels_ext = np.append(labels, label_target_class)  # placeholder; overridden by fuzzy_labels
        df_seq_ext = pd.concat([df_seq_train[[ut.COL_ENTRY]], pd.DataFrame({ut.COL_ENTRY: [_QUERY_ENTRY]})],
                               ignore_index=True)
        sm = ShapModel(random_state=random_state, verbose=False)  # default explainer + models
        sm.fit(X_ext, labels=labels_ext, label_target_class=label_target_class,
               fuzzy_labeling=True, fuzzy_aggregation="interpolate", n_rounds=1,
               df_seq=df_seq_ext, fuzzy_labels={_QUERY_ENTRY: float(proba)})
        out = sm.add_feat_impact(df_feat=df_feat.copy(), samples=_QUERY_ENTRY,
                                 names=_QUERY_ENTRY, df_seq=df_seq_ext, drop=True)
        # add_feat_impact writes 'feat_impact_<name>'; expose it under the requested col_imp
        created = f"{ut.COL_FEAT_IMPACT}_{_QUERY_ENTRY}"
        out = out.rename(columns={created: col_imp})
        out.attrs["proba"] = float(proba)
        if verbose:
            ut.print_out(f"CPPStructurePlot.explore: site p1={int(p1)} -> "
                         f"P(class {label_target_class})={proba:.3f}")
        return out

    return predictor
