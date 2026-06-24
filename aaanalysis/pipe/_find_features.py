"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``find_features`` golden
pipeline: a staged, interpretable CPP AutoML search. It sweeps the CPP feature space
(Split x Part x Scale -> Filter), selects the best configuration by cross-validated model
performance, ranks the winning features by tree-based importance, and draws the feature map.
"""
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature, CPP, CPPGrid, CPPPlot
from aaanalysis.explainable_ai import TreeModel
from aaanalysis.data_handling import load_scales


# I Helper Functions
# Optimization-mode sweep grids. Each mode names the CPP levers ``find_features`` sweeps:
# "fast" runs a single default configuration (no sweep, no selection); "balanced" sweeps the
# Split levers (which split types, granularity) and ``n_filter`` over default parts/scales;
# "exhaustive" additionally sweeps the Part region set and the Scale breadth. Held as a
# structured registry (not a flat constant bundle) so the whole grid stays in one place.
_MODES = {
    "fast":       {"sweep_parts": False, "sweep_scales": False, "sweep_split": False,
                   "scale_breadths": [30], "n_split_max_vals": [15], "n_filter_vals": [100],
                   "simplify_strategy": "greedy"},
    "balanced":   {"sweep_parts": False, "sweep_scales": False, "sweep_split": True,
                   "scale_breadths": [50], "n_split_max_vals": [10, 15],
                   "n_filter_vals": [25, 50, 75, 100, 125, 150],
                   "simplify_strategy": "greedy"},
    "exhaustive": {"sweep_parts": True, "sweep_scales": True, "sweep_split": True,
                   "scale_breadths": [50, None], "n_split_max_vals": [15],
                   "n_filter_vals": [25, 50, 75, 100, 125, 150],
                   "simplify_strategy": "consolidate"},
}
# Named-region part sets the exhaustive sweep varies over. Only composite regions (each >= ~20
# residues: tmd, the half-overlapping flanks, the whole jmd-tmd-jmd span) are used, never a bare
# 10-residue jmd_n / jmd_c: a short part empty-buckets under fine Segment/Pattern splits, which
# would only add configurations that error and get dropped.
_PART_SETS = [
    ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"],
    ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c", "tmd_jmd"],
    ["tmd_jmd", "jmd_n_tmd_n", "tmd_c_jmd_c"],
    ["tmd", "tmd_jmd"],
]
# Split-type sets (the "pattern_mode" facade): which of Pattern / PeriodicPattern join the
# always-present Segment splits.
_SPLIT_TYPE_SETS = [
    [ut.STR_SEGMENT],
    [ut.STR_SEGMENT, ut.STR_PATTERN],
    [ut.STR_SEGMENT, ut.STR_PERIODIC_PATTERN],
    [ut.STR_SEGMENT, ut.STR_PATTERN, ut.STR_PERIODIC_PATTERN],
]
_PATTERN_MODE = {(ut.STR_SEGMENT,): "none",
                 (ut.STR_SEGMENT, ut.STR_PATTERN): "p1",
                 (ut.STR_SEGMENT, ut.STR_PERIODIC_PATTERN): "p2",
                 (ut.STR_SEGMENT, ut.STR_PATTERN, ut.STR_PERIODIC_PATTERN): "p1+p2"}
# Levers a power user may pin via the bounded ``kws`` dict (unknown keys raise). Each override
# fixes that lever to a single value, collapsing its sweep.
_KWS_KEYS = {"n_explain", "n_split_max", "n_filter", "simplify_strategy", "max_cor", "max_overlap"}
# Robustness tolerance: among configurations within this margin of the best CV score, the
# simplest one (fewest features, then smallest n_filter) is selected.
_ROBUST_TOL = 0.01


def _resolve_model(model, random_state=None):
    """Construct the cross-validation estimator (mirrors the CPP.simplify model presets)."""
    if model == ut.MODEL_SVM:
        return SVC(class_weight="balanced", random_state=random_state)
    if model == ut.MODEL_RF:
        return RandomForestClassifier(random_state=random_state)
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _cv_score(X, labels, model="svm", cv=5, metric="balanced_accuracy", random_state=None):
    """Cross-validated (mean, std) model score for a feature matrix (consistent with CPP.simplify)."""
    estimator = _resolve_model(model=model, random_state=random_state)
    scores = cross_val_score(estimator, X, y=labels, cv=cv, scoring=metric)
    return float(scores.mean()), float(scores.std())


def _load_scales_breadth(top_explain_n=None, subcategories=None):
    """Load a scale set (optionally restricted to AAontology subcategories / interpretable breadth)."""
    if top_explain_n is None:
        df_scales = load_scales(name="scales")
    else:
        df_scales = load_scales(name="scales", top_explain_n=top_explain_n)
    if subcategories is not None:
        df_cat = load_scales(name="scales_cat")
        keep = set(df_cat[df_cat[ut.COL_SUBCAT].isin(subcategories)][ut.COL_SCALE_ID])
        cols = [c for c in df_scales.columns if c in keep]
        if len(cols) == 0:
            raise ValueError(f"'subcategories' ({subcategories}) should be names that match "
                             f"at least one scale.")
        df_scales = df_scales[cols]
    return df_scales


def _resolve_config(optimization="balanced", kws=None):
    """Merge the optimization-mode grid with the bounded ``kws`` overrides (unknown keys raise)."""
    mode = _MODES[optimization]
    cfg = {
        "sweep_parts": mode["sweep_parts"], "sweep_split": mode["sweep_split"],
        "sweep_scales": mode["sweep_scales"],
        "part_sets": list(_PART_SETS) if mode["sweep_parts"] else [_PART_SETS[0]],
        "split_type_sets": list(_SPLIT_TYPE_SETS) if mode["sweep_split"] else [_SPLIT_TYPE_SETS[-1]],
        "n_split_max_vals": list(mode["n_split_max_vals"]),
        "scale_breadths": list(mode["scale_breadths"]),
        "n_filter_vals": list(mode["n_filter_vals"]),
        "simplify_strategy": mode["simplify_strategy"],
        "max_cor": 0.5, "max_overlap": 0.5,
    }
    if kws is not None:
        ut.check_dict(name="kws", val=kws)
        unknown = set(kws) - _KWS_KEYS
        if unknown:
            raise ValueError(f"'kws' keys ({sorted(unknown)}) should be among {sorted(_KWS_KEYS)}.")
        if "n_explain" in kws:
            cfg["scale_breadths"] = [kws["n_explain"]]
            cfg["sweep_scales"] = False
        if "n_split_max" in kws:
            cfg["n_split_max_vals"] = [kws["n_split_max"]]
        if "n_filter" in kws:
            cfg["n_filter_vals"] = [kws["n_filter"]]
        if "simplify_strategy" in kws:
            cfg["simplify_strategy"] = kws["simplify_strategy"]
        if "max_cor" in kws:
            cfg["max_cor"] = kws["max_cor"]
        if "max_overlap" in kws:
            cfg["max_overlap"] = kws["max_overlap"]
    return cfg


def _refine_rfe(df_feat=None, X=None, labels=None, base_mean=0.0, model="svm", cv=5,
                metric="balanced_accuracy", random_state=None):
    """Recursive-feature-elimination post-step: keep the reduced set only if CV does not drop.

    Bounded and deterministic; a no-op when there are too few features to recurse. Worst case
    leaves ``df_feat`` unchanged, so it never degrades the selected feature set.
    """
    n_feat = len(df_feat)
    # Skip the (expensive) RFE fit when recursion is impossible (too few features) or pointless
    # (the score is already at the metric ceiling, so a smaller set can at best tie).
    if n_feat < 8 or base_mean >= 1.0 - 1e-9:
        return df_feat, base_mean, False
    n_feat_min = max(4, n_feat // 3)
    tm = TreeModel(random_state=random_state, verbose=False)
    tm.fit(X, labels=labels, use_rfe=True, n_cv=cv, n_feat_min=n_feat_min,
           n_feat_max=n_feat, metric=metric)
    # is_selected_ is (n_rounds, n_features); keep features chosen in the majority of rounds.
    mask = np.asarray(tm.is_selected_).mean(axis=0) >= 0.5
    if mask.sum() < n_feat_min or mask.all():
        return df_feat, base_mean, False
    df_feat_rfe = df_feat[mask].reset_index(drop=True)
    mean_rfe, _ = _cv_score(X[:, mask], labels, model=model, cv=cv, metric=metric,
                            random_state=random_state)
    if mean_rfe >= base_mean:
        return df_feat_rfe, mean_rfe, True
    return df_feat, base_mean, False


# II Main Functions
def find_features(labels: ut.ArrayLike1D,
                  df_seq: pd.DataFrame,
                  optimization: str = "balanced",
                  simplify: bool = True,
                  model: str = "svm",
                  cv: int = 5,
                  metric: str = "balanced_accuracy",
                  kws: Optional[dict] = None,
                  subcategories: Optional[List[str]] = None,
                  top_n: Optional[int] = None,
                  label_test: int = 1,
                  label_ref: int = 0,
                  name_test: str = "TEST",
                  name_ref: str = "REF",
                  plot: bool = True,
                  random_state: Optional[int] = None,
                  n_jobs: Optional[int] = None,
                  verbose: bool = False,
                  ) -> Tuple[pd.DataFrame, Optional[Axes], pd.DataFrame]:
    """
    Identify discriminating features in one call via a staged, interpretable CPP AutoML search.

    Sweeps the CPP feature space (which sequence Splits, which Part regions, which Scale breadth)
    together with the ``n_filter`` selection, scores every configuration by cross-validated model
    performance, and selects a **robust** winner (the simplest configuration within 1% of the best
    score). The winning feature set is then refined (:meth:`CPP.simplify` and recursive feature
    elimination, each kept only if it does not lower the cross-validated score), ranked by
    tree-based importance, and visualized as the CPP feature map. At ``optimization="fast"`` no
    search is run: the result is byte-identical to the explicit single-CPP path.

    The search is staged so its cost stays interpretable: the ``optimization`` grade scopes which
    levers vary, the bounded ``kws`` dict pins any single lever, and ``model`` / ``cv`` / ``metric``
    define the selection criterion (CPP's own feature-quality ranking is monotone in ``n_filter`` and
    cannot pick it, so a model-based cross-validation is used instead).

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Class labels for the samples (typically, 1=positive/test, 0=negative/reference).
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        Sequence DataFrame, row-aligned to ``labels``. Required, because the Part regions are a
        swept lever and must be (re)built from the sequences via :meth:`SequenceFeature.get_df_parts`.
    optimization : str, default="balanced"
        Search breadth: ``"fast"`` (single default configuration, no search), ``"balanced"`` (sweep
        the Split levers and ``n_filter`` over the default parts/scales), or ``"exhaustive"`` (also
        sweep the Part region set and the Scale breadth).
    simplify : bool, default=True
        If ``True``, refine the selected feature set with :meth:`CPP.simplify` (kept only if the
        cross-validated score does not drop).
    model : str, default="svm"
        Model used to score configurations during selection: ``"svm"``, ``"rf"``, or ``"log_reg"``.
    cv : int, default=5
        Number of cross-validation folds for the selection score, must be > 1.
    metric : str, default="balanced_accuracy"
        Cross-validation scoring metric (any scikit-learn classification scorer).
    kws : dict, optional
        Bounded power-user overrides; each pins a swept lever to a single value (unknown keys raise).
        Recognized keys: ``n_explain``, ``n_split_max``, ``n_filter``, ``simplify_strategy``,
        ``max_cor``, ``max_overlap``.
    subcategories : list of str, optional
        AAontology subcategories to restrict the scale set to. If ``None``, all scales of the grade.
    top_n : int, optional
        If given, keep only the top ``top_n`` features (after importance ranking).
    label_test : int, default=1
        Class label of the test/positive group passed to :meth:`CPP.run`.
    label_ref : int, default=0
        Class label of the reference/negative group passed to :meth:`CPP.run`.
    name_test : str, default="TEST"
        Display name of the test/positive group in the feature map.
    name_ref : str, default="REF"
        Display name of the reference/negative group in the feature map.
    plot : bool, default=True
        If ``True``, draw the CPP feature map and return its ``Axes``; if ``False``, return ``None``.
    random_state : int, optional
        The seed used by the random number generator. If a positive integer, results of stochastic
        processes are reproducible.
    n_jobs : int, optional
        Number of CPU cores (>=1) for the sweep and feature-matrix builds. If ``None``, the
        optimized number is used.
    verbose : bool, default=False
        If ``True``, verbose progress information is printed.

    Returns
    -------
    df_feat : pd.DataFrame
        Feature DataFrame of the selected configuration in the canonical CPP schema, ranked by
        tree-based importance.
    ax : matplotlib.axes.Axes or None
        The feature-map ``Axes`` if ``plot=True``, else ``None`` (Figure via ``ax.figure``).
    df_eval : pd.DataFrame
        Per-configuration sweep table with the configuration descriptors and the cross-validated
        ``cv_bacc_mean`` / ``cv_bacc_std``, ``rank``, and ``is_selected`` columns.

    See Also
    --------
    * :class:`CPPGrid` for the configuration sweep this pipeline drives.
    * :meth:`CPP.run` and :meth:`CPP.simplify` for the underlying feature engineering.
    * :meth:`CPPPlot.feature_map` for the visualization.

    Examples
    --------
    .. include:: examples/aap_find_features.rst
    """
    # Validate (thin facade: the wrapped primitives validate the rest)
    ut.check_str_options(name="optimization", val=optimization, list_str_options=list(_MODES))
    ut.check_df_seq(df_seq=df_seq)
    ut.check_bool(name="simplify", val=simplify)
    ut.check_str_options(name="model", val=model,
                         list_str_options=[ut.MODEL_SVM, ut.MODEL_RF, ut.MODEL_LOG_REG])
    ut.check_number_range(name="cv", val=cv, min_val=2, just_int=True)
    ut.check_str(name="metric", val=metric)
    ut.check_number_range(name="top_n", val=top_n, min_val=1, accept_none=True, just_int=True)
    ut.check_bool(name="plot", val=plot)
    ut.check_number_range(name="random_state", val=random_state, min_val=0,
                          accept_none=True, just_int=True)
    ut.check_bool(name="verbose", val=verbose)
    cfg = _resolve_config(optimization=optimization, kws=kws)

    sf = SequenceFeature(verbose=verbose)
    df_scales_all = load_scales(name="scales")
    if optimization == "fast":
        df_feat, df_parts_win, df_eval = _run_fast(
            sf=sf, labels=labels, df_seq=df_seq, cfg=cfg, simplify=simplify, model=model,
            cv=cv, metric=metric, subcategories=subcategories, label_test=label_test,
            label_ref=label_ref, df_scales_all=df_scales_all, random_state=random_state,
            n_jobs=n_jobs, verbose=verbose)
    else:
        df_feat, df_parts_win, df_eval = _run_search(
            sf=sf, labels=labels, df_seq=df_seq, cfg=cfg, simplify=simplify, model=model,
            cv=cv, metric=metric, subcategories=subcategories, label_test=label_test,
            label_ref=label_ref, df_scales_all=df_scales_all, random_state=random_state,
            n_jobs=n_jobs, verbose=verbose)

    # Rank the winning features by tree-based importance (also required by the feature map). The
    # full scale set is used here: simplify can swap to correlated scales outside a subcategory
    # filter, and feature_matrix/feature_map only read the scales the resulting features reference.
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts_win,
                          df_scales=df_scales_all, n_jobs=n_jobs)
    tm = TreeModel(random_state=random_state, verbose=verbose)
    tm.fit(X, labels=labels)
    df_feat = tm.add_feat_importance(df_feat=df_feat, sort=True)
    if top_n is not None:
        df_feat = df_feat.head(top_n)
    ax = None
    if plot:
        _, ax = CPPPlot(df_scales=df_scales_all, verbose=verbose).feature_map(
            df_feat=df_feat, name_test=name_test, name_ref=name_ref)
    return df_feat, ax, df_eval


def _run_fast(sf=None, labels=None, df_seq=None, cfg=None, simplify=True, model="svm", cv=5,
              metric="balanced_accuracy", subcategories=None, label_test=1, label_ref=0,
              df_scales_all=None, random_state=None, n_jobs=None, verbose=False):
    """Single default configuration: the explicit CPP path, byte-identical to writing it by hand."""
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=cfg["part_sets"][0])
    df_scales = _load_scales_breadth(top_explain_n=cfg["scale_breadths"][0], subcategories=subcategories)
    cpp = CPP(df_parts=df_parts, df_scales=df_scales, random_state=random_state, verbose=verbose)
    df_feat = cpp.run(labels=labels, label_test=label_test, label_ref=label_ref,
                      n_filter=cfg["n_filter_vals"][0], max_cor=cfg["max_cor"],
                      max_overlap=cfg["max_overlap"], n_jobs=n_jobs)
    if simplify:
        # Unconditional simplify (no cross-validated keep-guard) keeps fast byte-identical to the
        # explicit single-CPP chain; the search modes instead keep the refine only if CV improves.
        df_feat = cpp.simplify(df_feat=df_feat, labels=labels, strategy=cfg["simplify_strategy"],
                               label_test=label_test, label_ref=label_ref)
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts,
                          df_scales=df_scales_all, n_jobs=n_jobs)
    cv_mean, cv_std = _cv_score(X, labels, model=model, cv=cv, metric=metric, random_state=random_state)
    df_eval = pd.DataFrame([{
        "list_parts": ",".join(cfg["part_sets"][0]),
        "split_types": ",".join(_SPLIT_TYPE_SETS[-1]),
        "pattern_mode": _PATTERN_MODE[tuple(_SPLIT_TYPE_SETS[-1])],
        "n_split_max": cfg["n_split_max_vals"][0], "n_explain": cfg["scale_breadths"][0],
        "n_filter": cfg["n_filter_vals"][0], "n_features": len(df_feat),
        "cv_bacc_mean": cv_mean, "cv_bacc_std": cv_std, "rank": 1, "is_selected": True,
    }])
    return df_feat, df_parts, df_eval


def _run_search(sf=None, labels=None, df_seq=None, cfg=None, simplify=True, model="svm", cv=5,
                metric="balanced_accuracy", subcategories=None, label_test=1, label_ref=0,
                df_scales_all=None, random_state=None, n_jobs=None, verbose=False):
    """Sweep the configuration grid (CPPGrid), score each by model CV, and refine the robust winner."""
    # Build the scale candidates (one df_scales per swept breadth) and the CPPGrid parameter dicts.
    scale_sets = [_load_scales_breadth(top_explain_n=n, subcategories=subcategories)
                  for n in cfg["scale_breadths"]]
    params_parts = {"list_parts": cfg["part_sets"]} if cfg["sweep_parts"] else None
    params_split = {"split_types": cfg["split_type_sets"], "n_split_max": cfg["n_split_max_vals"]}
    params_cpp = {"n_filter": cfg["n_filter_vals"], "label_test": label_test, "label_ref": label_ref,
                  "max_cor": cfg["max_cor"], "max_overlap": cfg["max_overlap"]}
    cppg = CPPGrid(df_seq=df_seq, labels=labels, random_state=random_state, verbose=verbose,
                   n_jobs=n_jobs)
    list_df_feat, df_params = cppg.run(params_parts=params_parts, params_split=params_split,
                                       params_scales=scale_sets, params_cpp=params_cpp)
    # Cache df_parts per Part set (reused for the per-configuration feature matrices).
    parts_cache = {}

    def _parts_for(list_parts):
        key = tuple(list_parts)
        if key not in parts_cache:
            parts_cache[key] = sf.get_df_parts(df_seq=df_seq, list_parts=list(list_parts))
        return parts_cache[key]

    # Score every (non-errored) configuration by cross-validated model performance.
    rows, idx_keep = [], []
    for i, df_feat_i in enumerate(list_df_feat):
        if df_feat_i is None or len(df_feat_i) == 0:
            continue
        rec = df_params.iloc[i]
        list_parts = cfg["part_sets"][int(rec["list_parts"])] if cfg["sweep_parts"] else cfg["part_sets"][0]
        split_types = cfg["split_type_sets"][int(rec["split_types"])]
        breadth = cfg["scale_breadths"][int(rec["df_scales"])]
        df_parts_i = _parts_for(list_parts)
        X_i = sf.feature_matrix(features=df_feat_i[ut.COL_FEATURE], df_parts=df_parts_i,
                                df_scales=df_scales_all, n_jobs=n_jobs)
        cv_mean, cv_std = _cv_score(X_i, labels, model=model, cv=cv, metric=metric,
                                    random_state=random_state)
        rows.append({
            "list_parts": ",".join(list_parts), "split_types": ",".join(split_types),
            "pattern_mode": _PATTERN_MODE[tuple(split_types)], "n_split_max": int(rec["n_split_max"]),
            "n_explain": breadth, "n_filter": int(rec["n_filter"]), "n_features": len(df_feat_i),
            "cv_bacc_mean": cv_mean, "cv_bacc_std": cv_std,
        })
        idx_keep.append(i)
    if not rows:
        raise RuntimeError("'find_features' produced no valid configurations; relax 'kws' / "
                           "'subcategories' or use a less restrictive 'optimization'.")
    df_eval = pd.DataFrame(rows)
    # Robust selection: among configurations within the tolerance of the best score, the simplest
    # (fewest features, then smallest n_filter) wins. Ranking is best-score-first.
    order = df_eval.sort_values("cv_bacc_mean", ascending=False).index
    df_eval = df_eval.loc[order].reset_index(drop=True)
    idx_keep = [idx_keep[i] for i in order]
    best = df_eval["cv_bacc_mean"].max()
    cand = df_eval[df_eval["cv_bacc_mean"] >= best - _ROBUST_TOL]
    sel_pos = cand.sort_values(["n_features", "n_filter"], ascending=True).index[0]
    df_eval["rank"] = np.arange(1, len(df_eval) + 1)
    df_eval["is_selected"] = False
    df_eval.loc[sel_pos, "is_selected"] = True

    # Refine the winner (simplify + RFE), each kept only if the CV score does not drop.
    winner_i = idx_keep[sel_pos]
    df_feat_win = list_df_feat[winner_i].reset_index(drop=True)
    rec = df_params.iloc[winner_i]
    list_parts_win = cfg["part_sets"][int(rec["list_parts"])] if cfg["sweep_parts"] else cfg["part_sets"][0]
    df_parts_win = _parts_for(list_parts_win)
    df_scales_win = scale_sets[int(rec["df_scales"])]
    base_mean = float(df_eval.loc[sel_pos, "cv_bacc_mean"])
    if simplify:
        cpp_win = CPP(df_parts=df_parts_win, df_scales=df_scales_win,
                      random_state=random_state, verbose=verbose)
        df_feat_simpl = cpp_win.simplify(df_feat=df_feat_win, labels=labels,
                                         strategy=cfg["simplify_strategy"], ml_model=model,
                                         ml_metric=metric, ml_cv=cv, label_test=label_test,
                                         label_ref=label_ref)
        X_simpl = sf.feature_matrix(features=df_feat_simpl[ut.COL_FEATURE], df_parts=df_parts_win,
                                    df_scales=df_scales_all, n_jobs=n_jobs)
        mean_simpl, _ = _cv_score(X_simpl, labels, model=model, cv=cv, metric=metric,
                                  random_state=random_state)
        if mean_simpl >= base_mean:
            df_feat_win, base_mean = df_feat_simpl.reset_index(drop=True), mean_simpl
    X_win = sf.feature_matrix(features=df_feat_win[ut.COL_FEATURE], df_parts=df_parts_win,
                              df_scales=df_scales_all, n_jobs=n_jobs)
    df_feat_win, _, _ = _refine_rfe(df_feat=df_feat_win, X=X_win, labels=labels, base_mean=base_mean,
                                    model=model, cv=cv, metric=metric, random_state=random_state)
    return df_feat_win, df_parts_win, df_eval
