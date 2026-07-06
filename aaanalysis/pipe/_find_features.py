"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``find_features`` golden
pipeline: a staged, interpretable CPP AutoML search. Stage 1 cross-validates the full Cartesian
Part x Split x Scale grid and ranks each axis by its marginal-mean impact; Stage 2 refines the
single highest-impact axis against ``n_filter``; Stage 3 refines the winning feature set
(``CPP.simplify`` + RFE). Selection is multi-objective: per stage, the Pareto-optimal-then-simplest
configuration across all metrics wins, scored by the average cross-validated performance of one or
more models.
"""
from typing import Optional, List, Tuple, Union, Dict, Sequence
import warnings
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import aaanalysis.utils as ut
from aaanalysis.feature_engineering import SequenceFeature, CPP, CPPGrid, CPPPlot
from aaanalysis.explainable_ai import TreeModel
from aaanalysis.data_handling import load_scales
from ._eval_plot import plot_eval
from ._composition_baseline import (build_aac_df_feat, comp_kmer_signal, comp_kmer_df_feat,
                                    plot_composition_map, AAC_CAT_COLORS)


# I Helper Functions
# Optimization-mode sweep grids. The ``search`` grade names which CPP levers vary: "fast" runs a
# single default configuration (no search, byte-identical to the explicit CPP chain); "balanced"
# sweeps the Split levers + Scale + JMD length (n_jmd) + n_filter over default parts (~10 min);
# "exhaustive" also sweeps the Part region set and adds the orthogonal performance-ranked top60
# scale sets, a finer n_split_max step, and a wider n_jmd range. Held as a structured registry so the
# whole grid stays in one place and tunable. Scale candidates are (kind, value): ("explain", n) =
# top_explain_n interpretability tier (None = all); ("top60", k) = the k-th performance-ranked top60
# set. The ``n_jmd`` axis sweeps the JMD length symmetrically (jmd_n_len = jmd_c_len = n_jmd).
_MODES = {
    "fast": {
        "sweep_parts": False, "sweep_scales": False, "sweep_split": False, "sweep_jmd": False,
        "scale_specs": [("explain", 30)], "n_split_max_vals": [15], "n_filter_vals": [100],
        "n_jmd_vals": [10],
        "simplify_strategy": "greedy",
    },
    "balanced": {
        "sweep_parts": False, "sweep_scales": True, "sweep_split": True, "sweep_jmd": True,
        "scale_specs": [("explain", n) for n in (10, 20, 30, 40, 50, None)],
        "n_split_max_vals": [1, 5, 10, 15],
        "n_filter_vals": [25, 50, 75, 100, 125, 150],
        "n_jmd_vals": [6, 10, 14],
        "simplify_strategy": "greedy",
    },
    "exhaustive": {
        "sweep_parts": True, "sweep_scales": True, "sweep_split": True, "sweep_jmd": True,
        "scale_specs": [("explain", n) for n in (5, 10, 20, 30, 40, 50, 60)]
                       + [("top60", k) for k in range(1, 11)],
        "n_split_max_vals": [1, 3, 5, 7, 9, 11, 13, 15],
        "n_filter_vals": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        "n_jmd_vals": [2, 4, 6, 8, 10, 12, 14],
        "simplify_strategy": "consolidate",
    },
}
# Named-region part sets the exhaustive Part axis varies over. Composite regions only (each >= ~20
# residues): a bare 10-residue jmd_n / jmd_c empty-buckets under fine Segment/Pattern splits and
# would only add configurations that error and get dropped.
_PART_SETS = [
    ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"],
    ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c", "tmd_jmd"],
    ["tmd_jmd", "jmd_n_tmd_n", "tmd_c_jmd_c"],
    ["tmd", "tmd_jmd"],
]
# Split-type sets (the "pattern_mode" facade): which of Pattern / PeriodicPattern join Segment.
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
# Levers a power user may pin via the bounded ``kws`` dict (unknown keys raise).
_KWS_KEYS = {"n_explain", "n_split_max", "n_filter", "n_jmd", "len_max", "simplify_strategy",
             "max_cor", "max_overlap"}
_LIST_MODELS = [ut.MODEL_SVM, ut.MODEL_RF, ut.MODEL_LOG_REG]
# The first (default) PeriodicPattern step; a part must be at least this long to carry one.
_PERIODIC_STEP0 = 3


def _split_kws_for(split_types=None, n_split_max=15, len_max=15):
    """Build ``split_kws`` from the frontend ``SequenceFeature.get_split_kws``.

    Threads the Segment ``n_split_max`` and Pattern ``len_max`` levers through the public
    front door so ``find_features`` can request shorter splits (needed for free peptides).
    """
    return SequenceFeature.get_split_kws(split_types=list(split_types),
                                         n_split_max=n_split_max, len_max=len_max)


def _fit_split_kws_to_parts(split_types=None, n_split_max=15, len_max=15, df_parts=None):
    """Adapt the requested split config to the shortest sequence part (free-peptide safety net).

    A part of length ``L`` can only carry a ``Segment`` with ``n_split_max <= L``, a ``Pattern``
    with ``len_max <= L``, and a ``PeriodicPattern`` with ``steps[0] (=3) <= L``. When the shortest
    part is too short for the requested config, ``Pattern`` / ``PeriodicPattern`` are dropped and the
    ``Segment`` ``n_split_max`` is clamped so the run still works (``Segment``-only at minimum). A
    single ``UserWarning`` names what changed. For parts long enough for the requested config nothing
    is dropped or clamped, so the built ``split_kws`` is byte-identical to the requested one.
    """
    min_len = int(min(df_parts[c].map(len).min() for c in df_parts.columns))
    kept, changes = list(split_types), []
    if ut.STR_PERIODIC_PATTERN in kept and min_len < _PERIODIC_STEP0:
        kept.remove(ut.STR_PERIODIC_PATTERN)
        changes.append(f"dropped 'PeriodicPattern' (needs >= {_PERIODIC_STEP0} residues)")
    if ut.STR_PATTERN in kept and min_len < len_max:
        kept.remove(ut.STR_PATTERN)
        changes.append(f"dropped 'Pattern' (len_max={len_max} > shortest part n={min_len})")
    fitted_n_split_max = n_split_max
    if ut.STR_SEGMENT not in kept:
        # Never leave zero split types; Segment is the universal fallback (works down to n=1).
        kept.insert(0, ut.STR_SEGMENT)
    if n_split_max > min_len:
        fitted_n_split_max = min_len
        changes.append(f"clamped Segment 'n_split_max' {n_split_max} -> {min_len}")
    split_kws = _split_kws_for(split_types=kept, n_split_max=fitted_n_split_max, len_max=len_max)
    if changes:
        warnings.warn(
            f"'find_features': the shortest sequence part (n={min_len}) is too short for the "
            f"requested splits; {'; '.join(changes)}. This keeps the run working on free peptides / "
            f"short parts. Set 'kws' (n_split_max / len_max / n_jmd) or add flanking context to "
            f"control this.", UserWarning)
    return split_kws


def _cap_n_split_range(sf=None, df_seq=None, cfg=None):
    """Clamp the swept ``n_split_max`` values to the longest achievable shortest part, dedupe, warn.

    A ``Segment`` can be split into at most ``L`` pieces on a length-``L`` part, so CPP auto-caps any
    ``n_split_max`` above the part length. Across the swept parts / JMD lengths the least-constraining
    configuration has the longest shortest-part ``cap_len``; every swept value above ``cap_len`` would
    collapse to the same clamped run for **every** configuration, so it is clamped to ``cap_len`` and
    the list is deduped. Values ``<= cap_len`` are kept (some configuration runs them un-clamped), so
    on normal (long-part) inputs the sweep is unchanged and no warning fires — the cap only bites on
    genuinely short parts (free peptides). Mutates ``cfg['n_split_max_vals']`` and returns ``cap_len``.
    """
    cap_len = 0
    for lp in cfg["part_sets"]:
        for n_jmd in cfg["n_jmd_vals"]:
            df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list(lp),
                                       jmd_n_len=n_jmd, jmd_c_len=n_jmd)
            m = int(min(df_parts[c].map(len).min() for c in df_parts.columns))
            cap_len = max(cap_len, m)
    orig = list(cfg["n_split_max_vals"])
    capped = sorted({min(v, cap_len) for v in orig})
    if capped != orig:
        warnings.warn(
            f"'find_features': the shortest sequence part (n={cap_len}) caps 'n_split_max'; the "
            f"swept range {orig} was clamped and deduped to {capped} (values above it collapse to "
            f"the same clamped run). Add flanking context (n_jmd) to lift the cap.", UserWarning)
    cfg["n_split_max_vals"] = capped
    return cap_len


def _resolve_model(model, random_state=None):
    """Construct one cross-validation estimator (mirrors the CPP.simplify model presets)."""
    if model == ut.MODEL_SVM:
        return SVC(class_weight="balanced", random_state=random_state)
    if model == ut.MODEL_RF:
        return RandomForestClassifier(random_state=random_state)
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _resolve_models(model, random_state=None):
    """Resolve ``model`` (a name, an estimator, or a list of either) into a list of estimators."""
    items = model if isinstance(model, (list, tuple)) else [model]
    out = []
    for m in items:
        out.append(_resolve_model(m, random_state=random_state) if isinstance(m, str) else m)
    return out


def _cv_scores(X, labels, models=None, cv=5, metrics=None, random_state=None):
    """Cross-validate ``X`` and return ``{metric: (mean, std)}`` averaged over the models.

    All metrics are scored in one ``cross_validate`` pass per model (multi-scorer), so adding
    metrics is nearly free. The mean/std are averaged across the provided models.
    """
    means = {m: [] for m in metrics}
    stds = {m: [] for m in metrics}
    for est in models:
        res = cross_validate(est, X, y=labels, cv=cv, scoring=list(metrics))
        for m in metrics:
            arr = res["test_" + m]
            means[m].append(float(np.mean(arr)))
            stds[m].append(float(np.std(arr)))
    return {m: (float(np.mean(means[m])), float(np.mean(stds[m]))) for m in metrics}


def _load_scale_spec(spec, subcategories=None):
    """Build the ``df_scales`` for one scale spec ``("explain", n)`` / ``("top60", k)``."""
    kind, val = spec
    if kind == "top60":
        df_scales = load_scales(name="scales", top60_n=val)
    elif val is None:
        df_scales = load_scales(name="scales")
    else:
        df_scales = load_scales(name="scales", top_explain_n=val)
    if subcategories is not None:
        df_cat = load_scales(name="scales_cat")
        keep = set(df_cat[df_cat[ut.COL_SUBCAT].isin(subcategories)][ut.COL_SCALE_ID])
        cols = [c for c in df_scales.columns if c in keep]
        if len(cols) == 0:
            return None
        df_scales = df_scales[cols]
    return df_scales


def _scale_label(spec):
    """Readable Scale-axis level label for ``df_eval`` (e.g. ``explain:30`` / ``top60:3``)."""
    kind, val = spec
    return f"{kind}:{'all' if val is None else val}"


def _resolve_config(search="balanced", kws=None):
    """Merge the ``search``-mode grid with the bounded ``kws`` overrides (unknown keys raise)."""
    mode = _MODES[search]
    cfg = {
        "sweep_parts": mode["sweep_parts"], "sweep_split": mode["sweep_split"],
        "sweep_scales": mode["sweep_scales"], "sweep_jmd": mode["sweep_jmd"],
        "part_sets": list(_PART_SETS) if mode["sweep_parts"] else [_PART_SETS[0]],
        "split_type_sets": list(_SPLIT_TYPE_SETS) if mode["sweep_split"] else [_SPLIT_TYPE_SETS[-1]],
        "n_split_max_vals": list(mode["n_split_max_vals"]),
        "scale_specs": list(mode["scale_specs"]),
        "n_filter_vals": list(mode["n_filter_vals"]),
        "n_jmd_vals": list(mode["n_jmd_vals"]),
        "simplify_strategy": mode["simplify_strategy"],
        "max_cor": 0.5, "max_overlap": 0.5,
        # Pattern span (default 15 = the SequenceFeature.get_split_kws default). A power user lowers
        # it via kws["len_max"] to request shorter Pattern splits on short / free-peptide parts.
        "len_max": 15,
    }
    if kws is not None:
        ut.check_dict(name="kws", val=kws)
        unknown = set(kws) - _KWS_KEYS
        if unknown:
            raise ValueError(f"'kws' keys ({sorted(unknown)}) should be among {sorted(_KWS_KEYS)}.")
        if "n_explain" in kws:
            cfg["scale_specs"] = [("explain", kws["n_explain"])]
            cfg["sweep_scales"] = False
        if "n_split_max" in kws:
            cfg["n_split_max_vals"] = [kws["n_split_max"]]
        if "len_max" in kws:
            cfg["len_max"] = kws["len_max"]
        if "n_filter" in kws:
            cfg["n_filter_vals"] = [kws["n_filter"]]
        if "n_jmd" in kws:
            cfg["n_jmd_vals"] = [kws["n_jmd"]]
            cfg["sweep_jmd"] = False
        if "simplify_strategy" in kws:
            cfg["simplify_strategy"] = kws["simplify_strategy"]
        if "max_cor" in kws:
            cfg["max_cor"] = kws["max_cor"]
        if "max_overlap" in kws:
            cfg["max_overlap"] = kws["max_overlap"]
    # Free-peptide-like: no flanking context requested (every swept JMD length is 0). The composite
    # half-parts (jmd_n_tmd_n / tmd_c_jmd_c) then collapse to half-TMD fragments (an 8-aa peptide's
    # half-part is only 4 aa), needlessly dragging the shortest part length down. Use TMD-only so the
    # whole peptide is one part and 'n_split_max' can go up to the peptide length.
    if all(v == 0 for v in cfg["n_jmd_vals"]):
        cfg["part_sets"] = [["tmd"]]
        cfg["sweep_parts"] = False
    return cfg


def _pareto_mask(means):
    """Boolean mask of Pareto-optimal rows (non-dominated), maximizing every column of ``means``."""
    means = np.asarray(means, dtype=float)
    n = len(means)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(means[j] >= means[i]) and np.any(means[j] > means[i]):
                mask[i] = False
                break
    return mask


def _select_pareto_simplest(df_stage, metric_keys):
    """Mark ``is_pareto`` within a stage and return the index of the simplest Pareto winner."""
    means = df_stage[[m + "_mean" for m in metric_keys]].to_numpy()
    df_stage["is_pareto"] = _pareto_mask(means)
    cand = df_stage[df_stage["is_pareto"]]
    return cand.sort_values(["n_features", "n_filter"], ascending=True).index[0]


def _axis_impact(df_stage, axis_col, metric_keys):
    """Normalized marginal-mean impact of one axis, averaged across metrics (0 = no effect)."""
    impacts = []
    for m in metric_keys:
        col = m + "_mean"
        marg = df_stage.groupby(axis_col)[col].mean()
        rng = df_stage[col].max() - df_stage[col].min()
        impacts.append((marg.max() - marg.min()) / rng if rng > 1e-12 else 0.0)
    return float(np.mean(impacts))


def _refine_keep(df_feat_new, X_new, df_feat_cur, base_scores, labels=None, models=None, cv=5,
                 metrics=None, random_state=None):
    """Keep the refined set iff its CV scores are not Pareto-dominated by the current winner."""
    if df_feat_new is None or len(df_feat_new) == 0 or len(df_feat_new) == len(df_feat_cur):
        return df_feat_cur, base_scores, False
    new = _cv_scores(X_new, labels, models=models, cv=cv, metrics=metrics, random_state=random_state)
    new_means = np.array([new[m][0] for m in metrics])
    cur_means = np.array([base_scores[m][0] for m in metrics])
    dominated = np.all(cur_means >= new_means) and np.any(cur_means > new_means)
    if dominated:
        return df_feat_cur, base_scores, False
    return df_feat_new.reset_index(drop=True), new, True


def _resolve_baselines(baselines):
    """Resolve the ``baselines`` argument to a sorted list of k-mer orders (empty if disabled)."""
    if baselines is False or baselines is None:
        return []
    if baselines is True:
        return [1, 2]
    if isinstance(baselines, (str, bytes)) or not hasattr(baselines, "__iter__"):
        raise ValueError(f"'baselines' should be True/False or a sequence of ints in 1..4; got {baselines!r}.")
    ks = list(baselines)
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1 or int(k) > 4:
            raise ValueError(f"'baselines' entries should be ints in 1..4 (1=AAC, 2=DPC, ...); got {k!r}.")
    return sorted({int(k) for k in ks})


def _run_baselines(ks=None, sf=None, df_seq=None, labels=None, models=None, cv=5, metrics=None,
                   label_test=1, label_ref=0, name_test="TEST", name_ref="REF", n_jmd=10,
                   n_filter=100, max_cor=None, random_state=None, n_jobs=None, verbose=False, plot=True):
    """Composition baselines: AAC (k=1) as a CPP ``df_feat`` + feature map; DPC/k-mer CPP-filtered.

    Each baseline selects up to ``n_filter`` features with CPP's discriminative statistics (AAC via
    :meth:`CPP.run` over one-hot scales; DPC/k-mer via :func:`comp_kmer_df_feat`), cross-validates that
    selected set with the same models/CV as the CPP configs, and returns a reference ``df_eval`` row
    (``is_selected=False``) plus a dict of drawn objects (a ``df_feat`` + feature map for AAC; the
    signal + composition map + filtered ``df_feat`` table for DPC/k-mer) keyed ``AAC`` / ``DPC`` / ``<k>-mer``.
    """
    rows, artifacts = [], {}
    y = np.asarray(labels)
    for k in ks:
        label = {1: "AAC", 2: "DPC"}.get(k, f"{k}-mer")
        if k == 1:
            df_feat_b, df_parts_b, df_scales_b, df_cat_b = build_aac_df_feat(
                sf=sf, df_seq=df_seq, labels=labels, jmd_n_len=n_jmd, jmd_c_len=n_jmd,
                label_test=label_test, label_ref=label_ref, n_filter=n_filter,
                random_state=random_state, n_jobs=n_jobs, verbose=verbose)
            X = sf.feature_matrix(features=df_feat_b[ut.COL_FEATURE], df_parts=df_parts_b,
                                  df_scales=df_scales_b, n_jobs=n_jobs)
            scores = _cv_scores(X, labels, models=models, cv=cv, metrics=metrics, random_state=random_state)
            n_features = len(df_feat_b)
            art = {"df_feat": df_feat_b, "df_scales": df_scales_b, "df_cat": df_cat_b, "ax": None}
            if plot:
                tm = TreeModel(random_state=random_state, verbose=verbose)
                tm.fit(X, labels=labels)
                df_feat_b = tm.add_feat_importance(df_feat=df_feat_b, sort=True)
                _, ax_b = CPPPlot(df_scales=df_scales_b, df_cat=df_cat_b, jmd_n_len=n_jmd,
                                  jmd_c_len=n_jmd, verbose=verbose).feature_map(
                    df_feat=df_feat_b, name_test=name_test, name_ref=name_ref, dict_color=AAC_CAT_COLORS)
                art["df_feat"], art["ax"] = df_feat_b, ax_b
        else:
            # CPP-style discriminative filter of the k-mer composition (positional / scale-category
            # redundancy do not apply), then CV-score exactly the selected k-mers.
            signal, kmers = comp_kmer_signal(sf=sf, df_seq=df_seq, labels=labels, k=k,
                                             label_test=label_test, label_ref=label_ref)
            df_kmer = comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=k, label_test=label_test,
                                        label_ref=label_ref, n_filter=n_filter, max_cor=max_cor)
            X = np.asarray(sf.kmer_composition(df_seq=df_seq, k=k), dtype=float)
            col_idx = {km: i for i, km in enumerate(kmers)}
            sel = [col_idx[km] for km in df_kmer[ut.COL_FEATURE]]
            X_sel = X[:, sel]
            keep = np.isfinite(X_sel).all(axis=1)                # drop spans shorter than k (all-NaN)
            scores = _cv_scores(X_sel[keep], y[keep], models=models, cv=cv, metrics=metrics,
                                random_state=random_state)
            n_features = len(df_kmer)
            art = {"signal": signal, "kmers": kmers, "df_feat": df_kmer, "ax": None}
            if plot:
                art["ax"] = plot_composition_map(signal=signal, k=k, df_feat=df_kmer,
                                                 name_test=name_test, name_ref=name_ref)
        row = {"stage": "baseline", "scale": label, "n_features": n_features,
               "is_pareto": False, "is_selected": False}
        for m in metrics:
            row[m + "_mean"], row[m + "_std"] = scores[m]
        rows.append(row)
        artifacts[label] = art
    return rows, artifacts


# II Main Functions
def find_features(labels: ut.ArrayLike1D,
                  df_seq: pd.DataFrame,
                  search: str = "balanced",
                  simplify: bool = True,
                  model: Union[str, BaseEstimator, List] = "svm",
                  cv: int = 5,
                  metric: Union[str, List[str]] = "balanced_accuracy",
                  kws: Optional[dict] = None,
                  subcategories: Optional[List[str]] = None,
                  top_n: Optional[int] = None,
                  baselines: Union[bool, Sequence[int]] = False,
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

    The search is **staged** so its cost stays interpretable. Stage 1 cross-validates the full
    Cartesian Part × Split × Scale grid (at a reference ``n_filter``) and ranks each axis by its
    **marginal-mean impact**; Stage 2 refines only the single highest-impact axis against
    ``n_filter`` (the others pinned at the stage optimum); Stage 3 refines the winning feature set
    with :meth:`CPP.simplify` and recursive feature elimination. Selection is **multi-objective**:
    within each stage the Pareto-optimal-then-simplest configuration across all ``metric`` wins,
    scored by the average cross-validated performance of one or more ``model`` s. The winner is then
    ranked by tree-based importance and drawn as the CPP feature map. At ``search="fast"`` no search
    is run — the result is byte-identical to the explicit single-CPP path.

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Class labels for the samples (typically, 1=positive/test, 0=negative/reference).
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        Sequence DataFrame, row-aligned to ``labels``. Required, because the Part regions are a
        swept lever and must be (re)built from the sequences via :meth:`SequenceFeature.get_df_parts`.
    search : str, default="balanced"
        Search effort: ``"fast"`` (single default configuration, no search), ``"balanced"`` (sweep
        the Split levers + Scale + symmetric JMD length ``n_jmd`` + ``n_filter``), or
        ``"exhaustive"`` (also sweep the Part region set and the performance-ranked scale sets, with
        a finer grid and a wider ``n_jmd`` range).
    simplify : bool, default=True
        If ``True``, refine the winning feature set with :meth:`CPP.simplify` (kept only if it is
        not Pareto-dominated).
    model : str, estimator, or list, default="svm"
        Selection model(s): ``"svm"``, ``"rf"``, ``"log_reg"``, a scikit-learn estimator, or a list
        of these. A list averages the cross-validated scores across models.
    cv : int, default=5
        Number of cross-validation folds for the selection score, must be > 1.
    metric : str or list of str, default="balanced_accuracy"
        Cross-validation scoring metric(s). A list triggers multi-objective Pareto selection.
    kws : dict, optional
        Bounded power-user overrides; each pins a swept lever to a single value (unknown keys raise).
        Recognized keys: ``n_explain``, ``n_split_max`` (max ``Segment`` splits), ``len_max`` (max
        ``Pattern`` span), ``n_filter``, ``n_jmd`` (the symmetric JMD length ``jmd_n_len =
        jmd_c_len``), ``simplify_strategy``, ``max_cor``, ``max_overlap``. For **free peptides / short
        parts** (no flanking context), pass ``kws={"n_jmd": 0}`` so no JMD is carved out; the search
        then uses **TMD-only** parts (the whole peptide is one part, rather than half-TMD fragments)
        and **caps the swept ``n_split_max``** range to the shortest part length (deduped), with a
        ``UserWarning``. The split config also auto-caps to the shortest part (``Pattern`` /
        ``PeriodicPattern`` that cannot fit are dropped and ``n_split_max`` is clamped). On normal
        (long-part) inputs the range cap is a no-op. Lower ``n_split_max`` / ``len_max`` yourself to
        control which splits are used.
    subcategories : list of str, optional
        AAontology subcategories to restrict the scale sets to. If ``None``, all scales of the grade.
    top_n : int, optional
        If given, keep only the top ``top_n`` features (after importance ranking).
    baselines : bool or sequence of int, default=False
        Add composition baselines for the "how much does positional CPP add over plain composition?"
        comparison. ``True`` adds AAC (k=1) and DPC (k=2); a sequence of ints selects k-mer orders
        (1..4). Each baseline is cross-validated with the same ``model`` / ``cv`` / ``metric`` and
        appended to ``df_eval`` as a reference row (``stage="baseline"``, ``is_selected=False``; the
        returned ``df_feat`` and feature map stay the CPP winner). AAC (k=1) is a genuine CPP
        ``df_feat`` (a one-hot identity scale set with the whole-part ``Segment(1,1)`` split) whose
        feature map is attached; DPC / higher k-mers are attached as composition **signal maps**
        (per-k-mer ``test − ref`` composition: a 20×20 heatmap for k=2, top-N bars for k≥3). When
        ``plot=True`` the drawn objects are attached as ``ax.baselines`` (dict keyed ``AAC`` / ``DPC``
        / ``<k>-mer``).
    label_test : int, default=1
        Class label of the test/positive group passed to :meth:`CPP.run`.
    label_ref : int, default=0
        Class label of the reference/negative group passed to :meth:`CPP.run`.
    name_test : str, default="TEST"
        Display name of the test/positive group in the feature map.
    name_ref : str, default="REF"
        Display name of the reference/negative group in the feature map.
    plot : bool, default=True
        If ``True``, draw the CPP feature map (returned as ``ax``) and the publication eval figures
        (attached as ``ax.eval``); if ``False``, draw nothing and return ``None``.
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
        The feature-map ``Axes`` if ``plot=True``, else ``None``. When a search was run, the
        publication eval figures are attached as ``ax.eval`` (a list of
        :class:`matplotlib.figure.Figure`; empty for a single-configuration ``fast`` search) — see
        :func:`plot_eval`.
    df_eval : pd.DataFrame
        Per-configuration sweep table: the configuration descriptors, one ``<metric>_mean`` /
        ``<metric>_std`` column per metric, plus ``stage``, ``is_pareto`` (Pareto-optimal within its
        stage), ``rank``, and ``is_selected`` (the single winner).

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
    ut.check_str_options(name="search", val=search, list_str_options=list(_MODES))
    ut.check_df_seq(df_seq=df_seq)
    ut.check_bool(name="simplify", val=simplify)
    for m in (model if isinstance(model, (list, tuple)) else [model]):
        if isinstance(m, str):
            ut.check_str_options(name="model", val=m, list_str_options=_LIST_MODELS)
    ut.check_number_range(name="cv", val=cv, min_val=2, just_int=True)
    metrics = list(metric) if isinstance(metric, (list, tuple)) else [metric]
    for m in metrics:
        ut.check_str(name="metric", val=m)
    ut.check_number_range(name="top_n", val=top_n, min_val=1, accept_none=True, just_int=True)
    ut.check_bool(name="plot", val=plot)
    ks_baseline = _resolve_baselines(baselines)
    ut.check_number_range(name="random_state", val=random_state, min_val=0,
                          accept_none=True, just_int=True)
    ut.check_bool(name="verbose", val=verbose)
    cfg = _resolve_config(search=search, kws=kws)
    models = _resolve_models(model, random_state=random_state)

    sf = SequenceFeature(verbose=verbose)
    # Cap the swept 'n_split_max' range to the shortest achievable part and dedupe, so free peptides
    # / short parts do not run redundant configs that all collapse to the same clamped value. On
    # normal (long-part) inputs this is a no-op (no warning, sweep unchanged).
    _cap_n_split_range(sf=sf, df_seq=df_seq, cfg=cfg)
    df_scales_all = load_scales(name="scales")
    common = dict(sf=sf, labels=labels, df_seq=df_seq, cfg=cfg, simplify=simplify, models=models,
                  cv=cv, metrics=metrics, subcategories=subcategories, label_test=label_test,
                  label_ref=label_ref, df_scales_all=df_scales_all, random_state=random_state,
                  n_jobs=n_jobs, verbose=verbose)
    if search == "fast":
        df_feat, df_parts_win, df_eval = _run_fast(**common)
    else:
        df_feat, df_parts_win, df_eval = _run_search(**common)

    # Rank the winning features by tree-based importance (also required by the feature map); use the
    # full scale set since simplify can swap to correlated scales the feature_matrix still reads.
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts_win,
                          df_scales=df_scales_all, n_jobs=n_jobs)
    tm = TreeModel(random_state=random_state, verbose=verbose)
    tm.fit(X, labels=labels)
    df_feat = tm.add_feat_importance(df_feat=df_feat, sort=True)
    if top_n is not None:
        df_feat = df_feat.head(top_n)
    ax = None
    n_jmd_win = int(df_eval.loc[df_eval["is_selected"], "n_jmd"].iloc[0]) if (plot or ks_baseline) else 10
    if plot:
        # Draw the feature map with the winning JMD length so the map geometry matches the parts the
        # winning features were computed on (CPPPlot defaults jmd_n_len = jmd_c_len = 10).
        _, ax = CPPPlot(df_scales=df_scales_all, jmd_n_len=n_jmd_win, jmd_c_len=n_jmd_win,
                        verbose=verbose).feature_map(
            df_feat=df_feat, name_test=name_test, name_ref=name_ref)
        # The feature map is the primary figure (returned as ``ax``); the publication eval figures
        # are attached as ``ax.eval`` (a list — empty for a single-configuration ``fast`` search) so
        # the user can save each individually. This keeps the uniform (df_feat, ax, df_eval) triple.
        ax.eval = plot_eval(df_eval)
    # Composition baselines: AAC (k=1) as a first-class CPP df_feat + feature map, DPC / k-mer as
    # composition signal maps. Scored by the same models/CV and appended to df_eval as reference rows
    # (is_selected=False); the drawn objects are attached as ``ax.baselines`` (dict keyed AAC/DPC/<k>-mer).
    if ks_baseline:
        # Match the CPP winner's feature count so the comparison is apples-to-apples; reuse the CPP
        # redundancy threshold for the k-mer correlation filter.
        base_rows, base_artifacts = _run_baselines(
            ks=ks_baseline, sf=sf, df_seq=df_seq, labels=labels, models=models, cv=cv, metrics=metrics,
            label_test=label_test, label_ref=label_ref, name_test=name_test, name_ref=name_ref,
            n_jmd=n_jmd_win, n_filter=len(df_feat), max_cor=cfg["max_cor"], random_state=random_state,
            n_jobs=n_jobs, verbose=verbose, plot=plot)
        df_eval = pd.concat([df_eval, pd.DataFrame(base_rows)], ignore_index=True)
        if ax is not None:
            ax.baselines = base_artifacts
    return df_feat, ax, df_eval


def _eval_row(stage=None, list_parts=None, split_types=None, n_split_max=None, scale_spec=None,
              n_filter=None, n_features=None, n_jmd=None, scores=None, metrics=None):
    """Assemble one ``df_eval`` row (descriptors + per-metric mean/std columns)."""
    row = {"stage": stage, "list_parts": ",".join(list_parts),
           "split_types": ",".join(split_types), "pattern_mode": _PATTERN_MODE[tuple(split_types)],
           "n_split_max": n_split_max, "scale": _scale_label(scale_spec), "n_jmd": n_jmd,
           "n_filter": n_filter, "n_features": n_features}
    for m in metrics:
        row[m + "_mean"], row[m + "_std"] = scores[m]
    return row


def _run_fast(sf=None, labels=None, df_seq=None, cfg=None, simplify=True, models=None, cv=5,
              metrics=None, subcategories=None, label_test=1, label_ref=0, df_scales_all=None,
              random_state=None, n_jobs=None, verbose=False):
    """Single default configuration: the explicit CPP path, byte-identical to writing it by hand."""
    list_parts, split_types = cfg["part_sets"][0], _SPLIT_TYPE_SETS[-1]
    spec = cfg["scale_specs"][0]
    # Pin the JMD length to the default 10 (no sweep) so fast stays byte-identical to the explicit
    # single-CPP chain (get_df_parts defaults jmd_n_len = jmd_c_len = 10).
    n_jmd = cfg["n_jmd_vals"][0]
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts, jmd_n_len=n_jmd, jmd_c_len=n_jmd)
    df_scales = _load_scale_spec(spec, subcategories=subcategories)
    if df_scales is None:
        raise ValueError(f"'subcategories' ({subcategories}) should be names that match a scale.")
    # Thread the requested Segment n_split_max / Pattern len_max through to the split_kws (was
    # always the default before, silently ignoring kws), and auto-fit to the shortest part so a
    # free peptide / short part drops Pattern-type splits + clamps n_split_max instead of hard
    # erroring. For long-enough parts this is byte-identical to the default split_kws.
    split_kws = _fit_split_kws_to_parts(split_types=split_types,
                                        n_split_max=cfg["n_split_max_vals"][0],
                                        len_max=cfg["len_max"], df_parts=df_parts)
    cpp = CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws,
              random_state=random_state, verbose=verbose)
    df_feat = cpp.run(labels=labels, label_test=label_test, label_ref=label_ref,
                      n_filter=cfg["n_filter_vals"][0], max_cor=cfg["max_cor"],
                      max_overlap=cfg["max_overlap"], n_jobs=n_jobs)
    if simplify:
        # Unconditional simplify (no Pareto keep-guard) keeps fast byte-identical to the explicit
        # single-CPP chain; the search modes instead keep the refine only if not Pareto-dominated.
        df_feat = cpp.simplify(df_feat=df_feat, labels=labels, strategy=cfg["simplify_strategy"],
                               label_test=label_test, label_ref=label_ref)
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts,
                          df_scales=df_scales_all, n_jobs=n_jobs)
    scores = _cv_scores(X, labels, models=models, cv=cv, metrics=metrics, random_state=random_state)
    df_eval = pd.DataFrame([_eval_row(stage="single", list_parts=list_parts, split_types=split_types,
                                      n_split_max=cfg["n_split_max_vals"][0], scale_spec=spec,
                                      n_filter=cfg["n_filter_vals"][0], n_features=len(df_feat),
                                      n_jmd=n_jmd, scores=scores, metrics=metrics)])
    df_eval["is_pareto"] = True
    df_eval["rank"] = 1
    df_eval["is_selected"] = True
    return df_feat, df_parts, df_eval


def _grid_stage(sf=None, df_seq=None, parts=None, split_sets=None, n_split_vals=None, specs=None,
                n_filters=None, n_jmd_vals=None, cfg=None, labels=None, label_test=1, label_ref=0,
                models=None, cv=5, metrics=None, subcategories=None, df_scales_all=None,
                random_state=None, n_jobs=None, verbose=False, stage=None, parts_cache=None):
    """Run a CPPGrid sweep over the given axis levels, CV-score every config, return eval rows.

    Returns ``(rows, payloads)`` aligned: ``payloads[i]`` carries the kept ``df_feat`` and the parts
    plus the axis levels needed to rebuild / pin the winner. The JMD length is swept symmetrically:
    ``jmd_n_len`` and ``jmd_c_len`` are both varied over ``n_jmd_vals`` and only the diagonal
    (``jmd_n_len == jmd_c_len == n_jmd``) is kept, so a winning ``n_jmd`` is unambiguous.
    """
    scale_dfs, scale_specs = [], []
    for spec in specs:
        df_s = _load_scale_spec(spec, subcategories=subcategories)
        if df_s is not None:
            scale_dfs.append(df_s)
            scale_specs.append(spec)
    if not scale_dfs:
        raise ValueError(f"'subcategories' ({subcategories}) should match at least one scale.")
    params_parts = {"list_parts": [list(p) for p in parts],
                    "jmd_n_len": list(n_jmd_vals), "jmd_c_len": list(n_jmd_vals)}
    params_split = {"split_types": [list(s) for s in split_sets], "n_split_max": list(n_split_vals),
                    "len_max": cfg["len_max"]}
    params_cpp = {"n_filter": list(n_filters), "label_test": label_test, "label_ref": label_ref,
                  "max_cor": cfg["max_cor"], "max_overlap": cfg["max_overlap"]}
    cppg = CPPGrid(df_seq=df_seq, labels=labels, random_state=random_state, verbose=verbose,
                   n_jobs=n_jobs)
    list_df_feat, df_params = cppg.run(params_parts=params_parts, params_split=params_split,
                                       params_scales=scale_dfs, params_cpp=params_cpp)

    def _parts_for(lp, n_jmd):
        key = (tuple(lp), n_jmd)
        if key not in parts_cache:
            parts_cache[key] = sf.get_df_parts(df_seq=df_seq, list_parts=list(lp),
                                               jmd_n_len=n_jmd, jmd_c_len=n_jmd)
        return parts_cache[key]

    rows, payloads = [], []
    for i, df_feat_i in enumerate(list_df_feat):
        if df_feat_i is None or len(df_feat_i) == 0:
            continue
        rec = df_params.iloc[i]
        # Keep only the symmetric JMD-length diagonal (jmd_n_len == jmd_c_len); the off-diagonal
        # combinations CPPGrid also produces are dropped so the kept n_jmd is unambiguous.
        n_jmd_n, n_jmd_c = int(rec["jmd_n_len"]), int(rec["jmd_c_len"])
        if n_jmd_n != n_jmd_c:
            continue
        n_jmd = n_jmd_n
        lp = parts[int(rec["list_parts"])] if len(parts) > 1 else parts[0]
        sts = split_sets[int(rec["split_types"])] if len(split_sets) > 1 else split_sets[0]
        spec = scale_specs[int(rec["df_scales"])] if len(scale_specs) > 1 else scale_specs[0]
        df_parts_i = _parts_for(lp, n_jmd)
        X_i = sf.feature_matrix(features=df_feat_i[ut.COL_FEATURE], df_parts=df_parts_i,
                                df_scales=df_scales_all, n_jobs=n_jobs)
        scores = _cv_scores(X_i, labels, models=models, cv=cv, metrics=metrics,
                            random_state=random_state)
        rows.append(_eval_row(stage=stage, list_parts=lp, split_types=sts,
                              n_split_max=int(rec["n_split_max"]), scale_spec=spec,
                              n_filter=int(rec["n_filter"]), n_features=len(df_feat_i),
                              n_jmd=n_jmd, scores=scores, metrics=metrics))
        payloads.append({"df_feat": df_feat_i.reset_index(drop=True), "df_parts": df_parts_i,
                         "list_parts": lp, "split_types": sts, "n_split_max": int(rec["n_split_max"]),
                         "scale_spec": spec, "n_filter": int(rec["n_filter"]), "n_jmd": n_jmd})
    return rows, payloads


def _run_search(sf=None, labels=None, df_seq=None, cfg=None, simplify=True, models=None, cv=5,
                metrics=None, subcategories=None, label_test=1, label_ref=0, df_scales_all=None,
                random_state=None, n_jobs=None, verbose=False):
    """Staged sensitivity search: Cartesian P×S×Scale → dominant axis × n_filter → simplify+RFE."""
    parts_cache = {}
    common = dict(sf=sf, df_seq=df_seq, cfg=cfg, labels=labels, label_test=label_test,
                  label_ref=label_ref, models=models, cv=cv, metrics=metrics,
                  subcategories=subcategories, df_scales_all=df_scales_all,
                  random_state=random_state, n_jobs=n_jobs, verbose=verbose, parts_cache=parts_cache)
    ref_nfilter = max(cfg["n_filter_vals"])

    # Stage 1 — full Cartesian Part × Split × Scale × JMD-length at the reference n_filter.
    rows1, pay1 = _grid_stage(parts=cfg["part_sets"], split_sets=cfg["split_type_sets"],
                              n_split_vals=cfg["n_split_max_vals"], specs=cfg["scale_specs"],
                              n_filters=[ref_nfilter], n_jmd_vals=cfg["n_jmd_vals"],
                              stage="sensitivity", **common)
    if not rows1:
        raise RuntimeError("'find_features' produced no valid configurations; relax 'kws' / "
                           "'subcategories' or use a less restrictive 'search'.")
    df1 = pd.DataFrame(rows1)
    win1_pos = _select_pareto_simplest(df1, metrics)
    win1 = pay1[df1.index.get_loc(win1_pos)]

    # Rank the swept axes by normalized marginal-mean impact; the dominant axis gets the n_filter sweep.
    df1["_split_lvl"] = df1["pattern_mode"] + "/" + df1["n_split_max"].astype(str)
    axis_cols = {"part": "list_parts", "split": "_split_lvl", "scale": "scale", "n_jmd": "n_jmd"}
    swept = [a for a, col in axis_cols.items() if df1[col].nunique() > 1]
    impacts = {a: _axis_impact(df1, axis_cols[a], metrics) for a in swept}
    dominant = max(impacts, key=impacts.get) if impacts else "scale"

    # Stage 2 — sweep the dominant axis's levels × n_filter, the others pinned at the Stage-1 winner.
    parts2 = cfg["part_sets"] if dominant == "part" else [win1["list_parts"]]
    if dominant == "split":
        split2, nsplit2 = cfg["split_type_sets"], cfg["n_split_max_vals"]
    else:
        split2, nsplit2 = [win1["split_types"]], [win1["n_split_max"]]
    specs2 = cfg["scale_specs"] if dominant == "scale" else [win1["scale_spec"]]
    njmd2 = cfg["n_jmd_vals"] if dominant == "n_jmd" else [win1["n_jmd"]]
    rows2, pay2 = _grid_stage(parts=parts2, split_sets=split2, n_split_vals=nsplit2, specs=specs2,
                              n_filters=cfg["n_filter_vals"], n_jmd_vals=njmd2, stage="n_filter",
                              **common)
    df2 = pd.DataFrame(rows2)
    win2_pos = _select_pareto_simplest(df2, metrics)
    win2 = pay2[df2.index.get_loc(win2_pos)]

    # Stage 3 — refine the winner (simplify + RFE), each kept only if not Pareto-dominated.
    df_feat_win, df_parts_win = win2["df_feat"], win2["df_parts"]
    df_scales_win = _load_scale_spec(win2["scale_spec"], subcategories=subcategories)
    X_win = sf.feature_matrix(features=df_feat_win[ut.COL_FEATURE], df_parts=df_parts_win,
                              df_scales=df_scales_all, n_jobs=n_jobs)
    base = _cv_scores(X_win, labels, models=models, cv=cv, metrics=metrics, random_state=random_state)
    rows3 = []
    if simplify:
        # Rebuild the winner's split_kws (Segment n_split_max / Pattern len_max). CPP auto-caps it to
        # short / free-peptide parts (no raise); simplify operates on the existing df_feat (it never
        # reads split_kws), so this does not change the result for normal parts.
        split_kws_win = _split_kws_for(split_types=win2["split_types"],
                                       n_split_max=win2["n_split_max"], len_max=cfg["len_max"])
        cpp_win = CPP(df_parts=df_parts_win, df_scales=df_scales_win, split_kws=split_kws_win,
                      random_state=random_state, verbose=verbose)
        df_simpl = cpp_win.simplify(df_feat=df_feat_win, labels=labels,
                                    strategy=cfg["simplify_strategy"], ml_cv=cv,
                                    label_test=label_test, label_ref=label_ref)
        X_simpl = sf.feature_matrix(features=df_simpl[ut.COL_FEATURE], df_parts=df_parts_win,
                                    df_scales=df_scales_all, n_jobs=n_jobs)
        df_feat_win, base, kept = _refine_keep(df_simpl, X_simpl, df_feat_win, base, labels=labels,
                                               models=models, cv=cv, metrics=metrics,
                                               random_state=random_state)
        if kept:
            rows3.append(_eval_row(stage="refine", list_parts=win2["list_parts"],
                                   split_types=win2["split_types"], n_split_max=win2["n_split_max"],
                                   scale_spec=win2["scale_spec"], n_filter=win2["n_filter"],
                                   n_features=len(df_feat_win), n_jmd=win2["n_jmd"],
                                   scores=base, metrics=metrics))
    df_feat_win, df_parts_win = _refine_rfe_winner(
        df_feat_win, df_parts_win, df_scales_all, base, sf=sf, labels=labels, models=models, cv=cv,
        metrics=metrics, random_state=random_state, n_jobs=n_jobs, rows3=rows3, win=win2)

    # Assemble df_eval: all stages stacked. is_selected marks the actually-returned df_feat (the
    # final refined winner = the last kept refine row, else the Stage-2 winner) — which need NOT be
    # rank 1 under multi-objective selection. rank orders by the first metric for readability.
    df3 = pd.DataFrame(rows3)
    if len(df3):
        df3["is_pareto"] = True   # a kept refine config survived the Pareto non-domination check
    df_eval = pd.concat([df1.drop(columns=["_split_lvl"]), df2, df3], ignore_index=True)
    df_eval["is_pareto"] = df_eval["is_pareto"].fillna(False)
    winner_pos = (len(df_eval) - 1) if len(df3) else (len(df1) + df2.index.get_loc(win2_pos))
    df_eval["is_selected"] = False
    df_eval.loc[winner_pos, "is_selected"] = True
    primary = metrics[0] + "_mean"
    df_eval["rank"] = df_eval[primary].rank(ascending=False, method="first").astype(int)
    df_eval = df_eval.sort_values("rank").reset_index(drop=True)
    return df_feat_win, df_parts_win, df_eval


def _refine_rfe_winner(df_feat, df_parts, df_scales_all, base, sf=None, labels=None, models=None,
                       cv=5, metrics=None, random_state=None, n_jobs=None, rows3=None, win=None):
    """RFE post-step on the winner: keep the reduced set only if it is not Pareto-dominated."""
    n_feat = len(df_feat)
    base_means = np.array([base[m][0] for m in metrics])
    if n_feat < 8 or np.all(base_means >= 1.0 - 1e-9):
        return df_feat, df_parts
    n_feat_min = max(4, n_feat // 3)
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts,
                          df_scales=df_scales_all, n_jobs=n_jobs)
    tm = TreeModel(random_state=random_state, verbose=False)
    tm.fit(X, labels=labels, use_rfe=True, n_cv=cv, n_feat_min=n_feat_min, n_feat_max=n_feat,
           metric=metrics[0])
    mask = np.asarray(tm.is_selected_).mean(axis=0) >= 0.5
    if mask.sum() < n_feat_min or mask.all():
        return df_feat, df_parts
    df_rfe = df_feat[mask].reset_index(drop=True)
    df_feat_new, base_new, kept = _refine_keep(df_rfe, X[:, mask], df_feat, base, labels=labels,
                                               models=models, cv=cv, metrics=metrics,
                                               random_state=random_state)
    if kept and rows3 is not None:
        rows3.append(_eval_row(stage="refine", list_parts=win["list_parts"],
                               split_types=win["split_types"], n_split_max=win["n_split_max"],
                               scale_spec=win["scale_spec"], n_filter=win["n_filter"],
                               n_features=len(df_feat_new), n_jmd=win["n_jmd"],
                               scores=base_new, metrics=metrics))
    return df_feat_new, df_parts
