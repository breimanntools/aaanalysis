"""
This is a script for the backend of CPP's simplify method: interpretability-
guided scale swapping. For each feature (``PART-SPLIT-SCALE``) a more
interpretable, correlated scale is substituted (``PART-SPLIT`` preserved), the
feature stats are recomputed, and the swap is validated by a cross-validation
non-regression gate. The set is then redundancy-reduced so the result speaks in
fewer, more interpretable subcategories.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import aaanalysis.utils as ut
from .utils_feature import _feature_value, _get_dict_all_scales
from ._utils_feature_stat import add_stat_

# candidate_search='fast' cap: evaluate at most this many candidates per feature
# (the top-k of the ranked best-first list). Chosen as the smallest k that keeps
# the heuristic within the documented quality band (kept-feature Jaccard >= 0.95,
# ΔavgABS_AUC <= 0.005 vs exact) with margin on the canonical DOM_GSEC cell; see
# the validation harness (dev_scripts/perf_simplify_validate.py) and the T3
# regression anchor (tests/unit/cpp_tests/test_cpp_simplify_regression.py).
_FAST_TOP_K = 10


# I Helper Functions
def _cap_candidates_(candidates=None, top_k=None):
    """Top-``top_k`` prefix of the ranked (best-first) candidate list.

    ``top_k=None`` (``candidate_search='exact'``) returns the list unchanged, so
    the exact path stays byte-identical; a finite ``top_k``
    (``candidate_search='fast'``) keeps only the most promising candidates."""
    if top_k is None:
        return candidates
    return candidates[:top_k]


def _interp(scale_id=None, dict_interp=None):
    """Interpretability rating of a scale's subcategory (NaN if unrated/absent)."""
    val = dict_interp.get(scale_id, np.nan)
    if val is None or pd.isna(val):
        return np.nan
    return float(val)


def _load_candidate_pool_():
    """Load the full rated AAontology pool and precompute the scale-correlation matrix.

    Returns ``(df_scales_pool, df_cor, dict_all_scales, dict_interp, dict_meta)``
    where ``dict_interp`` maps scale id to its per-subcategory interpretability
    rating (1 = best) and ``dict_meta`` maps scale id to its
    ``(category, subcategory, scale_name, scale_description)``.
    """
    df_scales_pool = ut.load_default_scales()  # (20 AA, n_scales)
    df_cat_pool = ut.load_default_scales(scale_cat=True)  # per-scale classification
    df_cor = df_scales_pool.corr()  # scale x scale Pearson
    dict_all_scales = _get_dict_all_scales(df_scales=df_scales_pool)
    # Interpretability is a per-subcategory rating (single source: df_subcat); map it onto scales.
    df_subcat = ut.load_default_subcat()
    interp_by_subcat = dict(zip(df_subcat[ut.COL_SUBCAT], df_subcat[ut.COL_INTERPRET_GRADE]))
    dict_interp = {
        sid: interp_by_subcat.get(sub)
        for sid, sub in zip(df_cat_pool[ut.COL_SCALE_ID], df_cat_pool[ut.COL_SUBCAT])
    }
    dict_meta = {
        sid: (cat, sub, name, des)
        for sid, cat, sub, name, des in zip(
            df_cat_pool[ut.COL_SCALE_ID],
            df_cat_pool[ut.COL_CAT],
            df_cat_pool[ut.COL_SUBCAT],
            df_cat_pool[ut.COL_SCALE_NAME],
            df_cat_pool[ut.COL_SCALE_DES],
        )
    }
    return df_scales_pool, df_cor, dict_all_scales, dict_interp, dict_meta


def _resolve_ml_model_(ml_model=None, random_state=None):
    """Resolve a string preset or a custom estimator into an sklearn estimator.

    Presets: ``'svm'`` (default), ``'rf'``, ``'log_reg'`` (constructed with the
    resolved ``random_state``). A non-string ``ml_model`` is a user-configured
    estimator instance, returned as-is (the user owns its parameters)."""
    if not isinstance(ml_model, str):
        return ml_model
    if ml_model == ut.MODEL_SVM:
        return SVC(class_weight="balanced", random_state=random_state)
    if ml_model == ut.MODEL_RF:
        return RandomForestClassifier(random_state=random_state)
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _score_feature_set_(
    X=None, labels=None, ml_cv=5, ml_metric="balanced_accuracy", estimator=None
):
    """Mean cross-validation score of ``estimator`` on a feature matrix ``X``.

    Mirrors the ``cross_val_score`` primitive used by ``TreeModel.eval`` so the
    score is consistent with the rest of the package. ``cross_val_score`` clones
    the estimator per fold, so one instance can be reused across many calls."""
    return float(
        cross_val_score(estimator, X, y=labels, cv=ml_cv, scoring=ml_metric).mean()
    )


def _recompute_swapped_row_(
    feat_id=None,
    scale_new=None,
    df_parts=None,
    dict_all_scales=None,
    dict_meta=None,
    labels=None,
    label_test=1,
    label_ref=0,
    parametric=False,
    accept_gaps=False,
    positions=None,
):
    """Recompute a single ``PART-SPLIT-scale_new`` feature: values, stats, metadata.

    ``PART-SPLIT`` is preserved, so ``positions`` are copied from the original
    feature row (no recompute needed). Returns ``(row_df, col_values)``."""
    part, split, _ = ut.split_feat_id(feat_id=feat_id)
    new_feat_id = ut.join_feat_id(part=part, split=split, scale_id=scale_new)
    col = _feature_value(
        df_parts=df_parts[part.lower()],
        split=split,
        dict_scale=dict_all_scales[scale_new],
        accept_gaps=accept_gaps,
    )
    row_df = pd.DataFrame({ut.COL_FEATURE: [new_feat_id]})
    row_df = add_stat_(
        df=row_df,
        X=col[:, None],
        labels=labels,
        parametric=parametric,
        label_test=label_test,
        label_ref=label_ref,
    )
    cat, sub, name, des = dict_meta[scale_new]
    row_df[ut.COL_CAT] = cat
    row_df[ut.COL_SUBCAT] = sub
    row_df[ut.COL_SCALE_NAME] = name
    row_df[ut.COL_SCALE_DES] = des
    row_df[ut.COL_POSITION] = [positions]
    return row_df, col


def _interp_array_(df_cor=None, dict_interp=None):
    """Per-scale interpretability ratings aligned to ``df_cor.index`` (float, NaN if
    unrated). Hoist this ONCE per simplify call and pass it to
    :func:`_eligible_candidates_` as ``interp_arr`` so the per-scale ``_interp``
    scan is not repeated for every targeted feature (the #186 cross-feature hoist).
    ``dict_interp`` is constant across a simplify call, so the array is too."""
    scales = df_cor.index.to_numpy()
    interp_arr = np.array(
        [_interp(scale_id=s, dict_interp=dict_interp) for s in scales], dtype=float
    )
    return scales, interp_arr


def _eligible_candidates_(
    feat_id=None, df_cor=None, dict_interp=None, min_cor=0.7, interp_arr=None
):
    """Eligible candidate scales for one feature, ranked by (interpretability, |corr|).

    Eligible iff the candidate subcategory rating is strictly better (lower) than
    the feature's current scale rating AND ``|corr(candidate, original)| >=
    min_cor`` (anti-correlation allowed via ``abs``). Returns a list of
    ``(scale_cand, interp_cand, abs_cor, cor)``.

    The per-candidate scan over the ``df_cor[scale_old]`` column (a Python
    ``_interp`` + ``float`` per scale) is vectorized with numpy: the
    interpretability ratings and the correlation column are hoisted to arrays,
    filtered in one numpy pass. The small surviving candidate list is then ranked
    with the original Python ``list.sort(key=lambda t: (t[1], -t[2]))`` (primary:
    ``interp_cand`` ascending, secondary: ``abs_cor`` descending) so the output
    list — values, dtypes (Python ``float``), AND tie order (including NaN
    correlation cells, which are kept and stay in ``df_cor`` index order within
    their interp group) — is byte-identical to the original.

    ``interp_arr`` is an OPTIONAL precomputed ``(scales, interp_arr)`` pair from
    :func:`_interp_array_` (the cross-feature hoist): since ``dict_interp`` is
    constant across a simplify call, callers compute it once and pass it for every
    feature instead of rebuilding it per feature. Defaulting to ``None`` rebuilds
    it locally, so the standalone behavior is unchanged."""
    scale_old = ut.split_feat_id(feat_id=feat_id)[2]
    interp_old = _interp(scale_id=scale_old, dict_interp=dict_interp)
    if np.isnan(interp_old) or scale_old not in df_cor.columns:
        return []
    # Hoist the interp rating per scale and the correlation column to arrays,
    # following df_cor's index order (the original loop iterated cors.items()).
    if interp_arr is None:
        scales, interp_arr = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
    else:
        scales, interp_arr = interp_arr
    cor_arr = df_cor[scale_old].to_numpy(dtype=float)
    abs_cor_arr = np.abs(cor_arr)
    # Filter mirrors the original element-by-element guards. A NaN correlation
    # cell passes the min_cor test (nan < x is False), matching the original.
    keep = (
        (scales != scale_old)
        & ~np.isnan(interp_arr)
        & (interp_arr < interp_old)
        & ~(abs_cor_arr < min_cor)
    )
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    # Build the kept candidates in original df_cor index order (flatnonzero is
    # ascending), then rank with the SAME Python sort as the original
    # (key = (interp_cand, -abs_cor)). Sorting in Python — not np.lexsort —
    # reproduces list.sort's exact tie behavior, including NaN correlation cells:
    # these are kept (nan < min_cor is False) but must stay in index order within
    # their interp group, whereas np.lexsort would push the NaN secondary key to
    # the tail of the group. The survivor list is small, so this is not the
    # bottleneck (the per-scale interp/correlation scan was).
    candidates = [
        (
            str(scales[k]),
            float(interp_arr[k]),
            float(abs_cor_arr[k]),
            float(cor_arr[k]),
        )
        for k in idx
    ]
    candidates.sort(key=lambda t: (t[1], -t[2]))
    return candidates


def _select_targets_(df_feat=None, dict_interp=None, max_interpret_grade=None):
    """Order of features to attempt (worst-interpretability first).

    Only features whose current scale is rated (present in the pool) are
    targetable. Returns a list of ``(row_index, feat_id, interp_old)``."""
    rated = []
    for i, feat_id in enumerate(df_feat[ut.COL_FEATURE]):
        interp_old = _interp(
            scale_id=ut.split_feat_id(feat_id=feat_id)[2], dict_interp=dict_interp
        )
        if not np.isnan(interp_old):
            rated.append((i, feat_id, interp_old))
    rated.sort(key=lambda t: -t[2])
    if max_interpret_grade is not None:
        return [t for t in rated if t[2] > max_interpret_grade]
    return [t for t in rated if t[2] > 1]


def _build_base_matrix_(
    df_feat=None, df_parts=None, df_scales_self=None, accept_gaps=False
):
    """Recompute the (n_samples, n_features) matrix for the current feature set."""
    dict_self = _get_dict_all_scales(df_scales=df_scales_self)
    features = list(df_feat[ut.COL_FEATURE])
    X = np.empty((len(df_parts), len(features)))
    for j, feat in enumerate(features):
        part, split, scale = ut.split_feat_id(feat_id=feat)
        X[:, j] = _feature_value(
            df_parts=df_parts[part.lower()],
            split=split,
            dict_scale=dict_self[scale],
            accept_gaps=accept_gaps,
        )
    return X


def _merged_scale_corr_(df_feat=None, df_scales_pool=None, df_scales_self=None):
    """Correlation lookup covering every scale present in ``df_feat``.

    Swapped scales come from the pool; unchanged (possibly custom) scales come
    from ``df_scales_self``. Build the union once and correlate only the present
    scales (pool values win for shared ids; they are identical AAontology scales)."""
    present = [ut.split_feat_id(feat_id=f)[2] for f in df_feat[ut.COL_FEATURE]]
    extra = [c for c in df_scales_self.columns if c not in df_scales_pool.columns]
    df_all = (
        pd.concat([df_scales_pool, df_scales_self[extra]], axis=1)
        if extra
        else df_scales_pool
    )
    present = [s for s in dict.fromkeys(present) if s in df_all.columns]
    return df_all[present].corr()


def _is_redundant_(feat, kept, dict_c, dict_p, dict_scale, cor_np, cpos,
                   max_overlap, max_cor, check_cat):
    """A feature is redundant with one of ``kept`` when their positions overlap
    (``>= max_overlap`` or one is a subset) AND their scales positively correlate
    (signed ``cor > max_cor``, matching ``CPP.run``'s ``filtering`` — anti-correlated
    scales are NOT redundant), within the same category when ``check_cat``.

    The scale-correlation lookup is done against the numpy view ``cor_np`` plus the
    column->index map ``cpos`` (built once by the caller) instead of a per-pair
    double pandas ``__getitem__``; ``dict_scale`` maps feat id -> its scale id. This
    is a pure access change — the sequential greedy ``kept`` scan is unchanged, so
    the kept set and its order are byte-identical to the pandas form."""
    pos = dict_p[feat]
    si = cpos.get(dict_scale[feat])
    for top in kept:
        if check_cat and dict_c[feat] != dict_c[top]:
            continue
        top_pos = dict_p[top]
        overlap = len(top_pos.intersection(pos)) / len(top_pos.union(pos))
        if overlap >= max_overlap or pos.issubset(top_pos):
            ti = cpos.get(dict_scale[top])
            if si is not None and ti is not None:
                if float(cor_np[ti, si]) > max_cor:
                    return True
    return False


def _apply_redundancy_(
    df_feat=None,
    df_cor=None,
    dict_interp=None,
    swapped_ids=None,
    max_overlap=0.5,
    max_cor=0.5,
    check_cat=True,
    tie_break=ut.TIE_BREAK_INTERPRETABILITY,
):
    """Drop only *swapped* features that became redundant; **protect originals**.

    Original (unswapped) features are always kept — simplify never drops a feature
    the user already had, it only removes a swapped (new) feature when the swap made
    it redundant with a kept feature (an original or an already-kept swap). Swapped
    features are considered in ``tie_break`` order: ``interpretability`` (most
    interpretable first, then ``abs_auc``) or ``performance`` (``abs_auc``)."""
    swapped_ids = swapped_ids or set()
    df = df_feat.copy().reset_index(drop=True)
    feats = list(df[ut.COL_FEATURE])
    dict_c = dict(zip(feats, df[ut.COL_CAT]))
    dict_p = dict(zip(feats, [set(x) for x in df[ut.COL_POSITION]]))
    dict_auc = dict(zip(feats, df[ut.COL_ABS_AUC]))
    # Convert df_cor to a numpy view + a column->index map ONCE so the inner
    # redundancy test does an O(1) numpy lookup instead of a double pandas
    # __getitem__ (the profiled hot spot). dict_scale maps feat id -> scale id.
    dict_scale = {f: ut.split_feat_id(feat_id=f)[2] for f in feats}
    cor_np = df_cor.to_numpy()
    cpos = {c: k for k, c in enumerate(df_cor.columns)}
    originals = [f for f in feats if f not in swapped_ids]
    swapped = [f for f in feats if f in swapped_ids]
    if tie_break == ut.TIE_BREAK_INTERPRETABILITY:

        def _key(f):
            i = _interp(
                scale_id=ut.split_feat_id(feat_id=f)[2], dict_interp=dict_interp
            )
            return (i if not np.isnan(i) else np.inf, -dict_auc[f])

        swapped.sort(key=_key)
    else:
        swapped.sort(key=lambda f: -dict_auc[f])
    kept = list(originals)  # originals are protected — always kept
    for feat in swapped:
        if not _is_redundant_(
            feat, kept, dict_c, dict_p, dict_scale, cor_np, cpos,
            max_overlap, max_cor, check_cat
        ):
            kept.append(feat)
    return df[df[ut.COL_FEATURE].isin(kept)].reset_index(drop=True)


def _build_df_candidates_(records=None):
    """Tidy long-form report of every candidate considered (one row per candidate)."""
    cols = [
        "feature",
        "candidate_scale",
        "interpretability_orig",
        "interpretability_cand",
        "cor",
        "std_test",
        "accepted",
        "cv_score",
        "reason",
    ]
    return pd.DataFrame(records, columns=cols)


# II Main Functions
def _greedy_simplify_(
    df_feat=None,
    df_parts=None,
    labels=None,
    X=None,
    df_cor=None,
    dict_all_scales=None,
    dict_interp=None,
    dict_meta=None,
    max_interpret_grade=None,
    min_cor=0.7,
    ml_metric="balanced_accuracy",
    ml_th=0.0,
    ml_cv=5,
    on_unimprovable=ut.ON_UNIMPROVABLE_KEEP,
    label_test=1,
    label_ref=0,
    parametric=False,
    accept_gaps=False,
    max_std_test=0.2,
    estimator=None,
    top_k=None,
):
    """Per-feature swap with a CV non-regression gate. Returns
    ``(df_feat_new, X_kept, records, baseline)``.

    ``top_k`` (``candidate_search='fast'``) caps the candidates evaluated per
    feature; ``None`` (``'exact'``) evaluates them all (byte-identical path)."""
    n = len(df_feat)
    baseline = _score_feature_set_(
        X=X, labels=labels, ml_cv=ml_cv, ml_metric=ml_metric, estimator=estimator
    )
    targets = _select_targets_(
        df_feat=df_feat,
        dict_interp=dict_interp,
        max_interpret_grade=max_interpret_grade,
    )
    new_rows = {}  # row_index -> recomputed one-row df (accepted swaps)
    dropped = set()  # row_index dropped via on_unimprovable
    records = []
    interp_arr = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
    for i, feat_id, interp_old in targets:
        positions = df_feat.iloc[i][ut.COL_POSITION]
        cands = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
            min_cor=min_cor, interp_arr=interp_arr,
        )
        cands = _cap_candidates_(candidates=cands, top_k=top_k)
        accepted = False
        for scale_cand, interp_cand, abs_cor, cor in cands:
            row_df, col = _recompute_swapped_row_(
                feat_id=feat_id,
                scale_new=scale_cand,
                df_parts=df_parts,
                dict_all_scales=dict_all_scales,
                dict_meta=dict_meta,
                labels=labels,
                label_test=label_test,
                label_ref=label_ref,
                parametric=parametric,
                accept_gaps=accept_gaps,
                positions=positions,
            )
            std_test = float(row_df[ut.COL_STD_TEST].iloc[0])
            if std_test > max_std_test:
                records.append(
                    [
                        feat_id,
                        scale_cand,
                        interp_old,
                        interp_cand,
                        cor,
                        std_test,
                        False,
                        np.nan,
                        "max_std_test",
                    ]
                )
                continue
            # Score in place: cross_val_score only READS X (it clones the
            # estimator per fold), so mutate the single trial column instead of
            # copying the whole matrix. Save the old column to restore on reject;
            # the scored matrix is byte-identical to a full X.copy() trial.
            old_col = X[:, i].copy()
            X[:, i] = col
            score = _score_feature_set_(
                X=X,
                labels=labels,
                ml_cv=ml_cv,
                ml_metric=ml_metric,
                estimator=estimator,
            )
            if score >= baseline - ml_th:
                # accepted -> col already in place
                new_rows[i] = row_df
                baseline = score
                accepted = True
                records.append(
                    [
                        feat_id,
                        scale_cand,
                        interp_old,
                        interp_cand,
                        cor,
                        std_test,
                        True,
                        score,
                        "accepted",
                    ]
                )
                break
            # rejected -> restore the original column so X is unchanged
            X[:, i] = old_col
            records.append(
                [
                    feat_id,
                    scale_cand,
                    interp_old,
                    interp_cand,
                    cor,
                    std_test,
                    False,
                    score,
                    "cv_drop",
                ]
            )
        if not accepted and on_unimprovable != ut.ON_UNIMPROVABLE_KEEP:
            n_kept = n - len(dropped)
            if n_kept > 1:  # never drop the last feature
                if on_unimprovable == ut.ON_UNIMPROVABLE_DROP:
                    dropped.add(i)
                elif on_unimprovable == ut.ON_UNIMPROVABLE_DROP_IF_PERF:
                    keep_idx = [j for j in range(n) if j not in dropped and j != i]
                    score_drop = _score_feature_set_(
                        X=X[:, keep_idx],
                        labels=labels,
                        ml_cv=ml_cv,
                        ml_metric=ml_metric,
                        estimator=estimator,
                    )
                    if score_drop >= baseline - ml_th:
                        dropped.add(i)
                        baseline = score_drop
    # Assemble simplified df_feat (swapped rows replace originals; drop unimprovable)
    keep_idx = [i for i in range(n) if i not in dropped]
    rows = [new_rows[i] if i in new_rows else df_feat.iloc[[i]] for i in keep_idx]
    df_feat_new = pd.concat(rows, ignore_index=True)
    X_kept = X[:, keep_idx]
    return df_feat_new, X_kept, records, baseline


def _swap_all_simplify_(
    df_feat=None,
    df_parts=None,
    labels=None,
    df_cor=None,
    dict_all_scales=None,
    dict_interp=None,
    dict_meta=None,
    max_interpret_grade=None,
    min_cor=0.7,
    on_unimprovable=ut.ON_UNIMPROVABLE_KEEP,
    label_test=1,
    label_ref=0,
    parametric=False,
    accept_gaps=False,
    max_std_test=0.2,
):
    """Apply every eligible best-candidate swap, no CV scoring (fastest).

    ``on_unimprovable``: only ``'drop'`` is meaningful without scoring;
    ``'drop_if_perf_allows'`` degrades to ``'keep'``. Returns
    ``(df_feat_new, records)``."""
    n = len(df_feat)
    targets = _select_targets_(
        df_feat=df_feat,
        dict_interp=dict_interp,
        max_interpret_grade=max_interpret_grade,
    )
    new_rows = {}
    dropped = set()
    records = []
    interp_arr = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
    for i, feat_id, interp_old in targets:
        positions = df_feat.iloc[i][ut.COL_POSITION]
        cands = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
            min_cor=min_cor, interp_arr=interp_arr,
        )
        swapped = False
        for scale_cand, interp_cand, abs_cor, cor in cands:
            row_df, col = _recompute_swapped_row_(
                feat_id=feat_id,
                scale_new=scale_cand,
                df_parts=df_parts,
                dict_all_scales=dict_all_scales,
                dict_meta=dict_meta,
                labels=labels,
                label_test=label_test,
                label_ref=label_ref,
                parametric=parametric,
                accept_gaps=accept_gaps,
                positions=positions,
            )
            std_test = float(row_df[ut.COL_STD_TEST].iloc[0])
            if std_test > max_std_test:
                records.append(
                    [
                        feat_id,
                        scale_cand,
                        interp_old,
                        interp_cand,
                        cor,
                        std_test,
                        False,
                        np.nan,
                        "max_std_test",
                    ]
                )
                continue
            new_rows[i] = row_df
            swapped = True
            records.append(
                [
                    feat_id,
                    scale_cand,
                    interp_old,
                    interp_cand,
                    cor,
                    std_test,
                    True,
                    np.nan,
                    "accepted",
                ]
            )
            break
        if (
            not swapped
            and on_unimprovable == ut.ON_UNIMPROVABLE_DROP
            and n - len(dropped) > 1
        ):
            dropped.add(i)
    keep_idx = [i for i in range(n) if i not in dropped]
    rows = [new_rows[i] if i in new_rows else df_feat.iloc[[i]] for i in keep_idx]
    df_feat_new = pd.concat(rows, ignore_index=True)
    return df_feat_new, records


def _consolidate_simplify_(
    df_feat=None,
    df_parts=None,
    labels=None,
    X=None,
    df_cor=None,
    dict_all_scales=None,
    dict_interp=None,
    dict_meta=None,
    max_interpret_grade=None,
    min_cor=0.7,
    ml_metric="balanced_accuracy",
    ml_th=0.0,
    ml_cv=5,
    on_unimprovable=ut.ON_UNIMPROVABLE_KEEP,
    label_test=1,
    label_ref=0,
    parametric=False,
    accept_gaps=False,
    max_std_test=0.2,
    estimator=None,
    top_k=None,
):
    """Batch-by-subcategory swaps toward the fewest interpretable subcategories.

    Interpretable subcategories are processed best-first; for each, every still-
    unclaimed target with an eligible candidate in that subcategory is swapped to
    its best in-subcat candidate, the whole batch is CV-scored, and the batch is
    accepted only if the set score stays within ``ml_th`` of the baseline. Returns
    ``(df_feat_new, X_kept, records)``.

    ``top_k`` (``candidate_search='fast'``) caps the candidates per feature
    before subcategory ranking, so the heuristic is internally consistent; ``None``
    (``'exact'``) evaluates them all (byte-identical path)."""
    n = len(df_feat)
    baseline = _score_feature_set_(
        X=X, labels=labels, ml_cv=ml_cv, ml_metric=ml_metric, estimator=estimator
    )
    targets = _select_targets_(
        df_feat=df_feat,
        dict_interp=dict_interp,
        max_interpret_grade=max_interpret_grade,
    )
    # Per target: (feat_id, interp_old, eligible candidates) + the subcategory rankings.
    interp_arr = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
    target_cands = {
        i: (
            feat_id,
            interp_old,
            _cap_candidates_(
                candidates=_eligible_candidates_(
                    feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
                    min_cor=min_cor, interp_arr=interp_arr,
                ),
                top_k=top_k,
            ),
        )
        for i, feat_id, interp_old in targets
    }
    subcat_rating = {}
    for i in target_cands:
        for scale_cand, interp_cand, abs_cor, cor in target_cands[i][2]:
            subcat_rating.setdefault(dict_meta[scale_cand][1], interp_cand)
    ranked_subcats = sorted(subcat_rating, key=lambda s: subcat_rating[s])
    new_rows = {}
    claimed = set()
    records = []
    for sub in ranked_subcats:
        batch = {}  # i -> (scale_cand, interp_cand, cor, row_df, col, std_test)
        for i in target_cands:
            if i in claimed:
                continue
            feat_id, interp_old, cands = target_cands[i]
            best = next(
                ((sc, ic, ac, c) for sc, ic, ac, c in cands if dict_meta[sc][1] == sub),
                None,
            )
            if best is None:
                continue
            scale_cand, interp_cand, abs_cor, cor = best
            positions = df_feat.iloc[i][ut.COL_POSITION]
            row_df, col = _recompute_swapped_row_(
                feat_id=feat_id,
                scale_new=scale_cand,
                df_parts=df_parts,
                dict_all_scales=dict_all_scales,
                dict_meta=dict_meta,
                labels=labels,
                label_test=label_test,
                label_ref=label_ref,
                parametric=parametric,
                accept_gaps=accept_gaps,
                positions=positions,
            )
            std_test = float(row_df[ut.COL_STD_TEST].iloc[0])
            if std_test > max_std_test:
                records.append(
                    [
                        feat_id,
                        scale_cand,
                        interp_old,
                        interp_cand,
                        cor,
                        std_test,
                        False,
                        np.nan,
                        "max_std_test",
                    ]
                )
                continue
            batch[i] = (scale_cand, interp_cand, cor, row_df, col, std_test)
        if not batch:
            continue
        X_trial = X.copy()
        for i, (scale_cand, interp_cand, cor, row_df, col, std_test) in batch.items():
            X_trial[:, i] = col
        score = _score_feature_set_(
            X=X_trial,
            labels=labels,
            ml_cv=ml_cv,
            ml_metric=ml_metric,
            estimator=estimator,
        )
        accepted = score >= baseline - ml_th
        if accepted:
            X = X_trial
            baseline = score
        for i, (scale_cand, interp_cand, cor, row_df, col, std_test) in batch.items():
            feat_id, interp_old = target_cands[i][0], target_cands[i][1]
            if accepted:
                new_rows[i] = row_df
                claimed.add(i)
            records.append(
                [
                    feat_id,
                    scale_cand,
                    interp_old,
                    interp_cand,
                    cor,
                    std_test,
                    accepted,
                    score,
                    "accepted" if accepted else "cv_drop",
                ]
            )
    # Unclaimed targets -> on_unimprovable.
    dropped = set()
    if on_unimprovable != ut.ON_UNIMPROVABLE_KEEP:
        for i in target_cands:
            if i in claimed or n - len(dropped) <= 1:
                continue
            if on_unimprovable == ut.ON_UNIMPROVABLE_DROP:
                dropped.add(i)
            elif on_unimprovable == ut.ON_UNIMPROVABLE_DROP_IF_PERF:
                keep_idx = [j for j in range(n) if j not in dropped and j != i]
                score_drop = _score_feature_set_(
                    X=X[:, keep_idx],
                    labels=labels,
                    ml_cv=ml_cv,
                    ml_metric=ml_metric,
                    estimator=estimator,
                )
                if score_drop >= baseline - ml_th:
                    dropped.add(i)
                    baseline = score_drop
    keep_idx = [i for i in range(n) if i not in dropped]
    rows = [new_rows[i] if i in new_rows else df_feat.iloc[[i]] for i in keep_idx]
    df_feat_new = pd.concat(rows, ignore_index=True)
    return df_feat_new, X[:, keep_idx], records


def simplify_cpp_(
    df_feat=None,
    df_parts=None,
    df_scales_self=None,
    labels=None,
    strategy=ut.STRATEGY_GREEDY,
    candidate_search=ut.CANDIDATE_SEARCH_EXACT,
    max_interpret_grade=None,
    min_cor=0.7,
    ml_metric="balanced_accuracy",
    ml_th=0.0,
    ml_cv=5,
    on_unimprovable=ut.ON_UNIMPROVABLE_KEEP,
    redundancy_tie_break=ut.TIE_BREAK_INTERPRETABILITY,
    label_test=1,
    label_ref=0,
    max_std_test=0.2,
    max_cor=0.5,
    max_overlap=0.5,
    check_cat=True,
    parametric=False,
    accept_gaps=False,
    return_details=False,
    ml_model=ut.MODEL_SVM,
    random_state=None,
    verbose=False,
    allow_drop=True,
):
    """Backend entry for ``CPP.simplify``: df_feat or (df_feat, df_candidates)."""
    df_feat = df_feat.reset_index(drop=True).copy()
    # allow_drop=False -> swap-only: never drop a feature (no unimprovable-drop, no
    # redundancy reduction), so the output keeps every input feature 1:1.
    if not allow_drop:
        on_unimprovable = ut.ON_UNIMPROVABLE_KEEP
    # candidate_search='fast' caps candidates evaluated per feature; 'exact'
    # (top_k=None) evaluates them all (byte-identical path). swap_all has no CV
    # gate and breaks at the first viable candidate, so it ignores the cap.
    top_k = _FAST_TOP_K if candidate_search == ut.CANDIDATE_SEARCH_FAST else None
    df_scales_pool, df_cor, dict_all_scales, dict_interp, dict_meta = (
        _load_candidate_pool_()
    )
    # Skip-and-return when nothing is rated (e.g. a pure run_num / custom df_feat).
    targets = _select_targets_(
        df_feat=df_feat,
        dict_interp=dict_interp,
        max_interpret_grade=max_interpret_grade,
    )
    if len(targets) == 0:
        # Distinguish "nothing graded" (Case 1) from "all already good enough" (Case 2).
        n_graded = sum(
            1 for f in df_feat[ut.COL_FEATURE]
            if not np.isnan(_interp(scale_id=ut.split_feat_id(feat_id=f)[2], dict_interp=dict_interp))
        )
        if n_graded == 0:
            warnings.warn(
                "'df_feat' has no AAontology-graded features to simplify (scales are "
                "ungraded / from 'run_num'); returning it unchanged.",
                RuntimeWarning,
            )
        elif verbose:
            cut = max_interpret_grade if max_interpret_grade is not None else 1
            ut.print_out(
                f"{n_graded} graded feature(s), none graded worse than {cut}; "
                f"returning 'df_feat' unchanged."
            )
        out = ut.sort_cols_feat(df_feat=df_feat)
        return (out, _build_df_candidates_(records=[])) if return_details else out
    # swap_all needs no feature matrix (no CV scoring); greedy/consolidate do.
    if strategy == ut.STRATEGY_SWAP_ALL:
        df_feat_new, records = _swap_all_simplify_(
            df_feat=df_feat,
            df_parts=df_parts,
            labels=labels,
            df_cor=df_cor,
            dict_all_scales=dict_all_scales,
            dict_interp=dict_interp,
            dict_meta=dict_meta,
            max_interpret_grade=max_interpret_grade,
            min_cor=min_cor,
            on_unimprovable=on_unimprovable,
            label_test=label_test,
            label_ref=label_ref,
            parametric=parametric,
            accept_gaps=accept_gaps,
            max_std_test=max_std_test,
        )
    else:
        X = _build_base_matrix_(
            df_feat=df_feat,
            df_parts=df_parts,
            df_scales_self=df_scales_self,
            accept_gaps=accept_gaps,
        )
        estimator = _resolve_ml_model_(ml_model=ml_model, random_state=random_state)
        common = dict(
            df_feat=df_feat,
            df_parts=df_parts,
            labels=labels,
            X=X,
            df_cor=df_cor,
            dict_all_scales=dict_all_scales,
            dict_interp=dict_interp,
            dict_meta=dict_meta,
            max_interpret_grade=max_interpret_grade,
            min_cor=min_cor,
            ml_metric=ml_metric,
            ml_th=ml_th,
            ml_cv=ml_cv,
            on_unimprovable=on_unimprovable,
            label_test=label_test,
            label_ref=label_ref,
            parametric=parametric,
            accept_gaps=accept_gaps,
            max_std_test=max_std_test,
            estimator=estimator,
            top_k=top_k,
        )
        if strategy == ut.STRATEGY_GREEDY:
            df_feat_new, _X_kept, records, _ = _greedy_simplify_(**common)
        else:  # STRATEGY_CONSOLIDATE
            df_feat_new, _X_kept, records = _consolidate_simplify_(**common)

    # Set-level redundancy reduction: drop only swapped features that became
    # redundant; original (unswapped) features are protected. Swapped features are
    # exactly the ones whose feature id is new (a swap changes the SCALE → the id).
    swapped_ids = set(df_feat_new[ut.COL_FEATURE]) - set(df_feat[ut.COL_FEATURE])
    df_cor_present = _merged_scale_corr_(
        df_feat=df_feat_new,
        df_scales_pool=df_scales_pool,
        df_scales_self=df_scales_self,
    )
    if allow_drop:
        df_feat_new = _apply_redundancy_(
            df_feat=df_feat_new,
            df_cor=df_cor_present,
            dict_interp=dict_interp,
            swapped_ids=swapped_ids,
            max_overlap=max_overlap,
            max_cor=max_cor,
            check_cat=check_cat,
            tie_break=redundancy_tie_break,
        )
    df_feat_new = ut.sort_cols_feat(df_feat=df_feat_new)
    if return_details:
        return df_feat_new, _build_df_candidates_(records=records)
    return df_feat_new
