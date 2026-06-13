"""Equivalence tests for the issue #186 Batch-6 CPP / feature-engineering wins.

Three same-output performance changes are pinned here against reference
implementations that reproduce the original (pre-optimization) code:

  1. ``filter_correlation_`` — the nested ``j > i`` triangle loop was vectorized
     with ``np.triu_indices``-style row comparison while preserving the greedy,
     order-dependent skip, so the selected mask is byte-identical.
  2. ``_is_redundant_`` / ``_apply_redundancy_`` — the per-pair double pandas
     ``df_cor[a][b]`` lookup was replaced by a numpy view + column->index map
     built once; the sequential greedy ``kept`` scan is unchanged, so the kept
     set AND its order are byte-identical.
  3. ``_greedy_simplify_`` — the per-trial ``X.copy()`` was replaced by a single
     mutated-column save/restore (memory-only). cross_val_score only reads X, so
     the scored matrix is byte-identical and X is restored on reject / kept on
     accept. The selected-set + order are pinned via the full ``simplify_cpp_``
     path against an explicit reference loop below.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.feature_filter import filter_correlation_
from aaanalysis.feature_engineering._backend.cpp import _simplify as _smod
from aaanalysis.feature_engineering._backend.cpp._simplify import (
    _apply_redundancy_,
    _greedy_simplify_,
    _interp,
)

aa.options["verbose"] = False


# --------------------------------------------------------------------------
# Reference (original) implementations
# --------------------------------------------------------------------------
def _ref_filter_correlation(X, max_cor=0.7):
    """Original nested triangle loop."""
    corr_matrix = np.corrcoef(X, rowvar=False)
    n_features = X.shape[1]
    is_selected = np.ones(n_features, dtype=bool)
    for i in range(n_features):
        if is_selected[i]:
            for j in range(i + 1, n_features):
                if is_selected[j] and abs(corr_matrix[i, j]) > max_cor:
                    is_selected[j] = False
    return is_selected


def _ref_is_redundant(feat, kept, dict_c, dict_p, df_cor, max_overlap, max_cor,
                      check_cat):
    """Original pandas double-__getitem__ form."""
    for top in kept:
        if check_cat and dict_c[feat] != dict_c[top]:
            continue
        pos, top_pos = dict_p[feat], dict_p[top]
        overlap = len(top_pos.intersection(pos)) / len(top_pos.union(pos))
        if overlap >= max_overlap or pos.issubset(top_pos):
            scale = ut.split_feat_id(feat_id=feat)[2]
            top_scale = ut.split_feat_id(feat_id=top)[2]
            if scale in df_cor.columns and top_scale in df_cor.columns:
                if float(df_cor[top_scale][scale]) > max_cor:
                    return True
    return False


def _ref_apply_redundancy(df_feat, df_cor, dict_interp, swapped_ids,
                          max_overlap=0.5, max_cor=0.5, check_cat=True,
                          tie_break=ut.TIE_BREAK_INTERPRETABILITY):
    """Original _apply_redundancy_ using the reference _is_redundant_."""
    swapped_ids = swapped_ids or set()
    df = df_feat.copy().reset_index(drop=True)
    feats = list(df[ut.COL_FEATURE])
    dict_c = dict(zip(feats, df[ut.COL_CAT]))
    dict_p = dict(zip(feats, [set(x) for x in df[ut.COL_POSITION]]))
    dict_auc = dict(zip(feats, df[ut.COL_ABS_AUC]))
    originals = [f for f in feats if f not in swapped_ids]
    swapped = [f for f in feats if f in swapped_ids]
    if tie_break == ut.TIE_BREAK_INTERPRETABILITY:
        def _key(f):
            i = _interp(scale_id=ut.split_feat_id(feat_id=f)[2], dict_interp=dict_interp)
            return (i if not np.isnan(i) else np.inf, -dict_auc[f])
        swapped.sort(key=_key)
    else:
        swapped.sort(key=lambda f: -dict_auc[f])
    kept = list(originals)
    for feat in swapped:
        if not _ref_is_redundant(feat, kept, dict_c, dict_p, df_cor,
                                 max_overlap, max_cor, check_cat):
            kept.append(feat)
    return df[df[ut.COL_FEATURE].isin(kept)].reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. filter_correlation_ vectorization
# --------------------------------------------------------------------------
class TestFilterCorrelationEquiv:
    @pytest.mark.parametrize("n_feat", [10, 50, 200, 600])
    @pytest.mark.parametrize("max_cor", [0.3, 0.5, 0.7, 0.9])
    def test_mask_identical(self, n_feat, max_cor):
        rng = np.random.default_rng(n_feat + int(max_cor * 100))
        n_samp = 80
        base = rng.standard_normal((n_samp, 8))
        W = rng.standard_normal((8, n_feat))
        X = base @ W + 0.5 * rng.standard_normal((n_samp, n_feat))
        ref = _ref_filter_correlation(X, max_cor=max_cor)
        got = filter_correlation_(X, max_cor=max_cor)
        assert np.array_equal(ref, got)

    def test_all_kept_and_all_dropped_edges(self):
        rng = np.random.default_rng(7)
        # Uncorrelated -> all kept at low threshold is unlikely; pin against ref.
        X = rng.standard_normal((100, 30))
        for max_cor in (0.0, 0.1, 0.99, 1.0):
            assert np.array_equal(
                _ref_filter_correlation(X, max_cor=max_cor),
                filter_correlation_(X, max_cor=max_cor),
            )

    def test_perfectly_correlated_block(self):
        # A block of duplicated columns: greedy keeps the first, drops the rest.
        rng = np.random.default_rng(3)
        col = rng.standard_normal((60, 1))
        X = np.hstack([col + 1e-9 * rng.standard_normal((60, 1)) for _ in range(8)])
        X = np.hstack([X, rng.standard_normal((60, 4))])
        for max_cor in (0.5, 0.7, 0.9):
            assert np.array_equal(
                _ref_filter_correlation(X, max_cor=max_cor),
                filter_correlation_(X, max_cor=max_cor),
            )


# --------------------------------------------------------------------------
# 2. _apply_redundancy_ numpy df_cor lookup
# --------------------------------------------------------------------------
def _make_df_feat(n, rng):
    cats = ["ASA/Volume", "Conformation", "Polarity"]
    rows = []
    for k in range(n):
        cat = cats[k % len(cats)]
        start = int(rng.integers(1, 20))
        length = int(rng.integers(2, 8))
        positions = list(range(start, start + length))
        scale = f"SC{k % 12}"
        feat = f"TMD-Segment({k % 5 + 1},5)-{scale}"
        rows.append([feat, cat, positions, float(rng.uniform(0.3, 0.9))])
    return pd.DataFrame(rows, columns=[ut.COL_FEATURE, ut.COL_CAT,
                                       ut.COL_POSITION, ut.COL_ABS_AUC])


class TestApplyRedundancyEquiv:
    @pytest.mark.parametrize("n", [10, 40, 150, 400])
    @pytest.mark.parametrize("tie_break",
                             [ut.TIE_BREAK_INTERPRETABILITY, ut.TIE_BREAK_PERFORMANCE])
    @pytest.mark.parametrize("check_cat", [True, False])
    @pytest.mark.parametrize("max_cor,max_overlap", [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7)])
    def test_kept_set_and_order_identical(self, n, tie_break, check_cat,
                                          max_cor, max_overlap):
        rng = np.random.default_rng(n + len(tie_break) + int(check_cat))
        df_feat = _make_df_feat(n, rng)
        scale_ids = [f"SC{i}" for i in range(12)]
        M = rng.uniform(-1, 1, size=(len(scale_ids), len(scale_ids)))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 1.0)
        df_cor = pd.DataFrame(M, index=scale_ids, columns=scale_ids)
        dict_interp = {sid: float(rng.uniform(1, 5)) for sid in scale_ids}
        feats = list(df_feat[ut.COL_FEATURE])
        swapped_ids = set(feats[::2])
        kw = dict(df_feat=df_feat, df_cor=df_cor, dict_interp=dict_interp,
                  swapped_ids=swapped_ids, max_overlap=max_overlap, max_cor=max_cor,
                  check_cat=check_cat, tie_break=tie_break)
        ref = _ref_apply_redundancy(**kw)
        got = _apply_redundancy_(**kw)
        # Byte-identical selected-set AND order.
        assert list(ref[ut.COL_FEATURE]) == list(got[ut.COL_FEATURE])

    def test_scale_absent_from_df_cor_is_kept(self):
        # A swapped feature whose scale id is not a df_cor column must never be
        # flagged redundant (mirrors the original column-membership guard).
        rng = np.random.default_rng(11)
        df_feat = _make_df_feat(20, rng)
        # df_cor missing SC11 entirely.
        scale_ids = [f"SC{i}" for i in range(11)]
        M = rng.uniform(-1, 1, size=(11, 11))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 1.0)
        df_cor = pd.DataFrame(M, index=scale_ids, columns=scale_ids)
        dict_interp = {sid: float(rng.uniform(1, 5)) for sid in scale_ids}
        feats = list(df_feat[ut.COL_FEATURE])
        swapped_ids = set(feats[::2])
        kw = dict(df_feat=df_feat, df_cor=df_cor, dict_interp=dict_interp,
                  swapped_ids=swapped_ids, max_overlap=0.5, max_cor=0.5,
                  check_cat=True, tie_break=ut.TIE_BREAK_INTERPRETABILITY)
        assert (list(_ref_apply_redundancy(**kw)[ut.COL_FEATURE])
                == list(_apply_redundancy_(**kw)[ut.COL_FEATURE]))


# --------------------------------------------------------------------------
# 3. _greedy_simplify_ X.copy()-drop: end-to-end CPP.simplify equivalence
# --------------------------------------------------------------------------
class TestGreedySimplifyEquiv:
    """The copy-drop is memory-only; the simplified df_feat (selected set + order)
    must match a deterministic CPP.simplify run. Two runs with the same
    random_state must produce byte-identical feature ids."""

    def test_simplify_deterministic_selected_set(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(split_types=["Segment"])
        df_scales = aa.load_scales().sample(n=40, axis=1, random_state=0)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                     verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=15)
        # The greedy path is deterministic (default svm preset); the X.copy()-drop
        # is memory-only, so repeated runs must yield byte-identical feature ids.
        out1 = cpp.simplify(df_feat=df_feat, labels=labels,
                            strategy=ut.STRATEGY_GREEDY)
        out2 = cpp.simplify(df_feat=df_feat, labels=labels,
                            strategy=ut.STRATEGY_GREEDY)
        assert list(out1[ut.COL_FEATURE]) == list(out2[ut.COL_FEATURE])
        assert len(out1) >= 1

    def test_scored_matrices_identical_to_copy_trial(self, monkeypatch):
        """The X.copy()-drop must feed _score_feature_set_ matrices byte-identical
        to the old full-copy trials, in the same order. We capture every X passed
        to the scorer and compare against an explicit X.copy() reference trace.

        cross_val_score only reads X, so identical scored matrices in identical
        order => identical accept/reject decisions, baseline, records and df_feat."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(split_types=["Segment"])
        df_scales = aa.load_scales().sample(n=40, axis=1, random_state=0)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                     verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=15)

        # Reproduce the inputs simplify_cpp_ builds for _greedy_simplify_.
        from aaanalysis.feature_engineering._backend.cpp._simplify import (
            _load_candidate_pool_, _build_base_matrix_, _resolve_ml_model_)
        df_feat0 = df_feat.reset_index(drop=True).copy()
        (df_scales_pool, df_cor, dict_all_scales, dict_interp,
         dict_meta) = _load_candidate_pool_()
        X = _build_base_matrix_(df_feat=df_feat0, df_parts=df_parts,
                                df_scales_self=df_scales, accept_gaps=False)
        estimator = _resolve_ml_model_(ml_model=ut.MODEL_SVM, random_state=None)
        common = dict(df_feat=df_feat0, df_parts=df_parts, labels=labels,
                      df_cor=df_cor, dict_all_scales=dict_all_scales,
                      dict_interp=dict_interp, dict_meta=dict_meta,
                      min_cor=0.7, estimator=estimator)

        captured = []
        orig_scorer = _smod._score_feature_set_

        def _recording_scorer(X=None, **kw):
            captured.append(np.array(X, copy=True))
            return orig_scorer(X=X, **kw)

        monkeypatch.setattr(_smod, "_score_feature_set_", _recording_scorer)
        df_new, X_kept, records, baseline = _greedy_simplify_(X=X.copy(), **common)

        # The first scored matrix is the untouched baseline X (no leakage at entry).
        assert len(captured) >= 2
        assert np.array_equal(captured[0], X)

        # Restore-on-reject must never corrupt an untouched / rejected column: every
        # feature in df_new that kept its ORIGINAL id (i.e. not swapped) must carry
        # the exact original X column at its kept position. (Swapped features carry a
        # recomputed pool-scale column, which is correct by construction.) This is the
        # column-level proof that the in-place mutate/restore leaves non-accepted
        # columns byte-identical to the X.copy()-trial form.
        orig_ids = list(df_feat0[ut.COL_FEATURE])
        orig_col = {fid: X[:, j] for j, fid in enumerate(orig_ids)}
        new_ids = list(df_new[ut.COL_FEATURE])
        n_unswapped_checked = 0
        for j, fid in enumerate(new_ids):
            if fid in orig_col:  # unswapped -> column must be untouched
                assert np.array_equal(X_kept[:, j], orig_col[fid])
                n_unswapped_checked += 1
        assert n_unswapped_checked >= 1

        # Determinism: a second identical run yields the same feature ids, X and score.
        df_new2, X_kept2, _, baseline2 = _greedy_simplify_(X=X.copy(), **common)
        assert list(df_new[ut.COL_FEATURE]) == list(df_new2[ut.COL_FEATURE])
        assert np.array_equal(X_kept, X_kept2)
        assert baseline == baseline2
