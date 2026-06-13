"""Equivalence test for the issue #186 ``_eligible_candidates_`` vectorization.

The per-feature ``cors.items()`` scan over ~586 scales (a Python ``_interp`` +
``float`` per element) was replaced by a numpy filter, then the original Python
``list.sort`` ranks the small survivor list (kept so the NaN-correlation tie
order is byte-identical), with an OPTIONAL precomputed ``interp_arr`` (the cross-feature hoist that
reuses the per-scale interpretability array across every targeted feature in one
simplify call). The returned candidate list — tuple values, Python-``float``
dtype, AND tie order — must be byte-identical to the original pre-optimization
implementation. This pins exactness against a reference impl that reproduces the
original code, across randomized df_cor / dict_interp / min_cor configs plus
deliberate tie / anti-correlation / NaN-corr / NaN-interp / scale-absent cases,
and on the real bundled AAontology pool.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._simplify import (
    _eligible_candidates_,
    _interp,
    _interp_array_,
    _load_candidate_pool_,
)

aa.options["verbose"] = False


# --------------------------------------------------------------------------
# Reference (original, pre-optimization) implementation
# --------------------------------------------------------------------------
def _ref_eligible_candidates(feat_id=None, df_cor=None, dict_interp=None, min_cor=0.7):
    """Verbatim original ``_eligible_candidates_`` (the per-element python loop)."""
    scale_old = ut.split_feat_id(feat_id=feat_id)[2]
    interp_old = _interp(scale_id=scale_old, dict_interp=dict_interp)
    if np.isnan(interp_old) or scale_old not in df_cor.columns:
        return []
    candidates = []
    cors = df_cor[scale_old]
    for scale_cand, cor in cors.items():
        if scale_cand == scale_old:
            continue
        interp_cand = _interp(scale_id=scale_cand, dict_interp=dict_interp)
        if np.isnan(interp_cand) or interp_cand >= interp_old:
            continue
        abs_cor = abs(float(cor))
        if abs_cor < min_cor:
            continue
        candidates.append((scale_cand, interp_cand, abs_cor, float(cor)))
    candidates.sort(key=lambda t: (t[1], -t[2]))
    return candidates


def _assert_exact_equal(ref, got):
    """Byte-identical: same length, same order, same values, Python-float dtype."""
    assert isinstance(ref, list) and isinstance(got, list)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert len(a) == 4 and len(b) == 4
        assert a[0] == b[0]
        for k in (1, 2, 3):
            assert type(b[k]) is float  # not np.float64
            if isinstance(a[k], float) and np.isnan(a[k]):
                assert np.isnan(b[k])
            else:
                assert a[k] == b[k]
        assert repr(a) == repr(b)


def _make_df_cor(rng, scales):
    n = len(scales)
    M = rng.uniform(-1, 1, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    return pd.DataFrame(M, index=scales, columns=scales)


# --------------------------------------------------------------------------
# 1. Randomized sweep (incl. interp ties via a small grade range)
# --------------------------------------------------------------------------
class TestEligibleCandidatesEquiv:
    @pytest.mark.parametrize("trial", list(range(60)))
    def test_sweep_identical(self, trial):
        rng = np.random.default_rng(trial)
        n = int(rng.integers(3, 60))
        scales = [f"SC{i}" for i in range(n)]
        df_cor = _make_df_cor(rng, scales)
        dict_interp = {}
        grades = rng.integers(1, 5, size=n)  # small range -> many interp ties
        for sid, g in zip(scales, grades):
            r = rng.random()
            if r < 0.15:
                continue  # absent
            if r < 0.25:
                dict_interp[sid] = np.nan
            elif r < 0.30:
                dict_interp[sid] = None
            else:
                dict_interp[sid] = float(g)
        min_cor = float(rng.choice([0.0, 0.3, 0.5, 0.7, 0.9]))
        scale_old = str(rng.choice(scales))
        feat_id = f"TMD-Segment(1,5)-{scale_old}"
        ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, min_cor)
        got = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp, min_cor=min_cor
        )
        _assert_exact_equal(ref, got)
        # precomputed interp_arr (cross-feature hoist) path must match too
        ia = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
        got_h = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
            min_cor=min_cor, interp_arr=ia,
        )
        _assert_exact_equal(ref, got_h)

    def test_full_tie_order_is_stable(self):
        # equal interp AND equal |cor| -> original df_cor column order must win.
        scales = [f"SC{i}" for i in range(8)]
        M = np.full((8, 8), 0.8)
        np.fill_diagonal(M, 1.0)
        df_cor = pd.DataFrame(M, index=scales, columns=scales)
        dict_interp = {s: 2.0 for s in scales}
        dict_interp["SC0"] = 5.0  # feature scale worst -> all others eligible & tied
        feat_id = "TMD-Segment(1,5)-SC0"
        ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, 0.5)
        got = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp, min_cor=0.5
        )
        _assert_exact_equal(ref, got)
        assert [t[0] for t in got] == [f"SC{i}" for i in range(1, 8)]

    def test_anti_correlation_kept_via_abs(self):
        df_cor = pd.DataFrame(
            [[1.0, -0.9, 0.6], [-0.9, 1.0, -0.95], [0.6, -0.95, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        dict_interp = {"A": 5.0, "B": 1.0, "C": 2.0}
        feat_id = "TMD-Segment(1,5)-A"
        ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, 0.7)
        got = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp, min_cor=0.7
        )
        _assert_exact_equal(ref, got)
        assert any(t[3] < 0 for t in got)

    def test_nan_correlation_cell_is_kept(self):
        # abs(float(nan)) < min_cor is False -> NaN-cor candidate is NOT skipped.
        df_cor = pd.DataFrame(
            [[1.0, np.nan, 0.8], [np.nan, 1.0, 0.3], [0.8, 0.3, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        dict_interp = {"A": 5.0, "B": 1.0, "C": 2.0}
        feat_id = "TMD-Segment(1,5)-A"
        ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, 0.7)
        got = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp, min_cor=0.7
        )
        _assert_exact_equal(ref, got)
        assert any(t[0] == "B" for t in got)  # NaN-cor candidate kept

    def test_multiple_nan_corr_ties_keep_index_order(self):
        # Regression for the lexsort tie bug: several NaN-correlation candidates
        # (B, C, D) tied in interp with real-correlation candidates (E, F). The
        # original Python list.sort keeps them in df_cor index order
        # (B, C, D, E, F); np.lexsort would push the NaN secondary keys to the
        # tail of the interp group (E, F, B, C, D). The output must match the
        # original byte-for-byte.
        scales = ["A", "B", "C", "D", "E", "F"]
        n = len(scales)
        M = np.full((n, n), 0.5)
        np.fill_diagonal(M, 1.0)
        a_col = [1.0, np.nan, np.nan, np.nan, 0.8, 0.8]
        for i, v in enumerate(a_col):
            M[0, i] = M[i, 0] = v
        df_cor = pd.DataFrame(M, index=scales, columns=scales)
        dict_interp = {s: 2.0 for s in scales}
        dict_interp["A"] = 5.0  # feature scale worst -> B..F all eligible & tied
        feat_id = "TMD-Segment(1,5)-A"
        ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, 0.7)
        got = _eligible_candidates_(
            feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp, min_cor=0.7
        )
        assert sum(np.isnan(t[2]) for t in ref) >= 2  # the case lexsort breaks
        _assert_exact_equal(ref, got)

    def test_feature_scale_unrated_returns_empty(self):
        df_cor = _make_df_cor(np.random.default_rng(0), ["A", "B", "C"])
        for di in ({"A": np.nan, "B": 1.0}, {"B": 1.0}):  # NaN and absent
            ref = _ref_eligible_candidates("TMD-Segment(1,5)-A", df_cor, di, 0.5)
            got = _eligible_candidates_(
                feat_id="TMD-Segment(1,5)-A", df_cor=df_cor, dict_interp=di, min_cor=0.5
            )
            assert ref == [] and got == []

    def test_scale_old_absent_from_df_cor_returns_empty(self):
        df_cor = _make_df_cor(np.random.default_rng(1), ["A", "B", "C"])
        di = {"ZZ": 5.0, "A": 1.0, "B": 2.0}
        ref = _ref_eligible_candidates("TMD-Segment(1,5)-ZZ", df_cor, di, 0.5)
        got = _eligible_candidates_(
            feat_id="TMD-Segment(1,5)-ZZ", df_cor=df_cor, dict_interp=di, min_cor=0.5
        )
        assert ref == [] and got == []


# --------------------------------------------------------------------------
# 2. Real bundled AAontology pool (df_cor + dict_interp from the package data)
# --------------------------------------------------------------------------
class TestEligibleCandidatesRealPool:
    def test_identical_on_real_pool(self):
        _df_scales_pool, df_cor, _das, dict_interp, _dm = _load_candidate_pool_()
        scales = list(df_cor.columns)
        ia = _interp_array_(df_cor=df_cor, dict_interp=dict_interp)
        rng = np.random.default_rng(0)
        # sample a spread of real scales (rated and possibly unrated) across min_cor.
        sampled = list(rng.choice(scales, size=min(40, len(scales)), replace=False))
        n_nonempty = 0
        for scale_old in sampled:
            feat_id = f"TMD-Segment(1,5)-{scale_old}"
            for min_cor in (0.3, 0.5, 0.7, 0.9):
                ref = _ref_eligible_candidates(feat_id, df_cor, dict_interp, min_cor)
                got = _eligible_candidates_(
                    feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
                    min_cor=min_cor,
                )
                _assert_exact_equal(ref, got)
                got_h = _eligible_candidates_(
                    feat_id=feat_id, df_cor=df_cor, dict_interp=dict_interp,
                    min_cor=min_cor, interp_arr=ia,
                )
                _assert_exact_equal(ref, got_h)
                n_nonempty += len(got) > 0
        assert n_nonempty > 0  # the real pool produced non-trivial candidate lists
