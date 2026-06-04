"""This is a script to test branches of CPP's fast feature-matrix builder
(``_backend/cpp/_filters/_get_feature_matrix_fast.py``) that the DOM_GSEC parity
fixtures don't reach: the Phase-C per-sample loop (variable part lengths),
non-contiguous Pattern positions, the accept_gaps paths, the AALookupCache reuse,
the gap-symbol guard, and the n_jobs>1 parallel path.

The bundled DOM_GSEC TMDs are all length 23 (uniform), so the vectorized uniform
path is always taken and Phase C stays dark. Here we build df_parts directly with
VARIABLE-length sequences to force the per-sample fallback, and assert a
hand-computed golden mean for the simple uniform case.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._filters._get_feature_matrix_fast import (
    get_feature_matrix_fast_,
    AALookupCache,
    build_scale_lookup,
    clear_scale_lookup_cache,
    _positions_are_contiguous,
    _hoisted_gap_check,
)

aa.options["verbose"] = False

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# I Helper Functions
def _df_scales():
    """20-AA scales: S1[aa]=rank(1..20), S2[aa]=rank*0.1 — deterministic golden."""
    s1 = {aa: float(i + 1) for i, aa in enumerate(AA_ORDER)}
    s2 = {aa: round((i + 1) * 0.1, 4) for i, aa in enumerate(AA_ORDER)}
    return pd.DataFrame({"S1": s1, "S2": s2})


def _df_parts(tmds):
    return pd.DataFrame({"tmd": list(tmds)})


# II Test Classes
class TestGetFeatureMatrixFastGoldenValues:
    """Hand-computed expected values, not just 'runs'."""

    def test_segment_whole_mean(self):
        # part 'AC' -> Segment(1,1) is the whole sequence -> mean(S1[A], S1[C]).
        # S1[A]=1, S1[C]=2 -> 1.5 exactly.
        X = get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                     df_parts=_df_parts(["AC"]),
                                     df_scales=_df_scales())
        assert X.shape == (1, 1)
        assert X[0, 0] == 1.5

    def test_segment_second_half(self):
        # 'ACDE' Segment(2,2) -> second half = 'DE' -> mean(S1[D]=3,S1[E]=4)=3.5.
        X = get_feature_matrix_fast_(features=["TMD-Segment(2,2)-S1"],
                                     df_parts=_df_parts(["ACDE"]),
                                     df_scales=_df_scales())
        assert X[0, 0] == 3.5

    def test_pattern_n_positions(self):
        # Pattern(N,1,3) -> positions 0,2 -> 'A','D' -> mean(1,3)=2.0.
        X = get_feature_matrix_fast_(features=["TMD-Pattern(N,1,3)-S1"],
                                     df_parts=_df_parts(["ACDE"]),
                                     df_scales=_df_scales())
        assert X[0, 0] == 2.0

    def test_two_features_two_scales(self):
        X = get_feature_matrix_fast_(
            features=["TMD-Segment(1,1)-S1", "TMD-Segment(1,1)-S2"],
            df_parts=_df_parts(["AC"]), df_scales=_df_scales())
        assert X.shape == (1, 2)
        assert X[0, 0] == 1.5
        assert X[0, 1] == round((0.1 + 0.2) / 2, 5)


class TestPhaseCVariableLength:
    """Variable-length df_parts force the per-sample Phase-C loop."""

    def _parts(self):
        # lengths 9, 13, 5 -> all_same_L False -> Phase C for Segment.
        return _df_parts(["ACDEFGHIK", "ACDEFGHIKLMNP", "ACDEF"])

    def test_segment_phase_c(self):
        X = get_feature_matrix_fast_(features=["TMD-Segment(1,2)-S1"],
                                     df_parts=self._parts(), df_scales=_df_scales())
        assert X.shape == (3, 1)
        assert np.all(np.isfinite(X))

    def test_pattern_c_phase_c(self):
        # Pattern terminus C is never 'uniform' -> Phase C Pattern-C branch.
        X = get_feature_matrix_fast_(features=["TMD-Pattern(C,1,2)-S1"],
                                     df_parts=self._parts(), df_scales=_df_scales())
        assert X.shape == (3, 1)

    def test_pattern_n_noncontiguous_phase_c(self):
        # Pattern(N,1,4) -> positions 0,3 non-contiguous -> uniform-but-not-contiguous
        # -> falls through to Phase C Pattern-N branch.
        X = get_feature_matrix_fast_(features=["TMD-Pattern(N,1,4)-S1"],
                                     df_parts=self._parts(), df_scales=_df_scales())
        assert X.shape == (3, 1)

    def test_periodicpattern_phase_c(self):
        X = get_feature_matrix_fast_(
            features=["TMD-PeriodicPattern(N,i+3/3,1)-S1"],
            df_parts=self._parts(), df_scales=_df_scales())
        assert X.shape == (3, 1)

    def test_segment_phase_c_accept_gaps(self):
        # accept_gaps -> nanmean branches in Phase C.
        parts = _df_parts(["ACDEFGHIK", "ACDEF"])
        X = get_feature_matrix_fast_(features=["TMD-Segment(1,2)-S1"],
                                     df_parts=parts, df_scales=_df_scales(),
                                     accept_gaps=True)
        assert X.shape == (2, 1)

    def test_pattern_n_phase_c_accept_gaps(self):
        # non-contiguous Pattern-N + variable length + accept_gaps -> nanmean.
        X = get_feature_matrix_fast_(features=["TMD-Pattern(N,1,4)-S1"],
                                     df_parts=self._parts(), df_scales=_df_scales(),
                                     accept_gaps=True)
        assert X.shape == (3, 1)

    def test_pattern_c_phase_c_accept_gaps(self):
        X = get_feature_matrix_fast_(features=["TMD-Pattern(C,1,2)-S1"],
                                     df_parts=self._parts(), df_scales=_df_scales(),
                                     accept_gaps=True)
        assert X.shape == (3, 1)

    def test_periodicpattern_phase_c_accept_gaps(self):
        X = get_feature_matrix_fast_(
            features=["TMD-PeriodicPattern(N,i+3/3,1)-S1"],
            df_parts=self._parts(), df_scales=_df_scales(), accept_gaps=True)
        assert X.shape == (3, 1)


class TestUniformAcceptGaps:
    """Uniform contiguous path with accept_gaps -> nanmean + post_check."""

    def test_uniform_accept_gaps_nanmean(self):
        # all same length (uniform) + contiguous Segment + accept_gaps -> nanmean view path.
        parts = _df_parts(["ACDE", "FGHI", "KLMN"])
        X = get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                     df_parts=parts, df_scales=_df_scales(),
                                     accept_gaps=True)
        assert X.shape == (3, 1)
        assert np.all(np.isfinite(X))


class TestGapGuard:
    """_hoisted_gap_check raises on gap symbol when accept_gaps=False."""

    def test_gap_raises(self):
        parts = _df_parts(["AC-EF"])  # contains '-'
        with pytest.raises(ValueError, match="gaps"):
            get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                     df_parts=parts, df_scales=_df_scales())

    def test_gap_check_helper_direct(self):
        with pytest.raises(ValueError, match="gaps"):
            _hoisted_gap_check(df_parts=_df_parts(["AC-EF"]))

    def test_gap_check_helper_clean(self):
        # no gaps -> returns None silently
        assert _hoisted_gap_check(df_parts=_df_parts(["ACEF"])) is None


class TestPositionsContiguous:
    """_positions_are_contiguous helper, incl. the empty-positions guard."""

    def test_contiguous_true(self):
        assert _positions_are_contiguous(np.array([2, 3, 4])) is True

    def test_contiguous_false(self):
        assert _positions_are_contiguous(np.array([2, 4, 6])) is False

    def test_empty_positions_false(self):
        assert _positions_are_contiguous(np.array([], dtype=np.int64)) is False


class TestAALookupCacheReuse:
    """AALookupCache short-circuits the per-call rebuild (Phase A.4)."""

    def test_cache_matches_and_used(self):
        parts = _df_parts(["ACDE", "FGHI"])
        scales = _df_scales()
        cache = AALookupCache.from_df(parts, scales)
        assert cache.matches(parts, scales)
        X_cached = get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                            df_parts=parts, df_scales=scales,
                                            aa_lookup_cache=cache)
        X_plain = get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                           df_parts=parts, df_scales=scales)
        assert np.allclose(X_cached, X_plain, equal_nan=True)

    def test_cache_mismatch_falls_back(self):
        parts = _df_parts(["ACDE"])
        scales = _df_scales()
        cache = AALookupCache.from_df(parts, scales)
        other = _df_parts(["FGHI"])
        assert not cache.matches(other, scales)
        # passing a non-matching cache -> rebuild path, still correct
        X = get_feature_matrix_fast_(features=["TMD-Segment(1,1)-S1"],
                                     df_parts=other, df_scales=scales,
                                     aa_lookup_cache=cache)
        assert X.shape == (1, 1)


class TestScaleLookupCache:
    """build_scale_lookup is content-cached; clear evicts it."""

    def test_same_content_same_object(self):
        clear_scale_lookup_cache()
        a = build_scale_lookup(df_scales=_df_scales())
        b = build_scale_lookup(df_scales=_df_scales())  # distinct df, same content
        # scale_matrix (index 2) is the cached object -> identical reference
        assert a[2] is b[2]

    def test_clear_forces_rebuild(self):
        a = build_scale_lookup(df_scales=_df_scales())
        clear_scale_lookup_cache()
        b = build_scale_lookup(df_scales=_df_scales())
        assert a[2] is not b[2]


class TestParallelPath:
    """n_jobs>1 with >1 feature exercises the joblib chunk path."""

    def test_parallel_matches_serial(self):
        from joblib import parallel_backend
        parts = _df_parts(["ACDEFGHIK", "ACDEFGHIKLMNP", "ACDEF"])
        scales = _df_scales()
        feats = ["TMD-Segment(1,2)-S1", "TMD-Segment(2,2)-S1",
                 "TMD-Pattern(C,1,2)-S2", "TMD-Segment(1,1)-S2"]
        # threading backend runs workers in-process (deterministic + coverable).
        with parallel_backend("threading"):
            X_par = get_feature_matrix_fast_(features=feats, df_parts=parts,
                                             df_scales=scales, n_jobs=2)
        X_ser = get_feature_matrix_fast_(features=feats, df_parts=parts,
                                         df_scales=scales, n_jobs=1)
        assert X_par.shape == (3, 4)
        assert np.allclose(X_par, X_ser, equal_nan=True)
