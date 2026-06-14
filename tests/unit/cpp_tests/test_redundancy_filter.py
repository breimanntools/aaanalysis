"""This is a script to test CPP's redundancy-reduction backend
(``_redundancy_filter.filtering`` / ``filtering_info_``), with a focus on the
``check_cat`` scale-category gate audited in issue #77.

The ``filtering`` backend is exercised directly on tiny synthetic frames so the
category, position-overlap, and scale-correlation branches can each be driven in
isolation (the surviving feature set is then hand-derivable and
environment-independent). A final class pins the public ``CPP.run`` path so the
valid-input output stays stable for ``check_cat=True``.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from pandas.testing import assert_frame_equal

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._filters._redundancy_filter import (
    filtering,
    filtering_info_,
)

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions
# Two perfectly correlated scales (SCA, SCB; corr=+1.0), one anti-correlated
# (SCC; corr=-1.0 with SCA) so the ``cor > max_cor`` arm can be driven both ways.
DF_SCALES = pd.DataFrame(
    {"SCA": [1, 2, 3, 4, 5], "SCB": [1, 2, 3, 4, 5], "SCC": [5, 4, 3, 2, 1],
     "SCD": [1, 2, 3, 4, 5]},
    index=list("ACDEF"),
)


def _feat(scale="SCA"):
    return f"TMD-Segment(1,1)-{scale}"


def _df_feat(rows):
    """Build a minimal filtering input frame from ``(feature, category,
    positions, abs_auc)`` rows; ``abs_mean_dif`` mirrors ``abs_auc``."""
    return pd.DataFrame(
        [(f, c, list(p), auc, auc) for (f, c, p, auc) in rows],
        columns=[ut.COL_FEATURE, ut.COL_CAT, ut.COL_POSITION,
                 ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF],
    )


def _survivors(df_filtered):
    return list(df_filtered[ut.COL_FEATURE])


class TestFilteringInfo:
    """``filtering_info_`` builds the category / position / correlation lookups."""

    def test_check_cat_true_builds_category_map(self):
        # check_cat=True arm: dict_c maps every feature id to its category.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2], 0.9),
                       (_feat("SCB"), "ASA", [1, 2], 0.8)])
        dict_c, dict_p, df_cor = filtering_info_(df=df, df_scales=DF_SCALES, check_cat=True)
        assert dict_c == {_feat("SCA"): "ASA", _feat("SCB"): "ASA"}

    def test_check_cat_false_leaves_category_map_empty(self):
        # check_cat=False arm: dict_c is empty (never indexed by the loop).
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2], 0.9),
                       (_feat("SCB"), "ASA", [1, 2], 0.8)])
        dict_c, dict_p, df_cor = filtering_info_(df=df, df_scales=DF_SCALES, check_cat=False)
        assert dict_c == {}

    def test_positions_become_sets(self):
        # dict_p maps each feature id to a set of its positions.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 2, 3], 0.9)])
        _, dict_p, _ = filtering_info_(df=df, df_scales=DF_SCALES, check_cat=True)
        assert dict_p[_feat("SCA")] == {1, 2, 3}

    def test_correlation_matrix_is_scale_corr(self):
        # df_cor is df_scales.corr(): SCA/SCB perfectly correlated, SCA/SCC anti.
        df = _df_feat([(_feat("SCA"), "ASA", [1], 0.9)])
        _, _, df_cor = filtering_info_(df=df, df_scales=DF_SCALES, check_cat=True)
        assert df_cor["SCA"]["SCB"] == pytest.approx(1.0)
        assert df_cor["SCA"]["SCC"] == pytest.approx(-1.0)


class TestFiltering:
    """Per-branch behaviour of the greedy ``filtering`` loop, one edge per test."""

    def test_single_candidate_feature_kept(self):
        # Edge (a): a one-row frame -> list_feat empties on pop(0), loop body
        # never runs, the single feature is returned (no IndexError).
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9)])
        for check_cat in (True, False):
            out = filtering(df=df, df_scales=DF_SCALES, check_cat=check_cat)
            assert _survivors(out) == [_feat("SCA")]

    def test_same_category_redundant_feature_dropped(self):
        # Edge (b): same category + full overlap + correlated scale -> the
        # lower-AUC feature is dropped (second arm of the gate is True).
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCB"), "ASA", [1, 2, 3], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert _survivors(out) == [_feat("SCA")]

    def test_distinct_categories_keep_both_when_check_cat(self):
        # Edge (c): identical overlap+correlation but DIFFERENT categories ->
        # the gate never fires, so both features are kept (gate's second arm False).
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCB"), "TMD", [1, 2, 3], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCB")}

    def test_check_cat_false_vs_true_differ_on_cross_category(self):
        # Edge (d): on the SAME cross-category input, check_cat=False drops the
        # redundant feature (first arm `not check_cat` True) while True keeps both.
        rows = [(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                (_feat("SCB"), "TMD", [1, 2, 3], 0.8)]
        df = _df_feat(rows)
        out_true = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        out_false = filtering(df=df, df_scales=DF_SCALES, check_cat=False)
        assert set(_survivors(out_true)) == {_feat("SCA"), _feat("SCB")}
        assert _survivors(out_false) == [_feat("SCA")]
        assert set(_survivors(out_true)) != set(_survivors(out_false))

    def test_nan_category_treated_as_own_category(self):
        # Edge (e): a NaN category compares unequal to everything (NaN != NaN and
        # NaN != "ASA"), so the gate never fires -> the feature is kept, no raise.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCB"), np.nan, [1, 2, 3], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCB")}

    def test_both_nan_categories_treated_as_distinct(self):
        # Edge (e) variant: two NaN categories are still distinct (NaN != NaN),
        # so neither is dropped despite full overlap + correlation.
        df = _df_feat([(_feat("SCA"), np.nan, [1, 2, 3], 0.9),
                       (_feat("SCB"), np.nan, [1, 2, 3], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCB")}

    def test_low_overlap_keeps_both(self):
        # Same category but disjoint positions -> overlap 0 and not a subset, so
        # the redundancy check is skipped and both are kept (overlap arm False).
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCB"), "ASA", [7, 8, 9], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCB")}

    def test_high_overlap_low_correlation_keeps_both(self):
        # Same category + full overlap but anti-correlated scales (cor=-1.0 <=
        # max_cor) -> not redundant (cor > max_cor arm False), both kept.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCC"), "ASA", [1, 2, 3], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCC")}

    def test_subset_positions_trigger_redundancy(self):
        # pos is a strict subset of the seed's positions while overlap < max_overlap
        # -> the `or pos.issubset(top_pos)` arm still triggers the correlation check
        # and (correlated scale) drops the subset feature.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3, 4, 5, 6, 7, 8], 0.9),
                       (_feat("SCB"), "ASA", [1, 2], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, max_overlap=0.5, check_cat=True)
        assert _survivors(out) == [_feat("SCA")]

    def test_n_filter_caps_kept_features(self):
        # n_filter=1 -> the loop breaks before considering the second feature.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                       (_feat("SCC"), "TMD", [7, 8, 9], 0.8)])
        out = filtering(df=df, df_scales=DF_SCALES, n_filter=1, check_cat=True)
        assert len(out) == 1
        assert _survivors(out) == [_feat("SCA")]

    @settings(max_examples=5, deadline=None)
    @given(auc=some.floats(min_value=0.51, max_value=0.99))
    def test_seed_feature_always_survives(self, auc):
        # Property: the top-AUC feature (the popped seed) is never dropped.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], auc),
                       (_feat("SCB"), "ASA", [1, 2, 3], auc - 0.5)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert _feat("SCA") in _survivors(out)


class TestFilteringComplex:
    """Cross-parameter interactions and the public-API stability anchor."""

    def test_check_cat_only_difference_is_cross_category_pairs(self):
        # Within one category the two modes agree; the modes can only diverge on
        # cross-category pairs (verified by the third, distinct-category feature).
        rows = [(_feat("SCA"), "ASA", [1, 2, 3], 0.9),
                (_feat("SCB"), "ASA", [1, 2, 3], 0.8),
                (_feat("SCD"), "TMD", [1, 2, 3], 0.7)]
        df = _df_feat(rows)
        out_true = set(_survivors(filtering(df=df, df_scales=DF_SCALES, check_cat=True)))
        out_false = set(_survivors(filtering(df=df, df_scales=DF_SCALES, check_cat=False)))
        # SCB dropped in both (same category as SCA, correlated). SCD is also
        # correlated with the seed but in a different category, so it is kept only
        # under check_cat=True (cross-category gate) and dropped under False.
        assert out_true == {_feat("SCA"), _feat("SCD")}
        assert out_false == {_feat("SCA")}

    def test_cpp_run_check_cat_true_is_deterministic(self):
        # KPI: valid-input CPP.run(check_cat=True) output is stable (byte-identical
        # across repeated runs on the same fixed cell).
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(8).T
        kwargs = dict(labels=labels, n_filter=10, n_jobs=1)
        df_a = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False).run(
            check_cat=True, **kwargs)
        df_b = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False).run(
            check_cat=True, **kwargs)
        assert_frame_equal(df_a, df_b)

    def test_cpp_run_check_cat_false_is_valid_and_may_differ(self):
        # check_cat=False produces a valid df_feat over the same fixed cell; both
        # modes return the standard CPP schema and at most n_filter features.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(8).T
        df_true = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False).run(
            labels=labels, n_filter=10, n_jobs=1, check_cat=True)
        df_false = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False).run(
            labels=labels, n_filter=10, n_jobs=1, check_cat=False)
        assert ut.COL_FEATURE in df_true.columns and ut.COL_FEATURE in df_false.columns
        assert len(df_true) <= 10 and len(df_false) <= 10
        assert len(df_true) >= 1 and len(df_false) >= 1


class TestFilteringGoldenValues:
    """Hand-derived survivor sets on tiny inputs (environment-independent)."""

    def test_golden_same_category_chain(self):
        # SCA (seed) drops SCB (same cat, correlated, full overlap) but keeps SCC
        # (same cat, anti-correlated). Expected survivors: {SCA, SCC}.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.95),
                       (_feat("SCB"), "ASA", [1, 2, 3], 0.90),
                       (_feat("SCC"), "ASA", [1, 2, 3], 0.85)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=True)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCC")}

    def test_golden_check_cat_false_drops_correlated_cross_category(self):
        # check_cat=False: SCB dropped (correlated with seed regardless of cat),
        # SCC kept (anti-correlated). Expected survivors: {SCA, SCC}.
        df = _df_feat([(_feat("SCA"), "ASA", [1, 2, 3], 0.95),
                       (_feat("SCB"), "TMD", [1, 2, 3], 0.90),
                       (_feat("SCC"), "PROFILE", [1, 2, 3], 0.85)])
        out = filtering(df=df, df_scales=DF_SCALES, check_cat=False)
        assert set(_survivors(out)) == {_feat("SCA"), _feat("SCC")}
