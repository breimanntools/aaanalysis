"""Tests for the SequenceFeature model-free pruning methods (issue #32).

Covers ``prune_by_variance`` and ``prune_by_correlation`` (feature pruning: variance +
empirical-correlation filtering of a ``df_feat``). Follows the house testing template: a
normal-case ``Test<Method>`` class (one parameter per test, positive via hypothesis +
negative via pytest.raises), a ``Test<Method>Complex`` class crossing parameters, and a
``Test<Method>GoldenValues`` class with hand-computed expectations (the issue KPIs:
variance-0 removes exactly the constant columns; correlation at ``t`` retains no pair
above ``t`` and is deterministic).
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False
SF = aa.SequenceFeature


# I Helper functions / fixtures
@pytest.fixture(scope="module")
def fitted():
    """Real (sf, df_feat, df_parts, X) from a small DOM_GSEC CPP run, built once."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_parts=df_parts)
    df_feat = cpp.run(labels=labels, n_filter=40)
    X = sf.feature_matrix(features=df_feat, df_parts=df_parts)
    return sf, df_feat.reset_index(drop=True), df_parts, np.asarray(X)


def _max_abs_offdiag_corr(X):
    """Maximum absolute off-diagonal Pearson correlation of a feature matrix."""
    if X.shape[1] < 2:
        return 0.0
    cm = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(cm, 0.0)
    return float(np.nanmax(np.abs(cm)))


# ======================================================================================
# prune_by_variance
# ======================================================================================
class TestPruneByVariance:
    """Normal cases for prune_by_variance (one parameter per test)."""

    def test_returns_df_feat(self, fitted):
        sf, df_feat, df_parts, _ = fitted
        out = sf.prune_by_variance(df_feat=df_feat, df_parts=df_parts)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == list(df_feat.columns)

    def test_default_threshold_keeps_non_constant(self, fitted):
        sf, df_feat, df_parts, X = fitted
        # No constant columns in a real run -> default threshold 0 keeps everything
        assert (X.var(axis=0) > 0).all()
        out = sf.prune_by_variance(df_feat=df_feat, df_parts=df_parts, threshold=0.0)
        assert len(out) == len(df_feat)

    @settings(max_examples=8, deadline=None)
    @given(threshold=some.floats(min_value=0.0, max_value=0.05))
    def test_threshold_monotonic(self, fitted, threshold):
        sf, df_feat, df_parts, X = fitted
        expected = int((X.var(axis=0) > threshold).sum())
        if expected == 0:
            # A threshold above every feature's variance must raise rather than empty-return
            with pytest.raises(ValueError):
                sf.prune_by_variance(df_feat=df_feat, X=X, threshold=threshold)
        else:
            out = sf.prune_by_variance(df_feat=df_feat, X=X, threshold=threshold)
            assert len(out) == expected

    def test_reset_index(self, fitted):
        sf, df_feat, df_parts, X = fitted
        X2 = X.copy()
        X2[:, 0] = 0.5  # constant -> dropped, so the index would otherwise skip 0
        out = sf.prune_by_variance(df_feat=df_feat, X=X2, threshold=0.0)
        assert list(out.index) == list(range(len(out)))

    def test_x_passthrough_matches_df_parts(self, fitted):
        sf, df_feat, df_parts, X = fitted
        a = sf.prune_by_variance(df_feat=df_feat, df_parts=df_parts, threshold=0.005)
        b = sf.prune_by_variance(df_feat=df_feat, X=X, threshold=0.005)
        assert a.equals(b)

    def test_output_passes_schema(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_variance(df_feat=df_feat, X=X, threshold=0.005)
        ut.check_df_feat(df_feat=out)  # #18 schema contract

    # --- negative cases (one per parameter) ---
    def test_invalid_df_feat(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [None, pd.DataFrame(), "x", df_feat.drop(columns=[ut.COL_FEATURE])]:
            with pytest.raises(Exception):
                sf.prune_by_variance(df_feat=bad, df_parts=df_parts)

    def test_invalid_threshold(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [-0.1, "x", None]:
            with pytest.raises(Exception):
                sf.prune_by_variance(df_feat=df_feat, X=X, threshold=bad)

    def test_invalid_X_shape(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [X[:, :-1], X[:, 0], np.full((X.shape[0], X.shape[1]), np.nan)]:
            with pytest.raises(ValueError):
                sf.prune_by_variance(df_feat=df_feat, X=bad)

    def test_missing_df_parts_and_X(self, fitted):
        sf, df_feat, df_parts, X = fitted
        with pytest.raises(Exception):
            sf.prune_by_variance(df_feat=df_feat)

    def test_empty_result_raises(self, fitted):
        sf, df_feat, df_parts, X = fitted
        with pytest.raises(ValueError):
            sf.prune_by_variance(df_feat=df_feat, X=X, threshold=1e9)


class TestPruneByVarianceComplex:
    """Cross-parameter cases for prune_by_variance."""

    @settings(max_examples=6, deadline=None)
    @given(k=some.integers(min_value=1, max_value=5))
    def test_k_injected_constants_dropped(self, fitted, k):
        sf, df_feat, df_parts, X = fitted
        k = min(k, len(df_feat) - 1)
        X2 = X.copy()
        X2[:, :k] = 0.123  # first k columns constant
        out = sf.prune_by_variance(df_feat=df_feat, X=X2, threshold=0.0)
        assert len(out) == len(df_feat) - k
        # exactly the constant features (the first k after build order) are gone
        dropped = set(df_feat[ut.COL_FEATURE].iloc[:k])
        assert dropped.isdisjoint(set(out[ut.COL_FEATURE]))

    def test_accept_gaps_passthrough(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_variance(df_feat=df_feat, df_parts=df_parts,
                                   threshold=0.0, accept_gaps=True)
        assert isinstance(out, pd.DataFrame)


class TestPruneByVarianceGoldenValues:
    """KPI: threshold 0 removes EXACTLY the zero-variance columns and leaves all others."""

    def test_threshold_zero_removes_exactly_constants(self, fitted):
        sf, df_feat, df_parts, X = fitted
        rng = np.random.default_rng(0)
        const_idx = sorted(rng.choice(len(df_feat), size=3, replace=False).tolist())
        X2 = X.copy()
        for j in const_idx:
            X2[:, j] = 7.0  # constant value
        out = sf.prune_by_variance(df_feat=df_feat, X=X2, threshold=0.0)
        kept_feats = set(out[ut.COL_FEATURE])
        const_feats = set(df_feat[ut.COL_FEATURE].iloc[const_idx])
        non_const_feats = set(df_feat[ut.COL_FEATURE]) - const_feats
        assert kept_feats == non_const_feats  # exactly the constants removed


# ======================================================================================
# prune_by_correlation
# ======================================================================================
class TestPruneByCorrelation:
    """Normal cases for prune_by_correlation (one parameter per test)."""

    def test_returns_df_feat(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, df_parts=df_parts)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == list(df_feat.columns)

    @settings(max_examples=8, deadline=None)
    @given(max_cor=some.floats(min_value=0.1, max_value=0.99))
    def test_guarantee_no_pair_above_threshold(self, fitted, max_cor):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=max_cor)
        Xk = X[:, [df_feat[ut.COL_FEATURE].tolist().index(f) for f in out[ut.COL_FEATURE]]]
        assert _max_abs_offdiag_corr(Xk) <= max_cor + 1e-9

    def test_sorted_by_abs_auc_desc(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.7)
        aucs = out[ut.COL_ABS_AUC].to_numpy()
        assert np.all(np.diff(aucs) <= 1e-12)  # non-increasing

    def test_reset_index(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.5)
        assert list(out.index) == list(range(len(out)))

    def test_x_passthrough_matches_df_parts(self, fitted):
        sf, df_feat, df_parts, X = fitted
        a = sf.prune_by_correlation(df_feat=df_feat, df_parts=df_parts, max_cor=0.6)
        b = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.6)
        assert a.equals(b)

    def test_lower_max_cor_prunes_more(self, fitted):
        sf, df_feat, df_parts, X = fitted
        loose = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.9)
        tight = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.3)
        assert len(tight) <= len(loose)

    def test_single_feature_noop(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat.head(1), df_parts=df_parts)
        assert len(out) == 1

    def test_output_passes_schema(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=0.5)
        ut.check_df_feat(df_feat=out)

    # --- negative cases (one per parameter) ---
    def test_invalid_df_feat(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [None, pd.DataFrame(), "x"]:
            with pytest.raises(Exception):
                sf.prune_by_correlation(df_feat=bad, df_parts=df_parts)

    def test_invalid_max_cor(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [-0.1, 1.5, "x", None]:
            with pytest.raises(Exception):
                sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=bad)

    def test_invalid_X_shape(self, fitted):
        sf, df_feat, df_parts, X = fitted
        for bad in [X[:, :-1], np.full((X.shape[0], X.shape[1]), np.nan)]:
            with pytest.raises(ValueError):
                sf.prune_by_correlation(df_feat=df_feat, X=bad)

    def test_missing_df_parts_and_X(self, fitted):
        sf, df_feat, df_parts, X = fitted
        with pytest.raises(ValueError):
            sf.prune_by_correlation(df_feat=df_feat)


class TestPruneByCorrelationComplex:
    """Cross-parameter and determinism cases for prune_by_correlation."""

    def test_determinism_byte_identical(self, fitted):
        sf, df_feat, df_parts, X = fitted
        a = sf.prune_by_correlation(df_feat=df_feat, df_parts=df_parts, max_cor=0.7)
        b = sf.prune_by_correlation(df_feat=df_feat, df_parts=df_parts, max_cor=0.7)
        assert a.equals(b)
        assert list(a[ut.COL_FEATURE]) == list(b[ut.COL_FEATURE])

    def test_constant_columns_retained(self, fitted):
        sf, df_feat, df_parts, X = fitted
        X2 = X.copy()
        X2[:, 0] = 0.5  # constant -> undefined corr -> retained, no spurious warning
        out = sf.prune_by_correlation(df_feat=df_feat, X=X2, max_cor=0.5)
        assert df_feat[ut.COL_FEATURE].iloc[0] in set(out[ut.COL_FEATURE])

    def test_compose_variance_then_correlation(self, fitted):
        sf, df_feat, df_parts, X = fitted
        step1 = sf.prune_by_variance(df_feat=df_feat, df_parts=df_parts, threshold=0.0)
        step2 = sf.prune_by_correlation(df_feat=step1, df_parts=df_parts, max_cor=0.5)
        assert len(step2) <= len(step1) <= len(df_feat)
        ut.check_df_feat(df_feat=step2)

    def test_max_cor_one_keeps_all_non_constant(self, fitted):
        sf, df_feat, df_parts, X = fitted
        out = sf.prune_by_correlation(df_feat=df_feat, X=X, max_cor=1.0)
        assert len(out) == len(df_feat)


class TestPruneByCorrelationGoldenValues:
    """KPI: deterministic abs_auc tie-break keeps the higher-abs_auc feature of a pair."""

    def test_tie_break_keeps_higher_abs_auc(self, fitted):
        sf, df_feat, df_parts, X = fitted
        sub = df_feat.head(2).copy().reset_index(drop=True)
        # Make the two feature columns perfectly correlated
        col = X[:, 0].copy()
        X2 = np.column_stack([col, col * 2.0 + 1.0])  # corr = 1.0
        # Row 1 has the higher abs_auc -> it must be the survivor
        sub.loc[0, ut.COL_ABS_AUC] = 0.10
        sub.loc[1, ut.COL_ABS_AUC] = 0.90
        out = sf.prune_by_correlation(df_feat=sub, X=X2, max_cor=0.7)
        assert len(out) == 1
        assert out[ut.COL_FEATURE].iloc[0] == sub[ut.COL_FEATURE].iloc[1]

    def test_two_uncorrelated_both_kept(self, fitted):
        sf, df_feat, df_parts, X = fitted
        sub = df_feat.head(2).copy().reset_index(drop=True)
        X2 = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                              np.array([6.0, 1.0, 5.0, 2.0, 4.0, 3.0])])
        out = sf.prune_by_correlation(df_feat=sub, X=X2, max_cor=0.7)
        assert len(out) == 2
