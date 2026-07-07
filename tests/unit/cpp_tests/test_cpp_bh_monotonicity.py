"""Anchor tests for BH-adjusted p-values (canonical Benjamini-Hochberg monotonicity).

``_bh_corrected_pvalues`` previously omitted the reverse-cumulative-minimum step, so
``p_val_fdr_bh`` could be non-monotone in p-value order and deviate from canonical BH.
These tests pin it to a canonical reference (an independent explicit suffix-minimum, so
the check does not merely re-run the implementation) and assert monotonicity. Only the
reported column is affected; feature selection (``abs_auc`` / ``abs_mean_dif``) is not.
"""
import numpy as np

from aaanalysis.feature_engineering._backend.cpp._utils_feature_stat import _bh_corrected_pvalues


def _reference_bh(pvals):
    """Canonical BH via an explicit suffix-minimum (independent of ``np.minimum.accumulate``)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    raw = p_sorted * n / np.arange(1, n + 1)
    adj = np.array([raw[i:].min() for i in range(n)])   # min over all later (higher) ranks
    adj = np.minimum(adj, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


class TestBHMonotonicity:
    def test_matches_canonical_bh_random(self):
        rng = np.random.RandomState(0)
        for _ in range(25):
            pvals = rng.uniform(0, 1, size=int(rng.randint(5, 60)))
            np.testing.assert_allclose(_bh_corrected_pvalues(pvals), _reference_bh(pvals), rtol=1e-12, atol=1e-15)

    def test_output_is_monotone_in_pvalue_order(self):
        rng = np.random.RandomState(1)
        pvals = rng.uniform(0, 1, size=50)
        adj = _bh_corrected_pvalues(pvals)
        order = np.argsort(pvals)
        assert np.all(np.diff(adj[order]) >= -1e-12)   # non-decreasing in p-value order

    def test_known_nonmonotone_case(self):
        # Raw p*n/rank is non-monotone here; canonical BH pulls the later value down.
        pvals = np.array([0.01, 0.90, 0.02, 0.85])
        adj = _bh_corrected_pvalues(pvals)
        np.testing.assert_allclose(adj, _reference_bh(pvals), rtol=1e-12)
        order = np.argsort(pvals)
        assert np.all(np.diff(adj[order]) >= -1e-12)

    def test_clipped_to_one(self):
        pvals = np.array([0.5, 0.6, 0.7, 0.99])
        adj = _bh_corrected_pvalues(pvals)
        assert np.all(adj <= 1.0)
