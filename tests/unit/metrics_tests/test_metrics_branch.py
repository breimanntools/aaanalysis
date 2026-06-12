"""Branch-coverage tests reaching the per-protein / detection / smoothing metric
arms in aaanalysis/_utils/metrics.py and the n_classes guard in
aaanalysis/metrics/_metrics.py, all through the public aa.comp_* surface.

House style: Test<X>Branch classes; negatives with pytest.raises(match=...);
positives with @given + @settings(max_examples=5, deadline=None).
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as some

import aaanalysis as aa


class TestCompBicScoreBranch:
    """metrics/_metrics.py L21/22 (n_classes >= n_samples)."""

    def test_n_classes_ge_n_samples_raises(self):
        # 3 samples, 3 distinct cluster labels -> n_classes == n_samples.
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        with pytest.raises(ValueError, match="must be smaller than n_samples"):
            aa.comp_bic_score(X=X, labels=[0, 1, 2])

    @given(some.integers(min_value=4, max_value=8))
    @settings(max_examples=5, deadline=None)
    def test_valid_two_clusters(self, n):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 3))
        labels = [0, 1] * (n // 2) + [0] * (n % 2)
        bic = aa.comp_bic_score(X=X, labels=labels)
        assert np.isfinite(bic)


class TestCompPerProteinApBranch:
    """_utils/metrics.py L171 (hits == 0 -> return 0.0)."""

    def test_no_hit_returns_zero(self):
        # Positive site is the NaN residue, which is dropped before ranking,
        # so no finite ranked residue matches -> hits == 0 -> 0.0.
        ap = aa.comp_per_protein_ap(list_scores=[[0.9, np.nan]],
                                    list_positions=[[1]])
        assert ap[0] == 0.0

    @given(some.integers(min_value=0, max_value=2))
    @settings(max_examples=5, deadline=None)
    def test_perfect_hit(self, tol):
        ap = aa.comp_per_protein_ap(list_scores=[[0.9, 0.1, 0.2]],
                                    list_positions=[[0]], tolerance=tol)
        assert ap[0] == 1.0


class TestCompDetectionMetricsBranch:
    """_utils/metrics.py L206 (tolerance window loop reached)."""

    def test_tolerance_window_credits_call(self):
        # A call one residue off the true site is credited as TP when tolerance=1.
        res = aa.comp_detection_metrics(list_scores=[[0.9, 0.9, 0.1]],
                                        list_positions=[[1]],
                                        threshold=0.5, tolerance=1)
        assert res["tp"] >= 1

    @given(some.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=5, deadline=None)
    def test_threshold_runs(self, thr):
        res = aa.comp_detection_metrics(list_scores=[[1.0, 0.0]],
                                        list_positions=[[0]], threshold=thr)
        assert set(res) >= {"recall", "precision", "f1", "mcc"}


class TestCompSmoothScoresBranch:
    """_utils/metrics.py L290 (wsum == 0 -> position stays NaN)."""

    def test_all_nan_stays_nan(self):
        out = aa.comp_smooth_scores(scores=[np.nan, np.nan, np.nan], window=2)
        assert np.all(np.isnan(out))

    @given(some.integers(min_value=1, max_value=3))
    @settings(max_examples=5, deadline=None)
    def test_smoothing_runs(self, window):
        out = aa.comp_smooth_scores(scores=[0.0, 1.0, 0.0, 1.0, 0.0], window=window)
        assert len(out) == 5
