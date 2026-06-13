"""Test that the parallel KLD path returns the same per-feature values as the serial path.

Each feature's KLD is independent, so chunking across workers must not change the result
(verified to floating-point tolerance)."""
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False


class TestKLDParallelEquivalence:
    """``kullback_leibler_divergence_`` must be invariant to ``n_jobs``."""

    # n_jobs reaching this private helper is already normalized by ut.check_n_jobs
    # (which converts -1 -> os.cpu_count()), mirroring auc_adjusted_; so test concrete counts.
    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    @pytest.mark.parametrize("n_jobs", [2, 4])
    def test_parallel_matches_serial(self, seed, n_jobs):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((80, 60))
        labels = np.array([0] * 40 + [1] * 40)
        args = dict(X=X, labels=labels, label_test=0, label_ref=1)
        serial = ut.kullback_leibler_divergence_(**args, n_jobs=1)
        parallel = ut.kullback_leibler_divergence_(**args, n_jobs=n_jobs)
        assert serial.shape == (60,)
        np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-10)

    def test_default_is_serial(self):
        # Public comp_kld calls without n_jobs -> private default n_jobs=1 (serial), unchanged.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 20))
        labels = np.array([0] * 30 + [1] * 30)
        default = ut.kullback_leibler_divergence_(X=X, labels=labels, label_test=0, label_ref=1)
        serial = ut.kullback_leibler_divergence_(X=X, labels=labels, label_test=0, label_ref=1, n_jobs=1)
        np.testing.assert_array_equal(default, serial)
