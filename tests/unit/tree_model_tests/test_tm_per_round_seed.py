"""Anchor tests for TreeModel per-round seeding.

Under a fixed ``random_state`` each round must reseed to ``random_state + i`` so the rounds
are independent: the multi-round importance mean is a real Monte-Carlo average and
``feat_importance_std`` is non-zero. Before the fix every round used the same seed, so the
std collapsed to exactly 0 and rounds 2..N were wasted.

These are structural invariants (non-zero std, same-seed reproducibility, single-round std=0)
rather than a frozen importance hash: RandomForest Gini importances are not bit-identical
across the CI matrix (BLAS / sklearn build), so an exact-value anchor would be flaky, whereas
these invariants are platform-independent and encode exactly what the fix guarantees.
"""
import numpy as np
import aaanalysis as aa
from sklearn.datasets import make_classification


def _xy():
    X, y = make_classification(n_samples=120, n_features=30, n_informative=12, random_state=0)
    return X, y.tolist()


class TestTreeModelPerRoundSeed:
    def test_fixed_seed_gives_nonzero_importance_std(self):
        # The core defect fix: fixed seed -> rounds differ -> non-zero std (was exactly 0).
        X, y = _xy()
        tm = aa.TreeModel(random_state=42).fit(X, labels=y, n_rounds=5, n_feat_min=10, n_feat_max=20)
        std = np.asarray(tm.feat_importance_std, dtype=float)
        assert np.nanmax(std) > 0
        assert (std > 0).sum() >= 1

    def test_fixed_seed_is_reproducible(self):
        # Reseeding is deterministic: same seed -> identical mean and std across runs.
        X, y = _xy()
        a = aa.TreeModel(random_state=42).fit(X, labels=y, n_rounds=5, n_feat_min=10, n_feat_max=20)
        b = aa.TreeModel(random_state=42).fit(X, labels=y, n_rounds=5, n_feat_min=10, n_feat_max=20)
        assert np.allclose(np.asarray(a.feat_importance, dtype=float),
                           np.asarray(b.feat_importance, dtype=float))
        assert np.allclose(np.asarray(a.feat_importance_std, dtype=float),
                           np.asarray(b.feat_importance_std, dtype=float))

    def test_different_seeds_give_different_importance(self):
        # Different base seeds -> different (but valid) importance signatures.
        X, y = _xy()
        a = aa.TreeModel(random_state=42).fit(X, labels=y, n_rounds=5, n_feat_min=10, n_feat_max=20)
        b = aa.TreeModel(random_state=7).fit(X, labels=y, n_rounds=5, n_feat_min=10, n_feat_max=20)
        assert not np.allclose(np.asarray(a.feat_importance, dtype=float),
                               np.asarray(b.feat_importance, dtype=float))

    def test_single_round_has_zero_std(self):
        # Sanity: with one round there is no spread, so std must be exactly 0.
        X, y = _xy()
        tm = aa.TreeModel(random_state=42).fit(X, labels=y, n_rounds=1, n_feat_min=10, n_feat_max=20)
        assert np.allclose(np.asarray(tm.feat_importance_std, dtype=float), 0.0)
