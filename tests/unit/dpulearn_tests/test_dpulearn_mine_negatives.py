"""
This script tests the dPULearn.mine_negatives() convenience method (issue #308).

mine_negatives is additive sugar over dPULearn.fit: it stacks X_pos over X_unlabeled,
builds a 1 (positive) / 2 (unlabeled) label vector, fits, and returns the boolean mask of
identified reliable negatives over the rows of X_unlabeled. The key contract is that the
mask equals the manual ``labels_[len(X_pos):] == 0`` result exactly, and that the existing
``fit`` path stays byte-identical (no algorithm change).
"""
import numpy as np
import pytest

import aaanalysis as aa


# Helper functions
def _make_data(n_pos=20, n_unl=50, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(0.0, 1.0, size=(n_pos, n_features))
    X_unl = rng.normal(0.6, 1.0, size=(n_unl, n_features))
    return X_pos, X_unl


def _manual_mask(X_pos, X_unl, random_state=42, **fit_kwargs):
    """Reproduce the notebook cell 18/24 manual stacking path."""
    X_pool = np.vstack([X_pos, X_unl])
    y_pool = np.array([1] * len(X_pos) + [2] * len(X_unl))
    dpul = aa.dPULearn(random_state=random_state, verbose=False)
    dpul.fit(X=X_pool, labels=y_pool, **fit_kwargs)
    return np.asarray(dpul.labels_)[len(X_pos):] == 0, dpul


# Normal Cases Test Class
class TestMineNegatives:
    """Test dPULearn.mine_negatives() for each parameter individually."""

    def test_returns_boolean_mask_over_unlabeled(self):
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (X_unl.shape[0],)
        assert mask.sum() == 10

    def test_X_pos_parameter(self):
        X_pos, X_unl = _make_data(n_pos=30)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=5)
        assert mask.shape[0] == X_unl.shape[0]

    def test_X_unlabeled_parameter(self):
        X_pos, X_unl = _make_data(n_unl=70)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=12)
        assert mask.shape[0] == 70
        assert mask.sum() == 12

    def test_n_neg_parameter(self):
        X_pos, X_unl = _make_data()
        for n in (1, 5, 25):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=n)
            assert mask.sum() == n

    def test_metric_parameter(self):
        X_pos, X_unl = _make_data()
        for metric in ("euclidean", "manhattan", "cosine"):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl,
                                       n_neg=10, metric=metric)
            assert mask.sum() == 10

    def test_n_components_parameter(self):
        X_pos, X_unl = _make_data()
        for n_components in (2, 3, 0.5):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl,
                                       n_neg=10, n_components=n_components)
            assert mask.sum() == 10

    def test_instance_attributes_set(self):
        """After mining, labels_ / df_pu_ are set so the plotting class works."""
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10)
        assert dpul.labels_ is not None
        assert dpul.labels_.shape[0] == X_pos.shape[0] + X_unl.shape[0]
        assert dpul.df_pu_ is not None


# Regression / golden equivalence
class TestMineNegativesEquivalence:
    """The mask must equal the manual stacking path exactly (KPI #308)."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_mask_equals_manual_pca(self, seed):
        X_pos, X_unl = _make_data(seed=seed)
        manual_mask, dpul_m = _manual_mask(X_pos, X_unl, n_unl_to_neg=10)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10)
        assert np.array_equal(mask, manual_mask)
        assert np.array_equal(np.asarray(dpul.labels_), np.asarray(dpul_m.labels_))

    def test_mask_equals_manual_metric(self):
        X_pos, X_unl = _make_data(seed=3)
        manual_mask, _ = _manual_mask(X_pos, X_unl, n_unl_to_neg=8, metric="cosine")
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl,
                                   n_neg=8, metric="cosine")
        assert np.array_equal(mask, manual_mask)

    def test_mask_equals_manual_few_positives(self):
        # n_pos < 3: the manual stacked path accepts it (the >=3 floor applies to the
        # stacked matrix), so mine_negatives must match it, not reject the small pos set.
        X_pos, X_unl = _make_data(n_pos=1, seed=5)
        manual_mask, _ = _manual_mask(X_pos, X_unl, n_unl_to_neg=6)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=6)
        assert np.array_equal(mask, manual_mask)


# Negative Cases Test Class
class TestMineNegativesNegative:
    """Invalid inputs must raise informative ValueErrors."""

    def test_feature_mismatch(self):
        X_pos, _ = _make_data(n_features=8)
        _, X_unl = _make_data(n_features=6)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        with pytest.raises(ValueError):
            dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=5)

    def test_n_neg_below_one(self):
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        with pytest.raises(ValueError):
            dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=0)

    def test_too_many_negatives_requested(self):
        X_pos, X_unl = _make_data(n_unl=10)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        with pytest.raises(ValueError):
            dpul.mine_negatives(X_pos=X_pos, X_unlabeled=X_unl, n_neg=999)

    def test_X_pos_none(self):
        _, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        with pytest.raises(ValueError):
            dpul.mine_negatives(X_pos=None, X_unlabeled=X_unl, n_neg=5)

    def test_X_unlabeled_none(self):
        X_pos, _ = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        with pytest.raises(ValueError):
            dpul.mine_negatives(X_pos=X_pos, X_unlabeled=None, n_neg=5)


# Existing-fit byte-identical regression
class TestFitUnchanged:
    """The pre-existing fit(X, labels=...) path stays byte-identical (#308 no-change)."""

    def test_fit_pca_unchanged(self):
        X_pos, X_unl = _make_data(seed=11)
        X_pool = np.vstack([X_pos, X_unl])
        y_pool = np.array([1] * len(X_pos) + [2] * len(X_unl))
        dpul = aa.dPULearn(random_state=42, verbose=False)
        dpul.fit(X=X_pool, labels=y_pool, n_unl_to_neg=10)
        labels = np.asarray(dpul.labels_)
        # contract: positives stay 1, exactly 10 mined negatives become 0, rest stay 2
        assert (labels[:len(X_pos)] == 1).all()
        assert (labels == 0).sum() == 10
        assert set(np.unique(labels)).issubset({0, 1, 2})
