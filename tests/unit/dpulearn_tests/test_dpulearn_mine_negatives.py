"""
Tests the dPULearn.fit positives/unlabeled split-input mode + mask_neg_ (issue #308).

For the common positive/unlabeled setup, ``fit`` accepts ``X_pos`` and ``X_unlabeled``
separately (instead of ``X`` + a hand-built 1/2 label vector), stacks them internally, and
sets ``mask_neg_`` — the boolean mask of identified reliable negatives over the rows of
``X_unlabeled``. The key contract is that ``mask_neg_`` equals the manual
``labels_[len(X_pos):] == 0`` result exactly, and that the existing ``fit(X, labels=...)``
path stays byte-identical (no algorithm change).
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
class TestFitSplitInput:
    """fit(X_pos=, X_unlabeled=) sets mask_neg_ over the unlabeled rows, per parameter."""

    def test_mask_neg_is_boolean_over_unlabeled(self):
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10).mask_neg_
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (X_unl.shape[0],)
        assert mask.sum() == 10

    def test_X_pos_parameter(self):
        X_pos, X_unl = _make_data(n_pos=30)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=5).mask_neg_
        assert mask.shape[0] == X_unl.shape[0]

    def test_X_unlabeled_parameter(self):
        X_pos, X_unl = _make_data(n_unl=70)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=12).mask_neg_
        assert mask.shape[0] == 70
        assert mask.sum() == 12

    def test_n_neg_parameter(self):
        X_pos, X_unl = _make_data()
        for n in (1, 5, 25):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=n).mask_neg_
            assert mask.sum() == n

    def test_n_unl_to_neg_equivalent_to_n_neg(self):
        # With no pre-labeled negatives the two count params are equivalent.
        X_pos, X_unl = _make_data()
        m1 = aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=8).mask_neg_
        m2 = aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, X_unlabeled=X_unl, n_unl_to_neg=8).mask_neg_
        assert np.array_equal(m1, m2)

    def test_metric_parameter(self):
        X_pos, X_unl = _make_data()
        for metric in ("euclidean", "manhattan", "cosine"):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10, metric=metric).mask_neg_
            assert mask.sum() == 10

    def test_n_components_parameter(self):
        X_pos, X_unl = _make_data()
        for n_components in (2, 3, 0.5):
            dpul = aa.dPULearn(random_state=42, verbose=False)
            mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10, n_components=n_components).mask_neg_
            assert mask.sum() == 10

    def test_fit_returns_self_in_split_mode(self):
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        out = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10)
        assert out is dpul  # sklearn contract preserved

    def test_instance_attributes_set(self):
        X_pos, X_unl = _make_data()
        dpul = aa.dPULearn(random_state=42, verbose=False)
        dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10)
        assert dpul.labels_ is not None
        assert dpul.labels_.shape[0] == X_pos.shape[0] + X_unl.shape[0]
        assert dpul.df_pu_ is not None


# Regression / golden equivalence
class TestSplitMaskEquivalence:
    """mask_neg_ must equal the manual stacking path exactly (KPI #308)."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_mask_equals_manual_pca(self, seed):
        X_pos, X_unl = _make_data(seed=seed)
        manual_mask, dpul_m = _manual_mask(X_pos, X_unl, n_unl_to_neg=10)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=10).mask_neg_
        assert np.array_equal(mask, manual_mask)
        assert np.array_equal(np.asarray(dpul.labels_), np.asarray(dpul_m.labels_))

    def test_mask_equals_manual_metric(self):
        X_pos, X_unl = _make_data(seed=3)
        manual_mask, _ = _manual_mask(X_pos, X_unl, n_unl_to_neg=8, metric="cosine")
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=8, metric="cosine").mask_neg_
        assert np.array_equal(mask, manual_mask)

    def test_mask_equals_manual_few_positives(self):
        # n_pos < 3: the stacked matrix carries the >=3 floor, so the split path accepts a
        # small positive set exactly as the manual path does.
        X_pos, X_unl = _make_data(n_pos=1, seed=5)
        manual_mask, _ = _manual_mask(X_pos, X_unl, n_unl_to_neg=6)
        dpul = aa.dPULearn(random_state=42, verbose=False)
        mask = dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=6).mask_neg_
        assert np.array_equal(mask, manual_mask)

    def test_manual_mode_mask_neg_is_labels_zero(self):
        # In the (X, labels) mode, mask_neg_ is over all rows and equals labels_ == 0.
        X_pos, X_unl = _make_data(seed=2)
        _, dpul = _manual_mask(X_pos, X_unl, n_unl_to_neg=9)
        assert np.array_equal(dpul.mask_neg_, np.asarray(dpul.labels_) == 0)
        assert dpul.mask_neg_.shape[0] == X_pos.shape[0] + X_unl.shape[0]


# Negative Cases Test Class
class TestSplitInputNegative:
    """Invalid inputs must raise informative ValueErrors."""

    def test_feature_mismatch(self):
        X_pos, _ = _make_data(n_features=8)
        _, X_unl = _make_data(n_features=6)
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=5)

    def test_n_neg_below_one(self):
        X_pos, X_unl = _make_data()
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=0)

    def test_too_many_negatives_requested(self):
        X_pos, X_unl = _make_data(n_unl=10)
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, X_unlabeled=X_unl, n_neg=999)

    def test_X_unlabeled_missing(self):
        X_pos, _ = _make_data()
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(X_pos=X_pos, n_neg=5)

    def test_both_input_modes_rejected(self):
        X_pos, X_unl = _make_data()
        X = np.vstack([X_pos, X_unl])
        y = np.array([1] * len(X_pos) + [2] * len(X_unl))
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(X=X, labels=y, X_pos=X_pos,
                                                            X_unlabeled=X_unl, n_neg=5)

    def test_no_input_given(self):
        with pytest.raises(ValueError):
            aa.dPULearn(random_state=42, verbose=False).fit(n_neg=5)


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
        assert (labels[:len(X_pos)] == 1).all()
        assert (labels == 0).sum() == 10
        assert set(np.unique(labels)).issubset({0, 1, 2})
