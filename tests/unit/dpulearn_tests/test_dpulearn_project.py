"""
Tests the dPULearn.project() method (issue #352).

After PCA-based fitting, ``project`` maps held-out samples from the same feature space into the
fitted principal-component coordinates (the ``PCi`` columns of ``df_pu_``). Every ``method``
reconstructs a linear map from the fit pairs ``(X, df_pu_)``, so it must reproduce ``df_pu_`` on the
fitted samples (exact on the fit pool when n_features >= n_samples) and interpolate for new samples.
The default ``lstsq`` must equal the hand-rolled affine projection used in the gamma-secretase use case.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


# Helper functions
def _make_data(n_pos=8, n_unl=12, n_features=30, seed=0):
    """n_features >= n_samples so the reconstructed map is exact on the fit pool."""
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(0.0, 1.0, size=(n_pos, n_features))
    X_unl = rng.normal(0.6, 1.0, size=(n_unl, n_features))
    return X_pos, X_unl


def _fit(X_pos, X_unl, random_state=42, **fit_kwargs):
    dpul = aa.dPULearn(random_state=random_state, verbose=False)
    dpul.fit(X_pos=X_pos, X_unlabeled=X_unl, n_unl_to_neg=4, **fit_kwargs)
    return dpul


def _pc_cols(df_pu):
    return [c for c in df_pu.columns if "PC" in c and "abs" not in c]


# Normal Cases Test Class
class TestdPULearnProject:
    """Test dPULearn.project() for each parameter individually."""

    def test_returns_dataframe_with_pc_columns(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        df_proj = dpul.project(X_pos)
        assert isinstance(df_proj, pd.DataFrame)
        assert list(df_proj.columns) == _pc_cols(dpul.df_pu_)
        assert len(df_proj) == len(X_pos)

    @pytest.mark.parametrize("method", ["lstsq", "components"])
    def test_method_parameter(self, method):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        X_new = np.random.default_rng(1).normal(0.2, 1.0, size=(5, X_pos.shape[1]))
        df_proj = dpul.project(X_new, method=method)
        assert df_proj.shape == (5, len(_pc_cols(dpul.df_pu_)))
        assert np.all(np.isfinite(df_proj.to_numpy()))

    def test_new_samples_shape(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        X_new = np.random.default_rng(3).normal(size=(1, X_pos.shape[1]))
        assert dpul.project(X_new).shape == (1, len(_pc_cols(dpul.df_pu_)))


# Golden / Property Test Class
class TestdPULearnProjectGoldenValues:
    """Exact-on-fit-pool guarantee and notebook parity."""

    @pytest.mark.parametrize("method", ["lstsq", "components"])
    def test_exact_on_fit_pool(self, method):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        X_fit = np.vstack([X_pos, X_unl])                 # rows aligned with df_pu_
        Z_ref = dpul.df_pu_[_pc_cols(dpul.df_pu_)].to_numpy()
        Z_proj = dpul.project(X_fit, method=method).to_numpy()
        assert np.allclose(Z_proj, Z_ref, atol=1e-8)

    def test_lstsq_matches_manual_affine_map(self):
        """Reproduce the use-case hand-rolled np.linalg.lstsq affine projection exactly."""
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        cols = _pc_cols(dpul.df_pu_)
        X_pool = np.vstack([X_pos, X_unl])
        Z_pool = dpul.df_pu_[cols].to_numpy()
        X_known = np.random.default_rng(7).normal(0.1, 1.0, size=(4, X_pos.shape[1]))
        W, *_ = np.linalg.lstsq(np.hstack([X_pool, np.ones((len(X_pool), 1))]), Z_pool, rcond=None)
        Z_manual = np.hstack([X_known, np.ones((len(X_known), 1))]) @ W
        Z_proj = dpul.project(X_known, method="lstsq").to_numpy()
        assert np.allclose(Z_proj, Z_manual, atol=1e-10)

    def test_deterministic(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        X_new = np.random.default_rng(9).normal(size=(5, X_pos.shape[1]))
        a = dpul.project(X_new, method="components").to_numpy()
        b = dpul.project(X_new, method="components").to_numpy()
        assert np.array_equal(a, b)


# Negative Cases Test Class
class TestdPULearnProjectNegative:
    """Test dPULearn.project() rejects invalid input."""

    def test_feature_mismatch(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        with pytest.raises(ValueError):
            dpul.project(np.random.default_rng(0).normal(size=(3, X_pos.shape[1] + 1)))

    def test_invalid_method(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        with pytest.raises(ValueError):
            dpul.project(X_pos, method="bogus")

    def test_X_none(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl)
        with pytest.raises(ValueError):
            dpul.project(None)

    def test_distance_based_has_no_pcs(self):
        X_pos, X_unl = _make_data()
        dpul = _fit(X_pos, X_unl, metric="euclidean")
        with pytest.raises(ValueError):
            dpul.project(X_pos)

    def test_unfitted(self):
        X_pos, _ = _make_data()
        with pytest.raises(ValueError):
            aa.dPULearn(verbose=False).project(X_pos)
