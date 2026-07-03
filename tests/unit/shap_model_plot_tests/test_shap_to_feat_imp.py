"""This script tests the shap_to_feat_imp() helper."""
import numpy as np
import pytest

pytest.importorskip("shap")  # shap_to_feat_imp lives in the pro extra (needs shap to import)
from aaanalysis.explainable_ai_pro import shap_to_feat_imp

RNG = np.random.default_rng(0)


def _manual_impact(v):
    v = np.asarray(v, dtype=float)
    return v / np.abs(v).sum() * 100


class TestShapToFeatImp:
    """Normal cases, one behaviour per test."""

    def test_returns_ndarray_shape(self):
        v = RNG.normal(size=12)
        out = shap_to_feat_imp(v)
        assert isinstance(out, np.ndarray)
        assert out.shape == (12,)

    def test_impact_abs_sums_to_100(self):
        v = RNG.normal(size=20)
        out = shap_to_feat_imp(v, impact=True)
        assert np.isclose(np.abs(out).sum(), 100.0)

    def test_importance_sums_to_100(self):
        v = RNG.normal(size=20)
        out = shap_to_feat_imp(v, impact=False)
        assert np.isclose(out.sum(), 100.0)

    def test_impact_matches_manual_normalization(self):
        v = np.array([0.2, -0.1, 0.3, -0.4])
        assert np.allclose(shap_to_feat_imp(v, impact=True), _manual_impact(v))

    def test_impact_keeps_sign(self):
        v = np.array([0.2, -0.1, 0.3, -0.4])
        out = shap_to_feat_imp(v, impact=True)
        assert np.array_equal(np.sign(out), np.sign(v))

    def test_importance_is_abs_of_impact(self):
        v = RNG.normal(size=15)
        assert np.allclose(np.abs(shap_to_feat_imp(v, impact=True)),
                           shap_to_feat_imp(v, impact=False))

    def test_importance_non_negative(self):
        v = RNG.normal(size=15)
        assert (shap_to_feat_imp(v, impact=False) >= 0).all()

    def test_accepts_list_input(self):
        out = shap_to_feat_imp([1.0, -2.0, 3.0])
        assert np.isclose(np.abs(out).sum(), 100.0)


class TestShapToFeatImpErrors:
    """Negative cases, one wrong parameter per test."""

    def test_2d_input_raises(self):
        with pytest.raises(ValueError):
            shap_to_feat_imp(RNG.normal(size=(3, 4)))

    def test_none_input_raises(self):
        with pytest.raises(ValueError):
            shap_to_feat_imp(None)

    def test_impact_not_bool_raises(self):
        with pytest.raises(ValueError):
            shap_to_feat_imp(RNG.normal(size=5), impact="yes")

    def test_all_zero_raises(self):
        # An all-zero SHAP vector has an undefined normalization; must error, not return nan.
        with pytest.raises(ValueError, match="undefined"):
            shap_to_feat_imp(np.zeros(5))
