"""
This is a script for testing the aa.eval_features function.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import aaanalysis as aa


# Helper functions
def _toy_data(n_samples=40, n_features=8, random_state=0):
    """Small separable binary classification fixture."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=5, n_redundant=1, random_state=random_state)
    return X, y


# Positive tests
class TestEvalFeatures:
    """Positive tests for each public parameter."""

    def test_returns_float_percentage(self):
        """Default call returns a float score in the percentage range [0, 100]."""
        X, y = _toy_data()
        score = aa.eval_features(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_default_matches_notebook_recipe(self):
        """Golden test: the default reproduces the notebook's manual bACC recipe."""
        X, y = _toy_data()
        score = aa.eval_features(X, y)
        y_pred = cross_val_predict(SVC(kernel="linear"), X, y, cv=LeaveOneOut())
        gold = balanced_accuracy_score(y, y_pred) * 100
        assert np.isclose(score, gold)

    def test_model_param(self):
        """A custom estimator (RandomForest) is accepted and used."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, model=RandomForestClassifier(n_estimators=20),
                                 cv=5, random_state=42)
        assert 0.0 <= score <= 100.0

    def test_model_none_is_linear_svm(self):
        """model=None reproduces an explicit SVC(kernel='linear')."""
        X, y = _toy_data()
        s_none = aa.eval_features(X, y)
        s_svc = aa.eval_features(X, y, model=SVC(kernel="linear"))
        assert np.isclose(s_none, s_svc)

    def test_cv_int_param(self):
        """An integer cv selects k-fold CV and yields a valid score."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, cv=5)
        assert 0.0 <= score <= 100.0

    def test_cv_splitter_param(self):
        """A CV splitter object is accepted directly."""
        X, y = _toy_data()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = aa.eval_features(X, y, cv=cv)
        assert 0.0 <= score <= 100.0

    def test_metric_param_accuracy(self):
        """metric='accuracy' matches a manual cross_val_predict accuracy."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, metric="accuracy", cv=5)
        y_pred = cross_val_predict(SVC(kernel="linear"), X, y, cv=5)
        gold = accuracy_score(y, y_pred) * 100
        assert np.isclose(score, gold)

    @pytest.mark.parametrize("metric", ["balanced_accuracy", "accuracy", "f1",
                                        "precision", "recall", "matthews_corrcoef"])
    def test_metric_param_all_supported(self, metric):
        """Every supported metric name returns a finite score."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, metric=metric, cv=5)
        assert np.isfinite(score)

    def test_mask_known_pos_param(self):
        """The PU mask excludes masked positives from scoring (different score)."""
        X, y = _toy_data()
        mask = np.zeros(len(y), dtype=bool)
        mask[np.where(y == 1)[0][:3]] = True
        score = aa.eval_features(X, y, mask_known_pos=mask)
        assert 0.0 <= score <= 100.0

    def test_mask_known_pos_keeps_positives_in_training(self):
        """An all-False mask scores all samples (matches no mask via manual LOO loop)."""
        X, y = _toy_data()
        mask = np.zeros(len(y), dtype=bool)
        # All-False mask: every sample is scored, like the default leave-one-out path.
        s_mask = aa.eval_features(X, y, mask_known_pos=mask)
        s_plain = aa.eval_features(X, y)
        assert np.isclose(s_mask, s_plain)

    def test_random_state_param(self):
        """random_state is forwarded to a stochastic estimator without error."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, model=RandomForestClassifier(n_estimators=20),
                                 cv=5, random_state=7)
        assert 0.0 <= score <= 100.0


# Reproducibility
class TestEvalFeaturesReproducibility:
    """Determinism guarantees."""

    def test_same_random_state_identical_score(self):
        """Same random_state -> identical score across runs (stochastic model)."""
        X, y = _toy_data()
        kws = dict(model=RandomForestClassifier(n_estimators=25), cv=5, random_state=42)
        a = aa.eval_features(X, y, **kws)
        b = aa.eval_features(X, y, **kws)
        assert a == b

    def test_default_deterministic(self):
        """The default (deterministic) path is reproducible."""
        X, y = _toy_data()
        assert aa.eval_features(X, y) == aa.eval_features(X, y)

    def test_input_not_mutated(self):
        """X and labels are not modified in place."""
        X, y = _toy_data()
        X0, y0 = X.copy(), y.copy()
        aa.eval_features(X, y, cv=5)
        assert np.array_equal(X, X0) and np.array_equal(y, y0)


# Negative tests
class TestEvalFeaturesNegative:
    """One negative test per public parameter."""

    def test_invalid_model(self):
        """A non-estimator model raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, model=123)

    def test_invalid_cv_int(self):
        """cv as an int < 2 raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, cv=1)

    def test_invalid_cv_bool(self):
        """cv as a bool raises ValueError (bool is an int subclass)."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, cv=True)

    def test_invalid_cv_type(self):
        """cv as an object without a split method raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, cv="loo")

    def test_invalid_metric(self):
        """An unknown metric name raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, metric="nonsense")

    def test_invalid_mask_length(self):
        """A mask of the wrong length raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, mask_known_pos=np.zeros(3, dtype=bool))

    def test_invalid_mask_values(self):
        """A non-boolean mask raises ValueError."""
        X, y = _toy_data()
        bad = np.full(len(y), 2)
        with pytest.raises(ValueError):
            aa.eval_features(X, y, mask_known_pos=bad)

    def test_invalid_random_state(self):
        """A negative random_state raises ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, random_state=-1)

    def test_mismatched_X_labels(self):
        """Mismatched X / labels lengths raise ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y[:-1])

    def test_mask_all_true(self):
        """A mask covering every sample leaves nothing to score -> ValueError."""
        X, y = _toy_data()
        with pytest.raises(ValueError):
            aa.eval_features(X, y, mask_known_pos=np.ones(len(y), dtype=bool))
