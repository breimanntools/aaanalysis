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


def _manual_balanced_acc(y, y_pred):
    """Hand-rolled balanced accuracy = mean per-class recall (independent of sklearn)."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    recalls = [np.mean(y_pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls)) * 100


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
        """Golden test: the default reproduces the notebook recipe, cross-checked against
        an independent hand-rolled balanced accuracy (mean per-class recall) so the
        assertion does not merely re-run ``balanced_accuracy_score`` against itself."""
        X, y = _toy_data()
        score = aa.eval_features(X, y)
        y_pred = cross_val_predict(SVC(kernel="linear"), X, y, cv=LeaveOneOut())
        gold_sklearn = balanced_accuracy_score(y, y_pred) * 100
        gold_manual = _manual_balanced_acc(y, y_pred)
        assert np.isclose(score, gold_sklearn)
        assert np.isclose(score, gold_manual)

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
                                        "precision", "recall"])
    def test_metric_param_all_supported(self, metric):
        """Every supported metric name returns a finite percentage in [0, 100]."""
        X, y = _toy_data()
        score = aa.eval_features(X, y, metric=metric, cv=5)
        assert np.isfinite(score) and 0.0 <= score <= 100.0

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

    def test_mask_with_int_cv(self):
        """mask_known_pos combined with an integer cv works (int -> stratified k-fold)."""
        X, y = _toy_data()
        mask = np.zeros(len(y), dtype=bool)
        mask[np.where(y == 1)[0][:3]] = True
        score = aa.eval_features(X, y, cv=5, mask_known_pos=mask)
        assert 0.0 <= score <= 100.0


class _RecordingSVC(SVC):
    """Linear SVM that records, on shared class-level lists, the feature rows it is
    trained on and predicted on per fold. Class-level storage survives ``sklearn.clone``
    (which deep-copies constructor params but not class attributes)."""
    train_rows = []
    test_rows = []

    def __init__(self):
        super().__init__(kernel="linear")

    def fit(self, X, y):
        _RecordingSVC.train_rows.append({tuple(np.round(r, 6)) for r in np.asarray(X)})
        return super().fit(X, y)

    def predict(self, X):
        _RecordingSVC.test_rows.append([tuple(np.round(r, 6)) for r in np.asarray(X)])
        return super().predict(X)


class TestEvalFeaturesPUMask:
    """The PU mask-known-positives invariant: masked positives train every scored fold
    and are never themselves scored."""

    def test_masked_positives_train_every_fold_and_never_scored(self):
        X, y = _toy_data(n_samples=24)
        pos_idx = np.where(y == 1)[0][:4]
        mask = np.zeros(len(y), dtype=bool)
        mask[pos_idx] = True
        masked_rows = {tuple(np.round(X[i], 6)) for i in pos_idx}

        _RecordingSVC.train_rows.clear()
        _RecordingSVC.test_rows.clear()
        aa.eval_features(X, y, model=_RecordingSVC(), mask_known_pos=mask)

        assert len(_RecordingSVC.train_rows) > 0
        # Every fold that produced a scored prediction trained on ALL masked positives ...
        for train in _RecordingSVC.train_rows:
            assert masked_rows.issubset(train)
        # ... and no masked positive was ever handed to predict (scored).
        scored = {row for fold in _RecordingSVC.test_rows for row in fold}
        assert masked_rows.isdisjoint(scored)

    def test_masked_positives_train_every_fold_with_kfold_cv(self):
        """Same invariant under a non-LOO splitter: a masked positive that lands in a
        k-fold test partition must be folded back into that fold's training set (never
        dropped from both), and must never be scored."""
        X, y = _toy_data(n_samples=24)
        pos_idx = np.where(y == 1)[0][:4]
        mask = np.zeros(len(y), dtype=bool)
        mask[pos_idx] = True
        masked_rows = {tuple(np.round(X[i], 6)) for i in pos_idx}

        _RecordingSVC.train_rows.clear()
        _RecordingSVC.test_rows.clear()
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        aa.eval_features(X, y, model=_RecordingSVC(), cv=cv, mask_known_pos=mask)

        assert len(_RecordingSVC.train_rows) == 4
        for train in _RecordingSVC.train_rows:
            assert masked_rows.issubset(train)
        scored = {row for fold in _RecordingSVC.test_rows for row in fold}
        assert masked_rows.isdisjoint(scored)

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
