"""This is a script to test that CPP features are consumable by scikit-learn.

Integration tier (cross-package seam). ``SequenceFeature.feature_matrix`` returns a
plain numeric matrix ``X`` (one column per feature), so ``X`` composes directly with
any stock scikit-learn estimator / ``Pipeline`` / cross-validator — no AAanalysis glue
and no new public symbol (issues #211, #24). This pins that contract as a regression
gate.

Scope note: feature *selection* (``CPP.run`` -> ``df_feat``) is done once up front in
the shared fixture, so ``cross_val_score`` below scores the **classifier** on a fixed
feature set, **not** the selection step. Leak-free per-fold reselection is a separate
concern, tracked on its own issue and deliberately out of scope for this seam.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import aaanalysis as aa  # noqa: F401  (the package under test)

pytestmark = pytest.mark.integration


class TestCPPFeaturesAreSklearnConsumable:
    """The ``X`` from ``feature_matrix`` drops into a stock sklearn estimator/Pipeline."""

    def test_X_is_plain_numeric_matrix(self, pipeline):
        """feature_matrix yields a finite 2-D numeric array shaped (n_samples, n_features)."""
        X = pipeline["X"]
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[0] == len(pipeline["labels"])
        assert X.shape[1] == len(pipeline["df_feat"])
        assert np.isfinite(X).all()

    def test_pipeline_fits_and_predicts(self, pipeline):
        """X composes in a stock Pipeline([StandardScaler, RandomForest]) and predicts."""
        X, y = pipeline["X"], pipeline["labels"]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert len(pred) == len(y)
        # Predictions are valid class labels from the training set.
        assert set(np.unique(pred)).issubset(set(y))

    def test_X_survives_cross_val_score(self, pipeline):
        """X is well-formed for sklearn CV: 5 finite scores, one per fold.

        This scores the classifier on a FIXED feature set (selection done once up
        front) — it is not a leak-free estimate of the full discover+fit pipeline.
        """
        X, y = pipeline["X"], pipeline["labels"]
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
        assert len(scores) == 5
        assert np.isfinite(scores).all()
