"""This is a script to test the scikit-learn ``Pipeline`` compatibility seam of CPP features.

Integration tier (ADR-0031): this proves the cross-component contract that CPP
features compose inside a *stock* ``sklearn.pipeline.Pipeline`` with **no new
AAanalysis public symbol** — the key observation being that
``SequenceFeature.feature_matrix(df_feat, df_parts)`` is **stateless**, so it can
be wrapped in a plain ``sklearn.preprocessing.FunctionTransformer`` (issues #211,
folding in #24).

Leakage-aware pattern: CPP feature *discovery* (``CPP.run`` → ``df_feat``) happens
**once, outside** any cross-validation loop; only the deterministic
``feature_matrix`` transform sits inside the ``Pipeline``. The transformer is
therefore parameter-free w.r.t. the fold and cannot leak fold-test rows into the
selected feature set.

Seams covered:
  1. FunctionTransformer(feature_matrix) -> Pipeline -> RandomForestClassifier
     fits and predicts; transformed X has one column per discovered feature.
  2. The Pipeline-routed X is byte-identical to a direct ``feature_matrix`` call
     (no Pipeline-induced drift).
  3. The same Pipeline composes inside ``cross_val_score`` (5 folds, one score
     per fold) — the leakage-aware recipe end-to-end.
"""
from functools import partial

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import aaanalysis as aa

pytestmark = pytest.mark.integration


def _feature_matrix_transform(df_parts, *, features, df_scales):
    """Stateless transform: build the CPP feature matrix ``X`` for ``df_parts``.

    This is the function wrapped in a stock ``FunctionTransformer``. It closes
    over the feature set and scales discovered **once outside** cross-validation,
    so each fold only re-runs the deterministic per-part aggregation.

    Parameters
    ----------
    df_parts : pd.DataFrame
        Sequence parts for the rows of this fold (the ``X`` an sklearn estimator
        is handed at ``fit``/``transform`` time).
    features : array-like
        The discovered CPP feature ids (``df_feat['feature']``).
    df_scales : pd.DataFrame
        The same amino acid scales used to discover ``features``.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix, one column per id in ``features``.
    """
    return aa.SequenceFeature().feature_matrix(
        features=features, df_parts=df_parts, df_scales=df_scales, n_jobs=1)


def _make_function_transformer(df_feat, df_scales):
    """Wrap the stateless ``feature_matrix`` in a stock ``FunctionTransformer``."""
    func = partial(_feature_matrix_transform,
                   features=df_feat["feature"], df_scales=df_scales)
    return FunctionTransformer(func=func)


# ---------------------------------------------------------------------------
# Seam 1: FunctionTransformer(feature_matrix) -> Pipeline -> classifier
# ---------------------------------------------------------------------------
class TestCPPFeaturesInSklearnPipeline:
    """CPP features compose into a stock sklearn Pipeline via FunctionTransformer."""

    def test_pipeline_fits_and_predicts(self, pipeline):
        # df_feat is discovered ONCE, outside any CV loop (leakage-aware).
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        pipe = Pipeline([("cpp_feats", ft),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        pipe.fit(pipeline["df_parts"], pipeline["labels"])
        pred = pipe.predict(pipeline["df_parts"])
        assert len(pred) == len(pipeline["df_parts"])
        # Predictions are valid class labels from the training set.
        assert set(np.unique(pred)).issubset(set(pipeline["labels"]))

    def test_transformed_X_has_one_column_per_feature(self, pipeline):
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        X = ft.fit_transform(pipeline["df_parts"])
        X = np.asarray(X)
        assert X.shape[0] == len(pipeline["df_parts"])
        assert X.shape[1] == len(pipeline["df_feat"])

    def test_pipeline_scores_on_fixture(self, pipeline):
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        pipe = Pipeline([("cpp_feats", ft),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        pipe.fit(pipeline["df_parts"], pipeline["labels"])
        # A model trained and scored on the same tiny fixture should fit it well;
        # this only asserts the seam produces a usable, in-range score.
        score = pipe.score(pipeline["df_parts"], pipeline["labels"])
        assert 0.0 <= score <= 1.0

    def test_pipeline_clones(self, pipeline):
        # cross_val_score clones the estimator per fold; a FunctionTransformer
        # closing over df_feat must survive sklearn.clone for the recipe to work.
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        pipe = Pipeline([("cpp_feats", ft),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        cloned = clone(pipe)
        cloned.fit(pipeline["df_parts"], pipeline["labels"])
        assert len(cloned.predict(pipeline["df_parts"])) == len(pipeline["df_parts"])


# ---------------------------------------------------------------------------
# Seam 2: byte-identical regression (no Pipeline-induced drift)
# ---------------------------------------------------------------------------
class TestPipelineFeatureMatrixByteIdentical:
    """The Pipeline-routed X equals a direct feature_matrix(...) call, byte-for-byte."""

    def test_transformer_X_byte_identical_to_direct_call(self, pipeline):
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        X_pipe = np.asarray(ft.fit_transform(pipeline["df_parts"]))
        X_direct = np.asarray(aa.SequenceFeature().feature_matrix(
            features=pipeline["df_feat"]["feature"], df_parts=pipeline["df_parts"],
            df_scales=pipeline["df_scales"], n_jobs=1))
        assert X_pipe.shape == X_direct.shape
        assert X_pipe.dtype == X_direct.dtype
        # Byte-identical: same shape, same dtype, same raw buffer.
        assert X_pipe.tobytes() == X_direct.tobytes()
        assert np.array_equal(X_pipe, X_direct)

    def test_X_inside_full_pipeline_matches_direct_call(self, pipeline):
        # Route X through a fitted Pipeline's transform step (named-step access),
        # not just the bare transformer, to prove the Pipeline wiring itself
        # introduces no drift.
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        pipe = Pipeline([("cpp_feats", ft),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        pipe.fit(pipeline["df_parts"], pipeline["labels"])
        X_step = np.asarray(
            pipe.named_steps["cpp_feats"].transform(pipeline["df_parts"]))
        X_direct = np.asarray(aa.SequenceFeature().feature_matrix(
            features=pipeline["df_feat"]["feature"], df_parts=pipeline["df_parts"],
            df_scales=pipeline["df_scales"], n_jobs=1))
        assert X_step.tobytes() == X_direct.tobytes()


# ---------------------------------------------------------------------------
# Seam 3: leakage-aware recipe inside cross_val_score
# ---------------------------------------------------------------------------
class TestCPPPipelineCrossValidation:
    """The leakage-aware Pipeline runs inside cross_val_score, one score per fold."""

    def test_cross_val_score_yields_one_score_per_fold(self, pipeline):
        # Feature discovery already happened once (the shared fixture's CPP.run);
        # only the deterministic transform sits inside CV — the documented pattern.
        ft = _make_function_transformer(pipeline["df_feat"], pipeline["df_scales"])
        pipe = Pipeline([("cpp_feats", ft),
                         ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
        labels = np.asarray(pipeline["labels"])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipe, pipeline["df_parts"], labels, cv=cv,
                                 scoring="accuracy")
        assert len(scores) == 5
        assert np.all((scores >= 0.0) & (scores <= 1.0))
