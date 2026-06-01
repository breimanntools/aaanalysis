"""This is a script to test SequenceFeature.feature_matrix routing through the fast/Cython builder.

Covers Stage-1 decision D2: ``SequenceFeature.feature_matrix`` now delegates to
``_pick_feature_matrix_builder()`` (the Cython kernel when built, else the
pure-Python fast path) instead of the legacy ``get_feature_matrix_``. The output
must stay byte-identical to legacy across feature counts, n_jobs settings, and
scale-set sizes.
"""
import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp.utils_feature import get_feature_matrix_

aa.options["verbose"] = False


# Helper functions
def _build_fixture(n=10, n_scales=10):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=38).T.head(n_scales).T
    features = sf.get_features(list_parts=list(df_parts), list_scales=list(df_scales))
    return sf, df_parts, df_scales, features


class TestFeatureMatrixBuilder:
    """Byte-exact parity between SequenceFeature.feature_matrix and legacy."""

    def test_default(self):
        sf, df_parts, df_scales, features = _build_fixture()
        features = features[:50]
        X_new = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=1))
        X_legacy = get_feature_matrix_(features=features, df_parts=df_parts,
                                       df_scales=df_scales, accept_gaps=False, n_jobs=1)
        assert np.array_equal(X_new, X_legacy)

    def test_shape(self):
        sf, df_parts, df_scales, features = _build_fixture()
        features = features[:30]
        X_new = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=1))
        assert X_new.shape == (len(df_parts), len(features))

    def test_single_feature(self):
        sf, df_parts, df_scales, features = _build_fixture()
        X_new = np.asarray(sf.feature_matrix(features=[features[0]], df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=1))
        X_legacy = get_feature_matrix_(features=[features[0]], df_parts=df_parts,
                                       df_scales=df_scales, accept_gaps=False, n_jobs=1)
        assert np.array_equal(X_new, X_legacy)

    def test_n_jobs_none(self):
        sf, df_parts, df_scales, features = _build_fixture()
        features = features[:50]
        X_new = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=None))
        X_legacy = get_feature_matrix_(features=features, df_parts=df_parts,
                                       df_scales=df_scales, accept_gaps=False, n_jobs=None)
        assert np.array_equal(X_new, X_legacy)

    def test_default_df_scales(self):
        # df_scales=None should fall back to the default scale set without error.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        features = sf.get_features(list_parts=list(df_parts))[:20]
        X = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=1))
        assert X.shape == (len(df_parts), len(features))


class TestFeatureMatrixBuilderComplex:
    """Larger fixtures and cross-parameter parity."""

    def test_large_features(self):
        sf, df_parts, df_scales, features = _build_fixture(n=20, n_scales=20)
        features = features[:200]
        X_new = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=1))
        X_legacy = get_feature_matrix_(features=features, df_parts=df_parts,
                                       df_scales=df_scales, accept_gaps=False, n_jobs=1)
        assert np.array_equal(X_new, X_legacy)

    def test_segment_only(self):
        sf, df_parts, df_scales, features = _build_fixture(n=15, n_scales=10)
        features = [f for f in features if "Segment" in f][:100]
        X_new = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                             df_scales=df_scales, n_jobs=1))
        X_legacy = get_feature_matrix_(features=features, df_parts=df_parts,
                                       df_scales=df_scales, accept_gaps=False, n_jobs=1)
        assert np.array_equal(X_new, X_legacy)

    def test_repeat_calls_identical(self):
        # Warm-cache (scale lookup) repeat call must be byte-identical.
        sf, df_parts, df_scales, features = _build_fixture()
        features = features[:40]
        X1 = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                          df_scales=df_scales, n_jobs=1))
        X2 = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts,
                                          df_scales=df_scales, n_jobs=1))
        assert np.array_equal(X1, X2)
