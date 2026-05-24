"""End-to-end byte-identical parity test for ``get_feature_matrix_fast_``.

Exercises the fast variant against legacy ``utils_feature.get_feature_matrix_``
in isolation — does NOT go through CPP.run / CPP.run_num. This lets us pin
the bit-exact contract independent of the rest of the CPP pipeline.
"""
import warnings

import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp.utils_feature import get_feature_matrix_
from aaanalysis.feature_engineering._backend.cpp._filters_num._get_feature_matrix_fast import (
    get_feature_matrix_fast_,
)


aa.options["verbose"] = False


def _build_fixture(n=10, n_scales=10):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=38).T.head(n_scales).T
    split_kws = sf.get_split_kws()
    list_scales = list(df_scales)
    features = sf.get_features(list_parts=list(df_parts), split_kws=split_kws,
                               list_scales=list_scales)
    return df_parts, df_scales, features


class TestGetFeatureMatrixFastParity:
    """Bit-exact parity between get_feature_matrix_fast_ and legacy."""

    def test_default(self):
        df_parts, df_scales, features = _build_fixture()
        # Subset features to keep the test fast.
        features = features[:50]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_fast = get_feature_matrix_fast_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        # Strict bit-equality.
        assert np.array_equal(X_legacy, X_fast), (
            f"max abs diff: {np.abs(X_legacy - X_fast).max():.2e}, "
            f"n_diff: {(np.abs(X_legacy - X_fast) > 0).sum()}"
        )

    def test_n_jobs_multi(self):
        df_parts, df_scales, features = _build_fixture()
        features = features[:50]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=None,
        )
        X_fast = get_feature_matrix_fast_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=None,
        )
        assert np.array_equal(X_legacy, X_fast)

    def test_single_feature(self):
        df_parts, df_scales, features = _build_fixture()
        X_legacy = get_feature_matrix_(
            features=[features[0]], df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_fast = get_feature_matrix_fast_(
            features=[features[0]], df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_fast)


class TestGetFeatureMatrixFastParityComplex:
    """Larger fixtures + multi-feature combinations."""

    def test_large_features(self):
        # Bigger feature batch to exercise more code paths.
        df_parts, df_scales, features = _build_fixture(n=20, n_scales=20)
        features = features[:200]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_fast = get_feature_matrix_fast_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_fast)

    def test_full_dom_gsec(self):
        # Full DOM_GSEC slice.
        df_parts, df_scales, features = _build_fixture(n=30, n_scales=10)
        # Take only Segment features to stress that path.
        features = [f for f in features if "Segment" in f][:100]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_fast = get_feature_matrix_fast_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_fast)
