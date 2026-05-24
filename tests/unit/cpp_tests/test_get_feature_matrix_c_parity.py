"""Bit-identical parity test for ``get_feature_matrix_c_`` (Cython).

Mirrors ``test_get_feature_matrix_fast_parity.py`` but exercises the
Cython-accelerated builder. Skipped automatically if the compiled
extension isn't available (pure-Python install).
"""
import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp.utils_feature import get_feature_matrix_

try:
    from aaanalysis.feature_engineering._backend.cpp._filters_c._get_feature_matrix_c import (
        get_feature_matrix_c_,
    )
    _HAS_CYTHON_EXT = True
except ImportError:
    _HAS_CYTHON_EXT = False

pytestmark = pytest.mark.skipif(
    not _HAS_CYTHON_EXT,
    reason="Cython extension not built — run setup_inner.py build_ext --inplace",
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


class TestGetFeatureMatrixCParity:

    def test_default(self):
        df_parts, df_scales, features = _build_fixture()
        features = features[:50]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_c = get_feature_matrix_c_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_c), (
            f"max abs diff: {np.abs(X_legacy - X_c).max():.2e}, "
            f"n_diff: {(np.abs(X_legacy - X_c) > 0).sum()}"
        )

    def test_n_jobs_multi(self):
        df_parts, df_scales, features = _build_fixture()
        features = features[:50]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=None,
        )
        X_c = get_feature_matrix_c_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=None,
        )
        assert np.array_equal(X_legacy, X_c)

    def test_segment_only(self):
        df_parts, df_scales, features = _build_fixture(n=20, n_scales=10)
        features = [f for f in features if "Segment" in f][:80]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_c = get_feature_matrix_c_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_c)

    def test_pattern_only(self):
        df_parts, df_scales, features = _build_fixture(n=20, n_scales=10)
        features = [f for f in features if "Pattern" in f and "Periodic" not in f][:80]
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_c = get_feature_matrix_c_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        assert np.array_equal(X_legacy, X_c)

    def test_periodic_pattern_only(self):
        df_parts, df_scales, features = _build_fixture(n=20, n_scales=10)
        features = [f for f in features if "Periodic" in f][:50]
        if not features:
            pytest.skip("No PeriodicPattern features in default split_kws fixture")
        X_legacy = get_feature_matrix_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        X_c = get_feature_matrix_c_(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=False, n_jobs=1,
        )
        # PeriodicPattern falls back to Phase-C Python; should still be bit-exact.
        assert np.array_equal(X_legacy, X_c)


# Note: the former ``TestRunCParity`` class was removed in PR5 when
# ``CPP.run_c`` was deleted from the public surface. The Cython builder
# (``get_feature_matrix_c_``) is now the default backend selected by
# ``cpp.run`` and ``cpp.run_num`` via ``_pick_feature_matrix_builder``;
# the builder-level parity test above (``TestGetFeatureMatrixCParity``)
# exercises that path directly. End-to-end ``cpp.run`` parity is verified
# in ``test_run_num_parity.py`` and the existing CPP test suite.
