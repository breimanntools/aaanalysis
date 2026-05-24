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
    from aaanalysis.feature_engineering._backend.cpp._filters_num_c._get_feature_matrix_c import (
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


class TestRunCParity:
    """End-to-end CPP.run vs CPP.run_c parity through the full pipeline."""

    def test_defaults_check_exact(self):
        import pandas as pd
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_old = cpp.run(labels=labels, n_jobs=1)
        df_c = cpp.run_c(df_seq=df_seq, labels=labels, n_jobs=1)
        pd.testing.assert_frame_equal(df_old, df_c, check_exact=True)

    def test_parametric_check_exact(self):
        import pandas as pd
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_old = cpp.run(labels=labels, n_jobs=1, parametric=True)
        df_c = cpp.run_c(df_seq=df_seq, labels=labels, n_jobs=1, parametric=True)
        pd.testing.assert_frame_equal(df_old, df_c, check_exact=True)

    def test_accept_gaps_true_check_exact(self):
        """``accept_gaps=True`` activates the nanmean Cython kernels; must still be bit-exact.

        Regression test for the bug where ``accept_gaps=True`` fell through to
        a pure-Python path, silently disabling all Cython speedup.
        """
        import pandas as pd
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, accept_gaps=True)
        df_old = cpp.run(labels=labels, n_jobs=1)
        df_c = cpp.run_c(df_seq=df_seq, labels=labels, n_jobs=1)
        pd.testing.assert_frame_equal(df_old, df_c, check_exact=True)

    def test_n_batches_check_exact(self):
        """``run_c(n_batches=N)`` routes through ``cpp_run_num_batch`` with the
        Cython builder; output must remain bit-exact with legacy ``run(n_batches=N)``.
        """
        import pandas as pd
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_old = cpp.run(labels=labels, n_jobs=1, n_batches=4)
        df_c = cpp.run_c(df_seq=df_seq, labels=labels, n_jobs=1, n_batches=4)
        pd.testing.assert_frame_equal(df_old, df_c, check_exact=True)
