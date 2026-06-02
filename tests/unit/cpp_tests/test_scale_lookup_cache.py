"""This is a script to test the df_scales content-hash lookup cache.

Covers the Stage-1 scale-tensor reuse (D3): the module-level ``@lru_cache`` on
``build_scale_lookup`` keyed by ``df_scales`` content (via ``_ScalesKey``) and the
internal ``clear_scale_lookup_cache`` evictor (the cache is self-bounding, so eviction
is an internal utility, not public API — see ADR-0014). The cached scale-matrix must
stay bit-exact, so ``CPP.run`` results are identical whether the cache is cold or warm.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp._filters import (
    _get_feature_matrix_fast as fm,
)

aa.options["verbose"] = False


# Helper functions
def _get_parts_labels(n=10):
    """Return (df_parts, labels) from DOM_GSEC; labels come from the dataset."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    return df_parts, labels


class TestBuildScaleLookup:
    """Normal-case tests for build_scale_lookup / the content-hash cache."""

    def test_returns_four_part_tuple(self):
        fm.clear_scale_lookup_cache()
        out = fm.build_scale_lookup(df_scales=aa.load_scales(top60_n=38))
        assert isinstance(out, tuple) and len(out) == 4
        dict_all_scales, scale_to_idx, scale_matrix_f64, n_aa = out
        assert isinstance(dict_all_scales, dict)
        assert isinstance(scale_to_idx, dict)
        assert isinstance(scale_matrix_f64, np.ndarray)
        assert isinstance(n_aa, int)

    def test_scale_matrix_is_float64(self):
        fm.clear_scale_lookup_cache()
        _, _, scale_matrix_f64, _ = fm.build_scale_lookup(df_scales=aa.load_scales(top60_n=38))
        assert scale_matrix_f64.dtype == np.float64

    def test_scale_matrix_shape(self):
        fm.clear_scale_lookup_cache()
        df_scales = aa.load_scales(top60_n=38)
        _, _, scale_matrix_f64, n_aa = fm.build_scale_lookup(df_scales=df_scales)
        # (n_aa + 1, n_scales): last row is the NaN padding/unknown sentinel.
        assert scale_matrix_f64.shape == (n_aa + 1, len(df_scales.columns))

    def test_hit_on_same_content_distinct_objects(self):
        # Two distinct DataFrame objects with identical content hash equal,
        # so the second build is a cache hit.
        fm.clear_scale_lookup_cache()
        ds1 = aa.load_scales(top60_n=38)
        ds2 = aa.load_scales(top60_n=38)
        assert ds1 is not ds2
        fm.build_scale_lookup(df_scales=ds1)
        c1 = fm._build_scale_lookup_cached.cache_info()
        fm.build_scale_lookup(df_scales=ds2)
        c2 = fm._build_scale_lookup_cached.cache_info()
        assert c2.hits == c1.hits + 1
        assert c2.currsize == 1

    def test_miss_on_different_content(self):
        fm.clear_scale_lookup_cache()
        ds_full = aa.load_scales(top60_n=38)
        ds_small = ds_full.T.head(5).T
        fm.build_scale_lookup(df_scales=ds_full)
        c1 = fm._build_scale_lookup_cached.cache_info()
        fm.build_scale_lookup(df_scales=ds_small)
        c2 = fm._build_scale_lookup_cached.cache_info()
        assert c2.misses == c1.misses + 1
        assert c2.currsize == 2

    def test_clear_evicts(self):
        fm.build_scale_lookup(df_scales=aa.load_scales(top60_n=38))
        assert fm._build_scale_lookup_cached.cache_info().currsize >= 1
        fm.clear_scale_lookup_cache()
        assert fm._build_scale_lookup_cached.cache_info().currsize == 0

    def test_scaleskey_equal_for_same_content(self):
        ds1 = aa.load_scales(top60_n=38)
        ds2 = aa.load_scales(top60_n=38)
        k1, k2 = fm._ScalesKey(ds1), fm._ScalesKey(ds2)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_scaleskey_differs_on_values(self):
        ds = aa.load_scales(top60_n=38)
        ds_mut = ds.copy()
        ds_mut.iloc[0, 0] = ds_mut.iloc[0, 0] + 1.0
        assert fm._ScalesKey(ds) != fm._ScalesKey(ds_mut)


class TestClearScaleLookupCache:
    """Tests for the internal clear_scale_lookup_cache() evictor (ADR-0014)."""

    def test_clear_cache_empties(self):
        fm.build_scale_lookup(df_scales=aa.load_scales(top60_n=38))
        assert fm._build_scale_lookup_cached.cache_info().currsize >= 1
        fm.clear_scale_lookup_cache()
        assert fm._build_scale_lookup_cached.cache_info().currsize == 0

    def test_clear_cache_callable_without_instance(self):
        # Module-level utility; no CPP instance needed.
        fm.clear_scale_lookup_cache()
        assert fm._build_scale_lookup_cached.cache_info().currsize == 0

    def test_clear_cache_idempotent(self):
        fm.clear_scale_lookup_cache()
        fm.clear_scale_lookup_cache()
        assert fm._build_scale_lookup_cached.cache_info().currsize == 0


class TestScaleLookupCacheComplex:
    """Combinations: warm-vs-cold cache must not change CPP.run output."""

    def test_run_identical_cold_vs_warm(self):
        df_parts, labels = _get_parts_labels(n=10)
        df_scales = aa.load_scales(top60_n=38)
        fm.clear_scale_lookup_cache()
        df_cold = aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=20, n_jobs=1)
        # Cache is now warm; a fresh instance must produce identical features.
        df_warm = aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=20, n_jobs=1)
        assert df_cold["feature"].tolist() == df_warm["feature"].tolist()

    def test_run_identical_after_clear(self):
        df_parts, labels = _get_parts_labels(n=10)
        df_scales = aa.load_scales(top60_n=38)
        df_a = aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=20, n_jobs=1)
        fm.clear_scale_lookup_cache()
        df_b = aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=20, n_jobs=1)
        assert df_a["feature"].tolist() == df_b["feature"].tolist()
        for col in ["abs_auc", "abs_mean_dif", "mean_dif"]:
            assert np.allclose(df_a[col].to_numpy(), df_b[col].to_numpy(), equal_nan=True)

    def test_two_scale_sets_coexist_in_cache(self):
        df_parts, labels = _get_parts_labels(n=10)
        ds_full = aa.load_scales(top60_n=38)
        ds_small = ds_full.T.head(8).T
        fm.clear_scale_lookup_cache()
        aa.CPP(df_parts=df_parts, df_scales=ds_full).run(labels=labels, n_filter=10, n_jobs=1)
        aa.CPP(df_parts=df_parts, df_scales=ds_small).run(labels=labels, n_filter=10, n_jobs=1)
        assert fm._build_scale_lookup_cached.cache_info().currsize == 2

    def test_maxsize_is_bounded(self):
        # The cache is bounded so long sweeps cannot grow memory without limit.
        assert fm._build_scale_lookup_cached.cache_info().maxsize == 32

    def test_warm_cache_hit_increments_on_repeat_run(self):
        df_parts, labels = _get_parts_labels(n=10)
        df_scales = aa.load_scales(top60_n=38)
        fm.clear_scale_lookup_cache()
        aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=10, n_jobs=1)
        hits_before = fm._build_scale_lookup_cached.cache_info().hits
        aa.CPP(df_parts=df_parts, df_scales=df_scales).run(labels=labels, n_filter=10, n_jobs=1)
        hits_after = fm._build_scale_lookup_cached.cache_info().hits
        assert hits_after > hits_before
