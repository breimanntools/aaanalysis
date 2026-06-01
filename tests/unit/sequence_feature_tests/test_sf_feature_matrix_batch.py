"""This is a script to test SequenceFeature.feature_matrix(..., batch=True) — amortized per-batch matrices."""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False


# Helper functions
def _setup(n=20):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    sf = aa.SequenceFeature(verbose=False)
    dp = sf.get_df_parts(df_seq=df_seq)
    feats = sf.get_features(list_parts=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"])[:40]
    return sf, dp, feats


def _batches(dp, size=1):
    return [dp.iloc[i:i + size] for i in range(0, len(dp), size)]


class TestFeatureMatrixBatch:
    """Normal cases for batch=True (one positive per parameter + parity)."""

    def test_returns_list(self):
        sf, dp, feats = _setup()
        out = sf.feature_matrix(features=feats, df_parts=_batches(dp), batch=True)
        assert isinstance(out, list)

    def test_single_mode_returns_array(self):
        sf, dp, feats = _setup()
        out = sf.feature_matrix(features=feats, df_parts=dp, batch=False)
        assert isinstance(out, np.ndarray) and out.shape == (len(dp), len(feats))

    def test_batch_count_matches_input(self):
        sf, dp, feats = _setup()
        batches = _batches(dp, size=2)
        out = sf.feature_matrix(features=feats, df_parts=batches, batch=True)
        assert len(out) == len(batches)

    def test_each_shape_correct(self):
        sf, dp, feats = _setup()
        batches = _batches(dp, size=3)
        out = sf.feature_matrix(features=feats, df_parts=batches, batch=True)
        assert all(X.shape == (len(b), len(feats)) for X, b in zip(out, batches))

    def test_exact_vs_per_call(self):
        sf, dp, feats = _setup()
        batches = _batches(dp, size=1)
        out = sf.feature_matrix(features=feats, df_parts=batches, batch=True)
        indep = [sf.feature_matrix(features=feats, df_parts=b) for b in batches]
        assert all(np.allclose(a, b) for a, b in zip(out, indep))

    def test_single_batch_equals_non_batch(self):
        sf, dp, feats = _setup()
        out = sf.feature_matrix(features=feats, df_parts=[dp], batch=True)
        assert np.allclose(out[0], sf.feature_matrix(features=feats, df_parts=dp))

    def test_custom_df_scales(self):
        sf, dp, feats = _setup()
        dfs = aa.load_scales()
        feats2 = sf.get_features(list_parts=["tmd"], list_scales=list(dfs)[:10])[:20]
        out = sf.feature_matrix(features=feats2, df_parts=_batches(dp, 5), df_scales=dfs, batch=True)
        assert len(out) == len(_batches(dp, 5))

    def test_accept_gaps_true(self):
        sf, dp, feats = _setup()
        out = sf.feature_matrix(features=feats, df_parts=_batches(dp, 4), accept_gaps=True, batch=True)
        assert len(out) == len(_batches(dp, 4))

    def test_n_jobs_variants(self):
        sf, dp, feats = _setup()
        for nj in (1, -1):
            out = sf.feature_matrix(features=feats, df_parts=_batches(dp, 5), n_jobs=nj, batch=True)
            assert len(out) == len(_batches(dp, 5))


class TestFeatureMatrixBatchComplex:
    """Negative cases and combinations (one negative per parameter)."""

    def test_df_parts_none_batch_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=None, batch=True)

    def test_empty_list_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=[], batch=True)

    def test_single_df_with_batch_true_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=dp, batch=True)

    def test_list_with_batch_false_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=_batches(dp, 2), batch=False)

    def test_bad_batch_type_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=dp, batch="yes")

    def test_mismatched_parts_columns_raises(self):
        sf, dp, feats = _setup()
        bad = dp.iloc[:2].drop(columns=[dp.columns[0]])
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=[dp.iloc[:2], bad], batch=True)

    def test_bad_features_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=["NOT-A-FEATURE"], df_parts=_batches(dp, 5), batch=True)

    def test_bad_df_scales_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=_batches(dp, 5), df_scales="not_a_df", batch=True)

    def test_bad_accept_gaps_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=_batches(dp, 5), accept_gaps="yes", batch=True)

    def test_bad_n_jobs_raises(self):
        sf, dp, feats = _setup()
        with pytest.raises(ValueError):
            sf.feature_matrix(features=feats, df_parts=_batches(dp, 5), n_jobs=0, batch=True)

    def test_varying_batch_sizes_concatenate_correctly(self):
        sf, dp, feats = _setup()
        batches = [dp.iloc[0:1], dp.iloc[1:5], dp.iloc[5:7]]
        out = sf.feature_matrix(features=feats, df_parts=batches, batch=True)
        indep = [sf.feature_matrix(features=feats, df_parts=b) for b in batches]
        assert [X.shape[0] for X in out] == [1, 4, 2]
        assert all(np.allclose(a, b) for a, b in zip(out, indep))
