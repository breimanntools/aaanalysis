"""Tests for the composition baselines behind ``find_features(baselines=...)``.

AAC (k=1) is a first-class CPP ``df_feat`` over a one-hot identity scale set with the whole-part
``Segment(1,1)`` split; DPC / higher k-mers are visualized as per-k-mer ``test − ref`` composition
signal maps. ``find_features(baselines=...)`` scores each as a reference ``df_eval`` row and attaches
the drawn objects as ``ax.baselines`` without changing the returned CPP winner.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.pipe._find_features import find_features, _resolve_baselines
from aaanalysis.pipe._composition_baseline import (
    build_aac_df_feat, comp_kmer_signal, comp_kmer_df_feat, plot_composition_map, AAC_CAT_COLORS)

aa.options["verbose"] = False
warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
    return df_seq, df_seq["label"].to_list()


# I return_scales (one-hot scales + df_cat)
class TestReturnScales:
    def test_k1_identity_scales_and_cat(self, data):
        df_seq, _ = data
        sf = aa.SequenceFeature(verbose=False)
        X, df_scales, df_cat = sf.kmer_composition(df_seq=df_seq, k=1, return_scales=True)
        assert df_scales.shape == (20, 20)
        assert np.array_equal(df_scales.to_numpy(), np.eye(20))          # one-hot identity
        assert list(df_scales.index) == list(ut.LIST_CANONICAL_AA)
        for col in (ut.COL_SCALE_ID, ut.COL_CAT, ut.COL_SUBCAT, ut.COL_SCALE_NAME, ut.COL_SCALE_DES):
            assert col in df_cat.columns
        assert set(df_cat[ut.COL_CAT]) <= set(AAC_CAT_COLORS)            # every category has a color

    def test_k2_no_scales_but_kmer_cat(self, data):
        df_seq, _ = data
        sf = aa.SequenceFeature(verbose=False)
        X, df_scales, df_cat = sf.kmer_composition(df_seq=df_seq, k=2, return_scales=True)
        assert df_scales is None                                         # a dipeptide is not a scale
        assert len(df_cat) == 400 and df_cat[ut.COL_SUBCAT].iloc[1].count("-") == 1

    def test_default_return_unchanged(self, data):
        df_seq, _ = data
        sf = aa.SequenceFeature(verbose=False)
        assert isinstance(sf.kmer_composition(df_seq=df_seq, k=1), np.ndarray)   # no tuple by default


# II AAC as CPP df_feat
class TestAacDfFeat:
    def test_aac_is_whole_part_segment_df_feat(self, data):
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        df_feat, df_parts, df_scales, df_cat = build_aac_df_feat(
            sf=sf, df_seq=df_seq, labels=labels, n_jobs=1, random_state=0)
        assert 0 < len(df_feat) <= 20
        assert all("Segment(1,1)" in f for f in df_feat[ut.COL_FEATURE])   # whole-part composition
        # reconstructable by feature_matrix on the one-hot scales
        X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts,
                              df_scales=df_scales, n_jobs=1)
        assert X.shape == (len(df_parts), len(df_feat)) and np.isfinite(X).all()


# III k-mer signal + plot
class TestKmerSignalPlot:
    @pytest.mark.parametrize("k,n", [(1, 20), (2, 400), (3, 8000)])
    def test_signal_shape(self, data, k, n):
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        signal, kmers = comp_kmer_signal(sf=sf, df_seq=df_seq, labels=labels, k=k)
        assert signal.shape == (n,) and len(kmers) == n

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_plot_renders(self, k):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=20 ** k)
        ax = plot_composition_map(signal=signal, k=k, top_n=15)
        assert ax is not None
        plt.close("all")


# III.b CPP-style k-mer filtering
class TestKmerFilter:
    def test_df_feat_shape_and_ranking(self, data):
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        df = comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=2, n_filter=15)
        assert len(df) == 15
        for col in (ut.COL_FEATURE, ut.COL_ABS_AUC, ut.COL_MEAN_DIF, ut.COL_STD_TEST, ut.COL_SUBCAT):
            assert col in df.columns
        assert (df[ut.COL_ABS_AUC].diff().dropna() <= 1e-9).all()        # ranked by abs_auc, best first

    def test_correlation_filter_can_only_drop(self, data):
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        n_none = len(comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=2, n_filter=400))
        n_cor = len(comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=2, n_filter=400, max_cor=0.5))
        assert n_cor <= n_none

    def test_top_features_are_valid_kmers(self, data):
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        df = comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=2, n_filter=10)
        assert all(len(f) == 2 and set(f) <= set(ut.LIST_CANONICAL_AA) for f in df[ut.COL_FEATURE])

    def test_min_count_guard_drops_sparse_kmers(self, data):
        """A higher min-occurrence guard keeps only k-mers present in many sequences (fewer eligible)."""
        df_seq, labels = data
        sf = aa.SequenceFeature(verbose=False)
        n_lo = len(comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=3, n_filter=8000, min_count=1))
        n_hi = len(comp_kmer_df_feat(sf=sf, df_seq=df_seq, labels=labels, k=3, n_filter=8000, min_count=8))
        assert n_hi <= n_lo                                              # guard can only reduce the pool


# IV find_features integration
class TestFindFeaturesBaselines:
    def test_default_has_no_baseline_rows(self, data):
        df_seq, labels = data
        _, _, df_eval = find_features(labels=labels, df_seq=df_seq, search="fast", plot=False,
                                      random_state=0, n_jobs=1)
        assert (df_eval["stage"] == "baseline").sum() == 0

    def test_baselines_true_adds_aac_dpc(self, data):
        df_seq, labels = data
        df_feat, ax, df_eval = find_features(labels=labels, df_seq=df_seq, search="fast", plot=True,
                                             baselines=True, random_state=0, n_jobs=1)
        base = df_eval[df_eval["stage"] == "baseline"]
        assert list(base["scale"]) == ["AAC", "DPC"]
        assert not base["is_selected"].any()                              # reference-only
        assert set(ax.baselines) == {"AAC", "DPC"}
        assert ax.baselines["AAC"]["ax"] is not None                      # AAC feature map drawn
        assert ax.baselines["DPC"]["signal"].shape == (400,)
        plt.close("all")

    def test_baselines_list_orders(self, data):
        df_seq, labels = data
        _, ax, df_eval = find_features(labels=labels, df_seq=df_seq, search="fast", plot=True,
                                       baselines=[1, 3], random_state=0, n_jobs=1)
        assert set(ax.baselines) == {"AAC", "3-mer"}
        plt.close("all")

    @pytest.mark.parametrize("bad", [2.5, [0], [5], "aac", [1, 2.0]])
    def test_invalid_baselines(self, data, bad):
        df_seq, labels = data
        with pytest.raises(ValueError):
            find_features(labels=labels, df_seq=df_seq, search="fast", plot=False, baselines=bad, n_jobs=1)

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_invalid_baseline_min_count(self, data, bad):
        df_seq, labels = data
        with pytest.raises(ValueError):
            find_features(labels=labels, df_seq=df_seq, search="fast", plot=False, baselines=[2],
                          baseline_min_count=bad, n_jobs=1)


# V resolver
class TestResolveBaselines:
    @pytest.mark.parametrize("val,exp", [(False, []), (None, []), (True, [1, 2]),
                                         ([2, 1, 2], [1, 2]), ((3,), [3])])
    def test_resolve(self, val, exp):
        assert _resolve_baselines(val) == exp
