"""Tests for CPP.run_composit() / CPP.run_aac().

run_composit builds a composition df_feat: composition="aac" is positional (one-hot AA scales +
whole-part Segment(1,1) -> feature-map-able df_feat with positions), while composition="dpc"/"kmer"
are non-positional (a k-mer is not a per-residue scale) df_feats scored with CPP's discriminative
statistics and filtered by adjusted AUC.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False
warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def cpp_data():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=30)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature(verbose=False)
    cpp = aa.CPP(df_parts=sf.get_df_parts(df_seq=df_seq), verbose=False)
    return cpp, labels


# I AAC (positional)
class TestRunCompositAAC:
    def test_aac_is_positional_df_feat(self, cpp_data):
        cpp, labels = cpp_data
        df = cpp.run_composit(labels=labels, composition="aac", n_filter=20, n_jobs=1)
        assert 0 < len(df) <= 20
        assert ut.COL_POSITION in df.columns                       # positional
        assert all("Segment(1,1)" in f for f in df[ut.COL_FEATURE])
        assert ut.COL_ABS_AUC in df.columns

    def test_run_aac_alias(self, cpp_data):
        cpp, labels = cpp_data
        a = cpp.run_aac(labels=labels, n_filter=20, n_jobs=1)
        b = cpp.run_composit(labels=labels, composition="aac", n_filter=20, n_jobs=1)
        assert list(a[ut.COL_FEATURE]) == list(b[ut.COL_FEATURE])


# II DPC / k-mer (non-positional)
class TestRunComositKmer:
    def test_dpc_non_positional_ranked(self, cpp_data):
        cpp, labels = cpp_data
        df = cpp.run_composit(labels=labels, composition="dpc", n_filter=15, n_jobs=1)
        assert len(df) == 15
        assert ut.COL_POSITION not in df.columns                   # non-positional
        assert all(len(f) == 2 for f in df[ut.COL_FEATURE])        # dipeptides
        assert (df[ut.COL_ABS_AUC].diff().dropna() <= 1e-9).all()  # ranked by abs_auc, best first

    def test_kmer_k3_tripeptides(self, cpp_data):
        cpp, labels = cpp_data
        df = cpp.run_composit(labels=labels, composition="kmer", k=3, n_filter=10, n_jobs=1)
        assert all(len(f) == 3 for f in df[ut.COL_FEATURE])

    def test_min_count_and_max_cor_reduce(self, cpp_data):
        cpp, labels = cpp_data
        n_base = len(cpp.run_composit(labels=labels, composition="kmer", k=3, n_filter=8000, n_jobs=1))
        n_min = len(cpp.run_composit(labels=labels, composition="kmer", k=3, n_filter=8000,
                                     min_count=8, n_jobs=1))
        n_cor = len(cpp.run_composit(labels=labels, composition="dpc", n_filter=400, max_cor=0.5, n_jobs=1))
        n_nocor = len(cpp.run_composit(labels=labels, composition="dpc", n_filter=400, n_jobs=1))
        assert n_min <= n_base and n_cor <= n_nocor

    @pytest.mark.parametrize("kwargs", [{"composition": "xyz"}, {"composition": "kmer", "k": 0},
                                        {"composition": "kmer", "k": 5}, {"n_filter": 0},
                                        {"min_count": 0}])
    def test_invalid(self, cpp_data, kwargs):
        cpp, labels = cpp_data
        with pytest.raises(ValueError):
            cpp.run_composit(labels=labels, n_jobs=1, **kwargs)


# III df_feat contract
class TestRunComositContract:
    def test_has_cpp_stat_columns(self, cpp_data):
        cpp, labels = cpp_data
        df = cpp.run_composit(labels=labels, composition="dpc", n_filter=10, n_jobs=1)
        for col in (ut.COL_FEATURE, ut.COL_CAT, ut.COL_SUBCAT, ut.COL_ABS_AUC,
                    ut.COL_MEAN_DIF, ut.COL_STD_TEST):
            assert col in df.columns
