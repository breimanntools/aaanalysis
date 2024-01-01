"""
This is a script for testing the CPP().run() method.
"""
import pytest
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False


def get_parts_splits_scales():
    """"""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=3)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
    df_scales = aa.load_scales().T.head(2).T
    split_kws = aa.SequenceFeature().get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
    return df_parts, labels, split_kws, df_scales 


class TestCPPRun:
    """
    Test class for positive simple test cases of the CPP class run() method.
    """

    # Positive tests
    def test_defaults(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_feat = cpp.run(labels=labels)
        assert isinstance(df_feat, pd.DataFrame)

    def test_all_parst(self):
        """"""
        _, _, split_kws, df_scales = get_parts_splits_scales()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=3)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, all_parts=True)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels)

    def test_valid_labels(self):
        all_data_set_names = [x for x in aa.load_dataset()["Dataset"].to_list() if "AA" not in x
                              and "AMYLO" not in x and "PU" not in x]
        df_scales = aa.load_scales().T.head(2).T
        for name in all_data_set_names:
            df_seq = aa.load_dataset(name=name, n=10, min_len=50)
            labels = df_seq["label"].to_list()
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
            split_kws = sf.get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
            cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
            df_feat = cpp.run(labels=labels, n_jobs=1)
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_label_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        labels = [10 if l == 1 else l for l in labels]
        df_feat = cpp.run(labels=labels, label_test=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_label_ref(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        split_kws = aa.SequenceFeature().get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        labels = [10 if l == 0 else l for l in labels]
        df_feat = cpp.run(labels=labels, label_ref=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
    
    def test_valid_n_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_filter=50)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_pre_filter=200, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_pct_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, pct_pre_filter=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_std_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_std_test=0.99, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_std_test=0.1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_overlap(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_overlap=1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_overlap=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_cor(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_cor=1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_cor=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_check_cat(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, check_cat=False, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_parametric(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, parametric=True, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_start(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, start=-4, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_tmd_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, tmd_len=25, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_jmd_n_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, jmd_n_len=5, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_jmd_c_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, jmd_c_len=2, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_jobs(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, n_jobs=4)
        assert isinstance(df_feat, pd.DataFrame)

    # Negative tests
    def test_invalid_n_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_filter=-1)

    def test_invalid_n_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_pre_filter=-10)

    def test_invalid_pct_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, pct_pre_filter=-5)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, pct_pre_filter=None)

    def test_invalid_max_std_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_std_test=-0.5)

    def test_invalid_max_overlap(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_overlap=-0.1)

    def test_invalid_max_cor(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_cor=-0.1)

    def test_empty_df_parts(self):
        df_scales = aa.load_scales().T.head(2).T
        with pytest.raises(ValueError):
            cpp = aa.CPP(df_parts=pd.DataFrame(), df_scales=df_scales)
            cpp.run(labels=[0, 1])

    def test_mismatched_labels(self):
        df_parts, _, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=[1])  # Assuming df_parts has more than 1 row

    def test_invalid_n_jobs(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_jobs=0)