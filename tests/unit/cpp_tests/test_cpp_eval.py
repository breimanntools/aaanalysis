"""
This is a script for testing the CPP().run() method.
"""
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import random
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False


def get_parts_splits_scales(n_samples=5):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales()
    split_kws = aa.SequenceFeature().get_split_kws()
    return df_parts, labels, split_kws, df_scales


def get_list_df_feat(size=2, n_feat=100):
    df_feat = aa.load_features()
    list_df_feat = []
    for _ in range(size):
        sampled_df = df_feat.sample(n=n_feat, axis=0, replace=False)  # Sample n features
        list_df_feat.append(sampled_df)
    return list_df_feat


class TestCPPEval:
    """Test class for positive test cases of the CPP.eval() method."""

    # Positive tests
    def test_valid_list_df_feat_labels(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        for i in range(3):
            n_feat = random.randint(10, 100)
            size = random.randint(2, 3)
            list_df_feat = get_list_df_feat(n_feat=n_feat, size=size)
            df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels)
            assert isinstance(df_eval, pd.DataFrame)

    def test_valid_label_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        labels = [10 if l == 1 else l for l in labels]
        df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, label_test=10)
        assert isinstance(df_eval, pd.DataFrame)

    def test_valid_label_ref(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        labels = [10 if l == 0 else l for l in labels]
        df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, label_ref=10)
        assert isinstance(df_eval, pd.DataFrame)

    def test_valid_min_th(self):
        for min_th in [0, 0.3, 1]:
            df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
            cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
            list_df_feat = get_list_df_feat()
            df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, min_th=min_th)
            assert isinstance(df_eval, pd.DataFrame)

    @settings(max_examples=10, deadline=8000)
    @given(st.lists(st.text()))
    def test_valid_names_feature_sets(self, names_feature_sets):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        if len(list_df_feat) == len(names_feature_sets):
            df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels,
                               names_feature_sets=names_feature_sets)
            assert isinstance(df_eval, pd.DataFrame)

    def test_valid_list_df_parts(self):
        for i in range(3):
            n_samples = random.randint(5, 60)
            size = random.randint(2, 3)
            df_parts, labels, split_kws, df_scales = get_parts_splits_scales(n_samples=n_samples)
            list_df_parts = [df_parts] * size
            list_df_feat = get_list_df_feat(size=size)
            cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
            df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, list_df_parts=list_df_parts)
            assert isinstance(df_eval, pd.DataFrame)

    def test_valid_n_jobs(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        list_df_feat = get_list_df_feat()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels,
                           n_jobs=None)
        assert isinstance(df_eval, pd.DataFrame)

    # Negative tests
    def test_invalid_list_df_feat_empty(self):
        """Test with empty list_df_feat"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = []
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels)

    def test_invalid_list_df_feat_wrong_type(self):
        """Test with invalid type in list_df_feat"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = ["not_a_dataframe"]
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels)

    def test_invalid_labels_wrong_type(self):
        """Test with invalid type in labels"""
        df_parts, _, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        labels = "not_a_list"
        list_df_feat = get_list_df_feat()
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels)

    def test_invalid_label_test_wrong_type(self):
        """Test with invalid type for label_test"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels, label_test="not_an_int")

    def test_invalid_min_th(self):
        for min_th in [None, -1.3, str(1)]:
            df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
            cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
            list_df_feat = get_list_df_feat()
            with pytest.raises(ValueError):
                df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, min_th=min_th)

    def test_invalid_label_ref_wrong_type(self):
        """Test with invalid type for label_ref"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels, label_ref="not_an_int")

    def test_invalid_names_feature_sets(self):
        """Test non invalid names_feature_sets"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        names_feature_sets = [123, None]  # Invalid types
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels,
                     names_feature_sets=names_feature_sets)
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels,
                     names_feature_sets=[1, 2,3 ,54 ,5 ,6 ])


    def test_invalid_list_df_parts(self):
        """Test with empty list_df_parts"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat()
        list_df_parts = [None, 12]
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels, list_df_parts=list_df_parts)

    def test_invalid_n_jobs_wrong_type(self):
        """Test with invalid type for n_jobs"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        list_df_feat = get_list_df_feat()
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels,
                     n_jobs="not_an_int")

class TestCPPEvalComplex:
    """Test class for positive test cases of the CPP.eval() method."""

    # Positive tests
    def test_complex_case_valid_combination_1(self):
        """Test with a valid combination of parameters"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales(n_samples=10)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat(size=3, n_feat=50)
        list_df_parts = [df_parts] * 3
        df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, list_df_parts=list_df_parts, n_jobs=2)
        assert isinstance(df_eval, pd.DataFrame)

    def test_complex_case_valid_combination_2(self):
        """Test with another valid combination of parameters"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales(n_samples=15)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat(size=2, n_feat=60)
        names_feature_sets = ['set1', 'set2']
        df_eval = cpp.eval(list_df_feat=list_df_feat, labels=labels, names_feature_sets=names_feature_sets, n_jobs=4)
        assert isinstance(df_eval, pd.DataFrame)

    # Negative tests
    def test_complex_case_invalid_combination_1(self):
        """Test with an invalid combination of parameters (mismatched list sizes)"""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales(n_samples=10)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat(size=2, n_feat=40)
        names_feature_sets = ['set1', 'set2', 'set3']  # Mismatch in size with list_df_feat
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels, names_feature_sets=names_feature_sets)

    def test_complex_case_invalid_combination_2(self):
        """Test with another invalid combination of parameters (wrong label types)"""
        df_parts, _, split_kws, df_scales = get_parts_splits_scales(n_samples=20)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        list_df_feat = get_list_df_feat(size=3, n_feat=30)
        labels = 'invalid_labels'  # Invalid label type
        with pytest.raises(ValueError):
            cpp.eval(list_df_feat=list_df_feat, labels=labels, n_jobs=1)
