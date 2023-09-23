"""
This is a script testing methods of SequenceFeature object
"""
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa


# I Unit Tests
class TestGetDfParts:
    """Unit test for loading DataFrame with sequence parts"""

    # Positive unit test
    def test_getting_df_parts_based_on_parts(self, df_seq):
        sf = aa.SequenceFeature()
        assert isinstance(sf.get_df_parts(df_seq=df_seq), pd.DataFrame)
        df = df_seq.drop(["sequence"], axis=1)
        assert isinstance(sf.get_df_parts(df_seq=df, list_parts=["tmd"]), pd.DataFrame)

    def test_getting_df_parts_based_on_seq_info(self, df_seq):
        sf = aa.SequenceFeature()
        assert isinstance(sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10), pd.DataFrame)
        df = df_seq.drop(["tmd"], axis=1)
        assert isinstance(sf.get_df_parts(df_seq=df, jmd_n_len=10, jmd_c_len=10), pd.DataFrame)

    def test_getting_df_parts_based_on_sequence(self, df_seq):
        sf = aa.SequenceFeature()
        assert isinstance(sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10), pd.DataFrame)
        df = df_seq.drop(["tmd", "tmd_start", "tmd_stop", "jmd_c"], axis=1)
        assert isinstance(sf.get_df_parts(df_seq=df, jmd_n_len=10, jmd_c_len=10), pd.DataFrame)
        assert isinstance(sf.get_df_parts(df_seq=df, jmd_n_len=0, jmd_c_len=0, ext_len=0), pd.DataFrame)

    # Negative unit tests
    def test_wrong_inputs(self, df_seq, df_cat, df_scales):
        sf = aa.SequenceFeature()
        for i in [None, "a", df_cat, df_scales, 1.1, -1]:
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=i)
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=df_seq, ext_len=i)
        for i in ["a", df_cat, df_scales, 1.1, -1]:
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=df_seq, jmd_n_len=i, jmd_c_len=i)
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=df_seq, jmd_n_len=i, jmd_c_len=10)
            with pytest.raises(ValueError):
                sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=i)

    def test_corrupted_df_seq(self, corrupted_df_seq):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=corrupted_df_seq)    # Via parametrized fixtures

    def test_wrong_parameter_combinations(self, df_seq, df_scales):
        sf = aa.SequenceFeature()
        df = df_seq.drop(["sequence"], axis=1)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df, jmd_n_len=10, jmd_c_len=10)
        df = df_seq.drop(["tmd"], axis=1)
        with pytest.raises(ValueError):
            sf.get_df_parts(df_seq=df)


class TestGetSplitKws:
    """Unit tests for getting split arguments"""

    # Positive unit test
    def test_get_split_kws(self, df_cat):
        sf = aa.SequenceFeature()
        for i in ["Segment", "Pattern", "PeriodicPattern"]:
            assert isinstance(sf.get_split_kws(n_split_min=2, steps_pattern=[1, 3, 4], split_types=i), dict)

    # Negative unit tests
    def test_wrong_integer_input(self, df_cat):
        sf = aa.SequenceFeature()
        list_int_args = ["n_split_min", "n_split_max", "n_min", "n_max", "len_max"]
        for i in ["a", 1.1, -1, df_cat, dict, None]:
            for arg_names in list_int_args:
                arg = {arg_names: i}
                with pytest.raises(ValueError):
                    sf.get_split_kws(**arg)

    def test_wrong_ordered_list_input(self, df_cat):
        sf = aa.SequenceFeature()
        list_args = [[1, None, df_cat], [2, 1], [-1, 9], [0.1, 0.2], ["a", 4]]
        for list_arg in list_args:
            with pytest.raises(ValueError):
                sf.get_split_kws(steps_pattern=list_arg)
            with pytest.raises(ValueError):
                sf.get_split_kws(steps_periodicpattern=list_arg)

    def test_wrong_combination_of_input(self, df_cat):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_max=4, n_split_min=6)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_max=4, n_min=6)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_max=3, n_min=6, len_max=1)


class TestFeatures:
    """Unit test for creating feature ids"""

    # Positive unit test
    def test_features(self, df_scales, list_parts):
        sf = aa.SequenceFeature()
        split_kws = sf.get_split_kws()
        assert isinstance(sf.get_features(), list)
        for parts in list_parts:
            assert isinstance(sf.get_features(list_parts=parts), list)
            for split_type in split_kws:
                args = dict(list_parts=parts, df_scales=df_scales, split_kws={split_type: split_kws[split_type]})
                assert isinstance(sf.get_features(**args), list)

    # Negative unit tests
    def test_wrong_input(self, df_cat, df_seq):
        sf = aa.SequenceFeature()
        for wrong_input in [1, -1, "TMD", ["TMD"], [1, 2], ["aa", "a"], [["tmd", "tmd_e"]], df_cat, [df_cat, df_seq]]:
            with pytest.raises(ValueError):
                sf.get_features(list_parts=wrong_input)
            with pytest.raises(ValueError):
                sf.get_features(list_parts=["tmd"], df_scales=wrong_input)
            with pytest.raises(ValueError):
                sf.get_features(list_parts=["tmd"], split_kws=wrong_input)

    def test_corrupted_list_parts(self, corrupted_list_parts):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(list_parts=corrupted_list_parts)  # Via parametrized fixtures

    def test_corrupted_df_scales(self, corrupted_df_scales):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(list_parts=["tmd"], df_scales=corrupted_df_scales)    # Via parametrized fixtures

    def test_corrupted_split_kws(self, corrupted_split_kws):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(list_parts=["tmd"], split_kws=corrupted_split_kws)    # Via parametrized fixtures


class TestFeatureName:
    """Unit tests for getting feature names"""

    # Positive unit test
    def test_feat_name(self, df_feat, df_cat):
        sf = aa.SequenceFeature()
        assert isinstance(sf.feat_names(features=df_feat["feature"]), list)
        assert isinstance(sf.feat_names(features=list(df_feat["feature"])), list)
        assert isinstance(sf.feat_names(features=list(df_feat["feature"])[0]), list)
        assert isinstance(sf.feat_names(features=df_feat["feature"], df_cat=df_cat), list)

    # Property based testing
    @given(tmd_len=some.integers(min_value=15, max_value=100),
           jmd_n_len=some.integers(min_value=5, max_value=20),
           jmd_c_len=some.integers(min_value=5, max_value=20),
           ext_len=some.integers(min_value=1, max_value=4),
           start=some.integers(min_value=0, max_value=50))
    @settings(max_examples=10, deadline=None)
    def test_feat_name_tmd_len(self, df_feat_module_scope, tmd_len, jmd_n_len, jmd_c_len, ext_len, start):
        sf = aa.SequenceFeature()
        feat_names = sf.feat_names(features=df_feat_module_scope["feature"],
                                   tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                   ext_len=ext_len, start=start)
        assert isinstance(feat_names, list)

    # Negative unit test
    def test_wrong_features(self, wrong_df):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            feat_names = sf.feat_names(features=wrong_df)

    def test_corrupted_feature(self, df_feat):
        sf = aa.SequenceFeature()
        for col in df_feat:
            if col != "feature":
                with pytest.raises(ValueError):
                    feat_names = sf.feat_names(features=df_feat[col])
        wrong_feat = list(df_feat["feature"])[0]
        wrong_feat = "WRONG" + "-" + wrong_feat.split("-")[1] + "-" +wrong_feat.split("-")[2]
        with pytest.raises(ValueError):
            feat_names = sf.feat_names(features=wrong_feat)

    def test_wrong_df_cat(self, df_feat, wrong_df):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            feat_names = sf.feat_names(features=df_feat["feature"], df_cat=wrong_df)

    def test_corrupted_df_cat(self, df_cat, df_feat):
        sf = aa.SequenceFeature()
        df_cat = df_cat[list(df_cat)[0:1]]
        with pytest.raises(ValueError):
            feat_names = sf.feat_names(features=df_feat["feature"], df_cat=df_cat)


class TestFeatureValue:
    """Unit tests for getting feature values"""

    # Positive unit test
    def test_feature_value(self, df_seq, df_scales, list_parts, list_splits):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        for parts in list_parts:
            for split in list_splits:
                for i in range(0, len(df_scales)):
                    dict_scale = df_scales.iloc[:, i].to_dict()
                    x = sf.add_feat_value(split=split, dict_scale=dict_scale, df_parts=df_parts[parts])
                    assert isinstance(x, np.ndarray)

    def test_accept_gaps(self, df_seq, list_parts, list_splits, df_scales):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        parts, split, dict_scale = list_parts[0], list_splits[0], df_scales.iloc[:, 0].to_dict()
        df = df_parts.copy()
        args = dict(split=split, dict_scale=dict_scale)
        df[parts] = "AAA-CCC"
        assert isinstance(sf.add_feat_value(**args, df_parts=df[parts], accept_gaps=True), np.ndarray)
        with pytest.raises(ValueError):
            sf.add_feat_value(**args, df_parts=df[parts], accept_gaps=False)
        df[parts] = "------"
        with pytest.raises(ValueError):
            sf.add_feat_value(**args, df_parts=df[parts], accept_gaps=True)
        args = dict(split=split, df_parts=df_parts[parts])
        dict_scale_na = dict_scale.copy()
        dict_scale_na["A"] = np.NaN
        assert isinstance(sf.add_feat_value(**args, dict_scale=dict_scale_na, accept_gaps=True), np.ndarray)
        with pytest.raises(ValueError):
            sf.add_feat_value(**args, dict_scale=dict_scale_na, accept_gaps=False)

    # Negative test
    def test_wrong_input(self, df_cat, df_seq, list_parts, list_splits, df_scales):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        parts, split, dict_scale = list_parts[0], list_splits[0], df_scales.iloc[:, 0].to_dict()
        list_wrong_input = [1, -1, "TMD", ["TMD"], None, [1, 2], ["aa", "a"], [["tmd", "tmd_e"]],
                            df_cat, [df_cat, df_seq], dict(a=1)]
        for wrong_input in list_wrong_input:
            with pytest.raises(ValueError):
                sf.add_feat_value(split=wrong_input, dict_scale=dict_scale, df_parts=df_parts[parts])
            with pytest.raises(ValueError):
                sf.add_feat_value(split=split, dict_scale=wrong_input, df_parts=df_parts[parts])
            with pytest.raises(ValueError):
                sf.add_feat_value(split=split, dict_scale=dict_scale, df_parts=wrong_input)

    def test_corrupted_split(self, df_seq, list_parts, df_scales, corrupted_list_splits):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        parts, dict_scale = list_parts[0], df_scales.iloc[:, 0].to_dict()
        with pytest.raises(ValueError):
            # Via parametrized fixtures
            sf.add_feat_value(split=corrupted_list_splits, dict_scale=dict_scale, df_parts=df_parts[parts])

    def test_corrupted_dict_scale(self, df_seq, list_parts, list_splits, df_scales):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        parts, split, dict_scale = list_parts[0], list_splits[0], df_scales.iloc[:, 0].to_dict()
        dict_scale1 = dict_scale.copy()
        dict_scale1["A"] = "A"
        dict_scale2 = dict_scale.copy()
        dict_scale2.pop("A")
        dict_scale3 = dict_scale.copy()
        dict_scale3["A"] = dict
        wrong_dict_scales = [dict(A=1, B=np.NaN), dict(a=0), dict_scale1, dict_scale2, dict_scale3, dict_scale3]
        for d in wrong_dict_scales:
            with pytest.raises(ValueError):
                sf.add_feat_value(split=split, dict_scale=d, df_parts=df_parts[parts])

    def test_corrupted_df_parts(self, list_splits, df_scales, corrupted_df_parts):
        sf = aa.SequenceFeature()
        split, dict_scale = list_splits[0], df_scales.iloc[:, 0].to_dict()
        with pytest.raises(ValueError):
            # Via parametrized fixtures
            sf.add_feat_value(split=split, dict_scale=dict_scale, df_parts=corrupted_df_parts)


class TestFeatureMatrix:
    """Unit tests for getting feature matrix"""

    # Positive unit test
    def test_feature_matrix(self, df_seq, df_scales):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        features = sf.get_features()[0:100]
        feat_matrix = sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=features)
        assert isinstance(feat_matrix, np.ndarray)
        assert feat_matrix.shape == (len(df_seq), len(features))
        feat_matrix = sf.feat_matrix(df_parts=df_parts, features=features)
        assert isinstance(feat_matrix, np.ndarray)

    # Negative test
    def test_missing_parameters(self, df_scales, df_seq):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        features = sf.get_features()[0:100]
        with pytest.raises(ValueError):
            sf.feat_matrix(df_parts=df_parts)
        with pytest.raises(ValueError):
            sf.feat_matrix(features=features)
        with pytest.raises(ValueError):
            sf.feat_matrix(df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.feat_matrix(df_parts=df_parts, df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.feat_matrix(df_scales=df_scales, features=features)

    def test_wrong_input(self, df_cat, df_seq, df_scales):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        features = sf.get_features()[0:100]
        list_wrong_input = [1, -1, "TMD", ["TMD"], None, [1, 2], ["aa", "a"],
                            [["tmd", "tmd_e"]], df_cat, [df_cat, df_seq], dict(a=1)]
        for wrong_input in list_wrong_input:
            with pytest.raises(ValueError):
                sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=wrong_input)
            if wrong_input is not None:
                with pytest.raises(ValueError):
                    sf.feat_matrix(df_parts=df_parts, df_scales=wrong_input, features=features)
            with pytest.raises(ValueError):
                sf.feat_matrix(df_parts=wrong_input, df_scales=df_scales, features=features)

    def test_corrupted_df_parts(self, corrupted_df_parts, df_scales):
        sf = aa.SequenceFeature()
        features = sf.get_features()[0:100]
        with pytest.raises(ValueError):
            # Via parametrized fixtures
            sf.feat_matrix(df_parts=corrupted_df_parts, df_scales=df_scales, features=features)

    def test_corrupted_df_scales(self, corrupted_df_scales, df_seq):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        features = sf.get_features()[0:100]
        with pytest.raises(ValueError):
            # Via parametrized fixtures
            sf.feat_matrix(df_parts=df_parts, df_scales=corrupted_df_scales, features=features)

    def test_corrupted_features(self, df_scales, df_seq):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
        features = sf.get_features()[0:100]
        corrupted_features = [features[0:5] + [np.NaN], features[0:3] + ["Test"],
                              "a",
                              [[features[0:4]]],
                              [x.upper() for x in features[0:5]],
                              [x[0:5] for x in features[0:5]],
                              ["a".join(x.split("-")) for x in features[0:6]]]
        for features in corrupted_features:
            with pytest.raises(ValueError):
                sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=features)


# II Regression test (Functional test)
def test_sequence_feature(list_splits):
    """Positive regression/functional test of all aa.SequenceFeature() methods"""
    sf = aa.SequenceFeature()
    # Get test set of sequences
    df_seq = aa.load_dataset()
    # Get feature components
    df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=False)
    df_scales = aa.load_scales()
    split_kws = sf.get_split_kws()
    # Get features (names, values, matrix)
    features = sf.get_features()[0:100]
    feat_matrix = sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=features)
    assert isinstance(feat_matrix, np.ndarray)
    assert feat_matrix.shape == (len(df_seq), len(features))

