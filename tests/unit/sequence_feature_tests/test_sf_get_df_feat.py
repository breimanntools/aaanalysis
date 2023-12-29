"""This is a script to test the SequenceFeature().get_df_feat() method ."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import pandas as pd
import aaanalysis as aa
import random
aa.options["verbose"] = False


def _get_df_feat(n_feat=10, n_samples=20):
    """"""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(n_feat)
    features = df_feat["feature"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)
    return df_feat


class TestGetDfFeat:
    """Test the get_df_feat method for individual parameters."""

    # Positive tests
    def test_valid_features(self):
        """Test valid 'features' inputs."""
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=1000)
    @given(list_parts=some.lists(some.sampled_from(
        ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n', 'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c',
         'ext_n_tmd_n', 'tmd_c_ext_c']), min_size=1))
    def test_valid_df_parts(self, list_parts):
        """Test valid 'df_parts' DataFrame inputs."""
        sf = aa.SequenceFeature()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_feat = aa.load_features(name="DOM_GSEC").head(50)
        features = df_feat["feature"].to_list()
        # Feature parts must always be covered
        list_feat_parts = list(set([x.split("-")[0].lower() for x in features]))
        list_parts += list_feat_parts
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts)
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels)
        assert isinstance(df_feat, pd.DataFrame)

    """
    @settings(max_examples=10)
    @given(labels=some.lists(some.integers(0, 1), min_size=1))
    def test_valid_labels(self, labels):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(labels=labels)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=10)
    @given(df_scales=some.data_frames())
    def test_valid_df_scales(self, df_scales):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_scales=df_scales)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=10)
    @given(df_cat=some.data_frames())
    def test_valid_df_cat(self, df_cat):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_cat=df_cat)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=10)
    @given(start=some.integers(min_value=1))
    def test_valid_start(self, start):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(start=start)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=10)
    @given(accept_gaps=some.booleans())
    def test_accept_gaps(self, accept_gaps):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(accept_gaps=accept_gaps)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=10)
    @given(parametric=some.booleans())
    def test_parametric(self, parametric):
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(parametric=parametric)
        assert isinstance(df_feat, pd.DataFrame)

    # Negative test
    """
