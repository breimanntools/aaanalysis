"""This is a script to test the SequenceFeature().get_df_feat() method ."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import pandas as pd
import aaanalysis as aa
import random
aa.options["verbose"] = False


def _get_df_feat_input(n_feat=10, n_samples=20, list_parts=None):
    """Create input for sf.get_df_feat()"""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(n_feat)
    features = df_feat["feature"].to_list()
    sf = aa.SequenceFeature()
    if list_parts is not None:
        list_feat_parts = list(set([x.split("-")[0].lower() for x in features]))
        list_parts += list_feat_parts
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts)
    else:
        df_parts = sf.get_df_parts(df_seq=df_seq)
    return features, df_parts, labels



class TestGetDfFeat:
    """Test the get_df_feat method for individual parameters."""

    # Positive tests
    def test_valid_features(self):
        """Test valid 'features' inputs."""
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)
            assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=1000)
    @given(list_parts=some.lists(some.sampled_from(
        ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n', 'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c',
         'ext_n_tmd_n', 'tmd_c_ext_c']), min_size=1))
    def test_valid_df_parts(self, list_parts):
        """Test valid 'df_parts' DataFrame inputs."""
        features, df_parts, labels = _get_df_feat_input(n_feat=10, n_samples=50, list_parts=list_parts)
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_labels(self):
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels)
            assert isinstance(df_feat, pd.DataFrame)
            labels = [5 if i == 0 else i for i in labels]
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, label_ref=5)
            assert isinstance(df_feat, pd.DataFrame)
            labels = [10 if i == 1 else i for i in labels]
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, label_ref=5, label_test=10)
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_df_scales(self):
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            df_scales = aa.load_scales()
            sf = aa.SequenceFeature()
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels,
                                     df_scales=df_scales)
            assert isinstance(df_feat, pd.DataFrame)
            scales = list(set([x.split("-")[2] for x in features]))
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels,
                                     df_scales=df_scales[scales])
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_df_cat(self):
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            df_cat = aa.load_scales(name="scales_cat")
            sf = aa.SequenceFeature()
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, df_cat=df_cat)
            assert isinstance(df_feat, pd.DataFrame)
            scales = list(set([x.split("-")[2] for x in features]))
            _df_cat = df_cat[df_cat["scale_id"].isin(scales)]
            df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, df_cat=df_cat)
            assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=1000)
    @given(start=some.integers(min_value=1))
    def test_valid_start(self, start):
        n_feat = random.randint(5, 100)
        n_samples = random.randint(5, 50)
        features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, start=start)
        assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=1000)
    @given(tmd_len=some.integers(min_value=15, max_value=100),
           jmd_n_len=some.integers(min_value=10, max_value=100),
           jmd_c_len=some.integers(min_value=10, max_value=100))
    def test_valid_tmd_jmd_len(self, tmd_len, jmd_n_len, jmd_c_len):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels,
                                 tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        assert isinstance(df_feat, pd.DataFrame)

    def test_min_tmd_jmd_len(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels,
                                 tmd_len=30, jmd_n_len=0, jmd_c_len=0)
        assert isinstance(df_feat, pd.DataFrame)

    def test_accept_gaps(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, accept_gaps=True)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, accept_gaps=False)
        assert isinstance(df_feat, pd.DataFrame)

    def test_parametric(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, parametric=True)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, parametric=False)
        assert isinstance(df_feat, pd.DataFrame)

    def test_n_jobs(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, n_jobs=2)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, n_jobs=None)
        assert isinstance(df_feat, pd.DataFrame)

    # Negative test
    def test_invalid_features(self):
        """Test with invalid 'features' inputs."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception):
            sf.get_df_feat(features=None, df_parts=df_parts, labels=labels)

    def test_invalid_df_parts(self):
        """Test with invalid 'df_parts' DataFrame inputs."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=None, features=features, labels=labels)

    def test_invalid_labels(self):
        """Test with invalid 'labels' inputs."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=None)
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels*10)
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels + [2])

    def test_invalid_df_scales(self):
        """Test with invalid 'df_scales' inputs."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, df_scales=pd.DataFrame())

    def test_invalid_df_cat(self):
        """Test with invalid 'df_cat' inputs."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, df_cat=pd.DataFrame())

    def test_invalid_start(self):
        """Test with invalid 'start' parameter."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        for i in ["str", None, ]:
            with pytest.raises(Exception):
                sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, start=i)

    def test_invalid_lengths(self):
        """Test with invalid length parameters."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        with pytest.raises(Exception): 
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, tmd_len=-1, jmd_n_len=-1, jmd_c_len=-1)
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, tmd_len="str")
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, tmd_len=0)
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, tmd_len=None)
        with pytest.raises(Exception):
            sf.get_df_feat(df_parts=df_parts, features=features, labels=labels, tmd_len=3, jmd_n_len=0, jmd_c_len=0)


class TestGetDfFeatComplexPositive:
    """Complex positive tests for the get_df_feat method."""

    @settings(max_examples=5, deadline=1000)
    @given(
        tmd_len=some.integers(min_value=15, max_value=30),
        jmd_n_len=some.integers(min_value=10, max_value=15),
        jmd_c_len=some.integers(min_value=10, max_value=15),
        parametric=some.booleans(),
        accept_gaps=some.booleans()
    )
    def test_complex_positive(self, tmd_len, jmd_n_len, jmd_c_len, parametric, accept_gaps):
        """Test with a complex combination of valid parameters."""
        n_feat = random.randint(5, 100)
        n_samples = random.randint(5, 50)
        features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
        sf = aa.SequenceFeature()
        df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels,
                                 tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                 parametric=parametric, accept_gaps=accept_gaps)
        assert isinstance(df_feat, pd.DataFrame)

    def test_complex_negative(self):
        """Test with invalid combinations of parameters."""
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        # Test with invalid 'tmd_len', 'jmd_n_len', 'jmd_c_len'
        with pytest.raises(ValueError):
            sf.get_df_feat(features=features, df_parts=df_parts, labels=labels, tmd_len=-5, jmd_n_len=-1, jmd_c_len=-1)
        # Test with invalid 'labels'
        with pytest.raises(ValueError):
            invalid_labels = [999] * len(labels)  # Invalid label values
            sf.get_df_feat(features=features, df_parts=df_parts, labels=invalid_labels)
        # Test with mismatched features and df_parts
        with pytest.raises(Exception):  # Replace with specific exception if applicable
            sf.get_df_feat(features=[], df_parts=df_parts, labels=labels)  # Empty features list