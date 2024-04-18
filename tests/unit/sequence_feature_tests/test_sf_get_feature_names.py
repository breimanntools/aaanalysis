"""This is a script to test the SequenceFeature().get_feature_names() method ."""
from hypothesis import given, settings, strategies as st
import pytest
import random
import aaanalysis as aa
aa.options["verbose"] = False

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


def get_random_features(n_feat=100):
    """"""
    sf = aa.SequenceFeature()
    features = sf.get_features()
    return random.sample(features, n_feat)


class TestGetFeatureNames:
    """Class for testing get_feature_names function in positive scenarios."""

    def test_valid_features(self):
        """Test valid 'features' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_names(features=features)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    def test_valid_df_cat(self):
        """Test valid 'df_cat' DataFrame input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        df_cat = aa.load_scales(name="scales_cat")
        result = sf.get_feature_names(features=features, df_cat=df_cat)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    @settings(max_examples=5, deadline=1000)
    @given(start=st.integers(min_value=1))
    def test_valid_start(self, start):
        """Test valid 'start' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_names(features=features, start=start)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    @settings(max_examples=5, deadline=1000)
    @given(tmd_len=st.integers(min_value=20, max_value=2000))
    def test_valid_tmd_len(self, tmd_len):
        """Test valid 'tmd_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_names(features=features, tmd_len=tmd_len)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    @settings(max_examples=5, deadline=1000)
    @given(jmd_c_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_c_len(self, jmd_c_len):
        """Test valid 'jmd_c_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_names(features=features, jmd_c_len=jmd_c_len)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    @settings(max_examples=5, deadline=1000)
    @given(jmd_n_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_n_len(self, jmd_n_len):
        """Test valid 'jmd_n_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_names(features=features, jmd_n_len=jmd_n_len)
        assert isinstance(result, list) and all(isinstance(name, str) for name in result)

    # Negative Tests
    def test_invalid_features(self):
        """Negative test for invalid 'features' input."""
        sf = aa.SequenceFeature()
        invalid_features = [None, 123, "invalid_input", {}]
        for features in invalid_features:
            with pytest.raises(ValueError):
                sf.get_feature_names(features=features)

    def test_invalid_df_cat(self):
        """Negative test for invalid 'df_cat' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        invalid_df_cats = [123, "invalid_input", [], {}]
        for df_cat in invalid_df_cats:
            with pytest.raises(ValueError):
                sf.get_feature_names(features=features, df_cat=df_cat)

    @settings(max_examples=5, deadline=1000)
    @given(start=st.one_of(st.none(), st.text(), st.floats()))
    def test_invalid_start(self, start):
        """Negative test for invalid 'start' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_names(features=features, start=start)

    @settings(max_examples=5, deadline=1000)
    @given(tmd_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_tmd_len(self, tmd_len):
        """Negative test for invalid 'tmd_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_names(features=features, tmd_len=tmd_len)

    @settings(max_examples=5, deadline=1000)
    @given(jmd_c_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_jmd_c_len(self, jmd_c_len):
        """Negative test for invalid 'jmd_c_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_names(features=features, jmd_c_len=jmd_c_len)

    @settings(max_examples=5, deadline=1000)
    @given(jmd_n_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_jmd_n_len(self, jmd_n_len):
        """Negative test for invalid 'jmd_n_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_names(features=features, jmd_n_len=jmd_n_len)