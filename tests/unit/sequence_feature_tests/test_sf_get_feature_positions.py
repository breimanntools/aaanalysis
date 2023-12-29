"""This is a script to test the SequenceFeature().get_feature_positions() method ."""
from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
import random
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False

def get_random_features(n_feat=100):
    """"""
    sf = aa.SequenceFeature()
    features = sf.get_features()
    return random.sample(features, n_feat)

SEQ_TMD = "A" * 20
SEQ_JMD = "B" * 10

class TestGetFeaturePositions:
    """Class for testing get_feature_positions function in positive scenarios."""

    def test_valid_features(self):
        sf = aa.SequenceFeature()
        for i in range(5):
            features = get_random_features()
            result = sf.get_feature_positions(features=features)
            assert isinstance(result, list) and len(result) == len(features)

    @settings(max_examples=10, deadline=1000)
    @given(start=st.integers(min_value=0, max_value=2000))
    def test_valid_start(self, start):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, start=start)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(tmd_len=st.integers(min_value=20, max_value=2000))
    def test_valid_tmd_len(self, tmd_len):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, tmd_len=tmd_len)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(jmd_n_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_n_len(self, jmd_n_len):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, jmd_n_len=jmd_n_len)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(jmd_c_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_c_len(self, jmd_c_len):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, jmd_c_len=jmd_c_len)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(tmd_seq=st.text(min_size=20, max_size=2000))
    def test_valid_tmd_seq(self, tmd_seq):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, tmd_seq=tmd_seq,
                                          jmd_c_seq=SEQ_JMD, jmd_n_seq=SEQ_JMD)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(jmd_n_seq=st.text(min_size=10, max_size=2000))
    def test_valid_jmd_n_seq(self, jmd_n_seq):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, jmd_n_seq=jmd_n_seq,
                                          tmd_seq=SEQ_TMD, jmd_c_seq=SEQ_JMD)
        assert isinstance(result, list)

    @settings(max_examples=10, deadline=1000)
    @given(jmd_c_seq=st.text(min_size=10, max_size=2000))
    def test_valid_jmd_c_seq(self, jmd_c_seq):
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, jmd_c_seq=jmd_c_seq,
                                          tmd_seq=SEQ_TMD, jmd_n_seq=SEQ_JMD)
        assert isinstance(result, list)

class TestGetFeaturePositionsComplex:
    """Class for testing get_feature_positions function in complex scenarios."""

    # Positive Complex Tests
    def test_valid_complex_case_1(self):
        """Test valid combination of parameters."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, start=10, tmd_len=25, jmd_n_len=15, jmd_c_len=15)
        assert isinstance(result, list) and len(result) == len(features)

    def test_valid_complex_case_2(self):
        """Test valid combination with sequences."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_positions(features=features, start=5, tmd_seq=SEQ_TMD, jmd_n_seq=SEQ_JMD, jmd_c_seq=SEQ_JMD)
        assert isinstance(result, list) and len(result) == len(features)

    # Negative Complex Tests
    def test_invalid_complex_case_1(self):
        """Test invalid combination due to mismatched lengths."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_positions(features=features, start=3, tmd_len=5, jmd_n_len=-1, jmd_c_len=20)

    def test_invalid_complex_case_2(self):
        """Test invalid combination with incorrect sequence types."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_positions(features=features, tmd_seq=123, jmd_n_seq=[], jmd_c_seq={})
