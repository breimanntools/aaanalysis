"""This is a script to test the SequenceFeature().get_feature() method ."""
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import numpy as np
import random
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False


class TestGetFeatures:
    """Positive tests for the get_features method."""

    # Positive tests
    def test_default(self):
        sf = aa.SequenceFeature()
        result = sf.get_features()
        assert isinstance(result, list)
        assert all(isinstance(feature, str) for feature in result)

    @settings(max_examples=5, deadline=1000)
    @given(list_parts=st.lists(st.sampled_from(['tmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c']), min_size=1))
    def test_valid_list_parts(self, list_parts):
        sf = aa.SequenceFeature()
        result = sf.get_features(list_parts=list_parts)
        assert isinstance(result, list)
        assert all(isinstance(feature, str) for feature in result)

    def test_valid_all_parts(self):
        sf = aa.SequenceFeature()
        result = sf.get_features(all_parts=True)
        assert isinstance(result, list)
        sf = aa.SequenceFeature()
        result = sf.get_features(all_parts=False)
        assert isinstance(result, list)

    def test_valid_split_kws(self):
        sf = aa.SequenceFeature()
        for i in range(2, 10):
            split_kws = sf.get_split_kws(n_split_min=i, n_split_max=i+1)
            result = sf.get_features(split_kws=split_kws)
            assert isinstance(result, list)
            split_kws = sf.get_split_kws(n_min=i, n_max=i + 1)
            result = sf.get_features(split_kws=split_kws)
            assert isinstance(result, list)
        for steps_pattern in [[2,3], [3,4,5], [4, 6], [5,3]]:
            split_kws = sf.get_split_kws(steps_pattern=steps_pattern)
            result = sf.get_features(split_kws=split_kws)
            assert isinstance(result, list)
        for split_types in ["Segment", "Pattern", "PeriodicPattern"]:
            split_kws = sf.get_split_kws(split_types=split_types)
            result = sf.get_features(split_kws=split_kws)
            assert isinstance(result, list)

    def test_valid_list_scales(self):
        sf = aa.SequenceFeature()
        for name in ["scales", "scales_raw", "scales_pc"]:
            list_scales = aa.load_scales(name=name)
            result = sf.get_features(list_scales=list(list_scales))
            assert isinstance(result, list)
        list_scales = pd.DataFrame({"scale1": [1, 2], "scale2": [2, 3]})
        result = sf.get_features(list_scales=list(list_scales))
        assert isinstance(result, list)

    # Negative tests
    def test_invalid_list_parts(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(list_parts="invalid_type")
        with pytest.raises(ValueError):
            sf.get_features(list_parts=123)
        with pytest.raises(ValueError):
            sf.get_features(list_parts=[1, 2, 3])
        with pytest.raises(ValueError):
            sf.get_features(list_parts=[])

    def test_invalid_all_parts(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(all_parts="invalid_type")
        with pytest.raises(ValueError):
            sf.get_features(all_parts=123)
        with pytest.raises(ValueError):
            sf.get_features(all_parts=[True, False])

    def test_invalid_split_kws(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(split_kws="invalid_type")
        with pytest.raises(ValueError):
            sf.get_features(split_kws=123)
        with pytest.raises(ValueError):
            sf.get_features(split_kws={"wrong_key": "wrong_value"})

    def test_invalid_list_scales(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_features(list_scales="invalid_type")
        with pytest.raises(ValueError):
            sf.get_features(list_scales=123)
        with pytest.raises(ValueError):
            sf.get_features(list_scales=[1, 2, None])



