"""
This is a script for testing the aa.dPULearn.compare_sets_negatives() method.
"""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import pandas as pd
import numpy as np
import random
import aaanalysis as aa



def _create_list_labels(size, num_labels):
    return [np.random.choice([0, 1, 2], size=size) for _ in range(num_labels)]


def _create_names(num_names):
    return ['name_' + str(i) for i in range(num_names)]


class TestCompareSetsNegatives:
    """Positive test cases for compare_sets_negatives function for each parameter individually."""

    # Positive tests
    @settings(max_examples=10)
    @given(size=st.integers(min_value=1, max_value=100), num_labels=st.integers(min_value=1, max_value=10))
    def test_list_labels_valid(self, size, num_labels):
        list_labels = _create_list_labels(size, num_labels)
        result = aa.dPULearn.compare_sets_negatives(list_labels)
        assert isinstance(result, pd.DataFrame)

    @settings(max_examples=10)
    @given(num_names=st.integers(min_value=1, max_value=10))
    def test_names_valid(self, num_names):
        names = _create_names(num_names)
        list_labels = _create_list_labels(100, num_names)
        result = aa.dPULearn.compare_sets_negatives(list_labels=list_labels, names_datasets=names)
        assert isinstance(result, pd.DataFrame)

    def test_df_seq_valid(self):
        df_info = aa.load_dataset()
        list_name = df_info["Dataset"].to_list()
        # Test all benchmark datasets
        for name in list_name:
            df_seq = aa.load_dataset(name=name, n=50)
            list_labels =  _create_list_labels(len(df_seq), 5)
            result = aa.dPULearn.compare_sets_negatives(list_labels=list_labels, df_seq=df_seq)
            assert isinstance(result, pd.DataFrame)

    def test_return_upset_data_valid(self):
        list_labels = _create_list_labels(100, 1)
        upset_data = aa.dPULearn.compare_sets_negatives(list_labels=list_labels, return_upset_data=True)
        assert isinstance(upset_data, pd.Series)
        df_neg_comp = aa.dPULearn.compare_sets_negatives(list_labels=list_labels, return_upset_data=False)
        assert isinstance(df_neg_comp, pd.DataFrame)

    # Negative tests
    @settings(max_examples=10)
    @given(size=st.integers(min_value=1, max_value=100), num_labels=st.integers(min_value=1, max_value=10))
    def test_list_labels_invalid(self, size, num_labels):
        list_labels = _create_list_labels(size, num_labels)
        list_labels.append(None)  # Adding an invalid entry
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels)

    # Negative tests for names with empty list or None
    @settings(max_examples=10)
    @given(num_names=st.integers(min_value=1, max_value=10))
    def test_names_invalid(self, num_names):
        names = _create_names(num_names)
        list_labels = _create_list_labels(100, num_names+1)
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, names_datasets=names)
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, names_datasets="wrong")

    def test_df_seq_invalid(self):
        df_seq = pd.DataFrame({'invalid_column': ['a', 'b', 'c']})  # Non-numeric DataFrame
        list_labels = _create_list_labels(3, 1)
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, df_seq=df_seq)

    def test_return_upset_data_invalid(self):
        list_labels = _create_list_labels(100, 1)
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, return_upset_data='not_a_boolean')  # Invalid boolean value


class TestCompareSetsNegativesComplex:
    """Complex test cases for compare_sets_negatives function combining multiple parameters."""

    # Positive tests
    @settings(max_examples=5)
    @given(num_labels=st.integers(min_value=1, max_value=10),
           return_upset_data=st.booleans())
    def test_complex_valid_combinations(self, num_labels, return_upset_data):
        df_info = aa.load_dataset()
        list_name = df_info["Dataset"].to_list()
        random_dataset_name = random.choice(list_name)
        # Test with the randomly selected dataset
        df_seq = aa.load_dataset(name=random_dataset_name, n=50)
        list_labels = _create_list_labels(len(df_seq), num_labels)
        names = _create_names(num_labels)
        result = aa.dPULearn.compare_sets_negatives(list_labels=list_labels, names_datasets=names,
                                                    df_seq=df_seq, return_upset_data=return_upset_data)
        assert isinstance(result, pd.DataFrame) or isinstance(result, pd.Series)

    # Negative tests
    @settings(max_examples=5)
    @given(size=st.integers(min_value=1, max_value=100),
           num_labels=st.integers(min_value=1, max_value=10))
    def test_complex_invalid_mismatch(self, size, num_labels):
        list_labels = _create_list_labels(size, num_labels)
        names = _create_names(num_labels + 1)  # One extra name to create a mismatch
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, names_datasets=names)

    @settings(max_examples=5)
    @given(size=st.integers(min_value=1, max_value=100),
           num_labels=st.integers(min_value=1, max_value=10))
    def test_complex_invalid_df_seq(self, size, num_labels):
        list_labels = _create_list_labels(size, num_labels)
        df_seq = pd.DataFrame({'invalid_column': ['a'] * size})  # Non-numeric DataFrame
        with pytest.raises(ValueError):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, df_seq=df_seq)