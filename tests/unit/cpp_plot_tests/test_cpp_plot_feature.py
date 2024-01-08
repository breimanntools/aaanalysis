"""
This script tests the feature method for plotting CPP feature distributions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import aaanalysis as aa

# Helper functions and common setups
def create_valid_df_seq(n_samples=100, n_seq_info=10):
    return pd.DataFrame(np.random.rand(n_samples, n_seq_info), columns=[f'col{i}' for i in range(n_seq_info)])

def create_valid_labels(n_samples=100):
    return np.random.randint(0, 2, size=n_samples)

class TestFeature:
    """
    Test class for feature method, focusing on individual parameters.
    """

    @settings(max_examples=10)
    @given(feature=st.text(min_size=1))
    def test_feature_input(self, feature):
        cpp_plot = aa.CPPPlot()
        df_seq = create_valid_df_seq()
        labels = create_valid_labels()
        ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels)
        assert isinstance(ax, plt.Axes)

    @settings(max_examples=10)
    @given(df_seq=st.lists(st.floats(), min_size=1, max_size=100))
    def test_df_seq_input(self, df_seq):
        cpp_plot = aa.CPPPlot()
        labels = create_valid_labels(n_samples=len(df_seq))
        ax = cpp_plot.feature(feature='test_feature', df_seq=pd.DataFrame(df_seq), labels=labels)
        assert isinstance(ax, plt.Axes)

    @settings(max_examples=10)
    @given(labels=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=100))
    def test_labels_input(self, labels):
        cpp_plot = aa.CPPPlot()
        df_seq = create_valid_df_seq(n_samples=len(labels))
        ax = cpp_plot.feature(feature='test_feature', df_seq=df_seq, labels=np.array(labels))
        assert isinstance(ax, plt.Axes)

    @settings(max_examples=10)
    @given(label_test=st.integers(), label_ref=st.integers())
    def test_label_test_ref_input(self, label_test, label_ref):
        cpp_plot = aa.CPPPlot()
        df_seq = create_valid_df_seq()
        labels = np.random.choice([label_test, label_ref], size=df_seq.shape[0])
        ax = cpp_plot.feature(feature='test_feature', df_seq=df_seq, labels=labels,
                              label_test=label_test, label_ref=label_ref)
        assert isinstance(ax, plt.Axes)
