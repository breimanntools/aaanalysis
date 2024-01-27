"""
This script tests the CPPPlot().update_seq_size() method.
"""
import matplotlib.pyplot as plt
import hypothesis.strategies as st
from hypothesis import given, settings
import pytest
import aaanalysis as aa

class TestUpdateSeqSize:
    """Test class for update_seq_size method, focusing on individual parameters."""

    # Positive tests
    @settings(max_examples=10)
    @given(ax=st.just(plt.subplots()[1]))
    def test_ax_positive(self, ax):
        cpp_plot = aa.CPPPlot()
        result = cpp_plot.update_seq_size(ax)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(fig=st.just(plt.figure()))
    def test_fig_positive(self, fig):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, fig=fig)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(tmd_seq=st.text(min_size=1, max_size=20))
    def test_tmd_seq_positive(self, tmd_seq):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, tmd_seq=tmd_seq)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(jmd_n_seq=st.text(min_size=1, max_size=20))
    def test_jmd_n_seq_positive(self, jmd_n_seq):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, jmd_n_seq=jmd_n_seq)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(jmd_c_seq=st.text(min_size=1, max_size=20))
    def test_jmd_c_seq_positive(self, jmd_c_seq):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, jmd_c_seq=jmd_c_seq)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(max_x_dist=st.floats(min_value=0, max_value=10))
    def test_max_x_dist_positive(self, max_x_dist):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, max_x_dist=max_x_dist)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(fontsize_tmd_jmd=st.one_of(st.none(), st.integers(min_value=1, max_value=50)))
    def test_fontsize_tmd_jmd_positive(self, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, fontsize_tmd_jmd=fontsize_tmd_jmd)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(weight_tmd_jmd=st.sampled_from(['normal', 'bold']))
    def test_weight_tmd_jmd_positive(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, weight_tmd_jmd=weight_tmd_jmd)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(tmd_color=st.text(min_size=1))
    def test_tmd_color_positive(self, tmd_color):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, tmd_color=tmd_color)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(jmd_color=st.text(min_size=1))
    def test_jmd_color_positive(self, jmd_color):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, jmd_color=jmd_color)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(tmd_seq_color=st.text(min_size=1))
    def test_tmd_seq_color_positive(self, tmd_seq_color):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, tmd_seq_color=tmd_seq_color)
        assert isinstance(result, plt.Axes)

    @settings(max_examples=10)
    @given(jmd_seq_color=st.text(min_size=1))
    def test_jmd_seq_color_positive(self, jmd_seq_color):
        cpp_plot = aa.CPPPlot()
        ax = plt.subplots()[1]
        result = cpp_plot.update_seq_size(ax, jmd_seq_color=jmd_seq_color)
        assert isinstance(result, plt.Axes)
