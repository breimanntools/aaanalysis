"""
This script tests the CPPPlot().ranking() method.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import aaanalysis as aa


# Setup and helper functions
def create_df_feat(num_features=50):
    """Creates a dummy DataFrame to mimic df_feat input."""
    df_feat = aa.load_features()
    return df_feat.head(num_features)


# Test Class for Normal Cases
class TestRanking:
    """Test class for the ranking method, focusing on individual parameters."""

    # Positive tests
    def test_df_feat(self):
        df_feat = create_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=df_feat)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(n_top=st.integers(min_value=2, max_value=20))
    def test_n_top(self, n_top):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=n_top)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    def test_shap_plot(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        values = np.array([1] * len(df_feat))
        df_feat.insert(0, "feat_impact", values)
        for shap_plot in [True, False]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, shap_plot=shap_plot)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=2000)
    @given(figsize=st.tuples(st.integers(5, 15), st.integers(5, 15)))
    def test_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig, axes = cpp_plot.ranking(df_feat=df_feat, figsize=figsize)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(tmd_len=st.integers(min_value=1, max_value=100))
    def test_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, tmd_len=tmd_len)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(tmd_jmd_space=st.integers(min_value=1, max_value=10))
    def test_tmd_jmd_space(self, tmd_jmd_space):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, tmd_jmd_space=tmd_jmd_space)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(tmd_color=st.sampled_from(["blue", "green", "red", "yellow"]))
    def test_tmd_color(self, tmd_color):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, tmd_color=tmd_color)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(jmd_color=st.sampled_from(["blue", "green", "red", "yellow"]))
    def test_jmd_color(self, jmd_color):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, jmd_color=jmd_color)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(tmd_jmd_alpha=st.floats(min_value=0, max_value=1))
    def test_tmd_jmd_alpha(self, tmd_jmd_alpha):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, tmd_jmd_alpha=tmd_jmd_alpha)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()


    def test_name_test(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        for name_test in ["Test1", "asdfadsfasdf", "erf"]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, name_test=name_test)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    def test_name_ref(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        for name_ref in ["Test1", "asdfadsfasdf", "erf"]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, name_ref=name_ref)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(fontsize_titles=st.one_of(st.none(), st.integers(min_value=5, max_value=20)))
    def test_fontsize_titles(self, fontsize_titles):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, fontsize_titles=fontsize_titles)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(fontsize_labels=st.one_of(st.none(), st.integers(min_value=5, max_value=20)))
    def test_fontsize_labels(self, fontsize_labels):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, fontsize_labels=fontsize_labels)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(fontsize_annotations=st.one_of(st.none(), st.integers(min_value=5, max_value=20)))
    def test_fontsize_annotations(self, fontsize_annotations):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, fontsize_annotations=fontsize_annotations)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()


    @settings(max_examples=3, deadline=1500)
    @given(xlim_dif=st.tuples(st.floats(min_value=-10, max_value=0), st.floats(min_value=0, max_value=10)))
    def test_xlim_dif(self, xlim_dif):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if xlim_dif[0] < xlim_dif[1]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, xlim_dif=xlim_dif)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(xlim_rank=st.tuples(st.floats(min_value=0, max_value=50), st.floats(min_value=51, max_value=100)))
    def test_xlim_rank(self, xlim_rank):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if xlim_rank[0] < xlim_rank[1]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, xlim_rank=xlim_rank)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=1500)
    @given(x_rank_info=st.one_of(st.none(), st.floats(min_value=0, max_value=100)))
    def test_x_rank_info(self, x_rank_info):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, x_rank_info=x_rank_info)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    # Negative Test
    def test_invalid_df_feat(self):
        cpp_plot = aa.CPPPlot()
        invalid_df_feat = "invalid_data"  # Non-DataFrame input
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=invalid_df_feat)
            plt.close()


    def test_invalid_n_top(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, n_top=None)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, n_top="str")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, n_top=-3)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, n_top=1000)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, n_top=True)


    def test_invalid_figsize(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, figsize=(1, "sr"))
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, figsize="invalid")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, figsize="")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, figsize=(-1, 5))

    @settings(max_examples=20, deadline=1500)
    @given(tmd_len=st.integers(max_value=0))
    def test_invalid_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_len=tmd_len)

    @settings(max_examples=20, deadline=1500)
    @given(xlim_dif=st.tuples(st.floats(min_value=0), st.floats(max_value=-101)))
    def test_invalid_xlim_dif(self, xlim_dif):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if xlim_dif[0] > xlim_dif[1]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, xlim_dif=xlim_dif)

    @settings(max_examples=10, deadline=1500)
    @given(xlim_rank=st.tuples(st.floats(min_value=10), st.floats(max_value=0)))
    def test_invalid_xlim_rank(self, xlim_rank):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if xlim_rank[0] > xlim_rank[1]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, xlim_rank=xlim_rank)

    def test_invalid_x_rank_info(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, x_rank_info=-1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, x_rank_info="str")

    @settings(max_examples=20, deadline=1500)
    @given(tmd_jmd_space=st.one_of(st.integers(max_value=0), st.floats(allow_nan=True)))
    def test_invalid_tmd_jmd_space(self, tmd_jmd_space):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if tmd_jmd_space < 0:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, tmd_jmd_space=tmd_jmd_space)

    def test_invalid_tmd_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_color="sr")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_color="tab:yellow")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_color=1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_color=None)


    def test_invalid_jmd_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, jmd_color="sr")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, jmd_color="tab:yellow")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, jmd_color=1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, jmd_color=None)

    @settings(max_examples=20, deadline=1500)
    @given(tmd_jmd_alpha=st.one_of(st.floats(max_value=-0.01), st.floats(min_value=1.01), st.text()))
    def test_invalid_tmd_jmd_alpha(self, tmd_jmd_alpha):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_jmd_alpha=tmd_jmd_alpha)

    def test_invalid_name_test(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_test=1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_test=None)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_test=["str"])


    def test_invalid_name_ref(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_ref=1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_ref=None)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, name_ref=["str"])

    @settings(max_examples=20, deadline=1500)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_titles(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_titles=fontsize)

    @settings(max_examples=20, deadline=1500)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_labels(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_labels=fontsize)

    @settings(max_examples=20, deadline=1500)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_annotations(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_annotations=fontsize)


# Test Class for Complex Cases
class TestRankingComplex:
    """Test class for the ranking method, focusing on combinations of parameters."""

    # Positive Complex Test
    def test_complex_valid(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        values = np.array([1] * len(df_feat))
        df_feat.insert(0, "feat_impact", values)

        # Complex valid combination of parameters
        n_top = 10
        figsize = (10, 8)
        tmd_len = 25
        tmd_jmd_space = 5
        tmd_color = "green"
        jmd_color = "blue"
        tmd_jmd_alpha = 0.5
        name_test = "TestDataset"
        name_ref = "RefDataset"
        fontsize_titles = 12
        fontsize_labels = 11
        fontsize_annotations = 10
        xlim_dif = (-20, 20)
        xlim_rank = (1, 10)
        x_rank_info = 5

        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=n_top, figsize=figsize,
                                     tmd_len=tmd_len, tmd_jmd_space=tmd_jmd_space,
                                     tmd_color=tmd_color, jmd_color=jmd_color,
                                     tmd_jmd_alpha=tmd_jmd_alpha, name_test=name_test,
                                     name_ref=name_ref, fontsize_titles=fontsize_titles,
                                     fontsize_labels=fontsize_labels,
                                     fontsize_annotations=fontsize_annotations,
                                     xlim_dif=xlim_dif, xlim_rank=xlim_rank,
                                     x_rank_info=x_rank_info)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()