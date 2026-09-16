"""
This script tests the CPPPlot().ranking() method.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import aaanalysis as aa
import random
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Setup and helper functions
def create_df_feat(num_features=50):
    """Creates a dummy DataFrame to mimic df_feat input."""
    df_feat = aa.load_features()
    return df_feat.head(num_features)


def create_df_feat_ci(num_features=20, n_nan=0):
    """Creates a df_feat with the bootstrap interval columns of 'mean_dif'.

    Mimics the output of CPP(bootstrap=True, bootstrap_kws={'ci': 0.95}).run(). The first ``n_nan``
    features get NaN bounds, as a feature selected in fewer than two rounds does."""
    df_feat = create_df_feat(num_features=num_features).copy().reset_index(drop=True)
    mean_dif = df_feat["mean_dif"].to_numpy(dtype=float)
    df_feat["mean_dif_ci_low"] = mean_dif - 0.02
    df_feat["mean_dif_ci_high"] = mean_dif + 0.03
    if n_nan > 0:
        df_feat.loc[df_feat.index[:n_nan], ["mean_dif_ci_low", "mean_dif_ci_high"]] = np.nan
    return df_feat


def get_ci_segments(ax=None):
    """Returns the drawn confidence-interval whiskers of ``ax`` as [[x_low, y], [x_high, y]] lists."""
    segments = []
    for container in ax.containers:
        if isinstance(container, ErrorbarContainer):
            segments += [s.tolist() for s in container[2][0].get_segments()]
    return segments


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

    @settings(max_examples=3, deadline=None)
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


    def test_rank(self):
        for rank in [True, False]:
            cpp_plot = aa.CPPPlot()
            df_feat = create_df_feat()
            fig, axes = cpp_plot.ranking(df_feat=df_feat, rank=rank)
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
            col_imp = "feat_impact" if shap_plot else "feat_importance"
            fig, axes = cpp_plot.ranking(df_feat=df_feat, shap_plot=shap_plot, col_imp=col_imp)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    def test_col_dif_valid(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        df_feat.insert(0, "feat_impact", [1] * len(df_feat))
        valid_col_difs = ['mean_dif', 'mean_dif_TestSample', 'mean_dif_test']
        for col_dif in valid_col_difs:
            if col_dif not in df_feat.columns:
                df_feat.insert(0, col_dif, [1] * len(df_feat))
            shap_plot = col_dif != "mean_dif"
            col_imp = "feat_impact" if shap_plot else "feat_importance"
            fig, axes = cpp_plot.ranking(df_feat=df_feat, shap_plot=shap_plot, col_imp=col_imp, col_dif=col_dif)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()

    def test_col_imp_valid(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        valid_col_imps = ['feat_importance', 'feat_impact_TestSample', 'feat_impact_col_imp']
        for col_imp in valid_col_imps:
            if col_imp not in df_feat.columns:
                df_feat.insert(0, col_imp, [1] * len(df_feat))
            shap_plot = col_imp != "feat_importance"
            fig, axes = cpp_plot.ranking(df_feat=df_feat, col_imp=col_imp, shap_plot=shap_plot)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            plt.close()

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
    @given(tmd_len=st.integers(min_value=17, max_value=100))
    def test_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, tmd_len=tmd_len)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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


    @settings(max_examples=3, deadline=None)
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

    @settings(max_examples=3, deadline=None)
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

    def test_rank_info_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        for _ in range(3):
            x = random.uniform(0, 10)
            y = random.uniform(0, 1)
            fig, axes = cpp_plot.ranking(df_feat=df_feat, rank_info_xy=(x, y))
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            assert isinstance(axes[0], plt.Axes)
            plt.close()

    def test_show_ci(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci()
        for show_ci in [True, False]:
            fig, axes = cpp_plot.ranking(df_feat=df_feat, show_ci=show_ci)
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert len(axes) == 3
            n_whiskers = len(get_ci_segments(ax=axes[1]))
            assert n_whiskers == (15 if show_ci else 0)
            plt.close()

    def test_ci_color(self):
        df_feat = create_df_feat_ci()
        for ci_color in ["black", "red", "tab:blue"]:
            cpp_plot = aa.CPPPlot()
            fig, axes = cpp_plot.ranking(df_feat=df_feat, show_ci=True, ci_color=ci_color)
            assert isinstance(fig, plt.Figure)
            assert len(get_ci_segments(ax=axes[1])) > 0
            colors = [c for c in axes[1].containers if isinstance(c, ErrorbarContainer)][0][2][0].get_colors()
            assert tuple(colors[0]) == to_rgba(ci_color)
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

    def test_invalid_rank(self):
        for rank in [[], None, 123]:
            cpp_plot = aa.CPPPlot()
            df_feat = create_df_feat()
            with pytest.raises(ValueError):
                fig, axes = cpp_plot.ranking(df_feat=df_feat, rank=rank)
            plt.close()

    def test_col_dif_invalid(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        df_feat.insert(0, "mean_dif_test", [2] * len(df_feat))
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, col_dif='invalid_col_name')
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, col_dif='invalid_col_test', shap_plot=False)

    def test_col_imp_invalid(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        df_feat.insert(0, "feat_impact_test", [2] * len(df_feat))
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, col_imp='invalid_col_name')
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, col_imp='feat_impact_test', shap_plot=False)

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

    @settings(max_examples=20, deadline=None)
    @given(tmd_len=st.integers(max_value=0))
    def test_invalid_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, tmd_len=tmd_len)

    def test_invalid_xlim_dif(self):
        for xlim_dif in [(2,1), (None, 1), (), (1, 1, 1), (-2, -4)]:
            cpp_plot = aa.CPPPlot()
            df_feat = create_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, xlim_dif=xlim_dif)

    @settings(max_examples=10, deadline=None)
    @given(xlim_rank=st.tuples(st.floats(min_value=10), st.floats(max_value=0)))
    def test_invalid_xlim_rank(self, xlim_rank):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        if xlim_rank[0] > xlim_rank[1]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, xlim_rank=xlim_rank)

    def test_invalid_rank_info_xy(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, rank_info_xy=-1)
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, rank_info_xy="str")
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, rank_info_xy=(None, 123, 234))

    @settings(max_examples=20, deadline=None)
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

    @settings(max_examples=20, deadline=None)
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

    @settings(max_examples=20, deadline=None)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_titles(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_titles=fontsize)

    @settings(max_examples=20, deadline=None)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_labels(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_labels=fontsize)

    @settings(max_examples=20, deadline=None)
    @given(fontsize=st.one_of(st.floats(max_value=-1), st.text()))
    def test_invalid_fontsize_annotations(self, fontsize):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, fontsize_annotations=fontsize)

    def test_invalid_show_ci(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci()
        for show_ci in ["yes", None, 1.5, []]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, show_ci=show_ci)
            plt.close()

    def test_invalid_show_ci_missing_cols(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci()
        # Both interval columns missing, and each one on its own
        for cols_drop in [["mean_dif_ci_low", "mean_dif_ci_high"], ["mean_dif_ci_low"], ["mean_dif_ci_high"]]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat.drop(columns=cols_drop), show_ci=True)
            plt.close()

    def test_invalid_show_ci_col_dif(self):
        # Intervals exist only for the group-level 'mean_dif', not for a sample-specific column
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci()
        df_feat["mean_dif_Protein4"] = df_feat["mean_dif"]
        df_feat["feat_impact_Protein4"] = df_feat["feat_importance"]
        with pytest.raises(ValueError):
            cpp_plot.ranking(df_feat=df_feat, col_dif="mean_dif_Protein4",
                             col_imp="feat_impact_Protein4", shap_plot=True, show_ci=True)
        plt.close()

    def test_invalid_ci_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci()
        for ci_color in ["not_a_color", 123, None, []]:
            with pytest.raises(ValueError):
                cpp_plot.ranking(df_feat=df_feat, show_ci=True, ci_color=ci_color)
            plt.close()


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
        rank_info_xy = (2, 4)

        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=n_top, figsize=figsize,
                                     tmd_len=tmd_len, tmd_jmd_space=tmd_jmd_space,
                                     tmd_color=tmd_color, jmd_color=jmd_color,
                                     tmd_jmd_alpha=tmd_jmd_alpha, name_test=name_test,
                                     name_ref=name_ref, fontsize_titles=fontsize_titles,
                                     fontsize_labels=fontsize_labels,
                                     fontsize_annotations=fontsize_annotations,
                                     xlim_dif=xlim_dif, xlim_rank=xlim_rank,
                                     rank_info_xy=rank_info_xy)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        assert isinstance(axes[0], plt.Axes)

    def test_complex_valid_show_ci(self):
        # Intervals combined with other parameters, and with NaN bounds for the first 3 features
        # (rank=False keeps the row order, so those three are among the 12 shown)
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci(num_features=20, n_nan=3)
        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=12, rank=False, show_ci=True,
                                     ci_color="tab:gray", figsize=(10, 8), tmd_len=25,
                                     name_test="TestDataset", name_ref="RefDataset",
                                     xlim_dif=(-20, 20), xlim_rank=(1, 10),
                                     fontsize_annotations=10)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 3
        # 12 bars are shown, 3 of the features have NaN bounds and get no whisker
        assert len(get_ci_segments(ax=axes[1])) == 9
        plt.close()

    def test_show_ci_all_nan_bounds(self):
        # Every feature selected in fewer than two rounds: bars are drawn, no whisker is
        cpp_plot = aa.CPPPlot()
        df_feat = create_df_feat_ci(num_features=10, n_nan=10)
        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=10, show_ci=True)
        assert isinstance(fig, plt.Figure)
        assert len(get_ci_segments(ax=axes[1])) == 0
        plt.close()


# Test Class for Golden Values
class TestRankingGoldenValues:
    """Test class for the ranking method, focusing on hand-computed whisker geometry."""

    @staticmethod
    def create_df_golden(ci_low=None, ci_high=None):
        """Three features with hand-picked mean differences and interval bounds."""
        df_feat = create_df_feat(num_features=3).copy().reset_index(drop=True)
        df_feat["mean_dif"] = [0.10, -0.08, 0.05]
        df_feat["feat_importance"] = [3.0, 2.0, 1.0]
        df_feat["mean_dif_ci_low"] = [0.06, -0.12, np.nan] if ci_low is None else ci_low
        df_feat["mean_dif_ci_high"] = [0.13, -0.03, np.nan] if ci_high is None else ci_high
        return df_feat

    def test_whisker_endpoints(self):
        # The subplot shows mean differences in percent (x100), so the bounds 0.06/0.13 of the
        # first feature give a whisker from 6 to 13 at y=0, and -0.12/-0.03 give -12 to -3 at y=1.
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=self.create_df_golden(), n_top=3, rank=False, show_ci=True)
        segments = get_ci_segments(ax=axes[1])
        np.testing.assert_allclose(segments, [[[6.0, 0.0], [13.0, 0.0]], [[-12.0, 1.0], [-3.0, 1.0]]])
        plt.close()

    def test_nan_bounds_get_no_whisker(self):
        # The third feature (NaN bounds) is not drawn: 3 bars, 2 whiskers, none at y=2
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=self.create_df_golden(), n_top=3, rank=False, show_ci=True)
        segments = get_ci_segments(ax=axes[1])
        assert len(segments) == 2
        assert [s[0][1] for s in segments] == [0.0, 1.0]
        plt.close()

    def test_interval_not_bracketing_point_estimate(self):
        # ci_low (0.11) above mean_dif (0.10): a percentile interval need not contain the
        # full-data estimate, so the left arm is clipped at zero and the whisker runs 10 to 13.
        df_feat = self.create_df_golden(ci_low=[0.11, -0.12, np.nan], ci_high=[0.13, -0.03, np.nan])
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=3, rank=False, show_ci=True)
        segments = get_ci_segments(ax=axes[1])
        np.testing.assert_allclose(segments, [[[10.0, 0.0], [13.0, 0.0]], [[-12.0, 1.0], [-3.0, 1.0]]])
        plt.close()

    def test_no_whiskers_without_show_ci(self):
        # Interval columns present but the option off: nothing is drawn
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=self.create_df_golden(), n_top=3, rank=False)
        assert len(get_ci_segments(ax=axes[1])) == 0
        plt.close()
        plt.close()