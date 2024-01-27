"""
This script tests the CPPPlot().update_seq_size() method.
"""
import matplotlib.pyplot as plt
import hypothesis.strategies as st
from hypothesis import given, settings
import random
import pytest
import aaanalysis as aa


# Helper functions
def get_args_seq(n=0):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SEQ)
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    return args_seq


def plot_profile(cpp_plot=None, shap_plot=False, args_seq=None):
    col_imp = COL_FEAT_IMPACT_TEST if shap_plot else "feat_importance"
    if args_seq is None:
        fig, ax = cpp_plot.profile(df_feat=_df_feat, col_imp=col_imp, shap_plot=shap_plot)
    else:
        fig, ax = cpp_plot.profile(df_feat=_df_feat, col_imp=col_imp, shap_plot=shap_plot, **args_seq)
    return fig, ax


aa.options["verbose"] = False
COL_FEAT_IMPACT_TEST = "feat_impact_test"
N_SEQ = 10
_df_feat = aa.load_features().head(5)
_df_feat.insert(0, COL_FEAT_IMPACT_TEST, [2]*len(_df_feat))

VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

INVALID_COLORS = ["asdf", "tab:black", 234, [], {}]


class TestUpdateSeqSize:
    """Test class for update_seq_size method, focusing on individual parameters."""

    # Positive tests
    def test_ax_fig(self):
        list_n = list(range(0, N_SEQ*2))
        random_samples = random.sample(list_n, 3)
        for n in random_samples:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq(n=n)
            fig, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax=ax, fig=fig)
            assert isinstance(result, plt.Axes)
            plt.close()

    @settings(max_examples=2, deadline=10000)
    @given(fontsize_tmd_jmd=st.one_of(st.none(), st.integers(min_value=1, max_value=15)))
    def test_fontsize_tmd_jmd(self, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq()
        _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
        result = cpp_plot.update_seq_size(ax=ax, fontsize_tmd_jmd=fontsize_tmd_jmd)
        assert isinstance(result, plt.Axes)
        plt.close()

    def test_weight_tmd_jmd(self):
        for weight_tmd_jmd in ["bold", "normal"]:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq(n=0)
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax=ax, weight_tmd_jmd=weight_tmd_jmd)
            assert isinstance(result, plt.Axes)
            plt.close()

    def test_tmd_color(self):
        for i in range(2):
            tmd_color = random.sample(VALID_COLORS, 1)[0]
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax, tmd_color=tmd_color)
            assert isinstance(result, plt.Axes)
            plt.close()

    def test_jmd_color(self):
        for i in range(2):
            jmd_color = random.sample(VALID_COLORS, 1)[0]
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq(n=1)
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax, jmd_color=jmd_color)
            assert isinstance(result, plt.Axes)
            plt.close()

    def test_tmd_seq_color(self):
        for i in range(2):
            tmd_seq_color = random.sample(VALID_COLORS, 1)[0]
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax, tmd_seq_color=tmd_seq_color)
            assert isinstance(result, plt.Axes)
            plt.close()

    def test_jmd_seq_color(self):
        for i in range(2):
            jmd_seq_color = random.sample(VALID_COLORS, 1)[0]
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            result = cpp_plot.update_seq_size(ax, jmd_seq_color=jmd_seq_color)
            assert isinstance(result, plt.Axes)
            plt.close()

    # Negative tests
    def test_ax_negative(self):
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.update_seq_size(ax=None)

    @settings(max_examples=2, deadline=10000)
    @given(fontsize_tmd_jmd=st.one_of(st.just(-1), st.just(0), st.text()))
    def test_fontsize_tmd_jmd_negative(self, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq()
        _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
        with pytest.raises(ValueError):
            cpp_plot.update_seq_size(ax=ax, fontsize_tmd_jmd=fontsize_tmd_jmd)
        plt.close()

    @settings(max_examples=2, deadline=10000)
    @given(weight_tmd_jmd=st.text(min_size=1).filter(lambda x: x not in ['normal', 'bold']))
    def test_weight_tmd_jmd_negative(self, weight_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq()
        _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
        with pytest.raises(ValueError):
            cpp_plot.update_seq_size(ax=ax, weight_tmd_jmd=weight_tmd_jmd)
        plt.close()

    def test_tmd_color_negative(self):
        for tmd_color in INVALID_COLORS:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            with pytest.raises(ValueError):
                cpp_plot.update_seq_size(ax=ax, tmd_color=tmd_color)
            plt.close()

    def test_jmd_color_negative(self):
        for jmd_color in INVALID_COLORS:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            with pytest.raises(ValueError):
                cpp_plot.update_seq_size(ax=ax, jmd_color=jmd_color)
            plt.close()

    def test_tmd_seq_color_negative(self):
        for tmd_seq_color in INVALID_COLORS:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            with pytest.raises(ValueError):
                cpp_plot.update_seq_size(ax=ax, tmd_seq_color=tmd_seq_color)
            plt.close()

    def test_jmd_seq_color_negative(self):
        for jmd_seq_color in INVALID_COLORS:
            cpp_plot = aa.CPPPlot()
            args_seq = get_args_seq()
            _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
            with pytest.raises(ValueError):
                cpp_plot.update_seq_size(ax=ax, jmd_seq_color=jmd_seq_color)
            plt.close()

class TestUpdateSeqSizeComplex:
    # ... (previous test cases)

    def test_complex_positive(self):
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq(n=0)
        _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
        tmd_color = random.choice(VALID_COLORS)
        jmd_color = random.choice(VALID_COLORS)
        tmd_seq_color = random.choice(VALID_COLORS)
        jmd_seq_color = random.choice(VALID_COLORS)
        result = cpp_plot.update_seq_size(
            ax=ax,
            fig=None,  # Assuming None is a valid value
            fontsize_tmd_jmd=12,
            weight_tmd_jmd='bold',
            tmd_color=tmd_color,
            jmd_color=jmd_color,
            tmd_seq_color=tmd_seq_color,
            jmd_seq_color=jmd_seq_color
        )
        assert isinstance(result, plt.Axes)
        plt.close()

    def test_complex_negative(self):
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq(n=0)
        _, ax = plot_profile(cpp_plot=cpp_plot, args_seq=args_seq)
        tmd_color = random.choice(INVALID_COLORS)  # Choosing an invalid color
        jmd_color = random.choice(VALID_COLORS)
        tmd_seq_color = random.choice(VALID_COLORS)
        jmd_seq_color = random.choice(VALID_COLORS)
        with pytest.raises(ValueError):
            cpp_plot.update_seq_size(ax=ax, fig=None,  # Assuming None is a valid value
                fontsize_tmd_jmd=12, weight_tmd_jmd='bold', tmd_color=tmd_color,
                # This is expected to trigger a ValueError
                jmd_color=jmd_color, tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        plt.close()