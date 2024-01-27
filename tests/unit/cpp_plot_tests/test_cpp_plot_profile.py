"""
This script tests the CPPPlot().profile() method.
"""
import pandas as pd
import matplotlib.pyplot as plt
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

aa.options["verbose"] = False


# Constants and Helper functions
N_SEQ = 10
COL_FEAT_IMPACT_TEST = "feat_impact_test"
VALID_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
INVALID_COLORS = ["invalid-color", "tab:black", 234, [], {}]
VALID_GRID_AXIS = ['x', 'y', 'both', None]
INVALID_GRID_AXIS = ["invalid-axis", 123, [], {}]
LIST_CAT = ['ASA/Volume', 'Conformation', 'Energy', 'Polarity', 'Shape', 'Composition', 'Structure-Activity', 'Others']
DICT_COLOR = dict(zip(LIST_CAT, VALID_COLORS))

def get_args_seq(n=0):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SEQ)
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    return args_seq


def get_df_feat(n=10):
    df_feat = aa.load_features().head(n)
    df_feat.insert(0, COL_FEAT_IMPACT_TEST, [2]*len(df_feat))
    return df_feat


class TestProfilePositive:
    """Positive test cases for the profile method, focusing on individual parameters."""

    def test_df_feat(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()


    def test_shap_plot_col_imp(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.profile(df_feat=df_feat, shap_plot=True, col_imp=COL_FEAT_IMPACT_TEST,
                                   **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_normalize(self):
        for normalize in [True, False]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.profile(df_feat=df_feat, normalize=normalize)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(figsize=st.tuples(st.floats(min_value=4.0, max_value=20.0), st.floats(min_value=5.0, max_value=20.0)))
    def test_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, figsize=figsize)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(start=st.integers(min_value=0, max_value=1000))
    def test_start(self, start):
        if start <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.profile(df_feat=df_feat, start=start)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_len=st.integers(min_value=20, max_value=100))
    def test_tmd_len(self, tmd_len):
        if tmd_len <= 1000:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            fig, ax = cpp_plot.profile(df_feat=df_feat, tmd_len=tmd_len)
            assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(jmd_color=st.sampled_from(VALID_COLORS), tmd_seq_color=st.sampled_from(VALID_COLORS),
           jmd_seq_color=st.sampled_from(VALID_COLORS))
    def test_color_parameters(self, jmd_color, tmd_seq_color, jmd_seq_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                                   jmd_seq_color=jmd_seq_color)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(seq_size=st.floats(min_value=8.0, max_value=14.0),
           fontsize_tmd_jmd=st.floats(min_value=8.0, max_value=14.0))
    def test_font_sizes(self, seq_size, fontsize_tmd_jmd):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(add_xticks_pos=st.booleans(),
           highlight_tmd_area=st.booleans(),
           add_legend_cat=st.booleans())
    def test_boolean_flags(self, add_xticks_pos, highlight_tmd_area, add_legend_cat):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, add_xticks_pos=add_xticks_pos,
                                   highlight_tmd_area=highlight_tmd_area, add_legend_cat=add_legend_cat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(highlight_alpha=st.floats(min_value=0.0, max_value=1.0),
           bar_width=st.floats(min_value=0.1, max_value=2.0))
    def test_numeric_styling_parameters(self, highlight_alpha, bar_width):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, highlight_alpha=highlight_alpha, bar_width=bar_width)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(edge_color=st.sampled_from(VALID_COLORS + [None]),
           grid_axis=st.sampled_from(VALID_GRID_AXIS))
    def test_edge_color_and_grid_axis(self, edge_color, grid_axis):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, edge_color=edge_color, grid_axis=grid_axis)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(xtick_size=st.floats(min_value=8.0, max_value=14.0),
           xtick_width=st.floats(min_value=0.5, max_value=2.0),
           xtick_length=st.floats(min_value=3.0, max_value=10.0))
    def test_xtick_styling(self, xtick_size, xtick_width, xtick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, xtick_size=xtick_size, xtick_width=xtick_width,
                                   xtick_length=xtick_length)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(ytick_size=st.floats(min_value=8.0, max_value=14.0),
           ytick_width=st.floats(min_value=0.5, max_value=2.0),
           ytick_length=st.floats(min_value=3.0, max_value=8.0))
    def test_ytick_styling(self, ytick_size, ytick_width, ytick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, ytick_size=ytick_size, ytick_width=ytick_width,
                                   ytick_length=ytick_length)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_sequence_parameters(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.profile(df_feat=df_feat, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_dict_color(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, dict_color=DICT_COLOR)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        legend_kws = {'title': 'Legend', 'loc': 'upper right'}
        fig, ax = cpp_plot.profile(df_feat=df_feat, add_legend_cat=True, legend_kws=legend_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(ylim_max=st.floats(min_value=35.0, max_value=55.0))
    def test_ylim(self, ylim_max):
        ylim = (0, ylim_max)
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.profile(df_feat=df_feat, ylim=ylim)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_predefined_ax(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, predefined_ax = plt.subplots()
        _, ax = cpp_plot.profile(df_feat=df_feat, ax=predefined_ax)
        assert ax is predefined_ax  # Ensure the same ax object is used
        plt.close()

    # Negative tests
    def test_invalid_df_feat(self):
        cpp_plot = aa.CPPPlot()
        df_feat = "invalid_df_feat"  # This should be a DataFrame
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(shap_plot=st.booleans(), col_imp=st.text(min_size=1))
    def test_invalid_col_imp(self, shap_plot, col_imp):
        if col_imp not in [COL_FEAT_IMPACT_TEST, None]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.profile(df_feat=df_feat, shap_plot=shap_plot, col_imp=col_imp)
            plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(normalize=st.text(min_size=1))  # Text input for boolean parameter
    def test_invalid_normalize(self, normalize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, normalize=normalize)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(figsize=st.tuples(st.just(-10.0), st.just(-10.0)))  # Negative values for figsize
    def test_invalid_figsize(self, figsize):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, figsize=figsize)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(start=st.just(-1))  # Negative start value
    def test_invalid_start(self, start):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, start=start)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_len=st.just(-1))  # Negative length
    def test_invalid_tmd_len(self, tmd_len):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, tmd_len=tmd_len)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(color=st.sampled_from(INVALID_COLORS))
    def test_invalid_color_parameters(self, color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, tmd_color=color, jmd_color=color, tmd_seq_color=color, jmd_seq_color=color)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(dict_color=st.just({'invalid_cat': 'blue'}))
    def test_invalid_dict_color(self, dict_color):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, dict_color=dict_color)
        plt.close()

    def test_invalid_legend_kws(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_legend_kws = 'not_a_dict'  # This should be a dictionary
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, add_legend_cat=True, legend_kws=invalid_legend_kws)
        plt.close()


    @settings(max_examples=3, deadline=5000)
    @given(highlight_alpha=st.just(-1), bar_width=st.just(-1))  # Negative values for alpha and bar_width
    def test_invalid_numeric_styling_parameters(self, highlight_alpha, bar_width):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, highlight_alpha=highlight_alpha, bar_width=bar_width)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(grid_axis=st.sampled_from(INVALID_GRID_AXIS))
    def test_invalid_grid_axis(self, grid_axis):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, grid_axis=grid_axis)
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(xtick_size=st.just(-1), xtick_width=st.just(-1), xtick_length=st.just(-1),
           ytick_size=st.just(-1), ytick_width=st.just(-1), ytick_length=st.just(-1))
    def test_invalid_tick_styling(self, xtick_size, xtick_width, xtick_length, ytick_size, ytick_width, ytick_length):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length,
                             ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        plt.close()

    def test_invalid_ax(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, ax="not_an_ax_object")  # ax should be a matplotlib.axes.Axes instance or None
        plt.close()

    @settings(max_examples=3, deadline=5000)
    @given(tmd_seq=st.text(), jmd_n_seq=st.text(), jmd_c_seq=st.text())
    def test_invalid_sequences(self, tmd_seq, jmd_n_seq, jmd_c_seq):
        if not isinstance(tmd_seq, str) or not isinstance(jmd_n_seq, str) or not isinstance(jmd_c_seq, str):
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.profile(df_feat=df_feat, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
            plt.close()


    def test_zero_or_negative_figsize(self):
        for figsize in [(None, None), [], "asdf", (-12, 1)]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.profile(df_feat=df_feat, figsize=figsize)
            plt.close()

    def test_invalid_ylim(self):
        for ylim in [(5, 3), (5, 5), (-1, -2), (None, "str"), ()]:
            cpp_plot = aa.CPPPlot()
            df_feat = get_df_feat()
            with pytest.raises(ValueError):
                cpp_plot.profile(df_feat=df_feat, ylim=ylim)
            plt.close()

    def test_invalid_dict_color_structure(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        invalid_dict_color = {'invalid_cat': 123}  # Value should be a valid color representation, not a number
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, dict_color=invalid_dict_color)
        plt.close()


class TestProfileComplex:
    """Complex test cases for the profile method, focusing on parameter combinations."""

    def test_complex_positive_scenario_1(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        fig, ax = cpp_plot.profile(
            df_feat=df_feat,
            shap_plot=True,
            col_imp=COL_FEAT_IMPACT_TEST,
            normalize=True,
            figsize=(10, 8),
            start=1,
            tmd_len=20,
            tmd_color="mediumspringgreen",
            jmd_color="blue",
            tmd_seq_color="black",
            jmd_seq_color="white",
            seq_size=10,
            fontsize_tmd_jmd=12,
            add_xticks_pos=True,
            highlight_tmd_area=True,
            add_legend_cat=False,
            dict_color=DICT_COLOR,
            legend_kws={'title': 'Legend'},
            bar_width=0.75,
            edge_color="black",
            grid_axis="both",
            ylim=(0, 100),
            xtick_size=11,
            xtick_width=1,
            xtick_length=5,
            ytick_size=11,
            ytick_width=1,
            ytick_length=5,
            **args_seq
        )
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_complex_positive_scenario_2(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq(n=5)
        fig, ax = cpp_plot.profile(
            df_feat=df_feat,
            shap_plot=False,
            col_imp=None,
            normalize=False,
            figsize=(8, 6),
            start=5,
            tmd_len=25,
            tmd_color="red",
            jmd_color="green",
            tmd_seq_color="white",
            jmd_seq_color="black",
            seq_size=9,
            fontsize_tmd_jmd=10,
            add_xticks_pos=False,
            highlight_tmd_area=False,
            add_legend_cat=False,
            dict_color=None,
            legend_kws=None,
            bar_width=1.0,
            edge_color=None,
            grid_axis=None,
            xtick_size=10,
            xtick_width=2,
            xtick_length=6,
            ytick_size=10,
            ytick_width=2,
            ytick_length=6,
            **args_seq
        )
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close()

    def test_complex_negative_scenario(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        args_seq = get_args_seq(n=5)
        with pytest.raises(ValueError):
            cpp_plot.profile(
                df_feat=df_feat,
                shap_plot=True,
                col_imp=COL_FEAT_IMPACT_TEST,
                normalize="not_a_boolean",
                figsize=(0, 0),  # Invalid figsize
                start=-1,  # Invalid start
                tmd_len=-20,  # Invalid tmd_len
                tmd_color="invalid_color",  # Invalid tmd_color
                jmd_color="invalid_color",  # Invalid jmd_color
                tmd_seq_color="invalid_color",  # Invalid tmd_seq_color
                jmd_seq_color="invalid_color",  # Invalid jmd_seq_color
                bar_width=-0.75,  # Invalid bar_width
                edge_color="invalid_color",  # Invalid edge_color
                grid_axis="invalid_axis",  # Invalid grid_axis
                ylim=(-10, -5),  # Invalid ylim
                xtick_size=-1,  # Invalid xtick_size
                xtick_width=-1,  # Invalid xtick_width
                xtick_length=-1,  # Invalid xtick_length
                ytick_size=-1,  # Invalid ytick_size
                ytick_width=-1,  # Invalid ytick_width
                ytick_length=-1,  # Invalid ytick_length
                **args_seq
            )
        plt.close()
