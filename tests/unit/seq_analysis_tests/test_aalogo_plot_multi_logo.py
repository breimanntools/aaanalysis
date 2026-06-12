"""
This is a script to test the AAlogoPlot.multi_logo method.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

# Set default deadline from 200 to 4000
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False


# Helper
def get_df_logo(n=50, logo_type="probability", label_test=1):
    """Load df_logo from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].values
    df_parts = sf.get_df_parts(df_seq=df_seq)
    return aa.AAlogo(logo_type=logo_type).get_df_logo(
        df_parts=df_parts, labels=labels, label_test=label_test)


def get_list_df_logo(n=50, logo_type="probability"):
    """Load [df_logo_pos, df_logo_neg] from DOM_GSEC dataset."""
    return [get_df_logo(n=n, logo_type=logo_type, label_test=lt) for lt in [1, 0]]


def get_aal_plot(jmd_n_len=10, jmd_c_len=10, logo_type="probability"):
    """Create default AAlogoPlot instance."""
    return aa.AAlogoPlot(logo_type=logo_type, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)


# ===========================================================================
# I Test multi_logo: list_df_logo
# ===========================================================================
class TestMultiLogoListDfLogo:
    """Test multi_logo 'list_df_logo' parameter."""

    def test_valid_list_df_logo(self):
        """Test valid 'list_df_logo' produces a figure with correct subplot count."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == len(list_df_logo)
        plt.close("all")

    def test_valid_single_element_list(self):
        """Test that a list with one DataFrame works (n_plots=1 edge case)."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(list_df_logo=[df_logo])
        assert len(axes) == 1
        plt.close("all")

    def test_valid_three_logos(self):
        """Test with three logos."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(list_df_logo=[df_logo, df_logo, df_logo])
        assert len(axes) == 3
        plt.close("all")

    def test_invalid_list_df_logo_none(self):
        """Test that list_df_logo=None raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=None)

    def test_invalid_list_df_logo_empty(self):
        """Test that empty list raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=[])

    def test_invalid_list_df_logo_not_list(self):
        """Test that non-list raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for val in [df_logo, "invalid", 1]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=val)

    def test_invalid_list_df_logo_contains_non_df(self):
        """Test that list containing non-DataFrame element raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for bad_element in [None, "invalid", 1, pd.Series([1, 2, 3])]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=[df_logo, bad_element])

    def test_invalid_list_df_logo_mismatched_lengths(self):
        """Test that logos with different position counts raises ValueError."""
        df_logo = get_df_logo()
        df_logo_short = df_logo.iloc[:5]   # truncate to different length
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=[df_logo, df_logo_short])

    def test_df_logos_not_mutated(self):
        """Test that list_df_logo entries are not mutated by multi_logo."""
        list_df_logo = get_list_df_logo(logo_type="probability")
        maxes_before = [df.values.max() for df in list_df_logo]
        aal_plot = get_aal_plot()
        aal_plot.multi_logo(list_df_logo=list_df_logo)
        for df, max_before in zip(list_df_logo, maxes_before):
            assert df.values.max() == max_before
        plt.close("all")

    def test_shared_y_axis(self):
        """Test that all subplots share the same y-axis maximum."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo)
        ylims = [ax.get_ylim() for ax in axes]
        assert len(set(ylims)) == 1
        plt.close("all")


# ===========================================================================
# II Test multi_logo: figsize_per_logo
# ===========================================================================
class TestMultiLogoFigsizePerLogo:
    """Test multi_logo 'figsize_per_logo' parameter."""

    def test_valid_figsize_per_logo(self):
        """Test valid 'figsize_per_logo' scales total figure height correctly."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for w, h in [(8, 3), (10, 2), (6, 4)]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, figsize_per_logo=(w, h))
            expected_height = h * len(list_df_logo)
            assert abs(fig.get_size_inches()[1] - expected_height) < 0.1
            plt.close("all")

    def test_invalid_figsize_per_logo(self):
        """Test invalid 'figsize_per_logo' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for figsize in ["invalid", (8,), (8, 3, 2), None]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, figsize_per_logo=figsize)


# ===========================================================================
# III Test multi_logo: list_name_data
# ===========================================================================
class TestMultiLogoListNameData:
    """Test multi_logo 'list_name_data' parameter."""

    def test_valid_list_name_data(self):
        """Test valid 'list_name_data' annotates each subplot."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        names = ["Positive", "Negative"]
        fig, axes = aal_plot.multi_logo(
            list_df_logo=list_df_logo, list_name_data=names)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_valid_list_name_data_none(self):
        """Test that list_name_data=None produces no annotation error."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(
            list_df_logo=list_df_logo, list_name_data=None)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_invalid_list_name_data_not_list(self):
        """Test that non-list list_name_data raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for val in ["Positive", 1, ("Positive", "Negative")]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, list_name_data=val)

    def test_invalid_list_name_data_length_mismatch(self):
        """Test that list_name_data with wrong length raises ValueError."""
        list_df_logo = get_list_df_logo()   # 2 logos
        aal_plot = get_aal_plot()
        for names in [["only_one"], ["a", "b", "c"]]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, list_name_data=names)


# ===========================================================================
# IV Test multi_logo: list_name_data_color
# ===========================================================================
class TestMultiLogoListNameDataColor:
    """Test multi_logo 'list_name_data_color' parameter."""

    def test_valid_list_name_data_color_string(self):
        """Test valid single string color applies to all subplots."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(
            list_df_logo=list_df_logo,
            list_name_data=["Positive", "Negative"],
            list_name_data_color="black")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_valid_list_name_data_color_list(self):
        """Test valid list of colors, one per logo."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.multi_logo(
            list_df_logo=list_df_logo,
            list_name_data=["Positive", "Negative"],
            list_name_data_color=["tab:green", "tab:gray"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_invalid_list_name_data_color_wrong_type(self):
        """Test that non-string, non-list raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for color in [1, None, ("red", "blue")]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo,
                                    list_name_data_color=color)

    def test_invalid_list_name_data_color_length_mismatch(self):
        """Test that color list with wrong length raises ValueError."""
        list_df_logo = get_list_df_logo()   # 2 logos
        aal_plot = get_aal_plot()
        for colors in [["red"], ["red", "blue", "green"]]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo,
                                    list_name_data_color=colors)


# ===========================================================================
# V Test multi_logo: name_data_pos
# ===========================================================================
class TestMultiLogoNameDataPos:
    """Test multi_logo 'name_data_pos' parameter."""

    def test_valid_name_data_pos(self):
        """Test all valid 'name_data_pos' options."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for pos in ["top", "right", "bottom", "left"]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo,
                list_name_data=["Positive", "Negative"],
                name_data_pos=pos)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_name_data_pos(self):
        """Test invalid 'name_data_pos' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for pos in ["center", "middle", None, 0]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, name_data_pos=pos)


# ===========================================================================
# VI Test multi_logo: logo_stack_order and weight_tmd_jmd
# ===========================================================================
class TestMultiLogoStringOptions:
    """Test multi_logo string option parameters."""

    def test_valid_logo_stack_order(self):
        """Test valid 'logo_stack_order' options."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for order in ["big_on_top", "small_on_top", "fixed"]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, logo_stack_order=order)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_logo_stack_order(self):
        """Test invalid 'logo_stack_order' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for order in ["largest_on_top", "random", None, 0]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, logo_stack_order=order)

    def test_valid_weight_tmd_jmd(self):
        """Test valid 'weight_tmd_jmd' options."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for weight in ["normal", "bold"]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, weight_tmd_jmd=weight)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_weight_tmd_jmd(self):
        """Test invalid 'weight_tmd_jmd' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for weight in ["italic", "heavy", None, 1]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, weight_tmd_jmd=weight)


# ===========================================================================
# VII Test multi_logo: highlight_tmd_area and highlight_alpha
# ===========================================================================
class TestMultiLogoHighlight:
    """Test multi_logo highlight parameters."""

    def test_valid_highlight_tmd_area(self):
        """Test valid 'highlight_tmd_area' values."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for val in [True, False]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, highlight_tmd_area=val)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_highlight_tmd_area(self):
        """Test invalid 'highlight_tmd_area' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for val in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, highlight_tmd_area=val)

    def test_valid_highlight_alpha(self):
        """Test valid 'highlight_alpha' values."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for alpha in [0.0, 0.15, 0.5, 1.0]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, highlight_alpha=alpha)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_highlight_alpha(self):
        """Test invalid 'highlight_alpha' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for alpha in [-0.1, 1.1, 2.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, highlight_alpha=alpha)


# ===========================================================================
# VIII Test multi_logo: logo_width, xtick_width, xtick_length
# ===========================================================================
class TestMultiLogoNumericParams:
    """Test multi_logo numeric range parameters."""

    def test_valid_logo_width(self):
        """Test valid 'logo_width' values."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for width in [0.0, 0.5, 0.96, 1.0]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, logo_width=width)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_logo_width(self):
        """Test invalid 'logo_width' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for width in [-0.1, 1.1, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, logo_width=width)

    def test_valid_xtick_width(self):
        """Test valid 'xtick_width' values."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for width in [0.0, 1.0, 2.0, 5.0]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, xtick_width=width)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_xtick_width(self):
        """Test invalid 'xtick_width' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for width in [-1.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, xtick_width=width)

    def test_valid_xtick_length(self):
        """Test valid 'xtick_length' values."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for length in [0.0, 5.0, 11.0, 20.0]:
            fig, axes = aal_plot.multi_logo(
                list_df_logo=list_df_logo, xtick_length=length)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_xtick_length(self):
        """Test invalid 'xtick_length' raises ValueError."""
        list_df_logo = get_list_df_logo()
        aal_plot = get_aal_plot()
        for length in [-1.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, xtick_length=length)


# ===========================================================================
# IX Test multi_logo: check_parts_len
# ===========================================================================
class TestMultiLogoPartsLen:
    """Test multi_logo part length validation via check_parts_len."""

    def test_valid_parts_len(self):
        """Test that jmd_n_len + jmd_c_len < len(df_logo) passes."""
        list_df_logo = get_list_df_logo()   # each len = 40
        for jmd_n, jmd_c in [(10, 10), (0, 0), (5, 5)]:
            aal_plot = aa.AAlogoPlot(jmd_n_len=jmd_n, jmd_c_len=jmd_c)
            fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo)
            assert isinstance(fig, plt.Figure)
            plt.close("all")


# Helper for list_aal_kws tests
def get_df_parts_labels(n=50):
    """Build df_parts + labels from DOM_GSEC for list_aal_kws-based tests."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].values
    df_parts = sf.get_df_parts(df_seq=df_seq)
    return df_parts, labels


# ===========================================================================
# XII Test multi_logo: list_aal_kws (internal AAlogo shortcut)
# ===========================================================================
class TestMultiLogoListAalKws:
    """Test multi_logo 'list_aal_kws' convenience parameter."""

    def test_valid_list_aal_kws(self):
        """Test that list_aal_kws computes list_df_logo internally and plots."""
        df_parts, labels = get_df_parts_labels()
        aal_plot = get_aal_plot()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        fig, axes = aal_plot.multi_logo(list_aal_kws=list_aal_kws)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == len(list_aal_kws)
        plt.close("all")

    def test_list_aal_kws_matches_explicit(self):
        """Test that list_aal_kws path matches the explicit list_df_logo path."""
        df_parts, labels = get_df_parts_labels()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        list_df_logo = [aa.AAlogo().get_df_logo(**kws) for kws in list_aal_kws]
        fig_a, axes_a = get_aal_plot().multi_logo(list_df_logo=list_df_logo)
        fig_b, axes_b = get_aal_plot().multi_logo(list_aal_kws=list_aal_kws)
        assert isinstance(fig_a, plt.Figure) and isinstance(fig_b, plt.Figure)
        assert len(axes_a) == len(axes_b)
        plt.close("all")

    def test_list_aal_kws_respects_logo_type(self):
        """Test that the plot's logo_type is used when computing logos via list_aal_kws."""
        df_parts, labels = get_df_parts_labels()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        for logo_type in ["probability", "counts", "weight", "information"]:
            aal_plot = get_aal_plot(logo_type=logo_type)
            fig, axes = aal_plot.multi_logo(list_aal_kws=list_aal_kws)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_list_aal_kws_not_list(self):
        """Test that a non-list list_aal_kws raises ValueError."""
        aal_plot = get_aal_plot()
        for list_aal_kws in ["invalid", 1, {"df_parts": 1}]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_aal_kws=list_aal_kws)

    def test_invalid_list_aal_kws_empty(self):
        """Test that an empty list_aal_kws raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_aal_kws=[])

    def test_invalid_list_aal_kws_element_not_dict(self):
        """Test that a non-dict element in list_aal_kws raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_aal_kws=["not-a-dict"])

    def test_invalid_list_aal_kws_with_list_df_logo(self):
        """Test that combining list_aal_kws with list_df_logo raises ValueError."""
        df_parts, labels = get_df_parts_labels()
        list_df_logo = get_list_df_logo()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=list_df_logo, list_aal_kws=list_aal_kws)

    def test_invalid_list_aal_kws_unknown_key(self):
        """Test that an unknown key in a list_aal_kws dict raises ValueError."""
        df_parts, labels = get_df_parts_labels()
        aal_plot = get_aal_plot()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=1),
                        dict(df_parts=df_parts, labels=labels, bad_key=1)]
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_aal_kws=list_aal_kws)

    def test_list_aal_kws_shows_bit_bars(self):
        """Test that list_aal_kws auto-computes info and returns (ax_logo, ax_info) tuples."""
        df_parts, labels = get_df_parts_labels()
        aal_plot = get_aal_plot(logo_type="information")
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        fig, axes = aal_plot.multi_logo(list_aal_kws=list_aal_kws)
        assert len(axes) == 2
        for pair in axes:
            assert isinstance(pair, tuple) and len(pair) == 2
            ax_logo, ax_info = pair
            assert len(ax_info.patches) > 0  # bit-score bars drawn
        plt.close("all")


# Helper for list_df_logo_info tests
def get_list_df_logo_and_info(n=50, logo_type="information"):
    """Build matching [df_logo, df_logo] and [df_logo_info, df_logo_info] from DOM_GSEC."""
    df_parts, labels = get_df_parts_labels(n=n)
    aal = aa.AAlogo(logo_type=logo_type)
    kws = [dict(df_parts=df_parts, labels=labels, label_test=lt) for lt in [1, 0]]
    list_df_logo = [aal.get_df_logo(**k) for k in kws]
    list_df_logo_info = [aal.get_df_logo_info(**k) for k in kws]
    return list_df_logo, list_df_logo_info


# ===========================================================================
# XIII Test multi_logo: list_df_logo_info (bit-score bars)
# ===========================================================================
class TestMultiLogoListDfLogoInfo:
    """Test multi_logo 'list_df_logo_info' parameter (precomputed bit-score bars)."""

    def test_valid_list_df_logo_info_returns_tuples(self):
        """Test that list_df_logo_info renders bars and returns (ax_logo, ax_info) tuples."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo,
                                        list_df_logo_info=list_df_logo_info)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == len(list_df_logo)
        for ax_logo, ax_info in axes:
            assert isinstance(ax_logo, plt.Axes) and isinstance(ax_info, plt.Axes)
            assert len(ax_info.patches) > 0
        plt.close("all")

    def test_valid_single_group_with_info(self):
        """Test n_plots=1 edge case with a bit-score bar."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo[:1],
                                        list_df_logo_info=list_df_logo_info[:1])
        assert len(axes) == 1
        assert isinstance(axes[0], tuple)
        plt.close("all")

    def test_shared_info_bar_ylim_auto(self):
        """Test that all bit-score bars share a common y-axis when ylim is auto."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo,
                                        list_df_logo_info=list_df_logo_info)
        info_ylims = [ax_info.get_ylim() for _, ax_info in axes]
        assert len(set(info_ylims)) == 1
        plt.close("all")

    @settings(max_examples=5, deadline=None)
    @given(ymax=st.floats(min_value=2.0, max_value=6.0))
    def test_explicit_info_bar_ylim_applied(self, ymax):
        """Test that an explicit info_bar_ylim is applied to every bar."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo,
                                        list_df_logo_info=list_df_logo_info,
                                        info_bar_ylim=(0, ymax))
        for _, ax_info in axes:
            assert ax_info.get_ylim() == (0, ymax)
        plt.close("all")

    def test_name_data_top_lands_on_info_bar(self):
        """Test that name_data_pos='top' annotates the info bar, not the logo."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        names = ["Positive", "Negative"]
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo,
                                        list_df_logo_info=list_df_logo_info,
                                        list_name_data=names, name_data_pos="top")
        for (ax_logo, ax_info), name in zip(axes, names):
            assert name in [t.get_text() for t in ax_info.texts]
            assert name not in [t.get_text() for t in ax_logo.texts]
        plt.close("all")

    def test_no_info_stays_backward_compatible(self):
        """Test that omitting info still returns a flat list of Axes (no bars)."""
        list_df_logo, _ = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        fig, axes = aal_plot.multi_logo(list_df_logo=list_df_logo)
        assert all(isinstance(ax, plt.Axes) for ax in axes)
        plt.close("all")

    def test_invalid_info_with_list_aal_kws(self):
        """Test that combining list_df_logo_info with list_aal_kws raises ValueError."""
        df_parts, labels = get_df_parts_labels()
        _, list_df_logo_info = get_list_df_logo_and_info()
        list_aal_kws = [dict(df_parts=df_parts, labels=labels, label_test=lt)
                        for lt in [1, 0]]
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_aal_kws=list_aal_kws,
                                list_df_logo_info=list_df_logo_info)

    def test_invalid_info_length_mismatch(self):
        """Test that list_df_logo_info with the wrong count raises ValueError."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=list_df_logo,
                                list_df_logo_info=list_df_logo_info[:1])

    def test_invalid_info_per_group_length_mismatch(self):
        """Test that an info Series of the wrong length raises ValueError."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        bad = [list_df_logo_info[0], list_df_logo_info[1].iloc[:5]]
        aal_plot = get_aal_plot(logo_type="information")
        with pytest.raises(ValueError):
            aal_plot.multi_logo(list_df_logo=list_df_logo, list_df_logo_info=bad)

    def test_invalid_info_not_list(self):
        """Test that a non-list list_df_logo_info raises ValueError."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        for bad in [list_df_logo_info[0], "invalid", 1, []]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo, list_df_logo_info=bad)

    def test_invalid_info_bar_ylim(self):
        """Test that a malformed info_bar_ylim raises ValueError."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        for ylim in [(1,), (5, 1), "invalid", (1, 2, 3)]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo,
                                    list_df_logo_info=list_df_logo_info,
                                    info_bar_ylim=ylim)

    def test_invalid_height_ratio(self):
        """Test that a malformed height_ratio raises ValueError."""
        list_df_logo, list_df_logo_info = get_list_df_logo_and_info()
        aal_plot = get_aal_plot(logo_type="information")
        for hr in [(1,), (0, 6), (-1, 6), "invalid"]:
            with pytest.raises(ValueError):
                aal_plot.multi_logo(list_df_logo=list_df_logo,
                                    list_df_logo_info=list_df_logo_info,
                                    height_ratio=hr)
