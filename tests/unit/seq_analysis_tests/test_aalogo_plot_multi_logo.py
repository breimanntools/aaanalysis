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

# Set default deadline from 200 to 20000
settings.register_profile("ci", deadline=20000)
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
