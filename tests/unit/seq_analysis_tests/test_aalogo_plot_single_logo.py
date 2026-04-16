"""
This is a script to test the AAlogoPlot.single_logo method.
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


def get_df_logo_info(n=50, label_test=1):
    """Load df_logo_info from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].values
    df_parts = sf.get_df_parts(df_seq=df_seq)
    return aa.AAlogo().get_df_logo_info(
        df_parts=df_parts, labels=labels, label_test=label_test)


def get_aal_plot(jmd_n_len=10, jmd_c_len=10, logo_type="probability"):
    """Create default AAlogoPlot instance."""
    return aa.AAlogoPlot(logo_type=logo_type, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)


# ===========================================================================
# I Test single_logo: df_logo
# ===========================================================================
class TestSingleLogoDfLogo:
    """Test single_logo 'df_logo' parameter."""

    def test_valid_df_logo(self):
        """Test valid 'df_logo' produces a figure."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        fig, ax = aal_plot.single_logo(df_logo=df_logo)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_invalid_df_logo_none(self):
        """Test that df_logo=None raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.single_logo(df_logo=None)

    def test_invalid_df_logo_empty(self):
        """Test that empty DataFrame raises ValueError."""
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.single_logo(df_logo=pd.DataFrame())

    def test_invalid_df_logo_not_df(self):
        """Test that non-DataFrame raises ValueError."""
        aal_plot = get_aal_plot()
        for df_logo in ["invalid", 1, [1, 2, 3]]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo)

    def test_df_logo_not_mutated(self):
        """Test that df_logo is not mutated by single_logo (probability ×100 fix)."""
        df_logo = get_df_logo(logo_type="probability")
        max_before = df_logo.values.max()
        aal_plot = get_aal_plot()
        aal_plot.single_logo(df_logo=df_logo)
        assert df_logo.values.max() == max_before
        plt.close("all")


# ===========================================================================
# II Test single_logo: df_logo_info
# ===========================================================================
class TestSingleLogoDfLogoInfo:
    """Test single_logo 'df_logo_info' parameter."""

    def test_valid_df_logo_info(self):
        """Test that valid df_logo_info adds a second panel."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        plt.close("all")

    def test_none_df_logo_info_returns_single_ax(self):
        """Test that df_logo_info=None returns a single Axes."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        fig, ax = aal_plot.single_logo(df_logo=df_logo, df_logo_info=None)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_invalid_df_logo_info_not_series(self):
        """Test that non-Series df_logo_info raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for df_logo_info in [df_logo, "invalid", 1, [1, 2, 3]]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info)

    def test_invalid_df_logo_info_length_mismatch(self):
        """Test that df_logo_info with wrong length raises ValueError."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        wrong_info = df_logo_info.iloc[:5]  # truncate to wrong length
        aal_plot = get_aal_plot()
        with pytest.raises(ValueError):
            aal_plot.single_logo(df_logo=df_logo, df_logo_info=wrong_info)

    def test_ax_info_ylabel_is_bits(self):
        """Test that the info bar panel has 'Bits' as y-label."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info)
        ax_logo, ax_info = axes
        assert ax_info.get_ylabel() == "Bits"
        plt.close("all")


# ===========================================================================
# III Test single_logo: info_bar_color and info_bar_ylim
# ===========================================================================
class TestSingleLogoInfoBar:
    """Test single_logo 'info_bar_color' and 'info_bar_ylim' parameters."""

    def test_valid_info_bar_ylim(self):
        """Test valid 'info_bar_ylim' sets the y-axis limits."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        ylim = (0.0, 2.0)
        fig, axes = aal_plot.single_logo(
            df_logo=df_logo, df_logo_info=df_logo_info, info_bar_ylim=ylim)
        ax_logo, ax_info = axes
        assert ax_info.get_ylim() == ylim
        plt.close("all")

    def test_invalid_info_bar_ylim_not_tuple(self):
        """Test that non-tuple info_bar_ylim raises ValueError."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        for ylim in [[0, 2], "invalid", 1.0]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info,
                                     info_bar_ylim=ylim)

    def test_invalid_info_bar_ylim_min_geq_max(self):
        """Test that info_bar_ylim with min >= max raises ValueError."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        for ylim in [(2.0, 1.0), (1.0, 1.0)]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info,
                                     info_bar_ylim=ylim)


# ===========================================================================
# IV Test single_logo: height_ratio
# ===========================================================================
class TestSingleLogoHeightRatio:
    """Test single_logo 'height_ratio' parameter."""

    def test_valid_height_ratio(self):
        """Test valid 'height_ratio' values."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        for ratio in [(1, 6), (1, 2), (2, 5)]:
            fig, axes = aal_plot.single_logo(
                df_logo=df_logo, df_logo_info=df_logo_info, height_ratio=ratio)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_height_ratio_not_tuple(self):
        """Test that non-tuple height_ratio raises ValueError."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        for ratio in [[1, 6], "invalid", 6]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info,
                                     height_ratio=ratio)

    def test_invalid_height_ratio_non_positive(self):
        """Test that height_ratio with non-positive values raises ValueError."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        for ratio in [(0, 6), (-1, 6), (1, -1)]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info,
                                     height_ratio=ratio)


# ===========================================================================
# V Test single_logo: figsize
# ===========================================================================
class TestSingleLogoFigsize:
    """Test single_logo 'figsize' parameter."""

    def test_valid_figsize(self):
        """Test valid 'figsize' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for figsize in [(8, 4), (12, 3), (6.5, 2.5)]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, figsize=figsize)
            assert tuple(round(x, 1) for x in fig.get_size_inches()) == figsize
            plt.close("all")

    def test_invalid_figsize(self):
        """Test invalid 'figsize' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for figsize in ["invalid", (8,), (8, 4, 2), None]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, figsize=figsize)


# ===========================================================================
# VI Test single_logo: name_data_pos
# ===========================================================================
class TestSingleLogoNameDataPos:
    """Test single_logo 'name_data_pos' parameter."""

    def test_valid_name_data_pos(self):
        """Test all valid 'name_data_pos' options."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for pos in ["top", "right", "bottom", "left"]:
            fig, ax = aal_plot.single_logo(
                df_logo=df_logo, name_data="Test", name_data_pos=pos)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_name_data_pos(self):
        """Test invalid 'name_data_pos' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for pos in ["center", "middle", None, 0]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, name_data_pos=pos)

    def test_name_data_top_with_info_bar_goes_to_ax_info(self):
        """Test that name_data with pos='top' and df_logo_info is placed on ax_info."""
        df_logo = get_df_logo()
        df_logo_info = get_df_logo_info()
        aal_plot = get_aal_plot()
        fig, axes = aal_plot.single_logo(
            df_logo=df_logo, df_logo_info=df_logo_info,
            name_data="Test label", name_data_pos="top")
        ax_logo, ax_info = axes
        # ax_info should have text artists; ax_logo should not have title-style text
        ax_info_texts = [t.get_text() for t in ax_info.texts]
        assert "Test label" in ax_info_texts
        plt.close("all")


# ===========================================================================
# VII Test single_logo: logo_stack_order and weight_tmd_jmd
# ===========================================================================
class TestSingleLogoStringOptions:
    """Test single_logo string option parameters."""

    def test_valid_logo_stack_order(self):
        """Test valid 'logo_stack_order' options."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for order in ["big_on_top", "small_on_top", "fixed"]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, logo_stack_order=order)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_logo_stack_order(self):
        """Test invalid 'logo_stack_order' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for order in ["largest_on_top", "random", None, 0]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, logo_stack_order=order)

    def test_valid_weight_tmd_jmd(self):
        """Test valid 'weight_tmd_jmd' options."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for weight in ["normal", "bold"]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, weight_tmd_jmd=weight)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_weight_tmd_jmd(self):
        """Test invalid 'weight_tmd_jmd' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for weight in ["italic", "heavy", None, 1]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, weight_tmd_jmd=weight)


# ===========================================================================
# VIII Test single_logo: highlight_tmd_area and highlight_alpha
# ===========================================================================
class TestSingleLogoHighlight:
    """Test single_logo highlight parameters."""

    def test_valid_highlight_tmd_area(self):
        """Test valid 'highlight_tmd_area' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for val in [True, False]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, highlight_tmd_area=val)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_highlight_tmd_area(self):
        """Test invalid 'highlight_tmd_area' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for val in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, highlight_tmd_area=val)

    def test_valid_highlight_alpha(self):
        """Test valid 'highlight_alpha' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for alpha in [0.0, 0.15, 0.5, 1.0]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, highlight_alpha=alpha)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_highlight_alpha(self):
        """Test invalid 'highlight_alpha' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for alpha in [-0.1, 1.1, 2.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, highlight_alpha=alpha)


# ===========================================================================
# IX Test single_logo: logo_width, xtick_width, xtick_length
# ===========================================================================
class TestSingleLogoNumericParams:
    """Test single_logo numeric range parameters."""

    def test_valid_logo_width(self):
        """Test valid 'logo_width' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for width in [0.0, 0.5, 0.96, 1.0]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, logo_width=width)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_logo_width(self):
        """Test invalid 'logo_width' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for width in [-0.1, 1.1, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, logo_width=width)

    def test_valid_xtick_width(self):
        """Test valid 'xtick_width' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for width in [0.0, 1.0, 2.0, 5.0]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, xtick_width=width)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_xtick_width(self):
        """Test invalid 'xtick_width' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for width in [-1.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, xtick_width=width)

    def test_valid_xtick_length(self):
        """Test valid 'xtick_length' values."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for length in [0.0, 5.0, 11.0, 20.0]:
            fig, ax = aal_plot.single_logo(df_logo=df_logo, xtick_length=length)
            assert isinstance(fig, plt.Figure)
            plt.close("all")

    def test_invalid_xtick_length(self):
        """Test invalid 'xtick_length' raises ValueError."""
        df_logo = get_df_logo()
        aal_plot = get_aal_plot()
        for length in [-1.0, "invalid", None]:
            with pytest.raises(ValueError):
                aal_plot.single_logo(df_logo=df_logo, xtick_length=length)


# ===========================================================================
# X Test single_logo: check_parts_len (jmd_n_len + jmd_c_len vs df_logo)
# ===========================================================================
class TestSingleLogoPartsLen:
    """Test single_logo part length validation via check_parts_len."""

    def test_valid_parts_len(self):
        """Test that jmd_n_len + jmd_c_len < len(df_logo) passes."""
        df_logo = get_df_logo()   # len = 40 (10 + 20 + 10)
        for jmd_n, jmd_c in [(10, 10), (0, 0), (5, 5), (0, 10)]:
            aal_plot = aa.AAlogoPlot(jmd_n_len=jmd_n, jmd_c_len=jmd_c)
            fig, ax = aal_plot.single_logo(df_logo=df_logo)
            assert isinstance(fig, plt.Figure)
            plt.close("all")


# ===========================================================================
# XI Complex: valid parameter combinations
# ===========================================================================
class TestSingleLogoComplex:
    """Test single_logo with valid parameter combinations."""

    @settings(max_examples=5)
    @given(
        logo_type=st.sampled_from(["probability", "counts", "weight", "information"]),
        highlight_alpha=st.floats(min_value=0.0, max_value=1.0),
        logo_width=st.floats(min_value=0.1, max_value=1.0),  # 0.0 causes tight_layout UserWarning
        highlight_tmd_area=st.booleans(),
    )
    def test_valid_combinations(self, logo_type, highlight_alpha, logo_width, highlight_tmd_area):
        """Test valid parameter combinations produce a figure without error."""
        df_logo = get_df_logo(logo_type=logo_type)
        aal_plot = aa.AAlogoPlot(logo_type=logo_type)
        fig, ax = aal_plot.single_logo(
            df_logo=df_logo,
            highlight_alpha=highlight_alpha,
            logo_width=logo_width,
            highlight_tmd_area=highlight_tmd_area,
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")
