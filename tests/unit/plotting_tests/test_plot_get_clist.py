"""
This is a script for testing the aa.plot_get_clist function.
"""
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _is_color_tuple(color):
    """RGB(A) tuple of floats as returned by seaborn.color_palette."""
    return isinstance(color, tuple) and len(color) in (3, 4)


class TestPlotGetCList:
    """Test plot_get_clist function (one parameter per test)."""

    # Positive testing - n_colors
    @given(n_colors=st.integers(min_value=2, max_value=9))
    def test_categorical_curated(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors)
        assert len(colors) == n_colors
        assert all(isinstance(c, str) for c in colors)

    @given(n_colors=st.integers(min_value=10, max_value=20))
    def test_categorical_husl_range(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors)
        assert len(colors) == n_colors

    # Positive testing - kind
    @given(n_colors=st.integers(min_value=3, max_value=50))
    def test_continuous_ramp(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="continuous", cmap="viridis")
        assert len(colors) == n_colors
        assert all(_is_color_tuple(c) for c in colors)

    def test_continuous_default_husl(self):
        colors = aa.plot_get_clist(n_colors=12, kind="continuous")
        assert len(colors) == 12
        assert all(_is_color_tuple(c) for c in colors)

    def test_continuous_no_upper_cap(self):
        colors = aa.plot_get_clist(n_colors=300, kind="continuous", cmap="viridis")
        assert len(colors) == 300

    @given(n_colors=st.integers(min_value=3, max_value=101))
    def test_diverging_cpp_default(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="diverging")
        assert len(colors) == n_colors
        assert all(_is_color_tuple(c) for c in colors)

    @given(n_colors=st.integers(min_value=3, max_value=60))
    def test_diverging_matplotlib(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="diverging", cmap="coolwarm")
        assert len(colors) == n_colors

    # Positive testing - cmap
    @given(cmap=st.sampled_from(["Set2", "tab20", "husl", "hls"]))
    def test_categorical_cmap(self, cmap):
        colors = aa.plot_get_clist(n_colors=6, kind="categorical", cmap=cmap)
        assert len(colors) == 6

    @given(cmap=st.sampled_from(["RdBu_r", "coolwarm", "Spectral", "CPP", "SHAP"]))
    def test_diverging_cmap_options(self, cmap):
        colors = aa.plot_get_clist(n_colors=11, kind="diverging", cmap=cmap)
        assert len(colors) == 11

    # Positive testing - facecolor_dark
    @given(facecolor_dark=st.booleans())
    def test_diverging_facecolor_dark(self, facecolor_dark):
        colors = aa.plot_get_clist(n_colors=7, kind="diverging", cmap="SHAP",
                                   facecolor_dark=facecolor_dark)
        center = (0, 0, 0) if facecolor_dark else (1, 1, 1)
        assert colors[3] == center

    # Backward-compatibility identity
    def test_default_identity(self):
        assert aa.plot_get_clist(n_colors=3) == ["tab:gray", "tab:blue", "tab:red"]
        assert aa.plot_get_clist() == aa.plot_get_clist(n_colors=3, kind="categorical", cmap=None)

    def test_return_type_categorical(self):
        assert all(isinstance(c, str) for c in aa.plot_get_clist(n_colors=7))

    # Negative testing - n_colors
    @given(n_colors=st.integers(min_value=-1000, max_value=1))
    def test_invalid_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=n_colors)

    @given(n_colors=st.floats())
    def test_non_integer_n_colors(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=n_colors)

    @given(n_colors=st.integers(min_value=21, max_value=500))
    def test_categorical_over_20_raises(self, n_colors):
        with pytest.raises(ValueError, match="20"):
            aa.plot_get_clist(n_colors=n_colors, kind="categorical")

    @given(n_colors=st.integers(min_value=-5, max_value=2))
    def test_continuous_below_3_raises(self, n_colors):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=n_colors, kind="continuous")

    # Negative testing - kind
    @given(kind=st.sampled_from(["cat", "cont", "Diverging", "", "sequential"]))
    def test_invalid_kind(self, kind):
        with pytest.raises(ValueError, match="kind"):
            aa.plot_get_clist(n_colors=3, kind=kind)

    # Negative testing - cmap
    @given(cmap=st.sampled_from(["not_a_cmap", "viridis_x", "rainbow_xyz"]))
    def test_invalid_cmap(self, cmap):
        with pytest.raises(ValueError, match="cmap"):
            aa.plot_get_clist(n_colors=3, kind="continuous", cmap=cmap)

    @given(cmap=st.sampled_from(["CPP", "SHAP"]))
    def test_house_cmap_rejected_for_categorical(self, cmap):
        with pytest.raises(ValueError, match="diverging"):
            aa.plot_get_clist(n_colors=3, kind="categorical", cmap=cmap)

    @given(cmap=st.sampled_from(["CPP", "SHAP"]))
    def test_house_cmap_rejected_for_continuous(self, cmap):
        with pytest.raises(ValueError, match="diverging"):
            aa.plot_get_clist(n_colors=3, kind="continuous", cmap=cmap)

    # Negative testing - facecolor_dark
    @given(facecolor_dark=st.sampled_from(["yes", 1, 0, None, "True"]))
    def test_invalid_facecolor_dark(self, facecolor_dark):
        with pytest.raises(ValueError, match="facecolor_dark"):
            aa.plot_get_clist(n_colors=3, kind="diverging", facecolor_dark=facecolor_dark)


class TestPlotGetCListComplex:
    """Test plot_get_clist with crossed parameters and edge interactions."""

    # Positive combinations
    @given(n_colors=st.integers(min_value=2, max_value=20))
    def test_categorical_cmap_any_size(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="categorical", cmap="tab20")
        assert len(colors) == n_colors

    @given(n_colors=st.integers(min_value=3, max_value=40),
           cmap=st.sampled_from(["viridis", "magma", "husl", "hls", "Blues"]))
    def test_continuous_various_cmaps(self, n_colors, cmap):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="continuous", cmap=cmap)
        assert len(colors) == n_colors

    @given(n_colors=st.integers(min_value=3, max_value=51).filter(lambda n: n % 2 == 1))
    def test_diverging_cpp_center_white(self, n_colors):
        colors = aa.plot_get_clist(n_colors=n_colors, kind="diverging", cmap="CPP",
                                   facecolor_dark=False)
        assert colors[n_colors // 2] == (1, 1, 1)

    def test_diverging_facecolor_toggle_differs(self):
        light = aa.plot_get_clist(n_colors=9, kind="diverging", cmap="CPP", facecolor_dark=False)
        dark = aa.plot_get_clist(n_colors=9, kind="diverging", cmap="CPP", facecolor_dark=True)
        assert light != dark

    def test_continuous_husl_default_matches_explicit(self):
        assert aa.plot_get_clist(n_colors=15, kind="continuous") == \
               aa.plot_get_clist(n_colors=15, kind="continuous", cmap="husl")

    def test_diverging_cpp_default_matches_explicit(self):
        assert aa.plot_get_clist(n_colors=11, kind="diverging") == \
               aa.plot_get_clist(n_colors=11, kind="diverging", cmap="CPP")

    # Negative combinations
    def test_categorical_over_cap_with_cmap(self):
        with pytest.raises(ValueError, match="20"):
            aa.plot_get_clist(n_colors=25, kind="categorical", cmap="tab20")

    def test_continuous_unknown_cmap(self):
        with pytest.raises(ValueError, match="cmap"):
            aa.plot_get_clist(n_colors=5, kind="continuous", cmap="definitely_not_real")

    def test_diverging_below_min_and_bad_facecolor(self):
        with pytest.raises(ValueError):
            aa.plot_get_clist(n_colors=2, kind="diverging")
        with pytest.raises(ValueError, match="facecolor_dark"):
            aa.plot_get_clist(n_colors=7, kind="diverging", facecolor_dark="dark")

    def test_invalid_kind_with_valid_cmap(self):
        with pytest.raises(ValueError, match="kind"):
            aa.plot_get_clist(n_colors=5, kind="sequential", cmap="viridis")

    def test_categorical_cmap_none_curated_strings(self):
        colors = aa.plot_get_clist(n_colors=4, kind="categorical", cmap=None)
        assert all(isinstance(c, str) for c in colors)
