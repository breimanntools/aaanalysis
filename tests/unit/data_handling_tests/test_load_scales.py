"""
This is a script for testing the aa.load_scales function.
"""
from hypothesis import given, settings, example
import hypothesis.strategies as some
import aaanalysis as aa
from pandas import DataFrame
import pytest

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


class TestLoadScales:
    """Test load_scales function"""

    # Basic positive tests
    def test_load_scales_default(self):
        """Test the default parameters."""
        df = aa.load_scales()
        assert isinstance(df, DataFrame)

    def test_load_scales_names(self):
        """Test different dataset names."""
        for name in ["scales", "scales_raw", "scales_cat", "scales_pc", "top60", "top60_eval"]:
            df = aa.load_scales(name=name)
            assert isinstance(df, DataFrame)

    def test_load_scales_just_aaindex(self):
        """Test the 'just_aaindex' parameter."""
        df = aa.load_scales(just_aaindex=True)
        assert isinstance(df, DataFrame)

    def test_load_scales_unclassified_in(self):
        """Test the 'unclassified_in' parameter."""
        df = aa.load_scales(unclassified_out=True)
        assert isinstance(df, DataFrame)

    @settings(max_examples=10, deadline=1500)
    @given(top60_n=some.integers(min_value=1, max_value=60))
    def test_load_scales_top60_n(self, top60_n):
        """Test the 'top60_n' parameter."""
        df = aa.load_scales(name="scales", top60_n=top60_n)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_name(self):
        """Test with an invalid dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name="invalid_name")

    def test_invalid_name_type(self):
        """Test with a non-string dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name=123)

    def test_empty_name(self):
        """Test with an empty string as dataset name."""
        with pytest.raises(ValueError):
            aa.load_scales(name="")


class TestLoadScalesComplex:
    """Test load_scales function with complex scenarios"""

    # Positive tests
    def test_load_scales_both_filters(self):
        """Test both 'just_aaindex' and 'unclassified_in' together."""
        df = aa.load_scales(just_aaindex=True, unclassified_out=True)
        assert isinstance(df, DataFrame)

    def test_load_scales_all_params(self):
        """Test all parameters together."""
        df = aa.load_scales(name="scales", just_aaindex=True, unclassified_out=True, top60_n=10)
        assert isinstance(df, DataFrame)

    def test_load_scales_name_and_filters(self):
        """Test 'name' with 'just_aaindex' and 'unclassified_in' together."""
        df = aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_combination(self):
        """Test with all invalid parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False, top60_n=-5)

    def test_invalid_complex_scenario_1(self):
        """Test a complex combination of parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="hydrophobicity", just_aaindex="yes", top60_n="60")


class TestLoadScalesVeryComplex:
    """Test load_scales function with very complex scenarios"""

    # Positive tests
    def test_load_scales_all_filters_with_top60(self):
        """Test all filters ('just_aaindex', 'unclassified_in', 'top60_n')."""
        df = aa.load_scales(just_aaindex=True, unclassified_out=False, top60_n=5)
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_all_filters(self):
        """Test with all invalid filters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales_cat", just_aaindex=True, unclassified_out=False, top60_n=100)

    def test_invalid_very_complex_scenario_1(self):
        """Test a very complex scenario with extreme values."""
        with pytest.raises(ValueError):
            aa.load_scales(name="hydrophobicity" * 1000, just_aaindex=True)

    def test_invalid_very_complex_scenario_2(self):
        """Test a very complex scenario with conflicting parameters."""
        with pytest.raises(ValueError):
            aa.load_scales(name="some_name", top60_n=-100, unclassified_out="yes")

    def test_invalid_very_complex_scenario_3(self):
        """Test a very complex scenario with both invalid and out-of-bounds values."""
        with pytest.raises(ValueError):
            aa.load_scales(name="some_invalid_name", just_aaindex="not a boolean", top60_n=2000)


VALID_TIERS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
VALID_MIN_THS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TestLoadScalesTopExplain:
    """Test the interpretability-tier selector (top_explain_n / top_explain_min_th)."""

    # Positive tests (one parameter per test)
    @given(top_explain_n=some.sampled_from(VALID_TIERS))
    @settings(max_examples=12, deadline=1500)
    def test_top_explain_n_scales(self, top_explain_n):
        """Each valid tier returns a non-empty scales DataFrame."""
        df = aa.load_scales(name="scales", top_explain_n=top_explain_n)
        assert isinstance(df, DataFrame) and df.shape[1] > 0

    @given(top_explain_n=some.sampled_from(VALID_TIERS))
    @settings(max_examples=12, deadline=1500)
    def test_top_explain_n_cat_columns_and_threshold(self, top_explain_n):
        """A tier selection on scales_cat adds the two columns and respects the tier."""
        df_cat = aa.load_scales(name="scales_cat", top_explain_n=top_explain_n)
        assert "interpretability" in df_cat.columns and "top_explain" in df_cat.columns
        assert (df_cat["top_explain"] <= top_explain_n).all()

    @given(top_explain_n=some.sampled_from(VALID_TIERS))
    @settings(max_examples=12, deadline=1500)
    def test_top_explain_n_scales_raw(self, top_explain_n):
        """The tier selector also works for scales_raw."""
        df = aa.load_scales(name="scales_raw", top_explain_n=top_explain_n)
        assert isinstance(df, DataFrame) and df.shape[1] > 0

    @given(min_th=some.sampled_from(VALID_MIN_THS))
    @settings(max_examples=7, deadline=2000)
    def test_top_explain_min_th(self, min_th):
        """Each valid min_th returns a redundancy-reduced DataFrame."""
        df = aa.load_scales(name="scales", top_explain_n=40, top_explain_min_th=min_th)
        assert isinstance(df, DataFrame) and df.shape[1] > 0

    def test_top_explain_just_aaindex_drops_non_aaindex(self):
        """just_aaindex removes all LINS/KOEH scales from a tier selection."""
        df = aa.load_scales(name="scales", top_explain_n=20, just_aaindex=True)
        assert not any(("LINS" in s) or ("KOEH" in s) for s in df.columns)

    def test_default_cat_schema_unchanged(self):
        """Without top_explain_n the default df_cat schema has no extra columns."""
        df_cat = aa.load_scales(name="scales_cat")
        assert "interpretability" not in df_cat.columns
        assert "top_explain" not in df_cat.columns

    # Negative tests
    def test_invalid_tier_value(self):
        """A tier not on the grid raises."""
        for bad in [0, 7, 62, 65, 67, 100]:
            with pytest.raises(ValueError):
                aa.load_scales(name="scales", top_explain_n=bad)

    def test_invalid_min_th_value(self):
        """A min_th not on the grid raises."""
        for bad in [0.0, 0.25, 0.55, 1.0, 2.0]:
            with pytest.raises(ValueError):
                aa.load_scales(name="scales", top_explain_n=20, top_explain_min_th=bad)

    def test_min_th_without_tier_raises(self):
        """min_th without top_explain_n raises."""
        with pytest.raises(ValueError, match="top_explain_min_th"):
            aa.load_scales(name="scales", top_explain_min_th=0.5)

    def test_invalid_name_for_tier(self):
        """top_explain_n is invalid for non-scale names."""
        for name in ["scales_pc", "top60", "top60_eval"]:
            with pytest.raises(ValueError):
                aa.load_scales(name=name, top_explain_n=20)


class TestLoadScalesTopExplainComplex:
    """Combinations and golden/determinism checks for the interpretability tier."""

    # Positive tests
    def test_golden_tier5_subcategories(self):
        """Tier 5 contains exactly the five most interpretable subcategories."""
        df_cat = aa.load_scales(name="scales_cat", top_explain_n=5)
        assert set(df_cat["subcategory"]) == {"Volume", "Coil", "α-helix", "β-sheet", "Hydrophobicity"}

    def test_golden_tier5_scale_count(self):
        """Tier 5 returns all 125 member scales (no redundancy reduction)."""
        df = aa.load_scales(name="scales", top_explain_n=5)
        assert df.shape[1] == 125

    def test_tiers_are_cumulative(self):
        """Higher tiers are supersets of lower tiers (column-level)."""
        c10 = set(aa.load_scales(name="scales", top_explain_n=10).columns)
        c5 = set(aa.load_scales(name="scales", top_explain_n=5).columns)
        assert c5.issubset(c10)

    def test_min_th_reduces_scale_count(self):
        """Redundancy reduction yields no more scales than the full tier."""
        full = aa.load_scales(name="scales", top_explain_n=30).shape[1]
        reduced = aa.load_scales(name="scales", top_explain_n=30, top_explain_min_th=0.5).shape[1]
        assert reduced <= full

    def test_deterministic(self):
        """Same selection returns identical scale columns (precomputed)."""
        a = aa.load_scales(name="scales", top_explain_n=20, top_explain_min_th=0.5)
        b = aa.load_scales(name="scales", top_explain_n=20, top_explain_min_th=0.5)
        assert list(a.columns) == list(b.columns)

    def test_just_aaindex_min_th_combo(self):
        """min_th + just_aaindex returns an AAindex-only reduced set."""
        df = aa.load_scales(name="scales", top_explain_n=20, top_explain_min_th=0.5, just_aaindex=True)
        assert df.shape[1] > 0
        assert not any(("LINS" in s) or ("KOEH" in s) for s in df.columns)

    # Negative tests
    def test_mutual_exclusion_with_top60(self):
        """top_explain_n and top60_n cannot be combined."""
        with pytest.raises(ValueError, match="top60_n"):
            aa.load_scales(name="scales", top_explain_n=20, top60_n=1)

    def test_invalid_tier_with_min_th(self):
        """An invalid tier still raises even when min_th is valid."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales", top_explain_n=7, top_explain_min_th=0.5)

    def test_invalid_min_th_with_just_aaindex(self):
        """An invalid min_th raises regardless of just_aaindex."""
        with pytest.raises(ValueError):
            aa.load_scales(name="scales", top_explain_n=20, top_explain_min_th=0.55, just_aaindex=True)
