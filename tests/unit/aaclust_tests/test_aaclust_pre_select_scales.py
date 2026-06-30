"""This is a script to test AAclust.pre_select_scales()."""

import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions
def get_df_scales():
    """Return the full bundled scales DataFrame (rows = amino acids, columns = scale IDs)."""
    return aa.load_scales()


def get_df_cat():
    """Return the bundled scale-category table."""
    return aa.load_scales(name="scales_cat")


def cats():
    """Unique categories present in the bundled category table."""
    return sorted(get_df_cat()["category"].unique())


def subcats():
    """Unique subcategories present in the bundled category table."""
    return sorted(get_df_cat()["subcategory"].unique())


def ids_with(df_cat, col, names):
    """scale_ids whose `col` value is in `names`."""
    return set(df_cat[df_cat[col].isin(names)]["scale_id"])


class TestPreSelectScales:
    """Positive and negative tests for pre_select_scales() — one per parameter."""

    # --- Positive: per parameter ---
    def test_df_scales_parameter(self):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales)
        assert isinstance(df_pre, pd.DataFrame)
        assert list(df_pre.index) == list(df_scales.index)
        # No exclusion -> all scales kept
        assert df_pre.shape[1] == df_scales.shape[1]

    def test_df_cat_parameter_explicit(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        assert isinstance(df_pre, pd.DataFrame)
        assert df_pre.shape[1] < df_scales.shape[1]

    def test_df_cat_none_uses_bundled(self):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        df_pre_default = aac.pre_select_scales(df_scales=df_scales, cat_out=["Composition"])
        df_pre_explicit = aac.pre_select_scales(df_scales=df_scales, df_cat=get_df_cat(),
                                                cat_out=["Composition"])
        assert list(df_pre_default.columns) == list(df_pre_explicit.columns)

    @settings(max_examples=5, deadline=None)
    @given(cat=some.sampled_from(cats()))
    def test_cat_out_parameter(self, cat):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=[cat])
        removed = ids_with(df_cat, "category", [cat])
        assert removed.isdisjoint(set(df_pre.columns))
        assert df_pre.shape[1] == df_scales.shape[1] - len(removed & set(df_scales.columns))

    @settings(max_examples=5, deadline=None)
    @given(subcat=some.sampled_from(subcats()))
    def test_subcat_out_parameter(self, subcat):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=[subcat])
        removed = ids_with(df_cat, "subcategory", [subcat])
        assert removed.isdisjoint(set(df_pre.columns))

    def test_cat_out_accepts_single_string(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_list = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        df_str = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out="Composition")
        assert list(df_list.columns) == list(df_str.columns)

    def test_subcat_out_accepts_single_string(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        sc = subcats()[0]
        aac = aa.AAclust()
        df_list = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=[sc])
        df_str = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=sc)
        assert list(df_list.columns) == list(df_str.columns)

    def test_cat_out_none_is_noop(self):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, cat_out=None)
        assert df_pre.shape[1] == df_scales.shape[1]

    def test_subcat_out_none_is_noop(self):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, subcat_out=None)
        assert df_pre.shape[1] == df_scales.shape[1]

    def test_column_order_preserved(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        kept_in_order = [c for c in df_scales.columns if c in set(df_pre.columns)]
        assert list(df_pre.columns) == kept_in_order

    def test_returns_value_subset(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=["Hydrophilicity"])
        # Kept columns carry the unchanged scale values
        common = list(df_pre.columns)[:3]
        for col in common:
            assert (df_pre[col] == df_scales[col]).all()

    # --- Negative: per parameter ---
    def test_cat_out_unknown_raises(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        with pytest.raises(ValueError, match="cat_out"):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["NoSuchCategory"])

    def test_subcat_out_unknown_raises(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        with pytest.raises(ValueError, match="subcat_out"):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=["NoSuchSubcat"])

    def test_cat_out_one_unknown_among_valid_raises(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        with pytest.raises(ValueError, match="cat_out"):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition", "Nope"])

    @pytest.mark.parametrize("bad", [123, 4.5, {"a": 1}])
    def test_cat_out_wrong_type_raises(self, bad):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.pre_select_scales(df_scales=df_scales, cat_out=bad)

    @pytest.mark.parametrize("bad", [123, 4.5, {"a": 1}])
    def test_subcat_out_wrong_type_raises(self, bad):
        df_scales = get_df_scales()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.pre_select_scales(df_scales=df_scales, subcat_out=bad)

    @pytest.mark.parametrize("bad", [None, "not_a_df", 42, [1, 2, 3]])
    def test_df_scales_invalid_raises(self, bad):
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.pre_select_scales(df_scales=bad, cat_out=["Composition"])

    def test_df_cat_missing_column_raises(self):
        df_scales = get_df_scales()
        df_cat_bad = get_df_cat().drop(columns=["subcategory"])
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat_bad, cat_out=["Composition"])


class TestPreSelectScalesComplex:
    """Combinations and edge interactions for pre_select_scales()."""

    def test_cat_and_subcat_out_combined(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        sc = [s for s in subcats() if s != "Hydrophilicity"][0]
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat,
                                       cat_out=["Composition"], subcat_out=[sc])
        removed = (ids_with(df_cat, "category", ["Composition"])
                   | ids_with(df_cat, "subcategory", [sc]))
        assert removed.isdisjoint(set(df_pre.columns))

    def test_combined_equals_union_of_separate(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        sc = subcats()[0]
        aac = aa.AAclust()
        df_both = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat,
                                        cat_out=["Composition"], subcat_out=[sc])
        df_a = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        df_b = aac.pre_select_scales(df_scales=df_a, df_cat=df_cat, subcat_out=[sc])
        assert list(df_both.columns) == list(df_b.columns)

    def test_idempotent(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        once = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        twice = aac.pre_select_scales(df_scales=once, df_cat=df_cat, cat_out=["Composition"])
        assert list(once.columns) == list(twice.columns)

    def test_no_excluded_subcat_remains(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        out = subcats()[:3]
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, subcat_out=out)
        kept_subcats = set(df_cat[df_cat["scale_id"].isin(df_pre.columns)]["subcategory"])
        assert kept_subcats.isdisjoint(set(out))

    def test_composes_with_select_scales(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        df_sel = aac.select_scales(df_scales=df_pre, n_clusters=20)
        assert df_sel.shape[1] == 20
        assert set(df_sel.columns).issubset(set(df_pre.columns))

    def test_composes_with_filter_coverage(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        ids = list(df_pre.columns)
        names_ref = df_cat[df_cat["scale_id"].isin(ids)]["subcategory"].to_list()
        selected = aac.filter_coverage(X=df_pre.T, scale_ids=ids, names_ref=names_ref,
                                       min_coverage=100, df_cat=df_cat)
        assert set(selected).issubset(set(ids))

    def test_excluding_unrelated_category_keeps_subcat_scales(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        # A subcategory that is NOT inside "Composition" survives a Composition exclusion
        non_comp = df_cat[df_cat["category"] != "Composition"]["subcategory"].iloc[0]
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        kept_subcats = set(df_cat[df_cat["scale_id"].isin(df_pre.columns)]["subcategory"])
        assert non_comp in kept_subcats

    def test_subset_input_only_drops_present_columns(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        small = df_scales[list(df_scales.columns)[:40]]
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=small, df_cat=df_cat, cat_out=["Composition"])
        assert set(df_pre.columns).issubset(set(small.columns))

    # --- Negative combinations ---
    def test_both_unknown_raises(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat,
                                  cat_out=["Nope"], subcat_out=["AlsoNope"])

    def test_valid_cat_unknown_subcat_raises(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        with pytest.raises(ValueError, match="subcat_out"):
            aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat,
                                  cat_out=["Composition"], subcat_out=["Nope"])


class TestPreSelectScalesGoldenValues:
    """Hand-checked invariants on the bundled data."""

    def test_composition_removal_count(self):
        df_scales = get_df_scales()
        df_cat = get_df_cat()
        aac = aa.AAclust()
        n_comp = df_cat[df_cat["category"] == "Composition"]["scale_id"].isin(df_scales.columns).sum()
        df_pre = aac.pre_select_scales(df_scales=df_scales, df_cat=df_cat, cat_out=["Composition"])
        assert df_pre.shape[1] == df_scales.shape[1] - int(n_comp)
        assert "Composition" not in set(
            df_cat[df_cat["scale_id"].isin(df_pre.columns)]["category"])

    def test_paper_set_shape(self):
        # The γ-secretase "interpretable (broad)" exclusion list
        df_scales_clf = aa.load_scales(unclassified_out=True)
        df_cat = get_df_cat()
        subcat_out = ["β-turn", "β-turn (C-term)", "β-turn (N-term)", "Hydrophilicity"]
        aac = aa.AAclust()
        df_pre = aac.pre_select_scales(df_scales=df_scales_clf, df_cat=df_cat,
                                       cat_out=["Composition"], subcat_out=subcat_out)
        removed = (ids_with(df_cat, "category", ["Composition"])
                   | ids_with(df_cat, "subcategory", subcat_out))
        expected = [c for c in df_scales_clf.columns if c not in removed]
        assert list(df_pre.columns) == expected
