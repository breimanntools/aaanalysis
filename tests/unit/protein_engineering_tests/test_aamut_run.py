"""Tests for AAMut.__init__ and AAMut.run (per-scale substitution impact).

House template: a normal-case ``Test<Method>`` class (one parameter per test, positive via
hypothesis + negative via ``pytest.raises``), a ``Test<Method>Complex`` cross-parameter class,
and a ``Test<Method>GoldenValues`` class with hand-computed expectations.
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

AA = ut.LIST_CANONICAL_AA


class TestAAMutInit:
    """Constructor of AAMut."""

    def test_default(self):
        aam = aa.AAMut()
        assert aam.df_scales.shape[0] == 20

    @settings(max_examples=5, deadline=None)
    @given(verbose=some.booleans())
    def test_verbose(self, verbose):
        aam = aa.AAMut(verbose=verbose)
        assert aam._verbose == verbose

    def test_custom_df_scales(self):
        df = ut.load_default_scales().iloc[:, :5]
        aam = aa.AAMut(df_scales=df)
        assert list(aam.df_scales.columns) == list(df.columns)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aa.AAMut(verbose="yes")

    def test_df_scales_not_dataframe(self):
        with pytest.raises(ValueError):
            aa.AAMut(df_scales=[1, 2, 3])

    def test_df_scales_missing_aa(self):
        df = ut.load_default_scales().drop(index="A")
        with pytest.raises(ValueError):
            aa.AAMut(df_scales=df)

    def test_df_scales_no_columns(self):
        df = ut.load_default_scales().iloc[:, :0]
        with pytest.raises(ValueError):
            aa.AAMut(df_scales=df)


class TestAAMutRun:
    """Normal cases for AAMut.run — one parameter at a time."""

    def test_default_all_pairs(self):
        df = aa.AAMut(df_scales=ut.load_default_scales().iloc[:, :2]).run()
        # 20*19 ordered pairs * 2 scales
        assert len(df) == 20 * 19 * 2
        assert list(df.columns) == ut.COLS_AAMUT

    @settings(max_examples=8, deadline=None)
    @given(from_aa=some.sampled_from(AA))
    def test_from_aa_single(self, from_aa):
        df = aa.AAMut(df_scales=ut.load_default_scales().iloc[:, :3]).run(from_aa=from_aa)
        assert set(df[ut.COL_FROM_AA]) == {from_aa}

    @settings(max_examples=8, deadline=None)
    @given(to_aa=some.sampled_from(AA))
    def test_to_aa_single(self, to_aa):
        df = aa.AAMut(df_scales=ut.load_default_scales().iloc[:, :3]).run(to_aa=to_aa)
        assert set(df[ut.COL_TO_AA]) == {to_aa}

    def test_from_aa_list(self):
        df = aa.AAMut().run(from_aa=["M", "L"], to_aa="V")
        assert set(df[ut.COL_FROM_AA]) == {"M", "L"}

    def test_to_aa_list(self):
        df = aa.AAMut().run(from_aa="M", to_aa=["V", "A"])
        assert set(df[ut.COL_TO_AA]) == {"V", "A"}

    def test_scales_subset(self):
        scales = list(ut.load_default_scales().columns[:3])
        df = aa.AAMut().run(from_aa="M", to_aa="V", scales=scales)
        assert set(df[ut.COL_SCALE_ID]) == set(scales)

    def test_self_pair_excluded(self):
        df = aa.AAMut().run(from_aa="M", to_aa="M")
        assert len(df) == 0

    def test_abs_delta_nonnegative(self):
        df = aa.AAMut().run(from_aa="M", to_aa=["V", "A"])
        assert (df[ut.COL_ABS_DELTA] >= 0).all()

    def test_category_annotated(self):
        df = aa.AAMut().run(from_aa="M", to_aa="V")
        assert df[ut.COL_CAT].notna().any()

    def test_invalid_from_aa(self):
        with pytest.raises(ValueError):
            aa.AAMut().run(from_aa="Z")

    def test_invalid_to_aa(self):
        with pytest.raises(ValueError):
            aa.AAMut().run(to_aa="B")

    def test_invalid_scales(self):
        with pytest.raises(ValueError):
            aa.AAMut().run(from_aa="M", to_aa="V", scales=["NOT_A_SCALE"])

    def test_lowercase_aa_rejected(self):
        with pytest.raises(ValueError):
            aa.AAMut().run(from_aa="m")


class TestAAMutRunComplex:
    """Cross-parameter behaviour of AAMut.run."""

    @settings(max_examples=6, deadline=None)
    @given(n_from=some.integers(min_value=1, max_value=4),
           n_to=some.integers(min_value=1, max_value=4))
    def test_pair_count(self, n_from, n_to):
        scales = list(ut.load_default_scales().columns[:2])
        list_from = AA[:n_from]
        list_to = AA[:n_to]
        df = aa.AAMut().run(from_aa=list_from, to_aa=list_to, scales=scales)
        n_pairs = sum(1 for f in list_from for t in list_to if f != t)
        assert len(df) == n_pairs * len(scales)

    def test_antisymmetry(self):
        scales = list(ut.load_default_scales().columns[:3])
        fwd = aa.AAMut().run(from_aa="M", to_aa="V", scales=scales)
        rev = aa.AAMut().run(from_aa="V", to_aa="M", scales=scales)
        np.testing.assert_allclose(
            fwd.sort_values(ut.COL_SCALE_ID)[ut.COL_DELTA].to_numpy(),
            -rev.sort_values(ut.COL_SCALE_ID)[ut.COL_DELTA].to_numpy())

    def test_columns_order_stable(self):
        df = aa.AAMut().run(from_aa=["M", "L"], to_aa=["V", "A"])
        assert list(df.columns) == ut.COLS_AAMUT


class TestAAMutRunGoldenValues:
    """Hand-computed expectations against df_scales directly."""

    def test_delta_matches_scale_difference(self):
        ds = ut.load_default_scales()
        scale = ds.columns[0]
        df = aa.AAMut(df_scales=ds).run(from_aa="M", to_aa="V", scales=[scale])
        expected = ds.loc["V", scale] - ds.loc["M", scale]
        assert df[ut.COL_DELTA].iloc[0] == pytest.approx(expected)

    def test_abs_delta_equals_abs_delta(self):
        df = aa.AAMut().run(from_aa="K", to_aa="D")
        np.testing.assert_allclose(df[ut.COL_ABS_DELTA], df[ut.COL_DELTA].abs())
