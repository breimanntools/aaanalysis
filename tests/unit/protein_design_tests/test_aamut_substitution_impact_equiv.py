"""Equivalence test for the Batch-6 vectorization of
``comp_substitution_impact`` (AAMut backend, #186).

The original built one small ``pd.DataFrame`` per ordered (from_aa, to_aa) pair
in a nested loop and ``pd.concat``-ed hundreds of them. The new path accumulates
the (from, to, delta) columns and constructs ONE DataFrame. This must be
byte-identical: same columns, dtypes, row order and values. Here the
slow per-pair-concat formulation is reproduced inline as the reference and
pinned against the shipped backend.
"""
import numpy as np
import pandas as pd

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.protein_design._backend.aamut.aamut import (
    comp_substitution_impact, _get_df_cat_lookup,
)

aa.options["verbose"] = False


def _ref_per_pair_concat(df_scales, df_cat, list_from, list_to, list_scales):
    """Original behaviour: one DataFrame per (from, to) pair, then concat."""
    sub = df_scales[list_scales]
    cat_map, subcat_map = _get_df_cat_lookup(df_cat=df_cat)
    records = []
    n_scales = len(list_scales)
    arr_scale_id = np.asarray(list_scales)
    arr_cat = np.asarray([cat_map.get(s, np.nan) for s in list_scales], dtype=object)
    arr_subcat = np.asarray([subcat_map.get(s, np.nan) for s in list_scales], dtype=object)
    for from_aa in list_from:
        row_from = sub.loc[from_aa].to_numpy(dtype=float)
        for to_aa in list_to:
            if to_aa == from_aa:
                continue
            delta = sub.loc[to_aa].to_numpy(dtype=float) - row_from
            records.append(pd.DataFrame({
                ut.COL_FROM_AA: np.repeat(from_aa, n_scales),
                ut.COL_TO_AA: np.repeat(to_aa, n_scales),
                ut.COL_SCALE_ID: arr_scale_id,
                ut.COL_CAT: arr_cat,
                ut.COL_SUBCAT: arr_subcat,
                ut.COL_DELTA: delta,
                ut.COL_ABS_DELTA: np.abs(delta),
            }))
    if not records:
        return pd.DataFrame(columns=ut.COLS_AAMUT)
    return pd.concat(records, axis=0, ignore_index=True)[ut.COLS_AAMUT].reset_index(drop=True)


class TestSubstitutionImpactEquivalence:
    """Vectorized comp_substitution_impact == per-pair-concat reference."""

    def _load(self):
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        return df_scales, df_cat

    def test_full_canonical_byte_identical(self):
        """All 20x20 canonical pairs over 60 scales: byte-identical frame."""
        df_scales, df_cat = self._load()
        list_scales = list(df_scales.columns[:60])
        aas = list(ut.LIST_CANONICAL_AA)
        got = comp_substitution_impact(df_scales=df_scales, df_cat=df_cat,
                                       list_from=aas, list_to=aas, list_scales=list_scales)
        ref = _ref_per_pair_concat(df_scales, df_cat, aas, aas, list_scales)
        assert list(got.columns) == ut.COLS_AAMUT
        assert list(got.dtypes) == list(ref.dtypes)
        assert got.equals(ref)

    def test_asymmetric_lists_byte_identical(self):
        """Distinct, shorter from/to lists keep nested row order + values."""
        df_scales, df_cat = self._load()
        list_scales = list(df_scales.columns[:25])
        list_from = ["A", "C", "D", "E"]
        list_to = ["K", "R", "A", "W"]
        got = comp_substitution_impact(df_scales=df_scales, df_cat=df_cat,
                                       list_from=list_from, list_to=list_to,
                                       list_scales=list_scales)
        ref = _ref_per_pair_concat(df_scales, df_cat, list_from, list_to, list_scales)
        assert got.equals(ref)

    def test_no_cat_lookup_byte_identical(self):
        """Without df_cat, category/subcategory columns stay NaN identically."""
        df_scales, _ = self._load()
        list_scales = list(df_scales.columns[:15])
        aas = ["A", "G", "L", "V", "F"]
        got = comp_substitution_impact(df_scales=df_scales, df_cat=None,
                                       list_from=aas, list_to=aas, list_scales=list_scales)
        ref = _ref_per_pair_concat(df_scales, None, aas, aas, list_scales)
        assert got.equals(ref)

    def test_all_pairs_skipped_returns_empty(self):
        """from==to for every pair -> empty frame with the canonical columns."""
        df_scales, df_cat = self._load()
        list_scales = list(df_scales.columns[:5])
        got = comp_substitution_impact(df_scales=df_scales, df_cat=df_cat,
                                       list_from=["A"], list_to=["A"], list_scales=list_scales)
        assert list(got.columns) == ut.COLS_AAMUT
        assert len(got) == 0

    def test_delta_is_signed_to_minus_from(self):
        """Spot-check the actual numeric contract on a single pair."""
        df_scales, df_cat = self._load()
        list_scales = list(df_scales.columns[:8])
        got = comp_substitution_impact(df_scales=df_scales, df_cat=df_cat,
                                       list_from=["A"], list_to=["L"], list_scales=list_scales)
        expected = (df_scales.loc["L", list_scales].to_numpy(dtype=float)
                    - df_scales.loc["A", list_scales].to_numpy(dtype=float))
        np.testing.assert_allclose(got[ut.COL_DELTA].to_numpy(), expected, atol=1e-12)
        np.testing.assert_allclose(got[ut.COL_ABS_DELTA].to_numpy(), np.abs(expected), atol=1e-12)
