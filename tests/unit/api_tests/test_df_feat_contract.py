"""This is a script to test the df_feat interface contract (the documented "data
dictionary" downstream consumers such as ProtXplain pin to).

The contract has two halves, both guarded here so drift is detectable in CI:
- the machine-readable schema ``ut.DICT_DF_FEAT`` stays internally consistent with the
  canonical column order ``ut.LIST_COLS_FEAT``;
- a real, committed ``df_feat`` (``aa.load_features()``) conforms to it — every required
  column is present with the contracted dtype, and the PART-SPLIT-SCALE feature id parses.

A rename/removal of a contracted column, a dtype change, or a feature-id format change
fails one of these tests.
"""
import re

import pandas.api.types as pdt
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _kind(series):
    """Map a pandas dtype to the coarse kind used in DICT_DF_FEAT."""
    if pdt.is_float_dtype(series):
        return "float"
    if pdt.is_integer_dtype(series):
        return "int"
    if pdt.is_string_dtype(series) or series.dtype == object:
        return "str"
    return str(series.dtype)


@pytest.fixture(scope="module")
def df_feat():
    return aa.load_features()


class TestDfFeatSchemaConsistency:
    """The internal data dictionary must agree with the canonical column order."""

    def test_required_set_equals_canonical_lower_bound(self):
        required = {col for col, (_, req, _, _) in ut.DICT_DF_FEAT.items() if req}
        assert required == set(ut.LIST_COLS_FEAT)

    def test_canonical_columns_all_documented(self):
        assert set(ut.LIST_COLS_FEAT).issubset(set(ut.DICT_DF_FEAT))

    def test_entries_well_formed(self):
        for col, entry in ut.DICT_DF_FEAT.items():
            assert isinstance(col, str) and col
            dtype, required, nullable, semantics = entry
            assert dtype in {"str", "float", "int"}
            assert isinstance(required, bool)
            assert isinstance(nullable, bool)
            assert isinstance(semantics, str) and semantics.endswith(".")

    def test_no_duplicate_columns(self):
        assert len(ut.DICT_DF_FEAT) == len(set(ut.DICT_DF_FEAT))


class TestDfFeatGolden:
    """A real df_feat (load_features) must conform to the contract."""

    def test_all_required_columns_present(self, df_feat):
        required = [c for c, (_, req, _, _) in ut.DICT_DF_FEAT.items() if req]
        missing = [c for c in required if c not in df_feat.columns]
        assert not missing, f"required df_feat columns missing: {missing}"

    def test_columns_in_canonical_order(self, df_feat):
        # The canonical columns appear first and in LIST_COLS_FEAT order.
        canonical_present = [c for c in df_feat.columns if c in ut.LIST_COLS_FEAT]
        assert canonical_present == ut.LIST_COLS_FEAT

    def test_dtypes_match_contract(self, df_feat):
        for col in df_feat.columns:
            if col not in ut.DICT_DF_FEAT:
                continue  # dynamic per-substrate column (feat_impact_<name>, ...)
            expected = ut.DICT_DF_FEAT[col][0]
            assert _kind(df_feat[col]) == expected, f"{col}: dtype kind drifted"

    def test_non_nullable_required_columns_have_no_nan(self, df_feat):
        for col, (_, required, nullable, _) in ut.DICT_DF_FEAT.items():
            if required and not nullable and col in df_feat.columns:
                assert df_feat[col].notna().all(), f"{col} unexpectedly contains NaN"


class TestFeatureIdFormat:
    """The PART-SPLIT-SCALE feature-id grammar is part of the contract."""

    SPLIT_TYPES = ("Segment", "Pattern", "PeriodicPattern")

    def test_feature_id_splits_into_three(self, df_feat):
        for feat_id in df_feat[ut.COL_FEATURE]:
            part, split, scale_id = ut.split_feat_id(feat_id)
            assert part and split and scale_id

    def test_split_type_is_known(self, df_feat):
        for feat_id in df_feat[ut.COL_FEATURE]:
            _, split, _ = ut.split_feat_id(feat_id)
            assert split.startswith(self.SPLIT_TYPES), f"unknown split in {feat_id}"

    def test_positions_are_1based_ints(self, df_feat):
        for pos in df_feat[ut.COL_POSITION]:
            nums = [int(p) for p in re.split(r"[,\s]+", str(pos).strip()) if p != ""]
            assert nums and all(n >= 1 for n in nums)
