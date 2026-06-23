"""This is a script to test the DataFrame data dictionary / interface contract
(``ut.DICT_DF_SCHEMAS``) for the key AAanalysis DataFrames.

Three layers, all guarded so drift is detectable in CI:
- structural: every frame + field record is well-formed, and the rich df_feat view agrees
  with the simple ``ut.DICT_DF_FEAT``;
- golden conformance: real frames (load_dataset / get_df_parts / load_scales /
  load_features / AAlogo) satisfy their schema (columns, dtypes, ranges, uniqueness);
- cross-frame contract: a real df_feat references valid scales/parts/categories, and its
  numeric columns are finite and in range;
- doc sync: the committed schemas doc page matches the rendered registry.
"""
import pathlib
import re

import numpy as np
import pandas.api.types as pdt
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False


def _kind(series):
    if pdt.is_bool_dtype(series):
        return "bool"
    if pdt.is_float_dtype(series):
        return "float"
    if pdt.is_integer_dtype(series):
        return "int"
    if pdt.is_string_dtype(series) or series.dtype == object:
        return "str"
    return str(series.dtype)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def df_seq():
    return aa.load_dataset(name="DOM_GSEC", n=5)


@pytest.fixture(scope="module")
def df_feat():
    return aa.load_features()


@pytest.fixture(scope="module")
def df_cat():
    return aa.load_scales(name="scales_cat")


@pytest.fixture(scope="module")
def df_scales():
    return aa.load_scales()


# ------------------------------------------------------------------------- structural
class TestSchemaStructure:
    def test_every_frame_documented(self):
        for frame, spec in ut.DICT_DF_SCHEMAS.items():
            assert isinstance(spec.get("description"), str) and spec["description"]
            assert any(k in spec for k in ("columns", "matrix", "dynamic_columns")), frame

    def test_field_records_well_formed(self):
        for frame, spec in ut.DICT_DF_SCHEMAS.items():
            for col, rec in spec.get("columns", {}).items():
                assert rec["dtype"] in {"str", "float", "int", "bool"}, (frame, col)
                assert isinstance(rec["required"], bool)
                assert isinstance(rec["nullable"], bool)
                assert isinstance(rec["unique"], bool)
                assert isinstance(rec["description"], str) and rec["description"].endswith(".")
                if "range" in rec:
                    assert len(rec["range"]) == 2
                if "allowed_values" in rec:
                    assert isinstance(rec["allowed_values"], list) and rec["allowed_values"]

    def test_expected_frames_present(self):
        for frame in ["df_seq", "df_parts", "df_scales", "df_cat", "df_subcat",
                      "df_feat", "df_eval", "X", "prediction"]:
            assert frame in ut.DICT_DF_SCHEMAS

    def test_rich_df_feat_agrees_with_simple_dict(self):
        # The rich registry and the simple DICT_DF_FEAT must not drift on the df_feat
        # columns (set, required flag, dtype).
        rich = ut.DICT_DF_SCHEMAS["df_feat"]["columns"]
        simple = ut.DICT_DF_FEAT
        assert set(rich) == set(simple)
        for col, (dtype, required, nullable, _semantics) in simple.items():
            assert rich[col]["dtype"] == dtype, col
            assert rich[col]["required"] == required, col


# -------------------------------------------------------------------- golden conformance
class TestGoldenConformance:
    @staticmethod
    def _check_columns(df, schema_cols):
        for col in df.columns:
            if col not in schema_cols:
                continue  # dynamic / post-fit extra column
            rec = schema_cols[col]
            assert _kind(df[col]) == rec["dtype"], f"{col}: dtype kind drift"
            if rec["required"] and not rec["nullable"]:
                assert df[col].notna().all(), f"{col} unexpectedly NaN"
            if rec["unique"]:
                assert df[col].is_unique, f"{col} not unique"
            if "range" in rec and _kind(df[col]) in {"int", "float"}:
                lo, hi = rec["range"]
                vals = df[col].dropna()
                if lo is not None:
                    assert (vals >= lo - 1e-9).all(), f"{col} below range"
                if hi is not None:
                    assert (vals <= hi + 1e-9).all(), f"{col} above range"

    def test_df_seq_conforms(self, df_seq):
        self._check_columns(df_seq, ut.DICT_DF_SCHEMAS["df_seq"]["columns"])
        assert ut.COL_ENTRY in df_seq.columns and df_seq[ut.COL_ENTRY].is_unique

    def test_df_cat_conforms(self, df_cat):
        self._check_columns(df_cat, ut.DICT_DF_SCHEMAS["df_cat"]["columns"])

    def test_df_feat_conforms(self, df_feat):
        self._check_columns(df_feat, ut.DICT_DF_SCHEMAS["df_feat"]["columns"])

    def test_df_parts_dynamic_columns(self):
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=aa.load_dataset(name="DOM_GSEC", n=3))
        allowed = set(ut.DICT_DF_SCHEMAS["df_parts"]["dynamic_columns"]["allowed_names"])
        assert set(df_parts.columns).issubset(allowed)
        assert all(_kind(df_parts[c]) == "str" for c in df_parts.columns)

    def test_df_scales_matrix(self, df_scales):
        spec = ut.DICT_DF_SCHEMAS["df_scales"]["matrix"]
        assert spec["dtype"] == "float"
        assert pdt.is_float_dtype(df_scales.iloc[:, 0])
        assert set(df_scales.index).issubset(set(ut.LIST_CANONICAL_AA))


# ---------------------------------------------------------------------- cross-frame
class TestCrossFrameContract:
    def test_feature_id_references_valid_scale_and_part(self, df_feat, df_scales):
        valid_scales = set(df_scales.columns)
        valid_parts = set(ut.LIST_ALL_PARTS)
        for feat_id in df_feat[ut.COL_FEATURE]:
            part, split, scale_id = ut.split_feat_id(feat_id)
            assert scale_id in valid_scales, f"{scale_id} not in df_scales"
            # The feature id uppercases the part name (e.g. 'TMD_C_JMD_C').
            assert part.lower() in valid_parts, f"{part} not a known part"
            assert split.startswith(("Segment", "Pattern", "PeriodicPattern"))

    def test_feature_unique(self, df_feat):
        assert df_feat[ut.COL_FEATURE].is_unique

    def test_category_in_allowed_values(self, df_feat):
        allowed = set(ut.DICT_DF_SCHEMAS["df_feat"]["columns"][ut.COL_CAT]["allowed_values"])
        assert set(df_feat[ut.COL_CAT]).issubset(allowed)

    def test_numeric_columns_finite_and_in_range(self, df_feat):
        for col, rec in ut.DICT_DF_SCHEMAS["df_feat"]["columns"].items():
            if rec["dtype"] != "float" or col not in df_feat.columns:
                continue
            vals = df_feat[col].to_numpy(dtype=float)
            assert np.isfinite(vals).all(), f"{col} has non-finite values"


# ------------------------------------------------------------------------- doc sync
class TestDocSync:
    def test_committed_doc_matches_registry(self):
        doc = (pathlib.Path(aa.__file__).resolve().parent.parent
               / "docs/source/index/usage_principles/df_schemas.rst")
        assert doc.exists(), "df_schemas.rst not generated"
        assert doc.read_text() == ut.render_schemas_rst(), (
            "df_schemas.rst out of sync; run .github/scripts/gen_df_schemas_doc.py")
