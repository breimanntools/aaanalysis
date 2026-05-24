"""This is a script to test StructurePreprocessor.build_scales()."""
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# I Helper Functions
def _all_dssp_keys():
    return ["ss3", "ss8", "asa", "rasa", "phi_psi", "phi_psi_sincos"]


def _all_pdb_keys():
    return ["bfactor", "depth"]


# II Test Classes
class TestStpBuildScales:
    """Single-parameter normal + invalid cases for build_scales."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_features_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=None)

    def test_invalid_features_empty(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=[])

    def test_invalid_features_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=["mystery"])

    def test_invalid_features_str_not_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features="ss3")

    def test_invalid_features_int(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=[7])

    def test_invalid_dim_names_override_wrong_length(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=["ss3"], dim_names_override=["a", "b"])

    def test_invalid_dim_names_override_not_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=["ss3"], dim_names_override="abc")

    def test_invalid_dim_names_override_int_items(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=["ss3"],
                             dim_names_override=[1, 2, 3])

    def test_invalid_features_with_none_item(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=[None])

    def test_invalid_features_mixed_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(features=["ss3", "not_a_key"])

    # ----- POSITIVES (≥10) -----
    def test_valid_df_scales_shape_20_by_D(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(
            features=["ss3", "rasa", "phi_psi_sincos"])
        assert df_scales.shape == (20, 3 + 1 + 4)

    def test_valid_df_cat_shape_D_by_5(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(
            features=["ss3", "rasa", "phi_psi_sincos"])
        assert df_cat.shape == (8, 5)

    def test_valid_index_is_canonical_aa(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["ss3"])
        assert list(df_scales.index) == list(ut.LIST_CANONICAL_AA)

    def test_valid_columns_use_registry_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["ss3"])
        assert list(df_scales.columns) == ["ss_helix", "ss_strand", "ss_coil"]

    def test_valid_ss8_dim_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["ss8"])
        assert list(df_scales.columns) == ["ss_H", "ss_B", "ss_E", "ss_G",
                                           "ss_I", "ss_T", "ss_S", "ss_blank"]

    def test_valid_phi_psi_raw_dim_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["phi_psi"])
        assert list(df_scales.columns) == ["phi", "psi"]

    def test_valid_phi_psi_sincos_dim_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["phi_psi_sincos"])
        assert list(df_scales.columns) == ["phi_sin", "phi_cos",
                                           "psi_sin", "psi_cos"]

    def test_valid_df_cat_columns_match_schema(self):
        stp = aa.StructurePreprocessor(verbose=False)
        _, df_cat = stp.build_scales(features=["ss3"])
        assert list(df_cat.columns) == [
            ut.COL_SCALE_ID, ut.COL_CAT, ut.COL_SUBCAT,
            ut.COL_SCALE_NAME, ut.COL_SCALE_DES]

    def test_valid_df_cat_category_dssp_ss(self):
        stp = aa.StructurePreprocessor(verbose=False)
        _, df_cat = stp.build_scales(features=["ss3"])
        assert (df_cat[ut.COL_CAT] == "DSSP_SS").all()
        assert (df_cat[ut.COL_SUBCAT] == "3-state").all()

    def test_valid_mixed_dssp_pdb_features(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(
            features=["ss3", "rasa", "bfactor"])
        assert df_scales.shape == (20, 3 + 1 + 1)
        assert df_cat.shape == (5, 5)

    def test_valid_dim_names_override_applied(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(
            features=["ss3"], dim_names_override=["A", "B", "C"])
        assert list(df_scales.columns) == ["A", "B", "C"]
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["A", "B", "C"]

    def test_valid_returns_dataframes(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(features=["bfactor"])
        assert isinstance(df_scales, pd.DataFrame)
        assert isinstance(df_cat, pd.DataFrame)


class TestStpBuildScalesComplex:
    """Cross-parameter combinations for build_scales."""

    def test_complex_all_features_total_dim(self):
        stp = aa.StructurePreprocessor(verbose=False)
        keys = _all_dssp_keys() + _all_pdb_keys()
        df_scales, df_cat = stp.build_scales(features=keys)
        # ss3(3)+ss8(8)+asa(1)+rasa(1)+phi_psi(2)+phi_psi_sincos(4)+bfactor(1)+depth(1)=21
        assert df_scales.shape == (20, 21)
        assert df_cat.shape == (21, 5)

    def test_complex_feature_order_preserved(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales1, _ = stp.build_scales(features=["ss3", "rasa"])
        df_scales2, _ = stp.build_scales(features=["rasa", "ss3"])
        # Different orderings yield different column orderings.
        assert list(df_scales1.columns) != list(df_scales2.columns)
        assert df_scales1.shape == df_scales2.shape

    def test_complex_default_values_are_zero(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["ss3"])
        # Values are unused in numerical mode but should be present and numeric.
        assert (df_scales.values == 0.0).all()

    def test_complex_scale_id_matches_columns(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(features=["ss3", "rasa"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == list(df_scales.columns)

    def test_complex_subcategory_distinguishes_ss3_vs_ss8(self):
        stp = aa.StructurePreprocessor(verbose=False)
        _, df_cat = stp.build_scales(features=["ss3", "ss8"])
        unique = set(df_cat[ut.COL_SUBCAT].tolist())
        assert {"3-state", "8-state"}.issubset(unique)

    def test_complex_override_with_pdb_features(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, df_cat = stp.build_scales(
            features=["bfactor", "depth"],
            dim_names_override=["temperature", "burial"])
        assert list(df_scales.columns) == ["temperature", "burial"]
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["temperature", "burial"]

    def test_complex_duplicate_feature_keys_double_dims(self):
        # Passing a key twice doubles its dims — by design (allowed by registry).
        stp = aa.StructurePreprocessor(verbose=False)
        df_scales, _ = stp.build_scales(features=["asa", "asa"])
        assert df_scales.shape == (20, 2)
