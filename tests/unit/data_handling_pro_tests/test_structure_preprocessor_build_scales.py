"""This is a script to test StructurePreprocessor.build_scales()
and StructurePreprocessor.build_cat() — v1.1 split of the v1 ``build_scales``
into a corpus-derived (per-AA-mean) pseudo-scales method and a corpus-free
metadata method.
"""
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions
def _df_seq(n=3, L=12, seed=0):
    rng = np.random.default_rng(seed)
    aa_letters = list(ut.LIST_CANONICAL_AA)
    seqs = ["".join(rng.choice(aa_letters, size=L)) for _ in range(n)]
    return pd.DataFrame({"entry": [f"P{i}" for i in range(n)],
                         "sequence": seqs})


def _dict_num(df_seq, D, fill=0.5):
    return {row["entry"]: np.full((len(row["sequence"]), D), fill,
                                   dtype=np.float64)
            for _, row in df_seq.iterrows()}


# II Test Classes
class TestStpBuildPseudoScales:
    """Single-parameter normal + invalid cases for build_scales."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=None, dict_num=None, features=["bfactor"])

    def test_invalid_dict_num_none(self):
        df = _df_seq()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=None,
                                    features=["bfactor"])

    def test_invalid_features_empty(self):
        df = _df_seq()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df,
                                    dict_num=_dict_num(df, D=1),
                                    features=[])

    def test_invalid_features_unknown(self):
        df = _df_seq()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df,
                                    dict_num=_dict_num(df, D=1),
                                    features=["mystery"])

    def test_invalid_features_str_not_list(self):
        df = _df_seq()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df,
                                    dict_num=_dict_num(df, D=1),
                                    features="bfactor")

    def test_invalid_dict_num_missing_entry(self):
        df = _df_seq(n=3)
        d = _dict_num(df, D=1)
        del d["P1"]
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"])

    def test_invalid_dict_num_L_mismatch(self):
        df = _df_seq(n=2)
        d = _dict_num(df, D=1)
        d["P0"] = np.zeros((len(df["sequence"][0]) + 1, 1))
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"])

    def test_invalid_dict_num_D_mismatch(self):
        df = _df_seq()
        d = _dict_num(df, D=3)
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"])

    def test_invalid_dict_num_not_2d(self):
        df = _df_seq()
        d = {row["entry"]: np.zeros(len(row["sequence"]))
             for _, row in df.iterrows()}
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"])

    def test_invalid_dim_names_override_wrong_length(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"],
                                    dim_names_override=["a", "b"])

    def test_invalid_dim_names_override_not_list(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"],
                                    dim_names_override="abc")

    def test_invalid_return_std_non_bool(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"], return_std="yes")

    # ----- POSITIVES (≥10) -----
    def test_valid_returns_dataframe(self):
        df = _df_seq()
        d = _dict_num(df, D=1, fill=0.5)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(df_seq=df, dict_num=d,
                                                features=["bfactor"])
        assert isinstance(df_scales, pd.DataFrame)

    def test_valid_shape_is_20_by_D(self):
        df = _df_seq()
        d = _dict_num(df, D=4, fill=0.5)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["phi_psi_sincos"])
        assert df_scales.shape == (20, 4)

    def test_valid_index_is_canonical_aa(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        assert list(df_scales.index) == list(ut.LIST_CANONICAL_AA)

    def test_valid_columns_use_registry_names(self):
        df = _df_seq()
        d = _dict_num(df, D=3)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["ss3"])
        assert list(df_scales.columns) == ["ss_helix", "ss_strand", "ss_coil"]

    def test_valid_emits_user_warning(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stp.build_scales(df_seq=df, dict_num=d,
                                    features=["bfactor"])
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("dataset-dependent" in str(x.message).lower()
                   for x in user_warns)

    def test_valid_constant_input_constant_means(self):
        df = _df_seq()
        d = _dict_num(df, D=1, fill=0.42)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        # AAs present in the corpus get 0.42; AAs absent get NaN.
        for v in df_scales["bfactor"]:
            assert (np.isclose(v, 0.42, atol=1e-9, equal_nan=False)
                    or np.isnan(v))

    def test_valid_absent_aa_is_nan(self):
        # Corpus that contains only A, C, G
        df = pd.DataFrame({"entry": ["P0"], "sequence": ["ACG"]})
        d = {"P0": np.ones((3, 1), dtype=np.float64)}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        present = {"A", "C", "G"}
        for aa_letter in ut.LIST_CANONICAL_AA:
            if aa_letter in present:
                assert df_scales.loc[aa_letter, "bfactor"] == 1.0
            else:
                assert np.isnan(df_scales.loc[aa_letter, "bfactor"])

    def test_valid_return_std_returns_pair(self):
        df = _df_seq()
        d = _dict_num(df, D=1)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"], return_std=True)
        assert isinstance(result, tuple) and len(result) == 2
        df_scales, df_stds = result
        assert df_scales.shape == df_stds.shape

    def test_valid_constant_input_zero_std(self):
        df = _df_seq()
        d = _dict_num(df, D=1, fill=0.5)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, df_stds = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"], return_std=True)
        # Present AAs should have std=0 (constant fill); absent NaN.
        for v in df_stds["bfactor"]:
            assert v == 0.0 or np.isnan(v)

    def test_valid_dim_names_override_applied(self):
        df = _df_seq()
        d = _dict_num(df, D=3)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["ss3"],
                dim_names_override=["A", "B", "C"])
        assert list(df_scales.columns) == ["A", "B", "C"]

    def test_valid_corr_no_longer_all_nan(self):
        # Verify the fix for v1's defect: df_scales.corr() must be non-trivial
        # so the redundancy filter's cor arm is actually active.
        rng = np.random.default_rng(1)
        df = _df_seq(n=4, L=20, seed=1)
        # Build a multi-D dict_num with deterministic varying values per AA.
        D = 5
        d = {row["entry"]:
             rng.uniform(0.0, 1.0, (len(row["sequence"]), D))
             for _, row in df.iterrows()}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["phi_psi_sincos", "bfactor"])
        corr = df_scales.corr()
        # Diagonal should be 1.0, off-diagonal finite (not NaN).
        assert np.isfinite(np.diag(corr)).all()
        off_diag = corr.values[~np.eye(D, dtype=bool)]
        # At least some off-diagonal entries should be finite.
        assert np.isfinite(off_diag).any()


class TestStpBuildPseudoScalesComplex:
    """Cross-parameter combinations for build_scales."""

    def test_complex_multiple_features_D_matches(self):
        df = _df_seq()
        D = 3 + 1   # ss3 + bfactor
        d = _dict_num(df, D=D)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["ss3", "bfactor"])
        assert df_scales.shape == (20, D)

    def test_complex_return_std_uses_same_dim_names(self):
        df = _df_seq()
        d = _dict_num(df, D=4)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales, df_stds = stp.build_scales(
                df_seq=df, dict_num=d, features=["phi_psi_sincos"],
                return_std=True)
        assert list(df_scales.columns) == list(df_stds.columns)

    def test_complex_override_persists_across_calls(self):
        df = _df_seq()
        d = _dict_num(df, D=3)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = stp.build_scales(df_seq=df, dict_num=d,
                                        features=["ss3"],
                                        dim_names_override=["x", "y", "z"])
            b = stp.build_scales(df_seq=df, dict_num=d,
                                        features=["ss3"],
                                        dim_names_override=["x", "y", "z"])
        assert list(a.columns) == list(b.columns)

    def test_complex_partial_corpus_per_dim_nan(self):
        # Single entry, single AA in sequence — all other AAs in df_scales are NaN.
        df = pd.DataFrame({"entry": ["P0"], "sequence": ["A"]})
        d = {"P0": np.array([[0.7]])}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        # Only A has a value.
        assert df_scales.loc["A", "bfactor"] == 0.7
        assert df_scales.drop("A")["bfactor"].isna().all()

    def test_complex_no_overlap_aas_in_dict_skipped(self):
        # Non-canonical letter in sequence — skipped.
        df = pd.DataFrame({"entry": ["P0"], "sequence": ["XAC"]})
        d = {"P0": np.array([[0.3], [0.5], [0.7]])}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        assert df_scales.loc["A", "bfactor"] == 0.5
        assert df_scales.loc["C", "bfactor"] == 0.7

    def test_complex_dict_num_nan_propagates(self):
        df = pd.DataFrame({"entry": ["P0"], "sequence": ["AAC"]})
        d = {"P0": np.array([[np.nan], [0.5], [0.7]])}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["bfactor"])
        # A appears twice (one NaN, one 0.5) → mean over non-NaN = 0.5
        assert df_scales.loc["A", "bfactor"] == 0.5

    def test_complex_corr_diagonal_is_unity(self):
        rng = np.random.default_rng(2)
        df = _df_seq(n=4, L=20, seed=2)
        D = 4
        d = {row["entry"]:
             rng.uniform(0.0, 1.0, (len(row["sequence"]), D))
             for _, row in df.iterrows()}
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = stp.build_scales(
                df_seq=df, dict_num=d, features=["phi_psi_sincos"])
        corr = df_scales.corr().values
        np.testing.assert_allclose(np.diag(corr), 1.0)


class TestStpBuildCat:
    """build_cat: corpus-free registry lookup for ``df_cat``."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_features_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=None)

    def test_invalid_features_empty(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=[])

    def test_invalid_features_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=["mystery"])

    def test_invalid_features_str_not_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features="ss3")

    def test_invalid_features_int_item(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=[7])

    def test_invalid_features_mixed_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=["ss3", "not_a_key"])

    def test_invalid_features_none_item(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=[None])

    def test_invalid_dim_names_override_wrong_length(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=["ss3"], dim_names_override=["a", "b"])

    def test_invalid_dim_names_override_not_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=["ss3"], dim_names_override="abc")

    def test_invalid_dim_names_override_int_items(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.build_cat(features=["ss3"],
                          dim_names_override=[1, 2, 3])

    # ----- POSITIVES (≥10) -----
    def test_valid_shape_D_by_5(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3", "rasa", "phi_psi_sincos"])
        assert df_cat.shape == (8, 5)

    def test_valid_columns_match_schema(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3"])
        assert list(df_cat.columns) == [
            ut.COL_SCALE_ID, ut.COL_CAT, ut.COL_SUBCAT,
            ut.COL_SCALE_NAME, ut.COL_SCALE_DES]

    def test_valid_all_keys_emit_structure_category(self):
        # The v1.1 locked palette: every key → category='Structure'.
        stp = aa.StructurePreprocessor(verbose=False)
        all_keys = ["ss3", "ss8", "rasa", "phi_psi_sincos",
                    "bfactor", "depth"]
        df_cat = stp.build_cat(features=all_keys)
        assert (df_cat[ut.COL_CAT] == "Structure").all()

    def test_valid_subcategory_distinguishes_ss3_vs_ss8(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3", "ss8"])
        unique = set(df_cat[ut.COL_SUBCAT].tolist())
        assert {"Secondary structure (3-state)",
                "Secondary structure (8-state)"}.issubset(unique)

    def test_valid_subcategory_uses_registry_strings(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["bfactor"])
        assert df_cat[ut.COL_SUBCAT].iloc[0] == "B-factor (CA mean)"

    def test_valid_scale_id_matches_dim_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3", "rasa"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == [
            "ss_helix", "ss_strand", "ss_coil", "rasa"]

    def test_valid_dim_names_override_applied(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3"],
                               dim_names_override=["X", "Y", "Z"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["X", "Y", "Z"]

    def test_valid_returns_dataframe(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["bfactor"])
        assert isinstance(df_cat, pd.DataFrame)

    def test_valid_feature_order_preserved(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat1 = stp.build_cat(features=["ss3", "rasa"])
        df_cat2 = stp.build_cat(features=["rasa", "ss3"])
        assert df_cat1[ut.COL_SCALE_ID].tolist() != df_cat2[ut.COL_SCALE_ID].tolist()
        assert df_cat1.shape == df_cat2.shape

    def test_valid_mixed_dssp_pdb_features(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3", "rasa", "bfactor"])
        assert df_cat.shape == (5, 5)


class TestStpBuildCatComplex:
    """Cross-parameter combinations for build_cat."""

    def test_complex_all_v1_features_total_dim(self):
        stp = aa.StructurePreprocessor(verbose=False)
        keys = ["ss3", "ss8", "rasa", "phi_psi_sincos", "bfactor", "depth"]
        df_cat = stp.build_cat(features=keys)
        # 3+8+1+4+1+1 = 18
        assert df_cat.shape == (18, 5)

    def test_complex_scale_id_matches_columns_after_override(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(
            features=["bfactor", "depth"],
            dim_names_override=["temperature", "burial"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["temperature", "burial"]

    def test_complex_duplicate_keys_double_dims(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["rasa", "rasa"])
        assert df_cat.shape == (2, 5)

    def test_complex_category_invariant_across_orders(self):
        stp = aa.StructurePreprocessor(verbose=False)
        a = stp.build_cat(features=["ss3", "bfactor"])
        b = stp.build_cat(features=["bfactor", "ss3"])
        assert set(a[ut.COL_CAT]) == set(b[ut.COL_CAT]) == {"Structure"}

    def test_complex_subcategory_unique_per_feature(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["ss3", "rasa", "bfactor", "depth"])
        # Each feature key contributes its own subcategory string;
        # within a single feature key the subcategory is replicated per dim.
        assert df_cat[ut.COL_SUBCAT].nunique() == 4

    def test_complex_override_with_pdb_features(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(
            features=["bfactor", "depth"],
            dim_names_override=["A", "B"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["A", "B"]

    def test_complex_phi_psi_sincos_dim_names(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["phi_psi_sincos"])
        assert df_cat[ut.COL_SCALE_ID].tolist() == ["phi_sin", "phi_cos",
                                                     "psi_sin", "psi_cos"]
