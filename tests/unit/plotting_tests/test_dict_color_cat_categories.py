"""This is a script to test that every feature ``category`` value emitted
by ``StructurePreprocessor.build_cat`` and
``EmbeddingPreprocessor.build_cat`` resolves to a color in
``ut.DICT_COLOR_CAT`` — closing the v1 defect where these categories
caused ``CPPPlot.heatmap()`` to raise ``ValueError``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False


# I Helper Functions
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def _make_pseudo_scales(D=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((20, D)),
        index=ALPHABET,
        columns=[f"dim_{i}" for i in range(D)],
    )


# II Test Classes
class TestDictColorCatPaletteIsLocked:
    """Pin the exact contents of ut.DICT_COLOR_CAT for the v1.1 palette."""

    def test_contains_eight_aaontology_categories(self):
        aaontology = ['ASA/Volume', 'Composition', 'Conformation', 'Energy',
                      'Others', 'Polarity', 'Shape', 'Structure-Activity']
        for c in aaontology:
            assert c in ut.DICT_COLOR_CAT

    def test_contains_three_v1_1_buckets(self):
        for c in ("Structure", "Embeddings", "PTMs"):
            assert c in ut.DICT_COLOR_CAT

    def test_v1_1_hex_values(self):
        assert ut.DICT_COLOR_CAT["Structure"] == "#2E6E5E"
        assert ut.DICT_COLOR_CAT["Embeddings"] == "#6B4FB5"
        assert ut.DICT_COLOR_CAT["PTMs"] == "#B36BCB"

    def test_list_cat_matches_dict(self):
        # LIST_CAT is the ordered roster used by CPPPlot for color iteration.
        assert set(ut.LIST_CAT) == set(ut.DICT_COLOR_CAT.keys())

    def test_no_legacy_plm_cat_keys(self):
        # v1 emitted "PLM_cat_<k>" / "PLM_subcat_<k>" categories that were
        # NEVER in DICT_COLOR_CAT — confirm we didn't accidentally add them
        # while migrating EmbeddingPreprocessor to 'Embeddings'.
        for k in ut.DICT_COLOR_CAT:
            assert not k.startswith("PLM_cat_")
            assert not k.startswith("PLM_subcat_")


class TestStructurePreprocessorCategoriesResolve:
    """Every category produced by build_cat must be in DICT_COLOR_CAT."""

    def test_single_feature_category_resolves(self):
        strp = aa.StructurePreprocessor(verbose=False)
        df_cat = strp.build_cat(features=["ss3"])
        for c in df_cat[ut.COL_CAT].unique():
            assert c in ut.DICT_COLOR_CAT

    def test_all_v1_keys_resolve(self):
        # The full v1 feature key set; v1.1 also includes these.
        strp = aa.StructurePreprocessor(verbose=False)
        all_keys = ["ss3", "ss8", "rasa", "phi_psi_sincos",
                    "bfactor", "depth"]
        df_cat = strp.build_cat(features=all_keys)
        for c in df_cat[ut.COL_CAT].unique():
            assert c in ut.DICT_COLOR_CAT

    def test_all_categories_are_structure(self):
        strp = aa.StructurePreprocessor(verbose=False)
        all_keys = ["ss3", "ss8", "rasa", "phi_psi_sincos",
                    "bfactor", "depth"]
        df_cat = strp.build_cat(features=all_keys)
        assert set(df_cat[ut.COL_CAT].unique()) == {"Structure"}


class TestEmbeddingPreprocessorCategoriesResolve:
    """Every category produced by build_cat must be in DICT_COLOR_CAT."""

    def test_mean_only_category_resolves(self):
        df_scales = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor().build_cat(
            df_scales=df_scales,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0)
        for c in df_cat[ut.COL_CAT].unique():
            assert c in ut.DICT_COLOR_CAT

    def test_std_aware_category_resolves(self):
        df_scales = _make_pseudo_scales(D=8, seed=0)
        df_stds = _make_pseudo_scales(D=8, seed=1).abs()
        df_stds.index = df_scales.index
        df_stds.columns = df_scales.columns
        df_cat = aa.EmbeddingPreprocessor().build_cat(
            df_scales=df_scales, df_stds=df_stds,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0)
        for c in df_cat[ut.COL_CAT].unique():
            assert c in ut.DICT_COLOR_CAT

    def test_all_categories_are_embeddings(self):
        df_scales = _make_pseudo_scales(D=10)
        df_cat = aa.EmbeddingPreprocessor().build_cat(
            df_scales=df_scales,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0)
        assert set(df_cat[ut.COL_CAT].unique()) == {"Embeddings"}
