"""This is a script to test EmbeddingPreprocessor.cluster_pseudo_scales()."""
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=2000)
settings.load_profile("ci")


# Helpers --------------------------------------------------------------
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _make_pseudo_scales(D=10, seed=0):
    """Build a deterministic (20, D) pseudo-scale DataFrame for tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((20, D)).astype("float64"),
        index=list(ALPHABET),
        columns=[f"dim_{i}" for i in range(D)],
    )


# Normal cases ---------------------------------------------------------
class TestClusterPseudoScales:
    """Positive and parameter-level negative tests for cluster_pseudo_scales."""

    # Positive cases
    def test_returns_dataframe(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert isinstance(df_cat, pd.DataFrame)

    def test_shape_rows_match_D(self):
        df_scales_emb = _make_pseudo_scales(D=12)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat.shape == (12, 5)

    def test_columns_mirror_aaontology_df_cat(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert list(df_cat.columns) == ["scale_id", "category", "subcategory", "scale_name", "scale_description"]

    def test_scale_id_matches_df_scales_emb_columns(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat["scale_id"].tolist() == list(df_scales_emb.columns)

    def test_cat_labels_use_plm_cat_prefix(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert all(c.startswith("PLM_cat_") for c in df_cat["category"])
        assert all(s.startswith("PLM_subcat_") for s in df_cat["subcategory"])

    def test_subcat_has_at_least_as_many_clusters_as_cat(self):
        """Finer threshold should yield ≥ as many clusters as coarser threshold."""
        df_scales_emb = _make_pseudo_scales(D=16)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.2, subcat_min_th=0.8, random_state=0,
        )
        assert df_cat["subcategory"].nunique() >= df_cat["category"].nunique()

    def test_deterministic_with_same_random_state(self):
        df_scales_emb = _make_pseudo_scales(D=10)
        df1 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df2 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        pd.testing.assert_frame_equal(df1, df2)

    @given(rs=some.integers(min_value=0, max_value=10**6))
    @settings(max_examples=5)
    def test_accepts_any_nonneg_random_state(self, rs):
        df_scales_emb = _make_pseudo_scales(D=6)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=rs,
        )
        assert df_cat.shape == (6, 5)

    def test_scale_name_equals_scale_id(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert (df_cat["scale_id"] == df_cat["scale_name"]).all()

    def test_scale_description_is_empty_string(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert (df_cat["scale_description"] == "").all()

    # Negative cases
    def test_invalid_df_scales_emb_none(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=None, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_df_scales_emb_not_dataframe(self):
        for bad in ["str", 1, np.zeros((20, 5))]:
            with pytest.raises(ValueError):
                aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                    df_scales_emb=bad, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                )

    def test_invalid_cat_min_th_negative(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=-0.1, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_cat_min_th_above_one(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=1.5, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_subcat_min_th_negative(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=-0.1, random_state=0,
            )

    def test_invalid_subcat_min_th_above_one(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=1.5, random_state=0,
            )

    def test_invalid_cat_min_th_geq_subcat_min_th(self):
        """Coarser threshold must be < finer threshold; reject equal or reversed."""
        df_scales_emb = _make_pseudo_scales(D=6)
        for cat_th, sub_th in [(0.5, 0.5), (0.7, 0.3)]:
            with pytest.raises(ValueError, match="should be <"):
                aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                    df_scales_emb=df_scales_emb, cat_min_th=cat_th, subcat_min_th=sub_th, random_state=0,
                )

    def test_invalid_random_state_negative(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=-1,
            )


# Complex / interaction cases ------------------------------------------
class TestClusterPseudoScalesComplex:
    """Combinations and edge interactions for cluster_pseudo_scales."""

    def test_pipeline_with_build_pseudo_scales(self):
        """End-to-end: feed build_pseudo_scales output directly into cluster_pseudo_scales."""
        rng = np.random.default_rng(0)
        df_seq = pd.DataFrame({
            "entry": [f"P{i}" for i in range(5)],
            "sequence": ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKM"],
        })
        D = 12
        embeddings = {
            e: rng.standard_normal((len(s), D)).astype("float32")
            for e, s in zip(df_seq["entry"], df_seq["sequence"])
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales_emb = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        # Skip rows with NaN (AAs absent from corpus) — AAclust needs complete rows
        df_scales_emb = df_scales_emb.dropna()
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat.shape == (D, 5)
        assert df_cat["scale_id"].tolist() == [f"dim_{i}" for i in range(D)]

    def test_thresholds_yield_different_partitions(self):
        """At very different thresholds, cat and subcat partitions should differ."""
        df_scales_emb = _make_pseudo_scales(D=20)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.2, subcat_min_th=0.9, random_state=0,
        )
        # Subcategory should produce more clusters than category
        assert df_cat["subcategory"].nunique() > df_cat["category"].nunique()

    def test_large_D_still_works(self):
        df_scales_emb = _make_pseudo_scales(D=64)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat.shape == (64, 5)

    def test_single_dim_input(self):
        """Single dimension → one cluster regardless of threshold."""
        df_scales_emb = _make_pseudo_scales(D=1)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat.shape == (1, 5)
        assert df_cat["category"].nunique() == 1
        assert df_cat["subcategory"].nunique() == 1

    def test_two_close_dims_cluster_together_at_low_threshold(self):
        """Two correlated dims should share a cluster at a low threshold."""
        # Build two highly correlated dims
        base = np.random.default_rng(0).standard_normal(20)
        df_scales_emb = pd.DataFrame(
            np.column_stack([base, base + 0.001 * np.random.default_rng(1).standard_normal(20)]),
            index=list(ALPHABET),
            columns=["dim_0", "dim_1"],
        )
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        # The two highly-correlated dims should be in the same category at threshold 0.3
        assert df_cat["category"].iloc[0] == df_cat["category"].iloc[1]

    # Combined-invalid negative cases
    def test_invalid_combined_none_inputs(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=None, cat_min_th=-1, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_combined_reversed_thresholds(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        with pytest.raises(ValueError, match="should be <"):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.9, subcat_min_th=0.1, random_state=0,
            )
