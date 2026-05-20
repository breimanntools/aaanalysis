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

    def test_invalid_df_scales_emb_too_few_dims(self):
        """D < 3 should be rejected (AAclust requires ≥3 samples to cluster)."""
        for D in [1, 2]:
            df_scales_emb = _make_pseudo_scales(D=D)
            with pytest.raises(ValueError, match="at least 3"):
                aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                    df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                )

    # metric parameter
    def test_metric_default_is_correlation(self):
        """Omitting metric should match metric='correlation' exactly."""
        df_scales_emb = _make_pseudo_scales(D=10)
        df_default = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df_explicit = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            metric="correlation",
        )
        pd.testing.assert_frame_equal(df_default, df_explicit)

    def test_metric_cosine_returns_valid_table(self):
        df_scales_emb = _make_pseudo_scales(D=10)
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            metric="cosine",
        )
        assert df_cat.shape == (10, 5)
        assert all(c.startswith("PLM_cat_") for c in df_cat["category"])
        assert all(s.startswith("PLM_subcat_") for s in df_cat["subcategory"])

    def test_metric_correlation_and_cosine_deterministic(self):
        df_scales_emb = _make_pseudo_scales(D=12)
        for metric in ["correlation", "cosine"]:
            df1 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                metric=metric,
            )
            df2 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                metric=metric,
            )
            pd.testing.assert_frame_equal(df1, df2)

    # Negative cases for metric
    def test_invalid_metric_unknown_string(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        for bad in ["euclidean", "manhattan", "pearson", "", "Correlation"]:
            with pytest.raises(ValueError):
                aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                    df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                    metric=bad,
                )

    def test_invalid_metric_non_string(self):
        df_scales_emb = _make_pseudo_scales(D=8)
        for bad in [None, 1, 0.5, ["correlation"]]:
            with pytest.raises(ValueError):
                aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                    df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
                    metric=bad,
                )

    # df_stds_emb parameter (std-aware clustering)
    def test_accepts_df_stds_emb(self):
        """Supplying df_stds_emb enables std-aware mode and returns the standard (D, 5) shape."""
        df_scales_emb = _make_pseudo_scales(D=10, seed=0)
        df_stds_emb = _make_pseudo_scales(D=10, seed=1).abs()  # nonneg stds
        df_stds_emb.index = df_scales_emb.index
        df_stds_emb.columns = df_scales_emb.columns
        df_cat = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        assert df_cat.shape == (10, 5)
        assert all(c.startswith("PLM_cat_") for c in df_cat["category"])

    def test_std_aware_deterministic_with_same_random_state(self):
        df_scales_emb = _make_pseudo_scales(D=10, seed=0)
        df_stds_emb = _make_pseudo_scales(D=10, seed=2).abs()
        df_stds_emb.index = df_scales_emb.index
        df_stds_emb.columns = df_scales_emb.columns
        df1 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df2 = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_std_aware_equivalent_to_manual_concat_zscore(self):
        """Validates the math: passing (df_scales_emb, df_stds_emb) to std-aware mode is
        equivalent to passing the pre-built per-column z-scored concat matrix in mean-only
        mode. Catches regressions in the pre-transform helpers."""
        df_scales_emb = _make_pseudo_scales(D=10, seed=0)
        df_stds_emb = _make_pseudo_scales(D=10, seed=1).abs()
        df_stds_emb.index = df_scales_emb.index
        df_stds_emb.columns = df_scales_emb.columns

        def _zsc(X):
            mu = X.mean(axis=0, keepdims=True)
            sig = X.std(axis=0, keepdims=True)
            sig = np.where(sig == 0, 1.0, sig)
            return (X - mu) / sig

        M = df_scales_emb.T.values  # (D, 20)
        S = df_stds_emb.T.values    # (D, 20)
        manual = np.concatenate([_zsc(M), _zsc(S)], axis=1)  # (D, 40)
        # Mean-only path expects df_scales_emb of shape (n_features, D), then transposes
        # internally to (D, n_features). So we pass the (40, D) frame here.
        df_manual = pd.DataFrame(
            manual.T,
            index=[f"feat_{i}" for i in range(40)],
            columns=df_scales_emb.columns,
        )
        df_via_mean_only_on_manual = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_manual, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df_via_std_aware = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        pd.testing.assert_frame_equal(df_via_mean_only_on_manual, df_via_std_aware)

    def test_dims_with_similar_mean_different_std_separate_under_std_aware(self):
        """Two dims with VERY SIMILAR per-AA means but very different per-AA stds: in
        mean-only mode they cluster together; in std-aware mode they should separate.
        (Truly identical rows would trip AAclust's unique-samples check, so we add
        microscopic noise to the second dim's means.)"""
        rng = np.random.default_rng(0)
        base_mean = rng.standard_normal(20)
        means = np.column_stack([
            base_mean,
            base_mean + 1e-6 * rng.standard_normal(20),  # near-identical, Pearson ≈ 1
            rng.standard_normal(20),
            rng.standard_normal(20),
        ])
        df_scales_emb = pd.DataFrame(
            means, index=list(ALPHABET),
            columns=["dim_0", "dim_1", "dim_2", "dim_3"],
        )
        stds = np.column_stack([
            np.full(20, 0.01),         # tiny constant variance
            np.linspace(1.0, 5.0, 20), # large AA-varying variance
            np.full(20, 1.0),
            np.full(20, 1.0),
        ])
        df_stds_emb = pd.DataFrame(
            stds, index=list(ALPHABET),
            columns=["dim_0", "dim_1", "dim_2", "dim_3"],
        )
        df_mean_only = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df_std_aware = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        cat = lambda df, dim: df.loc[df["scale_id"] == dim, "category"].iloc[0]
        # Mean-only: dim_0 and dim_1 should share the category (means near-identical).
        assert cat(df_mean_only, "dim_0") == cat(df_mean_only, "dim_1")
        # Std-aware: with very different stds, they should separate.
        assert cat(df_std_aware, "dim_0") != cat(df_std_aware, "dim_1")

    # Negative cases for df_stds_emb
    def test_invalid_df_stds_emb_wrong_shape(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        df_stds_emb = _make_pseudo_scales(D=8).abs()  # D mismatch
        df_stds_emb.index = df_scales_emb.index
        with pytest.raises(ValueError, match="same shape"):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
                cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_df_stds_emb_mismatched_index_or_columns(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        # Same shape but shuffled AA index
        df_stds_emb_bad_idx = df_scales_emb.abs().copy()
        df_stds_emb_bad_idx.index = list(reversed(df_scales_emb.index))
        with pytest.raises(ValueError, match="same index"):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb_bad_idx,
                cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            )
        # Same shape but renamed columns
        df_stds_emb_bad_cols = df_scales_emb.abs().copy()
        df_stds_emb_bad_cols.columns = [f"X_{i}" for i in range(df_scales_emb.shape[1])]
        with pytest.raises(ValueError, match="same columns"):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb_bad_cols,
                cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
            )

    def test_invalid_df_stds_emb_with_nan(self):
        df_scales_emb = _make_pseudo_scales(D=6)
        df_stds_emb = df_scales_emb.abs().copy()
        df_stds_emb.iloc[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            aa.EmbeddingPreprocessor.cluster_pseudo_scales(
                df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
                cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
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

    def test_correlated_dims_cluster_together_at_low_threshold(self):
        """Two correlated dims should share a cluster at a low threshold."""
        # Build two highly correlated dims plus one independent dim (AAclust needs D≥3)
        base = np.random.default_rng(0).standard_normal(20)
        df_scales_emb = pd.DataFrame(
            np.column_stack([
                base,
                base + 0.001 * np.random.default_rng(1).standard_normal(20),
                np.random.default_rng(2).standard_normal(20),
            ]),
            index=list(ALPHABET),
            columns=["dim_0", "dim_1", "dim_2"],
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

    def test_pipeline_into_cpp_run(self):
        """End-to-end: build_pseudo_scales → cluster_pseudo_scales → CPP.run.

        Verifies the (df_scales_emb, df_cat_emb) pair is a drop-in replacement
        for AAontology (df_scales, df_cat) in the existing CPP.run pipeline.
        """
        rng = np.random.default_rng(0)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        D = 6
        embeddings = {
            entry: rng.standard_normal((len(seq), D)).astype("float32")
            for entry, seq in zip(df_seq["entry"], df_seq["sequence"])
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales_emb = aa.EmbeddingPreprocessor.build_pseudo_scales(
                df_seq=df_seq, embeddings=embeddings,
            )
        # Drop AAs absent from corpus (CPP rejects NaN values in df_scales)
        df_scales_emb = df_scales_emb.dropna()
        df_cat_emb = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales_emb, df_cat=df_cat_emb)
        df_feat = cpp.run(labels=labels, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) > 0

    def test_pipeline_into_cpp_run_std_aware(self):
        """End-to-end std-aware variant: build_pseudo_scales(return_std=True) →
        cluster_pseudo_scales(df_stds_emb=...) → CPP.run."""
        rng = np.random.default_rng(0)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        D = 6
        embeddings = {
            entry: rng.standard_normal((len(seq), D)).astype("float32")
            for entry, seq in zip(df_seq["entry"], df_seq["sequence"])
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales_emb, df_stds_emb = aa.EmbeddingPreprocessor.build_pseudo_scales(
                df_seq=df_seq, embeddings=embeddings, return_std=True,
            )
        df_scales_emb = df_scales_emb.dropna()
        df_stds_emb = df_stds_emb.loc[df_scales_emb.index]  # drop the same rows
        df_cat_emb = aa.EmbeddingPreprocessor.cluster_pseudo_scales(
            df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb,
            cat_min_th=0.3, subcat_min_th=0.6, random_state=0,
        )
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales_emb, df_cat=df_cat_emb)
        df_feat = cpp.run(labels=labels, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) > 0
