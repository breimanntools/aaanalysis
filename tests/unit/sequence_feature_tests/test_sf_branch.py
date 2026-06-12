"""Branch-coverage tests for SequenceFeature public methods.

Targets the previously-uncovered guard / warning arms of get_df_parts,
get_df_feat, feature_matrix, and prune_by_correlation, reached exclusively
through the public ``aa.SequenceFeature`` API. Follows the house template:
focused negatives via ``pytest.raises(..., match=...)`` and warnings via
``pytest.warns(<Category>, match=...)``.
"""
import math
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

SF = aa.SequenceFeature


# I Helper functions / fixtures
def _df_feat_input(n_feat=8, n_samples=10, with_jmd=False):
    """Real (features, df_parts, labels) from a small DOM_GSEC slice."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    features = aa.load_features(name="DOM_GSEC")["feature"].head(n_feat).to_list()
    sf = aa.SequenceFeature(verbose=False)
    if with_jmd:
        feat_parts = list({f.split("-")[0].lower() for f in features})
        list_parts = list({"jmd_n", "tmd", "jmd_c", *feat_parts})
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts)
    else:
        df_parts = sf.get_df_parts(df_seq=df_seq)
    return features, df_parts, labels


# ======================================================================================
# get_df_parts — remove_entries_with_gaps warn + all-removed raise
# ======================================================================================
class TestGetDfPartsGapsBranch:
    """Gap-removal warning and the all-entries-removed guard."""

    def test_removed_entries_warns(self):
        # P1 is too short for jmd_n_len=10 -> introduces gaps -> removed -> UserWarning
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKL", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"],
            "tmd_start": [3, 11],
            "tmd_stop": [6, 20],
        })
        sf = aa.SequenceFeature(verbose=True)
        with pytest.warns(UserWarning, match="entries have been removed"):
            df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10,
                                       remove_entries_with_gaps=True)
        assert len(df_parts) == 1

    def test_all_removed_raises(self):
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFG", "ACDEFGH"],
            "tmd_start": [2, 2],
            "tmd_stop": [4, 4],
        })
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError, match="All entries have been removed"):
            sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10,
                            remove_entries_with_gaps=True)


# ======================================================================================
# get_df_feat — label/df_parts guards + sample-vs-group amino-acid columns
# ======================================================================================
class TestGetDfFeatLabelGuards:
    """The check_match_* guards reachable through get_df_feat."""

    def test_wrong_labels_raise(self):
        # labels carrying a value outside {label_test, label_ref} is caught upstream by
        # check_labels; allow_other_vals=False -> ValueError before the dedicated guard.
        features, df_parts, labels = _df_feat_input(n_feat=6, n_samples=8)
        bad = list(labels)
        bad[0] = 7
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError):
            sf.get_df_feat(features=features, df_parts=df_parts, labels=bad)

    def test_one_test_sample_missing_parts_raises(self):
        # Exactly one test sample but df_parts lacks jmd_n/jmd_c -> dedicated guard fires.
        features, df_parts, _ = _df_feat_input(n_feat=6, n_samples=8, with_jmd=False)
        labels = [1] + [0] * (len(df_parts) - 1)
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError, match="label_test"):
            sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)

    def test_one_ref_sample_missing_parts_raises(self):
        features, df_parts, _ = _df_feat_input(n_feat=6, n_samples=8, with_jmd=False)
        labels = [0] + [1] * (len(df_parts) - 1)
        sf = aa.SequenceFeature(verbose=False)
        with pytest.raises(ValueError, match="label_ref"):
            sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)


class TestGetDfFeatSampleVsGroup:
    """Sample-vs-group reaches the backend amino-acid retrieval arms."""

    def test_one_test_sample_adds_aa_test(self):
        features, df_parts, _ = _df_feat_input(n_feat=6, n_samples=10, with_jmd=True)
        labels = [1] + [0] * (len(df_parts) - 1)
        sf = aa.SequenceFeature(verbose=False)
        df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)
        cols = [c.lower() for c in df_feat.columns]
        assert "amino_acids_test" in cols
        assert "amino_acids_ref" not in cols

    def test_one_ref_sample_adds_aa_ref(self):
        features, df_parts, _ = _df_feat_input(n_feat=6, n_samples=10, with_jmd=True)
        labels = [0] + [1] * (len(df_parts) - 1)
        sf = aa.SequenceFeature(verbose=False)
        df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)
        cols = [c.lower() for c in df_feat.columns]
        assert "amino_acids_ref" in cols
        assert "amino_acids_test" not in cols

    def test_one_test_one_ref_adds_both(self):
        # Exactly one test (1) AND exactly one ref (0): a 2-row df_parts (one per group).
        features, df_parts, _ = _df_feat_input(n_feat=6, n_samples=10, with_jmd=True)
        df_parts2 = df_parts.iloc[:2]
        labels = [1, 0]
        sf = aa.SequenceFeature(verbose=False)
        df_feat = sf.get_df_feat(features=features, df_parts=df_parts2, labels=labels)
        cols = [c.lower() for c in df_feat.columns]
        assert "amino_acids_test" in cols
        assert "amino_acids_ref" in cols


class TestGetDfFeatVerboseWarn:
    """get_df_feat with verbose=True hits the feature-matrix-size warning call site."""

    def test_verbose_emits_creation_message(self):
        features, df_parts, labels = _df_feat_input(n_feat=6, n_samples=8)
        sf = aa.SequenceFeature(verbose=True)
        # Small matrix -> only the info print, no >10^6 warning; the call site still runs.
        df_feat = sf.get_df_feat(features=features, df_parts=df_parts, labels=labels)
        assert isinstance(df_feat, pd.DataFrame)


# ======================================================================================
# feature_matrix — n_vals > 10^6 warning
# ======================================================================================
class TestFeatureMatrixLargeWarn:
    """Reach the >=10^6-values warning arm of warn_creation_of_feature_matrix."""

    def test_large_matrix_warns(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=63)  # 126 samples (both classes)
        sf = aa.SequenceFeature(verbose=True)
        df_parts = sf.get_df_parts(df_seq=df_seq)
        n_samples = len(df_parts)
        features = sf.get_features()
        need = math.ceil(1_000_000 / n_samples) + 1
        feats = features[:need]
        assert len(feats) * n_samples > 1_000_000
        with pytest.warns(UserWarning, match=r">=10\^6 values"):
            X = sf.feature_matrix(features=feats, df_parts=df_parts)
        assert np.asarray(X).shape == (n_samples, len(feats))


# ======================================================================================
# prune_by_correlation — fewer than two non-constant columns (skip the filter)
# ======================================================================================
class TestPruneByCorrelationConstantBranch:
    """df_feat with >=2 rows but <2 non-constant X columns skips filter_correlation_."""

    @pytest.fixture(scope="class")
    def fitted(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature(verbose=False)
        df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=4, jmd_c_len=4)
        df_feat = aa.CPP(df_parts=df_parts).run(labels=labels, n_filter=5)
        return sf, df_feat.reset_index(drop=True), df_parts

    def test_single_non_constant_column_retains_all(self, fitted):
        sf, df_feat, df_parts = fitted
        df_feat2 = df_feat.head(2)
        n = len(df_parts)
        X = np.zeros((n, 2))
        X[:, 0] = np.arange(n, dtype=float)  # non-constant
        X[:, 1] = 1.0                        # constant
        out = sf.prune_by_correlation(df_feat=df_feat2, X=X, max_cor=0.7)
        # Only one non-constant column -> correlation filter skipped -> both kept.
        assert len(out) == 2

    def test_all_constant_columns_retains_all(self, fitted):
        sf, df_feat, df_parts = fitted
        df_feat2 = df_feat.head(3)
        n = len(df_parts)
        X = np.ones((n, 3))  # all constant -> no non-constant columns at all
        out = sf.prune_by_correlation(df_feat=df_feat2, X=X, max_cor=0.7)
        assert len(out) == 3
