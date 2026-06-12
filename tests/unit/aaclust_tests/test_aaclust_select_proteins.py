"""This script tests the select_proteins() method of the AAclust class."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
import warnings

# Silence the deliberate clustering advisories on tiny fixtures
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions
def get_df_seq(n_proteins=24):
    """Return a minimal df_seq (entry column + a few extras) with n_proteins rows."""
    return pd.DataFrame({
        "entry": [f"P{i:04d}" for i in range(n_proteins)],
        "label": [i % 2 for i in range(n_proteins)],
    })


def get_X(n_proteins=24, n_features=8, seed=0):
    """Return a random per-protein feature matrix (n_proteins, n_features)."""
    rng = np.random.default_rng(seed)
    return rng.random((n_proteins, n_features))


class TestSelectProteins:
    """Positive tests for select_proteins() — one per parameter."""

    def test_df_seq_parameter(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(df_seq)
        assert {"cluster", "is_representative", "dist_to_rep"}.issubset(df.columns)

    def test_X_parameter(self):
        df_seq, X = get_df_seq(), get_X(n_features=12)
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=5)
        assert int(df["is_representative"].sum()) == 5

    @settings(deadline=None, max_examples=8)
    @given(n_clusters=st.integers(min_value=2, max_value=12))
    def test_n_clusters_parameter(self, n_clusters):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=n_clusters)
        assert int(df["is_representative"].sum()) == n_clusters

    def test_n_clusters_none_optimizes(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=None)
        assert 0 < int(df["is_representative"].sum()) <= len(df_seq)

    @settings(deadline=None, max_examples=6)
    @given(min_th=st.floats(min_value=0.0, max_value=1.0))
    def test_min_th_parameter(self, min_th):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df = aac.select_proteins(df_seq=df_seq, X=X, min_th=min_th)
        assert isinstance(df, pd.DataFrame)

    @given(on_center=st.booleans())
    @settings(deadline=None, max_examples=2)
    def test_on_center_parameter(self, on_center):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df = aac.select_proteins(df_seq=df_seq, X=X, on_center=on_center)
        assert isinstance(df, pd.DataFrame)

    @given(metric=st.sampled_from(["correlation", "manhattan", "euclidean", "cosine"]))
    @settings(deadline=None, max_examples=4)
    def test_metric_parameter(self, metric):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6, metric=metric)
        assert int(df["is_representative"].sum()) == 6

    @given(return_data=st.sampled_from(["annotated", "filtered", "both"]))
    @settings(deadline=None, max_examples=3)
    def test_return_data_parameter(self, return_data):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        out = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6, return_data=return_data)
        if return_data == "annotated":
            assert isinstance(out, pd.DataFrame)
        else:
            assert isinstance(out, tuple) and len(out) == 2


class TestSelectProteinsComplex:
    """Behavioural / contract tests for select_proteins()."""

    def test_annotated_columns_and_length(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6, return_data="annotated")
        assert len(df) == len(df_seq)
        assert list(df_seq.columns) == [c for c in df.columns
                                        if c not in ("cluster", "is_representative", "dist_to_rep")]

    def test_filtered_alignment(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df_r, X_r = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6, return_data="filtered")
        assert len(df_r) == X_r.shape[0] == 6
        assert (df_r["is_representative"] == 1).all()
        assert (df_r["dist_to_rep"] == 0).all()

    def test_both_mode(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df_full, X_r = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6, return_data="both")
        assert len(df_full) == len(df_seq)
        assert X_r.shape[0] == int(df_full["is_representative"].sum()) == 6

    def test_reps_equal_is_representative_sum(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=7)
        df_r, X_r = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=7, return_data="filtered")
        assert int(df["is_representative"].sum()) == len(df_r) == 7

    def test_reproducible_with_random_state(self):
        df_seq, X = get_df_seq(), get_X()
        df1 = aa.AAclust(random_state=0).select_proteins(df_seq=df_seq, X=X, n_clusters=6)
        df2 = aa.AAclust(random_state=0).select_proteins(df_seq=df_seq, X=X, n_clusters=6)
        assert df1["cluster"].to_list() == df2["cluster"].to_list()
        assert df1["is_representative"].to_list() == df2["is_representative"].to_list()

    def test_entries_preserved(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=6)
        assert df["entry"].to_list() == df_seq["entry"].to_list()


class TestSelectProteinsGoldenValues:
    """Exact-value tests on a hand-built feature matrix."""

    def _toy(self):
        # Three tight groups of 3 points each in 2D feature space
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],          # cluster A
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],    # cluster B
            [0.0, 10.0], [0.1, 10.0], [0.0, 10.1],       # cluster C
        ])
        df_seq = get_df_seq(n_proteins=9)
        return df_seq, X

    def test_medoids_have_zero_distance(self):
        df_seq, X = self._toy()
        aac = aa.AAclust(random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=3, metric="euclidean")
        assert (df.loc[df["is_representative"] == 1, "dist_to_rep"] == 0).all()
        assert int(df["is_representative"].sum()) == 3

    def test_euclidean_distance_value(self):
        df_seq, X = self._toy()
        aac = aa.AAclust(random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df = aac.select_proteins(df_seq=df_seq, X=X, n_clusters=3, metric="euclidean")
        # Within each cluster, every non-rep point is within ~0.15 of its medoid
        non_rep = df.loc[df["is_representative"] == 0, "dist_to_rep"]
        assert (non_rep < 0.2).all()
        assert (non_rep >= 0).all()


class TestSelectProteinsNegative:
    """Negative tests for select_proteins() — invalid inputs raise ValueError."""

    def test_length_mismatch(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X[:5])

    def test_missing_entry_column(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq.drop(columns=["entry"]), X=X)

    def test_non_unique_entries(self):
        df_seq, X = get_df_seq(), get_X()
        df_seq = df_seq.copy()
        df_seq["entry"] = ["dup"] * len(df_seq)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X)

    def test_df_seq_not_dataframe(self):
        X = get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq="invalid", X=X)

    def test_invalid_X(self):
        df_seq = get_df_seq()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X="invalid")

    def test_invalid_n_clusters(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, n_clusters=0)
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=get_df_seq(8), X=get_X(8), n_clusters=10_000)

    def test_invalid_min_th(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, min_th=-0.1)
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, min_th=1.5)

    def test_invalid_on_center(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, on_center="invalid")

    def test_invalid_metric(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, metric="invalid")

    def test_invalid_return_data(self):
        df_seq, X = get_df_seq(), get_X()
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_proteins(df_seq=df_seq, X=X, return_data="invalid")
