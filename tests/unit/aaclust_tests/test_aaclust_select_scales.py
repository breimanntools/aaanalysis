"""This script tests the select_scales() method of the AAclust class."""

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
def get_df_scales(n_scales=40):
    """Return a real scales DataFrame (rows = amino acids, columns = scale IDs)."""
    df_scales = aa.load_scales()
    return df_scales[list(df_scales.columns)[:n_scales]]


def get_random_df_scales(n_scales=10):
    """Return a random but valid scales DataFrame (20 amino-acid rows)."""
    aa_letters = list("ARNDCQEGHILKMFPSTWYV")
    data = np.random.rand(len(aa_letters), n_scales)
    cols = [f"scale_{i}" for i in range(n_scales)]
    return pd.DataFrame(data, index=aa_letters, columns=cols)


class TestSelectScales:
    """Positive tests for select_scales() — one per parameter."""

    def test_df_scales_parameter(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=10)
        assert isinstance(df_sel, pd.DataFrame)
        assert df_sel.shape[1] == 10
        assert list(df_sel.index) == list(df_scales.index)
        assert set(df_sel.columns).issubset(set(df_scales.columns))

    @settings(deadline=None, max_examples=8)
    @given(n_clusters=st.integers(min_value=2, max_value=20))
    def test_n_clusters_parameter(self, n_clusters):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=n_clusters)
        assert df_sel.shape[1] == n_clusters

    def test_n_clusters_none_optimizes(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_sel = aac.select_scales(df_scales=df_scales, n_clusters=None)
        assert 0 < df_sel.shape[1] <= df_scales.shape[1]

    @settings(deadline=None, max_examples=8)
    @given(min_th=st.floats(min_value=0.0, max_value=1.0))
    def test_min_th_parameter(self, min_th):
        df_scales = get_df_scales(n_scales=30)
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_sel = aac.select_scales(df_scales=df_scales, min_th=min_th)
        assert isinstance(df_sel, pd.DataFrame)

    @given(on_center=st.booleans())
    @settings(deadline=None, max_examples=2)
    def test_on_center_parameter(self, on_center):
        df_scales = get_df_scales(n_scales=30)
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_sel = aac.select_scales(df_scales=df_scales, on_center=on_center)
        assert isinstance(df_sel, pd.DataFrame)

    @given(metric=st.sampled_from(["correlation", "manhattan", "euclidean", "cosine"]))
    @settings(deadline=None, max_examples=4)
    def test_metric_parameter(self, metric):
        df_scales = get_df_scales(n_scales=30)
        aac = aa.AAclust()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_sel = aac.select_scales(df_scales=df_scales, metric=metric, n_clusters=10)
        assert df_sel.shape[1] == 10


class TestSelectScalesComplex:
    """Behavioural / contract tests for select_scales()."""

    def test_columns_match_medoid_names(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=12)
        assert list(df_sel.columns) == aac.medoid_names_

    def test_fit_attributes_set(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        aac.select_scales(df_scales=df_scales, n_clusters=12)
        assert aac.n_clusters == 12
        assert aac.labels_ is not None
        assert len(aac.medoid_names_) == 12

    def test_values_preserved(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=10)
        for col in df_sel.columns:
            assert np.allclose(df_sel[col].values, df_scales[col].values)

    def test_random_df_scales(self):
        df_scales = get_random_df_scales(n_scales=15)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=5)
        assert df_sel.shape == (20, 5)

    def test_equivalent_to_manual_fit(self):
        """select_scales must equal the manual fit(...).medoid_names_ workflow (same seed)."""
        df_scales = get_df_scales(n_scales=40)
        aac_manual = aa.AAclust(random_state=42)
        sel = aac_manual.fit(np.asarray(df_scales).T, names=list(df_scales),
                             n_clusters=10).medoid_names_
        df_manual = df_scales[sel]
        aac = aa.AAclust(random_state=42)
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=10)
        assert list(df_sel.columns) == list(df_manual.columns)
        assert np.allclose(df_sel.values, df_manual.values)

    def test_feeds_into_cpp(self):
        """The returned frame is a valid df_scales for CPP (round-trip)."""
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        df_sel = aac.select_scales(df_scales=df_scales, n_clusters=10)
        df_cat = aa.load_scales(name="scales_cat")
        df_cat_sel = df_cat[df_cat["scale_id"].isin(df_sel.columns)]
        assert len(df_cat_sel) == df_sel.shape[1]


class TestSelectScalesNegative:
    """Negative tests for select_scales() — invalid inputs raise ValueError."""

    def test_invalid_df_scales(self):
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=None)
        with pytest.raises(ValueError):
            aac.select_scales(df_scales="invalid")
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=get_df_scales().values)

    def test_df_scales_with_nan(self):
        df_scales = get_df_scales(n_scales=10).copy()
        df_scales.iloc[0, 0] = np.nan
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales)

    def test_df_scales_duplicate_columns(self):
        df_scales = get_df_scales(n_scales=10).copy()
        df_scales.columns = ["dup"] * df_scales.shape[1]
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales)

    def test_invalid_n_clusters(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, n_clusters=0)
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, n_clusters="invalid")
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, n_clusters=10_000)

    def test_invalid_min_th(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, min_th=-0.1)
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, min_th=1.5)

    def test_invalid_on_center(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, on_center="invalid")

    def test_invalid_metric(self):
        df_scales = get_df_scales(n_scales=40)
        aac = aa.AAclust()
        with pytest.raises(ValueError):
            aac.select_scales(df_scales=df_scales, metric="invalid")
