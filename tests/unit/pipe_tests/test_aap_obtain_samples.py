"""This script tests the aaanalysis.pipe.obtain_samples() golden pipeline."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

import aaanalysis as aa
import aaanalysis.pipe as aap

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False

SEGMENT_COLS = ["entry_win", "entry", "sequence", "window", "source_position",
                "label", "role", "strategy"]


def _fixture_df_seq(n=20):
    """DOM_GSEC slice with a center anchor on positives; the rest are the unlabeled pool."""
    df = aa.load_dataset(name="DOM_GSEC", n=n)[["entry", "sequence", "label"]].copy()
    df = df.reset_index(drop=True)
    df["pos"] = (df["sequence"].str.len() // 2).astype(int)
    # Proteins labeled 0 carry no positive -> eligible cross-protein ("different") pool
    df.loc[df["label"] == 0, "pos"] = np.nan
    return df


df_seq = _fixture_df_seq()
df_feat = aa.load_features().head(8)


class TestObtainSamples:
    """Positive and negative tests for aap.obtain_samples(), one parameter per test."""

    # Positive tests
    def test_returns_triple(self):
        out = aap.obtain_samples(df_seq, strategy="same_protein", seed=0)
        assert isinstance(out, tuple) and len(out) == 3
        df_samples, ax, df_eval = out
        assert isinstance(df_samples, pd.DataFrame)
        assert ax is None
        assert isinstance(df_eval, pd.DataFrame) and len(df_eval) == 1

    def test_df_samples_segments_schema(self):
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein", seed=0)
        assert list(df_samples.columns) == SEGMENT_COLS

    def test_df_samples_has_both_classes(self):
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein", seed=0)
        assert set(df_samples["label"].unique()) == {0, 1}
        assert "Test" in df_samples["role"].unique()

    @pytest.mark.parametrize("strategy", ["same_protein", "different_protein", "synthetic"])
    def test_strategy_parameter(self, strategy):
        df_samples, _, df_eval = aap.obtain_samples(df_seq, strategy=strategy, seed=1)
        assert df_eval["n_positive"].iloc[0] > 0
        assert (df_samples["label"] == 0).sum() >= 0

    @settings(max_examples=4, deadline=None)
    @given(n=st.integers(min_value=1, max_value=15))
    def test_n_parameter(self, n):
        df_samples, _, df_eval = aap.obtain_samples(df_seq, strategy="same_protein", n=n, seed=2)
        assert df_eval["n_negative"].iloc[0] <= n

    @settings(max_examples=4, deadline=None)
    @given(window_size=st.integers(min_value=3, max_value=15))
    def test_window_size_parameter(self, window_size):
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein",
                                              window_size=window_size, seed=3)
        win_len = df_samples["window"].str.len()
        assert (win_len == window_size).all()

    def test_pos_col_parameter(self):
        df = df_seq.rename(columns={"pos": "anchor"})
        df_samples, _, _ = aap.obtain_samples(df, strategy="same_protein",
                                              pos_col="anchor", seed=0)
        assert (df_samples["label"] == 1).sum() > 0

    @settings(max_examples=3, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=50))
    def test_seed_parameter(self, seed):
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="different_protein", seed=seed)
        assert len(df_samples) > 0

    def test_max_similarity_to_test_parameter(self):
        df_samples, _, df_eval = aap.obtain_samples(
            df_seq, strategy="same_protein", max_similarity_to_test=0.5, seed=4)
        max_sim = df_eval["max_similarity_to_test"].iloc[0]
        assert np.isnan(max_sim) or max_sim <= 0.5 + 1e-9

    def test_plot_parameter(self):
        df_samples, ax_on, _ = aap.obtain_samples(df_seq, strategy="same_protein",
                                                  plot=True, seed=0)
        _, ax_off, _ = aap.obtain_samples(df_seq, strategy="same_protein", plot=False, seed=0)
        # One sequence-logo panel per role group (Test + the references).
        n_roles = df_samples["role"].nunique()
        assert ax_on is not None and len(ax_on) == n_roles
        assert ax_off is None

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein",
                                                 verbose=verbose, seed=0)
            assert len(df_samples) > 0

    def test_n_jobs_parameter(self):
        for n_jobs in [None, 1]:
            df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein",
                                                 n_jobs=n_jobs, seed=0)
            assert len(df_samples) > 0

    # Negative tests
    def test_invalid_df_seq(self):
        with pytest.raises(ValueError):
            aap.obtain_samples("invalid", strategy="same_protein")

    def test_invalid_pos_col(self):
        with pytest.raises(ValueError, match="pos_col"):
            aap.obtain_samples(df_seq, pos_col="not_a_column", strategy="same_protein")

    def test_invalid_strategy(self):
        for strategy in ["motif_matched", "bogus", 1]:
            with pytest.raises(ValueError):
                aap.obtain_samples(df_seq, strategy=strategy)

    def test_invalid_n(self):
        for n in [0, -1, 1.5]:
            with pytest.raises(ValueError):
                aap.obtain_samples(df_seq, strategy="same_protein", n=n)

    def test_invalid_window_size(self):
        for window_size in [0, -3, "bad"]:
            with pytest.raises(ValueError):
                aap.obtain_samples(df_seq, strategy="same_protein", window_size=window_size)

    def test_invalid_max_similarity_to_test(self):
        for val in [-0.1, 1.5, "bad"]:
            with pytest.raises(ValueError):
                aap.obtain_samples(df_seq, strategy="same_protein", max_similarity_to_test=val)

    def test_invalid_seed(self):
        for seed in [-1, "bad", 1.5]:
            with pytest.raises(ValueError):
                aap.obtain_samples(df_seq, strategy="same_protein", seed=seed)

    def test_invalid_plot(self):
        with pytest.raises(ValueError):
            aap.obtain_samples(df_seq, strategy="same_protein", plot="yes")

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aap.obtain_samples(df_seq, strategy="same_protein", verbose="loud")

    def test_reliable_negatives_without_df_feat(self):
        with pytest.raises(ValueError, match="df_feat"):
            aap.obtain_samples(df_seq, strategy="different_protein", reliable_negatives=True)

    def test_reliable_negatives_wrong_strategy(self):
        with pytest.raises(ValueError, match="different_protein"):
            aap.obtain_samples(df_seq, strategy="same_protein",
                               reliable_negatives=True, df_feat=df_feat)

    def test_no_positive_windows_raises(self):
        df = df_seq.copy()
        df["pos"] = np.nan
        with pytest.raises(ValueError, match="positive window"):
            aap.obtain_samples(df, strategy="same_protein")

    def test_missing_sequence_column_raises(self):
        df = df_seq.drop(columns=["sequence"])
        with pytest.raises(ValueError, match="sequence"):
            aap.obtain_samples(df, strategy="same_protein")

    def test_pos_col_with_nan_in_list_cell(self):
        # A list cell carrying a NaN must not crash positive extraction; the NaN is dropped
        # (synthetic strategy does not route pos_col through the sampler's own parser).
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQ", "MNPQRSTVWYACDE"],
            "pos": [[7, np.nan], np.nan],
        })
        df_samples, _, df_eval = aap.obtain_samples(df, strategy="synthetic",
                                                    window_size=5, n=1, seed=0)
        assert df_eval["n_positive"].iloc[0] == 1


class TestObtainSamplesComplex:
    """Combinations, reproducibility, parity, and the PU path."""

    def test_reproducible_same_seed(self):
        df_1, _, _ = aap.obtain_samples(df_seq, strategy="different_protein", n=8, seed=7)
        df_2, _, _ = aap.obtain_samples(df_seq, strategy="different_protein", n=8, seed=7)
        assert df_1.equals(df_2)

    def test_byte_identical_negatives_to_sampler(self):
        # The negatives must equal the explicit AAWindowSampler.sample_same_protein call.
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="same_protein", n=10, seed=5)
        df_neg = df_samples[df_samples["label"] == 0].reset_index(drop=True)
        aaws = aa.AAWindowSampler(verbose=False, random_state=5, max_similarity_to_test=None)
        df_ref = aaws.sample_same_protein(df_seq=df_seq, n=10, window_size=9,
                                          pos_col="pos", seed=5).reset_index(drop=True)
        assert df_neg[SEGMENT_COLS].equals(df_ref[SEGMENT_COLS])

    def test_default_n_balances(self):
        df_samples, _, df_eval = aap.obtain_samples(df_seq, strategy="same_protein", seed=0)
        # default n matches the number of positive windows -> at most balanced
        assert df_eval["n_negative"].iloc[0] <= df_eval["n_positive"].iloc[0]
        assert df_eval["balance_ratio"].iloc[0] <= 1.0 + 1e-9

    def test_synthetic_strategy_tags(self):
        df_samples, _, _ = aap.obtain_samples(df_seq, strategy="synthetic", n=6, seed=1)
        neg = df_samples[df_samples["label"] == 0]
        assert all(s.startswith("synthetic") for s in neg["strategy"].unique())

    def test_synthetic_reports_no_source_proteins(self):
        # Synthetic windows have no source protein -> coverage must not count empty entries.
        _, _, df_eval = aap.obtain_samples(df_seq, strategy="synthetic", n=6, seed=1)
        assert df_eval["n_source_proteins"].iloc[0] == 0
        assert df_eval["protein_coverage"].iloc[0] == 0.0

    def test_pu_path_returns_reliable_negatives(self):
        df_samples, _, df_eval = aap.obtain_samples(
            df_seq, strategy="different_protein", reliable_negatives=True,
            df_feat=df_feat, n=4, window_size=9, seed=1, n_jobs=1)
        neg = df_samples[df_samples["label"] == 0]
        assert (neg["role"] == "Negative").all()
        assert df_eval["n_negative"].iloc[0] <= 4

    def test_pu_path_reproducible(self):
        kws = dict(strategy="different_protein", reliable_negatives=True,
                   df_feat=df_feat, n=4, window_size=9, seed=2, n_jobs=1)
        df_1, _, _ = aap.obtain_samples(df_seq, **kws)
        df_2, _, _ = aap.obtain_samples(df_seq, **kws)
        assert df_1.equals(df_2)

    def test_combined_valid_parameters(self):
        df_samples, ax, df_eval = aap.obtain_samples(
            df_seq, strategy="same_protein", n=12, window_size=11,
            max_similarity_to_test=0.7, plot=True, seed=3, verbose=False)
        assert ax is not None
        assert (df_samples["window"].str.len() == 11).all()

    def test_plot_returns_logo_axes_per_group(self):
        from matplotlib.axes import Axes
        df_samples, ax, _ = aap.obtain_samples(df_seq, strategy="different_protein",
                                               n=10, plot=True, seed=1)
        # multi_logo returns one (logo, info-bar) Axes pair per sampled group.
        assert len(ax) == df_samples["role"].nunique()
        for logo_ax, info_ax in ax:
            assert isinstance(logo_ax, Axes) and isinstance(info_ax, Axes)

    def test_combined_invalid_parameters(self):
        with pytest.raises(ValueError):
            aap.obtain_samples(df_seq, strategy="bogus", n=-5, seed="bad")

    def test_eval_leakage_respects_threshold(self):
        _, _, df_eval = aap.obtain_samples(df_seq, strategy="same_protein",
                                           max_similarity_to_test=0.4, seed=8)
        max_sim = df_eval["max_similarity_to_test"].iloc[0]
        assert np.isnan(max_sim) or max_sim <= 0.4 + 1e-9


class TestObtainSamplesGoldenValues:
    """Hand-computed expectations on a tiny crafted input."""

    def _tiny(self):
        # Two short proteins; P1 anchor at position 5 (1-based), window_size=5.
        return pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKL", "MNPQRSTVWY"],
            "pos": [5, np.nan],
        })

    def test_positive_window_string_and_count(self):
        df = self._tiny()
        df_samples, _, df_eval = aap.obtain_samples(df, strategy="different_protein",
                                                    window_size=5, n=1, seed=0)
        pos = df_samples[df_samples["label"] == 1]
        # window_size=5 -> half_left=2; anchor p=5 -> start=2 (0-based) -> "DEFGH"
        assert df_eval["n_positive"].iloc[0] == 1
        assert pos["window"].iloc[0] == "DEFGH"
        assert pos["entry_win"].iloc[0] == "P1_3-7"
        assert pos["role"].iloc[0] == "Test"
        assert pos["strategy"].iloc[0] == "test"

    def test_balance_ratio_exact(self):
        df = self._tiny()
        _, _, df_eval = aap.obtain_samples(df, strategy="different_protein",
                                           window_size=5, n=1, seed=0)
        n_pos = df_eval["n_positive"].iloc[0]
        n_neg = df_eval["n_negative"].iloc[0]
        assert df_eval["balance_ratio"].iloc[0] == pytest.approx(n_neg / n_pos)

    def test_self_identity_is_one(self):
        # A negative drawn identical to the test window would leak (identity 1.0).
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAAAAAAAAA", "AAAAAAAAAA"],
            "pos": [5, np.nan],
        })
        _, _, df_eval = aap.obtain_samples(df, strategy="different_protein",
                                           window_size=5, n=1, seed=0)
        # Both proteins are poly-A, so any sampled window is identical to the test window.
        assert df_eval["max_similarity_to_test"].iloc[0] == pytest.approx(1.0)
