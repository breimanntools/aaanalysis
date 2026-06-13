"""This is a script to test the sequence-analysis / protein-design seams.

Integration tier (ADR-0031). Real components, no mocks.

Seams covered:
  9.  AAWindowSampler.sample_* windows -> AAlogo logo matrix
  10. SeqMut.mutate -> re-featurize via SequenceFeature (design loop closes);
      and SeqMut.mutate(df_feat=...) -> ΔCPP against a real CPP feature set
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from tests import _pipeline

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.integration

POS_COLS = ["entry", "sequence", "tmd_start", "tmd_stop", "label"]


# ---------------------------------------------------------------------------
# Seam 9: AAWindowSampler -> AAlogo
# ---------------------------------------------------------------------------
class TestSamplerToLogo:
    """Sampled fixed-length windows drive an AAlogo composition matrix."""

    def test_windows_feed_logo(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        df_win = aa.AAWindowSampler(random_state=0).sample_synthetic(
            df_seq=df_seq, n=8, window_size=9, generator="global_freq", seed=0)
        df_parts = pd.DataFrame({"tmd": df_win["window"].to_list()})
        df_logo = aa.AAlogo(logo_type="probability").get_df_logo(df_parts=df_parts, tmd_len=9)
        assert df_logo.shape[0] == 9  # one row per window position

    @settings(max_examples=4, deadline=None)
    @given(window_size=some.integers(min_value=5, max_value=12))
    def test_logo_rows_equal_window_length(self, window_size):
        # Property across the seam: the sampler emits window_size-length windows
        # AND the logo's position axis reflects that actual length.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        df_win = aa.AAWindowSampler(random_state=0).sample_synthetic(
            df_seq=df_seq, n=6, window_size=window_size, generator="global_freq", seed=0)
        # The sampler's real contract — every window is exactly window_size residues.
        assert (df_win["window"].str.len() == window_size).all()
        df_parts = pd.DataFrame({"tmd": df_win["window"].to_list()})
        # tmd_len omitted: the logo INFERS length from the windows, so the row count
        # reflects the actual sampled length rather than a forced tmd_len.
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts)
        assert df_logo.shape[0] == window_size

    def test_absent_label_empties_logo(self):
        # Composition failure: window rows carry a label; selecting a label_test
        # absent from the pool empties the set, and the logo step must say so.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        df_win = aa.AAWindowSampler(random_state=0).sample_synthetic(
            df_seq=df_seq, n=6, window_size=9, generator="global_freq", seed=0)
        df_parts = pd.DataFrame({"tmd": df_win["window"].to_list()})
        # Two distinct labels (0/2) so the homogeneity guard passes; neither is the
        # requested label_test=1, so the post-filter pool is empty.
        labels = np.array([0, 0, 0, 2, 2, 2])
        with pytest.raises(ValueError, match="label_test"):
            aa.AAlogo().get_df_logo(df_parts=df_parts, labels=labels, label_test=1, tmd_len=9)


# ---------------------------------------------------------------------------
# Seam 10: SeqMut.mutate -> re-featurize via SequenceFeature
# ---------------------------------------------------------------------------
class TestMutationToFeatures:
    """A mutated sequence flows back through SequenceFeature (and CPP ΔCPP)."""

    @staticmethod
    def _df_seq():
        return aa.load_dataset(name="DOM_GSEC", n=5)[POS_COLS].copy()

    def test_refeaturize_mutated_sequence(self):
        df_seq = self._df_seq()
        row = df_seq.iloc[0]
        muts = pd.DataFrame({"entry": [row["entry"]],
                             "pos": [int(row["tmd_start"]) + 1], "to_aa": ["A"]})
        df_mut = aa.SeqMut(verbose=False).mutate(df_seq=df_seq, mutations=muts)
        df_seq_mut = df_seq.copy()
        df_seq_mut.loc[df_seq_mut["entry"] == row["entry"], "sequence"] = df_mut["sequence_mut"].iloc[0]
        dp_orig = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        dp_mut = aa.SequenceFeature().get_df_parts(df_seq=df_seq_mut)
        assert dp_orig.shape == dp_mut.shape                       # same matrix shape
        assert (dp_orig["tmd"] != dp_mut["tmd"]).sum() == 1        # exactly one protein changed

    def test_point_mutation_preserves_length(self):
        # Metamorphic: a single substitution must not change sequence length.
        df_seq = self._df_seq()
        row = df_seq.iloc[0]
        muts = pd.DataFrame({"entry": [row["entry"]],
                             "pos": [int(row["tmd_start"]) + 1], "to_aa": ["A"]})
        df_mut = aa.SeqMut(verbose=False).mutate(df_seq=df_seq, mutations=muts)
        assert len(df_mut["sequence_mut"].iloc[0]) == len(row["sequence"])

    def test_delta_cpp_against_real_feat_set(self):
        # Closes the design loop: mutate scored against a real CPP feature set.
        base = _pipeline.build_pipeline(n=5, n_filter=15)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)[POS_COLS].copy()
        row = df_seq.iloc[0]
        muts = pd.DataFrame({"entry": [row["entry"]],
                             "pos": [int(row["tmd_start"]) + 1], "to_aa": ["A"]})
        df_mut = aa.SeqMut(verbose=False).mutate(df_seq=df_seq, mutations=muts,
                                                 df_feat=base["df_feat"])
        assert "delta_cpp" in df_mut.columns
        assert np.isfinite(df_mut["delta_cpp"]).all()

    def test_mutation_unknown_entry_rejected(self):
        # Composition failure: the mutations table must reference df_seq entries.
        df_seq = self._df_seq()
        muts = pd.DataFrame({"entry": ["NOT_AN_ENTRY"], "pos": [5], "to_aa": ["A"]})
        with pytest.raises(ValueError, match=r"is not in 'df_seq'"):
            aa.SeqMut(verbose=False).mutate(df_seq=df_seq, mutations=muts)
