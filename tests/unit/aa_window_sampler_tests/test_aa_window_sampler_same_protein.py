"""This is a script to test the AAWindowSampler().sample_same_protein() method."""
import warnings
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
from aaanalysis.seq_analysis._backend.aa_window_sampler._utils import window_identity

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")

SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]
SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]


# I Helper Functions
def _add_pos_col(df_seq, seed=0, p_has_positive=0.7, max_pos_per_protein=3):
    rng = np.random.default_rng(seed)
    pos_list = []
    for s in df_seq["sequence"]:
        if rng.random() < p_has_positive:
            n_pos = int(rng.integers(1, max_pos_per_protein + 1))
            n_pos = min(n_pos, len(s))
            picks = rng.choice(np.arange(1, len(s) + 1), size=n_pos, replace=False)
            pos_list.append(sorted(int(p) for p in picks))
        else:
            pos_list.append([])
    df_seq = df_seq.copy()
    df_seq["pos"] = pos_list
    return df_seq


def _small_df_seq():
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [
            "ACDEFGHIKLMNPQRSTVWY" * 2,
            "ACDEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKLMNPQRSTVWY",
        ],
        "pos": [[5, 25], [10], []],
    })


# II Test Classes
class TestSampleSameProtein:
    """Test sample_same_protein() of the AAWindowSampler class."""

    # Positive tests
    def test_valid_df_seq(self):
        aaws = aa.AAWindowSampler()
        df_seq = _add_pos_col(aa.load_dataset(name="DOM_GSEC", n=20), seed=0)
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=10, window_size=9, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SCHEMA_SEGMENTS

    @settings(max_examples=10, deadline=1500)
    @given(n=some.integers(min_value=1, max_value=15))
    def test_valid_n(self, n):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=n, window_size=5, seed=0)
        # Returns at most n unique windows total
        assert len(df) <= n

    @settings(max_examples=8, deadline=1500)
    @given(n=some.integers(min_value=2, max_value=20))
    def test_valid_uniform_quota_per_protein(self, n):
        """The total budget n is split roughly uniformly across eligible proteins."""
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=n, window_size=5,
                                   min_distance_to_pos=0, seed=0)
        # Two eligible proteins (P1, P2); the larger share is at most ceil(n/2).
        counts = df["entry"].value_counts()
        assert counts.max() <= (n + 1) // 2

    def test_valid_topup_redistributes_shortfall(self):
        """If a protein cannot supply its quota, the rest fills from others."""
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIK", "ACDEFGHIKLMNPQRSTVWY" * 3],
            "pos": [[5], [10]],
        })
        aaws = aa.AAWindowSampler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=5,
                                       min_distance_to_pos=0, seed=0)
        assert len(df) == 20
        # P1's candidate pool is tiny; the bulk of windows comes from P2.
        counts = dict(df["entry"].value_counts())
        assert counts.get("P2", 0) > counts.get("P1", 0)

    @settings(max_examples=10, deadline=1500)
    @given(window_size=some.integers(min_value=1, max_value=11))
    def test_valid_window_size(self, window_size):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                   n=6, window_size=window_size, seed=0)
        assert (df["window"].str.len() == window_size).all()

    @settings(max_examples=10, deadline=1500)
    @given(min_distance=some.integers(min_value=0, max_value=5))
    def test_valid_min_distance_to_pos(self, min_distance):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=4, window_size=5,
                                   min_distance_to_pos=min_distance, seed=0)
        for entry, pos_list in zip(df_seq["entry"], df_seq["pos"]):
            picks = df[df["entry"] == entry]["source_position"].tolist()
            for c in picks:
                assert all(abs(c - p) >= min_distance for p in pos_list)

    @settings(max_examples=10, deadline=1500)
    @given(max_distance=some.integers(min_value=1, max_value=10))
    def test_valid_max_distance_to_pos(self, max_distance):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=4, window_size=5,
                                   max_distance_to_pos=max_distance, seed=0)
        for entry, pos_list in zip(df_seq["entry"], df_seq["pos"]):
            picks = df[df["entry"] == entry]["source_position"].tolist()
            for c in picks:
                assert min(abs(c - p) for p in pos_list) <= max_distance

    def test_valid_label_test_label_ref(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                   n=4, window_size=5,
                                   label_test=7, label_ref=3, seed=0)
        assert (df["label"] == 3).all()

    def test_valid_role(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                   n=4, window_size=5,
                                   role="Background", seed=0)
        assert (df["role"] == "Background").all()
        assert (df["strategy"] == "same_protein").all()

    def test_valid_default_role_is_negative(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                   n=4, window_size=5, seed=0)
        assert (df["role"] == "Negative").all()

    def test_valid_output_mode_sequences(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=4, window_size=5,
                                   output_mode="sequences",
                                   label_test=1, label_ref=0, seed=0)
        assert list(df.columns) == SCHEMA_SEQUENCES
        assert len(df) == len(df_seq)
        for _, row in df.iterrows():
            assert len(row["labels"]) == len(row["sequence"])
        labels_p1 = df[df["entry"] == "P1"]["labels"].iloc[0]
        assert labels_p1[4] == 1
        assert labels_p1[24] == 1
        assert 0 in labels_p1
        labels_p3 = df[df["entry"] == "P3"]["labels"].iloc[0]
        assert all(x is None for x in labels_p3)

    @settings(max_examples=10, deadline=1500)
    @given(seed=some.integers(min_value=0, max_value=10000))
    def test_valid_seed_determinism(self, seed):
        aaws = aa.AAWindowSampler()
        df_a = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                     n=6, window_size=5, seed=seed)
        df_b = aaws.sample_same_protein(df_seq=_small_df_seq(), pos_col="pos",
                                     n=6, window_size=5, seed=seed)
        pd.testing.assert_frame_equal(df_a, df_b)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_max_similarity_to_test_filters(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["AAAACDEFGHAAA"],
            "pos": [[2]],
        })
        aaws = aa.AAWindowSampler(max_similarity_to_test=0.0)
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=5, window_size=3,
                                   min_distance_to_pos=0, seed=0)
        assert (df["window"] != "AAA").all()

    def test_valid_max_similarity_within_ref_filters(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 3],
            "pos": [[3]],
        })
        aaws = aa.AAWindowSampler(max_similarity_within_ref=0.99)
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=20, window_size=4,
                                   min_distance_to_pos=0, seed=0)
        assert df["window"].nunique() == len(df)

    def test_valid_max_similarity_within_ref_across_proteins(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": [seq, seq],
            "pos": [[3], [3]],
        })
        aaws = aa.AAWindowSampler(max_similarity_within_ref=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=4,
                                       min_distance_to_pos=0, seed=0)
        windows = df["window"].tolist()
        for i, w_i in enumerate(windows):
            for w_j in windows[i + 1:]:
                assert window_identity(w_i, w_j) <= 0.5

    # Negative tests
    def test_invalid_df_seq(self):
        aaws = aa.AAWindowSampler()
        for invalid in [None, [], dict(), 1]:
            with pytest.raises((ValueError, AttributeError, TypeError)):
                aaws.sample_same_protein(df_seq=invalid, pos_col="pos",
                                      window_size=5, seed=0)

    def test_invalid_pos_col(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        with pytest.raises(ValueError):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="missing",
                                  window_size=5, seed=0)
        for invalid in [None, 1, [], {}]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col=invalid,
                                      window_size=5, seed=0)

    def test_invalid_n(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [0, -1, None, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=invalid, window_size=5, seed=0)

    def test_invalid_max_distance_to_pos(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=5,
                                      max_distance_to_pos=invalid, seed=0)

    def test_invalid_distance_to_pos_band(self):
        """min must not exceed max when both are given."""
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        with pytest.raises(ValueError):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                  n=4, window_size=5,
                                  min_distance_to_pos=5,
                                  max_distance_to_pos=2, seed=0)

    def test_invalid_window_size(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [0, -1, None, "5", 5.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=invalid, seed=0)

    def test_invalid_min_distance_to_pos(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=5,
                                      min_distance_to_pos=invalid, seed=0)

    def test_invalid_output_mode(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in ["seg", "Sequences", None, 1, ""]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=5,
                                      output_mode=invalid, seed=0)

    def test_invalid_role(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [None, 1, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=5,
                                      role=invalid, seed=0)

    def test_invalid_seed(self):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=5, seed=invalid)

    # Per-residue context filter (aa_context_col / context_in / context_out)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_aa_context_col_in(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
            "pos": [[10]],
            "topo": ["MMMMMTTTTTTTTTTCCCCC"],  # 5 M, 10 T (incl. pos 10), 5 C
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=10, window_size=3,
                                       min_distance_to_pos=0,
                                       aa_context_col="topo", context_in="T", seed=0)
        assert (df["source_position"].between(6, 15)).all()

    def test_valid_aa_context_col_out(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
            "pos": [[10]],
            "topo": ["MMMMMTTTTTTTTTTCCCCC"],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=10, window_size=3,
                                       min_distance_to_pos=0,
                                       aa_context_col="topo", context_out="C", seed=0)
        assert not (df["source_position"] > 15).any()

    def test_invalid_aa_context_col_missing(self):
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos", window_size=3,
                                      aa_context_col="missing", context_in="A", seed=0)

    def test_invalid_aa_context_col_length_mismatch(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["ACDEFGHIKL"], "pos": [[3]],
            "topo": ["MMMTT"],  # length 5 != sequence length 10
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="length"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos", window_size=3,
                                      aa_context_col="topo", context_in="M", seed=0)

    def test_invalid_aa_context_col_without_in_or_out(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["ACDEFGHIKL"], "pos": [[3]],
            "topo": ["MMMTTCCCCC"],
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="context_in"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos", window_size=3,
                                      aa_context_col="topo", seed=0)

    # Motif-match filter (motif_pwm / motif_score_threshold / motif_match)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_match_in_keeps_high_score_windows(self):
        import aaanalysis.utils as ut
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["AAACDEFGHIKLMNPQRSTV"],
            "pos": [[15]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=3,
                                       min_distance_to_pos=0,
                                       motif_pwm=pwm,
                                       motif_score_threshold=2.5,
                                       motif_match="in", seed=0)
        for w in df["window"]:
            score = sum(float(pwm.loc[i, c]) if c in pwm.columns else 0.0
                        for i, c in enumerate(w))
            assert score >= 2.5

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_match_out_drops_high_score_windows(self):
        import aaanalysis.utils as ut
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["AAACDEFGHIKLMNPQRSTV"],
            "pos": [[15]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=3,
                                       min_distance_to_pos=0,
                                       motif_pwm=pwm,
                                       motif_score_threshold=2.5,
                                       motif_match="out", seed=0)
        for w in df["window"]:
            score = sum(float(pwm.loc[i, c]) if c in pwm.columns else 0.0
                        for i, c in enumerate(w))
            assert score < 2.5

    def test_invalid_motif_pwm_shape(self):
        import aaanalysis.utils as ut
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        bad_pwm = pd.DataFrame(0.0, index=range(4),
                               columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      window_size=3, motif_pwm=bad_pwm,
                                      motif_score_threshold=1.0, seed=0)

    def test_invalid_motif_pwm_without_threshold(self):
        import aaanalysis.utils as ut
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_score_threshold"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      window_size=3, motif_pwm=pwm, seed=0)

    def test_invalid_motif_match(self):
        import aaanalysis.utils as ut
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_match"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      window_size=3, motif_pwm=pwm,
                                      motif_score_threshold=1.0,
                                      motif_match="invalid_mode", seed=0)

    def test_invalid_motif_pwm_ndarray_rejected(self):
        """ndarray PWM is rejected; DataFrame required."""
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        ndarray_pwm = np.zeros((3, 20))
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      window_size=3, motif_pwm=ndarray_pwm,
                                      motif_score_threshold=1.0, seed=0)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_pwm_dataframe_shuffled_columns(self):
        """DataFrame PWM with shuffled canonical-AA columns is reindexed internally."""
        import aaanalysis.utils as ut
        cols_shuffled = list(ut.LIST_CANONICAL_AA)
        np.random.RandomState(0).shuffle(cols_shuffled)
        df_pwm = pd.DataFrame(0.0, index=range(3), columns=cols_shuffled)
        df_pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["AAAAACDEFGHIKLMNPQRS"],
            "pos": [[10]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=3, window_size=3,
                                       min_distance_to_pos=0,
                                       motif_pwm=df_pwm,
                                       motif_score_threshold=1.0, seed=0)
        # All returned 3-mers should contain at least one 'A' (score >= 1.0).
        assert all("A" in w for w in df["window"])

    def test_invalid_context_in_without_aa_context_col(self):
        """context_in / context_out require aa_context_col (silent-drop guard)."""
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                      n=4, window_size=3,
                                      context_in=["A"], seed=0)

    def test_valid_row_order_independence_under_seed(self):
        """sample_same_protein output is df_seq-row-order-independent under fixed seed."""
        aaws = aa.AAWindowSampler(max_similarity_within_ref=0.5,
                              filter_iteratively=True)
        df_seq = pd.DataFrame({
            "entry": ["A", "B", "C"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * 3,
            "pos": [[5], [10], [15]],
        })
        df_seq_rev = df_seq.iloc[::-1].reset_index(drop=True)
        r1 = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=6, window_size=5, seed=42)
        r2 = aaws.sample_same_protein(df_seq=df_seq_rev, pos_col="pos",
                                   n=6, window_size=5, seed=42)
        r1_sorted = r1.sort_values("entry_win").reset_index(drop=True)
        r2_sorted = r2.sort_values("entry_win").reset_index(drop=True)
        assert r1_sorted[["entry_win", "window"]].equals(
            r2_sorted[["entry_win", "window"]]
        )

    def test_valid_entry_win_format_start_end(self):
        """entry_win is ``<entry>_<start_pos>-<end_pos>`` (1-based inclusive)."""
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["ACDEFGHIKLMNPQRSTVWY"], "pos": [[10]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                   n=1, window_size=5,
                                   min_distance_to_pos=2, seed=0)
        for r in df.itertuples():
            start = r.source_position - 2
            end = r.source_position + 2
            assert r.entry_win == f"P1_{start}-{end}"


class TestSampleSameProteinComplex:
    """Test sample_same_protein() with combinations of parameters."""

    @settings(max_examples=10, deadline=2500)
    @given(n=some.integers(min_value=1, max_value=10),
           window_size=some.integers(min_value=1, max_value=9),
           min_distance=some.integers(min_value=0, max_value=4),
           output_mode=some.sampled_from(["segments", "sequences"]),
           seed=some.integers(min_value=0, max_value=10000))
    def test_valid_combination(self, n, window_size, min_distance,
                               output_mode, seed):
        aaws = aa.AAWindowSampler()
        df_seq = _small_df_seq()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=n, window_size=window_size,
                                       min_distance_to_pos=min_distance,
                                       output_mode=output_mode, seed=seed)
        assert isinstance(df, pd.DataFrame)
        if output_mode == "segments":
            assert (df["window"].str.len() == window_size).all()
            assert (df["role"] == "Negative").all()
            assert len(df) <= n
        else:
            for r in df.itertuples():
                assert len(r.labels) == len(r.sequence)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_p1_anchor_even_window_is_right_heavy(self):
        """For even ``window_size``, the emitted window for a sampled center ``p``
        is ``seq[p - (L-1)//2 - 1 : p - (L-1)//2 - 1 + L]`` — i.e. the P1
        residue is at index ``(L-1)//2`` of the window, with floor-left /
        ceil-right asymmetry under Schechter-Berger cleavage convention.
        """
        seq = "ACDEFGHIKLMNPQRSTVWY"
        df_seq = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "pos": [[3]]})
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=4,
                                       min_distance_to_pos=0, seed=0)
        # half_left = (4-1)//2 = 1; window for center p (1-based) = seq[p-2:p+2].
        for r in df.itertuples():
            p = r.source_position
            expected = seq[p - 2:p + 2]
            assert r.window == expected, (
                f"P1 anchor mismatch at p={p}: got {r.window!r}, "
                f"expected {expected!r}")
            # P1 = seq[p-1] is the 2nd residue of the 4-mer (index half_left=1).
            assert r.window[1] == seq[p - 1]

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_p1_anchor_odd_window_is_symmetric(self):
        """Odd ``window_size`` is unchanged: P1 sits at the geometric center."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        df_seq = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "pos": [[5]]})
        aaws = aa.AAWindowSampler()
        df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=20, window_size=5,
                                       min_distance_to_pos=0, seed=0)
        for r in df.itertuples():
            p = r.source_position
            expected = seq[p - 3:p + 2]
            assert r.window == expected
            assert r.window[2] == seq[p - 1]

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_n_caps_total_returned(self):
        """``n`` is an upper bound on total returned windows across all proteins."""
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        for n in [1, 3, 7]:
            df = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                       n=n, window_size=5, seed=0)
            assert len(df) <= n

    def test_invalid_combination_zero_n(self):
        """n=0 is rejected (min_val=1)."""
        df_seq = _small_df_seq()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError):
            aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                  n=0, window_size=5, seed=0)
