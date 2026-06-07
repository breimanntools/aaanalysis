"""This is a script to test the AAWindowSampler().sample_different_protein() method."""
import warnings
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]
SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]


# I Helper Functions
def _df_seq_mixed():
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3", "P4"],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2] * 4,
        "pos": [[5], [], [], []],
    })


# II Test Classes
class TestSampleDifferentProtein:
    """Test sample_different_protein() of the AAWindowSampler class."""

    # Positive tests
    def test_valid_df_seq(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=10, window_size=9, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SCHEMA_SEGMENTS

    @settings(max_examples=10, deadline=None)
    @given(n=some.integers(min_value=1, max_value=20))
    def test_valid_n(self, n):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=n, window_size=5, seed=0)
        assert len(df) <= n

    @settings(max_examples=10, deadline=None)
    @given(window_size=some.integers(min_value=1, max_value=11))
    def test_valid_window_size(self, window_size):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=5, window_size=window_size, seed=0)
        assert (df["window"].str.len() == window_size).all()

    def test_valid_default_role_is_unlabeled(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=5, window_size=5, seed=0)
        assert (df["role"] == "Unlabeled").all()

    def test_valid_role_override(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=5, window_size=5,
                                        role="Background", seed=0)
        assert (df["role"] == "Background").all()

    def test_valid_label_ref(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=5, window_size=5,
                                        label_test=99, label_ref=42, seed=0)
        assert (df["label"] == 42).all()

    def test_valid_candidate_proteins(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=10, window_size=5,
                                        candidate_proteins=["P3"], seed=0)
        assert set(df["entry"]) == {"P3"}

    def test_valid_output_mode_sequences(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                        n=10, window_size=5,
                                        output_mode="sequences",
                                        label_test=1, label_ref=0, seed=0)
        assert list(df.columns) == SCHEMA_SEQUENCES
        assert len(df) == len(df_seq)
        # Positive-protein P1 has label_test=1 at its positive position (5),
        # None everywhere else; sampled centers from candidate proteins carry
        # label_ref=0. The per-residue labels frame is mergeable across calls.
        labels_p1 = df[df["entry"] == "P1"]["labels"].iloc[0]
        assert labels_p1[4] == 1  # 1-based position 5 → index 4
        assert sum(1 for x in labels_p1 if x == 1) == 1
        eligible = df[df["entry"] != "P1"]
        any_ref = any(0 in lab for lab in eligible["labels"])
        assert any_ref

    @settings(max_examples=10, deadline=None)
    @given(seed=some.integers(min_value=0, max_value=10000))
    def test_valid_seed_determinism(self, seed):
        aaws = aa.AAWindowSampler()
        df_a = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                          n=10, window_size=5, seed=seed)
        df_b = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                          n=10, window_size=5, seed=seed)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_valid_excludes_positive_proteins(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=_df_seq_mixed(), pos_col="pos",
                                        n=20, window_size=5, seed=0)
        assert "P1" not in set(df["entry"])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_max_similarity_to_test_filters(self):
        aaws = aa.AAWindowSampler(max_similarity_to_test=0.0,
                              filter_iteratively=True)
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAAAAAAAAA", "AAAAAAAAAA"],
            "pos": [[3], []],
        })
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3, seed=0)
        assert (df["window"] != "AAA").all() or len(df) == 0

    # Negative tests
    def test_invalid_df_seq(self):
        aaws = aa.AAWindowSampler()
        for invalid in [None, [], dict(), 1]:
            with pytest.raises((ValueError, AttributeError, TypeError)):
                aaws.sample_different_protein(df_seq=invalid, pos_col="pos",
                                           n=5, window_size=5, seed=0)

    def test_invalid_n(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        for invalid in [0, -1, None, "5", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=invalid, window_size=5, seed=0)

    def test_invalid_window_size(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        for invalid in [0, -1, None, "5", 5.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=invalid, seed=0)

    def test_invalid_output_mode(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        for invalid in ["seg", "Sequences", None, 1]:
            with pytest.raises(ValueError):
                aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=5,
                                           output_mode=invalid, seed=0)

    def test_invalid_no_eligible_proteins(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed().assign(pos=[[1], [1], [1], [1]])
        with pytest.raises(ValueError, match="No eligible proteins"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=5, seed=0)

    def test_invalid_window_too_large(self):
        aaws = aa.AAWindowSampler()
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"], "sequence": ["AAAA", "BBBB"], "pos": [[], []],
        })
        with pytest.raises(ValueError, match="window_size"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=9, seed=0)

    def test_invalid_seed(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=5, seed=invalid)

    # Per-residue context filter (aa_context_col / context_in / context_out)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_aa_context_col_in(self):
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * 2,
            "pos": [[5], []],
            "topo": ["MMMMMMMMMMTTTTTTTTTT"] * 2,
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=10, window_size=3,
                                            aa_context_col="topo", context_in="T", seed=0)
        assert (df["source_position"] > 10).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_aa_context_col_out(self):
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * 2,
            "pos": [[5], []],
            "topo": ["MMMMMMMMMMTTTTTTTTTT"] * 2,
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=10, window_size=3,
                                            aa_context_col="topo", context_out="M", seed=0)
        assert (df["source_position"] > 10).all()

    def test_invalid_aa_context_col_missing(self):
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           aa_context_col="missing", context_in="A", seed=0)

    def test_invalid_aa_context_col_length_mismatch(self):
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKL"] * 2, "pos": [[3], []],
            "topo": ["MMMTT", "MMMMMTTTTT"],  # P1 mismatched
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="length"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           aa_context_col="topo", context_in="M", seed=0)

    def test_invalid_aa_context_col_without_in_or_out(self):
        df_seq = _df_seq_mixed().assign(topo=["MMMMMMMMMMTTTTTTTTTT" * 2] * 4)
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="context_in"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           aa_context_col="topo", seed=0)

    # Motif-match filter (motif_pwm / motif_score_threshold / motif_match)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_match_in_keeps_high_score_windows(self):
        import aaanalysis.utils as ut
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRSTV", "AAACDEFGHIKLMNPQRS"],
            "pos": [[5], []],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=3,
                                            motif_pwm=pwm,
                                            motif_score_threshold=2.5,
                                            motif_match="in", seed=0)
        for w in df["window"]:
            score = sum(float(pwm.loc[i, c]) if c in pwm.columns else 0.0
                        for i, c in enumerate(w))
            assert score >= 2.5

    def test_valid_motif_match_out_drops_high_score_windows(self):
        import aaanalysis.utils as ut
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRSTV", "AAACDEFGHIKLMNPQRS"],
            "pos": [[5], []],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=10, window_size=3,
                                            motif_pwm=pwm,
                                            motif_score_threshold=2.5,
                                            motif_match="out", seed=0)
        for w in df["window"]:
            score = sum(float(pwm.loc[i, c]) if c in pwm.columns else 0.0
                        for i, c in enumerate(w))
            assert score < 2.5

    def test_invalid_motif_pwm_shape(self):
        import aaanalysis.utils as ut
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        bad_pwm = pd.DataFrame(0.0, index=range(4),
                               columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=bad_pwm,
                                           motif_score_threshold=1.0, seed=0)

    def test_invalid_motif_pwm_without_threshold(self):
        import aaanalysis.utils as ut
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_score_threshold"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3, motif_pwm=pwm, seed=0)

    def test_invalid_motif_match(self):
        import aaanalysis.utils as ut
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        pwm = pd.DataFrame(0.0, index=range(3), columns=list(ut.LIST_CANONICAL_AA))
        with pytest.raises(ValueError, match="motif_match"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3, motif_pwm=pwm,
                                           motif_score_threshold=1.0,
                                           motif_match="invalid_mode", seed=0)

    def test_invalid_motif_pwm_ndarray_rejected(self):
        """ndarray PWM is rejected; DataFrame required."""
        import numpy as np
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        ndarray_pwm = np.zeros((3, 20))
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=ndarray_pwm,
                                           motif_score_threshold=1.0, seed=0)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_pwm_dataframe_shuffled_columns(self):
        """DataFrame PWM with shuffled canonical-AA columns is reindexed internally."""
        import numpy as np
        import aaanalysis.utils as ut
        cols_shuffled = list(ut.LIST_CANONICAL_AA)
        np.random.RandomState(0).shuffle(cols_shuffled)
        df_pwm = pd.DataFrame(0.0, index=range(3), columns=cols_shuffled)
        df_pwm["A"] = 1.0
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNPQRSTV", "AAACDEFGHIKLMNPQRS"],
            "pos": [[5], []],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=3,
                                            motif_pwm=df_pwm,
                                            motif_score_threshold=2.5, seed=0)
        # All returned windows should be 'AAA' (the only score-2.5+ trimer with this PWM).
        assert (df["window"] == "AAA").all()

    def test_invalid_motif_pwm_dataframe_missing_canonical_aa(self):
        """DataFrame PWM missing a canonical AA column is rejected."""
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        # Omit 'Y'.
        cols = [c for c in "ACDEFGHIKLMNPQRSTVWY" if c != "Y"]
        df_pwm = pd.DataFrame(0.0, index=range(3), columns=cols)
        with pytest.raises(ValueError, match="missing"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3, motif_pwm=df_pwm,
                                           motif_score_threshold=1.0, seed=0)

    def test_invalid_motif_pwm_dataframe_extra_column(self):
        """DataFrame PWM with a non-canonical column (e.g. 'X') is rejected."""
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        cols = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
        df_pwm = pd.DataFrame(0.0, index=range(3), columns=cols)
        with pytest.raises(ValueError, match="extra"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3, motif_pwm=df_pwm,
                                           motif_score_threshold=1.0, seed=0)

    def test_invalid_context_in_without_aa_context_col(self):
        """context_in / context_out require aa_context_col (silent-drop guard)."""
        df_seq = _df_seq_mixed()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           context_in=["A"], seed=0)
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           context_out=["M"], seed=0)


class TestSampleDifferentProteinComplex:
    """Test sample_different_protein() with combinations of parameters."""

    @settings(max_examples=10, deadline=None)
    @given(n=some.integers(min_value=1, max_value=30),
           window_size=some.integers(min_value=1, max_value=9),
           role=some.sampled_from(["Negative", "Background", "Unlabeled"]),
           output_mode=some.sampled_from(["segments", "sequences"]),
           seed=some.integers(min_value=0, max_value=10000))
    def test_valid_combination(self, n, window_size, role, output_mode, seed):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_mixed()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_different_protein(df_seq=df_seq, pos_col="pos",
                                            n=n, window_size=window_size,
                                            role=role, output_mode=output_mode,
                                            seed=seed)
        assert isinstance(df, pd.DataFrame)
        if output_mode == "segments":
            assert (df["window"].str.len() == window_size).all()
            assert (df["role"] == role).all()
            assert "P1" not in set(df["entry"])
        else:
            assert len(df) == len(df_seq)
