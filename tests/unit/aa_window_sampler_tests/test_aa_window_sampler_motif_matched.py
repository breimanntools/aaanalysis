"""This is a script to test the AAWindowSampler().sample_motif_matched() method."""
import warnings
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")

SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]
SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]
AA_IDX = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}


# I Helper Functions
def _pwm_for_a(window_size=3):
    """PWM that scores 'A' at every position. Score of 'AAA' = window_size."""
    pwm = np.zeros((window_size, len(ut.LIST_CANONICAL_AA)))
    pwm[:, AA_IDX["A"]] = 1.0
    return pwm


def _df_seq_with_aaa():
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [
            "ACDEFGHIKLAAA",         # P1: positive row, has 'AAA' window
            "ACDEFGHIKLAAA",         # P2: candidate, has 'AAA' window
            "ACDEFGHIKLMNPQR",       # P3: candidate, no 'AAA' window
        ],
        "pos": [[5], [], []],
    })


# II Test Classes
class TestSampleMotifMatched:
    """Test sample_motif_matched() of the AAWindowSampler class."""

    # Positive tests
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_df_seq(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=_df_seq_with_aaa(), pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5, seed=0)
        assert isinstance(df, pd.DataFrame)
        # sample_motif_matched extends the standard schema with `motif_score`.
        assert list(df.columns) == SCHEMA_SEGMENTS + ["motif_score"]
        # All scores must meet the threshold the user requested.
        assert (df["motif_score"] >= 2.5).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @settings(max_examples=10, deadline=1500)
    @given(n=some.integers(min_value=1, max_value=20))
    def test_valid_n(self, n):
        df_seq = pd.DataFrame({
            "entry": [f"P{i}" for i in range(5)],
            "sequence": ["AAACDEFGAAA"] * 5,
            "pos": [[]] * 5,
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=n, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5, seed=0)
        assert len(df) <= n

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @settings(max_examples=8, deadline=1500)
    @given(window_size=some.integers(min_value=2, max_value=7))
    def test_valid_window_size(self, window_size):
        seq = "A" * 20
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": [seq], "pos": [[]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=10, window_size=window_size,
                                        motif_pwm=_pwm_for_a(window_size),
                                        motif_score_threshold=float(window_size),
                                        seed=0)
        assert (df["window"].str.len() == window_size).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_score_threshold_strict(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        # Strict threshold: only 'AAA' (score 3.0) should pass.
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=10, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5, seed=0)
        assert (df["window"] == "AAA").all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_score_threshold_lax(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        # Lax threshold (any window with at least one A): more hits.
        df_strict = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                               n=50, window_size=3,
                                               motif_pwm=_pwm_for_a(3),
                                               motif_score_threshold=2.5,
                                               seed=0)
        df_lax = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=50, window_size=3,
                                            motif_pwm=_pwm_for_a(3),
                                            motif_score_threshold=0.5,
                                            seed=0)
        assert len(df_lax) >= len(df_strict)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_pwm_shape(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for ws in [3, 5, 7]:
            df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=ws,
                                            motif_pwm=_pwm_for_a(ws),
                                            motif_score_threshold=float(ws),
                                            seed=0)
            assert (df["window"].str.len() == ws).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_pos_col_excludes_positives(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=10, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5, seed=0)
        # P1 is the positive row and must not appear among hits.
        assert "P1" not in set(df["entry"])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_role_default_is_negative(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5, seed=0)
        assert (df["role"] == "Negative").all()
        assert (df["strategy"] == "motif_matched").all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_role_override(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5,
                                        role="custom_decoy", seed=0)
        assert (df["role"] == "custom_decoy").all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_label_ref(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5,
                                        label_ref=42, seed=0)
        assert (df["label"] == 42).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @settings(max_examples=8, deadline=1500)
    @given(seed=some.integers(min_value=0, max_value=10000))
    def test_valid_seed_determinism(self, seed):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        kw = dict(df_seq=df_seq, pos_col="pos", n=5, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5, seed=seed)
        pd.testing.assert_frame_equal(aaws.sample_motif_matched(**kw),
                                      aaws.sample_motif_matched(**kw))

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_output_mode_sequences(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5,
                                        output_mode="sequences", seed=0)
        assert list(df.columns) == SCHEMA_SEQUENCES
        assert len(df) == len(df_seq)

    def test_valid_ranking_by_score(self):
        # Build a df where the top window scores higher than the rest.
        seq = "A" * 5 + "C" * 10 + "AAAAA"
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": [seq], "pos": [[]],
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=3, window_size=5,
                                        motif_pwm=_pwm_for_a(5),
                                        motif_score_threshold=0.0, seed=0)
        # All-A windows (score 5.0) should be picked before any with C.
        assert "AAAAA" in df["window"].tolist()

    # Negative tests
    def test_invalid_df_seq(self):
        aaws = aa.AAWindowSampler()
        for invalid in [None, [], dict(), 1]:
            with pytest.raises((ValueError, AttributeError, TypeError)):
                aaws.sample_motif_matched(df_seq=invalid, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5, seed=0)

    def test_invalid_pos_col(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in [None, 1, [], {}]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col=invalid,
                                           n=5, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5, seed=0)

    def test_invalid_n(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in [0, -1, None, "5", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                           n=invalid, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5, seed=0)

    def test_invalid_window_size(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in [0, -1, None, "3", 3.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=invalid,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5, seed=0)

    def test_invalid_motif_pwm_missing(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_score_threshold=2.5, seed=0)

    def test_invalid_motif_pwm_shape(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        bad_pwm = np.zeros((4, 20))  # window_size is 3
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3, motif_pwm=bad_pwm,
                                       motif_score_threshold=2.5, seed=0)

    def test_invalid_motif_score_threshold_missing(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="motif_score_threshold"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3), seed=0)

    def test_invalid_role(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in [None, 1, []]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5,
                                           role=invalid, seed=0)

    def test_invalid_output_mode(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in ["seg", "Sequences", None, 1, ""]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5,
                                           output_mode=invalid, seed=0)

    def test_invalid_seed(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                           n=5, window_size=3,
                                           motif_pwm=_pwm_for_a(3),
                                           motif_score_threshold=2.5,
                                           seed=invalid)

    def test_invalid_no_eligible_candidates(self):
        # Every row has positives → no candidates → ValueError.
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAACDEFGHIKL", "AAACDEFGHIKL"],
            "pos": [[3], [3]],
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="No eligible"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5, seed=0)


class TestSampleMotifMatchedComplex:
    """Test sample_motif_matched() with combinations of parameters."""

    @settings(max_examples=10, deadline=2500)
    @given(n=some.integers(min_value=1, max_value=10),
           window_size=some.integers(min_value=2, max_value=6),
           seed=some.integers(min_value=0, max_value=10000))
    def test_valid_combination(self, n, window_size, seed):
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["A" * 20, "ACDEFGHIKLMNPQRSTVWY"],
            "pos": [[], []],
        })
        aaws = aa.AAWindowSampler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=n, window_size=window_size,
                                            motif_pwm=_pwm_for_a(window_size),
                                            motif_score_threshold=0.0,
                                            seed=seed)
        assert (df["window"].str.len() == window_size).all()
        assert (df["role"] == "Negative").all()

    def test_valid_aa_context_col_in_combined(self):
        # PWM scans for 'A' but only positions tagged 'M' are eligible.
        # Sequence has 5 M-tagged positions followed by 5 A's; the PWM-matching
        # 'AAA' windows fall outside the M region → expect no hits.
        seq = "MMMMMAAAAA"
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": [seq], "pos": [[]],
            "topo": ["MMMMMCCCCC"],
        })
        aaws = aa.AAWindowSampler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=3,
                                            motif_pwm=_pwm_for_a(3),
                                            motif_score_threshold=2.5,
                                            aa_context_col="topo",
                                            context_in="M", seed=0)
        # M region has no 'A' residues -> 0 motif-matched windows.
        assert len(df) == 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_aa_context_col_in_with_motif_match(self):
        # Same length, M and A both span the protein.
        seq = "AAACCCAAA"
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": [seq], "pos": [[]],
            "topo": ["MMMCCCMMM"],  # M aligns with both AAA stretches
        })
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=2.5,
                                        aa_context_col="topo",
                                        context_in="M", seed=0)
        assert len(df) >= 1
        assert (df["window"] == "AAA").all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_max_similarity_to_test_blocks_motif_hits(self):
        # AAA appears both in the positive row and in the candidate row.
        # max_similarity_to_test=0.0 should drop motif hits identical to the
        # known test window.
        df_seq = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAACDEFG", "AAACDEFG"],
            "pos": [[2], []],
        })
        aaws = aa.AAWindowSampler(max_similarity_to_test=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=3,
                                            motif_pwm=_pwm_for_a(3),
                                            motif_score_threshold=2.5,
                                            seed=0)
        assert (df["window"] != "AAA").all() or len(df) == 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_concat_with_other_methods(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        same = aaws.sample_same_protein(df_seq=df_seq, pos_col="pos",
                                         n_per_positive=1, window_size=3,
                                         seed=0)
        motif = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=2, window_size=3,
                                            motif_pwm=_pwm_for_a(3),
                                            motif_score_threshold=2.5, seed=0)
        merged = pd.concat([same, motif], ignore_index=True)
        # Merging the two outputs introduces NaN in `motif_score` for the
        # `same_protein` rows, but the union of columns must contain the
        # canonical schema.
        assert set(SCHEMA_SEGMENTS).issubset(set(merged.columns))

    def test_invalid_aa_context_col_missing(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5,
                                       aa_context_col="missing",
                                       context_in="X", seed=0)

    def test_invalid_aa_context_col_length_mismatch(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAACDEFGHIKL"], "pos": [[]],
            "topo": ["MM"],
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="length"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5,
                                       aa_context_col="topo",
                                       context_in="M", seed=0)

    def test_invalid_motif_pwm_shape_combo(self):
        # window_size=5 but PWM is 3 rows.
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="motif_pwm"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=5,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5, seed=0)

    def test_invalid_threshold_with_no_hits_returns_empty(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        # Impossibly high threshold → no hits → empty DataFrame, not an error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                            n=5, window_size=3,
                                            motif_pwm=_pwm_for_a(3),
                                            motif_score_threshold=999.0,
                                            seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_invalid_combination_no_candidates_with_context(self):
        # Every row is positive AND aa_context_col is set → should raise on
        # "no eligible candidates" before context check.
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAACDEFG"], "pos": [[3]],
            "topo": ["MMMCCCCC"],
        })
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="No eligible"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5,
                                       aa_context_col="topo",
                                       context_in="M", seed=0)

    def test_valid_p1_anchor_even_window_is_right_heavy(self):
        """For ``window_size=4``, the P1 residue must be at index ``half_left=1``
        of the returned window. Build a sequence where exactly one 4-mer scores
        the PWM maximum and assert the source_position points at P1.
        """
        # Single AAAA stretch at residues 8-11 (1-based). half_left=1, half_right=3,
        # so the P1 anchor of an AAAA window is at the 2nd 'A' = position 9.
        seq = "CDEFGHIAAAACDEFG"  # 16 residues; AAAA at 1-based 8..11
        df_seq = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "pos": [[]]})
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=1, window_size=4,
                                        motif_pwm=_pwm_for_a(4),
                                        motif_score_threshold=4.0, seed=0)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["window"] == "AAAA"
        # half_left=1 → for window covering 1-based 8..11, P1 is at index 1+8=9.
        assert row["source_position"] == 9
        # And the window slice matches seq[p - half_left - 1 : p - half_left - 1 + L].
        p = row["source_position"]
        assert seq[p - 2:p + 2] == row["window"]

    def test_valid_p1_anchor_odd_window_is_symmetric(self):
        """Odd ``window_size=3`` is unchanged: P1 sits at the geometric center."""
        seq = "CDEFGHIAAACDEFG"  # AAA at 1-based 8..10
        df_seq = pd.DataFrame({"entry": ["P1"], "sequence": [seq], "pos": [[]]})
        aaws = aa.AAWindowSampler()
        df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=1, window_size=3,
                                        motif_pwm=_pwm_for_a(3),
                                        motif_score_threshold=3.0, seed=0)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["window"] == "AAA"
        # half_left=1 → window covers 1-based 8..10, P1 at index 1+8=9 (the middle 'A').
        assert row["source_position"] == 9
        p = row["source_position"]
        assert seq[p - 2:p + 1] == row["window"]

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_motif_pwm_dataframe_input(self):
        """DataFrame PWM with named AA columns (any order) matches ndarray equivalent."""
        import aaanalysis.utils as ut
        cols_shuffled = list(ut.LIST_CANONICAL_AA)
        np.random.RandomState(0).shuffle(cols_shuffled)
        ndarray_pwm = _pwm_for_a(3)  # uses canonical alphabetical order
        df_pwm = pd.DataFrame(np.zeros((3, 20)), columns=cols_shuffled)
        df_pwm["A"] = 1.0
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        df_arr = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                        n=5, window_size=3,
                                        motif_pwm=ndarray_pwm,
                                        motif_score_threshold=2.5, seed=0)
        df_df = aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=df_pwm,
                                       motif_score_threshold=2.5, seed=0)
        assert df_arr["window"].tolist() == df_df["window"].tolist()

    def test_invalid_motif_pwm_dataframe_missing_canonical_aa(self):
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        cols = [c for c in "ACDEFGHIKLMNPQRSTVWY" if c != "Y"]
        df_pwm = pd.DataFrame(0.0, index=range(3), columns=cols)
        with pytest.raises(ValueError, match="missing"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3, motif_pwm=df_pwm,
                                       motif_score_threshold=1.0, seed=0)

    def test_invalid_context_in_without_aa_context_col(self):
        """context_in / context_out require aa_context_col (silent-drop guard)."""
        df_seq = _df_seq_with_aaa()
        aaws = aa.AAWindowSampler()
        with pytest.raises(ValueError, match="aa_context_col"):
            aaws.sample_motif_matched(df_seq=df_seq, pos_col="pos",
                                       n=5, window_size=3,
                                       motif_pwm=_pwm_for_a(3),
                                       motif_score_threshold=2.5,
                                       context_in=["A"], seed=0)
