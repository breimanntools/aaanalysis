"""This is a script to test aaanalysis.scan_motif()."""
import shutil
from unittest.mock import patch
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
SCHEMA_SEGMENTS_SCORED = SCHEMA_SEGMENTS + ["motif_score"]
SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]

# A FIMO subprocess call is slow relative to hypothesis' default deadline; the
# property tests below raise it per-test.
FIMO_DEADLINE = 5000

# Skip the FIMO-dependent classes if the binary isn't installed locally; the
# parity, property, and golden tests require it. The validation class below is
# binary-free (it mocks ``shutil.which``) and always runs.
fimo_required = pytest.mark.skipif(shutil.which("fimo") is None,
                                   reason="FIMO binary not on PATH")
MOCK_WHICH = ("aaanalysis.seq_analysis_pro._scan_motif.shutil.which",)


# I Helper Functions
def _pwm_for_a(window_size=3):
    """All-mass-on-Alanine PWM: an ``A``-run of length ``window_size`` scores
    ``window_size`` (one point per position)."""
    pwm = pd.DataFrame(0.0, index=range(window_size),
                       columns=list(ut.LIST_CANONICAL_AA))
    pwm["A"] = 1.0
    return pwm


def _df_seq_with_aaa():
    """P1 is a positive row (excluded); P2 is a candidate containing ``AAA``;
    P3 is a candidate with no ``AAA``."""
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [
            "ACDEFGHIKLAAA",      # P1: positive row
            "ACDEFGHIKLAAA",      # P2: candidate, has 'AAA'
            "ACDEFGHIKLMNPQR",    # P3: candidate, no 'AAA'
        ],
        "pos": [[5], [], []],
    })


def _df_seq_multi_hit():
    """``Pos`` is positive (excluded); ``C1`` carries three overlapping A-rich
    windows (``AAA``=3.0, ``DAA``=2.0, ``AAK``=2.0) and ``C2`` one (``AAD``=2.0)."""
    return pd.DataFrame({
        "entry": ["Pos", "C1", "C2"],
        "sequence": ["MMMMMMM", "ACDAAAKL", "AADKLMNP"],
        "pos": [[3], [], []],
    })


# II Test Classes
class TestFindMotifMatchedViaFimo:
    """Test scan_motif() validation paths (FIMO-binary-free).

    Every test mocks ``shutil.which`` so ``check_fimo_installed`` passes and the
    validation block (which runs before any subprocess call) is reached even
    where FIMO is absent. These run on the whole CI matrix, Windows included.
    """

    # --- pre-flight & required-arg guards ---
    def test_invalid_no_fimo_binary(self):
        with patch(*MOCK_WHICH, return_value=None):
            with pytest.raises(RuntimeError, match="fimo"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5)

    def test_invalid_motif_pwm_missing(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_score_threshold=2.5)

    def test_invalid_motif_score_threshold_missing(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="motif_score_threshold"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_pwm=_pwm_for_a(3))

    # --- df_seq / pos_col ---
    def test_invalid_df_seq_type(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, "not_a_df", 123, [1, 2, 3]]:
                with pytest.raises(ValueError):
                    aa.scan_motif(df_seq=bad, pos_col="pos", n=5, window_size=3,
                                  motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5)

    def test_invalid_df_seq_missing_sequence_column(self):
        df_bad = pd.DataFrame({"entry": ["P1"], "pos": [[]]})
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError):
                aa.scan_motif(df_seq=df_bad, pos_col="pos", n=5, window_size=3,
                              motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)

    def test_invalid_pos_col_type(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, 123, ["pos"], ""]:
                with pytest.raises(ValueError, match="pos_col"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col=bad, n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5)

    def test_invalid_pos_col_not_a_column(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="pos_col"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="missing", n=5,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5)

    # --- numeric / string params ---
    def test_invalid_n(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [0, -1, 1.5, "5", None]:
                with pytest.raises(ValueError, match=r"'n'"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=bad,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5)

    def test_invalid_window_size(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [0, -1, 2.5, "3", None]:
                with pytest.raises(ValueError, match="window_size"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=bad, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5)

    def test_invalid_role(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, 123, ["Negative"]]:
                with pytest.raises(ValueError, match="role"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5, role=bad)

    def test_invalid_output_mode(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in ["seg", "", "Segments", "frequencies", 1]:
                with pytest.raises(ValueError, match="output_mode"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5, output_mode=bad)

    # --- PWM shape / columns / threshold type ---
    def test_invalid_motif_pwm_shape(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            bad_pwm = pd.DataFrame(0.0, index=range(4),
                                   columns=list(ut.LIST_CANONICAL_AA))
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_pwm=bad_pwm,
                              motif_score_threshold=2.5)

    def test_invalid_motif_pwm_columns(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            # 20 columns but not the canonical amino acids (X replaces A).
            cols = ["X"] + list(ut.LIST_CANONICAL_AA)[1:]
            bad_pwm = pd.DataFrame(0.0, index=range(3), columns=cols)
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_pwm=bad_pwm,
                              motif_score_threshold=2.5)

    def test_invalid_motif_score_threshold_type(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in ["2.5", [2.5], {"t": 2.5}]:
                with pytest.raises(ValueError, match="motif_score_threshold"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=bad)

    # --- optional FIMO passthrough params ---
    def test_invalid_max_stored_scores(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for invalid in [0, -1, 1.5, "1000"]:
                with pytest.raises(ValueError, match="max_stored_scores"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5,
                                  max_stored_scores=invalid)

    def test_invalid_motif_pseudo(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for invalid in [-0.1, "0.5"]:
                with pytest.raises(ValueError, match="motif_pseudo"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=3, motif_pwm=_pwm_for_a(3),
                                  motif_score_threshold=2.5, motif_pseudo=invalid)

    def test_invalid_bg_file_missing(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="bg_file"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5,
                              bg_file="/nonexistent/path/to/bg.txt")

    # --- no eligible candidates (derived-invariant guard) ---
    def test_invalid_no_eligible_candidates(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAACDEFG"], "pos": [[3]],
        })
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="No eligible"):
                aa.scan_motif(df_seq=df_seq, pos_col="pos", n=5, window_size=3,
                              motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)

    # --- negative parameter combinations ---
    def test_invalid_window_size_pwm_mismatch(self):
        """window_size must equal the PWM's first dimension."""
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for ws, rows in [(3, 5), (5, 3), (9, 3)]:
                with pytest.raises(ValueError, match="motif_pwm"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=ws, motif_pwm=_pwm_for_a(rows),
                                  motif_score_threshold=2.5)

    def test_invalid_combo_bad_output_mode_valid_rest(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="output_mode"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5, role="Custom",
                              output_mode="not_a_mode")

    def test_invalid_combo_bad_n_valid_rest(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match=r"'n'"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=-5,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5,
                              output_mode="sequences")

    def test_invalid_combo_no_candidates_all_positive(self):
        df_all_pos = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLAAA", "AAACDEFGHIKL"],
            "pos": [[5], [3]],
        })
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="No eligible"):
                aa.scan_motif(df_seq=df_all_pos, pos_col="pos", n=5,
                              window_size=3, motif_pwm=_pwm_for_a(3),
                              motif_score_threshold=2.5)


@fimo_required
class TestFindMotifMatchedViaFimoIntegration:
    """End-to-end tests that require the FIMO binary (skipped if not installed)."""

    # --- smoke / schema ---
    def test_valid_smoke_returns_dataframe(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5)
        assert isinstance(df, pd.DataFrame)
        # Output extends the standard schema with `motif_score`.
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["motif_score"] >= 2.5).all()

    def test_valid_output_mode_segments_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, output_mode="segments")
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["strategy"] == "motif_matched").all()

    def test_valid_output_mode_sequences_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, output_mode="sequences")
        assert list(df.columns) == SCHEMA_SEQUENCES
        # One per-residue label list per input protein.
        assert len(df) == 3
        for seq, labels in zip(df["sequence"], df["labels"]):
            assert len(labels) == len(seq)

    def test_valid_smoke_excludes_positives(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5)
        assert "P1" not in set(df["entry"])

    def test_valid_role_passthrough(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, role="Unlabeled")
        assert (df["role"] == "Unlabeled").all()

    def test_valid_label_ref_passthrough(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, label_ref=7)
        assert (df["label"] == 7).all()

    # --- property tests ---
    @settings(max_examples=6, deadline=FIMO_DEADLINE)
    @given(n=some.integers(min_value=1, max_value=10))
    def test_valid_n_caps_output(self, n):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=n,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.0)
        assert len(df) <= n

    @settings(max_examples=4, deadline=FIMO_DEADLINE)
    @given(window_size=some.sampled_from([3, 5, 7]))
    def test_valid_window_size_schema(self, window_size):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=window_size,
                           motif_pwm=_pwm_for_a(window_size),
                           motif_score_threshold=0.0)
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["window"].str.len() == window_size).all()

    @settings(max_examples=6, deadline=FIMO_DEADLINE)
    @given(threshold=some.floats(min_value=0.0, max_value=3.0))
    def test_valid_scores_respect_threshold(self, threshold):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=threshold)
        assert (df["motif_score"] >= threshold).all()

    def test_valid_scores_sorted_descending(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.0)
        scores = df["motif_score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_valid_higher_threshold_is_subset(self):
        kw = dict(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                  window_size=3, motif_pwm=_pwm_for_a(3))
        loose = aa.scan_motif(motif_score_threshold=2.0, **kw)
        strict = aa.scan_motif(motif_score_threshold=3.0, **kw)
        loose_hits = set(zip(loose["entry"], loose["source_position"]))
        strict_hits = set(zip(strict["entry"], strict["source_position"]))
        assert strict_hits <= loose_hits

    def test_valid_empty_result_keeps_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=99.0)
        assert len(df) == 0
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED

    # --- reproducibility (scan_motif is deterministic; no seed) ---
    def test_valid_determinism(self):
        kw = dict(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.0)
        df_a = aa.scan_motif(**kw)
        df_b = aa.scan_motif(**kw)
        pd.testing.assert_frame_equal(df_a, df_b)

    # --- optional-param passthrough ---
    def test_valid_max_stored_scores_passthrough(self):
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_bumped = aa.scan_motif(max_stored_scores=1_000_000, **kw)
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_bumped["entry"], df_bumped["source_position"]))

    def test_valid_motif_pseudo_passthrough(self):
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_no_pseudo = aa.scan_motif(motif_pseudo=0.0, **kw)
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_no_pseudo["entry"], df_no_pseudo["source_position"]))

    def test_valid_bg_file_passthrough(self, tmp_path):
        """A uniform MEME background file must not change the wrapper's hits
        (we re-score in Python and run FIMO at --thresh 1.0)."""
        bg = tmp_path / "uniform_bg.txt"
        bg.write_text("\n".join(f"{aa_} 0.05" for aa_ in ut.LIST_CANONICAL_AA)
                      + "\n")
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_bg = aa.scan_motif(bg_file=str(bg), **kw)
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_bg["entry"], df_bg["source_position"]))

    # --- positive parameter combinations ---
    def test_valid_combo_sequences_label_propagation(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, output_mode="sequences",
                           label_test=1, label_ref=0)
        labels_p1 = df.loc[df["entry"] == "P1", "labels"].iloc[0]
        labels_p2 = df.loc[df["entry"] == "P2", "labels"].iloc[0]
        # P1 (positive) marked with label_test at its 1-based pos 5 -> index 4.
        assert labels_p1[4] == 1
        # P2 (matched candidate) marked with label_ref at the AAA center.
        assert 0 in labels_p2

    def test_valid_combo_custom_role_and_window(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           motif_score_threshold=0.0, role="Control")
        assert (df["role"] == "Control").all()
        assert (df["window"].str.len() == 5).all()

    @settings(max_examples=4, deadline=FIMO_DEADLINE)
    @given(output_mode=some.sampled_from(["segments", "sequences"]),
           n=some.integers(min_value=1, max_value=10))
    def test_valid_combo_mode_and_n(self, output_mode, n):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=n,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.0, output_mode=output_mode)
        expected = SCHEMA_SEGMENTS_SCORED if output_mode == "segments" \
            else SCHEMA_SEQUENCES
        assert list(df.columns) == expected

    def test_valid_combo_pseudo_zero_and_max_stored(self):
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_both = aa.scan_motif(motif_pseudo=0.0, max_stored_scores=1_000_000,
                                **kw)
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_both["entry"], df_both["source_position"]))

    # --- parity with the pure-Python core ---
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_strict_parity_with_python_core(self):
        """Strict parity: same df_seq + same PWM + same threshold => same set of
        ``(entry, source_position)`` AND same ``motif_score`` values. Both paths
        use the raw PWM-sum scoring; FIMO is only used as a position scanner."""
        df_seq = _df_seq_with_aaa()
        kw = dict(df_seq=df_seq, pos_col="pos", n=50, window_size=3,
                  motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5)
        py_df = aa.AAWindowSampler().sample_motif_matched(seed=0, **kw)
        cli_df = aa.scan_motif(**kw)
        py_hits = set(zip(py_df["entry"], py_df["source_position"]))
        cli_hits = set(zip(cli_df["entry"], cli_df["source_position"]))
        assert py_hits == cli_hits, (
            f"Hit set mismatch.\nPython: {py_hits}\nCLI: {cli_hits}")
        py_scores = {(e, p): s for e, p, s in
                     zip(py_df["entry"], py_df["source_position"],
                         py_df["motif_score"])}
        cli_scores = {(e, p): s for e, p, s in
                      zip(cli_df["entry"], cli_df["source_position"],
                          cli_df["motif_score"])}
        for key, py_score in py_scores.items():
            assert abs(py_score - cli_scores[key]) < 1e-9, (
                f"Score mismatch at {key}: py={py_score}, cli={cli_scores[key]}")


@fimo_required
class TestFindMotifMatchedViaFimoGoldenValues:
    """Hand-computed expected values for a known PWM/sequence (FIMO required).

    The PWM places all mass on Alanine, so a window's score is exactly its count
    of ``A`` residues (one point per position). All numbers below are derived by
    hand from that rule and the P1-anchor window convention.
    """

    def test_golden_single_hit_score_and_position(self):
        # P2 = 'ACDEFGHIKLAAA' (len 13); the only A-run of length 3 is at 1-based
        # positions 11-13. window_size=3 -> half_left=1 -> P1 anchor = position 12.
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["entry"] == "P2"
        assert row["window"] == "AAA"
        assert row["source_position"] == 12
        assert row["motif_score"] == 3.0

    def test_golden_entry_win_format(self):
        # entry_win = '<entry>_<start>-<end>', 1-based inclusive: AAA spans 11-13.
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5)
        assert df.iloc[0]["entry_win"] == "P2_11-13"

    def test_golden_threshold_is_inclusive(self):
        # A score exactly equal to the threshold (3.0) is kept (>=, not >).
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=3.0)
        assert len(df) == 1
        assert df.iloc[0]["window"] == "AAA"
        assert df.iloc[0]["motif_score"] == 3.0

    def test_golden_ranking_and_counts(self):
        # C1='ACDAAAKL': AAA(pos5,3.0), DAA(pos4,2.0), AAK(pos6,2.0);
        # C2='AADKLMNP': AAD(pos2,2.0). Pos excluded. Threshold 2.0 -> 4 hits,
        # AAA ranked first by descending score.
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=10,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.0)
        assert len(df) == 4
        assert df.iloc[0]["window"] == "AAA"
        assert df.iloc[0]["motif_score"] == 3.0
        assert set(df["motif_score"]) == {3.0, 2.0}
        assert "Pos" not in set(df["entry"])

    def test_golden_n_cap_keeps_top_score(self):
        # With n=1 only the single highest-scoring window survives.
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=1,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.0)
        assert len(df) == 1
        assert df.iloc[0]["motif_score"] == 3.0
        assert df.iloc[0]["window"] == "AAA"

    def test_golden_sequences_label_positions(self):
        # sequences mode: P1 positive at 1-based pos 5 (index 4) -> label_test=1;
        # P2 matched AAA center at 1-based pos 12 (index 11) -> label_ref=0.
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                           window_size=3, motif_pwm=_pwm_for_a(3),
                           motif_score_threshold=2.5, output_mode="sequences",
                           label_test=1, label_ref=0)
        labels_p1 = df.loc[df["entry"] == "P1", "labels"].iloc[0]
        labels_p2 = df.loc[df["entry"] == "P2", "labels"].iloc[0]
        assert labels_p1[4] == 1
        assert labels_p1.count(1) == 1
        assert labels_p2[11] == 0
        assert labels_p2.count(0) == 1
