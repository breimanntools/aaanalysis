"""This is a script to test aaanalysis.scan_motif()."""
import shutil
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

AA_IDX = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}
SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]

# Skip the entire module if FIMO isn't installed locally; the parity test and
# smoke tests require the binary.
fimo_required = pytest.mark.skipif(shutil.which("fimo") is None,
                                    reason="FIMO binary not on PATH")


# I Helper Functions
def _pwm_for_a(window_size=3):
    pwm = np.zeros((window_size, len(ut.LIST_CANONICAL_AA)))
    pwm[:, AA_IDX["A"]] = 1.0
    return pwm


def _df_seq_with_aaa():
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [
            "ACDEFGHIKLAAA",      # P1: positive row
            "ACDEFGHIKLAAA",      # P2: candidate, has 'AAA'
            "ACDEFGHIKLMNPQR",    # P3: candidate, no 'AAA'
        ],
        "pos": [[5], [], []],
    })


# II Test Classes
class TestFindMotifMatchedViaFimo:
    """Test scan_motif() validation paths (FIMO-binary-free)."""

    # Negative tests — these don't need the FIMO binary because validation
    # happens first OR check_fimo_installed raises before validation.
    def test_invalid_no_fimo_binary(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="fimo"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                                 pos_col="pos", n=5,
                                                 window_size=3,
                                                 motif_pwm=_pwm_for_a(3),
                                                 motif_score_threshold=2.5)

    def test_invalid_motif_pwm_missing(self):
        # check_fimo_installed runs first; mock it to bypass.
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                                 pos_col="pos", n=5,
                                                 window_size=3,
                                                 motif_score_threshold=2.5)

    def test_invalid_motif_score_threshold_missing(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="motif_score_threshold"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                                 pos_col="pos", n=5,
                                                 window_size=3,
                                                 motif_pwm=_pwm_for_a(3))

    def test_invalid_motif_pwm_shape(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            bad_pwm = np.zeros((4, 20))
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                                 pos_col="pos", n=5,
                                                 window_size=3,
                                                 motif_pwm=bad_pwm,
                                                 motif_score_threshold=2.5)

    def test_invalid_no_eligible_candidates(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAACDEFG"], "pos": [[3]],
        })
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="No eligible"):
                aa.scan_motif(df_seq=df_seq, pos_col="pos",
                                                 n=5, window_size=3,
                                                 motif_pwm=_pwm_for_a(3),
                                                 motif_score_threshold=2.5)

    def test_invalid_max_stored_scores(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            for invalid in [0, -1, 1.5, "1000"]:
                with pytest.raises(ValueError, match="max_stored_scores"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos",
                                   n=5, window_size=3,
                                   motif_pwm=_pwm_for_a(3),
                                   motif_score_threshold=2.5,
                                   max_stored_scores=invalid)

    def test_invalid_motif_pseudo(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            for invalid in [-0.1, "0.5"]:
                with pytest.raises(ValueError, match="motif_pseudo"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos",
                                   n=5, window_size=3,
                                   motif_pwm=_pwm_for_a(3),
                                   motif_score_threshold=2.5,
                                   motif_pseudo=invalid)

    def test_invalid_bg_file_missing(self):
        with patch("aaanalysis.seq_analysis_pro._scan_motif"
                   ".shutil.which", return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="bg_file"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos",
                               n=5, window_size=3,
                               motif_pwm=_pwm_for_a(3),
                               motif_score_threshold=2.5,
                               bg_file="/nonexistent/path/to/bg.txt")


@fimo_required
class TestFindMotifMatchedViaFimoIntegration:
    """End-to-end tests that require the FIMO binary (skipped if not installed)."""

    def test_valid_smoke_returns_dataframe(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                              pos_col="pos", n=5, window_size=3,
                                              motif_pwm=_pwm_for_a(3),
                                              motif_score_threshold=2.5)
        assert isinstance(df, pd.DataFrame)
        # Output extends the standard schema with `motif_score` (matches sample_motif_matched).
        assert list(df.columns) == SCHEMA_SEGMENTS + ["motif_score"]
        assert (df["motif_score"] >= 2.5).all()

    def test_valid_smoke_excludes_positives(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(),
                                              pos_col="pos", n=10, window_size=3,
                                              motif_pwm=_pwm_for_a(3),
                                              motif_score_threshold=2.5)
        assert "P1" not in set(df["entry"])

    def test_valid_max_stored_scores_passthrough(self):
        # Passing a high max_stored_scores must not change the returned hits
        # for a small candidate set (FIMO's default 100 000 is already plenty).
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                  window_size=3, motif_pwm=_pwm_for_a(3),
                  motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_bumped = aa.scan_motif(max_stored_scores=1_000_000, **kw)
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_bumped["entry"], df_bumped["source_position"]))

    def test_valid_motif_pseudo_passthrough(self):
        # motif_pseudo affects FIMO's internal motif but not the wrapper's
        # parity contract (we re-score in Python and use --thresh 1.0).
        kw = dict(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                  window_size=3, motif_pwm=_pwm_for_a(3),
                  motif_score_threshold=2.5)
        df_default = aa.scan_motif(**kw)
        df_no_pseudo = aa.scan_motif(motif_pseudo=0.0, **kw)
        # Same hit set under both pseudocount values.
        assert set(zip(df_default["entry"], df_default["source_position"])) == \
               set(zip(df_no_pseudo["entry"], df_no_pseudo["source_position"]))

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_strict_parity_with_python_core(self):
        """Strict parity: same df_seq + same PWM + same threshold ⇒ same set of
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
        # Score parity: same (entry, source_position) keys map to identical scores.
        py_scores = {(e, p): s for e, p, s in
                     zip(py_df["entry"], py_df["source_position"],
                         py_df["motif_score"])}
        cli_scores = {(e, p): s for e, p, s in
                      zip(cli_df["entry"], cli_df["source_position"],
                          cli_df["motif_score"])}
        for key, py_score in py_scores.items():
            assert abs(py_score - cli_scores[key]) < 1e-9, (
                f"Score mismatch at {key}: py={py_score}, cli={cli_scores[key]}")
