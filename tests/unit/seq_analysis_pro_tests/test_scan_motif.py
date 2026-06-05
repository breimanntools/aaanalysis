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
SCHEMA_SEGMENTS_SCORED = SCHEMA_SEGMENTS + ["motif_score", "p_value"]
SCHEMA_SEQUENCES = ["entry", "sequence", "labels"]

# A FIMO subprocess call is slow relative to hypothesis' default deadline; the
# property tests below raise it per-test.
FIMO_DEADLINE = 5000

# Skip the FIMO-dependent classes if the binary isn't installed locally. The
# validation class is binary-free (it mocks ``shutil.which``) and always runs.
fimo_required = pytest.mark.skipif(shutil.which("fimo") is None,
                                   reason="FIMO binary not on PATH")
MOCK_WHICH = ("aaanalysis.seq_analysis_pro._scan_motif.shutil.which",)


# I Helper Functions
def _pwm_for_a(window_size=5):
    """Alanine-dominant PWM (A=0.81, the other 19 AAs 0.01 each, rows sum to 1).
    An A-run scores highest and is most significant under FIMO."""
    pwm = pd.DataFrame(0.01, index=range(window_size),
                       columns=list(ut.LIST_CANONICAL_AA))
    pwm["A"] = 0.81
    return pwm


def _df_seq_with_aaa():
    """P1 is a positive row (excluded); P2 is a candidate containing an A-run;
    P3 is a candidate with no A-run."""
    return pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [
            "ACDEFGHIKLAAAAA",     # P1: positive row
            "ACDEFGHIKLAAAAA",     # P2: candidate, has 'AAAAA'
            "ACDEFGHIKLMNPQR",     # P3: candidate, no A-run
        ],
        "pos": [[5], [], []],
    })


def _df_seq_multi_hit():
    """Pos is positive (excluded); C1/C2 carry several A-rich windows of
    differing significance."""
    return pd.DataFrame({
        "entry": ["Pos", "C1", "C2"],
        "sequence": ["MMMMMMMMMM", "ACDAAAAAKLMN", "AADEFAAAAAGH"],
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
                              window_size=5, motif_pwm=_pwm_for_a(5))

    def test_invalid_motif_pwm_missing(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=5)

    # --- df_seq / pos_col ---
    def test_invalid_df_seq_type(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, "not_a_df", 123, [1, 2, 3]]:
                with pytest.raises(ValueError):
                    aa.scan_motif(df_seq=bad, pos_col="pos", n=5, window_size=5,
                                  motif_pwm=_pwm_for_a(5))

    def test_invalid_df_seq_missing_sequence_column(self):
        df_bad = pd.DataFrame({"entry": ["P1"], "pos": [[]]})
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError):
                aa.scan_motif(df_seq=df_bad, pos_col="pos", n=5, window_size=5,
                              motif_pwm=_pwm_for_a(5))

    def test_invalid_pos_col_type(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, 123, ["pos"], ""]:
                with pytest.raises(ValueError, match="pos_col"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col=bad, n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5))

    def test_invalid_pos_col_not_a_column(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="pos_col"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="missing", n=5,
                              window_size=5, motif_pwm=_pwm_for_a(5))

    # --- numeric / string params ---
    def test_invalid_n(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [0, -1, 1.5, "5", None]:
                with pytest.raises(ValueError, match=r"'n'"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=bad,
                                  window_size=5, motif_pwm=_pwm_for_a(5))

    def test_invalid_window_size(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [0, -1, 2.5, "3", None]:
                with pytest.raises(ValueError, match="window_size"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=bad, motif_pwm=_pwm_for_a(5))

    def test_invalid_role(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [None, 123, ["Negative"]]:
                with pytest.raises(ValueError, match="role"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5), role=bad)

    def test_invalid_output_mode(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in ["seg", "", "Segments", "frequencies", 1]:
                with pytest.raises(ValueError, match="output_mode"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5),
                                  output_mode=bad)

    # --- PWM shape / columns / p-value threshold ---
    def test_invalid_motif_pwm_shape(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            bad_pwm = pd.DataFrame(0.05, index=range(4),
                                   columns=list(ut.LIST_CANONICAL_AA))
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=5, motif_pwm=bad_pwm)

    def test_invalid_motif_pwm_columns(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            cols = ["X"] + list(ut.LIST_CANONICAL_AA)[1:]
            bad_pwm = pd.DataFrame(0.05, index=range(5), columns=cols)
            with pytest.raises(ValueError, match="motif_pwm"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=5, motif_pwm=bad_pwm)

    def test_invalid_pvalue_threshold(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for bad in [-0.1, 1.5, "0.01", None]:
                with pytest.raises(ValueError, match="pvalue_threshold"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5),
                                  pvalue_threshold=bad)

    # --- optional FIMO passthrough params ---
    def test_invalid_max_stored_scores(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for invalid in [0, -1, 1.5, "1000"]:
                with pytest.raises(ValueError, match="max_stored_scores"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5),
                                  max_stored_scores=invalid)

    def test_invalid_motif_pseudo(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for invalid in [-0.1, "0.5"]:
                with pytest.raises(ValueError, match="motif_pseudo"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=5, motif_pwm=_pwm_for_a(5),
                                  motif_pseudo=invalid)

    def test_invalid_bg_file_missing(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="bg_file"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                              window_size=5, motif_pwm=_pwm_for_a(5),
                              bg_file="/nonexistent/path/to/bg.txt")

    # --- no eligible candidates (derived-invariant guard) ---
    def test_invalid_no_eligible_candidates(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAAAACDEFG"], "pos": [[3]],
        })
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="No eligible"):
                aa.scan_motif(df_seq=df_seq, pos_col="pos", n=5, window_size=5,
                              motif_pwm=_pwm_for_a(5))

    # --- negative parameter combinations ---
    def test_invalid_window_size_pwm_mismatch(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            for ws, rows in [(5, 7), (7, 5), (9, 5)]:
                with pytest.raises(ValueError, match="motif_pwm"):
                    aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=5,
                                  window_size=ws, motif_pwm=_pwm_for_a(rows))

    def test_invalid_combo_bad_output_mode_valid_rest(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="output_mode"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                              window_size=5, motif_pwm=_pwm_for_a(5),
                              role="Custom", output_mode="not_a_mode")

    def test_invalid_combo_bad_n_valid_rest(self):
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match=r"'n'"):
                aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=-5,
                              window_size=5, motif_pwm=_pwm_for_a(5),
                              output_mode="sequences")

    def test_invalid_combo_no_candidates_all_positive(self):
        df_all_pos = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLAAAAA", "AAAAACDEFGHIKL"],
            "pos": [[5], [3]],
        })
        with patch(*MOCK_WHICH, return_value="/usr/local/bin/fimo"):
            with pytest.raises(ValueError, match="No eligible"):
                aa.scan_motif(df_seq=df_all_pos, pos_col="pos", n=5,
                              window_size=5, motif_pwm=_pwm_for_a(5))


@fimo_required
class TestFindMotifMatchedViaFimoIntegration:
    """End-to-end tests that require the FIMO binary (skipped if not installed)."""

    # --- smoke / schema ---
    def test_valid_smoke_returns_dataframe(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["p_value"] <= 1e-2).all()

    def test_valid_output_mode_segments_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, output_mode="segments")
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["strategy"] == "motif_matched").all()

    def test_valid_output_mode_sequences_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, output_mode="sequences")
        assert list(df.columns) == SCHEMA_SEQUENCES
        assert len(df) == 3
        for seq, labels in zip(df["sequence"], df["labels"]):
            assert len(labels) == len(seq)

    def test_valid_smoke_excludes_positives(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2)
        assert "P1" not in set(df["entry"])

    def test_valid_role_passthrough(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, role="Unlabeled")
        assert (df["role"] == "Unlabeled").all()

    def test_valid_label_ref_passthrough(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, label_ref=7)
        assert (df["label"] == 7).all()

    # --- property tests ---
    def test_valid_pvalues_below_threshold(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2)
        assert (df["p_value"] <= 1e-2).all()

    def test_valid_pvalues_sorted_ascending(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=5e-2)
        pvals = df["p_value"].to_list()
        assert pvals == sorted(pvals)

    def test_valid_stricter_threshold_is_subset(self):
        kw = dict(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50,
                  window_size=5, motif_pwm=_pwm_for_a(5))
        loose = aa.scan_motif(pvalue_threshold=5e-2, **kw)
        strict = aa.scan_motif(pvalue_threshold=1e-3, **kw)
        loose_hits = set(zip(loose["entry"], loose["source_position"]))
        strict_hits = set(zip(strict["entry"], strict["source_position"]))
        assert strict_hits <= loose_hits

    @settings(max_examples=6, deadline=FIMO_DEADLINE)
    @given(n=some.integers(min_value=1, max_value=10))
    def test_valid_n_caps_output(self, n):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=n,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=5e-2)
        assert len(df) <= n

    @settings(max_examples=3, deadline=FIMO_DEADLINE)
    @given(window_size=some.sampled_from([3, 5, 7]))
    def test_valid_window_size_schema(self, window_size):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=window_size,
                           motif_pwm=_pwm_for_a(window_size),
                           pvalue_threshold=5e-2)
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["window"].str.len() == window_size).all()

    def test_valid_empty_result_keeps_schema(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=10,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-30)
        assert len(df) == 0
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED

    # --- reproducibility (scan_motif is deterministic; no seed) ---
    def test_valid_determinism(self):
        kw = dict(df_seq=_df_seq_multi_hit(), pos_col="pos", n=50, window_size=5,
                  motif_pwm=_pwm_for_a(5), pvalue_threshold=5e-2)
        df_a = aa.scan_motif(**kw)
        df_b = aa.scan_motif(**kw)
        pd.testing.assert_frame_equal(df_a, df_b)

    # --- distinct from the pure-Python PWM-sum sampler (no parity) ---
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_valid_diverges_from_python_engine(self):
        """scan_motif selects by FIMO p-value and scores with FIMO's log-odds;
        AAWindowSampler.sample_motif_matched selects by raw PWM-sum. They are
        complementary, not identical: a window selected by both carries
        different ``motif_score`` values, and only scan_motif reports p-values."""
        pwm = _pwm_for_a(5)
        df_seq = _df_seq_with_aaa()
        cli = aa.scan_motif(df_seq=df_seq, pos_col="pos", n=50, window_size=5,
                            motif_pwm=pwm, pvalue_threshold=1e-2)
        py = aa.AAWindowSampler().sample_motif_matched(
            df_seq=df_seq, pos_col="pos", n=50, window_size=5,
            motif_pwm=pwm, motif_score_threshold=2.0, seed=0)
        assert "p_value" in cli.columns
        assert "p_value" not in py.columns
        py_scores = {(e, p): s for e, p, s in
                     zip(py["entry"], py["source_position"], py["motif_score"])}
        shared = [(e, p, s) for e, p, s in
                  zip(cli["entry"], cli["source_position"], cli["motif_score"])
                  if (e, p) in py_scores]
        assert shared, "expected at least one window selected by both engines"
        assert any(abs(s - py_scores[(e, p)]) > 1e-6 for e, p, s in shared)

    # --- optional-param passthrough ---
    def test_valid_max_stored_scores_accepted(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, max_stored_scores=1_000_000)
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["p_value"] <= 1e-2).all()

    def test_valid_motif_pseudo_accepted(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, motif_pseudo=0.0)
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED

    def test_valid_bg_file_accepted(self, tmp_path):
        """A uniform MEME background file is accepted and yields valid hits."""
        bg = tmp_path / "uniform_bg.txt"
        bg.write_text("\n".join(f"{aa_} 0.05" for aa_ in ut.LIST_CANONICAL_AA)
                      + "\n")
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, bg_file=str(bg))
        assert list(df.columns) == SCHEMA_SEGMENTS_SCORED
        assert (df["p_value"] <= 1e-2).all()

    # --- positive parameter combinations ---
    def test_valid_combo_custom_role_and_window(self):
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=7, motif_pwm=_pwm_for_a(7),
                           pvalue_threshold=5e-2, role="Control")
        assert (df["role"] == "Control").all()
        assert (df["window"].str.len() == 7).all()

    @settings(max_examples=4, deadline=FIMO_DEADLINE)
    @given(output_mode=some.sampled_from(["segments", "sequences"]),
           n=some.integers(min_value=1, max_value=10))
    def test_valid_combo_mode_and_n(self, output_mode, n):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=n,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=5e-2, output_mode=output_mode)
        expected = SCHEMA_SEGMENTS_SCORED if output_mode == "segments" \
            else SCHEMA_SEQUENCES
        assert list(df.columns) == expected


@fimo_required
class TestFindMotifMatchedViaFimoGoldenValues:
    """Hand-reasoned structural/relational expectations for a known PWM/sequence.

    FIMO's exact score/p-value are environment-sensitive, so these assert
    relationships (positions, ordering, schema, label placement) rather than
    frozen FIMO floats.
    """

    def test_golden_excludes_positive_rows(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=5e-2)
        assert "Pos" not in set(df["entry"])

    def test_golden_window_matches_source_position(self):
        # The reported window is exactly seq[start-1 : start-1+window_size] where
        # start = source_position - (window_size-1)//2 (P1-anchor convention).
        ws = 5
        half_left = (ws - 1) // 2
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=ws, motif_pwm=_pwm_for_a(ws),
                           pvalue_threshold=5e-2)
        seq_by_entry = dict(zip(_df_seq_with_aaa()["entry"],
                                _df_seq_with_aaa()["sequence"]))
        for _, row in df.iterrows():
            start0 = (row["source_position"] - 1) - half_left
            assert seq_by_entry[row["entry"]][start0:start0 + ws] == row["window"]

    def test_golden_entry_win_format(self):
        ws = 5
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=ws, motif_pwm=_pwm_for_a(ws),
                           pvalue_threshold=5e-2)
        for _, row in df.iterrows():
            start = row["source_position"] - (ws - 1) // 2
            end = start + ws - 1
            assert row["entry_win"] == f"{row['entry']}_{start}-{end}"

    def test_golden_first_row_is_most_significant(self):
        df = aa.scan_motif(df_seq=_df_seq_multi_hit(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=5e-2)
        assert df.iloc[0]["p_value"] == df["p_value"].min()

    def test_golden_motif_score_is_fimo_not_pwm_sum(self):
        # FIMO's log-odds score is not the raw per-position PWM-sum
        # (max PWM-sum for a width-5 A-run here is 5*0.81 = 4.05); FIMO scores
        # exceed that, proving the score is FIMO's, not a re-scored PWM-sum.
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2)
        assert (df["motif_score"] > 4.05).any()

    def test_golden_sequences_label_positions(self):
        # sequences mode: P1 positive at 1-based pos 5 (index 4) -> label_test=1;
        # P2 (matched A-run) carries label_ref=0 at the matched window center.
        df = aa.scan_motif(df_seq=_df_seq_with_aaa(), pos_col="pos", n=20,
                           window_size=5, motif_pwm=_pwm_for_a(5),
                           pvalue_threshold=1e-2, output_mode="sequences",
                           label_test=1, label_ref=0)
        labels_p1 = df.loc[df["entry"] == "P1", "labels"].iloc[0]
        labels_p2 = df.loc[df["entry"] == "P2", "labels"].iloc[0]
        assert labels_p1[4] == 1
        assert labels_p1.count(1) == 1
        assert 0 in labels_p2
