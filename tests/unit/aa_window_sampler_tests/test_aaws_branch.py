"""This is a script to test branch-coverage edge arms of the AAWindowSampler class.

Every test drives a public ``AAWindowSampler`` method (never a private backend
function) to reach an otherwise-uncovered conditional arm in the frontend
validators or the sampling backends.
"""
import warnings
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

AA = list(ut.LIST_CANONICAL_AA)
SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]


# I Helper Functions
def _pwm_for_a(window_size=3):
    """PWM that scores 'A' at every position. Score of an all-'A' window = window_size."""
    pwm = pd.DataFrame(0.0, index=range(window_size), columns=AA)
    pwm["A"] = 1.0
    return pwm


def _df_pos_cand(seq="ACDEFGHIKLMNPQRSTVWY"):
    """One positive row (P1) and one candidate row (P2) sharing a sequence."""
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": [seq, seq],
        "pos": [[5], []],
    })


# II Test Classes
class TestMotifArgValidation:
    """Frontend ``check_motif_args`` inconsistent-pair arms."""

    def test_threshold_without_pwm_same_protein(self):
        # _aa_window_sampler.py L168: motif_score_threshold given, motif_pwm None.
        with pytest.raises(ValueError, match="motif_score_threshold.*without.*motif_pwm"):
            aa.AAWindowSampler().sample_same_protein(
                df_seq=_df_pos_cand(), pos_col="pos", n=2, window_size=3,
                motif_score_threshold=1.0, seed=0)

    def test_threshold_without_pwm_different_protein(self):
        with pytest.raises(ValueError, match="motif_score_threshold.*without.*motif_pwm"):
            aa.AAWindowSampler().sample_different_protein(
                df_seq=_df_pos_cand(), pos_col="pos", n=2, window_size=3,
                motif_score_threshold=1.0, seed=0)


class TestArmsValidation:
    """Frontend ``check_arms`` arm-name arm."""

    def test_non_string_arm_name(self):
        # _aa_window_sampler.py L210: arms key is not a string.
        arms = {1: {"method": "synthetic", "n": 2, "window_size": 3}}
        with pytest.raises(ValueError, match="should be a string arm name"):
            aa.AAWindowSampler().sample_benchmark_set(df_seq=_df_pos_cand(), arms=arms)


class TestBenchmarkSetSeedArm:
    """``sample_benchmark_set`` master-seed-None arm."""

    def test_no_seed_no_random_state(self):
        # _aa_window_sampler.py L1120: master is None -> sub_seeds = [None] * n.
        aaws = aa.AAWindowSampler(random_state=None)
        arms = {"synth": {"method": "synthetic", "n": 3, "window_size": 4,
                          "generator": "uniform"}}
        df = aaws.sample_benchmark_set(df_seq=_df_pos_cand(), arms=arms, seed=None)
        assert "arm" in df.columns
        assert (df["arm"] == "synth").all()


class TestContextFilterArms:
    """Per-residue context filter arms in ``_filter_aa_context``."""

    def test_single_bound_context_in_list(self):
        # _aa_window_sampler.py L271: _to_set(list) coerces a list-like whitelist
        # to a set (context_out stays None -> the unset-bound arm at L269).
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKLMNP", "ACDEFGHIKLMNP"],
            "pos": [[5], []],
            "ctx": [list("HHHHHHHHHHHHH"), list("HHHHHHHHHHHHH")],
        })
        out = aa.AAWindowSampler().sample_different_protein(
            df_seq=df, n=2, window_size=3, pos_col="pos",
            aa_context_col="ctx", context_in=["H", "L"], seed=0)
        assert len(out) >= 1

    def test_nan_context_row_skipped(self):
        # _aa_window_sampler.py L280-281: a row whose context cell is NaN yields
        # no eligible residues and is skipped (other rows still supply windows).
        df = pd.DataFrame({
            "entry": ["P1", "P2", "P3"],
            "sequence": ["ACDEFGHIKLMNP", "ACDEFGHIKLMNP", "ACDEFGHIKLMNP"],
            "pos": [[5], [], []],
            "ctx": [list("HHHHHHHHHHHHH"), np.nan, list("HHHHHHHHHHHHH")],
        })
        out = aa.AAWindowSampler().sample_different_protein(
            df_seq=df, n=2, window_size=3, pos_col="pos",
            aa_context_col="ctx", context_in="H", seed=0)
        # Only P3 (NaN context on P2) supplies windows.
        assert (out["entry"] == "P3").all()


class TestPosColParsing:
    """``_parse_pos_value`` int-cast and 1-based guards."""

    def test_non_castable_in_list(self):
        # _utils.py L26-33: list element not int-castable.
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFGHIKL"], "pos": [["x"]]})
        with pytest.raises(ValueError, match="int-castable positions"):
            aa.AAWindowSampler().sample_same_protein(
                df_seq=df, pos_col="pos", n=2, window_size=3, seed=0)

    def test_non_castable_scalar(self):
        # _utils.py L29-34: scalar cell not int-castable.
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFGHIKL"], "pos": ["x"]})
        with pytest.raises(ValueError, match="int or iterable of"):
            aa.AAWindowSampler().sample_same_protein(
                df_seq=df, pos_col="pos", n=2, window_size=3, seed=0)

    def test_zero_position_rejected(self):
        # _utils.py L35-37: a position < 1 violates 1-based indexing.
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFGHIKL"], "pos": [[0]]})
        with pytest.raises(ValueError, match="1-based indexing"):
            aa.AAWindowSampler().sample_same_protein(
                df_seq=df, pos_col="pos", n=2, window_size=3, seed=0)


class TestPwmNonCanonicalScoring:
    """``score_window_pwm_`` non-canonical-residue arm (motif filter path)."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_non_canonical_residue_scores_zero(self):
        # _utils.py L198->196: a candidate window with a non-canonical residue ('X')
        # contributes zero at that position when motif-filtering in
        # sample_different_protein.
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAACDEFGHI", "AAXAAACDEF"],
            "pos": [[2], []],
        })
        out = aa.AAWindowSampler().sample_different_protein(
            df_seq=df, n=5, window_size=3, pos_col="pos",
            motif_pwm=_pwm_for_a(3), motif_score_threshold=2.5,
            motif_match="in", seed=0)
        # 'X' contributes zero, so any window touching it scores < 3 and cannot
        # clear the all-three-'A' threshold of 2.5 -> no surviving window has 'X'.
        assert all("X" not in w for w in out["window"])


class TestSameProteinNoWindows:
    """``sample_same_protein`` no-valid-window arms."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_protein_too_short_returns_empty(self):
        # sample_same_protein.py L113-120 (no centers, warning) + L125 (no prot_data).
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEF"], "pos": [[3]]})
        out = aa.AAWindowSampler(verbose=False).sample_same_protein(
            df_seq=df, n=5, window_size=9, pos_col="pos", seed=0)
        assert len(out) == 0

    def test_no_window_warns(self):
        # sample_same_protein.py L114-119: RuntimeWarning when a protein supplies no
        # valid centers (verbose=True).
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEF"], "pos": [[3]]})
        with pytest.warns(RuntimeWarning, match="No valid windows for entry"):
            aa.AAWindowSampler(verbose=True).sample_same_protein(
                df_seq=df, n=5, window_size=9, pos_col="pos", seed=0)


class TestSameProteinRoundRobin:
    """``sample_same_protein`` pass-2 round-robin top-up arm."""

    @pytest.mark.parametrize("n", [5, 9, 13])
    def test_round_robin_reaches_target(self, n):
        # sample_same_protein.py L166: the pass-2 break (``total_accepted >= n``) is
        # taken mid-loop. One big supplier (slot A) plus several single-window tiny
        # proteins forces a multi-round round-robin top-up that crosses n part-way
        # through a slot loop rather than exactly at its end.
        big = "ACDEFGHIKLMNPQRSTVWY" * 4
        tiny = "ACDEFGHIK"  # len 9, window 9 -> exactly 1 candidate center
        df = pd.DataFrame({
            "entry": ["A", "B", "C", "D", "E"],
            "sequence": [big, tiny, tiny, tiny, tiny],
            "pos": [[40], [5], [5], [5], [5]],
        })
        out = aa.AAWindowSampler(verbose=False).sample_same_protein(
            df_seq=df, n=n, window_size=9, pos_col="pos", seed=2)
        assert len(out) == n


class TestDifferentProteinCandidateWarning:
    """``sample_different_protein`` ineligible-candidate warning arm."""

    def test_missing_candidate_protein_warns(self):
        # sample_different_protein.py L62-67: UserWarning for ineligible candidates.
        df = _df_pos_cand("ACDEFGHIKLMNP")
        with pytest.warns(UserWarning, match="candidate_proteins.*were not eligible"):
            aa.AAWindowSampler(verbose=True).sample_different_protein(
                df_seq=df, n=3, window_size=5, pos_col="pos",
                candidate_proteins=["P2", "NOPE"], seed=0)


class TestMotifMatchedEdgeArms:
    """``sample_motif_matched`` short-sequence / non-canonical / empty-pool arms."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_candidate_shorter_than_window(self):
        # sample_motif_matched.py L27: a candidate sequence shorter than window_size
        # is skipped by _scan_protein_.
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAAAAACDEF", "AAA"],
            "pos": [[3], []],
        })
        out = aa.AAWindowSampler(verbose=False).sample_motif_matched(
            df_seq=df, n=5, window_size=5, motif_pwm=_pwm_for_a(5),
            motif_score_threshold=2.5, pos_col="pos", seed=0)
        assert isinstance(out, pd.DataFrame)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_non_canonical_in_scan(self):
        # sample_motif_matched.py L39->37: non-canonical residue contributes zero
        # during the PWM scan.
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["AAACDE", "AAXAAA"],
            "pos": [[2], []],
        })
        out = aa.AAWindowSampler(verbose=False).sample_motif_matched(
            df_seq=df, n=5, window_size=3, motif_pwm=_pwm_for_a(3),
            motif_score_threshold=2.5, pos_col="pos", seed=0)
        # The scan visits the 'X' position (scoring it zero); windows touching 'X'
        # cannot reach the 2.5 threshold, so none survive.
        assert all("X" not in w for w in out["window"])

    def test_empty_pool_warns(self):
        # sample_motif_matched.py L114-118: RuntimeWarning when no candidate window
        # meets the threshold (verbose=True).
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["CDEFGH", "CDEFGH"],
            "pos": [[3], []],
        })
        with pytest.warns(RuntimeWarning, match="No candidate windows met the motif"):
            aa.AAWindowSampler(verbose=True).sample_motif_matched(
                df_seq=df, n=5, window_size=3, motif_pwm=_pwm_for_a(3),
                motif_score_threshold=2.5, pos_col="pos", seed=0)


class TestSyntheticEdgeArms:
    """``sample_synthetic`` empty-frequency and shortfall-warning arms."""

    def test_global_freq_no_canonical_aa(self):
        # sample_synthetic.py L73-74 (+ L70->68 lookup miss): global_freq over
        # sequences with no canonical amino acids.
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["1111111111"], "pos": [[]]})
        with pytest.raises(ValueError, match="No canonical amino acids"):
            aa.AAWindowSampler().sample_synthetic(
                df_seq=df, n=3, window_size=4, generator="global_freq", seed=0)

    def test_position_specific_no_canonical_aa(self):
        # sample_synthetic.py L159-160 (+ L156->154 lookup miss): position_specific
        # over a test window with no canonical amino acids at a column.
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["11111111111111"],
                           "pos": [[6]]})
        with pytest.raises(ValueError, match="no.*canonical amino acids"):
            aa.AAWindowSampler().sample_synthetic(
                df_seq=df, n=3, window_size=4, generator="position_specific",
                pos_col="pos", seed=0)

    def test_synthetic_shortfall_warns(self):
        # sample_synthetic.py L254-255: RuntimeWarning when filtering keeps fewer
        # than n synthetic windows. A custom_filter that rejects everything forces
        # the shortfall with verbose=True.
        df = _df_pos_cand("ACDEFGHIKLMNP")
        aaws = aa.AAWindowSampler(verbose=True, custom_filter=lambda w, e, p: False)
        with pytest.warns(RuntimeWarning, match="synthetic windows kept after"):
            aaws.sample_synthetic(df_seq=df, n=5, window_size=4,
                                  generator="uniform", seed=0)


class TestBuildOutputSequencesArms:
    """``build_sequences_output`` out-of-range-test-position arm."""

    def test_test_position_beyond_sequence(self):
        # build_output.py L68->67: a test position beyond the sequence length is
        # not written into the per-residue labels list (sequences mode).
        df = pd.DataFrame({
            "entry": ["P1", "P2"],
            "sequence": ["ACDEFGHIKL", "ACDEFGHIKL"],
            "pos": [[99], []],
        })
        out = aa.AAWindowSampler().sample_different_protein(
            df_seq=df, n=2, window_size=3, pos_col="pos",
            output_mode="sequences", seed=0)
        # P1's out-of-range position 99 leaves its labels free of label_test (1).
        p1_labels = out.loc[out["entry"] == "P1", "labels"].iloc[0]
        assert 1 not in p1_labels
        assert len(p1_labels) == 10
