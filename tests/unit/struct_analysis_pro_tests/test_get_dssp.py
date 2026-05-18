"""This is a script to test aaanalysis.get_dssp()."""
import shutil
import warnings
from pathlib import Path
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

MODULE = "aaanalysis.struct_analysis_pro._get_dssp"
RUNNER = f"{MODULE}.run_dssp_for_entry_"

mkdssp_required = pytest.mark.skipif(
    shutil.which("mkdssp") is None and shutil.which("dssp") is None,
    reason="mkdssp/dssp binary not on PATH")


# I Helper Functions
def _mock_binary_present():
    """Patch shutil.which in the frontend so check_mkdssp_installed succeeds."""
    return patch(f"{MODULE}.shutil.which",
                 side_effect=lambda name: f"/usr/bin/{name}")


def _df_one(seq="ACDEFGHIK"):
    return pd.DataFrame({"entry": ["P1"], "sequence": [seq]})


def _df_two():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIK", "MNPQRSTVW"],
    })


def _canned_chains_perfect(seq):
    """Stub run_dssp_for_entry_: one chain whose atom seq matches `seq` exactly,
    SS = all H (helix) — used as the 'happy path' canned response."""
    return [("A", seq, ["H"] * len(seq))]


def _write_pdb(path, lines):
    """Write a minimal PDB stub at ``path`` (content matches ``lines``)."""
    path.write_text("\n".join(lines) + "\n")


# II Test Classes
class TestGetDssp:
    """Single-parameter normal + invalid cases for get_dssp()."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_no_mkdssp_binary(self):
        with patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="mkdssp"):
                aa.get_dssp(df_seq=_df_one(), pdb_folder="/tmp")

    def test_invalid_df_seq_none(self, tmp_path):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="df_seq"):
                aa.get_dssp(df_seq=None, pdb_folder=str(tmp_path))

    def test_invalid_df_seq_missing_sequence_column(self, tmp_path):
        df = pd.DataFrame({"entry": ["P1"], "tmd": ["ACDE"]})
        with _mock_binary_present():
            with pytest.raises(ValueError):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_df_seq_missing_entry_column(self, tmp_path):
        df = pd.DataFrame({"sequence": ["ACDE"]})
        with _mock_binary_present():
            with pytest.raises(ValueError):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_pdb_folder_none(self):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                aa.get_dssp(df_seq=_df_one(), pdb_folder=None)

    def test_invalid_pdb_folder_nonexistent(self):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                aa.get_dssp(df_seq=_df_one(),
                            pdb_folder="/nonexistent/__no_such_dir__")

    def test_invalid_pdb_folder_is_file(self, tmp_path):
        bogus = tmp_path / "not_a_dir.txt"
        bogus.write_text("hi")
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                aa.get_dssp(df_seq=_df_one(), pdb_folder=str(bogus))

    def test_invalid_ss_mode_value(self, tmp_path):
        with _mock_binary_present():
            for bad in ["ss4", "SS3", "", "raw"]:
                with pytest.raises(ValueError, match="ss_mode"):
                    aa.get_dssp(df_seq=_df_one(),
                                pdb_folder=str(tmp_path), ss_mode=bad)

    def test_invalid_ss_mode_none(self, tmp_path):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="ss_mode"):
                aa.get_dssp(df_seq=_df_one(),
                            pdb_folder=str(tmp_path), ss_mode=None)

    def test_invalid_gap_handling_value(self, tmp_path):
        with _mock_binary_present():
            for bad in ["drop", "PAD", "fill", ""]:
                with pytest.raises(ValueError, match="gap_handling"):
                    aa.get_dssp(df_seq=_df_one(),
                                pdb_folder=str(tmp_path), gap_handling=bad)

    def test_invalid_gap_handling_none(self, tmp_path):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="gap_handling"):
                aa.get_dssp(df_seq=_df_one(),
                            pdb_folder=str(tmp_path), gap_handling=None)

    def test_invalid_verbose_non_bool(self, tmp_path):
        with _mock_binary_present():
            for bad in [1, 0, "true", None]:
                with pytest.raises(ValueError):
                    aa.get_dssp(df_seq=_df_one(),
                                pdb_folder=str(tmp_path), verbose=bad)

    def test_invalid_existing_ss_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_SS] = [["H"] * 9]
        with _mock_binary_present():
            with pytest.raises(ValueError, match="ss"):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_existing_dssp_ok_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_DSSP_OK] = [True]
        with _mock_binary_present():
            with pytest.raises(ValueError, match="dssp_ok"):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_unsafe_entry_path_traversal(self, tmp_path):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        with _mock_binary_present():
            with pytest.raises(ValueError, match="entry"):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    # ----- POSITIVES (≥10) -----
    def test_valid_returns_dataframe(self, tmp_path):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              verbose=False)
        assert isinstance(out, pd.DataFrame)

    def test_valid_columns_added(self, tmp_path):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              verbose=False)
        assert ut.COL_SS in out.columns
        assert ut.COL_DSSP_OK in out.columns

    def test_valid_ss3_alphabet(self, tmp_path):
        df = _df_one()
        # Canned SS: every DSSP letter present
        chains = [("A", "ACDEFGHIK",
                   ["H", "G", "I", "E", "B", "T", "S", " ", "H"])]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              ss_mode="ss3", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert set(ss).issubset({"H", "E", "C", "-"})

    def test_valid_ss8_alphabet(self, tmp_path):
        df = _df_one()
        chains = [("A", "ACDEFGHIK",
                   ["H", "G", "I", "E", "B", "T", "S", " ", "H"])]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              ss_mode="ss8", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        # ss8 returns raw DSSP letters; literal space remapped to '-'
        assert set(ss).issubset(set("HBEGITS-"))

    def test_valid_gap_handling_pad_length(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              gap_handling="pad", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert len(ss) == 9

    def test_valid_gap_handling_omit_no_dashes(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        # Chain is shorter than the target sequence → padding would insert '-'.
        chains = [("A", "ACDEF", ["H"] * 5)]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              gap_handling="omit", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert "-" not in ss
        assert len(ss) <= 9

    def test_valid_verbose_true_runs(self, tmp_path, capsys):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              verbose=True)
        assert bool(out[ut.COL_DSSP_OK].iloc[0])
        captured = capsys.readouterr()
        assert "P1" in captured.out

    def test_valid_verbose_false_quiet(self, tmp_path, capsys):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path), verbose=False)
        captured = capsys.readouterr()
        assert "P1" not in captured.out

    def test_valid_missing_pdb_warns_and_continues(self, tmp_path):
        df = _df_one()
        with _mock_binary_present():
            with pytest.warns(UserWarning, match="not found"):
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  verbose=False)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])
        assert out[ut.COL_SS].iloc[0] is None
        assert len(out) == 1

    def test_valid_multi_row_all_preserved(self, tmp_path):
        df = _df_two()
        chains_by_entry = {
            "P1": _canned_chains_perfect("ACDEFGHIK"),
            "P2": _canned_chains_perfect("MNPQRSTVW"),
        }
        # Look up the canned response by entry name from the path basename.
        def _runner(p):
            entry = Path(p).stem
            return chains_by_entry[entry]
        with _mock_binary_present(), patch(RUNNER, side_effect=_runner):
            (tmp_path / "P1.pdb").write_text("dummy")
            (tmp_path / "P2.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              verbose=False)
        assert len(out) == 2
        assert all(out[ut.COL_DSSP_OK])

    def test_valid_input_not_mutated(self, tmp_path):
        df = _df_one()
        df_copy = df.copy()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path), verbose=False)
        # Original df_seq must not have gained the new columns.
        assert ut.COL_SS not in df.columns
        assert ut.COL_DSSP_OK not in df.columns
        pd.testing.assert_frame_equal(df, df_copy)

    def test_valid_pdb_folder_accepts_pathlib(self, tmp_path):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=tmp_path, verbose=False)
        assert bool(out[ut.COL_DSSP_OK].iloc[0])

    @given(seq=some.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=5, max_size=20))
    def test_valid_arbitrary_sequence_pad_lengths_match(self, seq):
        df = pd.DataFrame({"entry": ["P1"], "sequence": [seq]})
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: _canned_chains_perfect(seq)):
            # tmp_path isn't usable inside hypothesis @given; use a real folder
            # via tempfile.
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "P1.pdb").write_text("dummy")
                out = aa.get_dssp(df_seq=df, pdb_folder=td,
                                  ss_mode="ss3", gap_handling="pad",
                                  verbose=False)
        assert len(out[ut.COL_SS].iloc[0]) == len(seq)


class TestGetDsspComplex:
    """Combinations and edge-case interactions for get_dssp()."""

    # ----- POSITIVES (≥5) -----
    def test_complex_multi_chain_best_match_used(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        # 3 chains, only chain C matches; the other two are unrelated.
        chains = [
            ("A", "WWWWWWWWW", ["E"] * 9),
            ("B", "YYYYYYYYY", ["E"] * 9),
            ("C", "ACDEFGHIK", ["H"] * 9),
        ]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                              ss_mode="ss3", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        # The best-matching chain has all-H, which maps to all-H in ss3.
        assert ss == ["H"] * 9

    def test_complex_partial_mismatch_pad_inserts_gap(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        # Chain only covers 5 of 9 residues; padding fills the rest with '-'.
        chains = [("A", "ACDEF", ["H"] * 5)]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  ss_mode="ss3", gap_handling="pad",
                                  verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert len(ss) == 9
        assert "-" in ss

    def test_complex_partial_mismatch_omit_drops_gap(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        chains = [("A", "ACDEF", ["H"] * 5)]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  gap_handling="omit", verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert "-" not in ss
        assert len(ss) == 5

    def test_complex_mixed_success_and_failure(self, tmp_path):
        df = pd.DataFrame({
            "entry": ["P1", "P2", "P3"],
            "sequence": ["ACDEFGHIK", "MNPQRSTVW", "AAAAAAAAA"],
        })
        chains_by_entry = {
            "P1": _canned_chains_perfect("ACDEFGHIK"),
            "P3": _canned_chains_perfect("AAAAAAAAA"),
        }
        # P2 missing on disk → row should fail; P1 and P3 succeed.
        def _runner(p):
            return chains_by_entry[Path(p).stem]
        with _mock_binary_present(), patch(RUNNER, side_effect=_runner):
            (tmp_path / "P1.pdb").write_text("dummy")
            (tmp_path / "P3.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  verbose=False)
        assert len(out) == 3
        assert [bool(x) for x in out[ut.COL_DSSP_OK]] == [True, False, True]
        assert out[ut.COL_SS].iloc[1] is None

    def test_complex_ss8_with_omit_no_dashes(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        chains = [("A", "ACDEFG",
                   ["H", "G", "I", "E", "B", "T"])]
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  ss_mode="ss8", gap_handling="omit",
                                  verbose=False)
        ss = out[ut.COL_SS].iloc[0]
        assert "-" not in ss
        assert set(ss).issubset(set("HBEGITS"))

    def test_complex_perfect_match_no_warning(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_chains_perfect("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  verbose=False)
        assert bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_complex_no_chains_returned_marks_failure(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: []):
            (tmp_path / "P1.pdb").write_text("dummy")
            with pytest.warns(UserWarning, match="No chains"):
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  verbose=False)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])
        assert out[ut.COL_SS].iloc[0] is None

    # ----- NEGATIVES (≥5) -----
    def test_complex_negative_collision_ss_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_SS] = [["H"] * 9]
        with _mock_binary_present():
            with pytest.raises(ValueError):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_complex_negative_collision_dssp_ok_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_DSSP_OK] = [True]
        with _mock_binary_present():
            with pytest.raises(ValueError):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_complex_negative_invalid_ss_and_gap_handling(self, tmp_path):
        with _mock_binary_present():
            with pytest.raises(ValueError, match="ss_mode"):
                aa.get_dssp(df_seq=_df_one(), pdb_folder=str(tmp_path),
                            ss_mode="bogus", gap_handling="also_bad")

    def test_complex_negative_unsafe_entry_with_slash(self, tmp_path):
        df = pd.DataFrame({"entry": ["foo/bar"], "sequence": ["ACDE"]})
        with _mock_binary_present():
            with pytest.raises(ValueError, match="entry"):
                aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_complex_negative_dssp_runtime_failure(self, tmp_path):
        df = _df_one()
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=RuntimeError("simulated DSSP crash")):
            (tmp_path / "P1.pdb").write_text("dummy")
            with pytest.warns(UserWarning, match="DSSP failed"):
                out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                  verbose=False)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])
        assert out[ut.COL_SS].iloc[0] is None

    def test_complex_negative_pdb_folder_nonexistent_before_loop(self):
        # Validation should fail before any row processing.
        df = _df_two()
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                aa.get_dssp(df_seq=df, pdb_folder="/__definitely_not_a_dir__")


@mkdssp_required
class TestGetDsspSmoke:
    """End-to-end smoke that actually invokes mkdssp on a tiny PDB.

    Skipped automatically when no DSSP binary is on PATH. The fixture PDB
    holds three alpha-helix residues; mkdssp should run cleanly even though
    only a stub geometry is provided (it will assign 'no SS' if helix
    geometry isn't satisfied — that's fine; we only check that the pipeline
    runs end-to-end and returns dssp_ok=True).
    """

    def test_smoke_runs_on_minimal_pdb(self, tmp_path):
        # Minimal valid PDB: 4 alanines, plausible CA backbone positions.
        # Coordinates aren't physically realistic but mkdssp accepts the format.
        pdb_lines = [
            "HEADER    TEST",
            "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00  0.00           C",
            "ATOM      3  C   ALA A   1      26.913  26.639   3.531  1.00  0.00           C",
            "ATOM      4  O   ALA A   1      27.886  26.463   4.263  1.00  0.00           O",
            "ATOM      5  N   ALA A   2      26.335  27.770   3.258  1.00  0.00           N",
            "ATOM      6  CA  ALA A   2      26.850  29.021   3.898  1.00  0.00           C",
            "ATOM      7  C   ALA A   2      26.100  29.253   5.202  1.00  0.00           C",
            "ATOM      8  O   ALA A   2      24.865  29.024   5.330  1.00  0.00           O",
            "ATOM      9  N   ALA A   3      26.847  29.694   6.204  1.00  0.00           N",
            "ATOM     10  CA  ALA A   3      26.348  29.974   7.547  1.00  0.00           C",
            "ATOM     11  C   ALA A   3      26.484  31.460   7.857  1.00  0.00           C",
            "ATOM     12  O   ALA A   3      27.521  32.026   7.512  1.00  0.00           O",
            "ATOM     13  N   ALA A   4      25.504  32.087   8.474  1.00  0.00           N",
            "ATOM     14  CA  ALA A   4      25.555  33.519   8.831  1.00  0.00           C",
            "ATOM     15  C   ALA A   4      26.748  33.781   9.741  1.00  0.00           C",
            "ATOM     16  O   ALA A   4      27.523  34.700   9.452  1.00  0.00           O",
            "TER      17      ALA A   4",
            "END",
        ]
        _write_pdb(tmp_path / "AAAA.pdb", pdb_lines)
        df = pd.DataFrame({"entry": ["AAAA"], "sequence": ["AAAA"]})
        out = aa.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                          ss_mode="ss3", verbose=False)
        assert isinstance(out, pd.DataFrame)
        assert bool(out[ut.COL_DSSP_OK].iloc[0])
        ss = out[ut.COL_SS].iloc[0]
        assert isinstance(ss, list)
        assert len(ss) == 4
        assert all(c in {"H", "E", "C", "-"} for c in ss)
