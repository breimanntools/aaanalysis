"""This is a script to test StructurePreprocessor.get_dssp()."""
import shutil
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=2000)
settings.load_profile("ci")

MODULE = "aaanalysis.struct_analysis_pro._structure_preprocessor"
RUNNER = f"{MODULE}.run_dssp_full_for_entry_"


# I Helper Functions
def _mock_binary_present():
    return patch(f"{MODULE}.shutil.which",
                 side_effect=lambda name: f"/usr/bin/{name}")


def _df_one(seq="ACDEFGHIK"):
    return pd.DataFrame({"entry": ["P1"], "sequence": [seq]})


def _df_two():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIK", "MNPQRSTVW"],
    })


def _canned_full(seq, ss_char="H", asa_val=80.0, phi_val=-60.0,
                 psi_val=-45.0):
    """Stub run_dssp_full_for_entry_: one chain, full feature streams."""
    L = len(seq)
    return [("A", seq,
             [ss_char] * L,
             [asa_val] * L,
             [phi_val] * L,
             [psi_val] * L)]


# II Test Classes
class TestStpGetDssp:
    """Single-parameter normal + invalid cases for StructurePreprocessor.get_dssp()."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_no_mkdssp_binary(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="mkdssp"):
                stp.get_dssp(df_seq=_df_one(), pdb_folder="/tmp")

    def test_invalid_df_seq_none(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=None, pdb_folder=str(tmp_path))

    def test_invalid_pdb_folder_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                stp.get_dssp(df_seq=_df_one(), pdb_folder=None)

    def test_invalid_pdb_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError, match="pdb_folder"):
                stp.get_dssp(df_seq=_df_one(),
                             pdb_folder="/nonexistent/__no_such_dir__")

    def test_invalid_features_empty(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError, match="features"):
                stp.get_dssp(df_seq=_df_one(),
                             pdb_folder=str(tmp_path), features=[])

    def test_invalid_features_unknown(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError, match="features"):
                stp.get_dssp(df_seq=_df_one(),
                             pdb_folder=str(tmp_path), features=["dihedrals"])

    def test_invalid_features_not_list(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=_df_one(),
                             pdb_folder=str(tmp_path), features="ss")

    def test_invalid_ss_mode_value(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            for bad in ["ss4", "SS3", "", "raw"]:
                with pytest.raises(ValueError, match="ss_mode"):
                    stp.get_dssp(df_seq=_df_one(),
                                 pdb_folder=str(tmp_path), ss_mode=bad)

    def test_invalid_gap_handling_value(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            for bad in ["drop", "PAD", "fill", ""]:
                with pytest.raises(ValueError, match="gap_handling"):
                    stp.get_dssp(df_seq=_df_one(),
                                 pdb_folder=str(tmp_path), gap_handling=bad)

    def test_invalid_existing_ss_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_SS] = [["H"] * 9]
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_existing_dssp_ok_column(self, tmp_path):
        df = _df_one()
        df[ut.COL_DSSP_OK] = [True]
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    def test_invalid_unsafe_entry(self, tmp_path):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError, match="entry"):
                stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))

    # ----- POSITIVES (≥10) -----
    def test_valid_returns_dataframe(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert isinstance(out, pd.DataFrame)

    def test_valid_ss_column_added(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["ss"])
        assert ut.COL_SS in out.columns
        assert ut.COL_DSSP_OK in out.columns

    def test_valid_asa_column_added(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["asa"])
        assert "asa" in out.columns

    def test_valid_phi_psi_columns_added(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["phi_psi"])
        assert "phi" in out.columns
        assert "psi" in out.columns

    def test_valid_default_features_all_three(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert {"ss", "asa", "phi", "psi"}.issubset(set(out.columns))

    def test_valid_ss3_length_matches_seq(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["ss"], ss_mode="ss3")
        ss = out[ut.COL_SS].iloc[0]
        assert len(ss) == 9
        assert set(ss).issubset({"H", "E", "C", "-"})

    def test_valid_ss8_alphabet(self, tmp_path):
        df = _df_one()
        chains = [("A", "ACDEFGHIK",
                   ["H", "G", "I", "E", "B", "T", "S", " ", "H"],
                   [50.0] * 9, [-60.0] * 9, [-45.0] * 9)]
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["ss"], ss_mode="ss8")
        ss = out[ut.COL_SS].iloc[0]
        assert set(ss).issubset(set("HBEGITS-"))

    def test_valid_gap_omit_drops_unresolved(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        chains = [("A", "ACDEF",
                   ["H"] * 5, [80.0] * 5, [-60.0] * 5, [-45.0] * 5)]
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               gap_handling="omit",
                               features=["ss", "asa", "phi_psi"])
        assert "-" not in out[ut.COL_SS].iloc[0]
        assert len(out["asa"].iloc[0]) == len(out[ut.COL_SS].iloc[0])

    def test_valid_dssp_ok_true(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_valid_missing_pdb_warns(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
            assert any("not found" in str(x.message) for x in w)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_valid_two_rows(self, tmp_path):
        df = _df_two()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: _canned_full(
                 "ACDEFGHIK" if "P1" in str(p) else "MNPQRSTVW")):
            (tmp_path / "P1.pdb").write_text("dummy")
            (tmp_path / "P2.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert len(out) == 2
        assert all(out[ut.COL_DSSP_OK])

    def test_valid_verbose_override_quiet(self, tmp_path, capsys):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=True)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path), verbose=False)
        cap = capsys.readouterr()
        assert "P1" not in cap.out


class TestStpGetDsspComplex:
    """Cross-parameter combinations for get_dssp."""

    def test_complex_ss_only_omits_other_columns(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["ss"])
        assert "asa" not in out.columns
        assert "phi" not in out.columns
        assert "psi" not in out.columns

    def test_complex_phi_psi_pair_added(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               features=["phi_psi"])
        assert "phi" in out.columns and "psi" in out.columns

    def test_complex_gap_pad_keeps_length(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        chains = [("A", "ACDEF",
                   ["H"] * 5, [80.0] * 5, [-60.0] * 5, [-45.0] * 5)]
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: chains):
            (tmp_path / "P1.pdb").write_text("dummy")
            out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path),
                               gap_handling="pad")
        assert len(out[ut.COL_SS].iloc[0]) == 9

    def test_complex_failed_entry_keeps_row(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert len(out) == 1
        assert out[ut.COL_SS].iloc[0] is None

    def test_complex_mixed_ok_failed_rows(self, tmp_path):
        df = _df_two()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert out[ut.COL_DSSP_OK].tolist() == [True, False]

    def test_complex_features_list_idempotent(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            out1 = stp.get_dssp(df_seq=df.copy(), pdb_folder=str(tmp_path),
                                features=["ss"])
            out2 = stp.get_dssp(df_seq=df.copy(), pdb_folder=str(tmp_path),
                                features=["ss"])
        assert out1[ut.COL_SS].iloc[0] == out2[ut.COL_SS].iloc[0]

    def test_complex_negative_features_with_garbage(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=_df_one(), pdb_folder=str(tmp_path),
                             features=[None])

    def test_complex_negative_pdb_folder_is_file(self, tmp_path):
        bogus = tmp_path / "not_a_dir.txt"
        bogus.write_text("hi")
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present():
            with pytest.raises(ValueError):
                stp.get_dssp(df_seq=_df_one(), pdb_folder=str(bogus))
