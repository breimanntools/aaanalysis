"""This is a script to test StructurePreprocessor.encode_dssp()."""
import warnings
from unittest.mock import patch

import numpy as np
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


def _canned_full(seq, ss_char="H", asa_val=80.0, phi_val=-60.0,
                 psi_val=-45.0):
    L = len(seq)
    return [("A", seq, [ss_char] * L,
             [asa_val] * L, [phi_val] * L, [psi_val] * L)]


def _df_pre(seq="ACDEFGHIK"):
    """A df_seq pre-populated with get_dssp-style list columns."""
    L = len(seq)
    return pd.DataFrame({
        "entry": ["P1"],
        "sequence": [seq],
        ut.COL_SS: [["H"] * L],
        "asa": [[80.0] * L],
        "phi": [[-60.0] * L],
        "psi": [[-45.0] * L],
        ut.COL_DSSP_OK: [True],
    })


# II Test Classes
class TestStpEncodeDssp:
    """Single-parameter normal + invalid cases for encode_dssp."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=None, pdb_folder=str(tmp_path),
                            features=["ss3"])

    def test_invalid_features_empty(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_one(), pdb_folder=str(tmp_path),
                            features=[])

    def test_invalid_features_unknown(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_one(), pdb_folder=str(tmp_path),
                            features=["unknown_key"])

    def test_invalid_features_wrong_method(self, tmp_path):
        # bfactor / depth belong to encode_pdb
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_one(), pdb_folder=str(tmp_path),
                            features=["bfactor"])

    def test_invalid_ss_mode_value(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="ss_mode"):
            stp.encode_dssp(df_seq=_df_pre(), features=["ss3"],
                            ss_mode="ss4")

    def test_invalid_dropped_key_asa(self, tmp_path):
        # v1.1: 'asa' (absolute) is dropped from the registry; only 'rasa'
        # remains. Passing the dropped key must raise.
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_pre(), features=["asa"])

    def test_invalid_dropped_key_phi_psi_raw(self, tmp_path):
        # v1.1: 'phi_psi' (raw degrees) is dropped; only 'phi_psi_sincos'.
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_pre(), features=["phi_psi"])

    def test_invalid_gap_handling(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="gap_handling"):
            stp.encode_dssp(df_seq=_df_pre(), features=["ss3"],
                            gap_handling="PAD")

    def test_invalid_on_failure(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            stp.encode_dssp(df_seq=_df_pre(), features=["ss3"],
                            on_failure="skip")

    def test_invalid_pdb_folder_none_no_precomputed(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_dssp(df_seq=_df_one(), pdb_folder=None,
                            features=["ss3"])

    def test_invalid_pdb_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_dssp(df_seq=_df_one(),
                            pdb_folder="/__nope__/__here__",
                            features=["ss3"])

    def test_invalid_raise_on_failure_propagates(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        df = _df_one()
        with _mock_binary_present():  # mkdssp "present"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    stp.encode_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                    features=["ss3"], on_failure="raise")

    # ----- POSITIVES (≥10) -----
    def test_valid_returns_dict(self):
        stp = aa.StructurePreprocessor(verbose=False)
        d, df_out = stp.encode_dssp(df_seq=_df_pre(), features=["ss3"])
        assert isinstance(d, dict)

    def test_valid_ss3_shape_is_L_by_3(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq), features=["ss3"])
        assert d["P1"].shape == (len(seq), 3)

    def test_valid_ss8_shape_is_L_by_8(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq), features=["ss8"])
        assert d["P1"].shape == (len(seq), 8)

    def test_valid_rasa_shape_is_L_by_1(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq), features=["rasa"])
        assert d["P1"].shape == (len(seq), 1)

    def test_valid_phi_psi_sincos_shape_is_L_by_4(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq),
                               features=["phi_psi_sincos"])
        assert d["P1"].shape == (len(seq), 4)

    def test_valid_multi_features_concat(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq),
                               features=["ss3", "rasa", "phi_psi_sincos"])
        assert d["P1"].shape == (len(seq), 3 + 1 + 4)

    def test_valid_ss3_onehot_is_binary(self):
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(), features=["ss3"])
        vals = d["P1"]
        nonnan = vals[~np.isnan(vals).any(axis=1)]
        assert set(np.unique(nonnan)).issubset({0.0, 1.0})

    def test_valid_helix_lands_in_helix_column(self):
        # All-H should activate column 0 (ss_helix)
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(), features=["ss3"])
        assert np.all(d["P1"][:, 0] == 1.0)

    def test_valid_rasa_in_unit_interval_when_valid(self):
        # 80 / max_ASA_AA values must yield finite values; all in [0, ~1]
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(), features=["rasa"])
        vals = d["P1"].ravel()
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0)

    def test_valid_runs_dssp_inline(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER,
                   side_effect=lambda p: _canned_full("ACDEFGHIK")):
            (tmp_path / "P1.pdb").write_text("dummy")
            d, df_out = stp.encode_dssp(
                df_seq=df, pdb_folder=str(tmp_path), features=["ss3"])
        assert d["P1"].shape == (9, 3)

    def test_valid_on_failure_nan_fills(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_dssp(
                df_seq=df, pdb_folder=str(tmp_path), features=["ss3"],
                on_failure="nan")
        assert d["P1"].shape == (9, 3)
        assert np.isnan(d["P1"]).all()

    def test_valid_on_failure_drop_removes_entry(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_dssp(
                df_seq=df, pdb_folder=str(tmp_path), features=["ss3"],
                on_failure="drop")
        assert "P1" not in d
        assert len(df_out) == 0


class TestStpEncodeDsspComplex:
    """Cross-parameter combinations for encode_dssp."""

    def test_complex_ss3_plus_phi_psi_sincos_total_dims(self):
        seq = "ACDEFGHIK"
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(seq),
                               features=["ss3", "phi_psi_sincos"])
        # 3 (ss3) + 4 (phi_psi_sincos) = 7
        assert d["P1"].shape == (len(seq), 7)

    def test_complex_rasa_in_unit_interval_after_normalize(self):
        # v1.1 normalization: rasa values are clipped to [0, 1].
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(), features=["rasa"])
        vals = d["P1"].ravel()
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_complex_phi_psi_sincos_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(),
                               features=["phi_psi_sincos"])
        assert d["P1"].shape[1] == 4
        # v1.1: sincos values are shifted to [0, 1].
        vals = d["P1"]
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_complex_uses_precomputed_skips_dssp(self):
        # No pdb_folder needed if columns already exist.
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=_df_pre(), pdb_folder=None,
                               features=["ss3"])
        assert d["P1"].shape == (9, 3)

    def test_complex_unresolved_position_becomes_nan_row(self):
        seq = "ACDEFGHIK"
        L = len(seq)
        df = pd.DataFrame({
            "entry": ["P1"], "sequence": [seq],
            ut.COL_SS: [["H", "-", "H", "-", "H", "H", "H", "H", "H"]],
            ut.COL_DSSP_OK: [True],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        d, _ = stp.encode_dssp(df_seq=df, features=["ss3"])
        # Rows 1 and 3 should be NaN; the rest should be valid one-hot.
        assert np.isnan(d["P1"][1]).all()
        assert np.isnan(d["P1"][3]).all()
        assert d["P1"][0, 0] == 1.0

    def test_complex_negative_drop_failed_keeps_succeeded(self, tmp_path):
        df = pd.DataFrame({"entry": ["P1", "P2"],
                           "sequence": ["ACDEFGHIK", "MNPQRSTVW"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: _canned_full(
                 "ACDEFGHIK" if "P1" in str(p) else "MNPQRSTVW")):
            (tmp_path / "P1.pdb").write_text("dummy")  # P2 missing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d, df_out = stp.encode_dssp(
                    df_seq=df, pdb_folder=str(tmp_path),
                    features=["ss3"], on_failure="drop")
        assert "P1" in d and "P2" not in d
        assert df_out["entry"].tolist() == ["P1"]

    def test_complex_negative_raise_when_any_failed(self, tmp_path):
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFGHIK"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with _mock_binary_present(), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_dssp(df_seq=df, pdb_folder=str(tmp_path),
                                features=["ss3"], on_failure="raise")

    def test_complex_features_with_int_rejected(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_pre(), features=[3])
