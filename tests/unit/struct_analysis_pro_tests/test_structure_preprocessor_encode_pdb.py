"""This is a script to test StructurePreprocessor.encode_pdb()."""
import shutil
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=2000)
settings.load_profile("ci")

PDB_FIXTURES = Path(__file__).resolve().parents[3] / \
    "aaanalysis" / "_data" / "pdb_test"

msms_required = pytest.mark.skipif(
    shutil.which("msms") is None,
    reason="msms binary not on PATH")


# I Helper Functions
def _df_one():
    return pd.DataFrame({
        "entry": ["P1"],
        "sequence": ["ACDEFGHIKLMNPQRS"],
    })


def _df_two():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIKLMNPQRS", "VLIMKRSTGADE"],
    })


# II Test Classes
class TestStpEncodePdb:
    """Single-parameter normal + invalid cases for encode_pdb."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=None, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_pdb_folder_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=None,
                           features=["bfactor"])

    def test_invalid_pdb_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_one(),
                           pdb_folder="/__nope__/__here__",
                           features=["bfactor"])

    def test_invalid_features_empty(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=[])

    def test_invalid_features_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["mystery"])

    def test_invalid_features_wrong_method(self):
        # ss3 belongs to encode_dssp
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["ss3"])

    def test_invalid_on_failure(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"], on_failure="ignore")

    def test_invalid_pdb_ok_collision(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df = _df_one()
        df["pdb_ok"] = [True]
        with pytest.raises(ValueError, match="pdb_ok"):
            stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_depth_without_msms(self):
        # When msms is missing, requesting 'depth' must raise.
        stp = aa.StructurePreprocessor(verbose=False)
        if shutil.which("msms") is None:
            with pytest.raises(RuntimeError, match="msms"):
                stp.encode_pdb(df_seq=_df_one(),
                               pdb_folder=str(PDB_FIXTURES),
                               features=["depth"])
        else:
            pytest.skip("msms available — skip absence assertion")

    def test_invalid_unsafe_entry(self):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="entry"):
            stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_raise_on_failure_propagates(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                               features=["bfactor"], on_failure="raise")

    # ----- POSITIVES (≥10) -----
    def test_valid_bfactor_returns_dict(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["bfactor"])
        assert isinstance(d, dict)
        assert "P1" in d

    def test_valid_bfactor_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)

    def test_valid_bfactor_values_finite(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        vals = d["P1"].ravel()
        assert np.all(np.isfinite(vals))

    def test_valid_pdb_ok_column(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["bfactor"])
        assert "pdb_ok" in df_out.columns
        assert bool(df_out["pdb_ok"].iloc[0])

    def test_valid_missing_pdb_yields_nan(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="nan")
        assert d["P1"].shape == (16, 1)
        assert np.isnan(d["P1"]).all()
        assert not bool(df_out["pdb_ok"].iloc[0])

    def test_valid_on_failure_drop(self, tmp_path):
        df = _df_two()
        # Copy only P1 from fixtures; P2 missing.
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df, pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="drop")
        assert "P1" in d and "P2" not in d
        assert df_out["entry"].tolist() == ["P1"]

    def test_valid_two_proteins_keys_present(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert set(d.keys()) == {"P1", "P2"}

    def test_valid_verbose_runs(self):
        stp = aa.StructurePreprocessor(verbose=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)

    def test_valid_p2_strand_bfactor_lower(self):
        # P1 helix has bfactor=30, P2 strand has bfactor=25.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        p1_mean = float(np.nanmean(d["P1"]))
        p2_mean = float(np.nanmean(d["P2"]))
        assert p1_mean > p2_mean

    @msms_required
    def test_valid_depth_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["depth"])
        assert d["P1"].shape == (16, 1)


class TestStpEncodePdbComplex:
    """Cross-parameter combinations for encode_pdb."""

    def test_complex_bfactor_with_two_entries_independent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)
        assert d["P2"].shape == (12, 1)

    def test_complex_drop_keeps_succeeded(self, tmp_path):
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_two(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="drop")
        assert set(d.keys()) == {"P1"}
        assert df_out["pdb_ok"].tolist() == [True]

    def test_complex_nan_default_two_entries(self, tmp_path):
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_two(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"])
        assert np.isnan(d["P2"]).all()
        assert not np.isnan(d["P1"]).all()
        assert df_out["pdb_ok"].tolist() == [True, False]

    def test_complex_raise_when_any_missing(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_two(), pdb_folder=str(tmp_path),
                               features=["bfactor"], on_failure="raise")

    def test_complex_features_typed_as_int_rejected(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=[7])

    def test_complex_features_int_rejected_with_typed_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features="bfactor")  # str, not list

    def test_complex_default_on_failure_is_nan(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=pd.DataFrame({
                "entry": ["P_missing"],
                "sequence": ["ACDE"]}),
                pdb_folder=str(PDB_FIXTURES),
                features=["bfactor"])
        assert np.isnan(d["P_missing"]).all()
