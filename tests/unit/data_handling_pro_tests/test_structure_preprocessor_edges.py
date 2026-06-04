"""This is a script to test StructurePreprocessor.get_dssp() failure / edge
branches not covered by the main test: DSSP RuntimeError -> dssp_ok=False,
no-matching-chains, residue-mismatch warning, and verbose output.

Reuses the mock pattern from test_structure_preprocessor_get_dssp.py (patch the
DSSP runner + shutil.which).
"""
import shutil
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

PDB_FIXTURES = Path(__file__).resolve().parents[3] / "aaanalysis" / "_data" / "pdb_test"
AF_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"  # len 30, matches AF_TINY fixture

aa.options["verbose"] = False

MODULE = "aaanalysis.data_handling_pro._struct_preproc"
RUNNER = f"{MODULE}.run_dssp_full_for_entry_"


def _mock_binary_present():
    return patch(f"{MODULE}.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")


def _df_one(seq="ACDEFGHIK"):
    return pd.DataFrame({"entry": ["P1"], "sequence": [seq]})


def _canned(seq, ss="H"):
    L = len(seq)
    return [("A", seq, [ss] * L, [80.0] * L, [-60.0] * L, [-45.0] * L,
            [-4] * L, [-2.0] * L, [4] * L, [-2.0] * L)]


class TestGetDsspEdges:
    def test_dssp_runtimeerror_sets_not_ok(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        (tmp_path / "P1.pdb").write_text("dummy")
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=RuntimeError("dssp boom")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert any("DSSP failed" in str(x.message) for x in w)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_no_matching_chains_sets_not_ok(self, tmp_path):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=False)
        (tmp_path / "P1.pdb").write_text("dummy")
        with _mock_binary_present(), patch(RUNNER, side_effect=lambda p: []):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert any("No chains" in str(x.message) for x in w)
        assert not bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_residue_mismatch_warns_but_ok(self, tmp_path):
        df = _df_one("ACDEFGHIK")
        stp = aa.StructurePreprocessor(verbose=False)
        (tmp_path / "P1.pdb").write_text("dummy")
        # chain sequence differs from target -> mismatch warning, still dssp_ok
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: _canned("MMMMMMMMM")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert any("mismatch" in str(x.message) for x in w)
        assert bool(out[ut.COL_DSSP_OK].iloc[0])

    def test_verbose_prints_entry(self, tmp_path, capsys):
        df = _df_one()
        stp = aa.StructurePreprocessor(verbose=True)
        (tmp_path / "P1.pdb").write_text("dummy")
        with _mock_binary_present(), \
             patch(RUNNER, side_effect=lambda p: _canned("ACDEFGHIK")):
            stp.get_dssp(df_seq=df, pdb_folder=str(tmp_path))
        assert "P1" in capsys.readouterr().out


class TestEncodePdbEdges:
    """encode_pdb failure branches: parse fail, encoder fail, on_failure policy."""

    def test_parse_fail_sets_not_ok(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.load_structure", side_effect=ValueError("bad pdb")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, df_out = stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                                           features=["bfactor"], return_df=True)
        assert any("parse failed" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])  # *_ok column False

    def test_encoder_fail_sets_not_ok(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.load_structure", return_value=object()), \
             patch(f"{MODULE}.encode_bfactor", side_effect=RuntimeError("enc boom")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, df_out = stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                                           features=["bfactor"], return_df=True)
        assert any("encoder" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])

    def test_on_failure_raise(self, tmp_path):
        # no PDB file -> failure -> raise (a 'not found' warning fires first)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                               features=["bfactor"], on_failure="raise")

    def test_on_failure_drop(self, tmp_path):
        # no PDB file -> failure -> drop the entry
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, df_out = stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                                         features=["bfactor"], on_failure="drop",
                                         return_df=True)
        assert len(df_out) == 0  # the only entry was dropped

    def test_verbose_pdb(self, tmp_path, capsys):
        (tmp_path / "P1.pdb").write_text("x")
        import numpy as np
        stp = aa.StructurePreprocessor(verbose=True)
        with patch(f"{MODULE}.load_structure", return_value=object()), \
             patch(f"{MODULE}.encode_bfactor",
                   return_value=(np.zeros((9, 1)), 1.0)):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                           features=["bfactor"])
        assert "P1" in capsys.readouterr().out


class TestEncodePaeEdges:
    """encode_pae failure policy (missing sidecar)."""

    def test_missing_sidecar_nan(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = stp.encode_pae(df_seq=_df_one(), pae_folder=str(tmp_path),
                                 features=["pae_row_mean"])
        assert "P1" in out  # NaN tensor present under 'nan' policy

    def test_missing_sidecar_raise(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pae(df_seq=_df_one(), pae_folder=str(tmp_path),
                               features=["pae_row_mean"], on_failure="raise")

    def test_missing_sidecar_drop(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, df_out = stp.encode_pae(df_seq=_df_one(), pae_folder=str(tmp_path),
                                         features=["pae_row_mean"], on_failure="drop",
                                         return_df=True)
        assert len(df_out) == 0


def _pae_df(tmp_path):
    """Copy the bundled AF_TINY PAE sidecar so encode_pae resolves it."""
    shutil.copy(PDB_FIXTURES / "AF_TINY_pae.json", Path(tmp_path) / "AF_TINY.json")
    return pd.DataFrame({"entry": ["AF_TINY"], "sequence": [AF_SEQ]})


class TestEncodePaeSuccess:
    """encode_pae success path (valid sidecar): encoder loop + verbose + encoder-fail."""

    def test_success(self, tmp_path):
        df = _pae_df(tmp_path)
        stp = aa.StructurePreprocessor(verbose=False)
        out = stp.encode_pae(df_seq=df, pae_folder=str(tmp_path),
                             features=["pae_row_mean"])
        assert "AF_TINY" in out
        assert out["AF_TINY"].shape[0] == len(AF_SEQ)

    def test_success_verbose(self, tmp_path, capsys):
        df = _pae_df(tmp_path)
        stp = aa.StructurePreprocessor(verbose=True)
        stp.encode_pae(df_seq=df, pae_folder=str(tmp_path),
                       features=["pae_row_mean"])
        assert "AF_TINY" in capsys.readouterr().out

    def test_encoder_fail_sets_not_ok(self, tmp_path):
        df = _pae_df(tmp_path)
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.encode_pae_row_mean", side_effect=RuntimeError("boom")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, df_out = stp.encode_pae(df_seq=df, pae_folder=str(tmp_path),
                                           features=["pae_row_mean"], return_df=True)
        assert any("encoder" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])


class TestEncodeDomainsInline:
    """encode_domains via the inline 'chopping' column (no domain files needed)."""

    def _df(self, chopping="1-10,15-30"):
        return pd.DataFrame({"entry": ["P1"], "sequence": ["A" * 30],
                             "chopping": [chopping]})

    DOMAIN_FEATS = ["domain_boundary", "domain_relative_position",
                    "domain_size", "n_domains_in_protein"]

    def test_success_all_features(self):
        stp = aa.StructurePreprocessor(verbose=False)
        out = stp.encode_domains(df_seq=self._df(), features=self.DOMAIN_FEATS)
        assert "P1" in out
        assert out["P1"].shape[0] == 30

    def test_success_verbose(self, capsys):
        stp = aa.StructurePreprocessor(verbose=True)
        stp.encode_domains(df_seq=self._df(), features=["domain_boundary"])
        assert "P1" in capsys.readouterr().out

    def test_inline_chopping_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, df_out = stp.encode_domains(df_seq=self._df(chopping=None),
                                           features=["domain_boundary"],
                                           return_df=True)
        assert any("missing" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])

    def test_inline_chopping_parse_fail(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, df_out = stp.encode_domains(df_seq=self._df(chopping="garbage!!"),
                                           features=["domain_boundary"],
                                           return_df=True)
        assert any("parse failed" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])

    def test_encoder_fail_sets_not_ok(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.encode_domain_boundary",
                   side_effect=RuntimeError("dom boom")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, df_out = stp.encode_domains(df_seq=self._df(),
                                               features=["domain_boundary"],
                                               return_df=True)
        assert any("encoder" in str(x.message) for x in w)
        assert not bool(df_out.iloc[0, -1])
