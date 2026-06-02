"""This is a script to test StructurePreprocessor.get_domains() —
ChainSaw + AFragmenter wrappers (v1.2 commit 4).

Both upstream tools are mocked so the test suite does NOT require the
actual ``afragmenter`` package or a local ChainSaw clone.
The wrappers' subprocess / lazy-import boundaries are the targets.
"""
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=2000)
settings.load_profile("ci")

# Patch targets — use the same module path the frontend imports from.
MODULE = "aaanalysis.data_handling_pro._struct_preproc"
RUN_AF = f"{MODULE}.run_afragmenter_on_pae"
RUN_CS = f"{MODULE}.run_chainsaw_on_entry"
RESOLVE_CS = f"{MODULE}.resolve_chainsaw_path"
CHECK_AF = f"{MODULE}.check_afragmenter_available"


# I Helper Functions
def _df_one(seq_len=30, entry="P1"):
    return pd.DataFrame({"entry": [entry], "sequence": ["A" * seq_len]})


def _df_two():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["A" * 30, "A" * 20],
    })


def _write_fake_pae(folder: Path, entry: str, L: int = 30):
    """Write a (L, L) zero-PAE matrix to the resolver-canonical filename."""
    pae = [[0.0] * L for _ in range(L)]
    (folder / f"{entry}.json").write_text(
        json.dumps({"predicted_aligned_error": pae}))


def _write_fake_pdb(folder: Path, entry: str):
    """Write a minimal stub that ChainSaw won't actually parse (we mock the runner)."""
    (folder / f"{entry}.pdb").write_text("HEADER STUB\nEND\n")


# II Test Classes — AFragmenter path
class TestGetDomainsAfragmenter:
    """get_domains(tool='afragmenter') — pip-installable, lazy-imported."""

    # ----- NEGATIVES (≥5) -----
    def test_invalid_tool_unknown(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="tool"):
            stp.get_domains(df_seq=_df_one(), pae_folder=str(tmp_path),
                            tool="merizo")

    def test_invalid_afragmenter_missing_pae_folder(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF):
            with pytest.raises(ValueError, match="pae_folder"):
                stp.get_domains(df_seq=_df_one(), pae_folder=None,
                                tool="afragmenter")

    def test_invalid_afragmenter_pae_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_folder"):
            stp.get_domains(df_seq=_df_one(),
                            pae_folder="/__nope__/__here__",
                            tool="afragmenter")

    def test_invalid_afragmenter_chopping_collision(self, tmp_path):
        df = _df_one()
        df["chopping"] = ["1-30"]
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="chopping"):
            stp.get_domains(df_seq=df, pae_folder=str(tmp_path),
                            tool="afragmenter")

    def test_invalid_afragmenter_resolution_out_of_range(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF):
            with pytest.raises(ValueError, match="resolution"):
                stp.get_domains(df_seq=_df_one(),
                                pae_folder=str(tmp_path),
                                tool="afragmenter", resolution=-1.0)

    def test_invalid_afragmenter_threshold_out_of_range(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF):
            with pytest.raises(ValueError, match="threshold"):
                stp.get_domains(df_seq=_df_one(),
                                pae_folder=str(tmp_path),
                                tool="afragmenter", threshold=50.0)

    def test_invalid_afragmenter_on_failure_value(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            stp.get_domains(df_seq=_df_one(), pae_folder=str(tmp_path),
                            tool="afragmenter", on_failure="skip")

    # ----- POSITIVES (≥6) -----
    def test_valid_afragmenter_returns_chopping(self, tmp_path):
        _write_fake_pae(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF, return_value="1-10,15-25"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pae_folder=str(tmp_path),
                                     tool="afragmenter")
        assert df_out["chopping"].iloc[0] == "1-10,15-25"
        assert bool(df_out["domain_ok"].iloc[0])

    def test_valid_afragmenter_appends_two_columns(self, tmp_path):
        _write_fake_pae(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF, return_value="1-30"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pae_folder=str(tmp_path),
                                     tool="afragmenter")
        assert "chopping" in df_out.columns
        assert "domain_ok" in df_out.columns

    def test_valid_afragmenter_missing_pae_warns_and_nan(self, tmp_path):
        # No PAE file written → resolver returns None → row fails.
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF):  # would not be called
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_out = stp.get_domains(df_seq=_df_one(),
                                         pae_folder=str(tmp_path),
                                         tool="afragmenter")
        assert df_out["chopping"].iloc[0] == ""
        assert not bool(df_out["domain_ok"].iloc[0])

    def test_valid_afragmenter_kwargs_pass_through(self, tmp_path):
        _write_fake_pae(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF, return_value="1-30") as run_mock:
            stp.get_domains(df_seq=_df_one(), pae_folder=str(tmp_path),
                            tool="afragmenter", resolution=0.42,
                            threshold=3.5)
        kwargs = run_mock.call_args.kwargs
        assert kwargs["resolution"] == 0.42
        assert kwargs["threshold"] == 3.5

    def test_valid_afragmenter_drop_failed(self, tmp_path):
        _write_fake_pae(tmp_path, "P1")   # only P1 has PAE; P2 missing
        df = _df_two()
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF, return_value="1-30"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_out = stp.get_domains(df_seq=df,
                                         pae_folder=str(tmp_path),
                                         tool="afragmenter",
                                         on_failure="drop")
        assert df_out["entry"].tolist() == ["P1"]

    def test_valid_afragmenter_raise_on_failure(self, tmp_path):
        # No file → on_failure='raise' should raise.
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    stp.get_domains(df_seq=_df_one(),
                                    pae_folder=str(tmp_path),
                                    tool="afragmenter",
                                    on_failure="raise")

    def test_valid_afragmenter_then_encode_domains(self, tmp_path):
        # End-to-end: get_domains output flows into encode_domains via
        # the in-memory 'chopping' column.
        _write_fake_pae(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(CHECK_AF), \
             patch(RUN_AF, return_value="1-10,15-25"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pae_folder=str(tmp_path),
                                     tool="afragmenter")
            d, df_aug = stp.encode_domains(return_df=True, 
                df_seq=df_out,
                features=["domain_boundary",
                          "domain_relative_position"])
        assert d["P1"].shape == (30, 2)
        # domain_ok gets overwritten by encode_domains' per-feature outcome.
        assert bool(df_aug["domain_ok"].iloc[0])


# II Test Classes — ChainSaw path
class TestGetDomainsChainsaw:
    """get_domains(tool='chainsaw') — BYO local clone + subprocess."""

    # ----- NEGATIVES (≥5) -----
    def test_invalid_chainsaw_missing_pdb_folder(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS):
            with pytest.raises(ValueError, match="pdb_folder"):
                stp.get_domains(df_seq=_df_one(), pdb_folder=None,
                                tool="chainsaw",
                                chainsaw_path=str(tmp_path))

    def test_invalid_chainsaw_pdb_folder_nonexistent(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS):
            with pytest.raises(ValueError, match="pdb_folder"):
                stp.get_domains(df_seq=_df_one(),
                                pdb_folder="/__nope__",
                                tool="chainsaw",
                                chainsaw_path=str(tmp_path))

    def test_invalid_chainsaw_path_none(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(RuntimeError, match="chainsaw_path"):
            stp.get_domains(df_seq=_df_one(),
                            pdb_folder=str(tmp_path),
                            tool="chainsaw", chainsaw_path=None)

    def test_invalid_chainsaw_path_not_a_dir(self, tmp_path):
        bogus = tmp_path / "not_a_dir.txt"
        bogus.write_text("hi")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(RuntimeError, match="chainsaw_path"):
            stp.get_domains(df_seq=_df_one(),
                            pdb_folder=str(tmp_path),
                            tool="chainsaw", chainsaw_path=str(bogus))

    def test_invalid_chainsaw_path_missing_entry_script(self, tmp_path):
        # Directory exists but has no get_predictions.py.
        cs_dir = tmp_path / "fake_chainsaw"
        cs_dir.mkdir()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(RuntimeError, match="get_predictions.py"):
            stp.get_domains(df_seq=_df_one(),
                            pdb_folder=str(tmp_path),
                            tool="chainsaw", chainsaw_path=str(cs_dir))

    # ----- POSITIVES (≥6) -----
    def test_valid_chainsaw_returns_chopping(self, tmp_path):
        _write_fake_pdb(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS, return_value="1-30"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pdb_folder=str(tmp_path),
                                     tool="chainsaw",
                                     chainsaw_path="/fake")
        assert df_out["chopping"].iloc[0] == "1-30"
        assert bool(df_out["domain_ok"].iloc[0])

    def test_valid_chainsaw_appends_two_columns(self, tmp_path):
        _write_fake_pdb(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS, return_value="1-30"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pdb_folder=str(tmp_path),
                                     tool="chainsaw",
                                     chainsaw_path="/fake")
        assert "chopping" in df_out.columns
        assert "domain_ok" in df_out.columns

    def test_valid_chainsaw_missing_pdb_warns(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS):  # would not be called
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_out = stp.get_domains(df_seq=_df_one(),
                                         pdb_folder=str(tmp_path),
                                         tool="chainsaw",
                                         chainsaw_path="/fake")
        assert df_out["chopping"].iloc[0] == ""
        assert not bool(df_out["domain_ok"].iloc[0])

    def test_valid_chainsaw_runtime_failure_warns(self, tmp_path):
        _write_fake_pdb(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS,
                   side_effect=RuntimeError("subprocess exit 1")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_out = stp.get_domains(df_seq=_df_one(),
                                         pdb_folder=str(tmp_path),
                                         tool="chainsaw",
                                         chainsaw_path="/fake")
        assert df_out["chopping"].iloc[0] == ""
        assert not bool(df_out["domain_ok"].iloc[0])

    def test_valid_chainsaw_two_entries_independent(self, tmp_path):
        _write_fake_pdb(tmp_path, "P1")
        _write_fake_pdb(tmp_path, "P2")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS,
                   side_effect=["1-30", "1-10,12-20"]):
            df_out = stp.get_domains(df_seq=_df_two(),
                                     pdb_folder=str(tmp_path),
                                     tool="chainsaw",
                                     chainsaw_path="/fake")
        assert df_out["chopping"].tolist() == ["1-30", "1-10,12-20"]
        assert df_out["domain_ok"].tolist() == [True, True]

    def test_valid_chainsaw_then_encode_domains(self, tmp_path):
        # End-to-end through ChainSaw mock + encode_domains.
        _write_fake_pdb(tmp_path, "P1")
        stp = aa.StructurePreprocessor(verbose=False)
        with patch(RESOLVE_CS, return_value=Path("/fake")), \
             patch(RUN_CS, return_value="1-10,15-25"):
            df_out = stp.get_domains(df_seq=_df_one(),
                                     pdb_folder=str(tmp_path),
                                     tool="chainsaw",
                                     chainsaw_path="/fake")
            d, df_aug = stp.encode_domains(return_df=True, 
                df_seq=df_out,
                features=["domain_boundary",
                          "n_domains_in_protein"])
        assert d["P1"].shape == (30, 2)
        np.testing.assert_allclose(d["P1"][0, 1], 0.2)  # 2 domains / 10


class TestEncodeDomainsInlineChoppingColumn:
    """encode_domains' new in-memory chopping path (v1.2 commit 4)."""

    def test_inline_chopping_used_when_present(self):
        df = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["A" * 30],
            "chopping": ["1-10,15-25"],
            "domain_ok": [True],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        # No domain_folder needed when 'chopping' column present.
        d = stp.encode_domains(df_seq=df,
                                  features=["domain_boundary"])
        v = d["P1"][:, 0]
        for endpoint_1based in (1, 10, 15, 25):
            assert v[endpoint_1based - 1] == 1.0

    def test_inline_chopping_missing_folder_ok(self):
        # When 'chopping' column is present, domain_folder is not required.
        df = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["A" * 30],
            "chopping": ["1-30"],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_domains(df_seq=df, domain_folder=None,
                                  features=["domain_boundary"])
        assert d["P1"].shape == (30, 1)

    def test_inline_chopping_malformed_warns(self):
        df = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["A" * 30],
            "chopping": ["not_a_chopping!!!"],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_aug = stp.encode_domains(return_df=True, 
                df_seq=df, features=["domain_boundary"])
        assert np.isnan(d["P1"]).all()
        assert not bool(df_aug["domain_ok"].iloc[0])

    def test_inline_chopping_empty_string_treated_as_no_domains(self):
        df = pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["A" * 30],
            "chopping": [""],
        })
        stp = aa.StructurePreprocessor(verbose=False)
        d, df_aug = stp.encode_domains(return_df=True, 
            df_seq=df, features=["domain_boundary"])
        # Empty chopping → 0 domains → all residues unassigned (NaN).
        assert np.isnan(d["P1"]).all()
        # Still OK because parsing succeeded (just no domains).
        assert bool(df_aug["domain_ok"].iloc[0])
