"""This is a script to test the AlphaFold download backend in
``data_handling_pro/_backend/struct_preproc/_alphafold.py``
(fetch_af_file / fetch_alphafold_bulk and the URL/filename helpers).

The network endpoint is never hit by the suite; the backend's ``http_get_``
transport seam is mocked.
A parity test asserts that the filenames written by the backend are exactly
the ones the StructurePreprocessor file resolvers find, so a download feeds
``encode_pdb`` / ``encode_pae`` / ``get_dssp`` with no glue.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

from aaanalysis.data_handling_pro._backend.struct_preproc import _alphafold as af
from aaanalysis.data_handling_pro._backend.struct_preproc._file_format import (
    resolve_structure_path, resolve_pae_path)

MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc._alphafold"


# I Helper Functions
def _resp(status=200, content=b"DATA"):
    m = MagicMock()
    m.status_code = status
    m.content = content
    return m


def _seq_responses(*statuses):
    """A side_effect list of mocked responses for the given status codes."""
    return [_resp(s) for s in statuses]


# II Test Classes
def _api_resp(status=200,
              pdb_url="https://alphafold.ebi.ac.uk/files/AF-P1-F1-model_v6.pdb",
              pae_url="https://alphafold.ebi.ac.uk/files/AF-P1-F1-predicted_aligned_error_v6.json",
              records="default"):
    """Mock an AlphaFold prediction-API response (carries .json())."""
    m = MagicMock()
    m.status_code = status
    if records == "default":
        records = [{"pdbUrl": pdb_url, "cifUrl": pdb_url.replace(".pdb", ".cif"),
                    "paeDocUrl": pae_url}] if status == 200 else None
    m.json.return_value = records
    return m


class TestUrlAndFilenameHelpers:
    """Local filename helpers (download URLs are resolved via the API now)."""

    def test_model_filename_pdb(self):
        assert af._af_model_filename("P1", "pdb") == "P1.pdb"

    def test_model_filename_cif(self):
        assert af._af_model_filename("P1", "cif") == "P1.cif"


class TestResolveUrls:
    """_af_resolve_urls: version-agnostic URL resolution via the API.

    This is the regression guard for the v4 -> v6 breakage: it pins the API
    fields the resolver reads, so a code-side change is caught networklessly
    (the live-endpoint guard is the network test in tests/integration/)."""

    def test_valid_returns_pdb_and_pae_urls(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp()):
            model_url, pae_url = af._af_resolve_urls("P1", "pdb", 5.0)
        assert model_url.endswith("model_v6.pdb")
        assert pae_url.endswith("predicted_aligned_error_v6.json")

    def test_valid_cif_uses_cif_url(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp()):
            model_url, _ = af._af_resolve_urls("P1", "cif", 5.0)
        assert model_url.endswith("model_v6.cif")

    def test_valid_404_returns_none(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp(404)):
            assert af._af_resolve_urls("P1", "pdb", 5.0) is None

    def test_valid_400_returns_none(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp(400)):
            assert af._af_resolve_urls("BADACC", "pdb", 5.0) is None

    def test_valid_empty_records_returns_none(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp(records=[])):
            assert af._af_resolve_urls("P1", "pdb", 5.0) is None

    def test_valid_no_pae_doc_url(self):
        with patch(f"{MODULE}.http_get_",
                   return_value=_api_resp(records=[{"pdbUrl": "http://m.pdb"}])):
            model_url, pae_url = af._af_resolve_urls("P1", "pdb", 5.0)
        assert model_url == "http://m.pdb" and pae_url is None

    def test_invalid_500_raises(self):
        with patch(f"{MODULE}.http_get_", return_value=_api_resp(500)):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                af._af_resolve_urls("P1", "pdb", 5.0)

    def test_invalid_missing_model_key_raises(self):
        with patch(f"{MODULE}.http_get_",
                   return_value=_api_resp(records=[{"paeDocUrl": "x"}])):
            with pytest.raises(RuntimeError, match="no 'pdbUrl'"):
                af._af_resolve_urls("P1", "pdb", 5.0)

    def test_invalid_transport_error_raises(self):
        with patch(f"{MODULE}.http_get_",
                   side_effect=requests.RequestException("boom")):
            with pytest.raises(RuntimeError, match="failed"):
                af._af_resolve_urls("P1", "pdb", 5.0)


class TestFetchAfFile:
    """fetch_af_file: 200 writes, 404 soft, other status / transport raise."""

    def test_valid_200_writes_and_returns_true(self, tmp_path):
        dest = tmp_path / "P1.pdb"
        with patch(f"{MODULE}.http_get_", return_value=_resp(200, b"XYZ")):
            ok = af.fetch_af_file("http://x", dest, timeout=5.0)
        assert ok is True
        assert dest.read_bytes() == b"XYZ"

    def test_valid_404_returns_false_no_file(self, tmp_path):
        dest = tmp_path / "P1.pdb"
        with patch(f"{MODULE}.http_get_", return_value=_resp(404)):
            ok = af.fetch_af_file("http://x", dest)
        assert ok is False
        assert not dest.exists()

    def test_valid_passes_timeout(self, tmp_path):
        dest = tmp_path / "P1.pdb"
        with patch(f"{MODULE}.http_get_",
                   return_value=_resp(200)) as mg:
            af.fetch_af_file("http://x", dest, timeout=12.0)
        assert mg.call_args.kwargs["timeout"] == 12.0

    def test_valid_atomic_no_part_left(self, tmp_path):
        dest = tmp_path / "P1.pdb"
        with patch(f"{MODULE}.http_get_", return_value=_resp(200)):
            af.fetch_af_file("http://x", dest)
        assert not (tmp_path / "P1.pdb.part").exists()

    def test_invalid_500_raises(self, tmp_path):
        with patch(f"{MODULE}.http_get_", return_value=_resp(500)):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                af.fetch_af_file("http://x", tmp_path / "P1.pdb")

    def test_invalid_transport_error_raises(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=requests.RequestException("boom")):
            with pytest.raises(RuntimeError, match="failed"):
                af.fetch_af_file("http://x", tmp_path / "P1.pdb")


class TestFetchAlphafoldBulk:
    """fetch_alphafold_bulk: single/multiple/skip/partial/mixed/verbose.

    URL resolution is stubbed here so these exercise the download/skip/404-file
    logic; _af_resolve_urls itself is covered by TestResolveUrls."""

    @pytest.fixture(autouse=True)
    def _stub_resolve_urls(self):
        with patch(f"{MODULE}._af_resolve_urls",
                   return_value=("https://af/model", "https://af/pae")):
            yield

    def test_valid_accession_not_in_afdb(self, tmp_path):
        # API resolves to None (404/400) -> both files not-ok, no crash.
        with patch(f"{MODULE}._af_resolve_urls", return_value=None):
            with pytest.warns(UserWarning):
                df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                             30.0, True, False)
        assert not bool(df.iloc[0]["alphafold_ok"])
        assert not bool(df.iloc[0]["model_ok"]) and not bool(df.iloc[0]["pae_ok"])

    def test_valid_single_entry_both_ok(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        row = df.iloc[0]
        assert row["model_ok"] and row["pae_ok"] and row["alphafold_ok"]
        assert (tmp_path / "P1.pdb").is_file()
        assert (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").is_file()

    def test_valid_multiple_entries(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200, 200, 200)):
            df = af.fetch_alphafold_bulk(["P1", "P2"], tmp_path, "pdb",
                                         30.0, True, False)
        assert len(df) == 2
        assert df["alphafold_ok"].all()

    def test_valid_cif_format(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            af.fetch_alphafold_bulk(["P1"], tmp_path, "cif", 30.0, True, False)
        assert (tmp_path / "P1.cif").is_file()

    def test_valid_skip_existing_skips(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").write_text("{}")
        with patch(f"{MODULE}.http_get_") as mg:
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        mg.assert_not_called()
        assert bool(df.iloc[0]["skipped"]) is True
        assert bool(df.iloc[0]["alphafold_ok"]) is True

    def test_valid_partial_refetches_only_missing(self, tmp_path):
        # Model already present, PAE missing -> only one GET (for PAE).
        (tmp_path / "P1.pdb").write_text("x")
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200)) as mg:
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        assert mg.call_count == 1
        assert df.iloc[0]["alphafold_ok"]

    def test_valid_skip_existing_false_always_downloads(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").write_text("{}")
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)) as mg:
            af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                    30.0, False, False)
        assert mg.call_count == 2

    def test_valid_mixed_success_and_404(self, tmp_path):
        # P1 ok; P2 model ok, PAE 404.
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200, 200, 404)):
            with pytest.warns(UserWarning, match="PAE for 'P2'"):
                df = af.fetch_alphafold_bulk(["P1", "P2"], tmp_path, "pdb",
                                             30.0, True, False)
        assert bool(df.iloc[0]["alphafold_ok"]) is True
        assert bool(df.iloc[1]["alphafold_ok"]) is False
        assert bool(df.iloc[1]["model_ok"]) is True
        assert df.iloc[1]["pae_path"] == ""

    def test_valid_404_model_warns(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(404, 200)):
            with pytest.warns(UserWarning, match="model for 'P1'"):
                df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                             30.0, True, False)
        assert bool(df.iloc[0]["model_ok"]) is False
        assert df.iloc[0]["model_path"] == ""

    def test_valid_verbose_prints_entry(self, tmp_path, capsys):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb", 30.0, True, True)
        assert "P1" in capsys.readouterr().out

    def test_valid_columns_schema(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        assert list(df.columns) == af.COLS_AF_STATUS

    def test_invalid_network_error_propagates(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=requests.RequestException("down")):
            with pytest.raises(RuntimeError):
                af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                        30.0, True, False)


class TestFetchAlphafoldBulkComplex:
    """Cross-cutting: parity with the resolvers + path bookkeeping."""

    @pytest.fixture(autouse=True)
    def _stub_resolve_urls(self):
        with patch(f"{MODULE}._af_resolve_urls",
                   return_value=("https://af/model", "https://af/pae")):
            yield

    def test_written_names_resolve_via_resolvers(self, tmp_path):
        # The no-glue contract: backend-written filenames are exactly what
        # encode_pdb / encode_pae find.
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            af.fetch_alphafold_bulk(["P12345"], tmp_path, "pdb",
                                    30.0, True, False)
        struct_path, fmt = resolve_structure_path(tmp_path, "P12345")
        pae_path = resolve_pae_path(tmp_path, "P12345")
        assert struct_path is not None and fmt == "pdb"
        assert pae_path is not None

    def test_cif_names_resolve(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            af.fetch_alphafold_bulk(["P9"], tmp_path, "cif", 30.0, True, False)
        struct_path, fmt = resolve_structure_path(tmp_path, "P9")
        assert struct_path is not None and fmt == "cif"

    def test_status_paths_point_at_real_files(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        assert Path(df.iloc[0]["model_path"]).is_file()
        assert Path(df.iloc[0]["pae_path"]).is_file()

    def test_skipped_false_when_downloaded(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        assert bool(df.iloc[0]["skipped"]) is False

    def test_returns_dataframe(self, tmp_path):
        with patch(f"{MODULE}.http_get_",
                   side_effect=_seq_responses(200, 200)):
            df = af.fetch_alphafold_bulk(["P1"], tmp_path, "pdb",
                                         30.0, True, False)
        assert isinstance(df, pd.DataFrame)
