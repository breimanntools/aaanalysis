"""This is a script to test StructurePreprocessor.fetch_alphafold()."""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

import aaanalysis as aa

aa.options["verbose"] = False

# http_get_ (the pooled transport seam) lives in the download backend; patch it
# there (no network).
BACKEND = "aaanalysis.data_handling_pro._backend.struct_preproc._alphafold"


# I Helper Functions
def _resp(status=200, content=b"DATA"):
    m = MagicMock()
    m.status_code = status
    m.content = content
    return m


def _responses(*statuses):
    return [_resp(s) for s in statuses]


def _df(entries=("P1",)):
    return pd.DataFrame({
        "entry": list(entries),
        "sequence": ["ACDEFGHIK"] * len(entries),
    })


def _patch_get(*statuses):
    return patch(f"{BACKEND}.http_get_", side_effect=_responses(*statuses))


@pytest.fixture(autouse=True)
def _stub_af_resolve_urls():
    """Resolve URLs without the network so the ``http_get_`` mocks below
    drive only the two file downloads (model + PAE). The resolver itself is
    unit-tested in test_alphafold_backend.py::TestResolveUrls and live-checked
    by the network test."""
    with patch(f"{BACKEND}._af_resolve_urls",
               return_value=("https://af/model", "https://af/pae")):
        yield


# II Test Classes
class TestFetchAlphafold:
    """Normal cases and per-parameter validation (one parameter per test)."""

    def test_valid_default_returns_status(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            out = strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path))
        assert isinstance(out, pd.DataFrame)
        assert bool(out.iloc[0]["alphafold_ok"]) is True

    def test_valid_status_columns(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            out = strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path))
        assert list(out.columns) == [
            "entry", "model_ok", "pae_ok", "alphafold_ok", "skipped",
            "model_path", "pae_path"]

    def test_valid_file_format_pdb(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                file_format="pdb")
        assert (tmp_path / "P1.pdb").is_file()

    def test_valid_file_format_cif(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                file_format="cif")
        assert (tmp_path / "P1.cif").is_file()

    def test_valid_timeout_passed_through(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{BACKEND}.http_get_",
                   side_effect=_responses(200, 200)) as mg:
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                timeout=7.0)
        assert mg.call_args.kwargs["timeout"] == 7.0

    def test_valid_skip_existing_true(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").write_text("{}")
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{BACKEND}.http_get_") as mg:
            out = strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                      skip_existing=True)
        mg.assert_not_called()
        assert bool(out.iloc[0]["skipped"]) is True

    def test_valid_skip_existing_false(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").write_text("{}")
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{BACKEND}.http_get_",
                   side_effect=_responses(200, 200)) as mg:
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                skip_existing=False)
        assert mg.call_count == 2

    def test_valid_on_failure_nan_keeps_row(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 404):
            with pytest.warns(UserWarning):
                out = strp.fetch_alphafold(df_seq=_df(),
                                          out_folder=str(tmp_path),
                                          on_failure="nan")
        assert len(out) == 1
        assert bool(out.iloc[0]["alphafold_ok"]) is False

    def test_valid_return_df_tuple(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            out = strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                      return_df=True)
        assert isinstance(out, tuple) and len(out) == 2
        assert "alphafold_ok" in out[1].columns

    def test_valid_bulk_multiple_entries(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 200):
            out = strp.fetch_alphafold(df_seq=_df(["P1", "P2"]),
                                      out_folder=str(tmp_path))
        assert len(out) == 2 and out["alphafold_ok"].all()

    def test_valid_creates_missing_leaf_folder(self, tmp_path):
        target = tmp_path / "new_leaf"
        assert not target.exists()
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(target))
        assert target.is_dir()

    def test_valid_verbose_prints(self, tmp_path, capsys):
        strp = aa.StructurePreprocessor(verbose=True)
        with _patch_get(200, 200):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path))
        assert "P1" in capsys.readouterr().out

    def test_invalid_df_seq_none(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=None, out_folder=str(tmp_path))

    def test_invalid_df_seq_missing_entry(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        df = pd.DataFrame({"sequence": ["ACDEF"]})
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=df, out_folder=str(tmp_path))

    def test_invalid_out_folder_none(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="out_folder"):
            strp.fetch_alphafold(df_seq=_df(), out_folder=None)

    def test_invalid_out_folder_parent_missing(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        bad = tmp_path / "does_not_exist" / "leaf"
        with pytest.raises(ValueError, match="parent"):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(bad))

    def test_invalid_file_format(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                file_format="mmcif")

    def test_invalid_timeout_too_small(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                timeout=0.0)

    def test_invalid_timeout_type(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                timeout="fast")

    def test_invalid_skip_existing_type(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                skip_existing="yes")

    def test_invalid_return_df_type(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                return_df="yes")

    def test_invalid_on_failure(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                on_failure="ignore")

    def test_invalid_existing_alphafold_ok_column(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        df = _df()
        df["alphafold_ok"] = [True]
        with pytest.raises(ValueError, match="alphafold_ok"):
            strp.fetch_alphafold(df_seq=df, out_folder=str(tmp_path))

    def test_invalid_unsafe_entry(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        df = pd.DataFrame({"entry": ["../escape"], "sequence": ["ACDEF"]})
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=df, out_folder=str(tmp_path))


class TestFetchAlphafoldComplex:
    """Combinations: on_failure interaction with 404s and network errors."""

    def test_valid_mixed_status_per_row(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 404):
            with pytest.warns(UserWarning):
                out = strp.fetch_alphafold(df_seq=_df(["P1", "P2"]),
                                          out_folder=str(tmp_path))
        assert bool(out.iloc[0]["alphafold_ok"]) is True
        assert bool(out.iloc[1]["alphafold_ok"]) is False

    def test_valid_on_failure_drop_removes_status_row(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 404):
            with pytest.warns(UserWarning):
                out = strp.fetch_alphafold(df_seq=_df(["P1", "P2"]),
                                          out_folder=str(tmp_path),
                                          on_failure="drop")
        assert len(out) == 1
        assert list(out["entry"]) == ["P1"]

    def test_valid_on_failure_drop_with_return_df(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 404):
            with pytest.warns(UserWarning):
                df_status, df_seq_out = strp.fetch_alphafold(
                    df_seq=_df(["P1", "P2"]), out_folder=str(tmp_path),
                    on_failure="drop", return_df=True)
        assert len(df_seq_out) == 1
        assert bool(df_seq_out.iloc[0]["alphafold_ok"]) is True

    def test_valid_return_df_nan_marks_failures(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 404):
            with pytest.warns(UserWarning):
                _, df_seq_out = strp.fetch_alphafold(
                    df_seq=_df(["P1", "P2"]), out_folder=str(tmp_path),
                    on_failure="nan", return_df=True)
        assert list(df_seq_out["alphafold_ok"]) == [True, False]

    def test_valid_partial_present_one_get(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{BACKEND}.http_get_",
                   side_effect=_responses(200)) as mg:
            strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path))
        assert mg.call_count == 1

    def test_invalid_on_failure_raise_with_404(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 404):
            with pytest.warns(UserWarning):
                with pytest.raises(RuntimeError, match="fetch_alphafold"):
                    strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                        on_failure="raise")

    def test_invalid_http_500_raises_regardless(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(500):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path),
                                    on_failure="nan")

    def test_invalid_transport_error_raises(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{BACKEND}.http_get_",
                   side_effect=requests.RequestException("down")):
            with pytest.raises(RuntimeError):
                strp.fetch_alphafold(df_seq=_df(), out_folder=str(tmp_path))

    def test_invalid_raise_aborts_bulk(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with _patch_get(200, 200, 200, 404):
            with pytest.warns(UserWarning):
                with pytest.raises(RuntimeError):
                    strp.fetch_alphafold(df_seq=_df(["P1", "P2"]),
                                        out_folder=str(tmp_path),
                                        on_failure="raise")

    def test_invalid_unsafe_entry_in_bulk(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        df = pd.DataFrame({"entry": ["P1", "bad/entry"],
                           "sequence": ["ACDEF", "GHIKL"]})
        with pytest.raises(ValueError):
            strp.fetch_alphafold(df_seq=df, out_folder=str(tmp_path))
