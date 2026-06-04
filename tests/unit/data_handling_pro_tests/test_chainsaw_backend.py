"""This is a script to test the ChainSaw backend wrapper functions
(resolve_chainsaw_path / _parse_chainsaw_tsv / run_chainsaw_on_entry).

These are backend helpers under ``data_handling_pro/_backend/struct_preproc``.
ChainSaw is not on PyPI (GPL-3, user-cloned), so the subprocess that shells into
its ``get_predictions.py`` is never invoked by the suite. To cover the wrapper's
parsing + error-handling internals we test the backend functions directly and
mock ``subprocess.run`` — a deliberate, narrow exception to the otherwise
frontend-driven testing convention (the binary-dependent lines are unreachable
from the StructurePreprocessor frontend without a real ChainSaw clone).
"""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from aaanalysis.data_handling_pro._backend.struct_preproc._chainsaw import (
    resolve_chainsaw_path,
    run_chainsaw_on_entry,
    _parse_chainsaw_tsv,
    _chainsaw_entry_script,
)

MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc._chainsaw"


# I Helper Functions
def _make_chainsaw_clone(root):
    """Create a minimal valid ChainSaw clone dir (with get_predictions.py)."""
    (root / "get_predictions.py").write_text("# fake entry script\n")
    return root


def _proc(returncode=0, stdout="", stderr=""):
    """Build a stand-in for the subprocess.CompletedProcess return value."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


TSV_OK = "chain_id\tndom\tchopping\nP1\t2\t1-50,55-120\n"


# II Test Classes
class TestResolveChainsawPath:
    """resolve_chainsaw_path: one positive + one negative per input shape."""

    # ----- POSITIVES -----
    def test_valid_returns_path(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        out = resolve_chainsaw_path(str(tmp_path))
        assert out == tmp_path

    def test_valid_accepts_path_object(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        out = resolve_chainsaw_path(tmp_path)
        assert out.is_dir()

    def test_valid_entry_script_under_root(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        out = resolve_chainsaw_path(tmp_path)
        assert _chainsaw_entry_script(out).is_file()

    # ----- NEGATIVES -----
    def test_invalid_none(self):
        with pytest.raises(RuntimeError, match="required"):
            resolve_chainsaw_path(None)

    def test_invalid_not_a_directory(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("hi")
        with pytest.raises(RuntimeError, match="not a directory"):
            resolve_chainsaw_path(str(f))

    def test_invalid_nonexistent_path(self):
        with pytest.raises(RuntimeError, match="not a directory"):
            resolve_chainsaw_path("/no/such/__chainsaw__")

    def test_invalid_dir_without_script(self, tmp_path):
        with pytest.raises(RuntimeError, match="get_predictions.py"):
            resolve_chainsaw_path(str(tmp_path))


class TestParseChainsawTsv:
    """_parse_chainsaw_tsv: one positive + one negative per parse branch."""

    # ----- POSITIVES -----
    def test_valid_extracts_chopping(self):
        assert _parse_chainsaw_tsv(TSV_OK) == "1-50,55-120"

    def test_valid_strips_whitespace(self):
        tsv = "a\tchopping\nx\t  1-10,20-30  \n"
        assert _parse_chainsaw_tsv(tsv) == "1-10,20-30"

    def test_valid_blank_lines_skipped(self):
        tsv = "\n\nchain_id\tchopping\n\nP1\t1-9\n\n"
        assert _parse_chainsaw_tsv(tsv) == "1-9"

    def test_valid_chopping_first_column(self):
        tsv = "chopping\tndom\n1-5_8-12\t2\n"
        assert _parse_chainsaw_tsv(tsv) == "1-5_8-12"

    def test_valid_empty_chopping_cell(self):
        tsv = "chain_id\tchopping\nP1\t\n"
        assert _parse_chainsaw_tsv(tsv) == ""

    # ----- NEGATIVES -----
    def test_invalid_empty_output(self):
        with pytest.raises(RuntimeError, match="no output"):
            _parse_chainsaw_tsv("")

    def test_invalid_whitespace_only(self):
        with pytest.raises(RuntimeError, match="no output"):
            _parse_chainsaw_tsv("   \n\n  \n")

    def test_invalid_missing_chopping_column(self):
        with pytest.raises(RuntimeError, match="missing 'chopping'"):
            _parse_chainsaw_tsv("chain_id\tndom\nP1\t2\n")

    def test_invalid_header_only_no_data(self):
        with pytest.raises(RuntimeError, match="no data row"):
            _parse_chainsaw_tsv("chain_id\tchopping\n")

    def test_invalid_short_data_row(self):
        # 'chopping' is at index 2 but the data row has only 1 cell.
        with pytest.raises(RuntimeError, match="fewer columns"):
            _parse_chainsaw_tsv("a\tb\tchopping\nonlyone\n")


class TestRunChainsawOnEntry:
    """run_chainsaw_on_entry: subprocess success + every failure branch."""

    # ----- POSITIVES -----
    def test_valid_returns_chopping(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run", return_value=_proc(0, TSV_OK)):
            out = run_chainsaw_on_entry(pdb, str(tmp_path))
        assert out == "1-50,55-120"

    def test_valid_invokes_entry_script(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(0, TSV_OK)) as mrun:
            run_chainsaw_on_entry(pdb, str(tmp_path))
        cmd = mrun.call_args[0][0]
        assert cmd[1].endswith("get_predictions.py")
        assert "--structure_file" in cmd

    def test_valid_empty_chopping_returned(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(0, "chain_id\tchopping\nP1\t\n")):
            out = run_chainsaw_on_entry(pdb, str(tmp_path))
        assert out == ""

    # ----- NEGATIVES -----
    def test_invalid_nonzero_exit(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(1, "", "boom")):
            with pytest.raises(RuntimeError, match="failed"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_invalid_timeout(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        err = subprocess.TimeoutExpired(cmd="cs", timeout=300)
        with patch(f"{MODULE}.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="timed out"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_invalid_binary_not_found(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   side_effect=FileNotFoundError("no python")):
            with pytest.raises(RuntimeError, match="Failed to invoke"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_invalid_bad_path_propagates(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        # chainsaw_path None -> resolve_chainsaw_path raises before subprocess.
        with pytest.raises(RuntimeError, match="required"):
            run_chainsaw_on_entry(pdb, None)

    def test_invalid_unparseable_stdout(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(0, "no_chopping_here\nrow\n")):
            with pytest.raises(RuntimeError, match="missing 'chopping'"):
                run_chainsaw_on_entry(pdb, str(tmp_path))


class TestChainsawComplex:
    """Cross-cutting combinations across resolve + run + parse."""

    def test_complex_stderr_preferred_in_error(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(2, "some stdout", "STDERR_MARKER")):
            with pytest.raises(RuntimeError, match="STDERR_MARKER"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_complex_falls_back_to_stdout_when_no_stderr(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(2, "STDOUT_MARKER", "")):
            with pytest.raises(RuntimeError, match="STDOUT_MARKER"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_complex_discontinuous_chopping_roundtrip(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        tsv = "chain_id\tchopping\nP1\t1-50,55-120_125-200\n"
        with patch(f"{MODULE}.subprocess.run", return_value=_proc(0, tsv)):
            out = run_chainsaw_on_entry(pdb, str(tmp_path))
        assert out == "1-50,55-120_125-200"

    def test_complex_runs_in_chainsaw_cwd(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run",
                   return_value=_proc(0, TSV_OK)) as mrun:
            run_chainsaw_on_entry(pdb, str(tmp_path))
        assert mrun.call_args.kwargs["cwd"] == str(tmp_path)

    def test_complex_timeout_message_includes_seconds(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        err = subprocess.TimeoutExpired(cmd="cs", timeout=300)
        with patch(f"{MODULE}.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="300"):
                run_chainsaw_on_entry(pdb, str(tmp_path))

    def test_complex_resolve_then_run_consistent(self, tmp_path):
        _make_chainsaw_clone(tmp_path)
        resolved = resolve_chainsaw_path(str(tmp_path))
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        with patch(f"{MODULE}.subprocess.run", return_value=_proc(0, TSV_OK)):
            out = run_chainsaw_on_entry(pdb, resolved)
        assert out == "1-50,55-120"
