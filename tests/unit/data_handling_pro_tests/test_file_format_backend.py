"""This is a script to test the structure/PAE file-format resolver in
``data_handling_pro/_backend/struct_preproc/_file_format.py``
(resolve_structure_path / resolve_pae_path / _decompress_to_temp).

Pure-Python file I/O: tested directly against ``tmp_path`` fixtures, including
the gzip-decompression and missing-temp_dir error branches.
"""
import gzip

import pytest

from aaanalysis.data_handling_pro._backend.struct_preproc._file_format import (
    resolve_structure_path,
    resolve_pae_path,
    _af_canonical_pae_name,
    _decompress_to_temp,
)


# I Helper Functions
def _write(path, text="data"):
    path.write_text(text)
    return path


def _write_gz(path, text="data"):
    with gzip.open(path, "wb") as f:
        f.write(text.encode())
    return path


# II Test Classes
class TestResolveStructurePath:
    """resolve_structure_path: each extension + gz + missing-temp_dir."""

    # ----- POSITIVES -----
    def test_valid_pdb(self, tmp_path):
        _write(tmp_path / "P1.pdb")
        path, fmt = resolve_structure_path(tmp_path, "P1")
        assert fmt == "pdb" and path.name == "P1.pdb"

    def test_valid_cif(self, tmp_path):
        _write(tmp_path / "P1.cif")
        path, fmt = resolve_structure_path(tmp_path, "P1")
        assert fmt == "cif" and path.name == "P1.cif"

    def test_valid_pdb_gz_decompressed(self, tmp_path):
        _write_gz(tmp_path / "P1.pdb.gz", "ATOM")
        temp = tmp_path / "tmp"
        temp.mkdir()
        path, fmt = resolve_structure_path(tmp_path, "P1", temp_dir=temp)
        assert fmt == "pdb"
        assert path.read_text() == "ATOM"
        assert path.parent == temp

    def test_valid_cif_gz_decompressed(self, tmp_path):
        _write_gz(tmp_path / "P1.cif.gz", "loop_")
        temp = tmp_path / "tmp"
        temp.mkdir()
        path, fmt = resolve_structure_path(tmp_path, "P1", temp_dir=temp)
        assert fmt == "cif"
        assert path.read_text() == "loop_"

    def test_valid_priority_pdb_over_cif(self, tmp_path):
        _write(tmp_path / "P1.pdb")
        _write(tmp_path / "P1.cif")
        path, fmt = resolve_structure_path(tmp_path, "P1")
        assert fmt == "pdb"  # .pdb resolved first

    # ----- NEGATIVES -----
    def test_invalid_no_match_returns_none(self, tmp_path):
        path, fmt = resolve_structure_path(tmp_path, "MISSING")
        assert path is None and fmt is None

    def test_invalid_pdb_gz_without_temp_dir(self, tmp_path):
        _write_gz(tmp_path / "P1.pdb.gz")
        with pytest.raises(RuntimeError, match="temp_dir"):
            resolve_structure_path(tmp_path, "P1")

    def test_invalid_cif_gz_without_temp_dir(self, tmp_path):
        _write_gz(tmp_path / "P1.cif.gz")
        with pytest.raises(RuntimeError, match="temp_dir"):
            resolve_structure_path(tmp_path, "P1")


class TestResolvePaePath:
    """resolve_pae_path: direct json, gz, AF-DB canonical fallback."""

    # ----- POSITIVES -----
    def test_valid_direct_json(self, tmp_path):
        _write(tmp_path / "P1.json")
        out = resolve_pae_path(tmp_path, "P1")
        assert out.name == "P1.json"

    def test_valid_json_gz(self, tmp_path):
        _write_gz(tmp_path / "P1.json.gz", "{}")
        temp = tmp_path / "tmp"
        temp.mkdir()
        out = resolve_pae_path(tmp_path, "P1", temp_dir=temp)
        assert out.read_text() == "{}"

    def test_valid_af_canonical_fallback(self, tmp_path):
        canonical = _af_canonical_pae_name("P12345")
        _write(tmp_path / canonical)
        out = resolve_pae_path(tmp_path, "P12345")
        assert out.name == canonical

    def test_valid_af_canonical_gz(self, tmp_path):
        canonical = _af_canonical_pae_name("P12345")
        _write_gz(tmp_path / f"{canonical}.gz", "{}")
        temp = tmp_path / "tmp"
        temp.mkdir()
        out = resolve_pae_path(tmp_path, "P12345", temp_dir=temp)
        assert out.read_text() == "{}"

    def test_valid_direct_preferred_over_canonical(self, tmp_path):
        _write(tmp_path / "P12345.json", "direct")
        _write(tmp_path / _af_canonical_pae_name("P12345"), "canonical")
        out = resolve_pae_path(tmp_path, "P12345")
        assert out.read_text() == "direct"

    # ----- NEGATIVES -----
    def test_invalid_no_match_none(self, tmp_path):
        assert resolve_pae_path(tmp_path, "MISSING") is None

    def test_invalid_json_gz_without_temp_dir(self, tmp_path):
        _write_gz(tmp_path / "P1.json.gz")
        with pytest.raises(RuntimeError, match="temp_dir"):
            resolve_pae_path(tmp_path, "P1")


class TestFileFormatComplex:
    """Cross-cutting + the _decompress_to_temp helper directly."""

    def test_complex_canonical_name_format(self):
        assert _af_canonical_pae_name("Q9Y6K9") == \
            "AF-Q9Y6K9-F1-predicted_aligned_error_v4.json"

    def test_complex_decompress_suffix_applied(self, tmp_path):
        src = _write_gz(tmp_path / "X.gz", "hi")
        temp = tmp_path / "tmp"
        temp.mkdir()
        out = _decompress_to_temp(src, temp, ".pdb")
        # 'X.gz'.stem == 'X' (no suffix) -> .pdb suffix applied.
        assert out.suffix == ".pdb"
        assert out.read_text() == "hi"

    def test_complex_decompress_keeps_inner_suffix(self, tmp_path):
        src = _write_gz(tmp_path / "P1.pdb.gz", "ATOM")
        temp = tmp_path / "tmp"
        temp.mkdir()
        out = _decompress_to_temp(src, temp, ".pdb")
        assert out.name == "P1.pdb"
        assert out.read_text() == "ATOM"

    def test_complex_structure_then_pae_same_folder(self, tmp_path):
        _write(tmp_path / "P1.pdb")
        _write(tmp_path / "P1.json")
        spath, fmt = resolve_structure_path(tmp_path, "P1")
        ppath = resolve_pae_path(tmp_path, "P1")
        assert fmt == "pdb" and ppath.name == "P1.json"

    def test_complex_string_folder_accepted(self, tmp_path):
        _write(tmp_path / "P1.pdb")
        path, fmt = resolve_structure_path(str(tmp_path), "P1")
        assert fmt == "pdb"

    def test_complex_no_structure_but_pae_exists(self, tmp_path):
        _write(tmp_path / "P1.json")
        spath, fmt = resolve_structure_path(tmp_path, "P1")
        ppath = resolve_pae_path(tmp_path, "P1")
        assert spath is None and ppath is not None
