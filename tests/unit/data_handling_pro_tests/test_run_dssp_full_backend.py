"""This is a script to test the DSSP-runner backend
``run_dssp_full_for_entry_`` (and its helpers _resolve_dssp_binary /
_coerce_float / _residue_one_letter).

This backend function shells into the ``mkdssp``/``dssp`` binary via biopython's
``Bio.PDB.DSSP`` and is never executed by the suite (no DSSP binary on PATH).
To cover its per-residue parsing, relative→absolute ASA conversion, H-bond field
extraction and exception paths we mock ``shutil.which`` and the function-local
``Bio.PDB`` parsers/DSSP with light fakes — a deliberate, narrow exception to the
otherwise frontend-driven testing convention (these lines are unreachable from the
StructurePreprocessor frontend without a real DSSP install).
"""
import contextlib
import math
import sys
from unittest.mock import patch

import pytest

import Bio.PDB
import Bio.PDB.DSSP  # noqa: F401  (registers the submodule in sys.modules)
import Bio.PDB.Polypeptide  # noqa: F401

from aaanalysis.data_handling_pro._backend.struct_preproc.run_dssp_full import (
    run_dssp_full_for_entry_,
    _resolve_dssp_binary,
    _coerce_float,
    _residue_one_letter,
)

MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc.run_dssp_full"

# The runner does function-local ``from Bio.PDB.DSSP import DSSP``. Note that
# ``Bio.PDB.DSSP`` as an *attribute* resolves to the re-exported DSSP *class*
# (Bio/PDB/__init__.py shadows the submodule), so we patch via the real module
# objects fetched from sys.modules instead of attribute access.
_BIO_PDB = sys.modules["Bio.PDB"]
_BIO_DSSP_MOD = sys.modules["Bio.PDB.DSSP"]
_BIO_POLY_MOD = sys.modules["Bio.PDB.Polypeptide"]


# I Helper Functions
class _Res:
    def __init__(self, resname, rid):
        self._resname = resname
        self.id = rid

    def get_resname(self):
        return self._resname


class _Chain:
    def __init__(self, cid, residues):
        self.id = cid
        self._residues = residues

    def __iter__(self):
        return iter(self._residues)


class _Model:
    def __init__(self, chains):
        self._chains = chains

    def __iter__(self):
        return iter(self._chains)


class _Structure:
    def __init__(self, models):
        self._models = models

    def get_models(self):
        return iter(self._models)


def _entry(ss="H", rel_asa=0.5, phi=-60.0, psi=-45.0,
           hbd_off=-4, hbd_en=-2.0, hba_off=4, hba_en=-2.0):
    """A biopython DSSP value tuple (indices 0-9 used by the runner)."""
    return (1, "A", ss, rel_asa, phi, psi, hbd_off, hbd_en, hba_off, hba_en)


class _FakeDSSP:
    def __init__(self, data):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


@contextlib.contextmanager
def _patched(structure, dssp_data, is_aa_fn=None, dssp_raises=None):
    """Patch shutil.which + the function-local Bio.PDB parsers/DSSP."""
    if is_aa_fn is None:
        is_aa_fn = lambda residue, standard=False: residue.get_resname() != "HOH"

    def _make_parser(*a, **k):
        class _P:
            def get_structure(self, name, path):
                return structure
        return _P()

    def _make_dssp(model, path, dssp=None):
        if dssp_raises is not None:
            raise dssp_raises
        return _FakeDSSP(dssp_data)

    with contextlib.ExitStack() as es:
        es.enter_context(patch(f"{MODULE}.shutil.which",
                               side_effect=lambda name: f"/usr/bin/{name}"))
        es.enter_context(patch.object(_BIO_PDB, "PDBParser", _make_parser))
        es.enter_context(patch.object(_BIO_PDB, "MMCIFParser", _make_parser))
        es.enter_context(patch.object(_BIO_DSSP_MOD, "DSSP", _make_dssp))
        es.enter_context(patch.object(_BIO_POLY_MOD, "is_aa", is_aa_fn))
        yield


def _single_ala_chain(n=3, resname="ALA"):
    residues = [_Res(resname, (" ", i + 1, " ")) for i in range(n)]
    structure = _Structure([_Model([_Chain("A", residues)])])
    data = {("A", (" ", i + 1, " ")): _entry() for i in range(n)}
    return structure, data


# II Test Classes
class TestResolveDsspBinary:
    """_resolve_dssp_binary: present (mkdssp/dssp) vs none."""

    def test_valid_mkdssp(self):
        with patch(f"{MODULE}.shutil.which",
                   side_effect=lambda n: "/usr/bin/mkdssp" if n == "mkdssp" else None):
            assert _resolve_dssp_binary() == "mkdssp"

    def test_valid_dssp_fallback(self):
        with patch(f"{MODULE}.shutil.which",
                   side_effect=lambda n: "/usr/bin/dssp" if n == "dssp" else None):
            assert _resolve_dssp_binary() == "dssp"

    def test_invalid_no_binary(self):
        with patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="PATH"):
                _resolve_dssp_binary()


class TestCoerceFloat:
    """_coerce_float: numeric passthrough, NA/None -> NaN."""

    def test_valid_float(self):
        assert _coerce_float(1.5) == 1.5

    def test_valid_int_string(self):
        assert _coerce_float("3") == 3.0

    def test_valid_na_is_nan(self):
        assert math.isnan(_coerce_float("NA"))

    def test_valid_none_is_nan(self):
        assert math.isnan(_coerce_float(None))

    def test_invalid_garbage_is_nan(self):
        assert math.isnan(_coerce_float("not-a-number"))


class TestResidueOneLetter:
    """_residue_one_letter: known 3-letter -> 1-letter, unknown -> X."""

    def test_valid_known(self):
        assert _residue_one_letter(_Res("ALA", 1), {"ALA": "A"}) == "A"

    def test_valid_unknown_is_x(self):
        assert _residue_one_letter(_Res("XYZ", 1), {"ALA": "A"}) == "X"


class TestRunDsspFull:
    """run_dssp_full_for_entry_: parse / convert / error branches."""

    # ----- POSITIVES -----
    def test_valid_returns_one_chain(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=3)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert len(out) == 1
        cid, seq, ss, asa, phi, psi, *_ = out[0]
        assert cid == "A"
        assert seq == "AAA"
        assert ss == ["H", "H", "H"]

    def test_valid_abs_asa_conversion(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        # ALA Sander max ASA is finite (>0); rel 0.5 -> abs = 0.5 * max > 0.
        structure, data = _single_ala_chain(n=2)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        asa = out[0][3]
        assert all(v > 0 for v in asa)

    def test_valid_hbond_fields(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=2)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        # indices 6-9 = donor off/en, acceptor off/en
        assert out[0][6] == [-4, -4]
        assert out[0][7] == [-2.0, -2.0]
        assert out[0][8] == [4, 4]
        assert out[0][9] == [-2.0, -2.0]

    def test_valid_phi_psi_streams(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=2)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert out[0][4] == [-60.0, -60.0]
        assert out[0][5] == [-45.0, -45.0]

    def test_valid_skips_non_aa(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        residues = [_Res("ALA", (" ", 1, " ")),
                    _Res("HOH", ("W", 2, " ")),
                    _Res("ALA", (" ", 3, " "))]
        structure = _Structure([_Model([_Chain("A", residues)])])
        data = {("A", (" ", 1, " ")): _entry(),
                ("A", (" ", 3, " ")): _entry()}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert out[0][1] == "AA"  # HOH dropped

    def test_valid_skips_residue_not_in_records(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        residues = [_Res("ALA", (" ", 1, " ")), _Res("ALA", (" ", 2, " "))]
        structure = _Structure([_Model([_Chain("A", residues)])])
        data = {("A", (" ", 1, " ")): _entry()}  # residue 2 missing
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert out[0][1] == "A"

    def test_valid_unknown_resname_abs_asa_nan(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        residues = [_Res("XYZ", (" ", 1, " "))]
        structure = _Structure([_Model([_Chain("A", residues)])])
        data = {("A", (" ", 1, " ")): _entry()}
        with _patched(structure, data,
                      is_aa_fn=lambda r, standard=False: True):
            out = run_dssp_full_for_entry_(pdb)
        assert out[0][1] == "X"
        assert math.isnan(out[0][3][0])  # no Sander max -> nan

    def test_valid_two_chains(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        c1 = _Chain("A", [_Res("ALA", (" ", 1, " "))])
        c2 = _Chain("B", [_Res("ALA", (" ", 1, " "))])
        structure = _Structure([_Model([c1, c2])])
        data = {("A", (" ", 1, " ")): _entry(),
                ("B", (" ", 1, " ")): _entry()}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert {c[0] for c in out} == {"A", "B"}

    def test_valid_empty_chain_omitted(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        c1 = _Chain("A", [_Res("ALA", (" ", 1, " "))])
        c2 = _Chain("B", [_Res("HOH", ("W", 1, " "))])  # only water
        structure = _Structure([_Model([c1, c2])])
        data = {("A", (" ", 1, " ")): _entry()}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert [c[0] for c in out] == ["A"]  # B yields nothing

    def test_valid_na_asa_becomes_nan(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        residues = [_Res("ALA", (" ", 1, " "))]
        structure = _Structure([_Model([_Chain("A", residues)])])
        data = {("A", (" ", 1, " ")): _entry(rel_asa="NA")}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert math.isnan(out[0][3][0])

    # ----- NEGATIVES -----
    def test_invalid_no_models(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure = _Structure([])  # get_models() empty -> StopIteration
        with _patched(structure, {}):
            with pytest.raises(RuntimeError, match="no models"):
                run_dssp_full_for_entry_(pdb)

    def test_invalid_dssp_failure(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=1)
        with _patched(structure, data, dssp_raises=ValueError("bad pdb")):
            with pytest.raises(RuntimeError, match="DSSP failed"):
                run_dssp_full_for_entry_(pdb)

    def test_invalid_no_binary(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=1)
        # which() -> None makes _resolve_dssp_binary raise before parsing.
        with patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="PATH"):
                run_dssp_full_for_entry_(pdb)


class TestRunDsspFullComplex:
    """Cross-cutting combinations for run_dssp_full_for_entry_."""

    def test_complex_cif_dispatch(self, tmp_path):
        cif = tmp_path / "P1.cif"
        cif.write_text("dummy")
        structure, data = _single_ala_chain(n=2)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(cif)
        assert out[0][1] == "AA"

    def test_complex_ss8_alphabet_preserved(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        residues = [_Res("ALA", (" ", i + 1, " ")) for i in range(3)]
        structure = _Structure([_Model([_Chain("A", residues)])])
        data = {("A", (" ", 1, " ")): _entry(ss="H"),
                ("A", (" ", 2, " ")): _entry(ss="G"),
                ("A", (" ", 3, " ")): _entry(ss=" ")}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert out[0][2] == ["H", "G", " "]

    def test_complex_stream_lengths_consistent(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        structure, data = _single_ala_chain(n=5)
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        rec = out[0]
        lengths = {len(rec[i]) for i in range(2, 10)}
        assert lengths == {5}
        assert len(rec[1]) == 5

    def test_complex_mixed_chain_some_empty(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        c1 = _Chain("A", [_Res("ALA", (" ", 1, " ")),
                          _Res("HOH", ("W", 2, " "))])
        c2 = _Chain("B", [_Res("ALA", (" ", 1, " "))])
        structure = _Structure([_Model([c1, c2])])
        data = {("A", (" ", 1, " ")): _entry(),
                ("B", (" ", 1, " ")): _entry()}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        assert [c[0] for c in out] == ["A", "B"]
        assert out[0][1] == "A"

    def test_complex_first_model_only(self, tmp_path):
        pdb = tmp_path / "P1.pdb"
        pdb.write_text("dummy")
        m1 = _Model([_Chain("A", [_Res("ALA", (" ", 1, " "))])])
        m2 = _Model([_Chain("Z", [_Res("ALA", (" ", 9, " "))])])
        structure = _Structure([m1, m2])
        data = {("A", (" ", 1, " ")): _entry()}
        with _patched(structure, data):
            out = run_dssp_full_for_entry_(pdb)
        # Only the first NMR model is used.
        assert [c[0] for c in out] == ["A"]
