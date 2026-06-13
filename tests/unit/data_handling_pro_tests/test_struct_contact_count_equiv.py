"""Equivalence test: the vectorized CA-CA contact-count must equal the original
O(n^2) per-pair loop, byte-for-byte, on the bundled PDB fixtures."""
from pathlib import Path

import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb import (
    load_structure, _ca_coords_and_residues, _align_atom_values_to_target,
    encode_contact_count_8A, encode_contact_count_12A)
from aaanalysis.data_handling_pro._backend.struct_preproc.feature_registry import normalize

aa.options["verbose"] = False
PDB_FIXTURES = Path(__file__).resolve().parents[3] / "aaanalysis" / "_data" / "pdb_test"


def _ref_contact(structure, sequence, radius_A, min_seq_sep, feature_key):
    """Original per-pair O(n^2) implementation (reference)."""
    coords, _res, atom_seq, identity = _ca_coords_and_residues(structure, sequence)
    if coords is None or len(coords) == 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    n = len(coords)
    counts = np.zeros(n, dtype=np.float64)
    has = ~np.isnan(coords).any(axis=1)
    for i in range(n):
        if not has[i]:
            counts[i] = np.nan
            continue
        ci = coords[i]
        for j in range(n):
            if j == i or abs(j - i) < min_seq_sep or not has[j]:
                continue
            if float(np.linalg.norm(ci - coords[j])) <= radius_A:
                counts[i] += 1
    aligned = _align_atom_values_to_target(sequence, atom_seq, counts.tolist())
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize(feature_key, raw), identity


@pytest.mark.parametrize("pdb,seq", [
    ("P1.pdb", "ACDEFGHIKLMNPQRS"),
    ("P2.pdb", "VLIMKRSTGADE"),
    ("AF_TINY.pdb", None),  # use the structure's own CA sequence
])
class TestContactCountEquivalence:
    def test_matches_original_loop(self, pdb, seq):
        structure = load_structure(PDB_FIXTURES / pdb)
        if seq is None:
            _c, _r, atom_seq, _id = _ca_coords_and_residues(structure, "A")
            seq = atom_seq
        for enc, radius, key in [(encode_contact_count_8A, 8.0, "contact_count_8A"),
                                 (encode_contact_count_12A, 12.0, "contact_count_12A")]:
            new_arr, new_id = enc(structure, seq)
            ref_arr, ref_id = _ref_contact(structure, seq, radius, 5, key)
            np.testing.assert_array_equal(new_arr, ref_arr)
            assert new_id == ref_id
