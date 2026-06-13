"""Equivalence test for the encode_pdb alignment cache: every encoder must
produce byte-identical output whether the per-(target, atom) global alignment
is recomputed each call (the original behaviour) or served from the shared
``lru_cache``. The cache keys only on the two sequences and stores
``aligner.align(...)[0]`` (the deterministic first optimal alignment), so a
primed cache must never change a single value."""
from pathlib import Path

import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc import encode_pdb as ep
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb import (
    load_structure, _ca_coords_and_residues)

aa.options["verbose"] = False
PDB_FIXTURES = Path(__file__).resolve().parents[3] / "aaanalysis" / "_data" / "pdb_test"

# Encoders that align target vs atom_seq with no external binary / heavy
# biopython surface tool (depth=msms, hse=HSExposure, chi=internal coords are
# excluded — the suite never runs those). Each routes through
# ``_pick_best_chain_records`` (-> ``_identity_fraction``) and
# ``_align_atom_values_to_target``, i.e. the two callers of the cache.
ENCODERS = [
    ("bfactor", ep.encode_bfactor),
    ("plddt", ep.encode_plddt),
    ("plddt_disorder", ep.encode_plddt_disorder),
    ("plddt_tier", ep.encode_plddt_tier),
    ("ca_centroid_dist", ep.encode_ca_centroid_dist),
    ("ca_centroid_dist_norm", ep.encode_ca_centroid_dist_norm),
    ("contact_count_8A", ep.encode_contact_count_8A),
    ("contact_count_12A", ep.encode_contact_count_12A),
    ("disulfide", ep.encode_disulfide),
]


def _fresh_alignment_strings(target_seq, atom_seq):
    """Recompute the aligned (target, atom) string pair with NO cache — the
    reference for the original per-call alignment behaviour."""
    aligner = ep._make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    return str(alignment[0]), str(alignment[1])


@pytest.mark.parametrize("pdb,seq", [
    ("P1.pdb", "ACDEFGHIKLMNPQRS"),
    ("P2.pdb", "VLIMKRSTGADE"),
    ("AF_TINY.pdb", None),  # use the structure's own CA sequence
])
class TestAlignmentCacheEquivalence:
    def _resolve_seq(self, structure, seq):
        if seq is not None:
            return seq
        _c, _r, atom_seq, _id = _ca_coords_and_residues(structure, "A")
        return atom_seq

    def test_cached_strings_match_uncached(self, pdb, seq):
        """The cached alignment strings equal a fresh, uncached recompute."""
        structure = load_structure(PDB_FIXTURES / pdb)
        seq = self._resolve_seq(structure, seq)
        _c, _r, atom_seq, _id = _ca_coords_and_residues(structure, seq)
        ep._alignment_strings.cache_clear()
        cached = ep._alignment_strings(seq, atom_seq)
        assert cached == _fresh_alignment_strings(seq, atom_seq)

    def test_encoders_identical_with_vs_without_cache(self, monkeypatch, pdb, seq):
        """Every encoder is byte-identical: cache bypassed vs. shared cache."""
        structure = load_structure(PDB_FIXTURES / pdb)
        seq = self._resolve_seq(structure, seq)

        # Reference: bypass the lru_cache entirely (recompute every call).
        monkeypatch.setattr(ep, "_alignment_strings", _fresh_alignment_strings)
        ref = {name: enc(structure, seq) for name, enc in ENCODERS}
        monkeypatch.undo()

        # Real path: shared lru_cache, primed across encoders within the entry.
        ep._alignment_strings.cache_clear()
        got = {name: enc(structure, seq) for name, enc in ENCODERS}

        for name, _enc in ENCODERS:
            np.testing.assert_array_equal(got[name][0], ref[name][0])
            assert got[name][1] == ref[name][1]

    def test_priming_cache_does_not_change_output(self, pdb, seq):
        """A cold first pass and a primed second pass yield identical output."""
        structure = load_structure(PDB_FIXTURES / pdb)
        seq = self._resolve_seq(structure, seq)
        ep._alignment_strings.cache_clear()
        cold = {name: enc(structure, seq) for name, enc in ENCODERS}
        primed = {name: enc(structure, seq) for name, enc in ENCODERS}  # cache warm
        for name, _enc in ENCODERS:
            np.testing.assert_array_equal(cold[name][0], primed[name][0])
            assert cold[name][1] == primed[name][1]


def test_cache_does_not_cross_contaminate_distinct_inputs():
    """Distinct (target, atom) keys return their own correct alignment — a
    shared cache must not leak one pair's strings into another's."""
    ep._alignment_strings.cache_clear()
    pairs = [("ACDEFGHIK", "ACDEFGHIK"),
             ("ACDEFGHIK", "ACDFGHIK"),
             ("VLIMKRST", "VLIMKRST"),
             ("VLIMKRST", "VLIM")]
    for target, atom in pairs:  # prime in one order
        ep._alignment_strings(target, atom)
    for target, atom in pairs:  # each key still matches its own recompute
        assert ep._alignment_strings(target, atom) == _fresh_alignment_strings(target, atom)
