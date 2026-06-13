"""Equivalence test for the encode_pdb shared per-entry chain pick (#201, the
last same-output T1 tail of #186).

Every encoder re-walked the structure to collect amino-acid chain residues
(``_collect_chain_residues``) and rebuilt each chain's atom sequence to pick the
best-matching chain (``_pick_best_chain_records``) — ~13x per entry. That pick is
now computed once via ``_resolve_best_chain`` and threaded into each encoder as
``chain_pick``. The pick is a deterministic function of ``(structure, sequence)``,
so a shared pick must produce byte-identical output to each encoder recomputing
it independently (``chain_pick=None``), which is the original behaviour.

Mirrors ``test_struct_batch6_equiv.py`` section 3 (the pLDDT shared-vs-independent
test) and the alignment-cache equivalence test. Encoders needing an external
binary / heavy biopython surface tool (``depth``=msms, ``hse``=HSExposure,
``chi``=internal coords) are excluded — the suite never runs those.
"""
from pathlib import Path

import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc import encode_pdb as ep
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb import (
    load_structure, _collect_chain_residues, _residue_one_letter,
    _resolve_best_chain)

aa.options["verbose"] = False
PDB_FIXTURES = Path(__file__).resolve().parents[3] / "aaanalysis" / "_data" / "pdb_test"

# No-external-tool encoders (msms / HSExposure / internal coords excluded, as in
# the alignment-cache equivalence suite). Each routes its chain pick through
# ``chain_pick`` -> ``_resolve_best_chain``. The three pLDDT encoders accept the
# pick via ``chain_pick`` only when their own ``plddt`` share is omitted.
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


def _chain_seq(structure):
    """Atom 1-letter sequence of the first chain (perfect-identity target)."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    chains = _collect_chain_residues(structure)
    _c, residues = chains[0]
    return "".join(_residue_one_letter(r, protein_letters_3to1)
                   for r, _ in residues)


@pytest.mark.parametrize("pdb,seq", [
    ("P1.pdb", "ACDEFGHIKLMNPQRS"),
    ("P2.pdb", "VLIMKRSTGADE"),
    ("AF_TINY.pdb", None),    # use the structure's own CA sequence
    ("SS_BOND.pdb", None),    # real disulfide -> non-trivial disulfide path
])
class TestChainPickEquivalence:
    def _resolve_seq(self, structure, seq):
        return seq if seq is not None else _chain_seq(structure)

    def test_encoders_identical_shared_vs_independent(self, pdb, seq):
        """Each encoder is byte-identical whether it resolves the chain pick
        itself (``chain_pick=None``, the original behaviour) or receives one
        shared pick computed once per entry."""
        structure = load_structure(PDB_FIXTURES / pdb)
        seq = self._resolve_seq(structure, seq)
        # Independent: every encoder resolves its own chain pick.
        ind = {name: enc(structure, seq) for name, enc in ENCODERS}
        # Shared: compute the pick once, thread it into every encoder.
        chain_pick = _resolve_best_chain(structure, seq)
        shared = {name: enc(structure, seq, chain_pick=chain_pick)
                  for name, enc in ENCODERS}
        for name, _enc in ENCODERS:
            np.testing.assert_array_equal(shared[name][0], ind[name][0])
            assert shared[name][1] == ind[name][1]

    def test_resolve_best_chain_matches_inline_collect_pick(self, pdb, seq):
        """``_resolve_best_chain`` == the inline collect + pick it replaces."""
        from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb \
            import _pick_best_chain_records
        structure = load_structure(PDB_FIXTURES / pdb)
        seq = self._resolve_seq(structure, seq)
        chains = _collect_chain_residues(structure)
        ref = _pick_best_chain_records(seq, chains) if chains \
            else (None, None, None, 0.0)
        got = _resolve_best_chain(structure, seq)
        assert got[0] is ref[0]              # same chain object
        assert got[2] == ref[2]              # same atom_seq
        assert got[3] == ref[3]              # same identity fraction


def test_resolve_best_chain_no_aa_chains_returns_none():
    """A structure with no amino-acid chains yields (None, None, None, 0.0),
    which every encoder maps to an all-NaN block at identity 0.0 — collapsing
    the former ``not chains`` and ``chain is None`` early-returns into one."""

    class _EmptyStructure:
        def get_models(self):
            return iter(())

    chain, residues, atom_seq, identity = _resolve_best_chain(
        _EmptyStructure(), "ACDEF")
    assert chain is None and residues is None and atom_seq is None
    assert identity == 0.0
    # An encoder handed that pick returns the all-NaN block, identity 0.0.
    arr, ident = ep.encode_bfactor(_EmptyStructure(), "ACDEF",
                                   chain_pick=(None, None, None, 0.0))
    assert arr.shape == (5, 1) and np.isnan(arr).all() and ident == 0.0
