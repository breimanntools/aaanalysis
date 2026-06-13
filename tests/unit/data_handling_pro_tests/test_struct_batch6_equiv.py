"""Equivalence tests for the struct_pro Batch-6 same-output perf wins (#186):

1. ``encode_disulfide`` vectorized SG-SG pairwise distance == the original
   O(n^2) double loop (byte-identical: 2.5 A inclusive boundary, equidistant
   ties, nearest-partner pick).
2. The DSSP session-scoped ``PairwiseAligner`` (reused across entries) ==
   per-entry fresh aligners for ``pick_best_chain_full_`` /
   ``count_mismatches_full_`` / ``align_chain_full_to_sequence_``.
3. The shared ``_plddt_per_residue`` result fed into the three pLDDT encoders
   == each encoder recomputing it independently.

All three replace their originals in place because they are equivalent.
"""
from pathlib import Path

import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb import (
    load_structure, _collect_chain_residues, _residue_one_letter,
    _pick_best_chain_records, _align_atom_values_to_target,
    _plddt_per_residue,
    encode_disulfide, encode_plddt, encode_plddt_disorder, encode_plddt_tier)
from aaanalysis.data_handling_pro._backend.struct_preproc.feature_registry import (
    normalize)
from aaanalysis.data_handling_pro._backend.struct_preproc import align_dssp_full as ad

aa.options["verbose"] = False
PDB_FIXTURES = Path(__file__).resolve().parents[3] / "aaanalysis" / "_data" / "pdb_test"


def _chain_seq(structure):
    """Atom 1-letter sequence of the first chain (perfect-identity target)."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    chains = _collect_chain_residues(structure)
    _c, residues = chains[0]
    return "".join(_residue_one_letter(r, protein_letters_3to1)
                   for r, _ in residues)


# ---------------------------------------------------------------------------
# 1. encode_disulfide: vectorized == original O(n^2) double loop
# ---------------------------------------------------------------------------
def _ref_disulfide(structure, sequence):
    """Original per-pair O(n^2) SG-SG implementation (reference)."""
    chains = _collect_chain_residues(structure)
    if not chains:
        return np.full((len(sequence), 2), np.nan), 0.0
    chain, residues, atom_seq, identity = _pick_best_chain_records(
        sequence, chains)
    if chain is None:
        return np.full((len(sequence), 2), np.nan), 0.0
    sg_coords = []
    is_cys = []
    for residue, _ in residues:
        if residue.get_resname() != "CYS":
            sg_coords.append(None)
            is_cys.append(False)
            continue
        is_cys.append(True)
        sg = None
        for atom in residue.get_atoms():
            if atom.get_name() == "SG":
                sg = np.asarray(atom.get_coord(), dtype=np.float64)
                break
        sg_coords.append(sg)
    n_res = len(residues)
    atom_participates = []
    atom_partner_dist = []
    for i in range(n_res):
        if not is_cys[i] or sg_coords[i] is None:
            atom_participates.append(float("nan"))
            atom_partner_dist.append(float("nan"))
            continue
        best_dist = float("inf")
        for j in range(n_res):
            if i == j or not is_cys[j] or sg_coords[j] is None:
                continue
            d = float(np.linalg.norm(sg_coords[i] - sg_coords[j]))
            if d <= 2.5 and d < best_dist:
                best_dist = d
        if np.isfinite(best_dist):
            atom_participates.append(1.0)
            atom_partner_dist.append(best_dist)
        else:
            atom_participates.append(0.0)
            atom_partner_dist.append(float("nan"))
    aligned_part = _align_atom_values_to_target(
        sequence, atom_seq, atom_participates)
    aligned_dist = _align_atom_values_to_target(
        sequence, atom_seq, atom_partner_dist)
    raw = np.column_stack([np.asarray(aligned_part, dtype=np.float64),
                           np.asarray(aligned_dist, dtype=np.float64)])
    return normalize("disulfide", raw), identity


@pytest.mark.parametrize("pdb,seq", [
    ("SS_BOND.pdb", None),   # has an actual SS bond (bonded CYS pair)
    ("AF_TINY.pdb", None),
    ("P1.pdb", "ACDEFGHIKLMNPQRS"),
])
class TestDisulfideEquivalence:
    def test_matches_original_loop(self, pdb, seq):
        structure = load_structure(PDB_FIXTURES / pdb)
        if seq is None:
            seq = _chain_seq(structure)
        new_arr, new_id = encode_disulfide(structure, seq)
        ref_arr, ref_id = _ref_disulfide(structure, seq)
        # NaN masks identical, finite values identical.
        np.testing.assert_array_equal(np.isnan(new_arr), np.isnan(ref_arr))
        m = ~np.isnan(ref_arr)
        assert np.array_equal(new_arr[m], ref_arr[m])
        assert new_id == ref_id

    def test_ss_bond_detects_a_bond(self, pdb, seq):
        """At least the SS_BOND fixture must register one bonded CYS pair, so
        the equivalence is exercised on a non-trivial (participates=1) path."""
        if pdb != "SS_BOND.pdb":
            pytest.skip("only SS_BOND carries a real disulfide")
        structure = load_structure(PDB_FIXTURES / pdb)
        arr, _id = encode_disulfide(structure, _chain_seq(structure))
        # column 0 = participates; at least two CYS marked 1.0 (a bonded pair)
        participates = arr[:, 0]
        assert np.nansum(participates == 1.0) >= 2


# ---------------------------------------------------------------------------
# 2. DSSP session aligner: reused == per-entry fresh aligners
# ---------------------------------------------------------------------------
class TestDsspSessionAligner:
    # target vs atom sequence pairs: perfect identity + a 3-residue deletion.
    TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    GAPPED = TARGET[:30] + TARGET[33:]

    def _chain_record(self, atom_seq):
        n = len(atom_seq)
        return ("A", atom_seq, ["C"] * n, [0.0] * n, [0.0] * n, [0.0] * n,
                [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n)

    @pytest.mark.parametrize("atom", [TARGET, GAPPED])
    def test_pick_best_chain_session_equals_fresh(self, atom):
        chains = [self._chain_record(atom)]
        sess = ad._make_aligner()
        fresh = ad.pick_best_chain_full_(self.TARGET, chains)
        reused = ad.pick_best_chain_full_(self.TARGET, chains, aligner=sess)
        assert fresh is not None and reused is not None
        assert fresh[0][1] == reused[0][1]
        assert fresh[1] == reused[1]  # identity fraction byte-identical

    @pytest.mark.parametrize("atom", [TARGET, GAPPED])
    def test_count_mismatches_session_equals_fresh(self, atom):
        sess = ad._make_aligner()
        fresh = ad.count_mismatches_full_(self.TARGET, atom)
        reused = ad.count_mismatches_full_(self.TARGET, atom, aligner=sess)
        assert fresh == reused

    @pytest.mark.parametrize("atom", [TARGET, GAPPED])
    def test_align_chain_session_equals_fresh(self, atom):
        rec = self._chain_record(atom)
        args = (self.TARGET,) + rec[1:]
        sess = ad._make_aligner()
        fresh = ad.align_chain_full_to_sequence_(*args)
        reused = ad.align_chain_full_to_sequence_(*args, aligner=sess)
        # eight parallel streams: ss list + seven float lists.
        assert fresh[0] == reused[0]  # ss strings identical
        for f, r in zip(fresh[1:], reused[1:]):
            np.testing.assert_array_equal(
                np.asarray(f, dtype=np.float64),
                np.asarray(r, dtype=np.float64))

    def test_one_session_aligner_serves_many_entries(self):
        """Reusing one aligner across a stream of entries gives, for each entry,
        the same result a fresh aligner would."""
        sess = ad._make_aligner()
        for atom in (self.TARGET, self.GAPPED, self.TARGET, self.GAPPED):
            chains = [self._chain_record(atom)]
            fresh = ad.pick_best_chain_full_(self.TARGET, chains)
            reused = ad.pick_best_chain_full_(self.TARGET, chains, aligner=sess)
            assert fresh[1] == reused[1]
            assert ad.count_mismatches_full_(self.TARGET, atom) == \
                ad.count_mismatches_full_(self.TARGET, atom, aligner=sess)


# ---------------------------------------------------------------------------
# 3. Shared _plddt_per_residue feeds the three pLDDT encoders identically
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pdb,seq", [
    ("AF_TINY.pdb", None),
    ("P1.pdb", "ACDEFGHIKLMNPQRS"),
])
class TestPlddtSharedReuse:
    def test_shared_equals_independent(self, pdb, seq):
        structure = load_structure(PDB_FIXTURES / pdb)
        if seq is None:
            seq = _chain_seq(structure)
        # Independent: each encoder recomputes _plddt_per_residue (plddt=None).
        ind_plddt = encode_plddt(structure, seq)
        ind_dis = encode_plddt_disorder(structure, seq)
        ind_tier = encode_plddt_tier(structure, seq)
        # Shared: compute once, pass into all three.
        shared = _plddt_per_residue(structure, seq)
        sh_plddt = encode_plddt(structure, seq, plddt=shared)
        sh_dis = encode_plddt_disorder(structure, seq, plddt=shared)
        sh_tier = encode_plddt_tier(structure, seq, plddt=shared)
        for ind, sh in ((ind_plddt, sh_plddt), (ind_dis, sh_dis),
                        (ind_tier, sh_tier)):
            np.testing.assert_array_equal(ind[0], sh[0])
            assert ind[1] == sh[1]
