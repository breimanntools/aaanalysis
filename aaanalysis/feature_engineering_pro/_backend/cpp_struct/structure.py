"""
This is a script for the backend structure handling of the CPPStructurePlot class:
pick the chain (explicit id, best match to a target sequence, or the first
amino-acid chain) and extract per-residue Cα coordinates, **absolute** residue
numbers, and pLDDT (the AlphaFold B-factor column).

Parsing and best-chain selection are reused verbatim from the StructurePreprocessor
backend (``load_structure`` / ``_collect_chain_residues`` / ``_resolve_best_chain``)
rather than re-implemented — the only new code here is the thin chain-by-id
Cα/pLDDT extractor, which the encoder backend does not expose because its encoders
return sequence-aligned arrays that drop the absolute residue numbering this
renderer needs.

The top-level biopython import gates the whole CPPStructurePlot feature behind the
``pro`` extra, matching the other ``*_pro`` subpackages.
"""
import numpy as np
# Top-level biopython import: structure parsing has no fallback, so this is the
# dependency that gates CPPStructurePlot behind the ``pro`` extra.
from Bio.PDB.Polypeptide import protein_letters_3to1

import aaanalysis.utils as ut
# Reused (no duplication) from the StructurePreprocessor backend. These are
# load-bearing beyond StructurePreprocessor — CPPStructurePlot depends on their
# behaviour too; keep their contracts stable.
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb import (
    load_structure, _collect_chain_residues, _resolve_best_chain, _identity_fraction)


# I Helper Functions
def _residue_one_letter(residue):
    """Return the 1-letter code for a residue, or 'X' if unknown."""
    return protein_letters_3to1.get(residue.get_resname(), "X")


def _ca_coord(residue):
    """Return the residue's Cα coordinate as a ``(3,)`` array, NaN if absent."""
    for atom in residue.get_atoms():
        if atom.get_name() == "CA":
            return np.asarray(atom.get_coord(), dtype=np.float64)
    return np.array([np.nan, np.nan, np.nan])


def _ca_plddt(residue):
    """Return the residue's Cα B-factor (AlphaFold pLDDT); mean atom B-factor if no Cα."""
    for atom in residue.get_atoms():
        if atom.get_name() == "CA":
            return float(atom.get_bfactor())
    b_values = [atom.get_bfactor() for atom in residue.get_atoms()]
    return float(np.mean(b_values)) if b_values else float("nan")


def _records_from_residues(residues):
    """Build ``{"resi", "coord", "plddt", "aa"}`` records from biopython residues.

    ``residues`` is the ``[(residue, key)]`` form returned by
    ``_collect_chain_residues`` / ``_resolve_best_chain``; ``resi`` is the
    structure's absolute residue number (``residue.id[1]``).
    """
    records = []
    for residue, _key in residues:
        records.append({"resi": int(residue.id[1]),
                        "coord": _ca_coord(residue),
                        "plddt": _ca_plddt(residue),
                        "aa": _residue_one_letter(residue)})
    return records


def _chain_identity(residues, sequence):
    """Sequence-match fraction of a chain's residues against ``sequence`` (1.0 if no seq)."""
    if sequence is None:
        return 1.0
    atom_seq = "".join(_residue_one_letter(r) for r, _ in residues)
    return _identity_fraction(sequence, atom_seq)


def _select_chain_residues(structure, chain=None, sequence=None):
    """Select a chain's residues: explicit id, best match to ``sequence``, or first.

    Returns ``(residues, identity, chain_id)`` where ``residues`` is the
    ``[(residue, key)]`` form, ``identity`` is the sequence-match fraction (1.0 when no
    ``sequence`` is supplied), and ``chain_id`` is the selected chain's id (so the
    renderer can qualify per-residue selections and not leak onto same-numbered
    residues of other chains).
    """
    chains = _collect_chain_residues(structure)
    if not chains:
        raise ValueError("'pdb' (structure) should contain at least one "
                         "amino-acid chain")
    if chain is not None:
        for ch, residues in chains:
            if ch.id == chain:
                # Still score identity so a wrong explicit chain + sequence warns.
                return residues, _chain_identity(residues, sequence), ch.id
        available = [ch.id for ch, _ in chains]
        raise ValueError(f"'chain' ('{chain}') should be one of the available "
                         f"chains {available}")
    if sequence is not None:
        ch, residues, _atom_seq, identity = _resolve_best_chain(structure, sequence)
        if residues is None:
            raise ValueError("'pdb' (structure) should contain at least one "
                             "amino-acid chain")
        return residues, identity, ch.id
    # Default: first amino-acid chain.
    ch, residues = chains[0]
    return residues, 1.0, ch.id


# II Main Functions
def extract_chain_residues(structure, chain=None, sequence=None):
    """Extract per-residue records for the selected chain.

    Returns
    -------
    records : list of dict
        One ``{"resi": int, "coord": (3,) ndarray, "plddt": float, "aa": str}``
        per amino-acid residue, in chain order; ``resi`` is the structure's
        absolute residue number.
    identity : float
        Sequence-match fraction of the chosen chain (1.0 without ``sequence``).
    chain_id : str
        Id of the selected chain (qualifies per-residue rendering selections).
    """
    residues, identity, chain_id = _select_chain_residues(structure, chain=chain,
                                                          sequence=sequence)
    return _records_from_residues(residues), identity, chain_id


def residue_numbers(records):
    """Return the set of absolute residue numbers present in ``records``."""
    return {r["resi"] for r in records}


# Re-export the reused parser so the frontend imports a single structure module.
__all__ = ["load_structure", "extract_chain_residues", "residue_numbers"]
