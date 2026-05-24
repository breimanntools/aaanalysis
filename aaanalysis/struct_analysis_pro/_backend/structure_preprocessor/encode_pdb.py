"""
This is a script for the backend of the StructurePreprocessor: per-feature
encoders that pull values directly from PDB ATOM records (mean B-factor)
and from biopython's surface tools (``Bio.PDB.ResidueDepth``, which needs
the external ``msms`` binary). The frontend ``encode_pdb`` validates inputs,
opens the PDB once per entry, then dispatches by feature key.
"""
from typing import List, Tuple

import numpy as np

import aaanalysis.utils as ut
from ._extras import check_msms_available
from .feature_registry import normalize


# I Helper Functions
def _residue_one_letter(residue, protein_letters_3to1) -> str:
    """Return the 1-letter code for a residue, or 'X' if unknown."""
    return protein_letters_3to1.get(residue.get_resname(), "X")


def _collect_chain_residues(structure):
    """Iterate the first model's chains; yield (chain, [(residue, key)])."""
    from Bio.PDB.Polypeptide import is_aa
    try:
        model = next(structure.get_models())
    except StopIteration:
        return []
    out = []
    for chain in model:
        records = [(r, (chain.id, r.id))
                   for r in chain if is_aa(r, standard=False)]
        if records:
            out.append((chain, records))
    return out


def _make_aligner():
    """Identity-scored global aligner; same setup as get_dssp alignment."""
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def _identity_fraction(target_seq: str, atom_seq: str, aligner) -> float:
    if not target_seq or not atom_seq:
        return 0.0
    alignment = aligner.align(target_seq, atom_seq)[0]
    a, b = str(alignment[0]), str(alignment[1])
    return sum(1 for x, y in zip(a, b)
               if x == y and x != "-") / len(target_seq)


def _pick_best_chain_records(target_seq: str, chains):
    """Return (chain, residues, identity) for the chain best matching target."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    aligner = _make_aligner()
    best = None
    best_score = -1.0
    for chain, residues in chains:
        atom_seq = "".join(
            _residue_one_letter(r, protein_letters_3to1) for r, _ in residues)
        score = _identity_fraction(target_seq, atom_seq, aligner)
        if score > best_score:
            best_score = score
            best = (chain, residues, atom_seq)
    if best is None:
        return None, None, None, 0.0
    chain, residues, atom_seq = best
    return chain, residues, atom_seq, best_score


def _align_atom_values_to_target(target_seq: str,
                                 atom_seq: str,
                                 atom_values: List[float]) -> List[float]:
    """Map per-ATOM-residue floats onto target positions; gaps -> NaN."""
    aligner = _make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    a_aln, b_aln = str(alignment[0]), str(alignment[1])
    out: List[float] = []
    atom_idx = 0
    for ta, ab in zip(a_aln, b_aln):
        if ta == "-":
            if ab != "-":
                atom_idx += 1
            continue
        if ab == "-":
            out.append(float("nan"))
        else:
            out.append(atom_values[atom_idx])
            atom_idx += 1
    return out


# II Main Functions
def load_structure(pdb_path):
    """Parse a PDB or mmCIF file and return a Bio.PDB Structure (quiet mode).

    Dispatches by extension: ``.cif`` uses ``MMCIFParser``; everything else
    (``.pdb`` etc.) uses ``PDBParser``. Gz inputs are expected to have been
    decompressed by the file-format resolver before reaching here.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from pathlib import Path
    suffix = Path(str(pdb_path)).suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure("s", str(pdb_path))


def encode_bfactor(structure, sequence: str) -> Tuple[np.ndarray, float]:
    """Per-residue mean B-factor as ``(L, 1)`` ndarray, aligned to ``sequence``.

    Returns
    -------
    np.ndarray, shape (L, 1)
        Mean B-factor (over each residue's atoms). NaN for residues without
        an ATOM record or aligned gaps.
    float
        Identity fraction of the chosen chain vs. ``sequence`` — caller can
        use this for diagnostics.
    """
    chains = _collect_chain_residues(structure)
    if not chains:
        return np.full((len(sequence), 1), np.nan), 0.0
    chain, residues, atom_seq, identity = _pick_best_chain_records(sequence,
                                                                   chains)
    if chain is None:
        return np.full((len(sequence), 1), np.nan), 0.0
    atom_b: List[float] = []
    for residue, _ in residues:
        b_values = [atom.get_bfactor() for atom in residue.get_atoms()]
        if not b_values:
            atom_b.append(float("nan"))
        else:
            atom_b.append(float(np.mean(b_values)))
    aligned = _align_atom_values_to_target(sequence, atom_seq, atom_b)
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize("bfactor", raw), identity


def encode_depth(structure, sequence: str) -> Tuple[np.ndarray, float]:
    """Per-residue residue depth as ``(L, 1)`` ndarray, aligned to ``sequence``.

    Uses :class:`Bio.PDB.ResidueDepth.ResidueDepth`, which shells out to the
    external ``msms`` binary. The caller is expected to have verified that
    ``msms`` is on PATH via :func:`_extras.check_msms_available` before
    reaching this function.
    """
    check_msms_available()
    from Bio.PDB.ResidueDepth import ResidueDepth
    chains = _collect_chain_residues(structure)
    if not chains:
        return np.full((len(sequence), 1), np.nan), 0.0
    try:
        model = next(structure.get_models())
    except StopIteration:
        return np.full((len(sequence), 1), np.nan), 0.0
    try:
        rd = ResidueDepth(model)
    except Exception as e:
        raise RuntimeError(f"ResidueDepth failed (msms error?): {e}") from e
    chain, residues, atom_seq, identity = _pick_best_chain_records(sequence,
                                                                   chains)
    if chain is None:
        return np.full((len(sequence), 1), np.nan), 0.0
    atom_depth: List[float] = []
    for residue, key in residues:
        try:
            res_depth, _ca_depth = rd[(chain.id, residue.id)]
            atom_depth.append(float(res_depth))
        except Exception:
            atom_depth.append(float("nan"))
    aligned = _align_atom_values_to_target(sequence, atom_seq, atom_depth)
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize("depth", raw), identity
