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


# --- AF model-file encoders (commit 2) ---------------------------------------
def _plddt_per_residue(structure, sequence: str):
    """Shared CA-B-factor read used by all three pLDDT keys.

    AlphaFold stores pLDDT (0–100) in the B-factor column of model PDB / CIF
    files. We read the CA-atom B-factor (preferred when available; falls
    back to mean over all atoms in the residue) which is consistent with the
    AF-DB confidence convention. Returns (aligned values list, identity).
    """
    chains = _collect_chain_residues(structure)
    if not chains:
        return None, 0.0
    chain, residues, atom_seq, identity = _pick_best_chain_records(
        sequence, chains)
    if chain is None:
        return None, 0.0
    atom_plddt: List[float] = []
    for residue, _ in residues:
        ca = None
        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                ca = atom
                break
        if ca is not None:
            atom_plddt.append(float(ca.get_bfactor()))
            continue
        b_values = [atom.get_bfactor() for atom in residue.get_atoms()]
        if not b_values:
            atom_plddt.append(float("nan"))
        else:
            atom_plddt.append(float(np.mean(b_values)))
    aligned = _align_atom_values_to_target(sequence, atom_seq, atom_plddt)
    return aligned, identity


def encode_plddt(structure, sequence: str) -> Tuple[np.ndarray, float]:
    """Per-residue pLDDT (AF B-factor column) as ``(L, 1)``, ``[0, 1]``."""
    aligned, identity = _plddt_per_residue(structure, sequence)
    if aligned is None:
        return np.full((len(sequence), 1), np.nan), 0.0
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize("plddt", raw), identity


def encode_plddt_disorder(structure, sequence: str,
                          threshold: float = 70.0
                          ) -> Tuple[np.ndarray, float]:
    """Boolean ``pLDDT < threshold`` as ``(L, 1)``, ``{0, 1}``."""
    aligned, identity = _plddt_per_residue(structure, sequence)
    if aligned is None:
        return np.full((len(sequence), 1), np.nan), 0.0
    arr = np.asarray(aligned, dtype=np.float64)
    out = np.where(np.isnan(arr), np.nan,
                   (arr < threshold).astype(np.float64))
    return normalize("plddt_disorder", out.reshape(-1, 1)), identity


def encode_plddt_tier(structure, sequence: str
                      ) -> Tuple[np.ndarray, float]:
    """Per-residue pLDDT tier one-hot ``[<50, 50-70, 70-90, ≥90]`` as ``(L, 4)``.

    Boundaries follow the AlphaFold-DB documented confidence tiers:
      very_low <50, low 50-70, confident 70-90, very_high ≥90.
    """
    aligned, identity = _plddt_per_residue(structure, sequence)
    if aligned is None:
        return np.full((len(sequence), 4), np.nan), 0.0
    arr = np.asarray(aligned, dtype=np.float64)
    L = arr.shape[0]
    out = np.zeros((L, 4), dtype=np.float64)
    for i, v in enumerate(arr):
        if np.isnan(v):
            out[i, :] = np.nan
            continue
        if v < 50:
            out[i, 0] = 1.0
        elif v < 70:
            out[i, 1] = 1.0
        elif v < 90:
            out[i, 2] = 1.0
        else:
            out[i, 3] = 1.0
    return normalize("plddt_tier", out), identity


def _chi_sincos_for_residue(residue, chi_index: int):
    """Compute chi_{chi_index} (1 or 2) for a residue; return (sin, cos) or (nan, nan).

    Uses Bio.PDB.internal_coords (loaded lazily) to derive the side-chain
    dihedral. NaN for residues without the requested chi (e.g. GLY/ALA for
    chi1; everything-but-{R,N,D,E,F,H,I,K,L,M,Q,W,Y} for chi2 etc.).
    """
    try:
        ric = residue.internal_coord
        if ric is None:
            # internal_coord must be initialised at chain level; the caller
            # ensures the parent chain is atom_to_internal_coordinates'd.
            return float("nan"), float("nan")
        chi_name = f"chi{chi_index}"
        angle = ric.get_angle(chi_name)
        if angle is None:
            return float("nan"), float("nan")
        rad = np.deg2rad(float(angle))
        return float(np.sin(rad)), float(np.cos(rad))
    except Exception:
        return float("nan"), float("nan")


def _encode_chi_sincos(structure, sequence: str, chi_index: int,
                       feature_key: str) -> Tuple[np.ndarray, float]:
    """Shared chi1 / chi2 backbone for the two registry keys."""
    chains = _collect_chain_residues(structure)
    if not chains:
        return np.full((len(sequence), 2), np.nan), 0.0
    chain, residues, atom_seq, identity = _pick_best_chain_records(
        sequence, chains)
    if chain is None:
        return np.full((len(sequence), 2), np.nan), 0.0
    # Initialise internal coordinates on the chain (idempotent).
    try:
        chain.atom_to_internal_coordinates()
    except Exception:
        pass
    sins: List[float] = []
    coss: List[float] = []
    for residue, _ in residues:
        s, c = _chi_sincos_for_residue(residue, chi_index)
        sins.append(s)
        coss.append(c)
    aligned_sin = _align_atom_values_to_target(sequence, atom_seq, sins)
    aligned_cos = _align_atom_values_to_target(sequence, atom_seq, coss)
    raw = np.column_stack([np.asarray(aligned_sin, dtype=np.float64),
                           np.asarray(aligned_cos, dtype=np.float64)])
    return normalize(feature_key, raw), identity


def encode_chi1_sincos(structure, sequence: str
                       ) -> Tuple[np.ndarray, float]:
    """sin/cos of chi1 per residue as ``(L, 2)``, ``[0, 1]``."""
    return _encode_chi_sincos(structure, sequence, 1, "chi1_sincos")


def encode_chi2_sincos(structure, sequence: str
                       ) -> Tuple[np.ndarray, float]:
    """sin/cos of chi2 per residue as ``(L, 2)``, ``[0, 1]``."""
    return _encode_chi_sincos(structure, sequence, 2, "chi2_sincos")


def _ca_coords_and_residues(structure, sequence: str):
    """Pick best chain and return (CA coordinates ndarray, residues, atom_seq, identity)."""
    chains = _collect_chain_residues(structure)
    if not chains:
        return None, None, None, 0.0
    chain, residues, atom_seq, identity = _pick_best_chain_records(
        sequence, chains)
    if chain is None:
        return None, None, None, 0.0
    ca_coords: List[np.ndarray] = []
    for residue, _ in residues:
        ca = None
        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                ca = atom
                break
        if ca is None:
            ca_coords.append(np.array([np.nan, np.nan, np.nan]))
        else:
            ca_coords.append(np.asarray(ca.get_coord(), dtype=np.float64))
    coords = np.vstack(ca_coords)
    return coords, residues, atom_seq, identity


def encode_ca_centroid_dist(structure, sequence: str
                            ) -> Tuple[np.ndarray, float]:
    """Per-residue ``||CA - centroid(CA)||`` as ``(L, 1)``, ``[0, 1]``."""
    coords, _residues, atom_seq, identity = _ca_coords_and_residues(
        structure, sequence)
    if coords is None or len(coords) == 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    finite = coords[~np.isnan(coords).any(axis=1)]
    if len(finite) == 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    centroid = finite.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    aligned = _align_atom_values_to_target(sequence, atom_seq, dists.tolist())
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize("ca_centroid_dist", raw), identity


def encode_ca_centroid_dist_norm(structure, sequence: str
                                  ) -> Tuple[np.ndarray, float]:
    """Per-residue ``||CA - centroid|| / Rg`` as ``(L, 1)``, ``[0, 1]``.

    Rg (radius of gyration over CA atoms) is computed as
    ``sqrt(mean(||CA_i - centroid||^2))``; degenerate / single-residue
    inputs yield NaN.
    """
    coords, _residues, atom_seq, identity = _ca_coords_and_residues(
        structure, sequence)
    if coords is None or len(coords) == 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    finite = coords[~np.isnan(coords).any(axis=1)]
    if len(finite) < 2:
        return np.full((len(sequence), 1), np.nan), 0.0
    centroid = finite.mean(axis=0)
    sq_dists_finite = ((finite - centroid) ** 2).sum(axis=1)
    rg = float(np.sqrt(sq_dists_finite.mean()))
    if not np.isfinite(rg) or rg <= 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    dists = np.linalg.norm(coords - centroid, axis=1) / rg
    aligned = _align_atom_values_to_target(sequence, atom_seq, dists.tolist())
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize("ca_centroid_dist_norm", raw), identity


def _encode_contact_count(structure, sequence: str, radius_A: float,
                          min_seq_sep: int, feature_key: str
                          ) -> Tuple[np.ndarray, float]:
    """Shared CA-CA contact-count backbone for the two radius variants."""
    coords, _residues, atom_seq, identity = _ca_coords_and_residues(
        structure, sequence)
    if coords is None or len(coords) == 0:
        return np.full((len(sequence), 1), np.nan), 0.0
    n = len(coords)
    counts = np.zeros(n, dtype=np.float64)
    has_finite = ~np.isnan(coords).any(axis=1)
    for i in range(n):
        if not has_finite[i]:
            counts[i] = np.nan
            continue
        ci = coords[i]
        for j in range(n):
            if j == i or abs(j - i) < min_seq_sep:
                continue
            if not has_finite[j]:
                continue
            if float(np.linalg.norm(ci - coords[j])) <= radius_A:
                counts[i] += 1
    aligned = _align_atom_values_to_target(sequence, atom_seq,
                                            counts.tolist())
    raw = np.asarray(aligned, dtype=np.float64).reshape(-1, 1)
    return normalize(feature_key, raw), identity


def encode_contact_count_8A(structure, sequence: str
                            ) -> Tuple[np.ndarray, float]:
    """Per-residue CA-CA contacts within 8 Å, seq-sep ≥ 5; ``(L, 1)`` in ``[0, 1]``."""
    return _encode_contact_count(structure, sequence,
                                 radius_A=8.0, min_seq_sep=5,
                                 feature_key="contact_count_8A")


def encode_contact_count_12A(structure, sequence: str
                             ) -> Tuple[np.ndarray, float]:
    """Per-residue CA-CA contacts within 12 Å, seq-sep ≥ 5; ``(L, 1)`` in ``[0, 1]``."""
    return _encode_contact_count(structure, sequence,
                                 radius_A=12.0, min_seq_sep=5,
                                 feature_key="contact_count_12A")
