"""
This is a script for the backend of get_dssp; runs ``mkdssp`` (via biopython's
:class:`Bio.PDB.DSSP` wrapper) on a single PDB file and returns per-chain
``(atom_seq, atom_ss)`` records.
"""
from typing import List, Tuple
import shutil
import tempfile
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1


# I Helper Functions
def _residue_one_letter(residue) -> str:
    """Return the 1-letter code for a residue, or 'X' if unknown / non-standard."""
    return protein_letters_3to1.get(residue.get_resname(), "X")


def _resolve_dssp_binary() -> str:
    """Pick the DSSP binary name available on PATH.

    Returns ``'mkdssp'`` (preferred; biopython supports the modern name) or
    ``'dssp'`` (legacy 2.x). Raises ``RuntimeError`` if neither is present;
    callers should typically have checked already via
    :func:`aaanalysis.struct_analysis_pro._get_dssp.check_mkdssp_installed`.
    """
    for name in ("mkdssp", "dssp"):
        if shutil.which(name):
            return name
    raise RuntimeError("Neither 'mkdssp' nor 'dssp' is on PATH")


# II Main Functions
def run_dssp_for_entry_(pdb_path) -> List[Tuple[str, str, List[str]]]:
    """Run DSSP on a single PDB and return per-chain ``(chain_id, atom_seq, atom_ss)``.

    Uses the first model (NMR-safe) and copies the PDB into a temporary directory
    so DSSP's intermediate files do not pollute the user's ``pdb_folder``.

    Parameters
    ----------
    pdb_path : str or pathlib.Path
        Path to a single PDB file.

    Returns
    -------
    list of (str, str, list[str])
        One triple per chain. ``atom_seq`` is the 1-letter ATOM sequence; the
        SS list has the same length and holds raw DSSP codes (``H/B/E/G/I/T/S``
        or a single space for "no SS"). Chains with zero standard residues are
        omitted.

    Raises
    ------
    RuntimeError
        If DSSP fails (binary missing, malformed PDB, internal error). The
        caller is expected to convert this into a per-row UserWarning and
        ``dssp_ok=False``.
    """
    pdb_path = Path(pdb_path)
    dssp_bin = _resolve_dssp_binary()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdb = Path(tmp_dir) / pdb_path.name
        shutil.copy(pdb_path, tmp_pdb)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(tmp_pdb))
        try:
            model = next(structure.get_models())
        except StopIteration:
            raise RuntimeError(f"PDB '{pdb_path}' has no models")
        try:
            dssp = DSSP(model, str(tmp_pdb), dssp=dssp_bin)
        except Exception as e:
            raise RuntimeError(f"DSSP failed on '{pdb_path}': {e}") from e
        # Map (chain_id, res_id) -> ss
        ss_by_key = {key: dssp[key][2] for key in dssp.keys()}
        chains: List[Tuple[str, str, List[str]]] = []
        for chain in model:
            atom_seq_chars: List[str] = []
            atom_ss: List[str] = []
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                key = (chain.id, residue.id)
                if key not in ss_by_key:
                    continue
                atom_seq_chars.append(_residue_one_letter(residue))
                atom_ss.append(ss_by_key[key])
            if atom_seq_chars:
                chains.append((chain.id, "".join(atom_seq_chars), atom_ss))
    return chains
