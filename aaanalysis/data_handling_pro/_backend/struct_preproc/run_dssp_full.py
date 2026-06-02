"""
This is a script for the backend of the StructurePreprocessor: the DSSP
runner that captures per-residue ``(ss, asa_abs, phi, psi,
hbond_donor_offset, hbond_donor_energy, hbond_acceptor_offset,
hbond_acceptor_energy)`` for every chain in a single PDB / mmCIF file and
returns all eight feature streams.

The four H-bond fields (v1.2) come from biopython's DSSP tuple indices 6-9:
the primary NH→O (donor) partner residue offset + energy and the primary
O→HN (acceptor) partner offset + energy. Secondary / bifurcated H-bonds
(indices 10-13) are not captured in v1.2.
"""
from typing import List, Tuple
import shutil
import tempfile
from pathlib import Path


# I Helper Functions
def _resolve_dssp_binary() -> str:
    """Pick the DSSP binary name available on PATH."""
    for name in ("mkdssp", "dssp"):
        if shutil.which(name):
            return name
    raise RuntimeError("Neither 'mkdssp' nor 'dssp' is on PATH")


def _residue_one_letter(residue, protein_letters_3to1) -> str:
    """Return the 1-letter code for a residue, or 'X' if unknown."""
    return protein_letters_3to1.get(residue.get_resname(), "X")


def _coerce_float(v) -> float:
    """Convert DSSP cell to float; NaN sentinel ``'NA'`` / ``None`` -> NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# II Main Functions
ChainFull = Tuple[
    str,            # chain_id
    str,            # atom_seq
    List[str],      # ss
    List[float],    # asa (absolute, Å²)
    List[float],    # phi
    List[float],    # psi
    List[float],    # hbond_donor_offset       (signed; 0 = no partner)
    List[float],    # hbond_donor_energy       (kcal/mol; ≤ 0 if H-bond exists)
    List[float],    # hbond_acceptor_offset
    List[float],    # hbond_acceptor_energy
]


def run_dssp_full_for_entry_(pdb_path) -> List[ChainFull]:
    """Run DSSP and return per-chain 10-tuples with all 8 numerical streams.

    Uses the first model (NMR-safe) and copies the PDB into a temporary
    directory so DSSP intermediates do not pollute the user's ``pdb_folder``.

    Returns
    -------
    list of 10-tuples
        One record per chain that has at least one resolved residue:
        ``(chain_id, atom_seq, ss, asa_abs, phi, psi,
           hbond_donor_offset, hbond_donor_energy,
           hbond_acceptor_offset, hbond_acceptor_energy)``.
        ``asa`` is converted from biopython's relative ASA back to absolute
        Å² using ``residue_max_acc['Sander']``. H-bond fields come from
        biopython DSSP tuple indices 6-9 (primary NH→O and O→HN partner
        offset + energy). DSSP reports 0 for missing partner offsets and
        ~0 energy when no H-bond exists.

    Raises
    ------
    RuntimeError
        If DSSP fails (binary missing, malformed PDB, internal error). The
        caller is expected to convert this into a per-row UserWarning and
        ``dssp_ok=False``.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.DSSP import DSSP, residue_max_acc
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

    pdb_path = Path(pdb_path)
    dssp_bin = _resolve_dssp_binary()
    sander_max = residue_max_acc.get("Sander", {})

    # v1.1 file-format support: dispatch parser by extension. Gz inputs are
    # decompressed by the caller (`StructurePreprocessor` frontend via
    # `_file_format.resolve_structure_path`) before reaching this function,
    # so we never see ``.gz`` here.
    suffix = pdb_path.suffix.lower()
    is_cif = suffix in (".cif", ".cifs")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdb = Path(tmp_dir) / pdb_path.name
        shutil.copy(pdb_path, tmp_pdb)
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(tmp_pdb))
        try:
            model = next(structure.get_models())
        except StopIteration:
            raise RuntimeError(f"PDB '{pdb_path}' has no models")
        try:
            dssp = DSSP(model, str(tmp_pdb), dssp=dssp_bin)
        except Exception as e:
            raise RuntimeError(f"DSSP failed on '{pdb_path}': {e}") from e

        records = {}
        for key in dssp.keys():
            entry = dssp[key]
            # Bio.PDB.DSSP tuple layout (biopython >= 1.78):
            #   0 dssp_index, 1 aa, 2 ss, 3 rel_asa, 4 phi, 5 psi,
            #   6 NH_O_1_relidx, 7 NH_O_1_energy,
            #   8 O_NH_1_relidx, 9 O_NH_1_energy,
            #   10-13 secondary / bifurcated H-bonds (not captured in v1.2).
            records[key] = dict(
                ss=entry[2],
                rel_asa=_coerce_float(entry[3]),
                phi=_coerce_float(entry[4]),
                psi=_coerce_float(entry[5]),
                hb_donor_off=_coerce_float(entry[6]),
                hb_donor_en=_coerce_float(entry[7]),
                hb_acceptor_off=_coerce_float(entry[8]),
                hb_acceptor_en=_coerce_float(entry[9]),
            )

        chains: List[ChainFull] = []
        for chain in model:
            atom_seq_chars: List[str] = []
            atom_ss: List[str] = []
            atom_asa: List[float] = []
            atom_phi: List[float] = []
            atom_psi: List[float] = []
            atom_hb_d_off: List[float] = []
            atom_hb_d_en: List[float] = []
            atom_hb_a_off: List[float] = []
            atom_hb_a_en: List[float] = []
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                key = (chain.id, residue.id)
                if key not in records:
                    continue
                rec = records[key]
                resname = residue.get_resname()
                max_asa = sander_max.get(resname, float("nan"))
                if max_asa <= 0 or max_asa != max_asa:
                    abs_asa = float("nan")
                else:
                    abs_asa = rec["rel_asa"] * max_asa
                atom_seq_chars.append(
                    _residue_one_letter(residue, protein_letters_3to1))
                atom_ss.append(rec["ss"])
                atom_asa.append(abs_asa)
                atom_phi.append(rec["phi"])
                atom_psi.append(rec["psi"])
                atom_hb_d_off.append(rec["hb_donor_off"])
                atom_hb_d_en.append(rec["hb_donor_en"])
                atom_hb_a_off.append(rec["hb_acceptor_off"])
                atom_hb_a_en.append(rec["hb_acceptor_en"])
            if atom_seq_chars:
                chains.append((chain.id, "".join(atom_seq_chars),
                               atom_ss, atom_asa, atom_phi, atom_psi,
                               atom_hb_d_off, atom_hb_d_en,
                               atom_hb_a_off, atom_hb_a_en))
    return chains
