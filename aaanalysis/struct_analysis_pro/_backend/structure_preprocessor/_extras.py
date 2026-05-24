"""
This is a script for the backend of the StructurePreprocessor: shared helpers
that don't belong with the DSSP or PDB encoders directly. Currently the
per-AA maximum-ASA table used for relative-ASA (rASA) normalization, and the
runtime ``msms``-binary availability check used by the residue-depth encoder.
"""
import shutil
from typing import Dict


# I Helper Functions
# (none)


# II Main Functions
# Per-amino-acid maximum solvent-accessible surface area (Å²) from Tien et al.
# 2013, *Maximum allowed solvent accessibility of residues in proteins*,
# PLoS ONE 8(11):e80635 — "theoretical" Gly-X-Gly tripeptide reference values.
# Used to convert DSSP absolute ASA into rASA (relative ASA) by dividing the
# residue's ASA by ``MAX_ASA_PER_AA[residue]``.
MAX_ASA_PER_AA: Dict[str, float] = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def is_msms_available() -> bool:
    """Return ``True`` if the ``msms`` binary is on PATH, else ``False``."""
    return shutil.which("msms") is not None


def check_msms_available() -> None:
    """Raise ``RuntimeError`` if ``msms`` is not installed.

    Used by the residue-depth encoder, which depends on
    :class:`Bio.PDB.ResidueDepth` and therefore on an external ``msms``
    binary. The error message includes a short install hint.
    """
    if not is_msms_available():
        raise RuntimeError(
            "'msms' is not installed or not in PATH. The residue-depth "
            "feature requires the MSMS surface-mesh binary. Install it via "
            "`conda install -c bioconda msms` or download from "
            "https://ccsb.scripps.edu/msms/, then ensure 'msms' is on PATH.")
