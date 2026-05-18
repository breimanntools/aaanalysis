"""
This is a script for the frontend of get_dssp. Runs DSSP (via the ``mkdssp``
CLI and biopython's :class:`Bio.PDB.DSSP` wrapper) for every entry in a
``df_seq``-shaped DataFrame, looking up each PDB file by ``<entry>.pdb`` in
``pdb_folder``, and returns the original frame with per-residue secondary
structure aligned to ``df_seq[sequence]``.
"""
from typing import Optional, Union, List
import re
import shutil
import warnings
from pathlib import Path

import pandas as pd

import aaanalysis.utils as ut
from ._backend.get_dssp._dssp_runner import run_dssp_for_entry_
from ._backend.get_dssp._alignment import (pick_best_chain_,
                                            align_chain_to_sequence_,
                                            apply_ss_mode_,
                                            apply_gap_handling_,
                                            count_mismatches_)
from ._backend.get_dssp.build_output import build_get_dssp_output


# Entries are joined into filesystem paths; reject characters that would
# allow path escape or platform-specific quoting bugs. Matches UniProt
# accessions, plain identifiers, and dotted versions.
_SAFE_ENTRY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# I Helper Functions
def check_mkdssp_installed():
    """Raise ``RuntimeError`` if neither ``mkdssp`` nor ``dssp`` is on PATH."""
    if not (shutil.which("mkdssp") or shutil.which("dssp")):
        raise RuntimeError(
            "'mkdssp' is not installed or not in PATH. Install the DSSP suite "
            "(e.g. `conda install -c bioconda dssp` or `apt install dssp`) to "
            "use this function."
        )


def _check_no_output_collisions(df_seq):
    """Refuse to overwrite pre-existing ``ss`` / ``dssp_ok`` columns."""
    existing = [c for c in (ut.COL_SS, ut.COL_DSSP_OK) if c in df_seq.columns]
    if existing:
        raise ValueError(
            f"'df_seq' should not already contain DSSP output columns "
            f"(found: {existing}). Drop them before calling get_dssp.")


def _check_entry_is_filesystem_safe(entry):
    """Reject entries that would escape ``pdb_folder`` when joined into a path."""
    if not isinstance(entry, str) or not _SAFE_ENTRY_RE.match(entry):
        raise ValueError(
            f"'entry' ({entry!r}) should match [A-Za-z0-9_.-]+ to be safe "
            f"for filesystem lookup")


# II Main Functions
def get_dssp(df_seq: pd.DataFrame = None,
             pdb_folder: Union[str, Path] = None,
             ss_mode: str = "ss3",
             gap_handling: str = "pad",
             verbose: bool = True,
             ) -> pd.DataFrame:
    """Run DSSP [Kabsch83]_ on per-entry PDB files; return ``df_seq`` with
    per-residue secondary-structure annotations.

    Each row of ``df_seq`` is matched to ``<pdb_folder>/<entry>.pdb`` using the
    ``entry`` column. The chain whose ATOM sequence best matches
    ``df_seq[sequence]`` is selected automatically. Per-residue DSSP codes are
    aligned to ``df_seq[sequence]`` so the returned ``ss`` cell has the same
    length as the sequence (when ``gap_handling='pad'``).

    Raises ``RuntimeError`` if neither ``mkdssp`` nor ``dssp`` is on PATH.
    Per-entry failures (missing PDB, DSSP crash, no chain match) emit a
    ``UserWarning`` and appear in the output as ``ss=None`` and
    ``dssp_ok=False``; rows are never dropped.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        DataFrame containing an ``entry`` column with unique protein
        identifiers and a ``sequence`` column with full protein sequences.
        ``entry`` is used as the PDB-file basename (``<entry>.pdb``) and
        ``sequence`` is the target sequence to which DSSP output is aligned.
    pdb_folder : str or pathlib.Path
        Directory containing one ``<entry>.pdb`` file per row of ``df_seq``.
        Files that are missing are reported via ``UserWarning`` and surface
        as ``dssp_ok=False`` rows in the output.
    ss_mode : {'ss3', 'ss8'}, default='ss3'
        Secondary-structure encoding. ``'ss3'`` reduces DSSP's 8 native states
        to ``H`` / ``E`` / ``C`` (H/G/I → H; E/B → E; T/S/blank → C).
        ``'ss8'`` returns the raw DSSP codes ``H/B/E/G/I/T/S``; DSSP's blank
        ("no SS") is rendered as ``'-'`` for readability.
    gap_handling : {'pad', 'omit'}, default='pad'
        ``'pad'`` aligns DSSP output to ``df_seq[sequence]`` so
        ``len(ss) == len(sequence)``; positions without DSSP coverage
        (missing-coordinate residues, sequence mismatches, alignment gaps)
        are filled with ``'-'``. ``'omit'`` drops those positions so the
        returned list contains only DSSP-assigned states; in this mode
        ``len(ss) <= len(sequence)``.
    verbose : bool, default=True
        If ``True``, emit progress messages via ``ut.print_out``.

    Returns
    -------
    df_out : pd.DataFrame
        A copy of ``df_seq`` with two additional columns:

        - ``ss`` (``list[str]`` or ``None``): per-residue SS codes; ``None``
          when DSSP could not be run for the row.
        - ``dssp_ok`` (``bool``): ``True`` if DSSP ran and returned at least
          one residue for the matched chain; ``False`` otherwise.

    Notes
    -----
    * Only the first model in each PDB is used (NMR-safe).
    * Only ``.pdb`` files are recognized; ``.cif`` and ``.pdb.gz`` are not
      currently supported.
    * The chain whose ATOM sequence has the highest global-alignment identity
      to ``df_seq[sequence]`` is selected; a ``UserWarning`` is emitted when
      mismatched residues remain after alignment.
    * Entries must match the regex ``[A-Za-z0-9_.-]+`` so that the lookup
      ``<pdb_folder>/<entry>.pdb`` cannot escape ``pdb_folder``.
    * Requires ``aaanalysis[pro]`` (biopython) plus a ``mkdssp`` or ``dssp``
      binary on PATH.

    See Also
    --------
    * DSSP `documentation <https://swift.cmbi.umcn.nl/gv/dssp/>`__ and the
      MKDSSP `repository <https://github.com/PDB-REDO/dssp>`__.

    Examples
    --------
    .. include:: examples/get_dssp.rst
    """
    check_mkdssp_installed()
    # Validate
    ut.check_df_seq(df_seq=df_seq)
    if ut.COL_SEQ not in df_seq.columns:
        raise ValueError(
            f"'df_seq' should contain a '{ut.COL_SEQ}' column for get_dssp")
    _check_no_output_collisions(df_seq)
    if pdb_folder is None:
        raise ValueError("'pdb_folder' should not be None")
    ut.check_folder_path_exists(folder_path=str(pdb_folder), name="pdb_folder")
    ut.check_str_options(name="ss_mode", val=ss_mode,
                         list_str_options=ut.LIST_SS_MODES)
    ut.check_str_options(name="gap_handling", val=gap_handling,
                         list_str_options=ut.LIST_GAP_HANDLING)
    ut.check_bool(name="verbose", val=verbose)

    pdb_folder = Path(pdb_folder)
    entries = df_seq[ut.COL_ENTRY].tolist()
    sequences = df_seq[ut.COL_SEQ].tolist()
    ss_per_row: List[Optional[List[str]]] = []
    ok_per_row: List[bool] = []

    for entry, target_seq in zip(entries, sequences):
        _check_entry_is_filesystem_safe(entry)
        pdb_path = pdb_folder / f"{entry}.pdb"
        if not pdb_path.is_file():
            warnings.warn(
                f"PDB file for entry '{entry}' not found at '{pdb_path}'; "
                f"row will have ss=None, dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            ok_per_row.append(False)
            continue
        try:
            chains = run_dssp_for_entry_(pdb_path)
        except RuntimeError as e:
            warnings.warn(
                f"DSSP failed for entry '{entry}': {e}; "
                f"row will have ss=None, dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            ok_per_row.append(False)
            continue
        best = pick_best_chain_(target_seq, chains)
        if best is None:
            warnings.warn(
                f"No chains with assigned secondary structure for entry "
                f"'{entry}'; row will have ss=None, dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            ok_per_row.append(False)
            continue
        chain_id, atom_seq, atom_ss, identity = best
        n_mismatch = count_mismatches_(target_seq, atom_seq)
        if n_mismatch > 0:
            warnings.warn(
                f"Entry '{entry}': best-matching chain '{chain_id}' has "
                f"{n_mismatch} residue mismatch(es) against df_seq[sequence] "
                f"(identity={identity:.3f}). "
                f"Mismatched positions will be filled with "
                f"'{ut.STR_SS_GAP}' (gap_handling='{gap_handling}').",
                UserWarning)
        aligned = align_chain_to_sequence_(target_seq, atom_seq, atom_ss)
        encoded = apply_ss_mode_(aligned, ss_mode)
        final = apply_gap_handling_(encoded, gap_handling)
        ss_per_row.append(final)
        ok_per_row.append(True)
        if verbose:
            ut.print_out(
                f"   get_dssp: entry={entry}, chain={chain_id}, "
                f"identity={identity:.3f}, n_ss={len(final)}")

    return build_get_dssp_output(df_seq, ss_per_row, ok_per_row)
