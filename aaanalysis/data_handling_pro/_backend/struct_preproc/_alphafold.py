"""
This is a script for the backend of the StructurePreprocessor: downloading
AlphaFold-DB model files (.pdb / .cif) and their PAE sidecar JSONs for a list
of UniProt accessions, saving them under the canonical filenames the
StructurePreprocessor file resolvers expect.

Files are saved so the existing resolvers find them with no glue: the model as
``<entry>.pdb`` / ``<entry>.cif`` (found by ``resolve_structure_path``) and the
PAE sidecar under the AlphaFold-DB canonical name
``AF-<entry>-F1-predicted_aligned_error_v4.json`` (found by ``resolve_pae_path``).

Backend trusts the frontend for argument validation; it raises ``RuntimeError``
on network / response failures other than a 404 (the soft "accession not in
AlphaFold DB" case), so missing structures do not abort a bulk download.
"""
from pathlib import Path
from typing import List
import warnings

import pandas as pd
import requests

import aaanalysis.utils as ut
from ._file_format import _af_canonical_pae_name

# I Helper Functions
AF_FILES_URL = "https://alphafold.ebi.ac.uk/files/"

# Column order for the per-entry status table built by ``fetch_alphafold_bulk``.
# Positional-list rows are wrapped into a DataFrame with these columns.
COL_MODEL_OK = "model_ok"
COL_PAE_OK = "pae_ok"
COL_ALPHAFOLD_OK = "alphafold_ok"
COL_SKIPPED = "skipped"
COL_MODEL_PATH = "model_path"
COL_PAE_PATH = "pae_path"
COLS_AF_STATUS = [
    ut.COL_ENTRY, COL_MODEL_OK, COL_PAE_OK, COL_ALPHAFOLD_OK, COL_SKIPPED,
    COL_MODEL_PATH, COL_PAE_PATH,
]


def _af_model_filename(entry: str, file_format: str) -> str:
    """Local filename for the model file (found by ``resolve_structure_path``)."""
    return f"{entry}.{file_format}"


def _af_model_url(entry: str, file_format: str) -> str:
    """AlphaFold-DB URL for the F1 model file of a UniProt accession."""
    return f"{AF_FILES_URL}AF-{entry}-F1-model_v4.{file_format}"


def _af_pae_url(entry: str) -> str:
    """AlphaFold-DB URL for the F1 PAE sidecar of a UniProt accession."""
    return f"{AF_FILES_URL}{_af_canonical_pae_name(entry)}"


def fetch_af_file(url: str, dest_path: Path, timeout: float = 30.0) -> bool:
    """Download one AlphaFold-DB file to ``dest_path`` with an atomic write.

    Returns
    -------
    bool
        ``True`` on HTTP 200 (file written); ``False`` on HTTP 404 (the
        accession / sidecar is not in AlphaFold DB).

    Raises
    ------
    RuntimeError
        On any other non-200 response or transport error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"AlphaFold request for '{url}' failed: {e}") from e
    if resp.status_code == 404:
        return False
    if resp.status_code != 200:
        raise RuntimeError(
            f"AlphaFold request for '{url}' returned HTTP "
            f"{resp.status_code} (expected 200)")
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    tmp_path.write_bytes(resp.content)
    tmp_path.replace(dest_path)
    return True


# II Main Functions
def fetch_alphafold_bulk(
    entries: List[str],
    out_folder: Path,
    file_format: str = "pdb",
    timeout: float = 30.0,
    skip_existing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download model + PAE for every entry; return a per-entry status table.

    A partial entry (only one of the two files present) re-fetches just the
    missing file when ``skip_existing`` is set. Entries missing from AlphaFold
    DB (404) are reported as not-ok via a ``UserWarning`` rather than raising.

    Raises
    ------
    RuntimeError
        Propagated from :func:`fetch_af_file` on network failure (non-404).
    """
    out_folder = Path(out_folder)
    rows: List[list] = []
    for entry in entries:
        model_path = out_folder / _af_model_filename(entry, file_format)
        pae_path = out_folder / _af_canonical_pae_name(entry)
        model_present = skip_existing and model_path.is_file()
        pae_present = skip_existing and pae_path.is_file()
        if model_present and pae_present:
            if verbose:
                ut.print_out(
                    f"Skipping '{entry}' (model + PAE already present)")
            rows.append([entry, True, True, True, True,
                         str(model_path), str(pae_path)])
            continue
        if verbose:
            ut.print_out(
                f"Fetching AlphaFold {file_format} + PAE for '{entry}'")
        model_ok = model_present or fetch_af_file(
            _af_model_url(entry, file_format), model_path, timeout)
        pae_ok = pae_present or fetch_af_file(
            _af_pae_url(entry), pae_path, timeout)
        if not model_ok:
            warnings.warn(
                f"AlphaFold model for '{entry}' not found in AlphaFold DB "
                f"(404); row will have model_ok=False", UserWarning,
                stacklevel=2)
        if not pae_ok:
            warnings.warn(
                f"AlphaFold PAE for '{entry}' not found in AlphaFold DB "
                f"(404); row will have pae_ok=False", UserWarning,
                stacklevel=2)
        rows.append([entry, model_ok, pae_ok, model_ok and pae_ok, False,
                     str(model_path) if model_ok else "",
                     str(pae_path) if pae_ok else ""])
    return pd.DataFrame(rows, columns=COLS_AF_STATUS)
