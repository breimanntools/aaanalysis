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
# Version-agnostic source of truth for download URLs. The /files/ names carry a
# data-version suffix (…_v4 -> …_v6 -> …) that changes without notice; the API
# always reports the current URLs, so we resolve through it instead of guessing.
AF_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/"

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


def _af_resolve_urls(entry: str, file_format: str, timeout: float = 30.0):
    """Resolve the current AlphaFold-DB model + PAE download URLs for an accession.

    Queries the AlphaFold prediction API, which always returns the URLs for the
    latest data version, so this survives AlphaFold-DB version bumps (the file
    naming moved ``v4`` -> ``v6`` and will move again). Returns
    ``(model_url, pae_url)``, or ``None`` when the accession is not in
    AlphaFold DB (the soft, ``on_failure``-governed case). ``pae_url`` may be
    ``None`` if the API record has no PAE sidecar.

    Raises
    ------
    RuntimeError
        On a non-404 API failure, or if the record lacks the requested model URL
        (an unexpected API-shape change worth surfacing rather than silently
        falling back to a guessed, possibly-stale URL).
    """
    api_url = f"{AF_API_URL}{entry}"
    try:
        resp = requests.get(api_url, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(
            f"AlphaFold API request for '{entry}' ('{api_url}') failed: {e}") from e
    # The API answers an accession it has no model for with 404 or 400 (the
    # latter for malformed/unknown accessions); both are the soft "not in
    # AlphaFold DB" case governed by on_failure, not a transport error.
    if resp.status_code in (400, 404):
        return None
    if resp.status_code != 200:
        raise RuntimeError(
            f"AlphaFold API request for '{entry}' returned HTTP "
            f"{resp.status_code} (expected 200)")
    records = resp.json()
    if not records:
        return None
    record = records[0]
    model_key = "cifUrl" if file_format == "cif" else "pdbUrl"
    model_url = record.get(model_key)
    if model_url is None:
        raise RuntimeError(
            f"AlphaFold API record for '{entry}' has no '{model_key}' "
            f"(unexpected API shape; available URL keys: "
            f"{[k for k in record if k.endswith('Url')]})")
    pae_url = record.get("paeDocUrl")
    return model_url, pae_url


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
        # Resolve current download URLs via the API (version-agnostic). None =>
        # the accession is not in AlphaFold DB; both files are then not-ok.
        resolved = _af_resolve_urls(entry, file_format, timeout)
        if resolved is None:
            model_ok, pae_ok = model_present, pae_present
        else:
            model_url, pae_url = resolved
            model_ok = model_present or fetch_af_file(model_url, model_path, timeout)
            pae_ok = pae_present or (
                pae_url is not None and fetch_af_file(pae_url, pae_path, timeout))
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
