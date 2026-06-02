"""
This is a script for the backend of the StructurePreprocessor: a resolver
that finds an entry's structure file (``.pdb`` / ``.pdb.gz`` / ``.cif`` /
``.cif.gz``) in a folder, and its PAE sidecar (``.json`` / ``.json.gz``
with an AlphaFold-DB canonical-filename fallback).

The resolver decompresses gz inputs to a temporary file on disk because
``mkdssp`` and ``Bio.PDB.PDBParser`` / ``Bio.PDB.MMCIFParser`` both expect
plain files. Caller is responsible for tempdir lifetime (caller passes its
own ``tempfile.TemporaryDirectory()`` as ``temp_dir``).
"""
from pathlib import Path
from typing import Optional, Tuple
import gzip
import shutil


# I Helper Functions
_STRUCTURE_EXTENSIONS = (".pdb", ".pdb.gz", ".cif", ".cif.gz")


def _af_canonical_pae_name(entry: str) -> str:
    """The AF-DB canonical PAE sidecar filename for a UniProt entry."""
    return f"AF-{entry}-F1-predicted_aligned_error_v4.json"


def _decompress_to_temp(src: Path, temp_dir: Path,
                        target_suffix: str) -> Path:
    """Gunzip ``src`` into ``temp_dir`` with ``target_suffix``; return new path."""
    out = temp_dir / (src.stem if src.suffix == ".gz" else src.name)
    if not out.suffix:
        out = out.with_suffix(target_suffix)
    with gzip.open(src, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


# II Main Functions
def resolve_structure_path(folder: Path, entry: str,
                           temp_dir: Optional[Path] = None
                           ) -> Tuple[Optional[Path], Optional[str]]:
    """Find an entry's structure file in ``folder``.

    Resolution order: ``<entry>.pdb``, ``<entry>.pdb.gz``, ``<entry>.cif``,
    ``<entry>.cif.gz``. Gzipped inputs are decompressed into ``temp_dir`` so
    DSSP / parsers can read a plain file. CIF inputs are returned as-is.

    Parameters
    ----------
    folder : pathlib.Path
        Directory to search.
    entry : str
        The entry name (PDB-file basename without extension).
    temp_dir : pathlib.Path, optional
        Required only when a ``.gz`` file matches. Caller manages lifetime.

    Returns
    -------
    path : pathlib.Path or None
        Path to a readable structure file (decompressed if originally gz),
        or ``None`` if no matching file exists.
    file_format : {'pdb', 'cif'} or None
        Tag identifying which parser the caller should dispatch to.
        ``None`` when ``path is None``.
    """
    folder = Path(folder)
    for ext in _STRUCTURE_EXTENSIONS:
        candidate = folder / f"{entry}{ext}"
        if not candidate.is_file():
            continue
        if ext == ".pdb":
            return candidate, "pdb"
        if ext == ".cif":
            return candidate, "cif"
        if ext == ".pdb.gz":
            if temp_dir is None:
                raise RuntimeError(
                    "resolve_structure_path: gz input requires a "
                    "temp_dir from the caller")
            return _decompress_to_temp(candidate, temp_dir, ".pdb"), "pdb"
        if ext == ".cif.gz":
            if temp_dir is None:
                raise RuntimeError(
                    "resolve_structure_path: gz input requires a "
                    "temp_dir from the caller")
            return _decompress_to_temp(candidate, temp_dir, ".cif"), "cif"
    return None, None


def resolve_pae_path(folder: Path, entry: str,
                     temp_dir: Optional[Path] = None) -> Optional[Path]:
    """Find an entry's PAE sidecar in ``folder``.

    Resolution order: ``<entry>.json``, ``<entry>.json.gz``, then the
    AF-DB canonical filename ``AF-<entry>-F1-predicted_aligned_error_v4.json``
    (and its ``.gz`` variant). Gzipped inputs are decompressed into
    ``temp_dir``.

    Parameters
    ----------
    folder : pathlib.Path
        Directory to search.
    entry : str
        The entry name. For the canonical fallback, ``entry`` is interpreted
        as a UniProt accession (most AF-DB files use ``AF-<uniprot>-F1-…``).
    temp_dir : pathlib.Path, optional
        Required only when a ``.gz`` file matches.

    Returns
    -------
    path : pathlib.Path or None
        Path to a readable JSON sidecar (decompressed if gz), or ``None``.
    """
    folder = Path(folder)
    af_canonical = _af_canonical_pae_name(entry)
    candidates = [
        (folder / f"{entry}.json",           False),
        (folder / f"{entry}.json.gz",        True),
        (folder / af_canonical,              False),
        (folder / f"{af_canonical}.gz",      True),
    ]
    for candidate, is_gz in candidates:
        if not candidate.is_file():
            continue
        if not is_gz:
            return candidate
        if temp_dir is None:
            raise RuntimeError(
                "resolve_pae_path: gz input requires a temp_dir from the "
                "caller")
        return _decompress_to_temp(candidate, temp_dir, ".json")
    return None
