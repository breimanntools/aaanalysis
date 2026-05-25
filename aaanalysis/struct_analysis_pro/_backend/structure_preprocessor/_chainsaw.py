"""
This is a script for the backend of the StructurePreprocessor: a thin
subprocess wrapper around ChainSaw (Wells et al. 2024, Bioinformatics;
https://github.com/JudeWells/Chainsaw).

ChainSaw is not on PyPI and ships GPL-3-licensed code we cannot vendor
into this BSD-3 package. Users clone the upstream repo + run its
``bash setup.sh`` (which compiles the ``stride`` binary into the local
``stride/`` folder), then pass the resulting directory as
``chainsaw_path`` to :meth:`StructurePreprocessor.get_domains`.

The wrapper shells into ChainSaw's ``get_predictions.py`` for each PDB
file, parses the TSV output, and extracts the ``chopping`` column —
which is already in the Merizo/ChainSaw common format consumed by the
v1.2 ``encode_domains`` reader.
"""
from pathlib import Path
import subprocess
import sys


# I Helper Functions
def _chainsaw_entry_script(chainsaw_path: Path) -> Path:
    """Return the path to ChainSaw's ``get_predictions.py`` entry script."""
    return Path(chainsaw_path) / "get_predictions.py"


# II Main Functions
def resolve_chainsaw_path(chainsaw_path) -> Path:
    """Validate the user-supplied ChainSaw clone directory.

    Raises
    ------
    RuntimeError
        If ``chainsaw_path`` is None, missing, or does not contain
        ``get_predictions.py``. The error includes the upstream URL so
        the user knows where to clone from.
    """
    if chainsaw_path is None:
        raise RuntimeError(
            "'chainsaw_path' is required when tool='chainsaw'. Clone the "
            "upstream repo from https://github.com/JudeWells/Chainsaw "
            "and pass the local directory as chainsaw_path=...")
    p = Path(chainsaw_path)
    if not p.is_dir():
        raise RuntimeError(
            f"'chainsaw_path' ({chainsaw_path!r}) is not a directory. "
            f"Expected a local clone of "
            f"https://github.com/JudeWells/Chainsaw")
    script = _chainsaw_entry_script(p)
    if not script.is_file():
        raise RuntimeError(
            f"'chainsaw_path' ({chainsaw_path!r}) does not contain "
            f"'get_predictions.py'. Clone the upstream repo from "
            f"https://github.com/JudeWells/Chainsaw and pass its root "
            f"directory as chainsaw_path=...")
    return p


def _parse_chainsaw_tsv(stdout: str) -> str:
    """Extract the ``chopping`` value from ChainSaw's TSV stdout.

    ChainSaw prints a tab-separated header row + one data row per PDB.
    For single-file invocation we expect one data row.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            "ChainSaw produced no output; check stderr or run manually")
    # First line is the header.
    header = lines[0].split("\t")
    if "chopping" not in header:
        raise RuntimeError(
            f"ChainSaw output missing 'chopping' column "
            f"(header={header}); upstream version may have changed")
    idx = header.index("chopping")
    if len(lines) < 2:
        raise RuntimeError(
            "ChainSaw produced a header but no data row")
    cells = lines[1].split("\t")
    if idx >= len(cells):
        raise RuntimeError(
            "ChainSaw data row has fewer columns than the header")
    return cells[idx].strip()


def run_chainsaw_on_entry(pdb_path: Path,
                          chainsaw_path) -> str:
    """Run ChainSaw on a single PDB file and return its chopping string.

    Parameters
    ----------
    pdb_path : pathlib.Path
        Path to a single PDB or mmCIF file (already decompressed by the
        caller if originally ``.gz``).
    chainsaw_path : str or pathlib.Path
        Local clone of the ChainSaw repository (validated by
        :func:`resolve_chainsaw_path`).

    Returns
    -------
    chopping : str
        ``encode_domains``-compatible chopping string, e.g.
        ``"1-50,55-120,125-200"``. Empty string when ChainSaw assigns no
        domains.

    Raises
    ------
    RuntimeError
        If the subprocess fails, the output cannot be parsed, or the
        ``get_predictions.py`` script is missing.
    """
    cs_root = resolve_chainsaw_path(chainsaw_path)
    script = _chainsaw_entry_script(cs_root)
    # ChainSaw's CLI: python get_predictions.py --structure_file <pdb>
    # (per the upstream README). We use the same Python interpreter that
    # imported us so the user's virtualenv resolves correctly.
    cmd = [sys.executable, str(script),
           "--structure_file", str(pdb_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(cs_root), timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ChainSaw timed out on '{pdb_path}' after {e.timeout}s") \
            from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Failed to invoke ChainSaw: {e}") from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"ChainSaw failed on '{pdb_path}' (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:500]}")
    return _parse_chainsaw_tsv(proc.stdout)
