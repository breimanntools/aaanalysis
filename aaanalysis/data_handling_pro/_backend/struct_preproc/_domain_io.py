"""
This is a script for the backend of the StructurePreprocessor: parse a
per-entry domain-segmentation file into a list of domains, where each
domain is a list of ``(start, end)`` 1-based inclusive residue ranges.

Designed for the **Merizo / ChainSaw chopping format**: domains separated
by commas, segments within a domain separated by underscores, segments
are ``start-end`` 1-based inclusive — e.g. ``"6-18_296-459,19-156"``
means two domains (the first discontinuous: residues 6-18 ∪ 296-459;
the second: residues 19-156).

Two input file formats supported by the resolver:

  1. ``<entry>.txt`` — single line containing the chopping string.
  2. ``<entry>.tsv`` — any tab-separated file with a column named
     ``chopping`` (the Merizo/ChainSaw default). The first non-header
     row is read.

The user pre-runs Merizo / ChainSaw / AFragmenter on their PDB files and
writes the chopping string to disk. v1.2 does NOT bundle PyTorch or any
segmentation tool; the integration is intentionally BYO-segments to keep
the ``[pro]`` extra lean.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import re


# I Helper Functions
_CHOPPING_RE = re.compile(r"^[0-9_\-,]+$")


def _parse_chopping(chopping: str) -> List[List[Tuple[int, int]]]:
    """Parse a Merizo / ChainSaw chopping string into nested ranges.

    Returns
    -------
    list of list of (int, int)
        Outer list: one entry per domain. Inner list: one (start, end)
        1-based inclusive segment per (possibly discontinuous) part of
        that domain.

    Raises
    ------
    RuntimeError
        If the string is not parseable.
    """
    chopping = chopping.strip()
    if not chopping:
        return []
    if not _CHOPPING_RE.match(chopping):
        raise RuntimeError(
            f"chopping string {chopping!r} contains unexpected characters; "
            f"expected only digits, '-', '_', ','")
    domains: List[List[Tuple[int, int]]] = []
    for dom_str in chopping.split(","):
        dom_str = dom_str.strip()
        if not dom_str:
            continue
        segments: List[Tuple[int, int]] = []
        for seg_str in dom_str.split("_"):
            if "-" not in seg_str:
                raise RuntimeError(
                    f"segment {seg_str!r} in domain {dom_str!r} should be "
                    f"'start-end'")
            parts = seg_str.split("-")
            if len(parts) != 2:
                raise RuntimeError(
                    f"segment {seg_str!r} should split into exactly 2 "
                    f"endpoints on '-' (got {len(parts)})")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise RuntimeError(
                    f"segment {seg_str!r} endpoints should be integers")
            if start < 1 or end < start:
                raise RuntimeError(
                    f"segment ({start}, {end}) should satisfy 1 ≤ start "
                    f"≤ end")
            segments.append((start, end))
        domains.append(segments)
    return domains


# II Main Functions
def load_chopping(path: Path) -> List[List[Tuple[int, int]]]:
    """Read a chopping string from ``path`` and return parsed domain ranges.

    Format-agnostic: ``.txt`` files are read as a single chopping string
    (the first non-empty line); ``.tsv`` files are read column-wise with
    a ``chopping`` header. Unknown extensions are parsed as ``.txt``.
    """
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".tsv", ".csv"):
        delim = "\t" if suffix == ".tsv" else ","
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"domain file {path} is empty")
        header = lines[0].split(delim)
        if "chopping" not in header:
            raise RuntimeError(
                f"domain file {path} (header={header}) should have a "
                f"'chopping' column")
        idx = header.index("chopping")
        if len(lines) < 2:
            raise RuntimeError(f"domain file {path} has header but no data row")
        row = lines[1].split(delim)
        if idx >= len(row):
            raise RuntimeError(
                f"domain file {path} first data row has fewer columns "
                f"than the header")
        return _parse_chopping(row[idx])
    # Plain text fallback: first non-empty line is the chopping string.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _parse_chopping(line)
    return []


def resolve_domain_path(folder: Path, entry: str) -> Optional[Path]:
    """Find an entry's domain-segmentation file in ``folder``.

    Resolution order: ``<entry>.txt``, ``<entry>.tsv``, ``<entry>.csv``.
    """
    folder = Path(folder)
    for ext in (".txt", ".tsv", ".csv"):
        candidate = folder / f"{entry}{ext}"
        if candidate.is_file():
            return candidate
    return None
