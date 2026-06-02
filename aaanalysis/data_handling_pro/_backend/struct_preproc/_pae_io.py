"""
This is a script for the backend of the StructurePreprocessor: load
AlphaFold predicted_aligned_error (PAE) JSON sidecars into a square
``(L, L)`` numpy matrix. Validates shape and squareness.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np


# I Helper Functions
def _coerce_to_matrix(payload, expected_L: Optional[int]) -> np.ndarray:
    """Extract a square (L, L) PAE matrix from common AF JSON layouts.

    Three layouts are handled:
      1. ``{"predicted_aligned_error": [[...], ...]}`` — AF-DB v4.
      2. ``{"pae": [[...], ...]}`` — some local AF runs / older releases.
      3. Bare 2-D list — user-written compact form.
    """
    if isinstance(payload, dict):
        for key in ("predicted_aligned_error", "pae"):
            if key in payload:
                payload = payload[key]
                break
    arr = np.asarray(payload, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(
            f"PAE sidecar ndim={arr.ndim} should be 2; got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise RuntimeError(
            f"PAE sidecar shape {arr.shape} should be square (L, L)")
    if expected_L is not None and arr.shape[0] != expected_L:
        raise RuntimeError(
            f"PAE sidecar L={arr.shape[0]} should equal "
            f"len(sequence)={expected_L}")
    return arr


# II Main Functions
def load_pae_matrix(json_path: Path,
                    expected_L: Optional[int] = None) -> np.ndarray:
    """Parse a PAE JSON sidecar into a validated ``(L, L)`` ndarray.

    Parameters
    ----------
    json_path : pathlib.Path
        Path to a plain (already-decompressed) ``.json`` file.
    expected_L : int or None
        When provided, the loaded L must match (else ``RuntimeError``).

    Returns
    -------
    np.ndarray, shape (L, L)
        PAE matrix in Å. Values are NOT clipped here — the encoder applies
        the registry's normalization recipe.

    Raises
    ------
    RuntimeError
        If the JSON cannot be parsed, the payload is not a square 2-D
        array, or the dimensions disagree with ``expected_L``.
    """
    try:
        with open(json_path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse PAE sidecar at '{json_path}': {e}") from e
    return _coerce_to_matrix(payload, expected_L=expected_L)
