"""
This is a script for the backend of the StructurePreprocessor: per-feature
encoders that turn an AlphaFold predicted_aligned_error (PAE) matrix
``(L, L)`` into ``(L, D)`` per-residue summaries. Every encoder normalizes
to ``[0, 1]`` via the recipes in
``aaanalysis.data_handling_pro._backend.structure_preprocessor.feature_registry.NORMALIZATION_RECIPES``.

PAE asymmetry is by AlphaFold's design — ``PAE[i, j] != PAE[j, i]`` in
general — so a dedicated ``encode_pae_asymmetry`` keeps that signal.
"""
from typing import Tuple

import numpy as np

from .feature_registry import normalize


# I Helper Functions
def _row_reduce(pae: np.ndarray, op) -> np.ndarray:
    """Apply ``op`` along axis=1 with NaN-awareness; return shape (L, 1)."""
    with np.errstate(invalid="ignore"):
        out = op(pae, axis=1)
    return out.astype(np.float64).reshape(-1, 1)


def _local_distal_split(pae: np.ndarray, local_window: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Split PAE row-means into local (±window) and distal (outside) means.

    Returns (local_mean, distal_mean) each of shape (L, 1). When a row has
    no local neighbors (window=0) the local value is NaN; when a row has no
    distal residues (very short protein), distal is NaN.
    """
    L = pae.shape[0]
    local_mean = np.full(L, np.nan)
    distal_mean = np.full(L, np.nan)
    idx = np.arange(L)
    for i in range(L):
        local_mask = np.abs(idx - i) <= local_window
        distal_mask = ~local_mask
        # Exclude PAE[i, i] from local mean
        local_mask[i] = False
        if local_mask.any():
            local_mean[i] = float(np.nanmean(pae[i, local_mask]))
        if distal_mask.any():
            distal_mean[i] = float(np.nanmean(pae[i, distal_mask]))
    return local_mean.reshape(-1, 1), distal_mean.reshape(-1, 1)


def _band_means(pae: np.ndarray, band_edges: Tuple[int, int]) -> np.ndarray:
    """Per-residue mean PAE over three sequence-distance bands.

    Bands: [0, edges[0]], (edges[0], edges[1]], (edges[1], L].
    Returns shape (L, 3). NaN for empty bands at protein ends.
    """
    L = pae.shape[0]
    if not (isinstance(band_edges, (list, tuple)) and len(band_edges) == 2):
        raise RuntimeError(
            f"band_edges {band_edges!r} should be a length-2 tuple")
    lo, hi = band_edges
    if not (0 < lo < hi):
        raise RuntimeError(
            f"band_edges ({lo}, {hi}) should satisfy 0 < lo < hi")
    out = np.full((L, 3), np.nan, dtype=np.float64)
    idx = np.arange(L)
    for i in range(L):
        dist = np.abs(idx - i)
        masks = [
            (dist > 0) & (dist <= lo),
            (dist > lo) & (dist <= hi),
            dist > hi,
        ]
        for b, mask in enumerate(masks):
            if mask.any():
                out[i, b] = float(np.nanmean(pae[i, mask]))
    return out


# II Main Functions
def encode_pae_row_mean(pae: np.ndarray) -> np.ndarray:
    """Mean PAE per row (i.e. residue), shape ``(L, 1)``, ``[0, 1]``."""
    return normalize("pae_row_mean", _row_reduce(pae, np.nanmean))


def encode_pae_row_min(pae: np.ndarray) -> np.ndarray:
    """Min PAE per row, shape ``(L, 1)``, ``[0, 1]``."""
    return normalize("pae_row_min", _row_reduce(pae, np.nanmin))


def encode_pae_row_max(pae: np.ndarray) -> np.ndarray:
    """Max PAE per row, shape ``(L, 1)``, ``[0, 1]``."""
    return normalize("pae_row_max", _row_reduce(pae, np.nanmax))


def encode_pae_local_mean(pae: np.ndarray,
                          local_window: int = 5) -> np.ndarray:
    """Mean PAE within ±``local_window`` (exclusive of self), shape ``(L, 1)``."""
    local, _ = _local_distal_split(pae, local_window)
    return normalize("pae_local_mean", local)


def encode_pae_distal_mean(pae: np.ndarray,
                           local_window: int = 5) -> np.ndarray:
    """Mean PAE outside ±``local_window``, shape ``(L, 1)``."""
    _, distal = _local_distal_split(pae, local_window)
    return normalize("pae_distal_mean", distal)


def encode_pae_asymmetry(pae: np.ndarray) -> np.ndarray:
    """Per-residue mean of ``|PAE[i, j] - PAE[j, i]|``, shape ``(L, 1)``.

    Captures the directional uncertainty signal that AlphaFold encodes by
    design in its non-symmetric PAE matrix.
    """
    diff = np.abs(pae - pae.T)
    with np.errstate(invalid="ignore"):
        per_row = np.nanmean(diff, axis=1)
    return normalize("pae_asymmetry",
                     per_row.astype(np.float64).reshape(-1, 1))


def encode_pae_band_means(pae: np.ndarray,
                          band_edges: Tuple[int, int] = (5, 15)
                          ) -> np.ndarray:
    """Per-band mean PAE, shape ``(L, 3)``, ``[0, 1]``."""
    return normalize("pae_band_means",
                     _band_means(pae, band_edges=band_edges))
