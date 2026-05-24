"""
This is a script for the backend of the StructurePreprocessor: per-feature
encoders that turn the DSSP per-residue list output (``ss``, ``asa``,
``phi``, ``psi``) into ``(L, D)`` numerical tensors normalized to ``[0, 1]``
per the recipes in
``aaanalysis.struct_analysis_pro._backend.structure_preprocessor.feature_registry.NORMALIZATION_RECIPES``.

One private helper per feature kind; the frontend ``encode_dssp`` orchestrates
dispatch + concatenation. v1.1 drops the raw-`asa` and raw-`phi_psi` paths —
the registry exposes only ``rasa`` and ``phi_psi_sincos`` now.
"""
from typing import List, Optional

import numpy as np

import aaanalysis.utils as ut
from ._extras import MAX_ASA_PER_AA
from .feature_registry import normalize


# I Helper Functions
# Canonical column order for ss3 / ss8 one-hot encodings. The ss3 mapping
# follows ``ut.DICT_DSSP_3STATE`` (H/G/I -> H ; E/B -> E ; rest -> C); ss8
# uses the raw DSSP 8-state alphabet plus an explicit "blank" column for
# residues whose DSSP code was a literal space (rendered as ``'-'`` in the
# get_dssp output).
SS3_ORDER: List[str] = ["H", "E", "C"]
SS8_ORDER: List[str] = ["H", "B", "E", "G", "I", "T", "S", "-"]


def _ss3_index_for_code(code: str) -> Optional[int]:
    """Map a raw or 3-state SS code to its column index in the ss3 one-hot."""
    if code == ut.STR_SS_GAP:
        return None
    mapped = ut.DICT_DSSP_3STATE.get(code, code if code in SS3_ORDER else "C")
    if mapped not in SS3_ORDER:
        return None
    return SS3_ORDER.index(mapped)


def _ss8_index_for_code(code: str) -> Optional[int]:
    """Map a raw DSSP code to its column index in the ss8 one-hot."""
    if code == " ":
        return SS8_ORDER.index("-")
    if code == ut.STR_SS_GAP:
        return None
    if code not in SS8_ORDER:
        return None
    return SS8_ORDER.index(code)


def _safe_float(value) -> float:
    """Return ``float(value)``; coerce ``None`` and non-finite to NaN."""
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


# II Main Functions
def encode_ss(ss_list: List[str], feature_key: str) -> np.ndarray:
    """Encode a per-residue list of SS codes as a ``(L, D)`` one-hot ndarray.

    Parameters
    ----------
    ss_list : list of str
        Per-residue SS codes as produced by ``get_dssp(gap_handling='pad')``;
        unresolved positions are ``ut.STR_SS_GAP`` (``'-'``) and become NaN
        rows.
    feature_key : {'ss3', 'ss8'}
        Registry key. ``'ss3'`` returns ``(L, 3)`` over
        ``[ss_helix, ss_strand, ss_coil]``; ``'ss8'`` returns ``(L, 8)`` over
        ``[H, B, E, G, I, T, S, blank]``.

    Returns
    -------
    np.ndarray, shape (L, 3) or (L, 8)
        Already in `[0, 1]` (one-hot); the registry normalization recipe for
        these keys is identity.
    """
    L = len(ss_list)
    if feature_key == "ss3":
        out = np.zeros((L, 3), dtype=np.float64)
        for i, code in enumerate(ss_list):
            idx = _ss3_index_for_code(code)
            if idx is None:
                out[i, :] = np.nan
            else:
                out[i, idx] = 1.0
        return normalize(feature_key, out)
    # ss8
    out = np.zeros((L, 8), dtype=np.float64)
    for i, code in enumerate(ss_list):
        idx = _ss8_index_for_code(code)
        if idx is None:
            out[i, :] = np.nan
        else:
            out[i, idx] = 1.0
    return normalize(feature_key, out)


def encode_rasa(asa_list: List[float], sequence: str) -> np.ndarray:
    """Encode the DSSP ASA list as a ``(L, 1)`` relative-ASA ndarray.

    Divides each absolute-ASA value by ``MAX_ASA_PER_AA[residue]`` (Tien et al.
    2013); the registry normalization recipe then clips to ``[0, 1]`` (rare
    Tien-table overshoots for non-canonical contexts).

    Parameters
    ----------
    asa_list : list of float
        Per-residue absolute ASA in Å² (DSSP output); ``None`` / NaN entries
        mark unresolved positions and propagate as NaN.
    sequence : str
        The per-row protein sequence; length must equal ``len(asa_list)``.

    Returns
    -------
    np.ndarray, shape (L, 1)
        Relative ASA clipped to ``[0, 1]``.
    """
    L = len(asa_list)
    if len(sequence) != L:
        raise RuntimeError(
            f"asa/sequence length mismatch in encode_rasa: "
            f"len(asa_list)={L}, len(sequence)={len(sequence)}")
    out = np.zeros((L, 1), dtype=np.float64)
    for i, (val, aa_letter) in enumerate(zip(asa_list, sequence)):
        v = _safe_float(val)
        max_v = MAX_ASA_PER_AA.get(aa_letter)
        if np.isnan(v) or max_v is None or max_v <= 0:
            out[i, 0] = np.nan
        else:
            out[i, 0] = v / max_v
    return normalize("rasa", out)


def encode_dihedrals_sincos(phi_list: List[float],
                            psi_list: List[float]) -> np.ndarray:
    """Encode (phi, psi) per residue as ``(L, 4)`` sin/cos pairs in ``[0, 1]``.

    Always emits ``[sin(phi), cos(phi), sin(psi), cos(psi)]`` so the cyclic
    discontinuity at ±180° is removed. The raw values are in ``[-1, 1]`` and
    the registry recipe shifts/scales them to ``[0, 1]`` via ``(x + 1) / 2``.

    Parameters
    ----------
    phi_list, psi_list : list of float
        Per-residue dihedral angles in degrees (DSSP convention). Unresolved
        positions are NaN or ``None`` and propagate.

    Returns
    -------
    np.ndarray, shape (L, 4)
        Normalized to ``[0, 1]``; NaN where phi or psi was unresolved.
    """
    if len(phi_list) != len(psi_list):
        raise RuntimeError(
            f"phi/psi length mismatch in encode_dihedrals_sincos: "
            f"len(phi_list)={len(phi_list)}, len(psi_list)={len(psi_list)}")
    L = len(phi_list)
    out = np.zeros((L, 4), dtype=np.float64)
    for i, (phi, psi) in enumerate(zip(phi_list, psi_list)):
        phi_v = _safe_float(phi)
        psi_v = _safe_float(psi)
        if np.isnan(phi_v):
            out[i, 0] = out[i, 1] = np.nan
        else:
            phi_rad = np.deg2rad(phi_v)
            out[i, 0] = np.sin(phi_rad)
            out[i, 1] = np.cos(phi_rad)
        if np.isnan(psi_v):
            out[i, 2] = out[i, 3] = np.nan
        else:
            psi_rad = np.deg2rad(psi_v)
            out[i, 2] = np.sin(psi_rad)
            out[i, 3] = np.cos(psi_rad)
    return normalize("phi_psi_sincos", out)
