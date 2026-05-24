"""
This is a script for the backend of the StructurePreprocessor: per-feature
encoders that turn the DSSP per-residue list output (``ss``, ``asa``,
``phi``, ``psi``) into ``(L, D)`` numerical tensors. One private helper per
feature kind; the frontend ``encode_dssp`` orchestrates the dispatch and
concatenation.
"""
from typing import List, Optional

import numpy as np

import aaanalysis.utils as ut
from ._extras import MAX_ASA_PER_AA


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
def encode_ss(ss_list: List[str], ss_mode: str = "ss3") -> np.ndarray:
    """Encode a per-residue list of SS codes as a ``(L, D)`` one-hot ndarray.

    Parameters
    ----------
    ss_list : list of str
        Per-residue SS codes as produced by ``get_dssp(gap_handling='pad')``;
        unresolved positions are ``ut.STR_SS_GAP`` (``'-'``) and become NaN
        rows.
    ss_mode : {'ss3', 'ss8'}, default='ss3'
        Encoding alphabet. ``'ss3'`` returns ``(L, 3)`` over
        ``[ss_helix, ss_strand, ss_coil]``; ``'ss8'`` returns ``(L, 8)`` over
        ``[H, B, E, G, I, T, S, blank]``.

    Returns
    -------
    np.ndarray, shape (L, 3) or (L, 8)
        One-hot rows; NaN rows for ``ut.STR_SS_GAP`` (unresolved).
    """
    L = len(ss_list)
    if ss_mode == ut.SS_MODE_3:
        out = np.zeros((L, 3), dtype=np.float64)
        for i, code in enumerate(ss_list):
            idx = _ss3_index_for_code(code)
            if idx is None:
                out[i, :] = np.nan
            else:
                out[i, idx] = 1.0
        return out
    # ss8
    out = np.zeros((L, 8), dtype=np.float64)
    for i, code in enumerate(ss_list):
        idx = _ss8_index_for_code(code)
        if idx is None:
            out[i, :] = np.nan
        else:
            out[i, idx] = 1.0
    return out


def encode_asa(asa_list: List[float],
               sequence: str,
               kind: str = "rasa") -> np.ndarray:
    """Encode the DSSP ASA list as a ``(L, 1)`` ndarray.

    Parameters
    ----------
    asa_list : list of float
        Per-residue absolute ASA in Å² (DSSP output); ``None`` or NaN entries
        mark unresolved positions and propagate as NaN.
    sequence : str
        The per-row protein sequence; only used when ``kind='rasa'`` to look
        up the per-AA maximum ASA. Length must equal ``len(asa_list)``.
    kind : {'rasa', 'asa'}, default='rasa'
        ``'rasa'`` divides each value by ``MAX_ASA_PER_AA[residue]`` (Tien et
        al. 2013); ``'asa'`` returns the raw absolute value.

    Returns
    -------
    np.ndarray, shape (L, 1)
    """
    L = len(asa_list)
    if kind == "rasa" and len(sequence) != L:
        raise RuntimeError(
            f"asa/sequence length mismatch in encode_asa: "
            f"len(asa_list)={L}, len(sequence)={len(sequence)}")
    out = np.zeros((L, 1), dtype=np.float64)
    if kind == "rasa":
        for i, (val, aa_letter) in enumerate(zip(asa_list, sequence)):
            v = _safe_float(val)
            max_v = MAX_ASA_PER_AA.get(aa_letter)
            if np.isnan(v) or max_v is None or max_v <= 0:
                out[i, 0] = np.nan
            else:
                out[i, 0] = v / max_v
    else:
        for i, val in enumerate(asa_list):
            out[i, 0] = _safe_float(val)
    return out


def encode_dihedrals(phi_list: List[float],
                     psi_list: List[float],
                     encoding: str = "sin_cos") -> np.ndarray:
    """Encode (phi, psi) per residue as ``(L, 2)`` raw or ``(L, 4)`` sin/cos.

    Parameters
    ----------
    phi_list, psi_list : list of float
        Per-residue dihedral angles in degrees (DSSP convention). Unresolved
        positions are NaN or ``None`` and propagate.
    encoding : {'sin_cos', 'raw'}, default='sin_cos'
        ``'sin_cos'`` returns ``[sin(phi), cos(phi), sin(psi), cos(psi)]`` so
        the cyclic discontinuity at ±180° is removed; ``'raw'`` returns
        ``[phi, psi]`` in degrees.

    Returns
    -------
    np.ndarray, shape (L, 2) or (L, 4)
    """
    if len(phi_list) != len(psi_list):
        raise RuntimeError(
            f"phi/psi length mismatch in encode_dihedrals: "
            f"len(phi_list)={len(phi_list)}, len(psi_list)={len(psi_list)}")
    L = len(phi_list)
    if encoding == "raw":
        out = np.zeros((L, 2), dtype=np.float64)
        for i, (phi, psi) in enumerate(zip(phi_list, psi_list)):
            out[i, 0] = _safe_float(phi)
            out[i, 1] = _safe_float(psi)
        return out
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
    return out
