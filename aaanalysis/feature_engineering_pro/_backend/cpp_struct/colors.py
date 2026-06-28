"""
This is a script for the backend colour ramps of the CPPStructurePlot class:
the signed white->SHAP-red / white->SHAP-blue impact ramp (with the
``sign * sqrt(|t|)`` perceptual transform that keeps faint impacts visible) and
the AlphaFold-style pLDDT confidence ramp. All colours come from the ``ut``
constants barrel; no hex is hardcoded here.
"""
import numpy as np
import matplotlib.colors as mcolors

import aaanalysis.utils as ut

# White anchor of the signed impact ramp (a neutral, not a domain colour).
_WHITE = "#FFFFFF"


# I Helper Functions
def _lerp_hex(color_lo, color_hi, frac):
    """Linearly interpolate between two colours in RGB; return a hex string."""
    lo = np.asarray(mcolors.to_rgb(color_lo), dtype=np.float64)
    hi = np.asarray(mcolors.to_rgb(color_hi), dtype=np.float64)
    frac = float(np.clip(frac, 0.0, 1.0))
    return mcolors.to_hex(lo + (hi - lo) * frac)


# II Main Functions
def perceptual_transform(t):
    """Signed square-root transform ``sign(t) * sqrt(|t|)`` on ``t`` in ``[-1, 1]``.

    Compresses large magnitudes and stretches small ones so faint but real
    impacts stay visible; the sign is preserved and the output stays in
    ``[-1, 1]``.
    """
    t = np.asarray(t, dtype=np.float64)
    return np.sign(t) * np.sqrt(np.abs(t))


def impact_to_hex(impact, max_abs, color_pos=None, color_neg=None):
    """Map a signed impact to a hex colour, white -> SHAP-red (+) / SHAP-blue (-).

    ``impact`` is normalised by ``max_abs`` to ``[-1, 1]``; the blend fraction off white
    is ``sqrt(|impact| / max_abs)`` (the ``sign * sqrt`` perceptual transform), so the
    colour **intensity scales with the absolute impact** — faint impacts stay near white,
    strong ones reach full ``COLOR_SHAP_POS`` / ``COLOR_SHAP_NEG``. This is a **linear**
    white->colour interpolation, matching the deployed app's ``shapColor`` exactly
    (``color_pos`` / ``color_neg`` override the endpoints). Zero / non-finite impact and a
    non-positive ``max_abs`` map to white.
    """
    if max_abs is None or max_abs <= 0 or not np.isfinite(impact) or impact == 0:
        return _WHITE
    t = float(np.clip(impact / max_abs, -1.0, 1.0))
    frac = float(np.sqrt(abs(t)))  # |sign * sqrt(t)| -> blend fraction in [0, 1]
    if t > 0:
        return _lerp_hex(_WHITE, color_pos if color_pos is not None else ut.COLOR_SHAP_POS, frac)
    return _lerp_hex(_WHITE, color_neg if color_neg is not None else ut.COLOR_SHAP_NEG, frac)


def plddt_cmap():
    """Continuous low->high pLDDT colormap built from the ``ut.LIST_COLOR_PLDDT`` ramp."""
    return mcolors.LinearSegmentedColormap.from_list("plddt", ut.LIST_COLOR_PLDDT)


def plddt_to_hex(plddt):
    """Map a pLDDT value (0-100) to the discrete AlphaFold confidence colour.

    Uses the canonical AlphaFold bins (very high >=90, confident >=70, low >=50, very
    low <50) read from ``ut.LIST_COLOR_PLDDT`` (ordered low->high), matching the
    AlphaFold DB / deployed-app look rather than a smooth gradient. Non-finite -> gray.
    """
    if plddt is None or not np.isfinite(plddt):
        return ut.COLOR_STRUCT_MISSING
    c_low, c_lo_mid, c_hi_mid, c_high = ut.LIST_COLOR_PLDDT  # orange, yellow, cyan, blue
    if plddt >= 90:
        return c_high
    if plddt >= 70:
        return c_hi_mid
    if plddt >= 50:
        return c_lo_mid
    return c_low


def color_for_residue(resi, dict_impact, max_abs, plddt, mode,
                      color_pos=None, color_neg=None):
    """Resolve the colour of a single residue for ``mode`` ('impact' or 'plddt')."""
    if mode == "plddt":
        return plddt_to_hex(plddt)
    return impact_to_hex(dict_impact.get(resi, 0.0), max_abs,
                         color_pos=color_pos, color_neg=color_neg)
