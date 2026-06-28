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

# Reuse the package's SHAP ramps (``sns.light_palette`` via ``plot_get_cmap_``) so the
# 3D structure colours match CPPPlot.profile / feature_map exactly instead of a
# divergent linear interpolation. Both go white (index 0) -> saturated (index 100).
_N_RAMP = 101
_RAMP_POS = [mcolors.to_hex(c) for c in
             ut.plot_get_cmap_(cmap=ut.STR_CMAP_SHAP, n_colors=_N_RAMP, only_pos=True)]
_RAMP_NEG = [mcolors.to_hex(c) for c in
             ut.plot_get_cmap_(cmap=ut.STR_CMAP_SHAP, n_colors=_N_RAMP, only_neg=True)][::-1]


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
    """Map a signed impact to a hex colour on the white->SHAP-pos / white->SHAP-neg ramp.

    ``impact`` is normalised by ``max_abs`` to ``[-1, 1]`` and passed through the
    ``sign * sqrt`` transform to get a blend fraction in ``[0, 1]``. By default the
    colour is read off the package SHAP ramp (so it matches the 2D CPP plots); a
    custom ``color_pos`` / ``color_neg`` falls back to a white->colour interpolation.
    Zero / non-finite impact and a non-positive ``max_abs`` map to white.
    """
    if max_abs is None or max_abs <= 0 or not np.isfinite(impact) or impact == 0:
        return _WHITE
    t = float(np.clip(impact / max_abs, -1.0, 1.0))
    frac = float(np.sqrt(abs(t)))  # |sign * sqrt(t)| -> blend fraction in [0, 1]
    idx = int(round(frac * (_N_RAMP - 1)))
    if t > 0:
        return _RAMP_POS[idx] if color_pos is None else _lerp_hex(_WHITE, color_pos, frac)
    return _RAMP_NEG[idx] if color_neg is None else _lerp_hex(_WHITE, color_neg, frac)


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
