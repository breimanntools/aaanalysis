"""
This is a script for the backend renderer of the CPPStructurePlot class: the
py3Dmol cartoon (per-residue ``setStyle`` colouring, optional impact-scaled sticks,
fade / zoom focus), wrapped by ``StructureView``. ``py3Dmol`` is imported lazily so
the class imports without it; the rendering methods require it.
"""
import numpy as np

import aaanalysis.utils as ut
from .colors import color_for_residue
from .view import StructureView


# I Helper Functions
def py3dmol_available():
    """Return ``True`` if the optional ``py3Dmol`` renderer is importable."""
    try:
        import py3Dmol  # noqa: F401
        return True
    except ImportError:
        return False


# Stick radii (Angstrom) for impact residues, matching the deployed app: a visible
# floor so every impact-carrying residue shows a stick, growing with |impact|.
_STICK_MIN = 0.18
_STICK_SPAN = 0.55
_STICK_FLAT = 0.22


def _stick_radius(impact, max_abs, size_by_impact):
    """Stick radius (A) for an impact residue; 0 for zero / undefined impact.

    With ``size_by_impact`` the radius floors at ``_STICK_MIN`` and grows with
    ``|impact| / max_abs`` (so even faint impacts stay visible, like the app);
    otherwise every impact residue gets a constant ``_STICK_FLAT`` stick.
    """
    if max_abs is None or max_abs <= 0 or not np.isfinite(impact) or impact == 0:
        return 0.0
    if not size_by_impact:
        return _STICK_FLAT
    return float(_STICK_MIN + _STICK_SPAN * min(1.0, abs(impact) / max_abs))


def _read_structure_text(pdb_path):
    """Read raw PDB / CIF text to feed py3Dmol's ``addModel``."""
    fmt = "cif" if str(pdb_path).lower().endswith(".cif") else "pdb"
    with open(str(pdb_path), "r", encoding="utf-8") as f:
        return f.read(), fmt


# Opacity of ghosted (out-of-focus) residues, matching the deployed app's 0.45.
_FADE_OPACITY = 0.45


def _sel(resi, chain_id):
    """py3Dmol residue selection, chain-qualified so it never leaks onto other chains."""
    sel = {"resi": str(resi) if not isinstance(resi, (list, tuple)) else [str(r) for r in resi]}
    if chain_id is not None:
        sel["chain"] = chain_id
    return sel


# II Main Functions
def render_py3dmol(pdb_path, records, dict_impact, max_abs, mode,
                   focus, window_resis, size_by_impact, chain_id=None,
                   color_pos=None, color_neg=None, width=600, height=450):
    """Build a py3Dmol cartoon view and wrap it in a StructureView, mirroring the app.

    The cartoon gets a neutral gray base; impact residues are then painted on the
    white->SHAP-red/blue ramp with optional ``|impact|``-scaled sticks (impact mode),
    or the whole cartoon is coloured by the discrete AlphaFold pLDDT palette (plddt mode).
    With ``focus`` ``'fade'`` / ``'zoom'`` the out-of-focus context is ghosted at low
    opacity so the feature window stands out, and ``'zoom'`` points the camera at it.
    ``addModel`` loads the whole (possibly multi-chain) structure, so every selection is
    qualified by ``chain_id`` — otherwise residue 50 would be coloured on every chain.
    """
    import py3Dmol
    pdb_text, fmt = _read_structure_text(pdb_path)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, fmt)
    view.setBackgroundColor("white")

    present_resis = {res["resi"] for res in records}
    in_focus = present_resis if window_resis is None else (set(window_resis) & present_resis)
    faded = focus != "whole" and bool(in_focus)

    # Base cartoon: ghost everything when focusing, else a neutral gray.
    if faded:
        view.setStyle({}, {"cartoon": {"color": ut.COLOR_STRUCT_MISSING,
                                       "opacity": _FADE_OPACITY}})
    else:
        view.setStyle({}, {"cartoon": {"color": ut.COLOR_STRUCT_MISSING}})

    if mode == "plddt":
        # Colour the in-focus residues (or all, when not faded) by the pLDDT palette.
        targets = records if not faded else [r for r in records if r["resi"] in in_focus]
        for res in targets:
            color = color_for_residue(res["resi"], dict_impact, max_abs, res["plddt"],
                                      "plddt")
            view.setStyle(_sel(res["resi"], chain_id), {"cartoon": {"color": color}})
    else:
        # Impact mode: paint only the residues that carry signed impact (full opacity),
        # with sticks; everything else keeps the gray / ghosted base, like the app.
        for res in records:
            resi = res["resi"]
            impact = dict_impact.get(resi, 0.0)
            if not np.isfinite(impact) or impact == 0:
                continue
            color = color_for_residue(resi, dict_impact, max_abs, res["plddt"], "impact",
                                      color_pos=color_pos, color_neg=color_neg)
            style = {"cartoon": {"color": color}}
            radius = _stick_radius(impact, max_abs, size_by_impact)
            if radius > 0:
                style["stick"] = {"radius": radius, "color": color}
            view.setStyle(_sel(resi, chain_id), style)

    # Zoom to the in-focus window if asked (and it exists), else fit the whole model.
    if focus == "zoom" and in_focus:
        view.zoomTo(_sel(sorted(in_focus), chain_id))
    else:
        view.zoomTo()
    return StructureView(backend="py3dmol", view=view, dict_impact=dict_impact,
                         max_abs=max_abs, mode=mode)
