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

# Linked-selection highlight marker (the residue picked on the feature map): a bold, high-contrast
# stick + sphere drawn on top of the impact styling so the selection is unmistakable. The colour is
# shared with the feature-map column line (ut.COLOR_LINK_HIGHLIGHT) so the two panels read as linked.
_HIGHLIGHT_COLOR = ut.COLOR_LINK_HIGHLIGHT
_HIGHLIGHT_STICK = 0.9
_HIGHLIGHT_SPHERE = 0.6


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


def _is_int(val):
    """True if ``val`` is a plain / numpy integer (``bool`` is rejected)."""
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def _expand_regions(highlight, seq_len=None):
    """Expand ``highlight`` region(s) to the set of 1-based residue numbers they cover.

    Accepts ``None`` (-> empty set), a single ``(start, stop)`` tuple, or a list of such
    tuples (1-based, inclusive). Each region contributes ``range(start, stop + 1)``; the
    union is clamped to ``[1, seq_len]`` (only the lower bound of ``1`` when ``seq_len`` is
    ``None``). Every ``start`` / ``stop`` must be an integer with ``start <= stop``; anything
    else raises ``ValueError``. This mirrors the ``highlight`` shape of
    :meth:`AAPredPlot.predict_sample`, so a region marked on the sequence plot maps to the
    same residues on the structure. Pure (no py3Dmol), so it is unit-testable.
    """
    if highlight is None:
        return set()
    # A length-2 collection of scalars is one region; a list of (start, stop) pairs is many.
    is_single = (isinstance(highlight, (tuple, list)) and len(highlight) == 2
                 and not any(isinstance(v, (tuple, list)) for v in highlight))
    regions = [highlight] if is_single else highlight
    if not isinstance(regions, (list, tuple)):
        raise ValueError(f"'highlight' ({highlight!r}) should be a (start, stop) tuple or a "
                         f"list of (start, stop) tuples.")
    hi_bound = int(seq_len) if seq_len is not None else None
    resis = set()
    for region in regions:
        if not (isinstance(region, (tuple, list)) and len(region) == 2):
            raise ValueError(f"'highlight' region ({region!r}) should be a (start, stop) tuple.")
        start, stop = region
        if not (_is_int(start) and _is_int(stop)):
            raise ValueError(f"'highlight' region ({region!r}) should have integer 'start' and "
                             f"'stop' positions.")
        start, stop = int(start), int(stop)
        if start > stop:
            raise ValueError(f"'highlight' region ({start}, {stop}) should have 'start' <= 'stop'.")
        lo = max(1, start)
        hi = stop if hi_bound is None else min(stop, hi_bound)
        resis.update(range(lo, hi + 1))
    return resis


def _highlight_style_specs(highlight_resis, present_resis, chain_id=None):
    """py3Dmol ``(selection, style)`` specs colouring each highlighted, present residue cyan.

    ``highlight_resis`` is the residue set from :func:`_expand_regions`; only residues that
    actually exist in the structure (``present_resis``) are styled. Each spec sets the residue's
    cartoon to ``ut.COLOR_LINK_HIGHLIGHT`` (the shared linked-selection colour that
    :meth:`AAPredPlot.predict_sample` shades its feature-map region with), so a region picked on
    the sequence plot reads as the same selection here. Returned sorted by residue and pure (no
    view), so it is unit-testable without a live 3D render.
    """
    resis = sorted(set(highlight_resis or set()) & set(present_resis))
    return [(_sel(r, chain_id), {"cartoon": {"color": _HIGHLIGHT_COLOR}}) for r in resis]


# II Main Functions
def render_py3dmol(pdb_path, records, dict_impact, max_abs, mode,
                   focus, window_resis, size_by_impact, chain_id=None,
                   color_pos=None, color_neg=None, width=600, height=450,
                   highlight_resi=None, highlight_resis=None):
    """Build a py3Dmol cartoon view and wrap it in a StructureView, mirroring the app.

    The cartoon gets a neutral gray base; impact residues are then painted on the
    white->SHAP-red/blue ramp with optional ``|impact|``-scaled sticks (impact mode),
    or the whole cartoon is coloured by the discrete AlphaFold pLDDT palette (plddt mode).
    With ``focus`` ``'fade'`` / ``'zoom'`` the out-of-focus context is ghosted at low
    opacity so the feature window stands out, and ``'zoom'`` points the camera at it.
    ``addModel`` loads the whole (possibly multi-chain) structure, so every selection is
    qualified by ``chain_id`` — otherwise residue 50 would be coloured on every chain.

    ``highlight_resis`` (from :func:`_expand_regions`) paints whole ``(start, stop)`` residue
    ranges in ``ut.COLOR_LINK_HIGHLIGHT`` on top of the impact colouring, mirroring the region a
    user shaded on :meth:`AAPredPlot.predict_sample`; it is distinct from the single-pick
    ``highlight_resi`` marker (a bold stick + sphere for one hovered residue).
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

    # Region highlight: paint whole (start, stop) residue ranges cyan (COLOR_LINK_HIGHLIGHT),
    # the same colour AAPredPlot.predict_sample shades its feature-map region with, so a region
    # selected on the sequence plot mirrors here. Applied on top of the impact styling (and at
    # full opacity even when the context is faded); only residues present in the structure are
    # styled. Distinct from the single-pick `highlight_resi` marker below.
    for sel_spec, style_spec in _highlight_style_specs(highlight_resis, present_resis, chain_id):
        view.setStyle(sel_spec, style_spec)

    # Linked-selection marker: a bold highlight stick + sphere on the selected residue, added on
    # top of the impact styling (and full opacity even when the context is faded) so the residue
    # the user picked on the feature map stands out. Only marks residues present in the structure.
    if highlight_resi is not None and highlight_resi in present_resis:
        sel = _sel(highlight_resi, chain_id)
        view.addStyle(sel, {"stick": {"radius": _HIGHLIGHT_STICK, "color": _HIGHLIGHT_COLOR}})
        view.addStyle(sel, {"sphere": {"radius": _HIGHLIGHT_SPHERE, "color": _HIGHLIGHT_COLOR}})

    # Zoom to the in-focus window if asked (and it exists), else fit the whole model.
    if focus == "zoom" and in_focus:
        view.zoomTo(_sel(sorted(in_focus), chain_id))
    else:
        view.zoomTo()
    return StructureView(backend="py3dmol", view=view, dict_impact=dict_impact,
                         max_abs=max_abs, mode=mode)
