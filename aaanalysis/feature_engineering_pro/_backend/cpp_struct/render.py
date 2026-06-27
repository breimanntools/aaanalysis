"""
This is a script for the backend renderers of the CPPStructurePlot class: the
interactive py3Dmol cartoon (per-residue ``setStyle`` colouring, optional impact-
scaled sticks, fade / zoom focus) and the static matplotlib ``mplot3d`` Cα-scatter
fallback. Both paint the same colour ramp and are wrapped by ``StructureView``.
``py3Dmol`` is imported lazily so the matplotlib fallback works when it is absent.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


# II Main Functions
def render_py3dmol(pdb_path, records, dict_impact, max_abs, mode,
                   focus, window_resis, size_by_impact, chain_id=None,
                   color_pos=None, color_neg=None, width=600, height=450):
    """Build a py3Dmol cartoon view coloured per residue and wrap it in a StructureView.

    ``addModel`` loads the whole (possibly multi-chain) structure, so every
    per-residue ``setStyle`` / ``zoomTo`` selection is qualified by ``chain_id`` —
    otherwise residue number 50 would be coloured on every chain that has one.
    """
    import py3Dmol
    pdb_text, fmt = _read_structure_text(pdb_path)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, fmt)
    view.setStyle({}, {"cartoon": {"color": ut.COLOR_STRUCT_MISSING}})
    present_resis = {res["resi"] for res in records}
    for res in records:
        resi = res["resi"]
        color = color_for_residue(resi, dict_impact, max_abs, res["plddt"], mode,
                                  color_pos=color_pos, color_neg=color_neg)
        in_window = window_resis is None or resi in window_resis
        cartoon = {"color": color}
        if focus == "fade" and not in_window:
            cartoon["opacity"] = 0.2
        style = {"cartoon": cartoon}
        if mode == "impact":
            radius = _stick_radius(dict_impact.get(resi, 0.0), max_abs, size_by_impact)
            if radius > 0:
                style["stick"] = {"radius": radius, "color": color}
        sel = {"resi": str(resi)}
        if chain_id is not None:
            sel["chain"] = chain_id
        view.setStyle(sel, style)
    # Only zoom to window residues that actually exist in the structure, else the
    # camera silently fails to focus on an empty selection.
    zoom_resis = sorted((window_resis or set()) & present_resis)
    if focus == "zoom" and zoom_resis:
        sel = {"resi": [str(r) for r in zoom_resis]}
        if chain_id is not None:
            sel["chain"] = chain_id
        view.zoomTo(sel)
    else:
        view.zoomTo()
    view.setBackgroundColor("white")
    return StructureView(backend="py3dmol", view=view, dict_impact=dict_impact,
                         max_abs=max_abs, mode=mode)


def draw_structure_mpl(ax, records, dict_impact, max_abs, mode, focus, window_resis,
                       size_by_impact, color_pos=None, color_neg=None, set_title=True):
    """Draw the per-residue Cα scatter / backbone trace onto a provided 3D ``ax``.

    Shared by :func:`render_mpl` (which owns its own Figure) and the combined
    static figure (which supplies its left 3D panel), so both paint pixel-identical
    structures from one code path.
    """
    xs, ys, zs, rgba, sizes = [], [], [], [], []
    win_coords = []
    for res in records:
        coord = res["coord"]
        if not np.all(np.isfinite(coord)):
            continue
        resi = res["resi"]
        color = color_for_residue(resi, dict_impact, max_abs, res["plddt"], mode,
                                  color_pos=color_pos, color_neg=color_neg)
        in_window = window_resis is None or resi in window_resis
        alpha = 1.0 if (focus != "fade" or in_window) else 0.15
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])
        rgba.append(mcolors.to_rgba(color, alpha=alpha))
        size = 30.0
        if mode == "impact":
            radius = _stick_radius(dict_impact.get(resi, 0.0), max_abs, size_by_impact)
            if radius > 0:
                size = 30.0 + 300.0 * radius  # radius ~0.18..0.73 -> marker ~84..249
        sizes.append(size)
        if in_window:
            win_coords.append(coord)
    if xs:
        # Backbone trace plus the per-residue Cα scatter.
        ax.plot(xs, ys, zs, color=ut.COLOR_STRUCT_MISSING, linewidth=0.8, alpha=0.6)
        ax.scatter(xs, ys, zs, c=rgba, s=sizes, depthshade=False, edgecolors="none")
    if focus == "zoom" and win_coords:
        win = np.vstack(win_coords)
        lo, hi = win.min(axis=0), win.max(axis=0)
        pad = 0.1 * np.maximum(hi - lo, 1.0)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if set_title:
        label = "pLDDT" if mode == "plddt" else "CPP feature impact"
        ax.set_title(f"{label} on structure")
    return ax


def render_mpl(records, dict_impact, max_abs, mode, focus, window_resis,
               size_by_impact, color_pos=None, color_neg=None, figsize=(6, 6)):
    """Build a matplotlib ``mplot3d`` Cα scatter coloured per residue; wrap as StructureView."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    draw_structure_mpl(ax=ax, records=records, dict_impact=dict_impact, max_abs=max_abs,
                       mode=mode, focus=focus, window_resis=window_resis,
                       size_by_impact=size_by_impact, color_pos=color_pos,
                       color_neg=color_neg)
    return StructureView(backend="mpl", fig=fig, ax=ax, dict_impact=dict_impact,
                         max_abs=max_abs, mode=mode)
