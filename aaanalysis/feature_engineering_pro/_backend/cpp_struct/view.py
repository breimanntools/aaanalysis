"""
This is a script for the backend view wrappers of the CPPStructurePlot class:
``StructureView`` (a thin handle over a py3Dmol view) and ``CombinedView`` (a
py3Dmol cartoon next to the CPPPlot feature-map image). Both expose a uniform
``show`` / ``write_html`` / ``_repr_html_`` surface so the structure is always a
real 3D cartoon, never a matplotlib scatter.
"""


# I Helper Functions
def _html_page(body):
    """Wrap an HTML body fragment in a minimal standalone page."""
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{body}</body></html>"


# II Main Functions
class StructureView:
    """Thin handle over a py3Dmol view.

    Returned by :meth:`CPPStructurePlot.map_structure`. A pure delegator: it forwards
    ``show`` / ``write_html`` / ``_repr_html_`` to the underlying py3Dmol view and otherwise
    carries the mapped impact for inspection.

    Attributes
    ----------
    backend : str
        Always ``'py3dmol'`` (kept for backward compatibility / introspection).
    dict_impact : dict
        ``{resi: impact}`` painted onto the structure.
    max_abs : float
        Normalisation constant for the impact ramp.
    mode : str
        ``'impact'`` or ``'plddt'``.
    """

    def __init__(self, backend="py3dmol", view=None, dict_impact=None, max_abs=None, mode=None):
        """Store the py3Dmol view object and the mapped-impact metadata."""
        self.backend = backend
        self._view = view
        self.dict_impact = dict_impact
        self.max_abs = max_abs
        self.mode = mode

    @property
    def view(self):
        """The underlying ``py3Dmol`` view object."""
        return self._view

    def show(self):
        """Display the interactive py3Dmol viewer."""
        return self._view.show()

    def write_html(self, path):
        """Write a self-contained interactive HTML file (py3Dmol export)."""
        path = str(path)
        self._view.write_html(path)
        return path

    def _repr_html_(self):
        """Notebook rich display: the self-contained py3Dmol widget HTML."""
        return self._view._make_html()


class CombinedView:
    """Structure cartoon (py3Dmol) and the CPP feature map (image) shown side by side.

    Returned by :meth:`CPPStructurePlot.plot_combined`. Lays the interactive py3Dmol cartoon
    next to the rendered :meth:`CPPPlot.feature_map` image, reproducing the deployed app's
    layout. ``write_html`` exports the pair as one self-contained page; ``savefig`` saves the
    feature-map panel as a static image (PNG/PDF/...) - the 3D cartoon is interactive and has no
    headless image, so it is captured via ``write_html`` or the viewer's screenshot button.

    Attributes
    ----------
    dict_impact : dict
        ``{resi: impact}`` painted onto the structure.
    max_abs : float
        Normalisation constant for the impact ramp.
    mode : str
        ``'impact'`` or ``'plddt'``.
    """

    def __init__(self, view=None, feature_map_png_b64=None, dict_impact=None,
                 max_abs=None, mode=None):
        """Store the py3Dmol view, the feature-map PNG, and the mapped-impact metadata."""
        self._view = view
        self._png_b64 = feature_map_png_b64
        self.dict_impact = dict_impact
        self.max_abs = max_abs
        self.mode = mode

    @property
    def view(self):
        """The underlying ``py3Dmol`` view object (the structure panel)."""
        return self._view

    def _body(self):
        """Side-by-side HTML body: the py3Dmol cartoon (left) and the feature map (right)."""
        struct_html = self._view._make_html()
        img_html = (f"<img src='data:image/png;base64,{self._png_b64}' "
                    f"style='max-width:100%;height:auto;'/>") if self._png_b64 else ""
        return (f"<div style='display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;'>"
                f"<div>{struct_html}</div><div>{img_html}</div></div>")

    def show(self):
        """Display the combined view inline (structure + feature map)."""
        from IPython.display import HTML, display
        return display(HTML(self._body()))

    def write_html(self, path):
        """Write the combined structure + feature-map view as one self-contained HTML page."""
        path = str(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_html_page(self._body()))
        return path

    def savefig(self, path, dpi=None):
        """Save the **feature-map** panel as a static image (format inferred from ``path``).

        The 3D structure panel is interactive: py3Dmol renders it client-side and cannot be
        rasterized from Python (even in a notebook), so there is no headless image of the cartoon.
        ``savefig`` therefore captures the feature-map panel - the quantitative, paper-ready
        figure - in the format given by ``path``'s extension (e.g. ``.png``, ``.pdf``, ``.jpg``,
        ``.tiff``). For the 3D structure use :meth:`write_html` (the interactive viewer, which also
        has a built-in screenshot button) or compose the two panels yourself. The pixels are
        already rendered at the ``feature_map_dpi`` passed to :meth:`plot_combined`; ``dpi`` only
        sets the saved file's resolution metadata.

        Raises
        ------
        RuntimeError
            If this view has no feature-map panel (``feature_map`` was disabled).
        """
        if not self._png_b64:
            raise RuntimeError("This CombinedView has no feature-map panel to save.")
        import io
        import base64
        from PIL import Image
        path = str(path)
        save_kwargs = {"dpi": (dpi, dpi)} if dpi is not None else {}
        with Image.open(io.BytesIO(base64.b64decode(self._png_b64))) as img:
            # PDF / JPEG have no alpha channel; flatten RGBA onto white before saving to those.
            if path.lower().endswith((".pdf", ".jpg", ".jpeg")) and img.mode == "RGBA":
                bg = Image.new("RGB", img.size, "white")
                bg.paste(img, mask=img.split()[-1])
                bg.save(path, **save_kwargs)
            else:
                img.save(path, **save_kwargs)
        return path

    def _repr_html_(self):
        """Notebook rich display: the side-by-side structure + feature-map HTML."""
        return self._body()


class LinkedView:
    """Feature-map ↔ structure linked view.

    Returned by :meth:`CPPStructurePlot.plot_linked`. Holds a self-contained HTML body in
    which hovering a feature-map column highlights the corresponding residue in the 3Dmol
    cartoon (the deployed app's interaction). ``write_html`` exports it as a standalone page.

    Attributes
    ----------
    dict_impact : dict
        ``{resi: impact}`` painted onto the structure.
    max_abs : float
        Normalisation constant for the impact ramp.
    mode : str
        ``'impact'`` or ``'plddt'``.
    """

    def __init__(self, html_body=None, dict_impact=None, max_abs=None, mode=None):
        """Store the linked HTML body and the mapped-impact metadata."""
        self._html = html_body
        self.dict_impact = dict_impact
        self.max_abs = max_abs
        self.mode = mode

    def show(self):
        """Display the linked view inline (Jupyter)."""
        from IPython.display import HTML, display
        return display(HTML(self._html))

    def write_html(self, path):
        """Write the linked structure + feature-map view as one self-contained HTML page."""
        from .linked_html import page
        path = str(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(page(self._html))
        return path

    def _repr_html_(self):
        """Notebook rich display: the feature-map ↔ structure linked HTML."""
        return self._html
