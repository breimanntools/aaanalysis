"""
This is a script for the backend StructureView wrapper of the CPPStructurePlot
class: a thin, pure delegator that gives the interactive (py3Dmol) and static
(matplotlib) renderers one uniform surface — ``show`` / ``write_html`` /
``savefig`` / ``_repr_html_`` — so the return type of ``map_structure`` does not
vary with the backend that fired.
"""
import base64
import io


# I Helper Functions
def _figure_to_png_b64(fig):
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


# II Main Functions
class StructureView:
    """Uniform handle over a py3Dmol view or a matplotlib Figure.

    Returned by :meth:`CPPStructurePlot.map_structure`. A pure delegator: it
    forwards ``show`` / ``write_html`` / ``savefig`` / ``_repr_html_`` to whichever
    backend produced it and otherwise carries the mapped impact for inspection.

    Attributes
    ----------
    backend : str
        ``'py3dmol'`` or ``'mpl'``.
    dict_impact : dict
        ``{resi: impact}`` painted onto the structure.
    max_abs : float
        Normalisation constant for the impact ramp.
    mode : str
        ``'impact'`` or ``'plddt'``.
    """

    def __init__(self, backend, view=None, fig=None, ax=None,
                 dict_impact=None, max_abs=None, mode=None):
        """Store the backend object and the mapped-impact metadata."""
        self.backend = backend
        self._view = view
        self._fig = fig
        self._ax = ax
        self.dict_impact = dict_impact
        self.max_abs = max_abs
        self.mode = mode

    @property
    def fig(self):
        """The matplotlib :class:`~matplotlib.figure.Figure` (``mpl`` backend only)."""
        return self._fig

    @property
    def ax(self):
        """The matplotlib :class:`~matplotlib.axes.Axes` (``mpl`` backend only)."""
        return self._ax

    @property
    def view(self):
        """The underlying ``py3Dmol`` view object (``py3dmol`` backend only)."""
        return self._view

    def show(self):
        """Display the structure (py3Dmol viewer; the Figure for the mpl backend)."""
        if self.backend == "py3dmol":
            return self._view.show()
        return self._fig

    def write_html(self, path):
        """Write a self-contained HTML file (py3Dmol export, or an embedded PNG for mpl)."""
        path = str(path)
        if self.backend == "py3dmol":
            self._view.write_html(path)
            return path
        png_b64 = _figure_to_png_b64(self._fig)
        html = (f"<html><body>"
                f"<img src='data:image/png;base64,{png_b64}'/>"
                f"</body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def savefig(self, path, **kwargs):
        """Save a static image (mpl backend only; py3Dmol exports via ``write_html``)."""
        if self.backend == "mpl":
            self._fig.savefig(str(path), **kwargs)
            return str(path)
        raise RuntimeError("'savefig' is only available for the matplotlib backend; "
                           "use 'write_html' for the py3Dmol backend")

    def _repr_html_(self):
        """Notebook rich display: the py3Dmol widget or an embedded PNG."""
        if self.backend == "py3dmol":
            return self._view._repr_html_()
        png_b64 = _figure_to_png_b64(self._fig)
        return f"<img src='data:image/png;base64,{png_b64}'/>"
