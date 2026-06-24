"""Contract guard: every public ``*Plot`` method returns one uniform ``(fig, ax)`` pair.

Historically the plotting methods returned three different shapes — ``(Figure, Axes)``,
a bare ``Axes``, and ``(Axes, DataFrame)`` — so a caller could not predict whether
``fig, ax = plot(...)`` or ``ax = plot(...)`` was correct. The contract is now a single
shape: every public ``*Plot`` method returns a :class:`aaanalysis.utils.FigAxResult`
(a thin ``(fig, ax)`` tuple subclass). Methods that also produce data (``centers`` /
``medoids``) expose it via a trailing-underscore instance attribute (``df_components_``)
rather than a third tuple element.

This meta-test enforces the contract by introspection so it cannot silently drift:

* every public method of every public ``*Plot`` class is annotated to return a
  ``(Figure, Axes)`` tuple, and
* its numpydoc ``Returns`` section names exactly ``fig`` first and ``ax`` second.

The single deliberate exception is ``CPPStructurePlot`` (pro), which returns a
``StructureView`` wrapper because its py3Dmol / matplotlib backends cannot share a
native return type — see the class docstring for the rationale.
"""
import inspect
import typing

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import aaanalysis as aa
import aaanalysis.utils as ut


# Documented exception: returns a StructureView wrapper, not (fig, ax).
EXCLUDED_CLASSES = {"CPPStructurePlot"}


def _public_plot_classes():
    """Every public ``*Plot`` class re-exported by the package (minus exceptions)."""
    out = []
    for name in aa.__all__:
        if name.endswith("Plot") and name not in EXCLUDED_CLASSES:
            out.append((name, getattr(aa, name)))
    return out


def _own_public_methods(cls):
    """Public methods defined on ``cls`` itself (skip inherited / dunder helpers)."""
    members = inspect.getmembers(cls, predicate=lambda o: inspect.isfunction(o) or inspect.ismethod(o))
    return [(n, o) for n, o in members
            if not n.startswith("_") and o.__qualname__.split(".")[0] == cls.__name__]


def _all_plot_methods():
    for cls_name, cls in _public_plot_classes():
        for meth_name, meth in _own_public_methods(cls):
            yield pytest.param(cls, meth, id=f"{cls_name}.{meth_name}")


def _returns_fig_ax_annotation(annotation):
    """True iff the annotation is a 2-tuple whose first element is ``Figure``.

    The second element may be ``Axes`` (single panel) or a sequence of ``Axes``
    (multi-panel methods such as ``eval`` / ``multi_logo``).
    """
    if typing.get_origin(annotation) is not tuple:
        return False
    args = typing.get_args(annotation)
    return len(args) == 2 and args[0] is Figure


def _returns_section_names(doc):
    """Return the ordered field names declared in a numpydoc ``Returns`` section."""
    if not doc:
        return []
    lines = doc.splitlines()
    names, in_returns = [], False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line == "Returns":
            in_returns = True
            continue
        if in_returns:
            # Section underline immediately after the header.
            if set(line) == {"-"}:
                continue
            # A blank line or the next section header ends the Returns block.
            if line == "" or (set(lines[i + 1].strip()) == {"-"} if i + 1 < len(lines) else False):
                break
            # Field declarations look like ``name : type`` at the section's indent.
            if " : " in line and not raw.startswith(" " * 12):
                names.append(line.split(" : ", 1)[0].strip())
    return names


# I FigAxResult behaves as a (fig, ax) pair that proxies attribute access to ax
class TestFigAxResult:
    """The return type itself: a 2-tuple that also forwards attributes to ``ax``."""

    def _make(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return ut.FigAxResult(fig, ax), fig, ax

    def test_is_tuple_of_len_two(self):
        r, fig, ax = self._make()
        assert isinstance(r, tuple) and len(r) == 2

    def test_unpacks_to_fig_ax(self):
        r, fig, ax = self._make()
        f, a = r
        assert f is fig and a is ax

    def test_indexing_and_properties(self):
        r, fig, ax = self._make()
        assert r[0] is fig and r[1] is ax
        assert r.fig is fig and r.ax is ax

    def test_attribute_fallthrough_to_ax(self):
        # Legacy ``ax = plot(...); ax.set_title(...)`` must keep working.
        r, fig, ax = self._make()
        r.set_title("hello")
        assert ax.get_title() == "hello"
        assert r.get_title() == "hello"

    def test_pickle_roundtrip(self):
        import pickle
        r, fig, ax = self._make()
        restored = pickle.loads(pickle.dumps(r))
        assert isinstance(restored, ut.FigAxResult) and len(restored) == 2


# II Every public *Plot method conforms to the (fig, ax) contract
class TestPlotReturnContract:
    """Introspection guard over every public ``*Plot`` method."""

    def test_at_least_one_plot_class_found(self):
        assert _public_plot_classes(), "no public *Plot classes discovered"

    @pytest.mark.parametrize("cls, method", list(_all_plot_methods()))
    def test_return_annotation_is_fig_ax(self, cls, method):
        annotation = inspect.signature(method).return_annotation
        assert annotation is not inspect.Signature.empty, \
            f"{cls.__name__}.{method.__name__} has no return annotation"
        assert _returns_fig_ax_annotation(annotation), (
            f"{cls.__name__}.{method.__name__} must return a (Figure, Axes) tuple, "
            f"got annotation {annotation!r}")

    @pytest.mark.parametrize("cls, method", list(_all_plot_methods()))
    def test_returns_docstring_names_fig_then_ax(self, cls, method):
        names = _returns_section_names(method.__doc__)
        assert names[:2] == ["fig", "ax"], (
            f"{cls.__name__}.{method.__name__} numpydoc Returns must declare "
            f"'fig' then 'ax'; found {names}")
