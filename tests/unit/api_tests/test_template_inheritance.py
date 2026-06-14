"""Meta-test enforcing the Wrapper/Tool template contract on the public API.

``aaanalysis.template_classes`` defines two ABCs:

* :class:`Wrapper` — the ``.fit`` / ``.eval`` model contract.
* :class:`Tool` — the ``.run`` / ``.eval`` pipeline contract.

Historically the "Wrapper"/"Tool" classification was only a documentation
convention: some classes implemented the method set without inheriting the
matching ABC, so ``isinstance`` / IDEs / type checkers could not verify it
(issue #134). This test pins the contract: any public class that implements the
full method set of an ABC must actually inherit that ABC.

Keying:
* ``.fit`` + ``.eval``  -> must be a :class:`Wrapper`.
* ``.run`` + ``.eval``  -> must be a :class:`Tool`.

Classes carrying only ``.eval`` (the ``*Plot`` pairs) or only one half of a
pair (``SeqMut`` has ``.mutate``/``.scan``/``.suggest``/``.eval`` but no
``.run``) are intentionally not bound by either contract and are skipped.
Pro classes that degraded to a ``missing_feature_stub`` (a callable, not a
class) when their optional dependency is absent are skipped by the
``inspect.isclass`` filter.
"""
import inspect

import pytest

import aaanalysis as aa
from aaanalysis.template_classes import Tool, Wrapper


def _has(obj, method):
    return callable(getattr(obj, method, None))


def _public_classes():
    """Yield (name, class) for every public class exported by aaanalysis."""
    for name in aa.__all__:
        obj = getattr(aa, name)
        if inspect.isclass(obj):
            yield name, obj


WRAPPER_CLASSES = [
    (name, cls)
    for name, cls in _public_classes()
    if _has(cls, "fit") and _has(cls, "eval")
]
TOOL_CLASSES = [
    (name, cls)
    for name, cls in _public_classes()
    if _has(cls, "run") and _has(cls, "eval")
]


class TestTemplateInheritance:
    """Every public class implementing an ABC's method set inherits that ABC."""

    @pytest.mark.parametrize("name, cls", WRAPPER_CLASSES, ids=lambda x: x if isinstance(x, str) else "")
    def test_fit_eval_classes_are_wrappers(self, name, cls):
        assert issubclass(cls, Wrapper), (
            f"'{name}' implements .fit/.eval but does not inherit Wrapper "
            f"(template_classes.Wrapper); add the base."
        )

    @pytest.mark.parametrize("name, cls", TOOL_CLASSES, ids=lambda x: x if isinstance(x, str) else "")
    def test_run_eval_classes_are_tools(self, name, cls):
        assert issubclass(cls, Tool), (
            f"'{name}' implements .run/.eval but does not inherit Tool "
            f"(template_classes.Tool); add the base."
        )

    def test_some_wrapper_and_tool_classes_were_discovered(self):
        # Guard against the discovery silently yielding nothing (e.g. import break).
        wrapper_names = {name for name, _ in WRAPPER_CLASSES}
        tool_names = {name for name, _ in TOOL_CLASSES}
        assert {"AAclust", "dPULearn", "TreeModel"} <= wrapper_names
        assert {"CPP", "CPPGrid", "AAMut"} <= tool_names


class TestIssue134Acceptance:
    """The exact isinstance KPIs from issue #134 hold True."""

    def test_treemodel_is_wrapper(self):
        assert isinstance(aa.TreeModel(), Wrapper)

    def test_dpulearn_is_wrapper(self):
        assert isinstance(aa.dPULearn(), Wrapper)

    def test_shapmodel_is_wrapper(self):
        if not inspect.isclass(aa.ShapModel):
            pytest.skip("ShapModel is a missing-feature stub (aaanalysis[pro] not installed)")
        assert isinstance(aa.ShapModel(), Wrapper)

    def test_aamut_is_tool(self):
        assert isinstance(aa.AAMut(), Tool)


class TestSamplerNotBound:
    """AAWindowSampler is a sampler, not a Wrapper/Tool: it must NOT be bound."""

    def test_aa_window_sampler_is_not_wrapper_or_tool(self):
        # It only exposes sample_* methods (no .fit/.eval or .run/.eval pair),
        # so it is intentionally outside both contracts.
        assert not issubclass(aa.AAWindowSampler, (Wrapper, Tool))

    def test_seqmut_has_no_run_method(self):
        # SeqMut implements mutate/scan/suggest/eval but no .run, so it does not
        # satisfy the Tool ABC and is (correctly) not bound to it. Inheriting Tool
        # would require adding a .run method (a behavior/API change, out of scope
        # for the non-breaking #134).
        assert not _has(aa.SeqMut, "run")
        assert not issubclass(aa.SeqMut, Tool)
