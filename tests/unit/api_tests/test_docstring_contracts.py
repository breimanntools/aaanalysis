"""Discoverability guard: load-bearing API contracts must live in the numpydoc
docstrings users actually read, not only in ``CONTEXT.md`` (issue #135).

This codifies the KPI of that issue — *a reader can determine pro-gating,
``[0, 1]`` normalization, and the ``get_parts`` -> ``run_num`` call order without
running code* — so the contracts cannot silently drift back out of the docstrings:

* every public symbol of every ``aaanalysis/*_pro/`` subpackage carries a ``[pro]``
  install marker in its summary — **auto-discovered**, so a newly added pro feature
  is covered without editing this test (it fails until the marker is present);
* ``CPP.run_num`` and ``NumericalFeature.get_parts`` state the ``[0, 1]`` contract
  and the two-step call order;
* ``SeqMut``'s ``df_seq`` consumers cross-link the canonical ``df_seq`` format spec
  (:meth:`SequenceFeature.get_df_parts`).

Docstrings are read straight from source via ``ast`` so the guard holds even when
the optional ``pro`` extras are not installed (the runtime objects are stubs then).
"""
import ast
import pathlib

import pytest

import aaanalysis as aa

PKG = pathlib.Path(aa.__file__).resolve().parent


def _find(nodes, name):
    """Return the top-level class / function node named ``name`` (or ``None``)."""
    kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in nodes:
        if isinstance(node, kinds) and node.name == name:
            return node
    return None


def _module_doc(rel_path, *, cls=None, func=None):
    """Return the docstring of a top-level function, class, or method (``cls.func``)."""
    tree = ast.parse((PKG / rel_path).read_text(encoding="utf-8"))
    if cls is not None:
        cls_node = _find(tree.body, cls)
        assert isinstance(cls_node, ast.ClassDef), f"class {cls} not found in {rel_path}"
        if func is None:
            return ast.get_docstring(cls_node)
        meth = _find(cls_node.body, func)
        assert meth is not None, f"method {cls}.{func} not found in {rel_path}"
        return ast.get_docstring(meth)
    fn_node = _find(tree.body, func)
    assert fn_node is not None, f"function {func} not found in {rel_path}"
    return ast.get_docstring(fn_node)


def _discover_pro_symbols():
    """Every public symbol of every ``aaanalysis/*_pro/`` subpackage, paired with
    the source file and docstring of its definition.

    Discovery is authoritative: it reads each ``*_pro/__init__.py`` ``__all__`` and
    follows the ``from ._module import Name`` line to the defining module. A new
    ``*_pro`` package or a new export is therefore picked up automatically.
    """
    out = []
    for init_path in sorted(PKG.glob("*_pro/__init__.py")):
        pkg = init_path.parent
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
        exported, import_of = [], {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                exported = [e.value for e in node.value.elts
                            if isinstance(e, ast.Constant)]
            elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                for alias in node.names:
                    import_of[alias.asname or alias.name] = node.module
        for name in exported:
            mod = import_of.get(name)
            assert mod is not None, (
                f"{pkg.name}.__all__ exports {name!r} but no "
                f"'from .<module> import {name}' line was found"
            )
            src = pkg / f"{mod.lstrip('.')}.py"
            doc = _module_doc(src.relative_to(PKG).as_posix(),
                              cls=name if name[:1].isupper() else None,
                              func=None if name[:1].isupper() else name)
            out.append((f"{pkg.name}.{name}", doc))
    return out


PRO_SYMBOLS = _discover_pro_symbols()


def test_pro_discovery_is_not_vacuous():
    # Guard the guard: if discovery silently finds nothing, the parametrized test
    # below would pass vacuously. Pin a floor at the symbols known to exist today.
    assert len(PRO_SYMBOLS) >= 6, f"only discovered {len(PRO_SYMBOLS)} pro symbols"


@pytest.mark.parametrize("qualname, doc", PRO_SYMBOLS, ids=[q for q, _ in PRO_SYMBOLS])
def test_pro_marker_in_summary(qualname, doc):
    summary = doc.split("\n\n", 1)[0] if doc else ""
    assert "[pro]" in summary, f"{qualname} summary lacks a '[pro]' gating marker"
    assert "aaanalysis[pro]" in summary, f"{qualname} summary lacks the install hint"


def test_run_num_states_normalization_and_order():
    doc = _module_doc("feature_engineering/_cpp.py", cls="CPP", func="run_num")
    assert "[0, 1]" in doc
    assert "get_parts" in doc and "run_num" in doc


def test_get_parts_states_normalization_and_order():
    doc = _module_doc("feature_engineering/_numerical_feature.py",
                      cls="NumericalFeature", func="get_parts")
    assert "[0, 1]" in doc
    assert "run_num" in doc


@pytest.mark.parametrize("method", ["mutate", "scan", "suggest"])
def test_seqmut_df_seq_cross_links_format_spec(method):
    doc = _module_doc("protein_engineering/_seqmut.py", cls="SeqMut", func=method)
    assert "SequenceFeature.get_df_parts" in doc, (
        f"SeqMut.{method} df_seq doc must cross-link the canonical format spec"
    )
