"""Discoverability guard: load-bearing API contracts must live in the numpydoc
docstrings users actually read, not only in ``CONTEXT.md`` (issue #135).

This codifies the KPI of that issue — *a reader can determine pro-gating,
``[0, 1]`` normalization, and the ``get_parts`` -> ``run_num`` call order without
running code* — so the contracts cannot silently drift back out of the docstrings:

* every ``[pro]`` class / function carries a ``[pro]`` marker in its summary;
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


def _module_doc(rel_path, *, cls=None, func=None):
    """Return the docstring of a top-level function, class, or method (``cls.func``)."""
    tree = ast.parse((PKG / rel_path).read_text(encoding="utf-8"))

    def _find(nodes, name, kinds):
        for node in nodes:
            if isinstance(node, kinds) and node.name == name:
                return node
        return None

    func_kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
    if cls is not None:
        cls_node = _find(tree.body, cls, (ast.ClassDef,))
        assert cls_node is not None, f"class {cls} not found in {rel_path}"
        if func is None:
            return ast.get_docstring(cls_node)
        meth = _find(cls_node.body, func, func_kinds)
        assert meth is not None, f"method {cls}.{func} not found in {rel_path}"
        return ast.get_docstring(meth)
    fn_node = _find(tree.body, func, func_kinds)
    assert fn_node is not None, f"function {func} not found in {rel_path}"
    return ast.get_docstring(fn_node)


# (rel_path, cls, func) for each [pro] class / function that must advertise gating.
PRO_TARGETS = [
    ("explainable_ai_pro/_shap_model.py", "ShapModel", None),
    ("data_handling_pro/_struct_preproc.py", "StructurePreprocessor", None),
    ("data_handling_pro/_annot_preproc.py", "AnnotationPreprocessor", None),
    ("seq_analysis_pro/_comp_seq_sim.py", None, "comp_seq_sim"),
    ("seq_analysis_pro/_filter_seq.py", None, "filter_seq"),
    ("seq_analysis_pro/_scan_motif.py", None, "scan_motif"),
]


@pytest.mark.parametrize("rel_path, cls, func", PRO_TARGETS)
def test_pro_marker_in_summary(rel_path, cls, func):
    doc = _module_doc(rel_path, cls=cls, func=func)
    summary = doc.split("\n\n", 1)[0] if doc else ""
    name = cls or func
    assert "[pro]" in summary, f"{name} summary lacks a '[pro]' gating marker"
    assert "aaanalysis[pro]" in summary, f"{name} summary lacks the install hint"


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
    doc = _module_doc("protein_design/_seqmut.py", cls="SeqMut", func=method)
    assert "SequenceFeature.get_df_parts" in doc, (
        f"SeqMut.{method} df_seq doc must cross-link the canonical format spec"
    )
