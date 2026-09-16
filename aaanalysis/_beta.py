"""This is a script for the beta-feature (experimental API) overview of AAanalysis.

A public symbol is **beta** when its docstring carries a ``.. warning::`` block whose text
opens with the bold run ``**Experimental.**``. That marker in the code is the single source
of truth: ``get_beta_symbols`` discovers the beta set by inspecting ``aaanalysis.__all__``
and ``aaanalysis.pipe.__all__`` at run time, so the overview page can neither list a symbol
that has lost the marker nor miss one that has just gained it. ``render_beta_rst`` renders
the discovered set to the reStructuredText overview page, and the drift test in
``tests/unit/api_tests/test_beta_features.py`` asserts the committed page matches.

Both namespaces are scanned because a reader asking what is unstable does not care which
namespace a symbol lives in. The rendered name keeps the namespace visible: a top-level
class renders as ``AAPred``, a golden pipeline as ``ap.find_features``, so nobody reads the
table as promising a top-level ``aa.find_features``.

``DICT_BETA_PURPOSE`` holds the one-phrase "what it is for" of each beta symbol, keyed by
that rendered name, and is the only hand-written part. The set itself, and the release each
symbol was added in (read from the ``.. versionadded::`` directive of the same docstring),
come from the code; a symbol without a curated phrase falls back to its docstring summary
rather than failing.

Unlike the sibling ``._schemas``, this module imports ``aaanalysis`` lazily inside the
functions: it reads the assembled public API, so a module-level import would be circular.
Note that a ``[pro]``-gated symbol is not discovered in a base install (it is absent from
``__all__``, or present only as an install-hint stub without the marker) -- the drift test
accounts for that.
"""
import inspect
import re

# The docstring marker that makes a public symbol beta. Do not change it without also
# updating every ``.. warning::`` block that carries it.
BETA_MARKER = "**Experimental.**"

# One short "what it is for" phrase per beta symbol, keyed by rendered name (the only
# hand-written content).
DICT_BETA_PURPOSE = {
    # Top-level classes (``import aaanalysis as aa``).
    "SequenceFeatureTransformer": "Leak-free CPP feature selection as a scikit-learn "
                                  "transformer.",
    "SeqOpt": "Multi-objective directed evolution over sequence variants.",
    "SeqOptPlot": "Pareto-front and convergence plots for ``SeqOpt`` results.",
    "AAPred": "Evaluation and deployment of sequence-based prediction models.",
    "AAPredPlot": "Evaluation and prediction figures for ``AAPred`` results.",
    "ReliabilityModel": "Per-prediction trust: calibration, uncertainty, and "
                        "applicability domain.",
    "ReliabilityModelPlot": "Calibration and trust-axis plots for ``ReliabilityModel`` "
                            "outputs.",
    "ModelEvaluator": "Cross-validated evaluation and paired comparison of models.",
    "CPPStructurePlot": "CPP feature impact painted onto a 3D protein structure.",
    # Golden pipelines (``import aaanalysis.pipe as ap``).
    "ap.find_features": "Staged CPP AutoML search for a discriminating feature set.",
    "ap.predict_samples": "Training and comparison of predictors across feature sets and "
                          "models in one call.",
    "ap.explain_features": "Per-sample SHAP impact and the SHAP-coloured feature map.",
}

# The namespaces scanned, as (module path, rendered prefix). The prefix keeps the calling
# namespace visible in the table; top-level symbols carry no prefix.
LIST_BETA_NAMESPACES = [("aaanalysis", ""), ("aaanalysis.pipe", "ap.")]

VERSIONADDED_RE = re.compile(r"\.\.\s+versionadded::\s*(\S+)")
CITATION_RE = re.compile(r"\[[A-Z][A-Za-z0-9]+\]_")


# I Helper Functions
def _first_versionadded(doc):
    """Return the first ``.. versionadded::`` version in a docstring, else 'n/a'."""
    match = VERSIONADDED_RE.search(doc)
    return match.group(1) if match else "n/a"


def _summary_phrase(doc):
    """Fall back to the docstring's first sentence when no curated phrase is registered."""
    summary = doc.strip().split("\n\n")[0]
    summary = CITATION_RE.sub("", " ".join(summary.split())).replace(" .", ".")
    sentence = summary.split(". ")[0].strip()
    return sentence if sentence.endswith(".") else sentence + "."


def _scan_namespace(module, namespace, prefix):
    """Return one record per beta symbol of a single namespace, in its ``__all__`` order."""
    records = []
    for name in module.__all__:
        obj = getattr(module, name, None)
        doc = getattr(obj, "__doc__", None) or ""
        if BETA_MARKER not in doc:
            continue
        display = f"{prefix}{name}"
        records.append({"name": name,
                        "namespace": namespace,
                        "display": display,
                        "role": "class" if inspect.isclass(obj) else "func",
                        "purpose": DICT_BETA_PURPOSE.get(display) or _summary_phrase(doc),
                        "versionadded": _first_versionadded(doc)})
    return records


def _render_tool(rec):
    """Render one symbol as a cross-reference that shows how it is actually reached."""
    if not rec["display"].startswith("ap."):
        return f":class:`~{rec['namespace']}.{rec['name']}`"
    return f":{rec['role']}:`{rec['display']} <{rec['namespace']}.{rec['name']}>`"


# II Main Functions
def get_beta_symbols():
    """Discover the beta public symbols by scanning ``aaanalysis.__all__`` and
    ``aaanalysis.pipe.__all__`` for the ``**Experimental.**`` docstring marker; returns one
    record (name, namespace, display, role, purpose, versionadded) per symbol, top-level
    symbols first, each namespace in public-API order."""
    import importlib  # local: reads the assembled public API, circular at module level
    records = []
    for namespace, prefix in LIST_BETA_NAMESPACES:
        records.extend(_scan_namespace(importlib.import_module(namespace),
                                       namespace, prefix))
    return records


def render_beta_rst():
    """Render the discovered beta symbols to the reStructuredText overview page (single
    source of truth for the docs; a drift test asserts the committed page matches)."""
    out = []
    out.append(".. _beta_features:")
    out.append("")
    out.append("Beta Features")
    out.append("=============")
    out.append("")
    out.append("Some AAanalysis tools are still under active development and are marked "
               "beta. For a beta tool, the API (signatures, defaults, return objects) may "
               "change between minor releases without the usual deprecation cycle, so pin "
               "a version (e.g. ``pip install aaanalysis==1.1.0``) if you depend on the "
               "current behaviour. Everything else in the public API keeps the ordinary "
               "deprecation cycle.")
    out.append("")
    out.append("Each tool below repeats this warning in its own documentation. The list is "
               "generated from the code: a symbol appears here exactly when its docstring "
               "carries the ``**Experimental.**`` warning, and a drift test keeps this "
               "page in sync, so it cannot go stale.")
    out.append("")
    out.append("An entry written ``ap.<name>`` is a golden pipeline reached through "
               "``import aaanalysis.pipe as ap``; every other entry is a class reached "
               "through ``import aaanalysis as aa``.")
    out.append("")
    out.append(".. list-table::")
    out.append("   :header-rows: 1")
    out.append("   :widths: 26 56 18")
    out.append("")
    out.append("   * - Tool\n     - Purpose\n     - Added in")
    for rec in get_beta_symbols():
        out.append(f"   * - {_render_tool(rec)}\n"
                   f"     - {rec['purpose']}\n"
                   f"     - {rec['versionadded']}")
    out.append("")
    return "\n".join(out).rstrip() + "\n"
