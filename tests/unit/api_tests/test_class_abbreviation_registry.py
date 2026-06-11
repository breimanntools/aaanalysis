"""Registry guard: every public class has one canonical abbreviation, used
identically as the example-notebook instance variable and the notebook filename
stem (see the *Class abbreviations* section of ``docs/source/index/docstring_guide.rst``).

This codifies the abbreviation registry so it cannot drift:

* every public class is registered, and abbreviations are unique;
* the registry is documented in the published style guide;
* every held class instance across **examples + tutorials + protocols** uses the
  bare registered abbreviation (the only exception is a genuinely concurrent
  second instance, named ``<abbr>_<qualifier>`` and listed in ``ALLOWED_SECONDARY``);
* every docstring ``.. include:: examples/<stem>.rst`` resolves to a notebook.

To register a new class, add it to ``REGISTRY`` **and** the guide table. Sequential
reuse of a class must reuse the bare abbreviation (reassign), not a suffixed name.
"""
import glob
import inspect
import json
import pathlib
import re

import aaanalysis as aa

ROOT = pathlib.Path(aa.__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "source" / "index" / "docstring_guide.rst"

# Class -> canonical abbreviation (single source of truth; mirrors the guide table).
REGISTRY = {
    "SequencePreprocessor": "sp",
    "EmbeddingPreprocessor": "ep",
    "AAlogo": "aal",
    "AAlogoPlot": "aal_plot",
    "AAWindowSampler": "aaws",
    "AAclust": "aac",
    "AAclustPlot": "aac_plot",
    "SequenceFeature": "sf",
    "NumericalFeature": "nf",
    "CPP": "cpp",
    "CPPGrid": "cppg",
    "CPPPlot": "cpp_plot",
    "dPULearn": "dpul",
    "dPULearnPlot": "dpul_plot",
    "AAMut": "aamut",
    "AAMutPlot": "aamut_plot",
    "SeqMut": "seqmut",
    "SeqMutPlot": "seqmut_plot",
    "TreeModel": "tm",
    "ShapModel": "sm",
    "StructurePreprocessor": "stp",
    "AnnotationPreprocessor": "ap",
}

# A class instance is named the bare abbreviation, ALWAYS. The only exception is
# a genuinely *concurrent* second instance of the same class that cannot be
# restructured away — it takes the ``<abbr>_<qualifier>`` form (never an unrelated
# word). These are the only two in the codebase; do not grow this list to excuse
# sequential reuse (reassign the canonical abbr instead).
ALLOWED_SECONDARY = {
    ("AAWindowSampler", "aaws_strict"),  # strict-threshold sampler alive beside aaws in aaws_sample_same_protein
    ("AAlogoPlot", "aal_plot_t"),        # per-iteration plotter beside aal_plot in aal_plot_single_logo
}

# Notebook surfaces whose instance variables are checked.
NOTEBOOK_GLOBS = ("examples", "tutorials", "protocols")

# Chained methods that return the instance itself (sklearn ``fit`` convention) —
# the variable still holds the tool, so it must use the abbreviation. Any other
# chained method (``.run``/``.get_*``/``.scan``/``.eval``/``.add_*``) returns data
# and names an output, which is exempt.
_SELF_RETURNING = ("fit", "fit_transform")


def _public_classes():
    return {n for n in aa.__all__ if inspect.isclass(getattr(aa, n))}


def _instance_notebooks():
    nbs = []
    for top in NOTEBOOK_GLOBS:
        nbs += glob.glob(str(ROOT / top / "**" / "*.ipynb"), recursive=True)
    return sorted(nbs)


def _match_paren(text, open_idx):
    """Return the index of the ``)`` matching the ``(`` at ``open_idx``."""
    depth = 0
    for k in range(open_idx, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return len(text) - 1


def _ctor_assignments(text):
    """Yield ``(var, cls)`` for assignments where ``var`` holds a class *instance*.

    Classification is by the **last** method in the chained RHS:
    ``aa.Cls(...)`` (no chain) and ``aa.Cls(...).fit(...)`` (self-returning final
    call) hold the instance; any chain whose final call returns data
    (``aa.Cls().run(...)``, ``aa.Cls().fit(...).add_feat_importance(...)``) names an
    output and is skipped."""
    for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*aa\.([A-Za-z]+)\(", text):
        cls = m.group(2)
        if cls not in REGISTRY:
            continue
        j = _match_paren(text, m.end() - 1) + 1  # just past the constructor call
        last_method = None
        while True:
            chain = re.match(r"\s*\.([A-Za-z_]\w*)\s*\(", text[j:])
            if not chain:
                break
            last_method = chain.group(1)
            j = _match_paren(text, j + chain.end() - 1) + 1
        # A trailing non-call ``.attr`` (e.g. ``aa.CPPPlot()._df_scales``) is an
        # attribute read -> the variable holds data, not the instance.
        if text[j:].lstrip().startswith("."):
            continue
        if last_method is not None and last_method not in _SELF_RETURNING:
            continue  # data-returning chain -> output variable, not the instance
        yield m.group(1), cls


def _code_blobs(nb_path):
    nb = json.loads(pathlib.Path(nb_path).read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))


# --- tests -------------------------------------------------------------------

def test_registry_covers_every_public_class():
    missing = _public_classes() - set(REGISTRY)
    assert not missing, f"public classes absent from the abbreviation REGISTRY: {sorted(missing)}"


def test_registry_has_no_stale_or_duplicate_entries():
    # every registered name is (or can be) a public class name
    stale = {c for c in REGISTRY if not c[0].isalpha()}
    assert not stale
    dups = {a for a in REGISTRY.values() if list(REGISTRY.values()).count(a) > 1}
    assert not dups, f"abbreviation collision (must be unique): {sorted(dups)}"


def test_registry_documented_in_style_guide():
    text = GUIDE.read_text(encoding="utf-8")
    undocumented = [
        (cls, abbr) for cls, abbr in REGISTRY.items()
        if f"``{cls}``" not in text or f"``{abbr}``" not in text
    ]
    assert not undocumented, (
        "Class abbreviations missing from docstring_guide.rst 'Class abbreviations' "
        f"table: {undocumented}"
    )


def test_notebook_instance_variables_match_registry():
    """examples + tutorials + protocols: every held class instance uses its abbreviation."""
    violations = []
    for nb in _instance_notebooks():
        for blob in _code_blobs(nb):
            for var, cls in _ctor_assignments(blob):
                if var != REGISTRY[cls] and (cls, var) not in ALLOWED_SECONDARY:
                    rel = pathlib.Path(nb).relative_to(ROOT)
                    violations.append(f"{rel}: `{var} = aa.{cls}()` should use `{REGISTRY[cls]}`")
    assert not violations, "Non-registry instance variables:\n" + "\n".join(violations)


def test_docstring_example_includes_resolve_to_notebooks():
    example_nbs = glob.glob(str(ROOT / "examples" / "**" / "*.ipynb"), recursive=True)
    stems = {pathlib.Path(p).stem for p in example_nbs}
    missing = []
    for py in glob.glob(str(ROOT / "aaanalysis" / "**" / "*.py"), recursive=True):
        for m in re.finditer(r"include:: examples/([A-Za-z0-9_]+)\.rst",
                             pathlib.Path(py).read_text(encoding="utf-8")):
            if m.group(1) not in stems:
                missing.append(f"{pathlib.Path(py).relative_to(ROOT)} -> examples/{m.group(1)}.rst")
    assert not missing, "Docstring .. include:: with no matching notebook:\n" + "\n".join(missing)
