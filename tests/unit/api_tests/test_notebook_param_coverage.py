"""Test that every public parameter is *demonstrated* in its example notebook.

House rule (``.claude/rules/notebooks.md``): each public method/function has one
example notebook (wired via a ``.. include:: examples/<name>.rst`` in its docstring),
and that notebook must "cover every public parameter of the demonstrated method". This
gate makes the rule enforceable instead of aspirational — the AAPred notebooks shipped
without full parameter coverage precisely because nothing checked it.

Approach (deliberately name-based, mirroring ``test_param_coverage.py``):
1. For every symbol in ``aaanalysis.__all__``, enumerate its public methods/functions.
2. For each that carries an ``.. include:: examples/<name>.rst`` docstring directive,
   locate ``examples/**/<name>.ipynb`` and read its **code** cells.
3. Every public parameter of that method must appear by **name** in the notebook code
   (i.e. be passed, almost always as an explicit keyword — the house demo style).
Missing parameters fail the test unless justified in ``ALLOWLIST``.

This is a coarse, string-based signal: a param name appearing does not prove a
*meaningful* demonstration, but a name never appearing proves the parameter is
undemonstrated. Keeping the allowlist tiny keeps the gate honest.
"""
import ast
import inspect
import json
import re
from pathlib import Path

import pytest

import aaanalysis as aa

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_SKIP_PARAMS = {"self", "cls"}
_VAR_KINDS = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
_INCLUDE_RE = re.compile(r"\.\.\s+include::\s+examples/([\w\-]+)\.rst")

# Ambient params that are structurally hard to demo by name and add no teaching value.
# Keep this SMALL — behavioural params must be shown, not allowlisted.
_AMBIENT = {"ax"}

# (symbol, method_or_None, param) -> reason. Only genuine non-demonstrables.
ALLOWLIST: dict = {}

# Committed backlog of known-undemonstrated params (pyright-style ratchet). NEW notebooks
# and the prediction notebooks are NOT in it, so they must have zero gaps; existing rows are
# burned down over time and MUST be removed once fixed (guarded below).
_BASELINE_PATH = Path(__file__).with_name("notebook_param_coverage_baseline.txt")


def _load_baseline():
    if not _BASELINE_PATH.exists():
        return set()
    return {
        line.strip()
        for line in _BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _gap_key(symbol, method_name, param):
    return f"{symbol}::{method_name}::{param}"


def _current_gaps():
    """Set of ``symbol::method::param`` keys for every currently-undemonstrated param."""
    gaps = set()
    for symbol, method_name, _fn, nb, params in iter_documented_methods():
        if nb is None:
            continue
        code = _notebook_code(nb)
        for p in params:
            if p in _AMBIENT or (symbol, method_name, p) in ALLOWLIST:
                continue
            if not re.search(rf"\b{re.escape(p)}\b", code):
                gaps.add(_gap_key(symbol, method_name, p))
    return gaps


def is_missing_feature_stub(obj):
    """True if ``obj`` is the ``missing_feature_stub`` lambda for an absent extra."""
    if not (inspect.isfunction(obj) and obj.__name__ == "<lambda>"):
        return False
    kinds = [p.kind for p in inspect.signature(obj).parameters.values()]
    return kinds == [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]


def _iter_methods(cls):
    yield "__init__", cls.__init__
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if isinstance(inspect.getattr_static(cls, name, None), property):
            continue
        member = getattr(cls, name)
        if callable(member):
            yield name, member


def _include_name(fn):
    """Return the ``examples/<name>.rst`` stem from the docstring, or None."""
    doc = inspect.getdoc(fn) or ""
    m = _INCLUDE_RE.search(doc)
    return m.group(1) if m else None


def _find_notebook(name):
    hits = list(_EXAMPLES_ROOT.rglob(f"{name}.ipynb"))
    return hits[0] if hits else None


def _notebook_code(path):
    """Concatenated source of all code cells in the notebook."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )


def _public_params(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [p for p, v in sig.parameters.items()
            if p not in _SKIP_PARAMS and v.kind not in _VAR_KINDS]


def iter_documented_methods():
    """Yield ``(symbol, method_name, fn, notebook_path, [params])`` for methods that
    have an example notebook via their docstring include."""
    for symbol in aa.__all__:
        obj = getattr(aa, symbol)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if is_missing_feature_stub(obj):
            continue
        targets = _iter_methods(obj) if inspect.isclass(obj) else [(None, obj)]
        for method_name, fn in targets:
            name = _include_name(fn)
            if name is None:
                continue
            nb = _find_notebook(name)
            yield symbol, method_name, fn, nb, _public_params(fn)


class TestNotebookParamCoverageMachinery:
    """Guards so the gate can never pass vacuously."""

    def test_enumeration_is_non_trivial(self):
        rows = list(iter_documented_methods())
        assert len(rows) > 30, f"only {len(rows)} documented methods found — enumeration broke"

    def test_every_documented_notebook_exists(self):
        missing = [f"{s}.{m}" for s, m, _fn, nb, _p in iter_documented_methods() if nb is None]
        assert not missing, f"docstring includes with no example notebook: {missing}"

    def test_allowlist_entries_are_real(self):
        valid = {(s, m, p) for s, m, _fn, _nb, params in iter_documented_methods() for p in params}
        stale = [k for k in ALLOWLIST if k not in valid]
        assert not stale, f"stale ALLOWLIST entries (param no longer exists): {stale}"


class TestNotebookParamCoverage:
    def test_no_new_undemonstrated_params(self):
        """Every undemonstrated param must be a known-backlog baseline entry. Prediction
        notebooks and any new notebook are not baselined, so they must have zero gaps."""
        new = sorted(_current_gaps() - _load_baseline())
        assert not new, (
            "Example notebooks must demonstrate every public parameter "
            "(.claude/rules/notebooks.md). These are NOT in the baseline and must be fixed "
            "(add the param as an explicit kwarg in the notebook), not baselined:\n"
            + "\n".join(f"  {k}" for k in new)
        )

    def test_baseline_has_no_stale_entries(self):
        """Once a notebook is fixed, its baseline row must be removed — this forces the
        backlog to ratchet down and can never silently drift up."""
        stale = sorted(_load_baseline() - _current_gaps())
        assert not stale, (
            "These baseline rows no longer correspond to a real gap — the param is now "
            "demonstrated. Remove them from notebook_param_coverage_baseline.txt:\n"
            + "\n".join(f"  {k}" for k in stale)
        )

    def test_prediction_notebooks_fully_covered(self):
        """The prediction (AAPred/AAPredPlot) notebooks are held to zero gaps."""
        pred = sorted(k for k in _current_gaps() if k.startswith(("AAPred::", "AAPredPlot::")))
        assert not pred, "Prediction notebooks must demonstrate every param:\n" + "\n".join(f"  {k}" for k in pred)
