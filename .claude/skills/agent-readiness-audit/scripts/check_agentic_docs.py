#!/usr/bin/env python3
"""Deterministic front-door checker for the ``agent-readiness-audit`` skill.

Proves the *structural* half of agent-friendliness: every public subpackage has a
module-level ``__init__.py`` docstring (the local entry point), an ``__all__`` that
matches its actual re-exports (0 missing / 0 stale), no private leak, and a
top-level ``aaanalysis.__all__`` that equals the union of the subpackage surfaces.

Static + ``ast``-based: it never imports the package, so it is safe to run without
the optional ``pro`` / ``embed`` / ``dev`` dependencies installed (the very deps
that make a live import fail). Output mirrors ``docstrings/scripts/check_docstrings.py``:
**Defects** (exit != 0) vs **Advisory** (never fail).

Usage:
    python .claude/skills/agent-readiness-audit/scripts/check_agentic_docs.py [PATH]

PATH defaults to ``aaanalysis`` (the package root, i.e. the dir holding the
top-level ``__init__.py``). A single subpackage dir also works.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

# Dirs that are deliberately internal — no public front-door expected (see
# .claude/rules/repo-layout.md). A leading underscore already marks the rest.
SKIP_DIRS = {"_utils", "_backend", "_data", "__pycache__"}

# Defect codes fail the run; advisory codes never do (mirror check_docstrings.py).
ADVISORY_CODES = {"TOPLEVEL-ALL-DRIFT"}


class Finding:
    __slots__ = ("path", "code", "detail")

    def __init__(self, path: str, code: str, detail: str) -> None:
        self.path = path
        self.code = code
        self.detail = detail

    @property
    def is_defect(self) -> bool:
        return self.code not in ADVISORY_CODES


# ----------------------------------------------------------------------------- #
# I Helper Functions
# ----------------------------------------------------------------------------- #
def _parse(init_path: str) -> ast.Module:
    with open(init_path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=init_path)


def _all_list(tree: ast.Module):
    """Return the literal value assigned to ``__all__`` (a list[str]), or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                return list(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                return None
    return None


def _appended_names(tree: ast.Module):
    """Names added via ``__all__.append("X")`` (the top-level pro/dev stubs)."""
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "__all__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.append(node.args[0].value)
    return out


def _relative_reexports(tree: ast.Module):
    """Public names bound by a *relative* ``from .x import Y`` (the re-exports).

    Relative-only filtering cleanly drops absolute/stdlib imports (``import
    warnings``, ``from importlib.metadata import version``); underscore filtering
    drops private helpers.
    """
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
            for alias in node.names:
                bound = alias.asname or alias.name
                if not bound.startswith("_"):
                    names.add(bound)
    return names


def _public_subpackages(pkg_root: str):
    """Yield (name, init_path) for every public subpackage under pkg_root."""
    for name in sorted(os.listdir(pkg_root)):
        if name.startswith("_") or name in SKIP_DIRS:
            continue
        sub = os.path.join(pkg_root, name)
        init_path = os.path.join(sub, "__init__.py")
        if os.path.isdir(sub) and os.path.isfile(init_path):
            yield name, init_path


def _check_subpackage(name: str, init_path: str):
    findings = []
    rel = os.path.join(name, "__init__.py")
    tree = _parse(init_path)

    if ast.get_docstring(tree) is None:
        # PKG-NO-DOCSTRING (not INIT-NO-DOCSTRING): this is the *package module*
        # docstring. The docstrings skill's INIT-NO-DOCSTRING means a class
        # __init__ *method* without a docstring — a different thing.
        findings.append(Finding(rel, "PKG-NO-DOCSTRING",
                                "no module docstring (the front-door is empty)"))

    all_list = _all_list(tree)
    if all_list is None:
        findings.append(Finding(rel, "ALL-MISSING", "no __all__ defined"))
        return findings  # nothing more to compare against

    private = [x for x in all_list if x.startswith("_")]
    if private:
        findings.append(Finding(rel, "PRIVATE-LEAK",
                                f"private name(s) in __all__: {sorted(private)}"))

    reexports = _relative_reexports(tree)
    declared = set(all_list)
    missing = reexports - declared          # re-exported but not in __all__
    stale = declared - reexports - set(private)  # in __all__ but not re-exported
    if missing or stale:
        bits = []
        if missing:
            bits.append(f"missing from __all__: {sorted(missing)}")
        if stale:
            bits.append(f"stale in __all__ (not re-exported): {sorted(stale)}")
        findings.append(Finding(rel, "ALL-EXPORT-MISMATCH", "; ".join(bits)))

    return findings


def _check_toplevel(pkg_root: str, sub_all):
    """Advisory: top-level __all__ (static + appended) == union(subpackages) + options."""
    init_path = os.path.join(pkg_root, "__init__.py")
    if not os.path.isfile(init_path):
        return []
    tree = _parse(init_path)
    static = _all_list(tree) or []
    appended = _appended_names(tree)
    top = set(static) | set(appended)

    expected = set().union(*sub_all.values()) if sub_all else set()
    expected.add("options")  # re-exported from .config, not a subpackage

    missing = expected - top
    extra = top - expected
    findings = []
    if missing or extra:
        bits = []
        if missing:
            bits.append(f"subpackage symbols absent from aaanalysis.__all__: {sorted(missing)}")
        if extra:
            bits.append(f"in aaanalysis.__all__ but no subpackage exports it: {sorted(extra)}")
        findings.append(Finding("__init__.py", "TOPLEVEL-ALL-DRIFT", "; ".join(bits)))
    return findings


# ----------------------------------------------------------------------------- #
# II Main Functions
# ----------------------------------------------------------------------------- #
def run(path: str) -> int:
    path = os.path.abspath(path)
    # Resolve to a package root (a dir holding __init__.py).
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isfile(os.path.join(path, "__init__.py")):
        print(f"error: '{path}' is not a Python package (no __init__.py)", file=sys.stderr)
        return 2

    findings = []
    subpkgs = list(_public_subpackages(path))

    if subpkgs:
        # ``path`` is a package root (e.g. aaanalysis): check every child
        # subpackage and reconcile the top-level __all__ union.
        sub_all = {}
        for name, init_path in subpkgs:
            findings.extend(_check_subpackage(name, init_path))
            sub_all[name] = set(_all_list(_parse(init_path)) or [])
        findings.extend(_check_toplevel(path, sub_all))
        scanned = len(sub_all)
    else:
        # ``path`` is a single leaf subpackage (e.g. aaanalysis/metrics): check
        # just its own front-door — the union check applies only at the root.
        findings.extend(_check_subpackage(os.path.basename(path),
                                          os.path.join(path, "__init__.py")))
        scanned = 1

    defects = [f for f in findings if f.is_defect]
    advisory = [f for f in findings if not f.is_defect]

    print(f"agent-readiness-audit — front-door check on {path}")
    print(f"  scanned {scanned} public subpackage(s)\n")

    if defects:
        print("Defects:")
        for f in defects:
            print(f"  [{f.code}] {f.path}: {f.detail}")
    else:
        print("Defects: none — every public subpackage has a front-door docstring "
              "and an __all__ in sync with its re-exports.")

    if advisory:
        print("\nAdvisory:")
        for f in advisory:
            print(f"  [{f.code}] {f.path}: {f.detail}")

    print(f"\nSummary: {len(defects)} defect(s), {len(advisory)} advisory.")
    print("Note: 0 defects = structural front-door integrity (docstring exists + __all__ "
          "in sync), NOT prose/semantic quality — that is the docstrings skill's job.")
    return 1 if defects else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Front-door (__init__ docstring + __all__) checker.")
    ap.add_argument("path", nargs="?", default="aaanalysis",
                    help="package root or a subpackage dir (default: aaanalysis)")
    args = ap.parse_args()
    return run(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
