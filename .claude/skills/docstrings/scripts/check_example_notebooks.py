#!/usr/bin/env python3
"""Check example notebooks against the house standard (docstring_guide.rst:
'Notebook content & structure' + 'Examples & verification').

Two failure modes that the blocking CI does NOT catch (it only checks that cells
run), so they ship silently and render as blank pages / undocumented params:

  1. EMPTY-OUTPUT  — a code cell has no saved output, so it renders no table/figure
     on Read the Docs (docs use nbsphinx_execute='never'; they show stored output).
  2. UNCOVERED-PARAM — a public parameter of the demonstrated method/function never
     appears anywhere in the notebook source (markdown or code), i.e. the example
     does not introduce every parameter.

Usage:
    check_example_notebooks.py [PATH ...]      # notebooks and/or dirs (default: examples/)
    check_example_notebooks.py --no-params ... # skip the param-coverage check

Param coverage maps a notebook ``examples/<sub>/<name>.ipynb`` to the public
symbol whose docstring includes ``examples/<name>.rst`` (resolved by importing
``aaanalysis`` and scanning ``__init__``-exported classes/functions). When the
symbol can't be resolved the param check is skipped for that notebook (reported
as a soft note, never a hard failure).
"""
import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXAMPLES = REPO / "examples"

# Names whose call means the cell is meant to render something.
_DISPLAY_CALLS = {"display", "display_df", "print", "show"}


def _iter_notebooks(paths):
    for p in paths:
        p = Path(p).resolve()
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*.ipynb")
                              if ".ipynb_checkpoints" not in q.parts)
        elif p.suffix == ".ipynb" and ".ipynb_checkpoints" not in p.parts:
            yield p


def _code_cells(nb):
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _expects_output(src):
    """True if the cell is *meant* to render output: a trailing bare expression
    (e.g. ``df``, ``df.head()``, ``aa.plot_x()``) or a display/print call."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    if not tree.body:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in _DISPLAY_CALLS:
                return True
    # A trailing bare *value* expression (``df``, ``df.shape``, ``df[0]``) renders
    # its repr. A trailing bare *call* (``lst.extend(...)``, ``plt.show()``) is
    # ambiguous — only the display calls above count — so it is not flagged here.
    last = tree.body[-1]
    if isinstance(last, ast.Expr) and isinstance(
        last.value, (ast.Name, ast.Attribute, ast.Subscript)
    ):
        return True
    return False


def check_empty_outputs(nb):
    """Return list of (cell_index, first_line) for cells meant to render output but
    committed with none (renders blank on RTD)."""
    bad = []
    for i, c in enumerate(_code_cells(nb)):
        src = _src(c).strip()
        if src and _expects_output(src) and not c.get("outputs"):
            bad.append((i, src.splitlines()[-1][:70]))
    return bad


def _resolve_symbol(nb_name):
    """Find the public class/function whose docstring includes examples/<nb_name>.rst."""
    try:
        import aaanalysis as aa
    except Exception:
        return None
    include = f"examples/{nb_name}.rst"
    for obj_name in getattr(aa, "__all__", dir(aa)):
        obj = getattr(aa, obj_name, None)
        if obj is None:
            continue
        members = [obj]
        if inspect.isclass(obj):
            members += [m for _, m in inspect.getmembers(obj, inspect.isfunction)]
        for m in members:
            doc = inspect.getdoc(m) or ""
            if include in doc:
                return m
    return None


def check_param_coverage(nb, nb_name):
    """Return (missing_params, note). missing = public params absent from notebook text."""
    sym = _resolve_symbol(nb_name)
    if sym is None:
        return [], f"param-coverage skipped (no symbol includes examples/{nb_name}.rst)"
    try:
        sig = inspect.signature(sym)
    except (ValueError, TypeError):
        return [], "param-coverage skipped (no signature)"
    text = "\n".join(_src(c) for c in nb.get("cells", []))
    skip = {"self", "args", "kwargs"}
    params = [p for p in sig.parameters
              if p not in skip and not p.startswith("_")]
    missing = [p for p in params if p not in text]
    return missing, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[str(EXAMPLES)])
    ap.add_argument("--no-params", action="store_true", help="skip param-coverage check")
    args = ap.parse_args()

    notebooks = list(_iter_notebooks(args.paths or [str(EXAMPLES)]))
    n_fail = 0
    notes = []
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        rel = nb_path.relative_to(REPO) if nb_path.is_relative_to(REPO) else nb_path
        empty = check_empty_outputs(nb)
        if empty:
            n_fail += 1
            print(f"\nEMPTY-OUTPUT  {rel}")
            for idx, line in empty:
                print(f"    code cell {idx}: `{line}` has no saved output "
                      f"(renders blank on RTD — re-run & save with outputs)")
        if not args.no_params:
            missing, note = check_param_coverage(nb, nb_path.stem)
            if missing:
                n_fail += 1
                print(f"\nUNCOVERED-PARAM  {rel}")
                print(f"    parameters never mentioned in the notebook: {missing}")
            elif note:
                notes.append(f"  - {rel}: {note}")

    print(f"\nScanned {len(notebooks)} notebook(s); {n_fail} with defects.")
    if notes:
        print("Notes:")
        print("\n".join(notes))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
