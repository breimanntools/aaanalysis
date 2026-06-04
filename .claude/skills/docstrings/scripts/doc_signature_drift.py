#!/usr/bin/env python3
"""Heuristic doc-vs-signature drift detector for AAanalysis docstrings.

The structural checker (``check_docstrings.py``) proves a docstring has the
right *shape*; it cannot tell that a documented ``default=`` or parameter set has
drifted from the actual signature. Those mismatches are the highest-value
("Necessary") findings of a critical doc review — and they are deterministic, so
this script surfaces candidates for an agent to verify.

It flags, per public function/method (stubs, privates and ``check_*`` helpers
skipped):

* ``DEFAULT-DRIFT``    — documented ``default=X`` differs from the signature default.
* ``PARAM-UNDOCUMENTED`` — a signature parameter has no ``Parameters`` entry.
* ``PARAM-EXTRA``      — a documented parameter is not in the signature.

These are *candidates*: numpydoc is free-form, so verify each against the source
before editing. Usage::

    python doc_signature_drift.py [PATH ...]      # default: aaanalysis
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

KNOWN_SECTIONS = {
    "Parameters", "Returns", "Yields", "Raises", "Warns", "Warnings", "Notes",
    "See Also", "Examples", "References", "Attributes", "Other Parameters",
}
STUB_RE = re.compile(r"under construction|not yet implemented", re.IGNORECASE)
# Capture a *balanced* default value — tuple/list/dict, quoted string, or a
# number/identifier — so internal commas/dots (``(6, 4)``, ``30.0``) are kept.
DEFAULT_RE = re.compile(
    r"default\s*[=:]?\s*"
    r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|'[^']*'|\"[^\"]*\"|[-\w.]+)")


def _norm(val: str) -> str:
    """Normalize a default token for comparison (drop quotes / spaces)."""
    if val is None:
        return ""
    return val.strip().replace(" ", "").replace("'", "").replace('"', "")


def _eq_default(doc_val: str, sig_val: str) -> bool:
    """True if the documented default matches the signature default.

    String-equal after normalization, or numerically equal (so ``30`` == ``30.0``).
    """
    a, b = _norm(doc_val), _norm(sig_val)
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def is_stub(node) -> bool:
    doc = (ast.get_docstring(node, clean=True) or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    if STUB_RE.search(summary):
        return True
    if not isinstance(node, ast.FunctionDef):
        return False
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return (len(body) == 1 and isinstance(body[0], ast.Raise)
            and isinstance((body[0].exc.func if isinstance(body[0].exc, ast.Call)
                            else body[0].exc), ast.Name)
            and (body[0].exc.func if isinstance(body[0].exc, ast.Call)
                 else body[0].exc).id == "NotImplementedError")


def _is_literal(node) -> bool:
    """True if a default is a literal we can compare to a documented value.

    A non-literal default — a ``ut.COL_POS`` constant reference, a
    ``shap.TreeExplainer`` factory, a function call — cannot be compared to a
    documented literal without resolving it, and the docstring documenting the
    *resolved value* of such a default is correct by convention. Skip those.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_literal(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal(k) and _is_literal(v)
                   for k, v in zip(node.keys, node.values))
    return False


def sig_defaults(node):
    """{param: signature-default-source} for params with a *literal* default."""
    a = node.args
    out = {}
    pos = a.args
    for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        if _is_literal(d):
            out[arg.arg] = ast.unparse(d)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None and _is_literal(d):
            out[arg.arg] = ast.unparse(d)
    out.pop("self", None)
    out.pop("cls", None)
    return out


def sig_param_names(node):
    a = node.args
    names = [x.arg for x in a.args + a.kwonlyargs]
    return [n for n in names if n not in ("self", "cls")]


def doc_params(doc):
    """{param: default-token-or-None} parsed from the Parameters section."""
    lines = doc.split("\n")
    # locate the Parameters block
    start = None
    for i, l in enumerate(lines):
        if l.strip() == "Parameters" and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            start = i + 2
            break
    if start is None:
        return None  # no Parameters section at all
    body = []
    for l in lines[start:]:
        if l.strip() in KNOWN_SECTIONS:
            break
        body.append(l)
    if not body:
        return {}
    base = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    params, cur = {}, None
    for l in body:
        if not l.strip():
            continue
        indent = len(l) - len(l.lstrip())
        m = re.match(r"^([A-Za-z_][\w, *]*?)\s*:", l.strip())
        if indent == base and m:
            # numpydoc allows a combined entry "a, b : type" documenting both.
            names = [n.strip().lstrip("*") for n in m.group(1).split(",") if n.strip()]
            dm = DEFAULT_RE.search(l)
            dv = dm.group(1) if dm else None
            for nm in names:
                params[nm] = dv
            cur = names[-1] if names else None
        elif cur and params.get(cur) is None:
            dm = DEFAULT_RE.search(l)
            if dm:
                params[cur] = dm.group(1)
    return params


SCALAR_ANN = {"int", "float", "str", "bool"}
DOC_SCALAR = {"int": "int", "integer": "int", "float": "float", "str": "str",
              "string": "str", "bool": "bool", "boolean": "bool"}


def _ann_scalar(node):
    """The single scalar (int/float/str/bool) an annotation denotes, else None.

    Unwraps ``Optional[int]`` / ``Union[int, None]``; ambiguous unions (e.g.
    ``Union[int, str]``) and non-scalar annotations return None (not comparable).
    """
    if isinstance(node, ast.Name) and node.id in SCALAR_ANN:
        return node.id
    if isinstance(node, ast.Subscript):
        s = node.slice
        elts = s.elts if isinstance(s, ast.Tuple) else [s]
        found = {r for r in (_ann_scalar(e) for e in elts) if r}
        return next(iter(found)) if len(found) == 1 else None
    return None


def doc_param_types(doc):
    """{param: documented-type-string} from the Parameters section (the part after ':')."""
    lines = doc.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l.strip() == "Parameters" and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            start = i + 2
            break
    if start is None:
        return {}
    out, base = {}, None
    for l in lines[start:]:
        if l.strip() in KNOWN_SECTIONS:
            break
        if not l.strip():
            continue
        if base is None:
            base = len(l) - len(l.lstrip())
        if (len(l) - len(l.lstrip())) == base:
            m = re.match(r"^([A-Za-z_][\w, *]*?)\s*:\s*(.+)$", l.strip())
            if m:
                for nm in (n.strip().lstrip("*") for n in m.group(1).split(",") if n.strip()):
                    out[nm] = m.group(2)
    return out


def _doc_scalars(type_str):
    return {DOC_SCALAR[w] for w in re.findall(r"[a-z]+", type_str.lower())
            if w in DOC_SCALAR}


def sig_annotations(node):
    a = node.args
    return {arg.arg: arg.annotation for arg in a.args + a.kwonlyargs
            if arg.arg not in ("self", "cls") and arg.annotation is not None}


def check_fn(node, qual, emit):
    if is_stub(node):
        return
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return
    dparams = doc_params(doc)
    if dparams is None:
        return  # nothing documented -> not this script's concern
    sdef = sig_defaults(node)
    snames = sig_param_names(node)
    for p, sd in sdef.items():
        # Signature default ``None`` usually means "resolved internally"; the
        # docstring then legitimately documents the *effective* default, so a
        # difference there is not drift. Only compare concrete-vs-concrete.
        if _norm(sd) == "None":
            continue
        if p in dparams and dparams[p] is not None and not _eq_default(dparams[p], sd):
            emit("DEFAULT-DRIFT", node.lineno, qual,
                 f"{p}: doc default={dparams[p].strip()} vs signature={sd}")
    for p in snames:
        if p not in dparams and not p.startswith("*"):
            emit("PARAM-UNDOCUMENTED", node.lineno, qual, p)
    for p in dparams:
        if p not in snames:
            emit("PARAM-EXTRA", node.lineno, qual, p)
    # Doc type vs annotation — only flag a clear scalar conflict (annotation is a
    # single scalar, the doc names a *different* scalar and not the right one).
    dtypes = doc_param_types(doc)
    for p, ann in sig_annotations(node).items():
        sc = _ann_scalar(ann)
        if sc is None or p not in dtypes:
            continue
        ds = _doc_scalars(dtypes[p])
        if ds and sc not in ds:
            emit("TYPE-DRIFT", node.lineno, qual,
                 f"{p}: doc type '{dtypes[p].split(',')[0].strip()}' vs signature {sc}")


def load_public(api_path: Path):
    if not api_path or not api_path.is_file():
        return None
    src = api_path.read_text(encoding="utf-8")
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                names.update(e.value for e in node.value.elts
                             if isinstance(e, ast.Constant) and isinstance(e.value, str))
    names.update(re.findall(r'#\s*"([A-Za-z_][A-Za-z0-9_]*)"', src))
    return names


def iter_targets(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if "_backend" in f.parts or "_utils" in f.parts:
                    continue
                if f.name.startswith("_"):
                    yield f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["aaanalysis"])
    ap.add_argument("--api", default="aaanalysis/__init__.py")
    args = ap.parse_args(argv)
    public = load_public(Path(args.api)) if args.api else None

    def is_public(name):
        return public is None or name in public

    findings = []
    files = list(iter_targets(args.paths or ["aaanalysis"]))
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            print(f"!! could not parse {f}: {e}", file=sys.stderr)
            continue

        def emit(code, lineno, symbol, detail, _f=f):
            findings.append((code, str(_f), lineno, symbol, detail))

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and is_public(node.name):
                if is_stub(node):
                    continue
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and (
                            sub.name == "__init__" or not sub.name.startswith("_")):
                        check_fn(sub, f"{node.name}.{sub.name}", emit)
            elif isinstance(node, ast.FunctionDef) and is_public(node.name):
                check_fn(node, node.name, emit)

    findings.sort(key=lambda x: (x[1], x[2]))
    cur = None
    for code, path, lineno, symbol, detail in findings:
        if path != cur:
            print(f"\n{path}")
            cur = path
        print(f"  {lineno:>5}  {code:<18} {symbol}  — {detail}")
    counts = {}
    for code, *_ in findings:
        counts[code] = counts.get(code, 0) + 1
    print(f"\nScanned {len(files)} file(s); {len(findings)} drift candidate(s) "
          f"(verify each against the source before editing).")
    for code in sorted(counts):
        print(f"  {counts[code]:>4}  {code}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
