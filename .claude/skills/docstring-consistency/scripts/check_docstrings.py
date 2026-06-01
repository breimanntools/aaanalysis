#!/usr/bin/env python3
"""Deterministic AAanalysis docstring house-style checker.

Scans frontend modules (``aaanalysis/**/_<name>.py``, excluding ``_backend`` /
``_utils``) and flags STRUCTURAL drift from the CPP / dPULearn / AAclust house
style. Structure only — prose quality is a human call. See the skill's
``REFERENCE.md`` for the full rules behind each check code.

Usage::

    python check_docstrings.py [PATH ...]      # default: aaanalysis

Exit code is 1 if any finding is reported, else 0.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# Class summaries are noun phrases; these openers signal an imperative verb.
IMPERATIVE_VERBS = {
    "preprocess", "fetch", "run", "build", "compute", "encode", "load", "sample",
    "filter", "create", "generate", "extract", "apply", "get", "set", "plot",
    "make", "wrap", "convert", "identify", "cluster", "evaluate", "perform",
    "produce", "return", "calculate", "estimate", "map", "split", "scan",
}
KNOWN_SECTIONS = {
    "Parameters", "Returns", "Yields", "Raises", "Warns", "Warnings", "Notes",
    "See Also", "Examples", "References", "Attributes", "Other Parameters",
}
CANONICAL_DF_SEQ = "DataFrame containing an ``entry`` column"
# citation keys may carry digits + a trailing disambiguation letter: [Breimann24a]_
CITATION_RE = re.compile(r"\[[A-Z][A-Za-z0-9]+\]_")

# code -> one-line description (printed as a legend)
CODES = {
    "CLASS-SUMMARY-VERB": "class summary is an imperative verb, not a noun phrase",
    "CLASS-NO-CITATION": "class summary lacks a [Key]_ project citation",
    "CLASS-HAS-PARAMETERS": "Parameters in class docstring (move to __init__)",
    "NO-VERSIONADDED": "no '.. versionadded::' directive",
    "INIT-NO-DOCSTRING": "__init__ takes args but has no docstring",
    "MISSING-DOCSTRING": "public symbol has no docstring",
    "SUMMARY-ARROW": "summary line uses '->' / arrow shorthand",
    "METHOD-NO-EXAMPLES": "public method/function lacks an Examples '.. include::'",
    "WARNS-SECTION": "uses 'Warns' section; house style is 'Warnings'",
    "SEEALSO-NO-BULLET": "See Also entry is not a '* :role:' bullet",
    "SEEALSO-SPACE-COLON": "See Also gloss uses ' : ' (use ': ')",
    "INLINE-CITATION": "inline '.. [Key]' reference (cite [Key]_; defs in references.rst)",
    "FREETEXT-CITATION": "free-text '(Author et al. Year)' citation (use [Key]_)",
    "DFSEQ-BASELINE": "df_seq description diverges from the canonical baseline",
    "RETURNS-UNNAMED": "Returns value is unnamed (use 'name : type')",
    "NOTES-DASH-BULLET": "Notes uses '- ' bullets; house style is '* '",
    "EXAMPLES-PATH": "Examples include is not 'examples/<name>.rst'",
}
ROLE_TOKEN_RE = re.compile(r":(class|meth|func|ref|attr):`")


def parse_doc(doc: str):
    """Split a cleaned docstring into (first_line, sections{name: [lines]})."""
    lines = doc.split("\n")
    sections = {"_summary": []}
    cur = "_summary"
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (line.strip() in KNOWN_SECTIONS
                and set(nxt.strip()) == {"-"} and len(nxt.strip()) >= 3):
            cur = line.strip()
            sections.setdefault(cur, [])
            i += 2
            continue
        sections[cur].append(line)
        i += 1
    first_line = next((l.strip() for l in sections["_summary"] if l.strip()), "")
    return first_line, sections


def param_desc(param_lines, name):
    """Return the joined description text for parameter ``name`` (or None)."""
    desc, capturing, base = [], False, 0
    for line in param_lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not capturing:
            if re.match(rf"^{re.escape(name)}\s*:", stripped):
                capturing, base = True, indent
            continue
        if stripped == "":
            continue
        if indent <= base and re.match(r"^[\w.]+\s*:", stripped):
            break  # next parameter
        desc.append(stripped)
    return " ".join(desc).strip() if capturing else None


def has_examples_include(sections) -> bool:
    body = "\n".join(sections.get("Examples", []))
    return ".. include::" in body and "examples/" in body


def is_property(node) -> bool:
    for d in node.decorator_list:
        if isinstance(d, ast.Name) and d.id == "property":
            return True
        if isinstance(d, ast.Attribute) and d.attr in {"setter", "getter", "deleter"}:
            return True
    return False


def check_seealso(sections, add):
    lines = sections.get("See Also", [])
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return
    base = min(len(l) - len(l.lstrip()) for l in nonempty)  # bullet/entry level
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "` : " in line:
            add("SEEALSO-SPACE-COLON", s[:70])
        indent = len(line) - len(line.lstrip())
        if indent > base:
            continue  # continuation of a wrapped entry, not a new entry
        if not s.startswith("*"):
            # an entry (leading role token or 'name :') missing its '* ' bullet?
            if re.match(r"^:(class|meth|func|ref|attr):`", s) or re.match(r"^[\w.]+\s+:", s):
                add("SEEALSO-NO-BULLET", s[:70])


def check_common(doc, sections, first_line, add, *, is_method):
    if "→" in first_line or re.search(r"\s->\s", first_line):
        add("SUMMARY-ARROW", first_line[:70])
    if "Warns" in sections:
        add("WARNS-SECTION", "")
    if re.search(r"^\s*\.\.\s+\[[^\]]+\]", doc, re.MULTILINE):
        add("INLINE-CITATION", "")
    if re.search(r"\([A-Z][A-Za-z]+ et al\.?,?\s*\d{4}", doc):
        add("FREETEXT-CITATION", "")
    check_seealso(sections, add)
    if "Parameters" in sections:
        d = param_desc(sections["Parameters"], "df_seq")
        if d is not None and CANONICAL_DF_SEQ not in d:
            add("DFSEQ-BASELINE", d[:70])
    # Returns should be named ("name : type"). Two accepted type-only idioms are
    # excluded: a bare class name (sklearn self-return) and a polymorphic "X or Y".
    ret_first = next((l.strip() for l in sections.get("Returns", []) if l.strip()), "")
    if (ret_first and " : " not in ret_first
            and not re.match(r"^[A-Za-z_]\w*$", ret_first)
            and " or " not in ret_first):
        add("RETURNS-UNNAMED", ret_first[:70])
    # Notes use '*' bullets, not '-'.
    notes = [l for l in sections.get("Notes", []) if l.strip()]
    if notes:
        nbase = min(len(l) - len(l.lstrip()) for l in notes)
        for l in notes:
            if (len(l) - len(l.lstrip())) == nbase and l.strip().startswith("- "):
                add("NOTES-DASH-BULLET", l.strip()[:60])
                break
    # Examples include must target examples/<name>.rst.
    for l in sections.get("Examples", []):
        m = re.search(r"\.\.\s+include::\s+(\S+)", l)
        if m and not (m.group(1).startswith("examples/") and m.group(1).endswith(".rst")):
            add("EXAMPLES-PATH", m.group(1)[:70])


def load_public_names(api_path: Path):
    """Public API = names in ``__init__.__all__`` (+ appends + commented pro).

    Returns a set of names, or None if the API file can't be read (then the
    checker falls back to "every non-underscore symbol").
    """
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
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "__all__"):
            names.update(a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str))
    # commented-out pro entries inside __all__, e.g.  # "StructurePreprocessor",
    names.update(re.findall(r'#\s*"([A-Za-z_][A-Za-z0-9_]*)"', src))
    return names


def check_file(path: Path, findings, public):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    def emit(code, lineno, symbol, detail=""):
        findings.append((code, str(path), lineno, symbol, detail))

    def is_public(name):
        return public is None or name in public

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_public(node.name):
            _check_class(node, emit)
        elif isinstance(node, ast.FunctionDef) and is_public(node.name):
            _check_function(node, node.name, emit, is_method=False)


def _check_class(node, emit):
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        emit("MISSING-DOCSTRING", node.lineno, node.name)
        return
    first_line, sections = parse_doc(doc)
    add = lambda code, detail="": emit(code, node.lineno, node.name, detail)
    if first_line.split()[:1] and first_line.split()[0].lower() in IMPERATIVE_VERBS:
        add("CLASS-SUMMARY-VERB", first_line[:70])
    if not CITATION_RE.search(doc):
        add("CLASS-NO-CITATION", first_line[:70])
    if "Parameters" in sections:
        add("CLASS-HAS-PARAMETERS")
    if ".. versionadded::" not in doc:
        add("NO-VERSIONADDED")
    # __init__ + public methods
    for sub in node.body:
        if not isinstance(sub, ast.FunctionDef):
            continue
        if sub.name == "__init__":
            takes_args = len(sub.args.args) > 1 or sub.args.kwonlyargs
            if takes_args and not ast.get_docstring(sub, clean=True):
                emit("INIT-NO-DOCSTRING", sub.lineno, f"{node.name}.__init__")
        elif not sub.name.startswith("_") and not is_property(sub):
            _check_function(sub, f"{node.name}.{sub.name}", emit, is_method=True)


def _check_function(node, qualname, emit, *, is_method):
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        emit("MISSING-DOCSTRING", node.lineno, qualname)
        return
    first_line, sections = parse_doc(doc)
    add = lambda code, detail="": emit(code, node.lineno, qualname, detail)
    check_common(doc, sections, first_line, add, is_method=is_method)
    if not has_examples_include(sections):
        add("METHOD-NO-EXAMPLES")
    if ".. versionadded::" not in doc and not is_method:
        add("NO-VERSIONADDED")  # top-level public functions carry versionadded


def autofix_file(path: Path) -> int:
    """Apply the SAFE, deterministic fixes only (Warns->Warnings header rename,
    See-Also ' : ' gloss -> ': '). Everything else needs human/agent judgment and
    is left for the audit workflow. Returns the number of edits made."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out, n, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (line.strip() == "Warns"
                and set(nxt.strip()) == {"-"} and len(nxt.strip()) >= 3):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(indent + "Warnings")
            out.append(indent + "-" * len("Warnings"))
            n += 1
            i += 2
            continue
        if ROLE_TOKEN_RE.search(line) and "` : " in line:
            line = line.replace("` : ", "`: ")
            n += 1
        out.append(line)
        i += 1
    if n:
        path.write_text("\n".join(out), encoding="utf-8")
    return n


def iter_targets(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                parts = f.parts
                if "_backend" in parts or "_utils" in parts:
                    continue
                if not f.name.startswith("_"):  # skips utils.py, config.py, __init__.py
                    continue
                yield f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["aaanalysis"],
                    help="files or directories to scan (default: aaanalysis)")
    ap.add_argument("--api", default="aaanalysis/__init__.py",
                    help="package __init__.py defining __all__ (the public-API "
                         "filter); pass '' to check every non-underscore symbol")
    ap.add_argument("--fix", action="store_true",
                    help="apply the safe mechanical fixes (Warns->Warnings, "
                         "See-Also ' : ' -> ': ') in place, then report what remains")
    args = ap.parse_args(argv)

    public = load_public_names(Path(args.api)) if args.api else None
    if args.api and public is None:
        print(f"!! could not read public API from {args.api}; "
              f"checking all non-underscore symbols", file=sys.stderr)

    findings = []
    files = list(iter_targets(args.paths or ["aaanalysis"]))
    if args.fix:
        fixed = sum(autofix_file(f) for f in files)
        print(f"--fix applied {fixed} mechanical edit(s) across {len(files)} file(s).")
    for f in files:
        try:
            check_file(f, findings, public)
        except SyntaxError as e:
            print(f"!! could not parse {f}: {e}", file=sys.stderr)

    findings.sort(key=lambda x: (x[1], x[2]))
    cur_file = None
    for code, path, lineno, symbol, detail in findings:
        if path != cur_file:
            print(f"\n{path}")
            cur_file = path
        tail = f"  — {detail}" if detail else ""
        print(f"  {lineno:>5}  {code:<20} {symbol}{tail}")

    counts = {}
    for code, *_ in findings:
        counts[code] = counts.get(code, 0) + 1
    print(f"\nScanned {len(files)} file(s); {len(findings)} finding(s).")
    if counts:
        print("\nBy check:")
        for code in sorted(counts):
            print(f"  {counts[code]:>4}  {code:<20} {CODES.get(code, '')}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
