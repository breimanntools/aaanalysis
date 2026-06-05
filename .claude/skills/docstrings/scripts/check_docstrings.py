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
# same, but capturing the key so it can be checked against references.rst
CITATION_KEY_RE = re.compile(r"\[([A-Z][A-Za-z0-9]+)\]_")
# An UNDER CONSTRUCTION / not-yet-implemented stub: its docstring says so, or its
# body just raises NotImplementedError. Stubs are exempt from the whole convention
# (they are explicitly not ready) and are skipped, not reported as findings.
STUB_RE = re.compile(r"under construction|not yet implemented", re.IGNORECASE)
# Advisory findings: reported for visibility but never fail the run. A correct
# answer can be "no citation" (utility/helper classes legitimately omit), and the
# package convention often omits a Raises section for validator-style raises — so
# RAISES-UNDOCUMENTED is a "consider documenting" prompt, not a hard gate.
ADVISORY_CODES = {"CLASS-NO-CITATION", "RAISES-UNDOCUMENTED", "SUMMARY-ONLY"}
# Citation keys defined in docs/source/index/references.rst; populated in main().
# None means references.rst could not be located -> the undefined-citation check
# is skipped (it cannot be validated without the reference list).
REF_KEYS = None

# Cross-reference integrity: every :class:/:meth:/:func: role + See-Also target
# should resolve to a real public AAanalysis symbol. Registry filled in main()
# from a package-wide scan (so single-file runs still resolve cross-file refs).
KNOWN_CLASSES: set = set()
KNOWN_METHODS: set = set()   # "Class.method"
KNOWN_FUNCS: set = set()
ROLE_TARGET_RE = re.compile(r":(?:class|meth|func):`([^`]+)`")
# Roles to external libraries are valid and must not be flagged. A dotted target
# whose head is here (or a lowercase head we don't recognize) is treated external.
EXTERNAL_MODULES = {
    "pandas", "pd", "numpy", "np", "sklearn", "scipy", "matplotlib", "plt", "mpl",
    "shap", "Bio", "biopython", "requests", "logomaker", "seaborn", "sns",
    "upsetplot", "typing", "os", "sys", "re", "collections", "itertools",
    "functools", "pathlib", "warnings", "json", "math", "random", "abc",
    "python", "self", "cls",
}

# code -> one-line description (printed as a legend)
CODES = {
    "CLASS-SUMMARY-VERB": "class summary is an imperative verb, not a noun phrase",
    "CLASS-NO-CITATION": "class summary lacks a [Key]_ citation (ADVISORY — only add "
                         "one that genuinely describes the class; never invent one)",
    "CITATION-UNDEFINED": "[Key]_ citation has no '.. [Key]' definition in "
                          "references.rst (typo, or a reference that does not exist)",
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
    "XREF-UNRESOLVED": "a :class:/:meth:/:func: target does not resolve to a known "
                       "public AAanalysis symbol (typo or stale cross-reference)",
    "RAISES-UNDOCUMENTED": "body raises an exception but the docstring has no "
                           "'Raises' section",
    "SUMMARY-ONLY": "docstring is summary-only (ADVISORY — add a short plain-language "
                    "description: what it does, cited tool [Key]_, key cross-refs)",
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


def has_extended_description(summary_lines) -> bool:
    """True if the pre-section block has a description paragraph beyond the summary.

    Paragraphs are blank-line-separated blocks; directive blocks (``.. versionadded``,
    ``.. note``) are not prose. ``>= 2`` prose paragraphs means a one-line (or wrapped)
    summary IS followed by an expanded plain-language description.
    """
    paras, cur = [], []
    for line in summary_lines:
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    prose = [p for p in paras if not p[0].startswith("..")]
    return len(prose) >= 2


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
    check_citation_keys(doc, add)
    check_xrefs(doc, add)
    if not has_extended_description(sections.get("_summary", [])):
        add("SUMMARY-ONLY", first_line[:70])
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


def load_reference_keys(paths):
    """Parse the citation keys defined in ``docs/source/index/references.rst``.

    Locates references.rst by walking up from each scanned path (and the cwd),
    then collects every ``.. [Key]`` definition. Returns the set of defined keys,
    or ``None`` if references.rst cannot be found (the undefined-citation check is
    then skipped — it cannot be validated without the reference list).
    """
    seeds = [Path(p).resolve() for p in (paths or [])] + [Path.cwd().resolve()]
    for seed in seeds:
        for base in [seed] + list(seed.parents):
            cand = base / "docs" / "source" / "index" / "references.rst"
            if cand.is_file():
                text = cand.read_text(encoding="utf-8")
                return set(re.findall(r"^\.\.\s+\[([^\]]+)\]", text, re.MULTILINE))
    return None


def check_citation_keys(doc, add):
    """Flag any ``[Key]_`` whose ``Key`` is not defined in references.rst.

    Catches typo'd or fabricated citations. It cannot judge *relevance* — a
    defined-but-inappropriate citation (e.g. the project paper slapped on an
    unrelated utility class) is a human call; see the skill's guidance.
    """
    if REF_KEYS is None:
        return
    for key in sorted(set(CITATION_KEY_RE.findall(doc))):
        if key not in REF_KEYS:
            add("CITATION-UNDEFINED", key)


def collect_symbols(paths, public):
    """Populate KNOWN_CLASSES / KNOWN_METHODS / KNOWN_FUNCS from a package scan.

    Scans the whole package (located via the scanned paths / cwd) so a single-file
    run can still resolve cross-file references like ``:meth:`CPP.run_num```.
    """
    # Find the package root (the dir containing __init__.py with __all__).
    seeds = [Path(p).resolve() for p in (paths or [])] + [Path.cwd().resolve()]
    pkg = None
    for seed in seeds:
        for base in [seed] + list(seed.parents):
            cand = base / "aaanalysis"
            if (cand / "__init__.py").is_file():
                pkg = cand
                break
            if base.name == "aaanalysis" and (base / "__init__.py").is_file():
                pkg = base
                break
        if pkg:
            break
    if pkg is None:
        return
    for f in iter_targets([str(pkg)]):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and (public is None or node.name in public):
                KNOWN_CLASSES.add(node.name)
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and (
                            not sub.name.startswith("_") or sub.name == "__init__"):
                        KNOWN_METHODS.add(f"{node.name}.{sub.name}")
            elif isinstance(node, ast.FunctionDef) and (public is None or node.name in public):
                KNOWN_FUNCS.add(node.name)


def check_xrefs(doc, add):
    """Flag :class:/:meth:/:func: targets that don't resolve to a known symbol.

    Conservative: external-library refs (``pandas.DataFrame``) and ambiguous bare
    lowercase names are left alone; the high-value catches are a CamelCase class
    target that doesn't exist (``AALogo`` vs ``AAlogo``) and a wrong method on a
    real class.
    """
    if not (KNOWN_CLASSES or KNOWN_FUNCS):
        return
    for raw in ROLE_TARGET_RE.findall(doc):
        t = raw.strip().lstrip("~")
        for pre in ("aaanalysis.", "aa."):
            if t.startswith(pre):
                t = t[len(pre):]
        t = t.split("(")[0]
        if not t:
            continue
        if "." in t:
            head = t.split(".")[0]
            if head in EXTERNAL_MODULES or t in KNOWN_METHODS:
                continue
            if head in KNOWN_CLASSES:
                add("XREF-UNRESOLVED", f"{raw} (no such method on {head})")
            elif head[:1].isupper():            # CamelCase head, not a known class
                add("XREF-UNRESOLVED", raw)
            # lowercase unknown head -> assume unlisted external module; skip
        else:
            if t in KNOWN_CLASSES or t in KNOWN_FUNCS:
                continue
            if any(m.rsplit(".", 1)[1] == t for m in KNOWN_METHODS):
                continue                         # bare method name of some class
            if t[:1].isupper():                  # CamelCase looks like a class
                add("XREF-UNRESOLVED", raw)
            # bare lowercase -> could be external func/param; skip


def body_raises(node):
    """True if the function body explicitly ``raise``s an exception (not bare re-raise)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc is not None:
            return True
    return False


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


def _body_raises_notimplemented(node):
    """True if the function body (sans docstring) is just ``raise NotImplementedError``."""
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    name = (exc.func if isinstance(exc, ast.Call) else exc)
    return isinstance(name, ast.Name) and name.id == "NotImplementedError"


def is_stub(node):
    """An UNDER CONSTRUCTION / not-yet-implemented symbol, exempt from the convention.

    Two precise signals (so a *documented* limitation of a real method — e.g. a
    ``Raises`` clause "X is not yet implemented for numerical mode" — is NOT
    mistaken for a stub): the marker appears in the **summary line** (e.g.
    ``UNDER CONSTRUCTION - ...``), or the function **body is solely**
    ``raise NotImplementedError``.
    """
    doc = (ast.get_docstring(node, clean=True) or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    if STUB_RE.search(summary):
        return True
    return isinstance(node, ast.FunctionDef) and _body_raises_notimplemented(node)


def check_file(path: Path, findings, public, skipped):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    def emit(code, lineno, symbol, detail=""):
        findings.append((code, str(path), lineno, symbol, detail))

    def is_public(name):
        return public is None or name in public

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_public(node.name):
            if is_stub(node):  # whole class + its methods are exempt
                skipped.append((node.name, str(path)))
                continue
            _check_class(node, emit, skipped)
        elif isinstance(node, ast.FunctionDef) and is_public(node.name):
            if is_stub(node):
                skipped.append((node.name, str(path)))
                continue
            _check_function(node, node.name, emit, is_method=False)


def _check_class(node, emit, skipped):
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
    check_citation_keys(doc, add)
    check_xrefs(doc, add)
    if not has_extended_description(sections.get("_summary", [])):
        add("SUMMARY-ONLY", first_line[:70])
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
            if is_stub(sub):  # not-yet-implemented method, exempt
                skipped.append((f"{node.name}.{sub.name}", "method"))
                continue
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
    if body_raises(node) and "Raises" not in sections:
        add("RAISES-UNDOCUMENTED")


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

    global REF_KEYS
    REF_KEYS = load_reference_keys(args.paths or ["aaanalysis"])
    if REF_KEYS is None:
        print("!! could not locate docs/source/index/references.rst; "
              "skipping the undefined-citation check", file=sys.stderr)
    collect_symbols(args.paths or ["aaanalysis"], public)
    if not (KNOWN_CLASSES or KNOWN_FUNCS):
        print("!! could not build the public-symbol registry; "
              "skipping the cross-reference check", file=sys.stderr)

    findings, skipped = [], []
    files = list(iter_targets(args.paths or ["aaanalysis"]))
    if args.fix:
        fixed = sum(autofix_file(f) for f in files)
        print(f"--fix applied {fixed} mechanical edit(s) across {len(files)} file(s).")
    for f in files:
        try:
            check_file(f, findings, public, skipped)
        except SyntaxError as e:
            print(f"!! could not parse {f}: {e}", file=sys.stderr)

    # Split into hard defects (fail the run) and advisory notes (never fail):
    # a correct answer can be "no citation", so CLASS-NO-CITATION must not gate.
    defects = [x for x in findings if x[0] not in ADVISORY_CODES]
    advisory = [x for x in findings if x[0] in ADVISORY_CODES]

    def _print_group(title, items):
        if not items:
            return
        print(f"\n=== {title} ({len(items)}) ===")
        items.sort(key=lambda x: (x[1], x[2]))
        cur_file = None
        for code, path, lineno, symbol, detail in items:
            if path != cur_file:
                print(f"\n{path}")
                cur_file = path
            tail = f"  — {detail}" if detail else ""
            print(f"  {lineno:>5}  {code:<20} {symbol}{tail}")

    _print_group("DEFECTS", defects)
    _print_group("ADVISORY (does not fail)", advisory)

    def _by_check(items):
        counts = {}
        for code, *_ in items:
            counts[code] = counts.get(code, 0) + 1
        for code in sorted(counts):
            print(f"  {counts[code]:>4}  {code:<20} {CODES.get(code, '')}")

    print(f"\nScanned {len(files)} file(s); "
          f"{len(defects)} defect(s), {len(advisory)} advisory, "
          f"{len(skipped)} stub(s) skipped.")
    if skipped:
        names = ", ".join(sorted(name for name, _ in skipped))
        print(f"  skipped (UNDER CONSTRUCTION / NotImplementedError): {names}")
    if defects:
        print("\nDefects by check:")
        _by_check(defects)
    if advisory:
        print("\nAdvisory by check (review, do not auto-fix by inventing content):")
        _by_check(advisory)
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
