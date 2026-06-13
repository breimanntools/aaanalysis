"""Security guard: every network GET in the package must pass an explicit ``timeout``.

A ``requests.get(url)`` with no ``timeout`` blocks forever if the remote server
never responds, which can hang a bulk fetch indefinitely. This meta-test walks
every ``aaanalysis/**/*.py`` source file, finds calls whose function resolves to
a ``get`` (``requests.get``, ``session.get``, ``http_get_`` transport seam, …)
that perform an HTTP fetch, and asserts each one passes a ``timeout`` keyword.

It is an AST guard (no network access, no pro deps needed): a regression here
means a new fetch site shipped without a timeout. The audited call sites and
their disposition are documented in ``docs/guides/security_audit_88.md``.
"""
import ast
import pathlib

import aaanalysis

ROOT = pathlib.Path(aaanalysis.__file__).resolve().parent

# Attribute names that denote an HTTP GET we require a ``timeout`` on. ``http_get_``
# is the shared transport seam in data_handling_pro/_backend/_fetch.py; it defaults
# timeout itself but we still require callers to be explicit where they pass one.
_GET_ATTR_NAMES = {"get"}
# Module-level functions that perform a fetch and must receive an explicit timeout.
_GET_FUNC_NAMES = {"http_get_"}
# Modules that legitimately call ``.get`` on a non-network object (dict.get etc.)
# are excluded by also requiring the call to look like a request (see below).


def _iter_py_files():
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _has_timeout_kw(call: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in call.keywords)


def _is_requests_get(call: ast.Call) -> bool:
    """True for ``requests.get(...)`` / ``<session>.get(...)`` HTTP calls.

    Excludes ``dict.get`` / ``.get`` on plain objects by requiring the call to
    sit in a module that imports ``requests`` AND the attribute owner to not be a
    literal dict. We key on the source line text via the AST: a ``.get`` whose
    first positional arg is a URL-ish expression. To stay robust we instead scope
    to files importing ``requests`` and treat any ``requests.get`` /
    ``*.session.get`` / ``_get_session().get`` as a network GET.
    """
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _GET_ATTR_NAMES:
        return False
    owner = func.value
    # requests.get(...)
    if isinstance(owner, ast.Name) and owner.id == "requests":
        return True
    # session.get(...) / resp = _get_session().get(...)
    if isinstance(owner, ast.Call):
        inner = owner.func
        if isinstance(inner, ast.Name) and inner.id == "_get_session":
            return True
    if isinstance(owner, ast.Name) and owner.id in {"session", "_session"}:
        return True
    return False


def _is_http_get_func(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Name) and func.id in _GET_FUNC_NAMES


def test_every_network_get_passes_explicit_timeout():
    """No ``requests.get`` / session ``.get`` / ``http_get_`` ships without ``timeout=``."""
    offenders = []
    for path in _iter_py_files():
        source = path.read_text(encoding="utf-8")
        if "requests" not in source and "http_get_" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_requests_get(node) or _is_http_get_func(node):
                if not _has_timeout_kw(node):
                    rel = path.relative_to(ROOT.parent)
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "network GET call(s) without an explicit timeout= (hang risk): "
        + ", ".join(offenders)
    )


def test_guard_actually_detects_a_missing_timeout():
    """Self-check: the AST guard flags a synthetic ``requests.get(url)`` w/o timeout."""
    bad = "import requests\nrequests.get('http://x')\n"
    tree = ast.parse(bad)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    get_calls = [c for c in calls if _is_requests_get(c)]
    assert get_calls, "guard failed to recognize requests.get as a network GET"
    assert not _has_timeout_kw(get_calls[0])

    good = "import requests\nrequests.get('http://x', timeout=30)\n"
    good_call = [c for c in ast.walk(ast.parse(good))
                 if isinstance(c, ast.Call) and _is_requests_get(c)][0]
    assert _has_timeout_kw(good_call)
