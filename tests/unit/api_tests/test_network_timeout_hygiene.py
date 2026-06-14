"""Security guard: every network GET in the package is bounded by a ``timeout``.

A ``requests.get(url)`` with no ``timeout`` blocks forever if the remote server
never responds, which can hang a bulk fetch indefinitely. Two AST guards enforce
the bound (no network access, no pro deps needed):

* every low-level ``requests.get`` / session ``.get`` passes an explicit
  ``timeout=`` keyword;
* the shared transport seam ``http_get_`` *defines* a defaulted ``timeout``
  parameter — that default is the safety guarantee, so callers routing through
  the seam need not repeat it (requiring them to would be a false-positive trap).

A regression here means a new fetch site shipped unbounded. The audited call
sites and their disposition are documented in ``docs/guides/security_audit_88.md``.
"""
import ast
import pathlib

import aaanalysis

ROOT = pathlib.Path(aaanalysis.__file__).resolve().parent

# Attribute names that denote an HTTP GET we require an explicit ``timeout`` on.
_GET_ATTR_NAMES = {"get"}
# Shared transport seam(s) in data_handling_pro/_backend/_fetch.py: their *definition*
# must declare a defaulted ``timeout`` (the safety guarantee). Callers routing through
# the seam are not required to repeat ``timeout=`` — the default already bounds them.
_SEAM_FUNC_NAMES = {"http_get_"}
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

    Excludes ``dict.get`` / ``.get`` on plain objects: we treat a ``.get`` as a
    network GET only when its owner is ``requests``, a session-named binding
    (``session`` / ``_session`` / ``<obj>.session``), or a ``_get_session()``
    call — the shapes the pro fetch backends actually use.
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
    # <obj>.session.get(...) (e.g. self.session.get / client._session.get)
    if isinstance(owner, ast.Attribute) and owner.attr in {"session", "_session"}:
        return True
    return False


def _func_def_has_defaulted_timeout(func: ast.FunctionDef) -> bool:
    """True if ``func`` declares a ``timeout`` parameter that carries a default."""
    args = func.args
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    # defaults align to the tail of the positional parameters
    n_defaulted = len(args.defaults)
    defaulted_positional = {a.arg for a in positional[len(positional) - n_defaulted:]}
    if "timeout" in defaulted_positional:
        return True
    kwonly_with_default = {a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults)
                           if d is not None}
    return "timeout" in kwonly_with_default


def test_every_network_get_passes_explicit_timeout():
    """No ``requests.get`` / session ``.get`` ships without an explicit ``timeout=``."""
    offenders = []
    for path in _iter_py_files():
        source = path.read_text(encoding="utf-8")
        if "requests" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_requests_get(node) and not _has_timeout_kw(node):
                rel = path.relative_to(ROOT.parent)
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "network GET call(s) without an explicit timeout= (hang risk): "
        + ", ".join(offenders)
    )


def test_transport_seam_defines_default_timeout():
    """The ``http_get_`` seam *defines* a defaulted ``timeout`` (callers inherit it)."""
    found = {}
    for path in _iter_py_files():
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in _SEAM_FUNC_NAMES):
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _SEAM_FUNC_NAMES:
                found[node.name] = _func_def_has_defaulted_timeout(node)
    missing = sorted(name for name in _SEAM_FUNC_NAMES if not found.get(name))
    assert not missing, (
        "transport seam(s) without a defaulted timeout parameter "
        "(callers can no longer rely on a bounded default): " + ", ".join(missing)
    )


def test_guard_actually_detects_a_missing_timeout():
    """Self-check: the AST guards flag synthetic GETs w/o timeout and accept timed ones."""
    # raw requests.get
    bad = "import requests\nrequests.get('http://x')\n"
    get_calls = [c for c in ast.walk(ast.parse(bad))
                 if isinstance(c, ast.Call) and _is_requests_get(c)]
    assert get_calls, "guard failed to recognize requests.get as a network GET"
    assert not _has_timeout_kw(get_calls[0])

    good = "import requests\nrequests.get('http://x', timeout=30)\n"
    good_call = [c for c in ast.walk(ast.parse(good))
                 if isinstance(c, ast.Call) and _is_requests_get(c)][0]
    assert _has_timeout_kw(good_call)

    # session-style fetch: <obj>.session.get(...) must also be recognized
    sess = "self.session.get('http://x')\n"
    sess_call = [c for c in ast.walk(ast.parse(sess))
                 if isinstance(c, ast.Call) and _is_requests_get(c)][0]
    assert not _has_timeout_kw(sess_call)

    # seam-definition check: a def with a defaulted timeout passes, one without fails
    seam_ok = ast.parse("def http_get_(url, timeout=30.0):\n    return url\n").body[0]
    seam_bad = ast.parse("def http_get_(url):\n    return url\n").body[0]
    assert _func_def_has_defaulted_timeout(seam_ok)
    assert not _func_def_has_defaulted_timeout(seam_bad)
