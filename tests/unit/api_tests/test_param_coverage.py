"""This is a script to test that every public parameter is exercised by a test.

Parameter-coverage meta-test (issue #84). Line coverage proves a line *ran*; it
does not prove that a public parameter was ever called with a non-default value.
A parameter can be added to a public signature, documented, and never exercised
while coverage stays green. This meta-test closes that gap:

1. Enumerate every public parameter of every symbol in ``aaanalysis.__all__`` —
   for classes, ``__init__`` plus every public (non-underscore) method; for
   functions, their parameters. Properties, ``self``/``cls`` and ``*args/**kwargs``
   are excluded; non-callable exports (``options``) are skipped.
2. Build the set of keyword-argument *names* used anywhere under ``tests/`` by
   parsing each test file's AST. A parameter counts as covered iff its name is
   used as a keyword argument at some call site.
3. Fail if any enumerated parameter is neither covered nor on the ``ALLOWLIST``
   (intentionally-untested params, each with a reason).

Detection is intentionally **global and name-based**: a parameter named ``loc``
counts as covered if ``loc=`` appears at any call site, not necessarily a call to
the owning symbol. This accepts a known false-positive surface for ambient names
(``verbose``, ``random_state``, ``n_jobs``, ``df_seq`` ...) in exchange for a
check that is simple and robust to test-layout drift. Per-(symbol, method, param)
attribution would need call-site type resolution and is out of scope.

Pro symbols whose optional dependency is absent resolve to ``missing_feature_stub``
lambdas (signature ``(*args, **kwargs)``); these are skipped with a recorded
reason, so the check is green in a core-only (``[dev]``) environment and enforces
the pro surface only where the ``[pro]`` extra is installed.

Extends the honest, package-only, ratcheted coverage policy.
"""
import ast
import inspect
from pathlib import Path

import pytest

import aaanalysis as aa

P = inspect.Parameter
_SKIP_PARAMS = {"self", "cls"}
_VAR_KINDS = (P.VAR_POSITIONAL, P.VAR_KEYWORD)
_TESTS_ROOT = Path(__file__).resolve().parents[2]


# I Helper Functions
def is_missing_feature_stub(obj):
    """True if ``obj`` is the ``missing_feature_stub`` lambda for an absent extra.

    Combines two signals so a real ``(*args, **kwargs)`` callable elsewhere is not
    mistaken for a stub: the object is a lambda AND its only parameters are
    ``*args, **kwargs``.
    """
    if getattr(obj, "__name__", None) != "<lambda>":
        return False
    try:
        kinds = [p.kind for p in inspect.signature(obj).parameters.values()]
    except (TypeError, ValueError):
        return False
    return kinds == [P.VAR_POSITIONAL, P.VAR_KEYWORD]


def _iter_methods(cls):
    """Yield ``(method_name, callable)`` for ``__init__`` + every public method.

    Properties are excluded (accessors, no params); ``getattr`` unwraps
    static/class methods to plain functions whose signatures carry no ``self``.
    """
    yield "__init__", cls.__init__
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if isinstance(inspect.getattr_static(cls, name, None), property):
            continue
        member = getattr(cls, name)
        if callable(member):
            yield name, member


def iter_public_params():
    """Yield ``(symbol, method_or_None, param)`` triples for the public surface.

    Skips non-callable exports and missing-feature stubs (see module docstring).
    """
    for symbol in aa.__all__:
        obj = getattr(aa, symbol)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue  # non-callable export (e.g. ``options`` config singleton)
        if is_missing_feature_stub(obj):
            continue  # pro/dev feature whose optional dependency is absent
        targets = _iter_methods(obj) if inspect.isclass(obj) else [(None, obj)]
        for method_name, fn in targets:
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            for pname, p in sig.parameters.items():
                if pname in _SKIP_PARAMS or p.kind in _VAR_KINDS:
                    continue
                yield symbol, method_name, pname


def collect_test_kwarg_names():
    """Return the set of keyword-argument names used at any call site under tests/."""
    names = set()
    for path in _TESTS_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg is not None:  # skip ``**unpacking``
                        names.add(kw.arg)
    return names


# Intentionally-untested public params, keyed by (symbol, param) -> reason.
# Only visual-only styling passthroughs belong here; behavioural params must be
# exercised by a real test, not allowlisted. Keep this list small (issue #84
# targets >=95% covered with the remainder justified here).
ALLOWLIST = {
    # AAlogoPlot single_logo/multi_logo: cosmetic passthrough to logomaker.
    ("AAlogoPlot", "logo_vpad"): "visual-only passthrough to logomaker",
    ("AAlogoPlot", "logo_vsep"): "visual-only passthrough to logomaker",
    ("AAlogoPlot", "logo_font_name"): "visual-only passthrough to logomaker",
    ("AAlogoPlot", "logo_color_scheme"): "visual-only passthrough to logomaker",
    ("AAlogoPlot", "name_data_fontsize"): "visual-only label styling",
    ("AAlogoPlot", "name_data_color"): "visual-only label styling",
    ("AAlogoPlot", "info_bar_color"): "visual-only label styling",
    ("AAlogoPlot", "target_p1_site"): "visual-only annotation marker",
    # AAMut / SeqMut have no dedicated unit-test suite yet (a whole-class gap, not
    # specific to these params; tracked as a separate issue). The global name-based
    # detection only surfaces their uniquely-named mutation-direction params; there
    # is no existing call site to extend, so they are justified-allowlisted here
    # until a dedicated suite lands.
    ("AAMut", "from_aa"): "AAMut/SeqMut lack a dedicated test suite (separate issue)",
    ("AAMut", "to_aa"): "AAMut/SeqMut lack a dedicated test suite (separate issue)",
    ("SeqMut", "to_aa"): "AAMut/SeqMut lack a dedicated test suite (separate issue)",
    # plot_legend: thin styling wrapper over matplotlib legend kwargs.
    ("plot_legend", "edgecolor"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "fontsize_title"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "frameon"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "handlelength"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "hatchcolor"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "loc"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "title"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "title_align_left"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "weight_font"): "matplotlib legend styling passthrough (visual-only)",
    ("plot_legend", "weight_title"): "matplotlib legend styling passthrough (visual-only)",
}


# II Main Functions
class TestParamCoverageMachinery:
    """Guards so the check can never pass vacuously (broken enumeration/parsing)."""

    def test_enumeration_is_non_trivial(self):
        # Lower bound holds in both the core-only ([dev]) and [pro] environments;
        # a regression to ~0 (e.g. a broken predicate) would make the gate vacuous.
        assert len(list(iter_public_params())) > 500

    def test_kwarg_collection_is_non_trivial(self):
        assert len(collect_test_kwarg_names()) > 300

    def test_stub_detection(self):
        stub = aa.missing_feature_stub("X", ImportError("shap"), mode="pro")
        assert is_missing_feature_stub(stub)
        assert not is_missing_feature_stub(aa.CPP)
        assert not is_missing_feature_stub(aa.load_dataset)

    def test_allowlist_entries_are_real_params(self):
        # Every allowlist key must name a parameter that actually exists on the
        # public surface, so renames/typos surface as a failure rather than rot.
        real = {(sym, param) for sym, _m, param in iter_public_params()}
        stale = sorted(k for k in ALLOWLIST if k not in real)
        assert not stale, (
            f"ALLOWLIST entries no longer match a public parameter (rename/typo?): {stale}"
        )


class TestParamCoverage:
    """The gate: every public parameter is covered by a test or justified-allowlisted."""

    def test_every_public_param_is_covered_or_allowlisted(self):
        covered = collect_test_kwarg_names()
        triples = list(iter_public_params())
        n_by_test = sum(1 for _s, _m, p in triples if p in covered)
        uncovered = [
            (sym, method, param)
            for sym, method, param in triples
            if param not in covered and (sym, param) not in ALLOWLIST
        ]
        # Recorded for drift (issue #84 KPI): the test-covered percentage excludes
        # the allowlist and must stay >=95%; allowlisted + uncovered make up the rest.
        pct_by_test = 100.0 * n_by_test / len(triples) if triples else 0.0
        print(
            f"\n[param-coverage] {len(triples)} public params | "
            f"test-covered {pct_by_test:.1f}% | allowlisted {len(ALLOWLIST)} | "
            f"uncovered {len(uncovered)}"
        )
        assert not uncovered, (
            "Public parameters with no test reference and no ALLOWLIST entry "
            f"({len(uncovered)}):\n"
            + "\n".join(f"  {s}.{m or '<call>'}({p})" for s, m, p in sorted(uncovered))
            + "\n\nAdd a test that passes the parameter by name, or add it to "
            "ALLOWLIST with a reason (visual-only passthroughs only)."
        )
