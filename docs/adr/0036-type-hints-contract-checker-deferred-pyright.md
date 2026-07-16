# ADR-0036 — Ship `py.typed` and adopt pyright (non-blocking) now; type hints are the contract

Status: Accepted — 2026-06-22

(Filename slug retains "deferred" from the superseded draft; the decision below is to
**adopt now**, not defer. Rename the slug at commit if desired.)

## Context

An agent-readiness review recommended a static type checker so an agent gets a read-time
contract plus a checker that catches mistakes before runtime. The package already **mandates
type hints on every public parameter and return** (`code-conventions.md`) and validates
inputs at runtime via `ut.check_*` (ADR-0038 D11) — but it ships **no `py.typed` marker**, so
those annotations are **invisible to every downstream consumer**, including ProtXplain
(which wraps AAanalysis and would type-check against it). The standing `sharp-edges.md`
position was "no type checker until v2."

The **moat argument** resolves the tension in favour of acting now: agent-friendliness of the
OSS substrate is **pure upside** (the moat lives in ProtXplain + the science + the data, not
in package friction), and ProtXplain is the primary consumer — so making the substrate
type-legible benefits *us* first. The withholding that matters (the typed verb/MCP layer)
already lives in ProtXplain per ADR-0038; nothing is gained by keeping the package
type-opaque.

## Decision

D1. **Type hints on public signatures are the read-time contract** and stay mandatory
    (`code-conventions.md`; `Optional[...]` for `None`-defaults; `ut.ArrayLike1D/2D`).

D2. **Ship `py.typed` (PEP 561).** This is the keystone: it turns the annotations the package
    already has into a *consumable* contract for ProtXplain and any downstream type-checker.

D3. **Adopt pyright now — `basic` mode, non-blocking (advisory CI), public-API-first.**
    `pyrightconfig.json` includes the public surface and excludes `_backend` initially,
    expanding inward over time. pyright (not mypy) for speed + editor/Pylance parity. The CI
    job is **advisory and never gates merge** through v1.x.

## Rejected alternatives

- **Defer the checker to v2 (the prior `sharp-edges.md` stance).** Forgoes the cheap,
  high-value `py.typed` win and leaves ProtXplain without consumable types. The moat argument
  shows substrate type-legibility is pure upside, so deferral has no defensive payoff.
- **A blocking type gate.** Too disruptive for a scientific numpy/pandas codebase at TRL 4–5;
  advisory keeps annotations honest without freezing velocity.
- **mypy as the checker.** Slower, weaker inference, not the editor engine (Pylance = pyright).
- **Strict mode / whole-package at once.** Highest cost, lowest marginal value; public-API
  signatures are the smallest, highest-leverage surface, and being wrong there is most
  expensive (it is the published contract).

## Consequences

- **Shipping `py.typed` is a semver promise:** downstream now type-checks against us, so the
  public **signatures** must be *honest* (``Optional[...]`` for None-defaults + real return
  types) before the marker ships — **not** whole-body pyright-clean. The internal method-body
  pyright errors (the bulk of the first-run baseline) remain **advisory** and are burned down
  over time **without gating** anything. Breaking a public annotation becomes a breaking change.
- pyright runs advisory in CI; backend/internal annotations are tightened later, working inward.
- Validation remains `ut.check_*` (ADR-0038 D11); this ADR adds *static* checking, not runtime
  enforcement.

## Out of scope

- Backend/internal strict-mode annotation completeness; any runtime type enforcement.
