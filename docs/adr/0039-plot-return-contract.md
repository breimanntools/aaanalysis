# ADR-0039 — One uniform `(fig, ax)` return contract for every `*Plot` method

Status: Accepted — 2026-06-24

Relates to: the `plotting.md` path-scoped rule; the strict-semver deprecation
policy (ADR-0030); the `StructureView` wrapper exception (ADR-0028); the API
ergonomics epic (#126).

## Context

The public `*Plot` methods returned **three different shapes** with no
predictable rule:

- `(Figure, Axes)` — `AAclustPlot.eval`, `CPPPlot.eval` / `ranking` / `profile` /
  `heatmap` / `feature_map`, `dPULearnPlot.eval`, `AAlogoPlot.single_logo` /
  `multi_logo`.
- `Axes` only — `CPPPlot.feature` / `update_seq_size`, `AAclustPlot.correlation`,
  `dPULearnPlot.pca`, `AAMutPlot.*`, `SeqMutPlot.*`.
- `(Axes, DataFrame)` — `AAclustPlot.centers` / `medoids`.

So a caller could not assume what a plot call returns: `fig, ax = plot(...)`
breaks on the Axes-only methods, `ax = plot(...)` breaks on the tuple methods,
and `ax, df = centers(...)` is a third, method-specific shape. This is a
recurring stumbling block when composing figures and a clear inconsistency for
both human and agentic callers (the API-ergonomics epic, #126).

The package is Production/Stable and **semver-strict from v1.x** (ADR-0030):
PATCH never breaks the public API, MINOR may only add or deprecate, MAJOR may
complete removals. Any change to a return shape that an existing caller unpacks
is therefore release-vehicle-sensitive.

## Decision

**One contract: every public `*Plot` method returns a `(fig, ax)` pair.**

- **D1 — The shape is `(fig, ax)`.** Matplotlib-idiomatic and composable: the
  caller always gets the `Figure` handle (for `savefig`, shared axes, multi-panel
  composition) alongside the `Axes`. For multi-panel methods (`eval`,
  `multi_logo`) the second element is an array of `Axes`, exactly as before.

- **D2 — The concrete type is `FigAxResult`, a thin `tuple` subclass**
  (`aaanalysis/_utils/plotting.py`, re-exported as `ut.FigAxResult`). It unpacks
  and indexes exactly like a plain 2-tuple, and additionally **forwards attribute
  access to the `ax` element** (`__getattr__` → `self[1]`). This makes the change
  **backward-compatible for every previously Axes-only method**: legacy
  `ax = plot(...); ax.set_title(...)` keeps working because the returned object
  proxies to the real `Axes`, while `fig, ax = plot(...)` now also works. It is a
  pure value type (no rendering logic), in the same spirit as the `StructureView`
  wrapper (ADR-0028).

- **D3 — Methods that also produce data expose it via a trailing-underscore
  attribute, not a third tuple element.** `AAclustPlot.centers` / `medoids` set
  `self.df_components_` (sklearn / `Wrapper`-style fitted attribute) and return
  `(fig, ax)` like every other method. This keeps the return shape uniform
  instead of carving out a `(fig, ax, df)` exception that the contract test would
  have to special-case.

- **D4 — Release vehicle: land now, complete the break in the next major.** The
  proxy (D2) makes the whole migration backward-compatible **except** the
  `centers` / `medoids` unpacking: `ax, df = centers(...)` cannot be preserved by
  a 2-tuple wrapper (the arities collide), so that one shape genuinely breaks.
  Under strict semver this break belongs to a MAJOR. The code lands on `master`
  now (uniform contract + proxy + `df_components_`); the version is **not** bumped
  here, and the `centers` / `medoids` break is recorded in the changelog as
  scheduled for the next major (v2.0). The minor-compatible part (every other
  method also returning `fig`) is an additive enhancement.

- **D5 — The contract is enforced by an introspection meta-test**
  (`tests/unit/api_tests/test_plot_return_contract.py`): every public method of
  every public `*Plot` class must be annotated `-> Tuple[Figure, Axes]` and
  declare `fig` then `ax` in its numpydoc `Returns` section. The single documented
  exception is `CPPStructurePlot` (pro), which returns a `StructureView`
  (ADR-0028).

## Rejected alternatives

- **Always return a bare `Axes`.** Simpler single object, but discards the
  `Figure` handle (no uniform `savefig` / multi-panel composition) and would
  break every current `(fig, ax)` caller — a much larger break for no ergonomic
  gain.
- **Keep `(fig, ax, df)` for `centers` / `medoids`.** Uniform unpacking prefix,
  but breaks the "identical shape across all methods" guarantee and forces the
  contract meta-test to special-case two methods. The trailing-underscore
  attribute (D3) keeps the surface uniform and matches the fitted-attribute
  idiom used elsewhere.
- **Plain 2-tuples (no proxy).** Cleanest end state, but turns the whole
  migration into a hard break (every `ax = plot(...)` call site) requiring a
  major immediately. The proxy (D2) lets the bulk land backward-compatibly now.
- **Force everything to a major release immediately.** Rejected for timing: the
  next release on this line is a minor (v1.1.0); the proxy lets the uniform
  contract land now and confines the unavoidable break to the documented
  `centers` / `medoids` case, scheduled for the major.

## Consequences

- Callers can rely on a single shape: `fig, ax = AnyPlot().any_method(...)`.
- Previously Axes-only methods are backward-compatible via the attribute proxy;
  `isinstance(result, plt.Axes)` is the one thing that changes (it is now a
  `FigAxResult`), so code that *type-checks* the return rather than using it must
  unpack first.
- `centers` / `medoids` read their component DataFrame from `.df_components_`;
  `ax, df = centers(...)` no longer unpacks correctly — a breaking change carried
  to the next major.
- A new public-facing type exists in spirit but stays internal (`ut.FigAxResult`,
  not added to `aaanalysis.__all__`): docstrings describe the return in plain
  `fig` / `ax` language, so the stability surface is unchanged.
- The meta-test makes the contract self-enforcing: a new `*Plot` method that
  returns the wrong shape fails CI.
