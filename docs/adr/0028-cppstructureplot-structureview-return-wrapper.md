# ADR-0028 — `CPPStructurePlot` returns a `StructureView` wrapper, not a matplotlib `Axes`

Status: Accepted — 2026-06-11 (amended 2026-06-28: matplotlib backend removed)

## Amendment (2026-06-28) — py3Dmol-only, no matplotlib structure backend

The original decision below kept a **matplotlib `mplot3d` fallback** so a structure could
render without py3Dmol. In practice that fallback drew a per-residue Cα **scatter** that
looked nothing like a protein and leaked into the rendered example notebooks — a real
quality problem. py3Dmol also cannot render a cartoon into a matplotlib figure or to a PNG
headlessly, so a "static matplotlib structure" can only ever be that scatter.

Decision: **drop the matplotlib structure backend entirely.** All `CPPStructurePlot`
rendering is py3Dmol (a real interactive cartoon); the methods now *require* py3Dmol (a
friendly install hint is raised otherwise). The `backend` argument of `map_structure` is
gone. `plot_combined` no longer returns a matplotlib `(fig, ax)` — it returns a
`CombinedView` (py3Dmol cartoon + the `CPPPlot.feature_map` image side by side, HTML).
`interactive` returns an ipywidgets panel. The wrapper rationale below still holds: none of
these are a matplotlib `Axes`, so `CPPStructurePlot` remains the documented exception to the
"return `(fig, ax)`" rule — now because the outputs are genuinely 3D / HTML, not because two
backends disagree. `StructureView` is a pure py3Dmol delegator (`show` / `write_html` /
`_repr_html_`); `savefig` is gone (use `write_html`). The original 2026-06-11 context (which
weighed the two-backend split) is retained below for history.

## Context

`CPPStructurePlot` (new pro plotting class, `aaanalysis/feature_engineering_pro/`)
paints per-residue CPP-SHAP feature impact onto a 3D protein structure. Unlike
every other plotting method in the package, it has **two render backends whose
native return objects are incompatible**:

- **py3Dmol** (interactive primary) returns a `py3Dmol.view`: renders inline in
  Jupyter via its own `_repr_html_`, exports a self-contained interactive page via
  `write_html(path)`, but has **no `savefig`**.
- **matplotlib `mplot3d`** (static fallback, used when py3Dmol is absent or
  `backend="mpl"`) returns a `Figure`: has `savefig(path)`, but **no
  `write_html`**.

Every existing plot method (`CPPPlot.feature/ranking/profile/heatmap`,
`AAMutPlot.*`, …) returns `plt.Axes` or `Tuple[plt.Figure, plt.Axes]`, and
`.claude/rules/plotting.md` codifies "return fig/ax, never `plt.show()`". There is
no view-wrapper object anywhere in the codebase. So the question was how
`map_structure()` (and later `.interactive()`) should return, given the two
backends cannot share a native type and neither native type alone exposes both
`write_html` (the whole point — shareable interactive output) and `savefig`.

## Decision

**Return a thin `StructureView` wrapper** exposing a uniform surface across both
backends: `show()`, `write_html(path)`, `savefig(path)`, and `_repr_html_`. It is
a **pure delegator** — no state, no rendering logic — that forwards to the
underlying `py3Dmol.view` or `Figure`. On the matplotlib path `write_html`
degrades gracefully (wrap the static image or raise a clear "use the py3Dmol
backend for interactive HTML" message); on the py3Dmol path `savefig` uses
py3Dmol's PNG export. This is a **deliberate, documented exception** to the
"return fig/ax" rule and must not be "fixed" back to returning an `Axes`.

## Rejected alternatives

- **Return native objects** (`py3Dmol.view` or `Figure` depending on backend).
  Rejected: the return type *and* the available methods silently vary by backend,
  and `write_html` — the shareable-interactive feature that motivates the class —
  would not exist on the matplotlib path. Callers would have to branch on which
  object they got.
- **Two separate methods** (e.g. `map_structure_html()` / `map_structure_static()`),
  each returning a native object. Rejected: doubles the public surface for what is
  one conceptual operation, and forces the user to choose the backend up front
  rather than letting `backend=None` auto-select py3Dmol-if-available.
- **Return a bare `Axes`/`Figure` always** (force everything through matplotlib).
  Rejected: discards the interactive py3Dmol view that is the class's reason to
  exist and mirrors the deployed app one-to-one.

## Consequences

- First non-`Axes` plotting return type in the package; recorded here so a future
  reader does not regard it as an oversight. Per the house rule, this ADR is
  referenced from the issue, **never from user-facing docstrings** (the docstring
  explains `StructureView` in plain language).
- Uniform notebook UX (`view`, `view.show()`, `view.write_html(...)`,
  `view.savefig(...)`) regardless of which backend fired.
- `StructureView` stays a pure delegator — if it ever accretes logic, that is a
  signal the boundary was drawn wrong.
- Sets the precedent that Issue B's `.interactive()` returns an ipywidgets
  container (a different non-`Axes` type) for the same reason: the output is not a
  matplotlib figure.
