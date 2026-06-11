# ADR-0028 — `CPPStructurePlot` returns a `StructureView` wrapper, not a matplotlib `Axes`

Status: Accepted — 2026-06-11

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
