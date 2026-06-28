---
paths:
  - "aaanalysis/plotting/**/*.py"
  - "aaanalysis/**/_*plot*.py"
  - "aaanalysis/**/_*Plot*.py"
  - "aaanalysis/**/*_plot*.py"
---

# Plotting

- **One return contract: every public `*Plot` method returns `(fig, ax)`.**
  Return `ut.FigAxResult(fig, ax)` — a thin `tuple` subclass that unpacks as
  `fig, ax = ...` and also forwards attribute access to `ax` (so legacy
  `ax = ...; ax.set_title(...)` keeps working). For an Axes-only computation
  derive the figure with `ax.get_figure()`. The second element is a single `Axes`
  or an array of `Axes` (multi-panel `eval` / `multi_logo`). Methods that also
  produce data (`AAclustPlot.centers` / `medoids`) store it on a trailing-
  underscore attribute (`self.df_components_`), **not** a third tuple element.
  numpydoc `Returns` names `fig` then `ax`. Enforced by
  `tests/unit/api_tests/test_plot_return_contract.py`. The sole exception is
  `CPPStructurePlot` (excluded as a class): its outputs are 3D / HTML, not a matplotlib
  `Axes` — `map_structure` returns a py3Dmol `StructureView`, `plot_combined` a `CombinedView`
  (cartoon + feature-map image), `plot_linked` a `LinkedView` (feature-map↔structure linked
  HTML), and `interactive` an ipywidgets panel (see ADR-0028).
- Library code **never** calls `plt.show()`. Return the `(fig, ax)` pair or
  mutate the passed `ax`.
- Colors come from `ut.COLOR_*` and `ut.DICT_COLOR*`. Do not hardcode hex
  values.
- `plot_settings()` mutates rcParams globally — call only from user-facing
  entry points, never from inside library logic.
- **In notebooks (the user side of the boundary), the opposite holds:** every
  example/tutorial plot cell MUST end with `plt.tight_layout()` then
  `plt.show()` so the figure renders as an image — a bare `Plot().method(...)`
  leaves only a `<Axes …>` text repr. See `notebooks.md`.

## Direct-plot shortcut: `<abbrev>_kws` (LIGHT pairs only)

A logic/plot pair whose precompute is **cheap** (a getter or light transform) may
let users skip the logic class: the plot method takes an optional
`<base_abbrev>_kws` dict — the abbreviation from the class-abbreviation registry,
e.g. `aal_kws` for `AAlogoPlot` — holding the logic class's getter kwargs, and
computes the data internally. Reference: `AAlogoPlot.single_logo`'s `aal_kws`
(and `multi_logo`'s `list_aal_kws`).

Contract (mirror the reference exactly):
- `<abbrev>_kws` is **mutually exclusive** with the precomputed-data params
  (e.g. `df_logo` / `df_logo_info`) — provide one or the other; error if both.
- Validate keys against `inspect.signature(<LogicClass>.<getter>).parameters`
  (reject unknown keys early) and that the value is a `dict`.
- For list/multi plot methods, mirror as `list_<abbrev>_kws: list[dict]` (one dict
  per panel), mutually exclusive with the list-of-data param.
- numpydoc: state that the data is computed internally from these kwargs.

**Apply ONLY to LIGHT pairs** (cheap precompute): `AAlogoPlot` (`aal_kws`, done),
`AAMutPlot` (`aamut_kws`), `SeqMutPlot` (`seqmut_kws`).

**NEVER add it to pairs whose `.run` / `.fit` is compute-intensive** —
`CPPPlot` (`CPP.run` = feature discovery), `dPULearnPlot` (`dPULearn.fit` =
PU learning), `AAclustPlot` (`AAclust.fit` = clustering). There, require the user
to compute and pass the result explicitly, so the expensive step stays visible and
its output is reused across plots instead of being silently recomputed per call.
