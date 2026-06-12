# Handoff: `CPPStructurePlot` — map CPP-SHAP feature impact onto a 3D protein structure

## Context / motivation

The deployed ADAMTS-7 cleavage app (`scripts/app/pyodide_template.html` in the Amin project) has a
well-liked view: per-residue CPP-SHAP impact (red = raises the prediction, blue = lowers it) painted
onto the AlphaFold cartoon, with fade-context, zoom-to-site, and a pLDDT colouring mode. Today this
lives **only** in that HTML file — the 3D is rendered by **3Dmol.js** in the browser, and the
per-residue impact is recomputed by a hand-written `perpos` loop inside an embedded `render_merged`
Python function.

This should become a reusable AAanalysis function. It is **not** HTML-specific — the browser file is
the only browser-bound part. Both halves of the logic already exist in AAanalysis:

- **Feature → residue mapping already exists.** `get_positions_()`
  (`aaanalysis/feature_engineering/_backend/cpp/utils_feature.py:221`) turns each `df_feat` feature
  into the 1-based positions it covers, and `PlotPartPositions.get_df_pos(..., value_type="sum")`
  (`aaanalysis/feature_engineering/_backend/cpp/_utils_cpp_plot_positions.py:201`) already aggregates
  `feat_impact` per position. The app's manual `perpos` loop is a re-implementation of this.
- **Structure handling already exists** in the `[pro]` extra. `StructurePreprocessor`
  (`aaanalysis/data_handling_pro/_struct_preproc.py`) parses PDB/CIF via biopython, and
  `fetch_alphafold()` downloads AF models + PAE by UniProt accession. The backend
  (`aaanalysis/data_handling_pro/_backend/struct_preproc/encode_pdb.py`) already exposes
  `load_structure()`, `_ca_coords_and_residues()` (best-chain pick + Cα coords), and
  `_plddt_per_residue()` (Cα B-factor → pLDDT).

The only genuinely new dependency is a **Python** 3D viewer. `py3Dmol` is the Python binding to the
same 3Dmol.js the app uses — interactive in Jupyter, exports a self-contained HTML — so the output
mirrors the deployed app one-to-one. A matplotlib `mplot3d` fallback gives a static Cα scatter for
headless/docs/CI use.

**Outcome:** a new pro plotting class `CPPStructurePlot` whose `feature_structure(df_feat, pdb=...)`
returns an interactive (py3Dmol) or static (matplotlib) view of per-residue feature impact mapped onto
the protein, with fade/zoom focus, a pLDDT colouring mode, and optional AlphaFold auto-fetch.

## Suggested issue split

This handoff covers two stages, intended as **two separate issues**:

- **Issue 1 — `CPPStructurePlot` (static render class).** The `df_feat` + structure → coloured 3D view.
  A pure render function: data in, coloured structure out. Self-contained, ships first. (Sections
  *Public API* through *Verification* below.)
- **Issue 2 — interactive live-prediction (`.interactive(...)`).** An ipywidgets wrapper that calls a
  user-supplied predictor on every change (pick a site, mutate a residue, drag a threshold) and
  repaints the Issue-1 view. Depends on Issue 1. (Section *Issue 2* below.)

**Cost:** everything here is free to run — py3Dmol/3Dmol.js (BSD, runs locally in-browser), the
AlphaFold-DB fetch (free public endpoint), and the CPP/TreeModel prediction (local CPU). No LLM or
hosted-inference component anywhere.

## Design decisions

- **Placement / API:** new pro plotting class `CPPStructurePlot`, mirroring `CPPPlot` / `AAMutPlot`.
  Core stays dependency-free.
- **Renderer:** `py3Dmol` primary (interactive + standalone HTML); matplotlib `mplot3d` static fallback.
- **v1 scope:** (1) colour by SHAP impact, (2) fade-context + zoom-to-site, (3) pLDDT colouring mode,
  (4) AlphaFold auto-fetch by accession.

---

# Issue 1 — `CPPStructurePlot` (static render class)

## Public API

```python
import aaanalysis as aa

sp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=True)

view = sp.feature_structure(
    df_feat=df_feat,            # standard CPP df_feat (needs col_imp present)
    pdb="Q9NQ76.pdb",          # OR uniprot="Q9NQ76" to auto-fetch from AlphaFold-DB
    col_imp="feat_impact_site", # signed per-position impact column (shap_plot semantics)
    tmd_len=10,                 # TMD length used when the features were generated
    start=312,                  # absolute residue number of the FIRST jmd_n residue in the structure
    mode="impact",             # "impact" (red/blue) | "plddt" (AF palette)
    focus="fade",              # "whole" | "fade" (ghost the rest) | "zoom" (camera to window)
    size_by_impact=True,        # stick radius proportional to |impact|
    chain=None,                 # None = auto best-matching chain (reuses encode_pdb logic)
    backend=None,               # None = py3Dmol if available else mpl; "py3dmol" | "mpl" to force
)
view.show()                    # interactive in Jupyter (py3Dmol) / inline PNG (matplotlib)
view.write_html("out.html")    # self-contained interactive page (py3Dmol backend)
view.savefig("out.png")        # static render (matplotlib backend; py3Dmol can export PNG too)
```

`feature_structure()` returns a thin `StructureView` wrapper exposing `show()`, `write_html(path)`,
`savefig(path)`, and `_repr_html_`, so the return type is uniform across both backends.

## Implementation

New module `aaanalysis/feature_engineering_pro/` (sibling to `feature_engineering/`, where `CPPPlot`
lives), gated by the `pro` extra following the existing `ShapModel` pattern.

```
aaanalysis/feature_engineering_pro/
  __init__.py                       # exports CPPStructurePlot
  _cpp_structure_plot.py            # CPPStructurePlot class (public, numpydoc, input checks)
  _backend/
    cpp_structure/
      map_impact_to_residues.py     # df_feat -> {abs_residue: summed impact}
      render_py3dmol.py             # py3Dmol cartoon colouring + focus modes + HTML export
      render_mpl.py                 # matplotlib mplot3d Ca-scatter fallback
      colors.py                     # shap impact ramp + AF pLDDT palette (reuse package constants)
```

### Core logic (reuse, don't reinvent)

1. **`map_impact_to_residues`** — wrap `get_positions_(features, start, tmd_len, jmd_n_len, jmd_c_len)`
   to get each feature's positions, then aggregate `df_feat[col_imp]` per position by **sum** (the same
   `value_type="sum"` rule `PlotPartPositions.get_df_pos` uses for `feat_impact`). `start` shifts
   window positions to absolute structure residue numbers. Return `{resi: impact}` plus `max_abs` for
   normalisation. This replaces the app's hand-written `perpos` loop.

2. **Structure parsing** — reuse
   `aaanalysis/data_handling_pro/_backend/struct_preproc/encode_pdb.py`: `load_structure()` (PDB/CIF),
   `_ca_coords_and_residues()` (best chain + Cα xyz, honours the `chain` arg), `_plddt_per_residue()`
   (Cα B-factor → pLDDT). No new parsing code.

3. **AlphaFold auto-fetch** — when `uniprot=` is given instead of `pdb=`, call the existing
   `StructurePreprocessor.fetch_alphafold()` into a temp dir, then load the model. Reuses pro code.

### Colours (reuse package constants)

- Impact ramp: white→`COLOR_SHAP_POS` (`#FF0D57`) / white→`COLOR_SHAP_NEG` (`#1E88E5`) from
  `aaanalysis/utils.py:454`, with the app's `sign·sqrt(|t|)` perceptual transform
  (`shapColor`, `pyodide_template.html:318`).
- pLDDT palette: AlphaFold blue→green→yellow→orange (`#0053D6/#65CBF3/#FFDB13/#FF7D45`), matching the
  app's `plddtColor()` (`pyodide_template.html:208`).

### Renderers

- **py3Dmol** (`render_py3dmol.py`): `addModel(pdb,"pdb")`; per-residue
  `setStyle({resi},{cartoon:{color}})`; optional `stick` radius proportional to |impact| when
  `size_by_impact`; `focus="fade"` ghosts non-window residues (`opacity:0.45`), `focus="zoom"` calls
  `zoomTo({resi: window})`. The window = TMD ± jmd lengths derived from `start`/`tmd_len`/`jmd_*_len`.
  `write_html()` uses py3Dmol's HTML export. Mirrors `applyBaseStyle` / `show3D` in the app
  (`pyodide_template.html:329` / `:459`).
- **matplotlib** (`render_mpl.py`): `mplot3d` scatter of Cα coords coloured by the same ramp; faded
  residues at low alpha; returns a `Figure` for `savefig`. Used automatically when py3Dmol is
  unavailable or when `backend="mpl"`.

### Wiring (follow `ShapModel` exactly — see `.claude/rules/pro-core-boundary.md`)

- `aaanalysis/__init__.py`: add a `try: from .feature_engineering_pro import CPPStructurePlot` block
  with `missing_feature_stub("CPPStructurePlot", e, mode="pro")` on ImportError; append to `__all__`
  on success.
- `_EXTRA_MODULES["pro"]` in `aaanalysis/__init__.py:83`: add `"py3Dmol"` (its import name) so a missing
  viewer yields the friendly install hint rather than a raw traceback.
- `pyproject.toml` `[project.optional-dependencies] pro`: add `py3Dmol>=2.0`. (biopython/requests
  already present; matplotlib is already a core dep, so the fallback adds no new dependency.)

## Critical files

- New: `aaanalysis/feature_engineering_pro/_cpp_structure_plot.py` (+ backend files above)
- Reuse: `aaanalysis/feature_engineering/_backend/cpp/utils_feature.py:221` (`get_positions_`)
- Reuse: `aaanalysis/data_handling_pro/_backend/struct_preproc/encode_pdb.py` (`load_structure`,
  `_ca_coords_and_residues`, `_plddt_per_residue`)
- Reuse: `aaanalysis/data_handling_pro/_struct_preproc.py` (`StructurePreprocessor.fetch_alphafold`)
- Reuse: `aaanalysis/utils.py:454` (`COLOR_SHAP_POS` / `COLOR_SHAP_NEG`)
- Edit: `aaanalysis/__init__.py` (export + `_EXTRA_MODULES`), `pyproject.toml` (pro extra)
- Parity reference (Amin project): `scripts/app/pyodide_template.html` — `render_merged` perpos loop
  (~line 787), `shapColor` (318), `plddtColor` (208), `applyBaseStyle` (329), `show3D` (459).

## Verification

1. **Unit — mapping:** build a tiny `df_feat` (2–3 known features), call `map_impact_to_residues`, and
   assert per-residue sums equal a hand-computed `perpos` for a chosen `start`/`tmd_len`. Cross-check
   one case against the app's `render_merged` output for the same window.
2. **Integration — real data:** use the deployed ADAMTS-7 `df_feat` + an AlphaFold PDB; run
   `feature_structure(..., mode="impact", focus="fade")`, `write_html("/tmp/v.html")`, confirm the HTML
   opens with a coloured cartoon. Repeat with `mode="plddt"`.
3. **Auto-fetch:** `feature_structure(df_feat, uniprot="<acc>")` downloads the AF model and renders
   (network-gated; mock/skip in CI).
4. **Fallback:** force `backend="mpl"` (or simulate py3Dmol absent); assert a `Figure` is returned and
   `savefig` writes a PNG.
5. **Gating:** in an env without `py3Dmol`, `aa.CPPStructurePlot` resolves to the stub and raises the
   friendly `pip install 'aaanalysis[pro]'` hint on use.

## Out of scope for Issue 1

Live re-prediction (→ Issue 2); topology colouring mode; linked hover/tooltips across panels;
multi-fragment AlphaFold renumbering; per-residue secondary-structure/burial readouts. App
conveniences that can be layered later.

---

# Issue 2 — interactive live-prediction (`.interactive(...)`)

## Context / motivation

Issue 1 is a one-shot render: `df_feat` in, coloured structure out. Issue 2 closes the loop so the
prediction itself is **live and adjustable** in a notebook — pick a site, mutate a residue, or drag a
threshold, and the model re-runs and the cartoon repaints. This is the notebook-native equivalent of
the deployed `pyodide_template.html` app, but driven by the **real Python model on a live kernel**
(exact predictions, no JS/Pyodide port needed).

The key fork is *where it runs* (worth stating in the issue so expectations are set):

- **Live Jupyter kernel** — full live re-prediction works: a Python callback runs the model and repaints
  the py3Dmol view. This is the target of Issue 2.
- **Exported standalone `.html`** — has no Python kernel, so it cannot call back into the model.
  Shareable live prediction requires the model to run *in the browser* (Pyodide or a JS port) — which is
  exactly what the existing app already does. **Out of scope here**; Issue 2 targets the live-kernel
  path only.

## Public API

```python
import aaanalysis as aa

sp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10)

# `predictor` is a user-supplied callable: (sequence, p1) -> df_feat
# (e.g. wraps CPP.run(...) + ShapModel/TreeModel to produce a df_feat with col_imp at the site)
ui = sp.interactive(
    predictor=my_predictor,     # callable returning a df_feat for the current selection
    sequence="MK...",           # full protein sequence
    pdb="Q9NQ76.pdb",          # or uniprot="Q9NQ76"
    tmd_len=10,
    col_imp="feat_impact_site",
    mode="impact", focus="fade",
)
ui   # renders the widget panel + linked 3D view in the notebook output cell
```

`interactive()` returns an `ipywidgets` container (an `HBox`/`VBox`) that displays inline. It owns the
control widgets and re-renders by reusing the **same** Issue-1 mapping + render backend.

## Control flow

```
[site slider / sequence box / impact-threshold slider]
        --on_change-->  Python callback (debounced)
                predictor(sequence, p1)            -> df_feat   (user's model, runs on the kernel)
                map_impact_to_residues(df_feat, start=p1_anchor, tmd_len, jmd_*_len)
                render backend (py3Dmol)           -> repaint the view in-place
```

- **Anchor:** the selected P1 (or TMD start) sets `start`, the same anchor Issue 1 uses to map
  window-relative `df_feat` positions to absolute structure residues. Selection drives `start`.
- **Debounce:** coalesce rapid slider moves (e.g. `Output` + a short timer) so the model isn't re-run on
  every intermediate value — mirrors the app's "pause while exploring" behaviour.
- **In-place repaint:** update the py3Dmol view inside a single `ipywidgets.Output` so the camera/zoom
  state is preserved across re-predictions where possible.

## Implementation

Add one public method to `CPPStructurePlot` plus a small backend file. No change to the Issue-1 render
path — Issue 2 only *drives* it.

```
aaanalysis/feature_engineering_pro/
  _cpp_structure_plot.py            # + .interactive(predictor, sequence, ...) method
  _backend/
    cpp_structure/
      interactive_widgets.py        # ipywidgets panel, debounced callback, Output repaint
```

- `interactive_widgets.py` builds the widget panel, wires the debounced callback to
  `predictor → map_impact_to_residues → render_py3dmol`, and manages the `Output` cell.
- `predictor` is a plain callable supplied by the user — the class does **not** hard-code CPP/TreeModel,
  so any model that returns a `df_feat` (with `col_imp`) for a selection works. The docstring should show
  one worked example wiring `CPP` + `ShapModel`/`TreeModel` into such a callable.

## Dependency / gating

- Add `ipywidgets` to the `pro` extra in `pyproject.toml`, and `"ipywidgets"` to `_EXTRA_MODULES["pro"]`
  in `aaanalysis/__init__.py`. The import is **lazy inside `.interactive()`** so Issue-1 static use never
  requires ipywidgets — only the interactive path does.

## Verification

1. **Headless smoke:** instantiate `.interactive(...)` with a stub `predictor` (returns a fixed
   `df_feat`); assert it builds a widget container without a display backend (no exception, correct
   children).
2. **Callback wiring:** simulate a site-slider change; assert the stub predictor is called with the
   expected `(sequence, p1)` and that `map_impact_to_residues` is invoked with the matching `start`.
3. **Manual notebook check:** in JupyterLab, drive the real CPP/TreeModel predictor; confirm changing
   the site repaints the cartoon with updated colours and the camera is preserved.
4. **Gating:** without `ipywidgets`, `.interactive()` raises the friendly `pip install 'aaanalysis[pro]'`
   hint while `.feature_structure()` (Issue 1) still works.

## Out of scope for Issue 2

Shareable standalone-HTML live prediction (needs Pyodide / JS port — already solved in the deployed
app); mutation scanning UIs; multi-site comparison panels.
