# AAanalysis protocol quality rubric

The teaching-depth bar every **Protocol** must clear. This is the companion to
[`protocol_style_guide.md`](protocol_style_guide.md): the style guide fixes a
protocol's *structure and voice* (the seven fields, glossary language, tone, hard
rules); this rubric fixes its *teaching depth* — whether the reader can **see** each
concept the protocol names, or is asked to take the prose on faith.

**Core idea: show, don't tell.** A protocol that *names* a concept without
*visualizing* it teaches nothing a paragraph could not. The standard is set by
**Protocol 1 (CPP signature)**, the only protocol that currently renders every
decision its method makes. Each of protocols 2–10 is measured against it, in review,
per protocol, pass/fail.

---

## The exemplar: how Protocol 1 shows six things

Protocol 1 draws the **same signature six ways**, and each figure isolates one
concept by **contrasting it against its alternative** — the contrast is what makes
the concept legible:

| Figure | Concept it isolates | Contrast |
|---|---|---|
| Map 1 | the raw signature | after `run`, before refinement |
| Map 2 | the refined signature | after `simplify` — fewer, stronger features (vs Map 1) |
| Map 3 | **compositional** locality | whole-part features |
| Map 4 | **positional** locality | sub-segment features (vs Map 3) |
| Map 5 | the `Pattern` split type | fixed offsets vs `Segment`'s contiguous chunks |
| Map 6 | one AAontology property family | the category-filtered view vs the full signature |

The narrative skeleton is untouched: the six maps live in a *"The two concepts the
map makes visible"* block that sits between **Output** and **How to interpret**. The
concept figures deepen the protocol; they do not restructure it.

---

## The rubric

A protocol passes when **all four** hold. Apply it per protocol, binary.

### 1. Concept-contrast visualizations
- **Every core concept the protocol names gets its own figure**, and that figure
  **contrasts** the concept against its alternative (test vs reference, before vs
  after, Arm A vs Arm B, level vs level). A concept stated only in prose fails.
- A protocol that carries too few figures for the concepts it names must gain
  **≥ 1** additional concept-contrasting figure. (Baseline audit at time of writing:
  P2, P3, P4, P5, P7, P8 each carry a **single** figure and must gain at least one
  more; P10 has two figures and also gains at least one; P6 (four figures) and P9
  (three figures) are **audited-and-tightened** against this rubric rather than
  necessarily expanded.)
- The **first** output figure remains the protocol's gallery image (per the style
  guide); the concept figures come after it.

### 2. Full public-surface coverage
- **Every public parameter of the demonstrated method is passed by name**, grouped
  sensibly (the discipline already enforced for `examples/**` by
  `tests/unit/api_tests/test_notebook_param_coverage.py`). The protocol is where a
  reader sees the method used at full stretch, not in a one-arg call.
- Prefer keyword calls throughout, so each argument reads as documentation.

### 3. Narrative skeleton preserved
- The seven bold-lead-in fields stay, in order: **When to use it** (incl. *when not
  to*) → **Input** → **Run** → **Output** → **How to interpret** (+ 2–3 *Key
  takeaways*) → **Common mistakes** → **Next step**. New figures are inserted
  *within* this skeleton, never in place of it.
- Opens with intuition and a **Key mental model** call-out before any API.

### 4. Presentation mechanics (must render, must not rot)
- **Every DataFrame** via `aa.display_df(df, n_rows=10, show_shape=True)` — never a
  bare `df`, `df.head()`, or `print(df)`.
- **Every plot cell** ends with `plt.tight_layout()` then `plt.show()` (inline
  backend), so the figure flushes as an embedded PNG rather than an `<Axes …>` repr.
- **Executed outputs are committed** — tables and images — and re-run before pushing.
- Small, seeded fixtures (`load_dataset(name=…, n=<small>)`, `n_filter ≤ 50`,
  `random_state=42`), every cell under the 120 s nbmake budget.
- **No em dashes; no GitHub issue references** in the notebook (house style — see the
  style guide). Real, source-verified API only.

---

## How to apply it

1. Read the protocol against **§1–§4** with Protocol 1 open beside it.
2. For each concept the prose names, ask: *is there a figure that shows it, by
   contrast?* If not, that is the figure to add.
3. Confirm the method's public parameters are all exercised by name.
4. Re-execute the notebook clean under nbmake (0 errors) and commit the outputs.

A protocol is done when a reviewer can tick §1–§4 and every named concept has a
picture. The two standalone deliverables that make this rubric enforceable are this
document and the **nbmake execution gate** that runs `protocols/` in CI (so a broken
or stale protocol cell fails a check, the same way `tutorials/` and `examples/` are
gated today).

## See also

- [`protocol_style_guide.md`](protocol_style_guide.md) — structure, voice, the seven
  fields, glossary language, hard rules.
- `.claude/rules/notebooks.md` — the `display_df` / `plt.show()` / executed-output
  discipline shared with tutorials and examples.
- `protocols/protocol1_cpp_signature.ipynb` — the worked exemplar this rubric distills.
