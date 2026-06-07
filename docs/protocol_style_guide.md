# AAanalysis protocol style guide

How to write a **Protocol** for AAanalysis. Protocols live in the top-level
`protocols/` folder as executable notebooks (`protocol<N>_<slug>.ipynb`), are
auto-converted to RST by `docs/source/create_notebooks_docs.py`, and render under
**Examples : Protocols** on Read the Docs (`docs/source/protocols.rst`).

**Core idea.** A *tutorial* teaches one function; a *protocol* teaches a
**workflow**: it answers a single biological question end-to-end and, above all,
builds the reader's **mental model** for *when* and *why* to use a tool, not just
*how*. Derived from the seaborn tutorials and the scikit-learn example gallery,
adapted to AAanalysis.

---

## 1. Tutorials vs protocols (keep them distinct, no overlap)

Tutorials and protocols must not overlap. A **tutorial** documents one function
(what it does, its parameters). A **protocol** is a task-framed **workflow**. When
a protocol uses a function, **link to that function's tutorial** for the mechanics
instead of re-teaching it. A protocol that reads like a tutorial (parameter-by-
parameter tour of one function) is wrong: reframe it around the biological task.

## 2. The mental-model rule (most important)

Lead with **intuition before API**. Before any code, state the problem the
protocol solves and the one mental model the reader should walk away with. Name
the function by its *purpose*, not its signature. Make the abstraction explicit
("behind the scenes, CPP ...") rather than magical. Every protocol carries a short
**Key mental model** call-out near the top: 2 to 4 sentences a biologist can
repeat back.

## 3. Glossary-grounded language

Use the canonical terms from `CONTEXT.md` verbatim, and say them out loud:

- Frame the task by its **class**: *determinant discovery*, *prediction*, *design*.
- Name the **prediction level** (residue `AA_*` / domain `DOM_*` / protein `SEQ_*`)
  and the **unit of comparison**.
- Use **test group** (`label=1`) vs **reference group** (`label=0`); the selected
  feature set is the **signature** of the test group.
- Use **part / split / scale**, **compositional vs positional**, **window /
  reference / control** exactly as defined in the glossary.

## 4. Page structure

- **Title (the only heading).** A short H1 of the form `P<N>: <short title>` (e.g.
  `P1: CPP signature`). This is what shows in the sidebar, so keep it short. Put
  the longer descriptive phrase in the opening paragraph, not the title.
- **Field labels are bold lead-ins, not headings.** Write the seven fields as
  `**When to use it.** ...` (bold), *not* as `## When to use it`. Headings would
  clutter the RTD sidebar with a sub-level we do not want; bold keeps the sidebar
  to protocol titles only.
- The seven fields, in order: **When to use it** (incl. *when not to*) :
  **Input** (name the upstream protocol) : **Run** : **Output** (show the key
  figure early) : **How to interpret** (+ 2 to 3 *Key takeaways* bullets) :
  **Common mistakes** : **Next step** (link the next protocol by its `P<N>:` name).

## 5. Code & figures

- **Show dataframes with `aa.display_df(df=..., n_rows=...)`**, never `print(df)`
  or a bare `df.head()`: it renders a clean, scrollable table.
- No `print()` for library output; figures over console dumps.
- Real, meaningful datasets (`load_dataset(...)`), `random_state=42`,
  `aa.options["verbose"]=False`. `plt.tight_layout()`, labelled axes.
- Figure-forward, conversational tone ("A few things happened here ...",
  "Notice how ..."). The first output figure is the protocol's gallery image.
- Backtick + cross-reference public symbols (e.g. `CPP.run`).

## 6. Hard rules

- **No GitHub issue references** anywhere (no `#NN`, no issue links).
- **No em dashes.** Use a colon for label-like breaks and a comma or parentheses
  in prose.
- **Real API only**, verified against source. `CPP(df_parts=...)`;
  `CPPPlot().feature_map` is an instance method needing a `feat_importance` column.
- **Runs green** under a fresh kernel within the nbmake budget (120 s/cell): small
  fixtures (`load_dataset(name=..., n=<small>)`, `n_filter<=50`).
- **One protocol = one notebook** named `protocol<N>_<slug>.ipynb` in `protocols/`;
  add it to the `protocols.rst` toctree in catalog order.

## 7. Catalog order (pipeline)

CPP signature is P1; the exploratory no-label first look is P2:

```
1 CPP signature                2 exploratory sequence analysis (no labels)
3 sampling                     4 prediction levels
5 engineer features            6 compositional vs positional
7 select & reduce features     8 classifier
9 interpretability             10 validate ("can I trust this?")
```

This is a **living catalog**: append protocols as the package grows.

## 8. Checklist (tick before merging a protocol)

- [ ] Framed as a *workflow*, distinct from tutorials (links out, no overlap)
- [ ] Opens with intuition + a **Key mental model** call-out
- [ ] Glossary terms used correctly (level, unit of comparison, test/reference, signature, ...)
- [ ] Short `P<N>: <title>` H1; seven fields as **bold lead-ins**, not headings
- [ ] Tables via `aa.display_df`; key figure shown early; `plt.tight_layout()`; no `print()`
- [ ] **No GitHub issue references; no em dashes**
- [ ] Real API, verified; `random_state=42`; runs green under a fresh kernel (<=120 s/cell)
- [ ] Named `protocol<N>_<slug>.ipynb` in `protocols/`; added to `protocols.rst`
