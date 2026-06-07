# AAanalysis protocol style guide

How to write a **Protocol** for AAanalysis. Protocols live in the top-level
`protocols/` folder as executable notebooks (`protocol<N>_<slug>.ipynb`), are
auto-converted to RST by `docs/source/create_notebooks_docs.py`, and render under
**Examples → Protocols** on Read the Docs (`docs/source/protocols.rst`).

**Core idea.** A *tutorial* teaches one function; a *protocol* teaches a
**workflow** — it answers a single biological question end-to-end and, above all,
builds the reader's **mental model** for *when* and *why* to use a tool, not just
*how*. Derived from the conventions of the seaborn tutorials and the scikit-learn
example gallery, adapted to AAanalysis.

---

## 1. The mental-model rule (most important)

Lead with **intuition before API**. Before any code, state the problem the
protocol solves and the one mental model the reader should walk away with. Name
the function by its *purpose*, not its signature. Make the abstraction explicit
("behind the scenes, CPP …") rather than magical.

Every protocol carries a short **Key mental model** call-out near the top — 2–4
sentences a biologist can repeat back.

## 2. Glossary-grounded language

Use the canonical terms from `CONTEXT.md` verbatim, and say them out loud:

- Frame the task by its **class**: *determinant discovery*, *prediction*,
  *design*.
- Name the **prediction level** (residue `AA_*` / domain `DOM_*` / protein
  `SEQ_*`) and the **unit of comparison**.
- Use **test group** (`label=1`) vs **reference group** (`label=0`); the selected
  feature set is the **signature** of the test group.
- Use **part / split / scale**, **compositional vs positional**, **window /
  reference / control** exactly as defined in the glossary.

Terminology in a protocol must not drift from `CONTEXT.md` or the docstring guide.

## 3. The 7-field structure (pipeline-chained)

Markdown cells around minimal code cells, in this order:

1. **Title + intro** — action-oriented title; 1 short paragraph stating the
   biological question and naming the *Key mental model*.
2. **When to use it** — the question in plain language; typical examples; *and
   when not to use it*.
3. **Input** — the dataframe/format that flows in; name the **upstream** protocol
   it receives from.
4. **Run** — minimal, **real** code (no aspirational one-liners). Set
   `aa.options["verbose"]=False` and `random_state=42`.
5. **Output** — what flows out; show the key **figure early** (figures carry the
   explanation; `plt.tight_layout()`, labelled axes).
6. **How to interpret** — biological-language reading (a small table works well);
   plus *Key takeaways* (2–3 bullets).
7. **Common mistakes**, then **Next step** — link the **next** pipeline-stage
   protocol *by name* (this is what makes the catalog a pipeline).

## 4. Tone & figures

- Conversational, minimal jargon: "A few things happened here …", "Notice how …".
- Progressive complexity: simplest path first, options later.
- Figure-forward: the first output figure is the protocol's gallery image; prefer
  plots over console dumps; no `print()` for library output (show returned
  objects / `df.head()`).
- Real, meaningful datasets (`load_dataset(...)`), never arbitrary noise.
- Backtick + cross-reference public symbols (e.g. `CPP.run`, `CPPPlot.feature_map`).

## 5. Hard rules

- **No GitHub issue references** anywhere in a protocol (no `#NN`, no issue
  links) — protocols are user-facing docs, not tracker notes.
- **Real API only**, verified against source. `CPP(df_parts=...)` (not
  `df_seq`/`labels`); `CPPPlot().feature_map` is an instance method needing a
  `feat_importance` column.
- **Runs green** under a fresh kernel within the nbmake budget (120 s/cell): use
  small fixtures (`load_dataset(name=..., n=<small>)`, `n_filter<=50`).
- **One protocol = one notebook** named `protocol<N>_<slug>.ipynb` in
  `protocols/`; add it to the `protocols.rst` toctree in catalog (pipeline) order.
- Colab/GPU-only notebooks are excluded from the build (`LIST_EXCLUDE`) and
  linked from the relevant protocol's *Input* section instead.

## 6. Catalog order (pipeline)

Lead with the two canonical protocols, then follow the data-flow pipeline:

```
0 exploratory sequence analysis (no labels)
1 CPP signature            2 prediction tasks / levels
  ── then the pipeline ──
3 construct sets / sampling     4 engineer features (parts/splits; run_num)
5 compositional vs positional   6 select & reduce features
7 build a classifier            8 interpretability (SHAP, feature map)
9 validate / "can I trust this?"
```

This is a **living catalog** — append protocols as the package grows.

## 7. Checklist (tick before merging a protocol)

- [ ] Opens with intuition + a **Key mental model** call-out
- [ ] Glossary terms (level, unit of comparison, test/reference, signature, …) used correctly
- [ ] All 7 fields present; **Next step** links the next protocol by name
- [ ] Key figure shown early; `plt.tight_layout()`; no `print()` for output
- [ ] **No GitHub issue references**
- [ ] Real API, verified; `random_state=42`; runs green under a fresh kernel (≤120 s/cell)
- [ ] Named `protocol<N>_<slug>.ipynb` in `protocols/`; added to `protocols.rst`
