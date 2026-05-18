---
paths:
  - "aaanalysis/**/*.py"
  - "docs/source/**/*.rst"
---

# Docstring style and citations

## Docstring style (numpydoc)

numpydoc style throughout (`Parameters / Returns / Raises / Notes / See Also /
References`). Two non-obvious rules that catch most drift on touch:

- **Recurring parameter descriptions stay consistent across methods.** When a
  parameter appears in multiple public methods (e.g. `df_seq`, `pos_col`,
  `window_size`, `n`, `role`, `seed`), the *baseline* description is the same
  everywhere; method-specific behavior is a *suffix* on the baseline, not a
  replacement. Canonical example for `df_seq`:
  ```
  df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
      DataFrame containing an ``entry`` column with unique protein identifiers
      and a ``sequence`` column with full protein sequences. <Method-specific
      note: e.g. "Used here as the source of empirical amino-acid frequencies
      for ``mode='global_freq'``.">
  ```
  The canonical first sentence (`DataFrame containing an ``entry`` column...`)
  is what `SequenceFeature.get_df_parts` and `CPP` use; new methods must match.
- **Describe the structure first, the use second.** A `Parameters` block
  explains *what the value must look like*; behavioral notes go after. A
  description that only says "Source for amino-acid frequencies" without
  documenting the required columns is incomplete.

Other expectations:
- Module docstring opens with `"""This is a script for ..."""` (see CLAUDE.md §3).
- Cite literature via `[Key]_` only; never inline the full bibliography (see citations below).
- Don't write multi-paragraph parameter descriptions when one tight sentence
  works. Examples (`.. include::`) belong in the dedicated examples directory,
  not inside the parameter block.
- Keep `Parameters` concise; behavioral elaboration goes in `Notes`.

## Citations

- All bibliography entries live in **`docs/source/index/references.rst`**,
  grouped into topical sections (AAanalysis Algorithms, Sequence Algorithms,
  Machine Learning, Positive-Unlabeled Learning, Explainable AI, Datasets and
  Benchmarks, Use Cases, Further Information).
- Entry format: `.. [AuthorYY] Author Year, *Title*, `Venue <URL>`__.`
  - Single first-author: `[AuthorYY]` (e.g. `[Song12]`, `[Rawlings16]`).
  - Two-author: `[FirstSecondYY]` (e.g. `[BekkerDavis20]`, `[ElkanNoto08]`).
  - Multiple papers same author/year: append a lowercase letter
    (e.g. `[Breimann24a]`, `[Breimann24b]`).
- **Cite from docstrings via `[Key]_` only.** Never repeat the full citation
  inside a class or method docstring — Sphinx + numpydoc resolve the inline
  cite against `references.rst` and render the link automatically. The
  pattern is exactly: `... feature engineering algorithm [Breimann25a]_.`
- Choose **the few most relevant references** per class — typically 1–2 per
  major method / strategy, plus the project paper at the class level. Do not
  copy a long bibliography from a design document into a docstring; prune to
  what a reader actually needs to follow that specific code path.
- When adding a new reference, place it under the appropriate section in
  `references.rst` and re-use across docstrings — never duplicate the same
  source under different keys.
