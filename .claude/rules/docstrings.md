---
paths:
  - "aaanalysis/**/*.py"
  - "docs/source/**/*.rst"
---

# Docstrings (pointer)

> **Single source of truth:** the **Docstring Style Guide** at
> `docs/source/index/docstring_guide.rst` (published on Read the Docs). This file
> defines no rules of its own — it only flags the always-needed essentials and
> points there. Enforce with the `docstrings` skill / checker.

numpydoc throughout; mirror `CPP` / `AAclust` / `dPULearn`. Highest-frequency
traps to keep front-of-mind on touch:

- **Class summary** = noun phrase ending in a `[Key]_` citation; `Parameters` go
  in `__init__`, not the class docstring.
- **Recurring params** (`df_seq`, `labels`, `n_jobs`, `random_state`, ...) reuse
  the canonical baseline sentence verbatim; method specifics are a suffix.
- **`Returns` is named** (`name : type`); every public method ends with one
  `Examples` → `.. include:: examples/<name>.rst`.
- **Citations are `[Key]_` only** (definitions live in
  `docs/source/index/references.rst`); never inline a full reference or URL.
- **No ADR references in project code or GitHub (hard rule).** Never cite an
  ADR number / "see ADR-XXXX" / a `docs/adr/...` path in any `.py` file
  (docstrings **and** `#` comments) or in GitHub issues / PRs / review comments.
  ADRs change and aren't part of the RTD docs, so the pointer goes stale and
  leaks internal process. State the *why* in plain language; ADR cross-refs stay
  in the decision layer only: `docs/adr/`, `CONTEXT.md`, `.claude/rules/`,
  `docs/guides/`.
- Module docstring opens with `"""This is a script for ..."""` (CLAUDE.md §3).
- **Example/tutorial notebooks** (the `.ipynb` behind each `examples/<name>.rst`)
  show DataFrames with `aa.display_df(df, n_rows=10, show_shape=True)` — never
  `print(df)` / bare `df` / `df.head()` (CLAUDE.md §3).
- **Class abbreviations are registered.** Every public class has one canonical
  abbreviation, used identically as the instance variable (`aac = aa.AAclust()`)
  and the example-notebook filename stem (`aac_fit.ipynb`). Registry + rules: the
  **Class abbreviations** section of `docstring_guide.rst`; enforced across
  examples + tutorials + protocols by
  `tests/unit/api_tests/test_class_abbreviation_registry.py`. Highlights:
  `ShapModel`→`sm`, `SeqMut`→`seqm`, `AAWindowSampler`→`aaws`, `AAMut`→`aam`,
  `CPPGrid`→`cppg`, `AALogo`→`aal`; plot pair = `<base>_plot`.
- **Instance = the bare abbr, always.** `cpp = aa.CPP(...)`, never `cpp_res`/
  `cpp_dom`; reassign the bare name per iteration and let *outputs* carry the
  qualifier (`df_feat_res`). A `<abbr>_<qualifier>` instance name is allowed only
  for a genuinely concurrent second instance (`aaws_strict`).
- **Output / data-object names** are canonical too (`df_seq`, `df_parts`,
  `split_kws` — not `skw`/`sks` —, `df_feat`, `X`, `labels`, `df_eval`, …); see the
  **Output / data-object names** table in `docstring_guide.rst` (documentation,
  not test-gated). Qualifiers (`df_feat_res`, `df_top15`) live on the data level.

Run the checker before/after editing docstrings:
`python .claude/skills/docstrings/scripts/check_docstrings.py <path>`.
