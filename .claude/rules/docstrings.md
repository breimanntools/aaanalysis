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
- Module docstring opens with `"""This is a script for ..."""` (CLAUDE.md §3).

Run the checker before/after editing docstrings:
`python .claude/skills/docstrings/scripts/check_docstrings.py <path>`.
