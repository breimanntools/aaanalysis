---
name: docstring-consistency
description: Audit and author AAanalysis docstrings against the house style set by the mature classes CPP, dPULearn, and AAclust (numpydoc: noun-phrase class summary + project citation, Parameters in __init__, named Returns, a per-method Examples `.. include::`, `* :role:` See Also bullets, `[Key]_` citations only). Use when writing or reviewing the docstring of any public class/method/function under aaanalysis/, when a newly added class (e.g. a Preprocessor) needs to match existing conventions, or when the user asks to make the documentation consistent. Bundles a deterministic checker (scripts/check_docstrings.py).
---

# AAanalysis Docstring Consistency

> **Authoritative rules:** the **Docstring Style Guide** at
> `docs/source/index/docstring_guide.rst` (published on Read the Docs) is the single
> source of truth for templates, baselines, citations, versioning, examples, and the
> rule → checker-code table. This skill *enforces* that guide via
> `scripts/check_docstrings.py`; it defines no rules of its own. The notes below are a
> quick operational summary — when in doubt, the guide wins.

The mature classes **CPP** (`feature_engineering/_cpp.py`), **dPULearn**
(`pu_learning/_dpulearn.py`), and **AAclust** (`feature_engineering/_aaclust.py`)
are the gold standard. Every other public symbol should read as if written by the
same hand.

## Quick start

Run the deterministic checker, then read the findings:

```bash
python .claude/skills/docstring-consistency/scripts/check_docstrings.py aaanalysis/
# a single file:
python .claude/skills/docstring-consistency/scripts/check_docstrings.py aaanalysis/data_handling/_embed_preproc.py
# auto-fix the safe mechanical drift, then report what is left:
python .claude/skills/docstring-consistency/scripts/check_docstrings.py --fix aaanalysis/
```

It flags structural check codes (each printed with a one-line legend). It does
**not** judge prose quality — pair it with the checklist below and `REFERENCE.md`.
The public-API filter comes from `aaanalysis/__init__.py` (`--api` to override),
so file-local `check_*` helpers are never flagged.

**Reading the output — Defects vs Advisory vs skipped:**
- **Defects** are hard convention violations; **the exit code is non-zero iff
  there are defects.** `0 defect(s)` means the complete convention is satisfied
  for every real symbol.
- **Advisory** (`CLASS-NO-CITATION`, `RAISES-UNDOCUMENTED`) never fails the run.
  **Citations are the exception, reserved for important classes/methods** (a
  paper that describes them); the *norm* for utility/helper classes is **no
  citation**. So that list is **not a to-do** — it is a reminder to confirm an
  *important* class isn't missing its citation, **never** a cue to add one to
  utilities or to invent one (see rule 7). `RAISES-UNDOCUMENTED` likewise just
  flags a body that raises without a `Raises` section — document it if the
  exception is user-facing; the package omits it for routine validation.
- **Cross-references are validated** (`XREF-UNRESOLVED`, a defect): every
  `:class:`/`:meth:`/`:func:` target must resolve to a real public symbol
  (registry built package-wide, so single-file runs still resolve cross-file
  refs). External refs like `pandas.DataFrame` are ignored. Watch capitalization
  (`AAlogo` ≠ `AALogo`) and method-on-class typos.
- **The build is the final gate** (not run by this checker): `cd docs && make
  html` (ideally `SPHINXOPTS="-W"`) must finish clean — RST render errors and
  unresolved `.. include::` targets surface only there.
- **Stubs are skipped** (reported only as a count + names): a class whose
  summary starts `UNDER CONSTRUCTION` / a method whose body is just
  `raise NotImplementedError` is explicitly not-ready and exempt from the
  whole convention. A *documented limitation* of a real method (e.g. a `Raises`
  clause "X not yet implemented for numerical mode") is **not** a stub — the
  marker must be in the summary line, not buried in a section.

## Audit workflow

1. Run the checker on the target file(s). Note each `CODE  file:line  symbol`.
2. For every finding, open the symbol and compare against the canonical template
   in [REFERENCE.md](REFERENCE.md). The checker catches structure; you judge the
   noun-phrase summary, the citation choice, and the recurring-param baseline.
3. Fix in place. Re-run the checker until clean (or until remaining findings are
   deliberate and noted).
4. The checker has no opinion on backend code (`_backend/`, `_utils/`) — it only
   scans frontend modules (`aaanalysis/**/_<name>.py`).

## Fix workflow

1. `--fix` applies the **safe, deterministic** edits only: `Warns` → `Warnings`
   (header + underline) and See-Also ` : ` glosses → `: `. Re-run without `--fix`
   to see the rest.
2. Everything else needs judgment and is fixed by hand/agent against the
   [REFERENCE.md](REFERENCE.md) templates: rewrite imperative class summaries to a
   noun phrase + citation, move `Parameters` into `__init__`, add the per-method
   `Examples` include, replace inline/free-text citations with `[Key]_`, restore
   the `df_seq` baseline sentence, name unnamed `Returns`, etc.
3. Re-run the checker until only deliberate findings remain.

## Author workflow (new class/method)

Copy the matching template from [REFERENCE.md](REFERENCE.md) and fill it in:

- **Class** → noun-phrase summary `<Full Name> (**ACRONYM**) class ...` on the
  line *after* a blank first line, ending in a `[Key]_` citation **only if that
  reference genuinely describes the class** (see rule 7 — new data-prep /
  utility classes usually have none and omit it); `.. versionadded::` next; then
  `Attributes` only (sklearn `_`-state). Put **Parameters / Notes / See Also /
  Examples in `__init__`**, never in the class docstring.
- **Method** → verb-phrase summary (no `→`/`+` shorthand); sections in order
  `Parameters → Returns → Raises → Notes → Warnings → See Also → Examples`;
  named `Returns`; end with `Examples` → `.. include:: examples/<name>.rst`.
- **Plot pair** → summary `Plotting class for :class:`<X>` ...`; reciprocal
  `See Also` bullets between logic and plot class.

## House-style rules at a glance

1. Class summary = noun phrase + `**ACRONYM**` (not a verb); a `[Key]_` citation
   only when one genuinely applies (rule 7).
2. `Parameters` live in `__init__`; the class docstring carries at most `Attributes`.
3. Every public method ends with one `Examples` `.. include:: examples/<name>.rst`.
4. `Returns` value is named and matches the returned variable.
5. Recurring params (`df_seq`, `labels`, `n_jobs`, `random_state`) reuse the
   canonical baseline sentence verbatim; method specifics are a suffix. **Each
   docstring is self-contained: document every parameter as its own entry** —
   never lump them into `a, b, c : See :meth:`other`. Same semantics.`
6. `See Also` = `* :role:`Target`: gloss.` bullets (single colon, no bare names).
7. **Verify every citation — never invent one.** A `[Key]_` is valid only when
   (a) `Key` is defined in `docs/source/index/references.rst` *and* (b) that work
   actually describes this class/method. **Do NOT reflexively append the project
   paper `[Breimann25a]_`** — it covers the core γ-secretase CPP/dPULearn
   algorithms, *not* data-prep utilities (the Structure/Annotation/Embedding/
   Sequence preprocessors, loaders, …). When in doubt, **OMIT**: `CLASS-NO-CITATION`
   is advisory; satisfying it with a wrong citation is worse than leaving none.
   The checker's `CITATION-UNDEFINED` catches typo'd/fabricated keys; relevance is
   your call. Citations are `[Key]_` only — no inline `.. [Key]`, raw URL, or
   `(Author Year)`.
8. Section header is `Warnings`, not `Warns`; no `→`/`+` in any summary line.

Full 15-point checklist, annotated templates, and a worked example of the
Preprocessor drift: [REFERENCE.md](REFERENCE.md).
