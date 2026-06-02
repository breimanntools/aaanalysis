---
name: docstring-consistency
description: Audit and author AAanalysis docstrings against the house style set by the mature classes CPP, dPULearn, and AAclust (numpydoc: noun-phrase class summary + project citation, Parameters in __init__, named Returns, a per-method Examples `.. include::`, `* :role:` See Also bullets, `[Key]_` citations only). Use when writing or reviewing the docstring of any public class/method/function under aaanalysis/, when a newly added class (e.g. a Preprocessor) needs to match existing conventions, or when the user asks to make the documentation consistent. Bundles a deterministic checker (scripts/check_docstrings.py).
---

# AAanalysis Docstring Consistency

The mature classes **CPP** (`feature_engineering/_cpp.py`), **dPULearn**
(`pu_learning/_dpulearn.py`), and **AAclust** (`feature_engineering/_aaclust.py`)
are the gold standard. Every other public symbol should read as if written by the
same hand. This skill complements `.claude/rules/docstrings.md` (the always-on
style rule) with an invokable cross-API audit + authoring workflow.

## Quick start

Run the deterministic checker, then read the findings:

```bash
python .claude/skills/docstring-consistency/scripts/check_docstrings.py aaanalysis/
# a single file:
python .claude/skills/docstring-consistency/scripts/check_docstrings.py aaanalysis/data_handling/_embed_preproc.py
# auto-fix the safe mechanical drift, then report what is left:
python .claude/skills/docstring-consistency/scripts/check_docstrings.py --fix aaanalysis/
```

It flags 17 structural check codes (each printed with a one-line legend). It does
**not** judge prose quality — pair it with the checklist below and `REFERENCE.md`.
The public-API filter comes from `aaanalysis/__init__.py` (`--api` to override),
so file-local `check_*` helpers are never flagged.

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

- **Class** → noun-phrase summary `<Full Name> (**ACRONYM**) class ... [Breimann2x_]_.`
  on the line *after* a blank first line; `.. versionadded::` next; then
  `Attributes` only (sklearn `_`-state). Put **Parameters / Notes / See Also /
  Examples in `__init__`**, never in the class docstring.
- **Method** → verb-phrase summary (no `→`/`+` shorthand); sections in order
  `Parameters → Returns → Raises → Notes → Warnings → See Also → Examples`;
  named `Returns`; end with `Examples` → `.. include:: examples/<name>.rst`.
- **Plot pair** → summary `Plotting class for :class:`<X>` ...`; reciprocal
  `See Also` bullets between logic and plot class.

## House-style rules at a glance

1. Class summary = noun phrase + `**ACRONYM**` + `[Key]_` citation (not a verb).
2. `Parameters` live in `__init__`; the class docstring carries at most `Attributes`.
3. Every public method ends with one `Examples` `.. include:: examples/<name>.rst`.
4. `Returns` value is named and matches the returned variable.
5. Recurring params (`df_seq`, `labels`, `n_jobs`, `random_state`) reuse the
   canonical baseline sentence verbatim; method specifics are a suffix.
6. `See Also` = `* :role:`Target`: gloss.` bullets (single colon, no bare names).
7. Citations are `[Key]_` only — no inline `.. [Key]`, raw URL, or `(Author Year)`.
8. Section header is `Warnings`, not `Warns`; no `→`/`+` in any summary line.

Full 15-point checklist, annotated templates, and a worked example of the
Preprocessor drift: [REFERENCE.md](REFERENCE.md).
