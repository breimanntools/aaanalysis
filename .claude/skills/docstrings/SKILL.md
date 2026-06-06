---
name: docstrings
description: Author, check, and critically review AAanalysis docstrings against the house style of the mature classes (CPP, dPULearn, AAclust). Bundles a deterministic structural checker (scripts/check_docstrings.py — numpydoc shape, named Returns, per-method Examples include, cross-reference integrity, citation-defined, stub-skipping) and a doc-vs-signature drift detector (scripts/doc_signature_drift.py — documented default/type vs the real signature). For deep reviews it fans out per-subpackage reviewers, cross-checks each docstring against the implementation + sibling docstrings + the CONTEXT.md glossary for clarity and consistency, classifies fixes as Necessary / Nice-to-have / Can-be, optionally refines the plan with grill-with-docs, then refactors staged or in one pass. Use when writing or reviewing the docstring of any public class/method/function under aaanalysis/, when a new class must match conventions, or when the user asks to audit, grade, lint, or critically review docstring / API-doc quality or find doc-vs-code drift.
---

# AAanalysis docstrings — author, check, review

Two layers. The **structural** layer (the checker) proves a docstring has the
right *shape*; the **critical** layer judges whether it is *correct, clear, and
consistent* with the code. The mature classes **CPP**, **dPULearn**, **AAclust**
are the gold standard — every public symbol should read as if written by the
same hand.

> **Authoritative rules:** the published **Docstring Style Guide**
> (`docs/source/index/docstring_guide.rst`) is the single source of truth for
> templates, baselines, citations, versioning, examples, and the rule →
> checker-code table. This skill *enforces* it; when in doubt, the guide wins.
> Detailed pointers: [REFERENCE.md](REFERENCE.md).

## Quick start

```bash
SK=.claude/skills/docstrings/scripts
# 1. Structural shape (defects fail; advisory + stubs don't):
python $SK/check_docstrings.py aaanalysis        # or a single file; --fix for safe edits
# 2. Doc-vs-signature drift (high-signal, the Necessary tier):
python $SK/doc_signature_drift.py aaanalysis
```

- `check_docstrings.py` flags the structural codes (summary form, Parameters in
  `__init__`, named `Returns`, per-method `Examples` include, `See Also` bullets,
  `Notes` `*`-bullets, `CITATION-UNDEFINED`, `XREF-UNRESOLVED`, …). Output splits
  **Defects** (exit ≠ 0) from **Advisory** (`CLASS-NO-CITATION`,
  `RAISES-UNDOCUMENTED` — never fail) and **skips `UNDER CONSTRUCTION` stubs**.
  `0 defects` = the convention holds for every real symbol.
- `doc_signature_drift.py` flags `DEFAULT-DRIFT` and `TYPE-DRIFT` (documented
  default/scalar-type ≠ signature), plus noisier `PARAM-UNDOCUMENTED` /
  `PARAM-EXTRA`. It suppresses intentional patterns (signature default `None` /
  `ut.X` constants / non-literal factories / ambiguous unions), so the drift hits
  are almost always real.
- **The docs build is the final gate** neither script sees: `cd docs && make
  html` must succeed (RST render errors, broken `.. include::`, stale autosummary
  stubs surface only there).

## Authoring a new class / method

Copy the matching template from the guide / [REFERENCE.md](REFERENCE.md):
- **Class** → noun-phrase summary `<Full Name> (**ACRONYM**) class ...` after a
  blank first line, ending in a `[Key]_` citation **only if that reference
  genuinely describes the class** (rule 7); `.. versionadded::`; then `Attributes`
  only. **Parameters / Notes / See Also / Examples go in `__init__`.**
- **Method** → verb-phrase summary (no `→`/`+`); sections ordered `Parameters →
  Returns → Raises → Notes → Warnings → See Also → Examples`; named `Returns`;
  end with `Examples` → `.. include:: examples/<name>.rst`.
- **Plot pair** → `Plotting class for :class:`<X>` ...`; reciprocal `See Also`.

## House-style rules at a glance

1. Class summary = noun phrase + `**ACRONYM**` (not a verb); `[Key]_` only when one genuinely applies (rule 7).
2. `Parameters` live in `__init__`; the class docstring carries at most `Attributes`.
3. Every public method ends with one `Examples` `.. include:: examples/<name>.rst`.
4. `Returns` is named and matches the returned variable.
5. Recurring params (`df_seq`, `labels`, `n_jobs`, `random_state`) reuse the canonical baseline sentence; **each docstring is self-contained — document every parameter as its own entry**, never lump them into `a, b, c : See :meth:`other``.
6. `See Also` = `* :role:`Target`: gloss.` bullets. Every `:class:`/`:meth:`/`:func:` target must resolve (`XREF-UNRESOLVED`).
7. **Verify every citation — never invent one.** Valid only if `Key` is defined in `references.rst` *and* the work describes this symbol. **Citations are the exception** (core algorithm classes); utilities/loaders/preprocessors carry **none** — don't reflexively add `[Breimann25a]_`. When unsure, omit.
8. Header is `Warnings` (not `Warns`); no `→`/`+` in summaries.
9. **Every integrated external tool/method is cited + explained in one sentence.** A method wrapping/running an external tool (DSSP, Chainsaw, Merizo, AFragmenter, MEME/FIMO, cd-hit, mmseqs2, logomaker, SHAP, …) names it with a `[Key]_` citation (its paper in `references.rst`, not a bare repo URL) and a one-sentence "what it does".
10. **Summary + description.** A one-line summary is followed by a short plain-language description (what it does in simple words, the cited tool `[Key]_` if any, key `:role:` cross-refs) — for classes and non-trivial methods/functions. The checker's `SUMMARY-ONLY` (advisory) flags summary-only docstrings.
11. **Expand abbreviations on first use.** The first time an abbreviation/acronym appears in a docstring, spell it out with the short form in parentheses — e.g. "Command Line Interface (CLI)", "Position Weight Matrix (PWM)" — then use the short form. Each docstring is self-contained (re-introduce per docstring); the bold `(**ACRONYM**)` in a class summary is the class-level form. Glossary terms may use `:term:` instead; universally standard forms (DNA, 3D, ID, CPU, PDB) need no expansion.

## Critical review (audit → tier → refactor)

1. **Baseline.** Enumerate public symbols (`__init__.__all__`). Run both scripts.
2. **Fan-out.** One reviewer agent per subpackage (parallel), each reading the
   docstrings against the *signatures*, the *implementation*, a *sibling
   docstring*, and the *CONTEXT.md glossary*.
3. **Verify before asserting.** Every specific claim (wrong default/shape, a
   "requires X", a code bug) is a *candidate* until checked against source —
   never report a fix on an unconfirmed false positive.
4. **Tabulate + classify** into the three tiers (below).
5. **Refine (optional).** `grill-with-docs` per doc-group to settle *how* to fix.
6. **Refactor** staged or in one pass; finish when `check_docstrings.py` = 0
   defects, drift has no unexplained `DEFAULT-DRIFT`/`TYPE-DRIFT`, and `make html`
   builds.

**Fix tiers:** 🔴 **Necessary** — wrong / misleading / a code bug (drift, copy-paste,
contradictions, false claims, return shape errors). 🟡 **Nice-to-have** —
incomplete but not wrong (terse params, missing Notes/See-Also, vague prose).
⚪ **Can-be** — cosmetic wording. Stubs are exempt.

**Two review dimensions** (rubric for fan-out agents — rate ✅/🟡/🟠/⚪, note
specific & actionable or "—"): **(a) per-symbol correctness** (mostly the scripts);
**(b) consistency & clarity** — package consistency (terminology/depth vs siblings
+ the CONTEXT.md glossary), ease of understanding (followable without the source),
and faithfulness to the implementation (skim the primary methods). Map a
prose-vs-impl mismatch → Necessary; an inconsistency/unclear explanation →
Nice-to-have; wording → Can-be. **Guardrails:** never suggest adding a citation
to a utility, nor an `Examples` include where one exists.

## See also

- The published guide (`docs/source/index/docstring_guide.rst`) — authoritative rules.
- `grill-with-docs` — refine the per-doc-group update plan before editing.
