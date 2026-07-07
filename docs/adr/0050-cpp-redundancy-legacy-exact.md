# ADR-0050 — CPP redundancy criterion: `legacy` default, `exact` opt-in

Status: Accepted — 2026-07-05

## Context

CPP's redundancy-reduction step (`filtering`) drops a candidate feature when it overlaps
in *position* **and** correlates in *scale* with a higher-ranked kept feature. The
position-overlap gate has, since v1.0.0, compared the **characters** of the comma-joined
position string (`set("11,12,…")` → `{'0'–'9', ','}`) instead of the integer positions. For
multi-digit positions this makes the position gate largely inoperative, so redundancy
reduction behaves as a scale-correlation-within-category filter, and `max_overlap` thresholds
a digit-character overlap rather than true positional overlap.

An empirical audit compared this behaviour against a position-correct variant across 13
internal benchmarks (both γ-secretase sets) + 5 external UniProt tasks:

- The **selected feature set changes substantially** on multi-digit-position datasets
  (Jaccard 0.28–0.53; ~50–72 % of features differ); on single-digit-position data
  (AA 9-mers) it is nearly unchanged (Jaccard 0.82–1.0) — confirming the mechanism.
- **Predictive performance is unchanged** — mean ΔAUC ≈ −0.006 over 16 datasets, better in
  only 3, all within cross-validation noise. No benchmark improves meaningfully (γ-secretase:
  DOM_GSEC +0.005, DOM_GSEC_PU −0.002).

So the position-correct variant is an **interpretability** change (a more concentrated
signature), **not a performance improvement**, and switching the default would break the
reproducibility of every previously published CPP signature (and the ADR-0015 exact-value
anchor).

## Decision

Add `CPP.run(..., redundancy="legacy"|"exact")` (and `CPP.run_num`), **default `"legacy"`**:

- **`"legacy"` (default):** the original character-set position criterion. Byte-identical to
  all prior versions — the default path needs no regolden and keeps published results
  reproducible.
- **`"exact"` (opt-in):** compares the actual residue positions. Framed as an **enhancement,
  not a bug fix**: it yields a more concentrated signature (fewer redundant subcategories) but
  does **not** improve predictive accuracy. Documented in the `run` / `run_num` Notes, with a
  pointer to `CPP.simplify` for a stronger, more efficient reduction.

The switch is an **explicit parameter**, deliberately **not** tied to whether `max_overlap`
was changed — the same numeric value must never silently select a different algorithm.

## Rejected alternatives

- **Switch the default to the position-correct behaviour (regolden).** Rejected for now: no
  measured performance gain, and it reshuffles every published signature.
- **Tie the algorithm to whether `max_overlap` differs from its default.** Rejected: magic,
  surprising (`run()` vs `run(max_overlap=0.5)` would differ), and fragile to detect.
- **Do nothing / document only.** Rejected: the current `max_overlap` has a real but
  scientifically uninterpretable effect; an explicit opt-in gives users a correct alternative.

## Consequences

- Default output is unchanged; the exact-value CPP regression anchor (ADR-0015) stays valid.
- `"exact"` carries a T3 regression anchor under ADR-0032 (frozen feature set on the canonical
  DOM_GSEC cell), run in the non-gating nightly.
- Two redundancy code paths are maintained; `"exact"` may become the default in a future major
  version (with a deprecation cycle) once results are re-anchored.
- `CPP.simplify` keeps its own (stronger) reduction and current behaviour; the `redundancy`
  switch applies only to the `run` / `run_num` filtering stage.
