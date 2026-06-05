# ADR-0021 — `scan_motif` is a true FIMO significance scanner, not a parity twin

Status: Accepted — 2026-06-05

## Context

`aa.scan_motif` (pro, gated on the FIMO/MEME-suite binary) was introduced as a
*parity twin* of the pure-Python `AAWindowSampler.sample_motif_matched`: it ran
`fimo --text --thresh 1.0` (which reports **every** window position), then
**discarded FIMO's score and re-scored each window with the raw PWM-sum**
(`score_window_pwm_`) so its output was byte-for-byte identical to the core
method. A parity test asserted set-equality of `(entry, source_position)` and
numeric closeness of `motif_score`.

The consequence: the external FIMO dependency produced **no data the pure-Python
path didn't already produce**. FIMO's distinctive capability — probabilistic
matching against a background model, yielding a per-occurrence **p-value** — was
computed and thrown away. The CI cost (a from-source MEME build, ADR-0019) and
the maintenance of a second public symbol bought nothing but a slower way to get
the same windows. The "in-core / in-pro parity pattern" in
`pro-core-boundary.md` actively *prescribed* this redundancy.

## Decision

**D1 — Repurpose `scan_motif` to genuine FIMO matching.** It no longer
re-scores to parity. FIMO selects the hits itself: `fimo --text --thresh
<pvalue_threshold>` reports only occurrences whose match p-value is below the
threshold. The output keeps FIMO's `score` (log-odds) as `motif_score` and adds
a `p_value` column; hits are ranked by ascending p-value and capped at `n`. The
old `motif_score_threshold` parameter (raw PWM-sum, parity-only) is **replaced**
by `pvalue_threshold` (default `1e-4`). The three FIMO knobs (`bg_file`,
`motif_pseudo`, `max_stored_scores`) are now *meaningful* — they tune FIMO's
real scoring.

**D2 — Keep both, each in its natural home.** `sample_motif_matched` stays the
core, pure-Python, PWM-sum sampler; `scan_motif` stays in `seq_analysis_pro` as
the FIMO p-value scanner. They are **complementary, not redundant**: different
selection criterion → different windows → different `motif_score` semantics,
with `scan_motif` additionally reporting significance. The pro placement is
correct because FIMO is a fragile external binary.

**D3 — Test the difference, not parity.** The parity test is removed and
replaced by a divergence test (an overlapping window carries different
`motif_score`; only `scan_motif` reports `p_value`) plus structural/relational
golden checks. Exact FIMO floats are environment-sensitive and are **not**
frozen (any exact-value check would go behind `@pytest.mark.regression`, per
ADR-0015).

**D4 — The signature change is free.** v1.1.0 is unreleased
(`scan_motif` is `versionadded:: 1.1.0`, never shipped), so replacing
`motif_score_threshold` with `pvalue_threshold` needs no deprecation cycle.

## Rejected alternatives

- **Drop `scan_motif` entirely.** It was redundant, so removing it (rather than
  fixing it) is defensible. Rejected: FIMO's significance-based selection is a
  *useful, distinct* way to mine training windows; the fix recovers that value
  instead of discarding the feature.
- **Fold FIMO into `sample_motif_matched` as an `engine="fimo"` switch.**
  Maximises one-method discoverability, but the two engines need incompatible
  thresholds (PWM-sum vs p-value) and score columns in one signature, and it
  drags a fragile external binary into a core method. Rejected for a clean
  pro/core split.
- **Keep the parity twin.** Rejected outright — it is the redundancy this ADR
  exists to remove.

## Consequences

- The FIMO CI build (ADR-0019) and `@fimo_required` test gating remain valid and
  necessary — FIMO is still required to exercise the scanner end-to-end.
- `scan_motif` and `sample_motif_matched` now return different hit sets for the
  same PWM; the `motif_score` column means *FIMO log-odds* in one and *PWM-sum*
  in the other. This is documented in both docstrings.
- A helper `check_pwm` was extracted from `check_motif_args` so the PWM can be
  validated without a (now-absent) raw-score threshold; `check_motif_args`
  behaviour is unchanged (it delegates).
- The "in-core / in-pro parity pattern" in `pro-core-boundary.md` is superseded
  there by the "complementary pattern": a pro ex-CLI wrapper must produce
  genuinely different output than its core sibling, or it belongs in core.
- `[Bailey09]` / `[Grant11]` citations stay in use.
