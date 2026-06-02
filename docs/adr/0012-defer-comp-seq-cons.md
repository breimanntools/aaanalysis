# ADR-0012 — Defer `comp_seq_cons`: neither test nor integrate the orphaned conservation module

Status: Accepted — 2026-06-02

## Context

A coverage-hardening pass found `aaanalysis/seq_analysis_pro/_comp_seq_cons.py`
at 0% line coverage. Inspection showed it is not merely untested but
**orphaned and partly incorrect**:

- **Unreachable.** It is exported from no `__init__.py` (neither
  `seq_analysis_pro/__init__.py` nor the top-level pro-stub block) and is
  referenced nowhere else in the package or tests. No user can call it through
  the public API.
- **Logically broken.** `get_msa` fetches from the retired
  `www.uniprot.org/uniprot/<id>.fasta` endpoint (UniProt moved to
  `rest.uniprot.org`) and downloads a **single** sequence; `comp_seq_cons`
  then feeds that single-sequence file to `Bio.AlignIO.read` and computes
  per-column Shannon entropy — meaningless on one sequence. `check_is_tool` is
  defined but never called.
- **Non-compliant.** No validation block, no `ut.COL_*` constants
  (hardcodes `"Position"` / `"Conservation_Score"`), no numpydoc class/citation/
  `Examples` include, and no frontend/backend split — it conforms to none of
  the house rules its siblings (`comp_seq_sim`, `filter_seq`, `scan_motif`)
  follow.

The coverage pass had to decide what to do with it before reporting a number.

## Decision

**D1 — Exclude `_comp_seq_cons.py` from the coverage/test scope.** Writing
tests (even with a mocked network) for code that no public entry point can
reach, and whose core computation is incorrect, would lock in broken behaviour
and inflate the coverage number without protecting any user-facing surface.

**D2 — Do not delete it in this change.** Deletion is a hard-rule action
requiring explicit maintainer permission (root `CLAUDE.md` §0); it is recorded
here as the recommended next step, not performed.

**D3 — Do not integrate/export it in this change.** Proper integration would
touch two CONFIRM-FIRST surfaces (`seq_analysis_pro/__init__.py` and the
top-level `aaanalysis/__init__.py` pro-stub) and amounts to a feature PR, out
of scope for a coverage pass.

## Rejected alternatives

- **Test it with a mocked network now.** Rejected: it tests an unreachable,
  incorrect pipeline; the green checkmark would be misleading and the
  `get_msa`/`AlignIO` mismatch would survive untouched.
- **Export + integrate, then test now.** Rejected for this change: it is a
  feature addition (new public pro API, frontend/backend split, house
  docstring, example handling for a network-bound function) that changes the
  public surface and belongs in its own PR, not a coverage pass.
- **Delete it now.** Rejected here only on process grounds (§0 permission);
  it remains the recommended resolution if the conservation feature is not
  wanted.

## Consequences

- The 0% line for `_comp_seq_cons.py` persists by design; the coverage delta
  reported by this pass is measured over the reachable surface.
- Two future options remain open, each its own change with the required §0
  permission: (a) **remove** the module, or (b) **rebuild it properly** — fix
  `get_msa` to retrieve a real MSA from the current UniProt/EBI endpoint (or
  accept a precomputed alignment), give it a validating frontend +
  `_backend/comp_seq_cons.py`, house docstring, and register it in the three
  pro-export sites alongside `comp_seq_sim`.

## Out of scope

Integrating, exporting, deleting, or correcting the module — all deferred per
D1–D3.
