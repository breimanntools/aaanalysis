# ADR-0013 — Wire `cpp_run_sample_batched` to a public `CPP.run(n_sample_batches=)`

Status: Accepted — 2026-06-03

## Context

A coverage pass found `cpp_run_sample_batched` (in
`feature_engineering/_backend/cpp_run.py`) and its `_stat_filter` helpers
(`build_feature_index_map`, `accumulate_partial_stats`, `finalize_stats`)
**imported but never called** — no frontend method exposed them. It is a
complete, memory-bounded orchestration (peak `O(batch_size · L · D)` instead
of `O(n)`) intended for very large sample counts, sitting alongside the two
wired orchestrators: `cpp_run_single` (default) and `cpp_run_batch`
(scale-axis batching, reached via `CPP.run(n_batches=)`). Being unreachable,
it was untestable through the public API and accounted for the bulk of the
two files' coverage gaps.

## Decision

**D1 — Expose it via `CPP.run(n_sample_batches=None)`** (seq-mode only),
mirroring how `n_batches` routes to `cpp_run_batch`. `None` disables it;
`>=2` (up to the sample count) routes the run through `cpp_run_sample_batched`.
Validated with the standard `ut.check_number_range`.

**D2 — `n_batches` and `n_sample_batches` are mutually exclusive.** They are
two different batching axes (scales vs samples); setting both raises a
`ValueError`. `run_num` does not gain the parameter (the sample-batched path
uses the seq-mode `assign_scale_values_to_seq`, not pre-sliced tensors).

**D3 — Backward-compatible, semver-minor.** A new optional keyword defaulting
to `None` leaves every existing call unchanged.

## Rejected alternatives

- **Delete the dead orchestrator.** It is a working, useful large-`n`
  memory optimization; removing it would discard real capability, and
  deletion is a hard-rule action requiring separate permission. Wiring keeps
  the value and makes it testable.
- **Test it by calling the backend directly.** Rejected: 115/116 test files
  drive the public API and `frontend-backend.md` mandates backend-trusts-
  frontend; a direct-call test would both violate that convention and leave
  the function unreachable for users.
- **Reuse `n_batches` for both axes.** Rejected: scale-batching and
  sample-batching bound different memory dimensions and are not
  interchangeable; one overloaded knob would be a footgun.

## Consequences

- Public API gains `CPP.run(n_sample_batches=)`; a parity test asserts the
  sample-batched run returns the same feature set as the single-pass run
  (`cpp_run.py` 67%→94%, `_stat_filter.py` 53%→83%).
- The still-unused `_get_split_labels` / `_get_f_split_num` /
  `_get_vf_split_num` helpers in `_stat_filter.py` are genuinely dead
  (referenced nowhere) and are *not* covered by this wiring; their removal is
  a separate deletion pending maintainer permission.
- The raw-`dict_num` arm of `cpp_run_single` (`has_raw_dict_num`) remains a
  dev-bench path, unchanged here.
