# Session kickoff — Issue #66: General Sampling Strategy Class (`NegativeSampler`)

**Verdict:** ✅ Ready (prio:1, `type:feature`, topic:data) — but **scope it down hard**.
**Run as:** `/github-issue-handoff` (context) → `/grill-with-docs` (resolve decisions) → implement.

## TL;DR
The capability already exists as **`AAWindowSampler`** (`seq_analysis/_aa_window_sampler.py`, `.. versionadded:: 1.1.0`):
same-protein, different-protein, synthetic, motif-matched sampling with role tagging (N/U/Control), per-call `seed`,
distance-to-positive bands, residue-context & identity-similarity filters, unified `segments` schema. **#66 is ~80% a thin
orchestration + config + alias layer over it, not a new sampler.** The grill's central job: decide wrap-vs-fork and prune the duplicated 60%.

## Scope / standards
- **Core**, lives at `seq_analysis/_negative_sampler.py`, exported in `seq_analysis/__init__.py`. **Reuse the existing
  `_backend/aa_window_sampler/` backend — do not fork it.** No `*Plot` pair (utility class, like `AAWindowSampler`). No `.fit`/`.run`.
- **Pro touch point:** `similarity_metric ∈ {cpp, embedding}` relies on `seq_analysis_pro` (`comp_seq_sim`, `scan_motif`) →
  gate behind the `pro` extra via the `missing_feature_stub` path; `pwm` + identity filters stay core.

## Reuse map (issue requirement → existing code)
- same-protein/role=N → `sample_same_protein()` `_aa_window_sampler.py:363`
- different-protein/role=U → `sample_different_protein()` `:528` (has `role=`; `treat_as` is sugar over it)
- synthetic/role=Control → `sample_synthetic()` `:664` (uniform/global_freq/position_specific/scrambled + presets)
- distance-to-positive → `min/max_distance_to_pos` `:368`; residue-class → `aa_context_col`/`_filter_aa_context` `:176` + `match_residue_type` (`_annot_preproc.py:851`)
- similarity → motif PWM `:118`, identity `:319`, cpp hard-decoy `sample_motif_matched()` `:835`
- constants → `ut.ROLE_*`/`LIST_ROLES` (`utils.py:194`), `ut.STRATEGY_*` (`:201`), `ut.COLS_SEGMENTS` (`:184`)
- **Genuinely new:** `custom_filter` hook, `SamplingFilters` config, `sample_benchmark_set(arms)`, `provenance` column, structure filters.

## Decisions for grilling (recommended answers)
- **D1 Relationship** → new `NegativeSampler` frontend that **wraps/composes** `AAWindowSampler` and reuses its backends; do *not* subclass or edit AAWindowSampler's stable API. `__init__` stores `df_pos`/`df_seq`/`SamplingFilters`; each `sample_*` translates filters→AAWindowSampler kwargs and delegates.
- **D2 `SamplingFilters` dataclass?** → **Yes.** The no-typed-records rule (`sharp-edges.md`/`backend-dataframes.md`) is about DataFrame **rows**, not config objects. Keep it frontend-only, validated in `check_sampling_filters()`.
- **D3 Metric split** → `pwm`/identity = core; `cpp`/`embedding` = pro-gated with friendly install hint.
- **D4 Schema** → reuse `ut.COL_ENTRY` for `source_protein`; **add only** `ut.COL_PROVENANCE` (+ maybe `COL_ARM`) and a `COLS_NEG_SAMPLES` bundle. Don't rename to `source_protein` (breaks AAWindowSampler parity).
- **D5 `sample_benchmark_set(arms, seed)`** → `arms: Dict[name → {method, **kwargs}]`; per-arm sub-seed (`SeedSequence.spawn`), tag `ut.COL_ARM`, `concat`, dedupe on `entry_win` (reuse `build_output.py:36`). Pin: seed scheme, dedupe, YAML-now-or-defer.
- **D6 residue-class P1/P1'** → map onto `aa_context` machinery + `match_residue_type`; confirm Schechter–Berger anchor convention (`_aa_window_sampler.py:303`).

## Cut / defer (scope creep)
Structure filters (`match_structure`/`structure_tolerance`) → defer to a follow-up or ship as a `custom_filter` recipe (the issue allows this). YAML round-trip → defer unless trivial. Explicit `markov_order` generator → scope or map to existing modes. **Out of scope (honor issue):** PU learning, MEROPS/domain loaders.

## Test plan
New `tests/unit/negative_sampler_tests/` mirroring `aa_window_sampler_tests/`: schema/column-set golden; role tagging (+`treat_as`);
filter composition + order-stability; synthetic frequency match within tolerance; `seed` determinism per method **and** for `sample_benchmark_set`;
error-message tests (`match_structure` w/o `structure_features`; `cpp` metric w/o pro). One end-to-end multi-arm example under `examples/` (nbmake gate).

## Launch command (paste in a fresh session)
```
/grill-with-docs Implement issue #66 per docs/issue_kickoffs/issue_66.md. First load github-issue-handoff
context. Central decision: wrap/compose AAWindowSampler (reuse its backend) vs new code — resolve D1–D6,
prune the ~60% that already exists, and cut structure filters + YAML to a follow-up.
```
