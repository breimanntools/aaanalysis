# Session kickoff — Issue #17: tmd_start/tmd_stop indexing consistency

**Verdict:** 🔄 Revisit (prio:1, `type:bug`). Smallest of the three; correctness-critical.
**Run as:** `/github-issue-handoff` (context) → `/grill-with-docs` (resolve decisions below) → implement.

## Scope / standards
In-scope core bug on the position-based `df_seq` schema. Touches `SequenceFeature.get_df_parts`
behavior → **potentially semver-relevant** (`api-stability.md`: behavior change → major bump). The
grill must settle whether current output is *wrong* (breaking fix) or merely *under-derived* (clarity refactor).

## Ground truth (from code)
Convention is **1-based, start-inclusive, stop-inclusive** (`_sequence_feature.py:237,244-245`), and every
consumer implements it: `Parts.get_tmd` (`_backend/cpp/_part.py:35-37`), `_slice_dict_num_to_basic_parts`
(`_filters/_assign.py:279-291`), `expand_pos_anchors_` (`check_feature.py:318-319`, the correct reference).

**The smell:** `_get_tmd_positions` (`check_feature.py:215-223`) computes `tmd_stop = tmd_start + len(tmd)`
(0-based **exclusive**) then does `tmd_start += 1` but **never converts `tmd_stop`** → it equals the
1-based inclusive last position *only by arithmetic coincidence*. The two columns are defined under two
conventions that agree by luck; any future edit silently breaks it. The `seq_based` branch
(`check_feature.py:267-273`) is a third construction path to reconcile.

## Files
- `aaanalysis/feature_engineering/_backend/check_feature.py` — `_get_tmd_positions` (215-223); audit `seq_based` (267-273).
- `aaanalysis/feature_engineering/_sequence_feature.py` (237-245) + `_numerical_feature.py:148` — docstring wording.
- **Reuse, don't re-derive:** `Parts` (`_part.py`) canonical slicer; `expand_pos_anchors_` already encodes the correct geometry via `ut.get_window_offsets`.
- Tests: `tests/unit/sequence_feature_tests/test_sf_get_df_parts.py`, mirror in `numerical_feature_tests/test_nf_get_parts.py`.

## Decisions for grilling (recommended answers)
1. **Convention** → keep 1-based, start- & stop-inclusive (matches UniProt + all consumers). Do *not* switch to exclusive.
2. **Breaking change?** → Target **zero output-value change**: rewrite `_get_tmd_positions` so start & stop *both* express inclusive-1-based explicitly. → patch/minor. Confirm via round-trip test that no column value shifts (esp. len-1 TMD, `seq_based` branch).
3. **If a genuine off-by-one is found** (e.g. in `seq_based`) → that fix *is* a behavior change → escalate to `api-stability.md` (CONFIRM-FIRST, possibly `deprecated` shim).
4. **Single source of truth** → extract one documented helper `_one_based_inclusive_stop(start0, length)` shared by all three construction sites.
5. **Docstring lock** → resolve "inclusive" vs `len()`-exclusive wording; run the `docstrings` checker.
6. **Guards** → keep the `tmd_start == tmd_stop` empty-TMD guard (line 220) but ensure a legit **len-1 TMD** is not rejected.

## Test plan (edge cases)
len-1 TMD; TMD at sequence start (`tmd_start==1`); TMD at end (`tmd_stop==len`); over-long jmd flanks → gap pad;
**round-trip equivalence** (position-based vs part-based vs sequence-TMD give identical `df_parts` — the acceptance
criterion); dict_num tensor parity.

## Launch command (paste in a fresh session)
```
/grill-with-docs Implement issue #17 per docs/issue_kickoffs/issue_17.md. First load github-issue-handoff
context. Resolve decisions 1–6 (especially: is this a zero-output-change clarity refactor or a real
value fix → semver), proving no output deltas with the round-trip test before coding.
```
