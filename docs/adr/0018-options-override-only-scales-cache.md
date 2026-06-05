# ADR-0018 — `options['df_scales'|'df_cat']` are override-only; default memoization is internal

Status: Accepted — 2026-06-05

## Context

`load_default_scales` memoized the bundled default frame by **writing** it into
`aaanalysis.options['df_scales']` / `['df_cat']`. Those two keys are also a
**documented user override** — referenced in the public docstrings of `CPP`,
`CPPPlot`, and `SequenceFeature` as *"… unless specified in `options['df_scales']`"*,
and used as a user-facing knob in the `nf_extend_alphabet` and `options` example
notebooks. Using the same slot as an internal cache conflated two meanings ("the
user set this" vs. "the library cached this"): the default frame appeared in the
option after first use, state leaked across tests, and the write made the
loader unsafe under parallel execution. `config.md` already forbids `options[...]`
as an internal cache. Disk I/O was *already* cached by the module-level
`read_csv_cached` LRU, so the `options` write only memoized the extra
`.astype(float)` step.

## Decision

**D1 — `options['df_scales']` and `options['df_cat']` are override-only.** The
library never writes them. They reflect user intent only and read back `None`
until the user sets them.

**D2 — Memoize the default frame in a private `@lru_cache` loader.** A new
`_load_default_scales_cached(scale_cat)` in `utils.py` holds the default;
`load_default_scales` returns the user override when set, else a copy of the
cached default.

## Rejected alternatives

- **Remove the `df_scales` / `df_cat` keys entirely** (treat as purely
  internal). Rejected: they are a shipped, documented public override; removing
  them is a breaking change with no benefit.
- **Keep the auto-populate behavior but guard it.** Rejected: the conflation is
  the bug. There is no safe way to keep "the default appears in the user option
  after first use" without re-introducing the leak.
- **Deprecation shim warning on auto-populate reads.** Rejected as overkill for
  an internal leak: nothing in the package reads the key expecting
  auto-population, and the test fixture already resets it to `None`.

## Consequences

- Observable behavior change: after a CPP run, `options['df_scales']` /
  `['df_cat']` now read `None` (previously the default frame appeared there).
  No semver-relevant API change — only internal state stopped leaking into a
  user-visible slot.
- Regression guard: `tests/unit/utils_tests/test_load_default_scales.py` asserts
  the default path leaves both keys `None`, plus override/copy/mutation isolation.
- Complements `config.md`'s "do not use `options[...]` as an internal cache" rule.
