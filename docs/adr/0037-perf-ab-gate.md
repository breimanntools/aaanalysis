# ADR-0037 — Same-runner A/B vs the latest PyPI release makes wall-clock a merge gate

Status: Accepted — 2026-06-22

Supersedes the *non-gating* stance for the perf benchmark suite established in
[ADR-0015](0015-cpp-regression-anchor.md) and
[ADR-0016](0016-coverage-measurement-and-gates.md). Those ADRs keep their
correctness/coverage decisions; only the "wall-clock never gates merges" part as
it applies to the committed benchmark suite is revised here.

## Context

The committed benchmark suite ([ADR-0033](0033-performance-audit-outcomes.md),
#187/#193) shipped as a **non-gating nightly** that compared each run against a
**static committed baseline** (`tests/benchmarks/perf_baseline.json`). The
reasoning was sound at the time: shared-runner wall-clock is too noisy to gate
merges, so the suite was advisory only.

In practice the static baseline made the job flap red for days. The baseline was
captured once on one machine (and never refreshed via the prescribed
`workflow_dispatch` path); subsequent nightly runs landed on different runner
classes, so **seven of eight benchmarks read 1.1×–2.4× "slower" at once** — the
unmistakable signature of a hardware/baseline mismatch, not a code regression (a
real regression hits one path, not all of them uniformly). The drifting
dependency landscape (numpy/pandas/sklearn versions changing under the baseline)
compounds the same problem.

The root cause is comparing against a **frozen number captured on different
hardware under a different dependency set**. Remove that, and wall-clock becomes
stable enough to act on.

## Decision

**Replace the static baseline with a live same-runner A/B and promote the check
to a merge gate.**

- **D1. A/B on one runner, one job.** Benchmark twice in the same job: the
  current working tree, and the latest stable release installed from PyPI.
  Hardware, OS, Python, and dependency variance affect both runs equally and
  cancel — the only remaining variable is aaanalysis's own source.
- **D2. Shared dependency set (`--no-deps`).** Resolve dependencies once from the
  working-tree install, freeze them, and install the released build with
  `pip install aaanalysis --no-deps` onto that *same* set. The released code thus
  runs under identical numpy/pandas/sklearn versions, isolating the comparison to
  library source only.
- **D3. Baseline = latest stable, resolved dynamically.** No version pin and no
  committed baseline file: `pip install aaanalysis` (no version) selects the
  newest stable release each run, so the anchor never goes stale and needs no
  per-release bump. The comparison is "published artifact vs working tree" — note
  these can share a version *string* (both `1.0.3` today), so the comparison is by
  installed *source*, never by version number.
- **D4. Per-benchmark thresholds.** The residual jitter between the two suite
  runs (minutes apart on one runner) is small for the meaty paths but large in
  relative terms for the sub-2ms micro-benchmarks. So the gate uses **1.3×** on
  the meaty paths and **2.0×** on the sub-2ms micro-paths (the explicit
  `SUBMS_BENCHMARKS` set in `.github/scripts/check_perf_regression.py`).
- **D5. Benchmark on realistic inputs, not tiny ones.** A hot path dominated by
  fixed setup cost (e.g. `CPP.run` on a 10-scale fixture) measures overhead, not
  the kernel — and reads as a regression because the post-1.0.3 overhaul *adds*
  setup overhead that only pays off at scale. The CPP fixtures use the full
  ~586-scale set with **Segment** splits (Segment+Pattern is ~3× slower again,
  too costly for the released side); at that size master is correctly ~2.9×
  faster on `CPP.run` and ~2.4× on `feature_matrix`. Input size is bounded so the
  *slow released side* (which runs the un-optimized 1.0.3 code) stays a few
  seconds per benchmark.
- **D6. Import isolation is mandatory (and guarded).** Running `pytest` from the
  repo root puts the working-tree `aaanalysis/` source on `sys.path`, so the
  *released* venv would import **master**, silently making the A/B a
  master-vs-master comparison that can never see a regression (`tests/__init__.py`
  makes pytest's default prepend mode insert the repo root too). Both runs use
  **`PYTHONSAFEPATH=1`** + **`pytest --import-mode=importlib`**, and a **guard
  step fails the job** if the released venv's `aaanalysis.__file__` resolves into
  the repo. (1.0.3 also lacks `aaanalysis.__version__`; logging reads
  `importlib.metadata.version`.)
- **D7. Merge-gating on PRs and master push** (plus a daily schedule heartbeat),
  with the docs-only `paths-ignore` of the unit matrix. This is the part that
  revises ADR-0015/0016: wall-clock now blocks a merge, because the A/B has
  removed the variance that made gating unsafe.
- **D8. The release baseline only gates methods present in that release —
  accepted, but loudly surfaced.** Because the latest release (1.0.3) predates
  most of the 1.1 performance work, every method added since (`CPP.run_num`,
  `CPP.simplify`, `AAWindowSampler`, `StructurePreprocessor.encode_*`,
  `prune_by_correlation`, …) has no baseline and is **measured but not gated**
  until the next release ships. We keep the release baseline anyway (a stable,
  meaningful "did we regress vs what users run today?" anchor), and the comparator
  prints a `COVERAGE: N/M gated` line plus the explicit pending-gate list so a
  green run is never mistaken for full protection. As of this writing 11/16
  benchmarks are gated.
- **D8b. The pending-gate set is self-enforcing, not a hand-kept ledger.** Each
  benchmark stamps the release that first contains its method into the run JSON
  (`benchmark.extra_info["gated_since"]`). The comparator, given the released
  baseline version (`--released-version`), enforces the invariant: a benchmark
  whose `gated_since <= released version` **must** appear in the released run;
  if it is absent the method was renamed/removed and the gate silently stopped
  protecting it, so the comparator **fails**. Newer benchmarks
  (`gated_since > released`) are reported "pending-gate" and auto-gate when their
  release ships — no manual trigger. This turns the 1.0.3→1.1.0 transition into a
  checked event rather than a silent one.
- **D9. Benchmarks newer than the published release are not gated.** A benchmark
  exercising an API absent from the latest release errors out of the released
  run's JSON; the comparator (intersection-only) reports it as new/unbaselined
  and never fails on it. The released benchmark step is tolerant by construction.

## Consequences

- The static `tests/benchmarks/perf_baseline.json` and the `--update` refresh
  path are obsolete and removed; the comparator now takes two run JSONs.
- A genuine post-release slowdown on a hot path turns a PR red before merge,
  rather than surfacing days later in a nightly. The correctness anchors
  (ADR-0015 regression test, ADR-0032 tolerance tiers) are unchanged and remain
  the authority on *output* equivalence; this gate governs *speed* only.
- Cost: ~1–2 min added per PR (two installs + two suite runs on one runner), and
  wall-clock is back in the merge path — accepted because the A/B makes it
  reliable. If a sub-2ms path ever flaps despite the 2.0× band, widen its band or
  drop it from the gated set rather than reverting to a static baseline.
