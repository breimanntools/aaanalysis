# ADR-0019 — Build FIMO from source on the Linux CI matrix to gate `scan_motif` end-to-end

Status: Accepted — 2026-06-05

## Context

`aa.scan_motif` is a `pro` wrapper around the MEME suite's `fimo` CLI: it uses
FIMO purely as a position scanner (`--text --thresh 1.0 --no-qvalue`) and then
re-scores every reported window in Python with `score_window_pwm_`, the same
PWM-sum scoring as `AAWindowSampler.sample_motif_matched`. Its tests therefore
split in two:

- **validation tests** — binary-free; they mock `shutil.which` so the Validate
  block is reached even where FIMO is absent, and run on the whole matrix; and
- **end-to-end / property / golden / parity tests** — marked `@fimo_required`
  (`pytest.mark.skipif(shutil.which("fimo") is None)`), which need the real
  binary to exercise `_pwm_to_meme` (the MEME-alphabet column permutation, a
  documented sharp edge), `_df_seq_to_fasta`, `_run_fimo`, and the
  CLI-vs-Python parity contract (`pro-core-boundary.md`).

Until now CI never installed FIMO, so every `@fimo_required` test was silently
**skipped in the gate**. The coverage gate (ADR-0016, `--cov=aaanalysis
--cov-fail-under=88`) saw the entire FIMO code path as uncovered, and the
parity guard that `pro-core-boundary.md` mandates was never actually run in CI —
the meatiest behaviour of a published `pro` feature went unverified on every
merge. The existing matrix already installs Linux-only binary tools the suite
needs (`cd-hit`, `mmseqs2`) via `apt-get`, so binary scanners-in-CI is an
established pattern here; FIMO is the one that `apt` cannot satisfy.

## Decision

**D1 — Install FIMO in CI so the `@fimo_required` tests run in the gate.** The
blocking workflows (`main.yml`, `test_coverage.yml`) gain a FIMO install step;
the scan_motif end-to-end / property / golden / parity tests now execute and
count toward coverage instead of skipping.

**D2 — Build from source, not conda.** FIMO is installed by downloading the
pinned MEME tarball (`meme-5.5.8.tar.gz`) and running
`./configure --prefix=$HOME/meme-install --enable-build-libxml2
--enable-build-libxslt && make && make install`. The build is wrapped in an
`actions/cache@v4` keyed on the MEME version (`${{ runner.os }}-meme-5.5.8`), so
the ~5–10 min compile runs only on a cache miss (a version bump or cache
eviction); warm runs restore the prebuilt `~/meme-install` and only re-export
the binary onto `$GITHUB_PATH`.

**D3 — Full Linux matrix, Linux-only.** The install runs on every Linux job
across py 3.10–3.14 (the cache is shared across those jobs, so the cold build
happens once per key). It is gated `if: runner.os == 'Linux'`; on Windows the
`@fimo_required` tests skip, matching how `filter_seq` already skips on Windows.

## Rejected alternatives

- **Keep `@fimo_required` skipped (status quo).** Zero CI cost, but the
  published `pro` feature's end-to-end behaviour and the mandated parity guard
  never run in the gate, and its lines stay uncovered. Rejected — it defeats the
  purpose of having the tests.
- **Install via conda/bioconda (`conda install -c bioconda meme`).** The path
  the docstring suggests to end users and the simplest one-liner, but it pulls a
  full conda/mamba toolchain into a pip-based CI, is heavy, and bioconda
  solves/downloads are themselves a frequent flake source. Explicitly ruled out
  by the maintainer in favour of a source build.
- **`apt-get install meme` (mirror the cd-hit/mmseqs2 pattern).** Preferred for
  consistency, but the MEME suite is not packaged in the Ubuntu repositories, so
  there is nothing to install. Not available.
- **Single dedicated FIMO job (build once, run only `seq_analysis_pro` tests).**
  Cheapest compute and isolates the heavy dependency, but the `@fimo_required`
  tests then run on only one Python version, losing the cross-version signal the
  rest of the matrix provides. Rejected in favour of the full Linux matrix.

## Consequences

- A cold MEME build adds ~5–10 min to the first run on a new cache key (version
  bump or eviction); warm runs add only the tarball-less PATH export. Bumping
  the pinned MEME version means editing the `MEME_VERSION` value **and** the
  cache `key` in both workflows in the same PR.
- The `scan_motif` FIMO code path (`_pwm_to_meme`, `_df_seq_to_fasta`,
  `_run_fimo`) and the CLI-vs-Python parity test now execute under coverage on
  Linux; the parity guard required by `pro-core-boundary.md` is finally enforced
  in CI rather than only locally.
- The build depends on an external download from `meme-suite.org`; an outage
  there fails the cold-build step (warm cache runs are unaffected). The
  `--enable-build-libxml2 --enable-build-libxslt` flags make MEME compile its
  own bundled libxml2/libxslt, avoiding system-library drift; non-fatal Perl
  dependency warnings during `make install` do not affect the `fimo` C binary.
- Windows matrix cells report the `@fimo_required` tests as skipped, not failed.
