"""Perf-regression guard: same-runner A/B of two benchmark runs (issue #187).

Mirrors ``.github/scripts/check_branch_coverage.py``: a small, dependency-free
CI helper. It compares two ``pytest-benchmark`` JSON runs produced **on the same
runner in the same job** -- the *current working tree* against the *latest stable
release installed from PyPI* -- and exits non-zero when any covered hot path is
slower in the current run beyond its threshold.

Why two live runs instead of a committed static baseline: a frozen number is
captured on one machine and then compared against whatever (faster or slower)
runner the job lands on next, so cross-runner hardware variance -- and drifting
dependency versions -- masquerade as regressions. Benchmarking *both* builds on
the *one* runner, under the *one* dependency set, cancels hardware, OS, Python
and dependency variance: the only variable left is aaanalysis's own source.

Per-benchmark thresholds absorb the residual temporal jitter between the two
suite runs (they are minutes apart). The sub-2ms micro-benchmarks are far noisier
in relative terms than the meaty paths, so they get a looser band.

Pending-gate guard (``gated_since``): the baseline is the latest RELEASE, so a
benchmark whose method post-dates that release has no baseline and is measured-
but-not-gated -- it auto-gates the moment its release ships (the dynamic baseline
catches up). Each benchmark declares the release that first contains its method
via ``benchmark.extra_info["gated_since"]`` (rides into the JSON). With
``--released-version`` this script enforces the invariant: a benchmark whose
``gated_since <= released version`` MUST appear in the released run; if it is
absent (method renamed/removed before release, so the gate silently lost it) the
script FAILS. Benchmarks with ``gated_since > released version`` are reported as
"pending-gate" and are not failed on.

It lives under ``.github/scripts/`` (not ``dev_scripts/``, which is git-ignored)
so the CI checkout can run it. Like the coverage gate it prints to stdout for the
CI log; the package itself never calls ``print`` (it uses ``ut.print_out``).

Compare the current run against the released run::

    # current working tree
    pytest tests/benchmarks --benchmark-json=current_run.json -c tests/pytest.ini
    # latest stable release, installed --no-deps onto the same dependency set
    pytest tests/benchmarks --benchmark-json=released_run.json -c tests/pytest.ini
    python .github/scripts/check_perf_regression.py current_run.json \
        --baseline-run released_run.json
"""
import argparse
import json
import sys
from pathlib import Path

# A covered path is flagged once current/released exceeds its threshold. The
# meaty paths get a tight 1.3x band; the sub-2ms micro-benchmarks get a looser
# 2.0x band because their relative jitter on shared runners is far higher (a
# 0.7ms benchmark swinging +/-0.3ms is +/-40% on noise alone). Names match the
# benchmark functions in tests/benchmarks/test_perf_hot_paths.py.
BIG_PATH_THRESHOLD = 1.3
SUBMS_THRESHOLD = 2.0
SUBMS_BENCHMARKS = {
    "test_aa_window_sampler",
    "test_aaclust_fit",
    "test_encode_pdb",
    "test_dpulearn_fit",
    "test_get_sliding_aa_window",
    "test_encode_one_hot",
}


def threshold_for(name):
    """Per-benchmark slowdown threshold (looser for the sub-2ms micro-paths)."""
    return SUBMS_THRESHOLD if name in SUBMS_BENCHMARKS else BIG_PATH_THRESHOLD


def version_tuple(v):
    """Parse a dotted version (e.g. ``'1.0.3'``) to an int tuple for comparison.

    Stops at the first non-numeric component so a tag like ``1.1.0rc1`` parses to
    ``(1, 1, 0)``; returns ``()`` for an unparseable/empty value.
    """
    parts = []
    for chunk in str(v).split("."):
        lead = ""
        for c in chunk:
            if not c.isdigit():
                break
            lead += c
        if not lead:
            break
        parts.append(int(lead))
    return tuple(parts)


# I Helper Functions
def read_run_medians(run_json_path):
    """Return ``{benchmark_name: median_seconds}`` from a pytest-benchmark JSON.

    ``pytest-benchmark --benchmark-json=FILE`` writes a top-level ``benchmarks``
    list; each entry carries a ``name`` and a ``stats`` dict with a ``median``
    (seconds). We key on the short ``name`` (e.g. ``test_cpp_run``) so the
    report stays readable and machine-path-independent.
    """
    data = json.loads(Path(run_json_path).read_text())
    medians = {}
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        medians[name] = float(bench["stats"]["median"])
    return medians


def read_gated_since(run_json_path):
    """Return ``{benchmark_name: gated_since}`` from a run's per-benchmark extra_info.

    Each benchmark sets ``benchmark.extra_info["gated_since"] = "<release>"`` (the
    release that first contains its method); pytest-benchmark stores it in the JSON.
    Benchmarks without the tag are omitted.
    """
    data = json.loads(Path(run_json_path).read_text())
    out = {}
    for bench in data.get("benchmarks", []):
        gs = (bench.get("extra_info") or {}).get("gated_since")
        if gs:
            out[bench["name"]] = gs
    return out


def compare(current_medians, baseline_medians):
    """Compare the current run against the baseline (released) run.

    Returns ``(regressions, missing, added)`` where ``regressions`` is a list of
    ``(name, baseline_s, current_s, ratio, threshold)`` for paths slower than
    their per-benchmark threshold, ``missing`` are baseline names absent from the
    current run, and ``added`` are current names absent from the baseline run.

    ``added`` is the graceful path for benchmarks that exercise an API newer than
    the published release (the released suite errors on them, so they never reach
    its JSON): they are reported as new/unbaselined and are **not** gated.
    """
    regressions = []
    for name, base_s in baseline_medians.items():
        if name not in current_medians:
            continue
        cur_s = current_medians[name]
        ratio = cur_s / base_s if base_s > 0 else float("inf")
        threshold = threshold_for(name)
        if ratio > threshold:
            regressions.append((name, base_s, cur_s, ratio, threshold))
    missing = sorted(set(baseline_medians) - set(current_medians))
    added = sorted(set(current_medians) - set(baseline_medians))
    return regressions, missing, added


# II Main Functions
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_run",
                        help="pytest-benchmark JSON for the current working tree")
    parser.add_argument("--baseline-run", required=True,
                        help="pytest-benchmark JSON for the latest stable release "
                             "(benchmarked on the same runner in the same job)")
    parser.add_argument("--released-version", default=None,
                        help="version of the installed release baseline (e.g. 1.0.3); "
                             "enables the gated_since guard against silent coverage loss")
    args = parser.parse_args(argv)

    current_path = Path(args.current_run)
    baseline_path = Path(args.baseline_run)
    for label, path in (("current", current_path), ("baseline", baseline_path)):
        if not path.is_file():
            print(f"ERROR: {label} benchmark run not found: {path} "
                  "(run pytest with --benchmark-json=FILE first)")
            return 2

    current_medians = read_run_medians(current_path)
    baseline_medians = read_run_medians(baseline_path)
    gated_since = read_gated_since(current_path)
    rel_ver = version_tuple(args.released_version) if args.released_version else None

    regressions, missing, added = compare(current_medians, baseline_medians)

    gated = sorted(set(current_medians) & set(baseline_medians))
    print(f"[perf] current vs released A/B "
          f"(thresholds: {BIG_PATH_THRESHOLD}x meaty / {SUBMS_THRESHOLD}x sub-2ms)")
    for name in gated:
        cur_s, base_s = current_medians[name], baseline_medians[name]
        ratio = cur_s / base_s if base_s > 0 else float("inf")
        flag = "  REGRESSION" if ratio > threshold_for(name) else ""
        print(f"  {name:<40s} {cur_s*1e3:8.2f} ms  "
              f"({ratio:4.2f}x released, gate {threshold_for(name):.1f}x){flag}")

    # Classify the current-only benchmarks (absent from the released run) using
    # each benchmark's declared gated_since. With a known released version we can
    # tell "pending-gate" (method newer than the release; OK) apart from
    # "lost coverage" (method should be in the release but isn't — renamed/removed,
    # so the gate silently stopped protecting it -> FAIL).
    pending, lost, unknown = [], [], []
    for name in added:
        gs = gated_since.get(name)
        if rel_ver is None or gs is None:
            unknown.append(name)
        elif version_tuple(gs) <= rel_ver:
            lost.append((name, gs))
        else:
            pending.append((name, gs))

    # Coverage is loud on purpose: the baseline is the latest PyPI RELEASE, so any
    # benchmark whose method post-dates that release is measured-but-NOT-gated
    # (it auto-gates when its release ships). Surface it so a green run is never
    # mistaken for "every hot path is protected".
    total = len(set(current_medians) | set(baseline_medians))
    print(f"[perf] COVERAGE: {len(gated)}/{total} benchmark(s) gated vs the release baseline"
          + (f" ({args.released_version})" if args.released_version else "") + ".")
    if pending:
        print("[perf] PENDING-GATE — measured only, will gate when their release ships: "
              + ", ".join(f"{n} (since {gs})" for n, gs in pending))
    if unknown:
        print(f"[perf] NOT GATED — absent from the release (no gated_since declared): "
              f"{', '.join(unknown)}")
    if missing:
        print(f"[perf] released benchmarks not in the current run: {', '.join(missing)}")

    rc = 0
    if lost:
        print("FAIL: coverage lost — benchmark(s) whose method is in the release "
              f"({args.released_version}) but absent from the released run "
              "(renamed/removed? the gate silently stopped protecting them): "
              + ", ".join(f"{n} (gated_since {gs})" for n, gs in lost))
        rc = 1
    if regressions:
        worst = ", ".join(f"{n} {r:.2f}x" for n, _, _, r, _ in
                          sorted(regressions, key=lambda x: -x[3]))
        print(f"FAIL: {len(regressions)} path(s) slower than threshold: {worst}")
        rc = 1
    if rc == 0:
        print("OK: no covered path regressed; no coverage lost.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
