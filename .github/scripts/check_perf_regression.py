"""Perf-regression guard for the committed benchmark suite (issue #187).

Mirrors ``.github/scripts/check_branch_coverage.py``: a small, dependency-free
CI helper that compares a fresh ``pytest-benchmark`` run against a committed
baseline and exits non-zero when any covered hot path slows down beyond a
generous threshold (default ``1.5x``, to absorb shared-runner wall-clock noise).

The benchmark suite (``tests/benchmarks/``) is opt-in (the ``[bench]`` install
extra) and runs only in the perf nightly (``.github/workflows/perf_nightly.yml``),
never in the blocking matrix — wall-clock is too noisy to gate merges on. This
helper turns the nightly's raw timings into a clear pass/fail report.

It lives under ``.github/scripts/`` (not ``dev_scripts/``, which is git-ignored)
so the CI checkout can run it. Like the coverage gate it prints to stdout for the
CI log; the package itself never calls ``print`` (it uses ``ut.print_out``).

Baseline format (slim, hand-refreshable JSON)::

    {
      "threshold": 1.5,
      "benchmarks": {"test_cpp_run": 0.0123, ...}   # median seconds per benchmark
    }

Compare a run against the committed baseline::

    pytest tests/benchmarks --benchmark-json=perf_run.json -c tests/pytest.ini
    python .github/scripts/check_perf_regression.py perf_run.json

Refresh the baseline from a run (do this on the CI runner class — the perf
nightly's ``workflow_dispatch`` on ubuntu-latest — then commit the file)::

    python .github/scripts/check_perf_regression.py perf_run.json --update
"""
import argparse
import json
import sys
from pathlib import Path

# Committed default. A covered path is flagged once its median exceeds
# baseline * THRESHOLD. Generous on purpose: shared CI runners are noisy, and
# this guard is a non-gating nightly signal, not a merge gate.
DEFAULT_THRESHOLD = 1.5

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / "tests" / "benchmarks" / "perf_baseline.json"


# I Helper Functions
def read_run_medians(run_json_path):
    """Return ``{benchmark_name: median_seconds}`` from a pytest-benchmark JSON.

    ``pytest-benchmark --benchmark-json=FILE`` writes a top-level ``benchmarks``
    list; each entry carries a ``name`` and a ``stats`` dict with a ``median``
    (seconds). We key on the short ``name`` (e.g. ``test_cpp_run``) so the
    baseline stays readable and machine-path-independent.
    """
    data = json.loads(Path(run_json_path).read_text())
    medians = {}
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        medians[name] = float(bench["stats"]["median"])
    return medians


def read_baseline(baseline_path):
    """Return ``(threshold, {name: median_seconds})`` from a slim baseline JSON."""
    data = json.loads(Path(baseline_path).read_text())
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    benchmarks = {k: float(v) for k, v in data.get("benchmarks", {}).items()}
    return threshold, benchmarks


def compare(run_medians, baseline_medians, threshold):
    """Compare run vs baseline medians.

    Returns ``(regressions, missing, added)`` where ``regressions`` is a list of
    ``(name, baseline_s, run_s, ratio)`` for paths slower than
    ``baseline * threshold``, ``missing`` are baseline names absent from the run,
    and ``added`` are run names absent from the baseline (new, unbaselined).
    """
    regressions = []
    for name, base_s in baseline_medians.items():
        if name not in run_medians:
            continue
        run_s = run_medians[name]
        ratio = run_s / base_s if base_s > 0 else float("inf")
        if ratio > threshold:
            regressions.append((name, base_s, run_s, ratio))
    missing = sorted(set(baseline_medians) - set(run_medians))
    added = sorted(set(run_medians) - set(baseline_medians))
    return regressions, missing, added


def write_baseline(run_medians, baseline_path, threshold):
    """Write a slim baseline JSON from a run's medians (the ``--update`` path)."""
    payload = {
        "threshold": threshold,
        "benchmarks": {name: round(run_medians[name], 6)
                       for name in sorted(run_medians)},
    }
    Path(baseline_path).write_text(json.dumps(payload, indent=2) + "\n")


# II Main Functions
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json",
                        help="pytest-benchmark JSON (--benchmark-json output)")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                        help="committed baseline JSON (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="slowdown ratio to flag; overrides the baseline's "
                             f"value (committed default {DEFAULT_THRESHOLD})")
    parser.add_argument("--update", action="store_true",
                        help="refresh the baseline from run_json instead of "
                             "comparing (commit the result on the CI runner class)")
    args = parser.parse_args(argv)

    run_path = Path(args.run_json)
    if not run_path.is_file():
        print(f"ERROR: benchmark run not found: {run_path} "
              "(run pytest with --benchmark-json=FILE first)")
        return 2
    run_medians = read_run_medians(run_path)

    if args.update:
        threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
        if Path(args.baseline).is_file():
            threshold = read_baseline(args.baseline)[0] if args.threshold is None \
                else args.threshold
        write_baseline(run_medians, args.baseline, threshold)
        print(f"[perf] wrote baseline {args.baseline} "
              f"({len(run_medians)} benchmarks, threshold {threshold}x)")
        return 0

    if not Path(args.baseline).is_file():
        print(f"ERROR: baseline not found: {args.baseline} "
              "(create it with --update on the CI runner class)")
        return 2
    base_threshold, baseline_medians = read_baseline(args.baseline)
    threshold = args.threshold if args.threshold is not None else base_threshold

    regressions, missing, added = compare(run_medians, baseline_medians, threshold)

    print(f"[perf] {len(run_medians)} benchmarks vs baseline "
          f"(threshold {threshold}x)")
    for name in sorted(set(run_medians) & set(baseline_medians)):
        run_s, base_s = run_medians[name], baseline_medians[name]
        ratio = run_s / base_s if base_s > 0 else float("inf")
        flag = "  REGRESSION" if ratio > threshold else ""
        print(f"  {name:<40s} {run_s*1e3:8.2f} ms  "
              f"({ratio:4.2f}x baseline){flag}")
    if added:
        print(f"[perf] new (unbaselined) benchmarks: {', '.join(added)} "
              "— refresh the baseline with --update to track them")
    if missing:
        print(f"[perf] baseline benchmarks not in this run: {', '.join(missing)}")

    if regressions:
        worst = ", ".join(f"{n} {r:.2f}x" for n, _, _, r in
                          sorted(regressions, key=lambda x: -x[3]))
        print(f"FAIL: {len(regressions)} path(s) slower than {threshold}x: {worst}")
        return 1
    print("OK: no covered path regressed beyond threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
