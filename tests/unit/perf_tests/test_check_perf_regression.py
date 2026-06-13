"""Tests for the perf-regression guard (``.github/scripts/check_perf_regression.py``).

These exercise the comparison *logic* with synthetic data, so they run in the
normal unit matrix without ``pytest-benchmark`` installed. The benchmark suite
itself (``tests/benchmarks/``) is the opt-in nightly; here we pin the guard that
turns its timings into a pass/fail — in particular the issue-#187 KPI that a
deliberate ~2x slowdown is flagged.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_perf_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_perf_regression", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# I compare() — the core regression logic
def test_no_regression_when_equal(mod):
    base = {"a": 0.010, "b": 0.020}
    run = {"a": 0.010, "b": 0.020}
    regressions, missing, added = mod.compare(run, base, threshold=1.5)
    assert regressions == []
    assert missing == [] and added == []


def test_two_x_slowdown_is_flagged(mod):
    """KPI: a ~2x slowdown in a covered path is flagged (threshold 1.5x)."""
    base = {"hot": 0.010}
    run = {"hot": 0.020}  # 2.0x
    regressions, _, _ = mod.compare(run, base, threshold=1.5)
    assert len(regressions) == 1
    name, base_s, run_s, ratio = regressions[0]
    assert name == "hot"
    assert ratio == pytest.approx(2.0)


def test_within_threshold_not_flagged(mod):
    base = {"hot": 0.010}
    run = {"hot": 0.014}  # 1.4x < 1.5x
    regressions, _, _ = mod.compare(run, base, threshold=1.5)
    assert regressions == []


def test_missing_and_added_reported(mod):
    base = {"a": 0.010, "gone": 0.030}
    run = {"a": 0.010, "fresh": 0.040}
    regressions, missing, added = mod.compare(run, base, threshold=1.5)
    assert regressions == []
    assert missing == ["gone"]
    assert added == ["fresh"]


def test_zero_baseline_is_infinite_ratio(mod):
    base = {"hot": 0.0}
    run = {"hot": 0.010}
    regressions, _, _ = mod.compare(run, base, threshold=1.5)
    assert len(regressions) == 1
    assert regressions[0][3] == float("inf")


# II read/write round-trip and the pytest-benchmark JSON shape
def test_read_run_medians_parses_benchmark_json(mod, tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.0123}},
        {"name": "test_y", "stats": {"median": 0.4560}},
    ]}))
    medians = mod.read_run_medians(run_json)
    assert medians == {"test_x": 0.0123, "test_y": 0.4560}


def test_baseline_write_read_round_trip(mod, tmp_path):
    baseline = tmp_path / "perf_baseline.json"
    mod.write_baseline({"b": 0.002, "a": 0.001}, baseline, threshold=1.5)
    threshold, benchmarks = mod.read_baseline(baseline)
    assert threshold == 1.5
    assert benchmarks == {"a": 0.001, "b": 0.002}


# III main() end-to-end exit codes
def test_main_flags_regression_nonzero(mod, tmp_path):
    baseline = tmp_path / "perf_baseline.json"
    baseline.write_text(json.dumps({"threshold": 1.5,
                                    "benchmarks": {"test_x": 0.010}}))
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.025}},  # 2.5x
    ]}))
    rc = mod.main([str(run_json), "--baseline", str(baseline)])
    assert rc == 1


def test_main_passes_when_fast(mod, tmp_path):
    baseline = tmp_path / "perf_baseline.json"
    baseline.write_text(json.dumps({"threshold": 1.5,
                                    "benchmarks": {"test_x": 0.010}}))
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.009}},  # faster
    ]}))
    rc = mod.main([str(run_json), "--baseline", str(baseline)])
    assert rc == 0


def test_main_update_writes_baseline(mod, tmp_path):
    baseline = tmp_path / "perf_baseline.json"
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.011}},
    ]}))
    rc = mod.main([str(run_json), "--baseline", str(baseline), "--update"])
    assert rc == 0
    threshold, benchmarks = mod.read_baseline(baseline)
    assert benchmarks == {"test_x": 0.011}


def test_main_missing_run_file_returns_2(mod, tmp_path):
    rc = mod.main([str(tmp_path / "nope.json"),
                   "--baseline", str(tmp_path / "b.json")])
    assert rc == 2
