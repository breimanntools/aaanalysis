"""Tests for the perf-regression guard (``.github/scripts/check_perf_regression.py``).

These exercise the comparison *logic* with synthetic data, so they run in the
normal unit matrix without ``pytest-benchmark`` installed. The benchmark suite
itself (``tests/benchmarks/``) is the opt-in A/B job; here we pin the guard that
turns its timings into a pass/fail — in particular the issue-#187 KPI that a
deliberate ~2x slowdown on a hot path is flagged, and that the looser sub-2ms
band does not flag jitter-sized changes on the micro-benchmarks.
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


# I compare() — the core A/B regression logic (per-benchmark thresholds)
def test_no_regression_when_equal(mod):
    base = {"test_cpp_run": 0.010, "test_aaclust_fit": 0.0008}
    run = {"test_cpp_run": 0.010, "test_aaclust_fit": 0.0008}
    regressions, missing, added = mod.compare(run, base)
    assert regressions == []
    assert missing == [] and added == []


def test_two_x_slowdown_on_meaty_path_is_flagged(mod):
    """KPI: a ~2x slowdown on a covered meaty path is flagged (1.3x gate)."""
    base = {"test_cpp_run": 0.010}
    run = {"test_cpp_run": 0.020}  # 2.0x
    regressions, _, _ = mod.compare(run, base)
    assert len(regressions) == 1
    name, base_s, run_s, ratio, threshold = regressions[0]
    assert name == "test_cpp_run"
    assert ratio == pytest.approx(2.0)
    assert threshold == mod.BIG_PATH_THRESHOLD


def test_meaty_path_within_band_not_flagged(mod):
    base = {"test_cpp_run": 0.010}
    run = {"test_cpp_run": 0.012}  # 1.2x < 1.3x
    regressions, _, _ = mod.compare(run, base)
    assert regressions == []


def test_subms_path_uses_looser_band(mod):
    """A sub-2ms micro-benchmark gets the 2.0x band, not 1.3x."""
    base = {"test_aaclust_fit": 0.0008}
    # 1.5x would trip the meaty gate but is under the sub-2ms 2.0x band -> ok
    run_ok = {"test_aaclust_fit": 0.0012}
    assert mod.compare(run_ok, base)[0] == []
    # 2.5x exceeds even the loose band -> flagged
    run_bad = {"test_aaclust_fit": 0.0020}
    regressions, _, _ = mod.compare(run_bad, base)
    assert len(regressions) == 1
    assert regressions[0][4] == mod.SUBMS_THRESHOLD


def test_missing_and_added_reported(mod):
    base = {"test_cpp_run": 0.010, "gone": 0.030}
    run = {"test_cpp_run": 0.010, "fresh": 0.040}
    regressions, missing, added = mod.compare(run, base)
    assert regressions == []
    assert missing == ["gone"]
    assert added == ["fresh"]


def test_added_only_benchmark_is_not_gated(mod):
    """An API newer than the release errors out of the released run -> not gated."""
    base = {"test_cpp_run": 0.010}
    run = {"test_cpp_run": 0.010, "test_encode_pdb": 0.5}  # huge but unbaselined
    regressions, _, added = mod.compare(run, base)
    assert regressions == []
    assert added == ["test_encode_pdb"]


def test_zero_baseline_is_infinite_ratio(mod):
    base = {"test_cpp_run": 0.0}
    run = {"test_cpp_run": 0.010}
    regressions, _, _ = mod.compare(run, base)
    assert len(regressions) == 1
    assert regressions[0][3] == float("inf")


# II the pytest-benchmark JSON shape
def test_read_run_medians_parses_benchmark_json(mod, tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.0123}},
        {"name": "test_y", "stats": {"median": 0.4560}},
    ]}))
    medians = mod.read_run_medians(run_json)
    assert medians == {"test_x": 0.0123, "test_y": 0.4560}


def test_read_gated_since_from_extra_info(mod, tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"benchmarks": [
        {"name": "test_x", "stats": {"median": 0.01},
         "extra_info": {"gated_since": "1.1.0"}},
        {"name": "test_y", "stats": {"median": 0.02}},  # no extra_info -> omitted
    ]}))
    assert mod.read_gated_since(run_json) == {"test_x": "1.1.0"}


def test_version_tuple_parses_and_orders(mod):
    assert mod.version_tuple("1.0.3") == (1, 0, 3)
    assert mod.version_tuple("1.1.0rc1") == (1, 1, 0)
    assert mod.version_tuple("") == ()
    assert mod.version_tuple("1.0.0") <= mod.version_tuple("1.0.3")
    assert mod.version_tuple("1.1.0") > mod.version_tuple("1.0.3")


# III main() end-to-end exit codes (current vs released A/B)
def _write_run(path, medians):
    path.write_text(json.dumps({"benchmarks": [
        {"name": n, "stats": {"median": m}} for n, m in medians.items()]}))


def test_main_flags_regression_nonzero(mod, tmp_path):
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run(released, {"test_cpp_run": 0.010})
    _write_run(current, {"test_cpp_run": 0.025})  # 2.5x on a meaty path
    rc = mod.main([str(current), "--baseline-run", str(released)])
    assert rc == 1


def test_main_passes_when_fast(mod, tmp_path):
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run(released, {"test_cpp_run": 0.010})
    _write_run(current, {"test_cpp_run": 0.009})  # faster
    rc = mod.main([str(current), "--baseline-run", str(released)])
    assert rc == 0


def test_main_missing_run_file_returns_2(mod, tmp_path):
    rc = mod.main([str(tmp_path / "nope.json"),
                   "--baseline-run", str(tmp_path / "released.json")])
    assert rc == 2


# IV gated_since guard (pending-gate vs silent coverage loss)
def _write_run_ei(path, medians_since):
    """medians_since: {name: (median_s, gated_since_or_None)}."""
    benches = []
    for name, (med, since) in medians_since.items():
        b = {"name": name, "stats": {"median": med}}
        if since is not None:
            b["extra_info"] = {"gated_since": since}
        benches.append(b)
    path.write_text(json.dumps({"benchmarks": benches}))


def test_main_pending_gate_does_not_fail(mod, tmp_path):
    """A current-only benchmark newer than the release is 'pending-gate', not a fail."""
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run(released, {"test_cpp_run": 0.010})
    _write_run_ei(current, {"test_cpp_run": (0.010, "1.0.0"),
                            "test_new_method": (5.0, "1.1.0")})  # absent from release
    rc = mod.main([str(current), "--baseline-run", str(released),
                   "--released-version", "1.0.3"])
    assert rc == 0


def test_main_lost_coverage_fails(mod, tmp_path):
    """A benchmark whose method IS in the release but missing from the released run
    (renamed/removed) is silent coverage loss -> the guard fails."""
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run(released, {"test_cpp_run": 0.010})
    _write_run_ei(current, {"test_cpp_run": (0.010, "1.0.0"),
                            "test_old_method": (5.0, "1.0.0")})  # should be gated, isn't
    rc = mod.main([str(current), "--baseline-run", str(released),
                   "--released-version", "1.0.3"])
    assert rc == 1


def test_main_no_released_version_disables_guard(mod, tmp_path):
    """Without --released-version the gated_since guard is inactive (no fail)."""
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run(released, {"test_cpp_run": 0.010})
    _write_run_ei(current, {"test_cpp_run": (0.010, "1.0.0"),
                            "test_old_method": (5.0, "1.0.0")})
    rc = mod.main([str(current), "--baseline-run", str(released)])
    assert rc == 0


# V output A/B (byte-exact current-vs-released)
def _write_run_dig(path, medians_digests):
    """medians_digests: {name: (median_s, output_digest_or_None)}."""
    benches = []
    for name, (med, dig) in medians_digests.items():
        b = {"name": name, "stats": {"median": med}}
        if dig is not None:
            b["extra_info"] = {"output_digest": dig}
        benches.append(b)
    path.write_text(json.dumps({"benchmarks": benches}))


def test_read_output_digests(mod, tmp_path):
    run = tmp_path / "run.json"
    _write_run_dig(run, {"test_a": (0.01, "abc"), "test_b": (0.02, None)})
    assert mod.read_output_digests(run) == {"test_a": "abc"}


def test_main_output_identical_passes(mod, tmp_path):
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run_dig(released, {"test_cpp_run": (0.010, "samehash")})
    _write_run_dig(current, {"test_cpp_run": (0.010, "samehash")})
    assert mod.main([str(current), "--baseline-run", str(released)]) == 0


def test_main_output_drift_fails(mod, tmp_path):
    """Same method, byte-different output vs the release -> fail (review flag)."""
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run_dig(released, {"test_cpp_run": (0.010, "releasehash")})
    _write_run_dig(current, {"test_cpp_run": (0.010, "currenthash")})
    assert mod.main([str(current), "--baseline-run", str(released)]) == 1


def test_main_output_no_digest_not_compared(mod, tmp_path):
    """A model-returning method carries no digest -> not output-compared."""
    released = tmp_path / "released.json"
    current = tmp_path / "current.json"
    _write_run_dig(released, {"test_aaclust_fit": (0.001, None)})
    _write_run_dig(current, {"test_aaclust_fit": (0.001, None)})
    assert mod.main([str(current), "--baseline-run", str(released)]) == 0
