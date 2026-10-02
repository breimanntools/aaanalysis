"""Tests for the docs build-health gate (``.github/scripts/check_docs_build.py``).

These exercise the log *parsing and verdict* logic with fixtures, so no Sphinx
build runs in the unit matrix (the real build lives in the ``Docs Build (gate)``
workflow). The KPIs pinned here are the issue-#571 acceptance criteria: a build
log carrying a docutils ``ERROR`` fails the gate, and a clean log does not --
plus the trap that made a naive gate unusable, namely that committed notebook
output is echoed into the build log and must not be mistaken for a build error.

The gate is CI tooling rather than library code, so it is loaded from its script
path the same way the version-guard's tests load theirs.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_docs_build.py"

CLEAN_LOG = """\
Running Sphinx v8.1.3
loading intersphinx inventory from https://docs.python.org/3/objects.inv ...
building [html]: targets for 3 source files that are out of date
WARNING: Summarised items should not include the current module. [autosummary.import_cycle]
build succeeded, 1 warning.
"""

ERROR_LOG = CLEAN_LOG + (
    'source/index/api.rst:42: ERROR: Unknown target name: "cpp".\n'
)

CRITICAL_LOG = CLEAN_LOG + (
    "source/generated/examples/aal_get_conservation.rst:2: "
    "CRITICAL: Unexpected section title.\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("check_docs_build", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# I count_levels() -- the parser the whole gate rests on
class TestCountLevels:
    """Test the per-level message count in isolation."""

    def test_clean_log_has_no_errors(self, mod):
        counts = mod.count_levels(CLEAN_LOG)
        assert counts["ERROR"] == 0
        assert counts["SEVERE"] == 0

    def test_warning_is_counted(self, mod):
        assert mod.count_levels(CLEAN_LOG)["WARNING"] == 1

    def test_error_line_is_counted(self, mod):
        assert mod.count_levels(ERROR_LOG)["ERROR"] == 1

    def test_critical_is_counted_separately_from_error(self, mod):
        counts = mod.count_levels(CRITICAL_LOG)
        assert counts["CRITICAL"] == 1
        assert counts["ERROR"] == 0

    def test_bare_build_wide_message_is_counted(self, mod):
        """A message with no ``path:line:`` prefix still carries a level."""
        assert mod.count_levels("ERROR: config value is invalid\n")["ERROR"] == 1

    def test_notebook_output_is_not_a_build_error(self, mod):
        """KPI trap: committed cell output is echoed verbatim into the log.

        ``nbsphinx`` renders a code cell's stream output, so a notebook that
        printed a line containing ``ERROR:`` must not fail the docs build.
        """
        log = CLEAN_LOG + "    ERROR: model failed to converge (cell output)\n"
        assert mod.count_levels(log)["ERROR"] == 0

    def test_level_word_inside_a_message_is_not_counted(self, mod):
        log = CLEAN_LOG + "source/x.rst:1: WARNING: the word ERROR: appears here\n"
        counts = mod.count_levels(log)
        assert counts["ERROR"] == 0
        assert counts["WARNING"] == 2

    def test_empty_log_counts_nothing(self, mod):
        assert mod.count_levels("") == {}


# II collect_failures() -- what the CI log shows the author
class TestCollectFailures:
    """Test that a failing run names the offending lines."""

    def test_clean_log_collects_nothing(self, mod):
        assert mod.collect_failures(CLEAN_LOG) == []

    def test_error_line_is_reported_verbatim(self, mod):
        failures = mod.collect_failures(ERROR_LOG)
        assert len(failures) == 1
        assert failures[0].endswith('Unknown target name: "cpp".')

    def test_critical_is_not_a_failure_line(self, mod):
        """CRITICAL rides the ratchet, so it is never in the zero-gate list."""
        assert mod.collect_failures(CRITICAL_LOG) == []

    def test_severe_is_a_failure_line(self, mod):
        log = CLEAN_LOG + "source/x.rst:1: SEVERE: Unexpected section title.\n"
        assert len(mod.collect_failures(log)) == 1


# III evaluate() -- the exit code CI gates on
class TestEvaluate:
    """Test the verdict, which is what the workflow step returns."""

    def test_clean_log_passes(self, mod):
        code, _ = mod.evaluate(CLEAN_LOG)
        assert code == 0

    def test_error_fails_the_gate(self, mod):
        """KPI: a pull request introducing a docutils ERROR fails the check."""
        code, lines = mod.evaluate(ERROR_LOG)
        assert code == 1
        assert any("requires zero" in line for line in lines)

    def test_unseeded_ratchet_never_fails(self, mod):
        """KPI: adopting the gate turns no green pull request red."""
        code, lines = mod.evaluate(CRITICAL_LOG, baseline=None)
        assert code == 0
        assert any("not seeded" in line for line in lines)

    def test_critical_at_baseline_passes(self, mod):
        code, lines = mod.evaluate(CRITICAL_LOG, baseline=1)
        assert code == 0
        assert any("at baseline" in line for line in lines)

    def test_critical_above_baseline_fails(self, mod):
        code, lines = mod.evaluate(CRITICAL_LOG, baseline=0)
        assert code == 1
        assert any("REGRESSION" in line for line in lines)

    def test_critical_below_baseline_passes_and_asks_to_lower(self, mod):
        code, lines = mod.evaluate(CLEAN_LOG, baseline=5)
        assert code == 0
        assert any("IMPROVED" in line for line in lines)

    def test_improved_message_points_at_a_ci_run_not_a_local_count(self, mod):
        """A local build can undercount, so the message must not name a number to commit."""
        _, lines = mod.evaluate(CLEAN_LOG, baseline=5)
        improved = next(line for line in lines if "IMPROVED" in line)
        assert "CI" in improved
        assert "Lower the baseline to 0" not in improved

    def test_error_fails_even_when_critical_is_at_baseline(self, mod):
        code, _ = mod.evaluate(ERROR_LOG + CRITICAL_LOG, baseline=1)
        assert code == 1

    def test_counts_appear_in_the_first_report_line(self, mod):
        _, lines = mod.evaluate(CRITICAL_LOG, baseline=1)
        assert lines[0].startswith("[docs] 0 error")
        assert "1 critical" in lines[0]


# IV read_baseline() -- the committed high-water mark
class TestReadBaseline:
    """Test the ratchet's seed file."""

    def test_missing_file_is_unseeded(self, mod, tmp_path):
        assert mod.read_baseline(tmp_path / "absent.txt") is None

    def test_empty_file_is_unseeded(self, mod, tmp_path):
        path = tmp_path / "baseline.txt"
        path.write_text("", encoding="utf-8")
        assert mod.read_baseline(path) is None

    def test_value_is_read_as_int(self, mod, tmp_path):
        path = tmp_path / "baseline.txt"
        path.write_text("167\n", encoding="utf-8")
        assert mod.read_baseline(path) == 167


# V main() -- the CLI the workflow calls
class TestMain:
    """Test the entry point end to end on a log file."""

    def test_clean_log_exits_zero(self, mod, tmp_path, capsys):
        log = tmp_path / "build.log"
        log.write_text(CLEAN_LOG, encoding="utf-8")
        assert mod.main([str(log)]) == 0
        assert "zero docutils errors" in capsys.readouterr().out

    def test_error_log_exits_one(self, mod, tmp_path, capsys):
        log = tmp_path / "build.log"
        log.write_text(ERROR_LOG, encoding="utf-8")
        assert mod.main([str(log)]) == 1
        assert "FAILED" in capsys.readouterr().out

    def test_report_mode_never_fails(self, mod, tmp_path, capsys):
        log = tmp_path / "build.log"
        log.write_text(ERROR_LOG, encoding="utf-8")
        assert mod.main([str(log), "--report"]) == 0
        assert "FAILED" in capsys.readouterr().out
