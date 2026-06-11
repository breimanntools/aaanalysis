"""
This is a script for the CI branch- and line-coverage gate (issue #84).

It parses the cobertura ``coverage.xml`` produced by
``pytest --cov-branch --cov-report=xml`` and exits non-zero if either the
line-rate or the branch-rate falls below its committed gate. It runs as a step
*after* the coverage pytest invocation in
``.github/workflows/test_coverage.yml`` because ``coverage.xml`` is only written
at the end of the pytest session. It lives under ``.github/scripts/`` (not
``dev_scripts/``, which is git-ignored) so the CI checkout can run it.

Branch coverage cannot be gated with ``pytest --cov-fail-under`` directly:
once ``--cov-branch`` is on, that flag checks the *combined* line+branch number,
which would silently re-define the line gate. Parsing the two rates separately
keeps an honest, independent line floor and branch floor (ADR-0016 D4).

This is CI tooling, not library code, so it prints to stdout for the CI log
(the package itself never calls ``print`` — it uses ``ut.print_out``).

Local use::

    pytest tests -m "not regression" --cov=aaanalysis --cov-branch \\
        --cov-report=xml -n auto -c tests/pytest.ini
    python .github/scripts/check_branch_coverage.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Committed gates in percent. Ratcheted at-or-just-below the measured baseline
# (ADR-0016); raise only once the suite clears the next step. The line gate
# mirrors the historical --cov-fail-under=88.
LINE_GATE = 88.0
BRANCH_GATE = 80.0


# I Helper Functions
def read_rates(xml_path):
    """Return ``(line_rate_pct, branch_rate_pct)`` from a cobertura ``coverage.xml``.

    The root ``<coverage>`` element carries ``line-rate`` and ``branch-rate`` as
    floats in ``[0, 1]``; ``branch-rate`` is ``0`` unless the run used
    ``--cov-branch``.
    """
    root = ET.parse(str(xml_path)).getroot()
    line_rate = float(root.attrib["line-rate"]) * 100.0
    branch_rate = float(root.attrib["branch-rate"]) * 100.0
    return line_rate, branch_rate


# II Main Functions
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    xml_path = Path(argv[0]) if argv else Path("coverage.xml")
    if not xml_path.is_file():
        print(f"ERROR: coverage report not found: {xml_path} "
              "(run pytest with --cov-branch --cov-report=xml first)")
        return 2
    line_rate, branch_rate = read_rates(xml_path)
    print(f"[coverage] line {line_rate:.2f}% (gate {LINE_GATE:.0f}%) | "
          f"branch {branch_rate:.2f}% (gate {BRANCH_GATE:.0f}%)")
    failures = []
    if line_rate < LINE_GATE:
        failures.append(f"line coverage {line_rate:.2f}% < gate {LINE_GATE:.0f}%")
    if branch_rate < BRANCH_GATE:
        failures.append(f"branch coverage {branch_rate:.2f}% < gate {BRANCH_GATE:.0f}%")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("OK: line and branch coverage meet their gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
