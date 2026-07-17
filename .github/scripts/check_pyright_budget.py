"""
This is a script for the pyright diagnostics no-regression ratchet.

AAanalysis ships ``py.typed`` and runs pyright **public-API first**
(``pyrightconfig.json`` excludes ``aaanalysis/**/_backend``). The type contract
is being deepened by driving the diagnostic count down in small, per-subpackage
steps. This script records that progress as a committed high-water mark
(``.github/pyright_baseline.txt``) and reports each run against it, so every PR
shows a strict, non-increasing total.

It is a **no-regression gate**, not a clean/strict gate: the ``Type Check
(ratchet)`` workflow runs it *without* ``continue-on-error``, so the job fails
whenever the count rises ABOVE the committed budget (new type errors are
blocked) but passes at or below it (the existing backlog is allowed to merge).
Both CI and local runs exit non-zero on a *regression* (count above the
committed budget) and zero at or below it; an improvement prints the new,
lower number to commit. The pyright version is pinned in the workflow so the
committed budget is reproducible.

This is CI tooling, not library code, so it prints to stdout for the CI log
(the package itself never calls ``print`` — it uses ``ut.print_out``).

Local use::

    python .github/scripts/check_pyright_budget.py            # runs pyright itself
    pyright --outputjson > pr.json
    python .github/scripts/check_pyright_budget.py pr.json     # parse an existing report

When the count drops, lower the integer in ``.github/pyright_baseline.txt`` to
the new value in the same PR — that is what keeps the ratchet honest.
"""
import sys
import json
import subprocess
from collections import Counter
from pathlib import Path

# The committed high-water mark. Lower it (never raise it) as each burn-down PR
# clears diagnostics. Home: ``.github/pyright_baseline.txt`` next to this script.
BASELINE_PATH = Path(__file__).resolve().parents[1] / "pyright_baseline.txt"


# I Helper Functions
def read_budget(path=BASELINE_PATH):
    """Return the committed diagnostic budget (int) from ``pyright_baseline.txt``."""
    return int(path.read_text(encoding="utf-8").strip())


def load_report(argv):
    """Return pyright's parsed JSON report.

    Parse ``argv[0]`` as a pre-generated ``pyright --outputjson`` file when
    given; otherwise invoke pyright directly so the script is self-contained.
    """
    if argv:
        return json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    proc = subprocess.run(["pyright", "--outputjson"], capture_output=True, text=True)
    # pyright exits non-zero whenever diagnostics remain; that is expected here,
    # so we read stdout regardless of the return code.
    if not proc.stdout.strip():
        raise RuntimeError(f"pyright produced no JSON output:\n{proc.stderr}")
    return json.loads(proc.stdout)


def count_diagnostics(report):
    """Return ``(total, Counter(rule -> n))`` for the report's diagnostics."""
    diags = report.get("generalDiagnostics", [])
    by_rule = Counter(d.get("rule", "<no-rule>") for d in diags)
    return len(diags), by_rule


# II Main Functions
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    budget = read_budget()
    total, by_rule = count_diagnostics(load_report(argv))
    print(f"[pyright] {total} diagnostics (committed budget {budget})")
    for rule, n in by_rule.most_common():
        print(f"    {n:5d}  {rule}")
    if total > budget:
        print(f"REGRESSION: {total} > budget {budget} "
              f"(+{total - budget}). New diagnostics were introduced.")
        return 1
    if total < budget:
        print(f"IMPROVED: {total} < budget {budget} "
              f"(-{budget - total}). Lower .github/pyright_baseline.txt to {total}.")
        return 0
    print("OK: at budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
