"""
This is a script for the documentation build-health gate.

Nothing in the pipeline used to fail when a page stopped rendering: no workflow
built the docs, Read the Docs sets no ``fail_on_warning``, and **Sphinx exits 0
even with ``ERROR`` lines in its log**. Three broken references reached
``master`` through that gap. This script closes it by parsing the build log and
gating on the message level, never on Sphinx's return code.

Two signals, deliberately gated differently:

* ``ERROR`` / ``SEVERE`` -- a **plain zero gate**. The count was taken to zero,
  so there is no baseline file to maintain and no backlog to tolerate. One new
  broken reference fails the build.
* ``CRITICAL`` -- a **no-regression ratchet** against
  ``.github/docs_critical_baseline.txt``, the same pattern as the pyright
  budget. These are the pre-existing ``Unexpected section title`` lines in the
  *generated* example pages; they are a known backlog, so they may merge at or
  below the committed mark but never above it. **The mark is measured in CI,
  never from a local build** -- a local build differs (an offline runner cannot
  reach the intersphinx inventories, and a base install generates a different
  page set than ``[docs]``, which pulls ``[pro]``). When the baseline file is
  absent the ratchet reports and does not gate, so adopting the gate can never
  turn a green pull request red.

``WARNING`` is counted and printed but never gated: most of the ~340 are one
pre-existing ``autosummary.import_cycle`` pattern, so a warning ratchet would be
noise until that pattern is addressed. ``-W`` is not usable here for the same
reason.

The toolchain needs no pinning in the workflow: ``sphinx`` and ``docutils`` are
pinned in the ``docs`` extra in ``pyproject.toml``, so ``pip install .[docs]``
reproduces the counts.

This is CI tooling, not library code, so it prints to stdout for the CI log
(the package itself never calls ``print`` -- it uses ``ut.print_out``).

Local use::

    python .github/scripts/check_docs_build.py               # full build, then check
    python .github/scripts/check_docs_build.py build.log     # parse an existing log
    python .github/scripts/check_docs_build.py --report      # print counts, never fail
"""
import re
import sys
import shutil
import argparse
import tempfile
import subprocess
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "docs" / "source"
# The committed high-water mark for CRITICAL lines. Lower it (never raise it) as
# the generated-page backlog clears. Home: next to this script's siblings.
BASELINE_PATH = REPO_ROOT / ".github" / "docs_critical_baseline.txt"

# Sphinx/docutils emit ``<path>:<line>: LEVEL: message``, or a bare
# ``LEVEL: message`` for build-wide messages. Anchoring at the start of the line
# matters: committed notebook output is echoed into the log verbatim, and a cell
# that printed "ERROR: ..." mid-line must not be counted as a build error.
RE_MESSAGE = re.compile(
    r"^(?:.*?:\d*:\s*)?(?P<level>CRITICAL|SEVERE|ERROR|WARNING):\s",
)
# Gated at zero; CRITICAL rides the ratchet instead, and WARNING is advisory.
FAIL_LEVELS = ("ERROR", "SEVERE")


# I Helper Functions
def read_baseline(path=BASELINE_PATH):
    """Return the committed CRITICAL high-water mark, or None when unset.

    A missing file means the ratchet has not been seeded from a CI run yet: the
    count is reported and nothing is gated on it.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text else None


def build_docs(source_dir=SOURCE_DIR, out_dir=None):
    """Build the HTML docs from scratch and return the combined build log.

    The build always goes into a FRESH directory, because the counts are only
    meaningful for a full build. Sphinx is incremental: a second build into the
    same output directory re-reads only the changed pages, so every message from
    an untouched page is missing from the log. Locally that silently undercounts
    -- a re-run reported 89 critical against a true 167 and invited the baseline
    to be lowered to a number CI would never reproduce. CI always builds a fresh
    checkout, so this keeps a local run honest and agreeing with it.

    Sphinx writes its messages to stderr and returns 0 with errors present, so
    both streams are captured and the return code is deliberately ignored.
    """
    tmp_dir = None
    if out_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="aaanalysis-docs-gate-")
        out_dir = tmp_dir
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", str(source_dir), str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if tmp_dir is not None:
        # Always discard it: the log is the deliverable, and a kept tree would make the
        # NEXT run incremental, which is the very thing this function exists to avoid.
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if not log.strip():
        raise RuntimeError(f"sphinx produced no output (exit {proc.returncode})")
    return log


def count_levels(log):
    """Return ``Counter(level -> n)`` over the build log's messages."""
    counts = Counter()
    for line in log.splitlines():
        match = RE_MESSAGE.match(line)
        if match:
            counts[match.group("level")] += 1
    return counts


def collect_failures(log, levels=FAIL_LEVELS):
    """Return the log lines whose level is one of ``levels``."""
    out = []
    for line in log.splitlines():
        match = RE_MESSAGE.match(line)
        if match and match.group("level") in levels:
            out.append(line.rstrip())
    return out


# II Main Functions
def evaluate(log, baseline=None):
    """Return ``(exit_code, report_lines)`` for one build log.

    ``exit_code`` is 1 when an ERROR/SEVERE line is present, or when the
    CRITICAL count rises above ``baseline``; 0 otherwise.
    """
    counts = count_levels(log)
    n_fail = sum(counts[level] for level in FAIL_LEVELS)
    n_critical = counts["CRITICAL"]
    n_warning = counts["WARNING"]
    lines = [f"[docs] {n_fail} error (ERROR/SEVERE) · {n_critical} critical · "
             f"{n_warning} warning"]
    code = 0
    if n_fail:
        lines.append(f"FAILED: {n_fail} docutils error(s); the gate requires zero.")
        lines.extend(f"    {line}" for line in collect_failures(log))
        code = 1
    else:
        lines.append("OK: zero docutils errors.")
    if baseline is None:
        lines.append(f"CRITICAL ratchet not seeded (advisory). Commit {n_critical} "
                     f"to {BASELINE_PATH.relative_to(REPO_ROOT)} from a CI run to arm it.")
    elif n_critical > baseline:
        lines.append(f"REGRESSION: {n_critical} critical > baseline {baseline} "
                     f"(+{n_critical - baseline}).")
        code = 1
    elif n_critical < baseline:
        lines.append(f"IMPROVED: {n_critical} critical < baseline {baseline} "
                     f"(-{baseline - n_critical}). Lower the baseline to the count a CI "
                     f"run reports, not to a local one.")
    else:
        lines.append(f"OK: critical at baseline ({baseline}).")
    return code, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("log", nargs="?",
                        help="a pre-generated build log; omit to build the docs")
    parser.add_argument("--report", action="store_true",
                        help="print the counts and always exit 0")
    args = parser.parse_args(argv)
    log = Path(args.log).read_text(encoding="utf-8") if args.log else build_docs()
    code, lines = evaluate(log, baseline=read_baseline())
    for line in lines:
        print(line)
    return 0 if args.report else code


if __name__ == "__main__":
    sys.exit(main())
