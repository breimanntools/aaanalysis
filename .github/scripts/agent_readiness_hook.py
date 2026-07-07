#!/usr/bin/env python3
"""Stop-hook: run the fast agent-readiness checkers and surface any defect.

Wired from ``.claude/settings.json`` as a ``Stop`` hook. It closes the feedback
loop so a regression in the agent-facing invariants is caught at end-of-turn
instead of only in CI after a push:

  * subpackage front-doors + ``__all__`` in sync (``check_agentic_docs.py``)
  * decision-log hygiene (``check_adrs.py``)
  * internal dataflow-map coverage (``check_module_map.py``)

All three are AST/text-based, import-free, and sub-second. This is deliberately
*not* the deferred pre-commit / ruff / mypy migration — it wires the checkers the
repo already owns into the agent's edit loop, nothing more.

Trigger is **path-scoped**: each checker runs only when a file it actually reads
has changed in the working tree, so editing tests, notebooks, CI, or prose docs
never fires it. Mapping (relative to the repo root):

  * ``aaanalysis/**``        -> front-doors, ADR-in-code scan, module-map coverage
  * ``docs/adr/**``          -> ADR hygiene
  * ``docs/module_map.md``   -> module-map coverage

Behaviour:
  * Silent, exit 0, when no relevant path changed OR every relevant checker passes.
  * On a real defect: print a concise summary to stderr and exit 2 so Claude Code
    feeds it back to the model to self-correct — but at most once per chain
    (the ``stop_hook_active`` guard prevents an infinite loop).
  * Any internal error is swallowed (exit 0): a hook must never wedge a session.
"""
import json
import os
import subprocess
import sys

# checker exit codes: 0 = clean, 1 = defects, 2 = usage error (ignore).
_DEFECT = 1


def _run(cmd, cwd, timeout):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def _changed_paths(root):
    """Repo-relative paths with any working-tree change (staged, unstaged, untracked)."""
    status = _run(["git", "status", "--porcelain"], root, 10)
    if status is None:
        return set()
    paths = set()
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        entry = line[3:] if len(line) > 3 else line.strip()  # drop the 2-char status + space
        # renames/copies: "old -> new" — both endpoints are relevant.
        for part in entry.split(" -> "):
            part = part.strip().strip('"')
            if part:
                paths.add(part)
    return paths


def _main():
    # 1) Read the hook payload; never block twice in a row (loop guard).
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    if payload.get("stop_hook_active"):
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

    # 2) Decide which checkers are in scope from what changed (skip on chat turns).
    changed = _changed_paths(root)
    if not changed:
        return 0
    pkg_changed = any(p.startswith("aaanalysis/") for p in changed)
    adr_changed = any(p.startswith("docs/adr/") for p in changed)
    map_changed = "docs/module_map.md" in changed

    sk = os.path.join(root, ".claude", "skills", "agent-readiness-audit", "scripts")
    checks = []
    if pkg_changed:
        checks.append(("front-doors / __all__",
                       [sys.executable, os.path.join(sk, "check_agentic_docs.py"), "aaanalysis"]))
    if pkg_changed or adr_changed:
        checks.append(("ADR hygiene", [sys.executable, os.path.join(sk, "check_adrs.py")]))
    if pkg_changed or map_changed:
        checks.append(("module map", [sys.executable, os.path.join(sk, "check_module_map.py")]))
    if not checks:
        return 0

    # 3) Run the in-scope checkers.
    failures = []
    for label, cmd in checks:
        r = _run(cmd, root, 60)
        if r is not None and r.returncode == _DEFECT:
            failures.append((label, r.stdout))

    if not failures:
        return 0

    # 4) Report defects back to the model to self-correct.
    lines = ["Agent-readiness regression — fix before finishing (or state why it is intended):"]
    for label, out in failures:
        defects = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[")]
        lines.append(f"\n- {label}:")
        lines.extend(f"    {d}" for d in defects)
    lines.append("\nRe-run any checker: python3 .claude/skills/agent-readiness-audit/scripts/<name>.py")
    sys.stderr.write("\n".join(lines) + "\n")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except Exception:
        sys.exit(0)
