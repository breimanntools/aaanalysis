#!/usr/bin/env python3
"""Reconcile local branches / worktrees against GitHub PR state, and prune the merged ones.

Why this exists
---------------
This repo **squash-merges** every PR. A squash creates a brand-new commit on
``master`` with a different SHA, so the feature branch's own commits are never
reachable from ``master``. That breaks the obvious cleanup primitives:

* ``git branch --merged master``  -> does NOT list a squash-merged branch
* ``git branch -d <branch>``      -> refuses it ("not fully merged")

So reachability-based cleanup silently no-ops and merged branches pile up
(made worse by parallel sessions whose auto-merge lands *after* the session
ended). This tool instead asks the source of truth -- **PR state via ``gh``** --
and classifies every local branch / worktree into:

* MERGED   -> its PR is merged; the work is in master; safe to delete.
* OPEN     -> open PR; in-flight; KEEP.
* FORGOTTEN-> no PR and commits not in master; real work nobody is tracking.
             NEVER auto-deleted -- only flagged for a human.
* CONTAINED-> no PR but introduces nothing master lacks; safe to delete.

It is idempotent and read-only by default: it PRINTS recommendations and only
deletes when you ask it to. Decoupled from any single session -- run it whenever.

Usage
-----
    python .github/scripts/prune_merged_branches.py            # report only (default)
    python .github/scripts/prune_merged_branches.py --apply    # delete the merged/contained local branches
    python .github/scripts/prune_merged_branches.py do-it       # same as --apply (friendly alias)
    python .github/scripts/prune_merged_branches.py --apply --remote   # also delete stale REMOTE branches (a push!)
    python .github/scripts/prune_merged_branches.py --no-fetch  # skip the initial 'git fetch --prune'

FORGOTTEN branches are never deleted by ``--apply`` -- resolve them by hand
(open a PR, or delete deliberately once you've confirmed the work is unwanted).
"""
from __future__ import annotations

import json
import subprocess
import sys

APPLY_WORDS = {"--apply", "apply", "do", "do-it", "doit", "do_it", "run"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _git_ok(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def default_branch() -> str:
    rc, out = _git_ok("rev-parse", "--abbrev-ref", "origin/HEAD")
    if rc == 0 and out:
        return out.split("/", 1)[-1]  # 'origin/master' -> 'master'
    return "master"


def pr_status_map() -> dict[str, str]:
    """headRefName -> one of OPEN / MERGED / CLOSED, preferring OPEN > MERGED > CLOSED."""
    out = subprocess.run(
        ["gh", "pr", "list", "--state", "all", "--limit", "300",
         "--json", "state,headRefName"],
        capture_output=True, text=True, check=True,
    ).stdout
    rank = {"OPEN": 3, "MERGED": 2, "CLOSED": 1}
    best: dict[str, str] = {}
    for pr in json.loads(out):
        head, state = pr["headRefName"], pr["state"]
        if head not in best or rank[state] > rank[best[head]]:
            best[head] = state
    return best


def worktree_map() -> dict[str, str]:
    """branch-name -> worktree path, for branches currently checked out in a worktree."""
    out = _git("worktree", "list", "--porcelain")
    wt: dict[str, str] = {}
    path = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
        elif line.startswith("branch ") and path:
            name = line[len("branch "):].replace("refs/heads/", "", 1)
            wt[name] = path
    return wt


def is_clean(path: str) -> bool:
    rc, out = _git_ok("-C", path, "status", "--porcelain")
    return rc == 0 and out == ""


def classify(branch: str, base: str, prs: dict[str, str]) -> tuple[str, str]:
    """Return (verdict, detail). verdict in KEEP / DELETE / FORGOTTEN."""
    state = prs.get(branch)
    if state == "OPEN":
        return "KEEP", "open PR"
    if state == "MERGED":
        return "DELETE", "PR merged (work is in {})".format(base)
    # No open/merged PR (None or CLOSED-not-merged): decide on content.
    rc, _ = _git_ok("diff", "--quiet", "{}...{}".format(base, branch))
    if rc == 0:
        return "DELETE", "no open PR; introduces nothing {} lacks".format(base)
    ahead = _git("rev-list", "--count", "{}..{}".format(base, branch))
    return "FORGOTTEN", "NO PR and {} commit(s) not in {}".format(ahead, base)


def main() -> int:
    args = set(sys.argv[1:])
    apply = bool(args & APPLY_WORDS)
    do_remote = "--remote" in args
    fetch = "--no-fetch" not in args

    if fetch:
        _git_ok("fetch", "--prune", "origin")

    base = "origin/" + default_branch()
    rc, _ = _git_ok("rev-parse", "--verify", base)
    if rc != 0:
        base = default_branch()  # no remote tracking; fall back to local

    current = _git("rev-parse", "--abbrev-ref", "HEAD")
    protected = {current, default_branch()}
    prs = pr_status_map()
    wt = worktree_map()

    locals_ = [b for b in _git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
               if b and b not in protected]

    buckets: dict[str, list[tuple[str, str]]] = {"DELETE": [], "KEEP": [], "FORGOTTEN": []}
    for b in sorted(locals_):
        verdict, detail = classify(b, base, prs)
        buckets[verdict].append((b, detail))

    print("Branch reconciliation vs {} ({} mode)\n".format(
        base, "APPLY" if apply else "report-only"))

    if buckets["FORGOTTEN"]:
        print("FORGOTTEN -- real work, no PR, NOT in {} (never auto-deleted; open a PR or handle by hand):".format(base))
        for b, d in buckets["FORGOTTEN"]:
            print("  ! {:45s} {}".format(b, d))
        print()
    if buckets["KEEP"]:
        print("KEEP -- in-flight:")
        for b, d in buckets["KEEP"]:
            print("  = {:45s} {}".format(b, d))
        print()

    print("DELETE candidates -- merged / fully contained:")
    if not buckets["DELETE"]:
        print("  (none)")
    for b, d in buckets["DELETE"]:
        loc = "  [in worktree {}]".format(wt[b]) if b in wt else ""
        print("  - {:45s} {}{}".format(b, d, loc))
    print()

    # Stale remote branches whose PR is already merged (info; deleted only with --remote --apply).
    merged_heads = {h for h, s in prs.items() if s == "MERGED"}
    remote_heads = {r.replace("origin/", "", 1)
                    for r in _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").splitlines()
                    if r.startswith("origin/") and not r.endswith("/HEAD")}
    stale_remote = sorted((merged_heads & remote_heads) - {default_branch()})
    if stale_remote:
        print("Stale REMOTE branches (merged PR, origin branch still exists) -- {} total:".format(len(stale_remote)))
        for b in stale_remote:
            print("  ~ origin/{}".format(b))
        print("  (enable 'Automatically delete head branches' in repo settings to stop this; "
              "or run with --remote --apply to delete them now -- a push)\n")

    if not apply:
        print("Report only. Re-run with --apply (or 'do-it') to delete the local DELETE candidates.")
        return 0

    # --- apply: delete local DELETE-candidate branches ---
    print("Applying local deletions ...")
    for b, _ in buckets["DELETE"]:
        if b in wt:
            path = wt[b]
            if not is_clean(path):
                print("  SKIP {} -- worktree {} has uncommitted changes (remove manually with --force)".format(b, path))
                continue
            rc, out = _git_ok("worktree", "remove", path)
            print(("  worktree removed {}".format(path)) if rc == 0 else "  WARN worktree remove failed: " + out)
            if rc != 0:
                continue
        rc, out = _git_ok("branch", "-D", b)
        print(("  deleted {}".format(b)) if rc == 0 else "  WARN: " + out)
    _git_ok("worktree", "prune")

    if do_remote and stale_remote:
        print("Deleting stale remote branches (push) ...")
        for b in stale_remote:
            rc, out = _git_ok("push", "origin", "--delete", b)
            print(("  deleted origin/{}".format(b)) if rc == 0 else "  WARN: " + out)
    elif stale_remote:
        print("Left {} stale remote branches untouched (add --remote to delete them).".format(len(stale_remote)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
