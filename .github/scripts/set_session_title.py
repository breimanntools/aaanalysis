#!/usr/bin/env python3
"""Give every Claude Code session a descriptive terminal tab title, so parallel
sessions (each driving its own git worktree) are tellable apart at a glance.

Wired as a Stop hook (fires when Claude finishes a turn). It maintains the title
from the **very first turn** (no warm-up gate) and only re-writes it when the
computed label actually changes, so it is silent otherwise — refreshed only when
something meaningful changes (a PR is opened, an ADR is added, the active
worktree switches).

Title format — topic first, then the key IDs that exist:

    <topic> · PR#<n> · ADR<nnnn>     e.g.  adr-parallel-fix · PR#233 · ADR0038

  * topic : the branch slug ("doc/x-y" -> "x-y") — the human-authored "what this
            is about".
  * PR#   : the open PR for that branch.
  * ADR   : new docs/adr/NNNN-*.md files introduced on that branch.

Resolving the topic works **from any checkout, including the main one on
master**: if the session's own checkout is on a feature branch, that branch is
the topic; otherwise (master / main / detached — the common case when a session
is launched from the primary checkout and ``cd``s into per-task worktrees) it
attributes the session to the feature worktree **whose path appears most
recently in this session's own transcript** (the ``cd`` / edit targets a Stop
hook cannot otherwise see from a master ``cwd``). That correctly distinguishes
parallel sessions each driving a different worktree — a plain "most recent
commit" guess would mislabel one session with another's branch. Only when the
transcript names no feature worktree does it fall back to the repo dir name.

Writes only to /dev/tty (never stdout, which would be injected into the
conversation) and always exits 0 so it can never block a turn.
"""
import json
import os
import re
import subprocess
import sys
import tempfile


def _sh(args, timeout=8, cwd=None):
    try:
        return subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True, timeout=timeout, cwd=cwd).strip()
    except Exception:
        return ""


def _feature_worktrees(cwd):
    """Return ``[(branch, path), ...]`` for every worktree on a feature branch
    (i.e. not master/main), parsed from ``git worktree list --porcelain``."""
    out = _sh(["git", "-C", cwd, "worktree", "list", "--porcelain"])
    res = []
    path = branch = None

    def _flush():
        if path and branch and branch not in ("master", "main"):
            res.append((branch, path))

    for line in out.splitlines():
        if line.startswith("worktree "):
            _flush()
            path, branch = line[len("worktree "):], None
        elif line.startswith("branch "):
            branch = line[len("branch "):].replace("refs/heads/", "")
    _flush()
    return res


def _worktree_from_transcript(tpath, worktrees):
    """Pick the feature worktree this session is actually driving: the one whose
    path appears latest in the session transcript. ``("", "")`` if none appear.

    The Stop hook only sees the session ``cwd`` (often the master checkout), but
    the transcript records the ``cd`` / edit paths, so the most-recently-mentioned
    worktree path is this session's own — not a sibling session's."""
    if not tpath or not worktrees or not os.path.exists(tpath):
        return ("", "")
    try:
        # The tail is enough and bounds the read on long sessions.
        with open(tpath, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 262144))
            text = fh.read().decode("utf-8", "ignore")
    except Exception:
        return ("", "")
    best = None  # (last_index, branch, path)
    for branch, path in worktrees:
        idx = text.rfind(path)
        if idx >= 0 and (best is None or idx > best[0]):
            best = (idx, branch, path)
    return (best[1], best[2]) if best else ("", "")


try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

sid = str(data.get("session_id") or "nosid")
tpath = data.get("transcript_path") or ""
cwd = data.get("cwd") or os.getcwd()

tmp = tempfile.gettempdir()
state_path = os.path.join(tmp, "claude-title-%s.json" % sid)
try:
    state = json.load(open(state_path))
except Exception:
    state = {}

# Topic = branch slug. Prefer the session checkout's own branch; otherwise (the
# master/main/detached case) the most-recently-active feature worktree; finally
# the repo dir name. ``meta_tree`` / ``meta_branch`` are where PR + ADR metadata
# for that topic is read from.
repo = os.path.basename((_sh(["git", "-C", cwd, "rev-parse", "--show-toplevel"]) or cwd).rstrip("/")) or "session"
branch = _sh(["git", "-C", cwd, "branch", "--show-current"])
if branch and branch not in ("master", "main"):
    topic, meta_branch, meta_tree = branch.split("/", 1)[-1], branch, cwd
else:
    wt_branch, wt_path = _worktree_from_transcript(tpath, _feature_worktrees(cwd))
    if wt_branch:
        topic, meta_branch, meta_tree = wt_branch.split("/", 1)[-1], wt_branch, wt_path
    else:
        topic, meta_branch, meta_tree = repo, "", cwd

# PR number — network, so cache it once found; re-probe if the active branch
# changed under the session (e.g. a different worktree became the most recent).
pr = str(state.get("pr") or "")
if meta_branch and (not pr or state.get("pr_branch") != meta_branch):
    got = _sh(["gh", "pr", "view", meta_branch, "--json", "number", "-q", ".number"], cwd=meta_tree)
    pr = got if got.isdigit() else ""

# ADR(s) introduced on that branch: added-vs-origin/master + still-untracked.
ref = meta_branch or "HEAD"
adr_lines = _sh(["git", "-C", meta_tree, "diff", "--name-only", "--diff-filter=A",
                 "origin/master...%s" % ref, "--", "docs/adr"]).splitlines()
adr_lines += _sh(["git", "-C", meta_tree, "ls-files", "--others", "--exclude-standard",
                  "docs/adr"]).splitlines()
adrs = []
for f in adr_lines:
    m = re.search(r"/(\d{4})-", "/" + f)
    if m and m.group(1) not in adrs:
        adrs.append(m.group(1))
adrs.sort()

parts = [topic]
if pr:
    parts.append("PR#" + pr)
if adrs:
    parts.append("ADR" + "/".join(adrs))
label = " · ".join(parts)
if len(label) > 48:
    label = label[:47] + "…"

# Only touch the terminal when the label actually changed.
if label != state.get("label"):
    try:
        with open("/dev/tty", "w") as tty:
            tty.write("\033]0;%s\007" % label)
            tty.flush()
    except Exception:
        pass

state["label"] = label
state["pr"] = pr
state["pr_branch"] = meta_branch
try:
    json.dump(state, open(state_path, "w"))
except Exception:
    pass

sys.exit(0)
