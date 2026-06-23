#!/usr/bin/env python3
"""Give every Claude Code session a descriptive terminal tab title, so parallel
sessions running in different git worktrees are tellable apart at a glance.

Wired as a Stop hook (fires when Claude finishes a turn). Once the session has
used ~1% of the context window it starts maintaining the tab title; it only
re-writes the title when the computed label actually changes, so it is silent
otherwise (effectively "set once, then refreshed only when something meaningful
changes" — e.g. when a branch is created, a PR is opened, or an ADR is added).

Title format — topic first, then the key IDs that exist:

    <topic> · PR#<n> · ADR<nnnn>     e.g.  adr-parallel-fix · PR#233 · ADR0038

  * topic : the branch slug ("doc/x-y" -> "x-y") — the human-authored "what this
            is about"; falls back to the worktree/repo dir name on master/detached.
  * PR#   : the open PR for the branch (one `gh` call, cached once found).
  * ADR   : new docs/adr/NNNN-*.md files introduced on the branch.

Tunables (env vars):
    CLAUDE_TITLE_THRESHOLD_CHARS   transcript bytes that count as ~1% usage (default 8000)
    CLAUDE_TITLE_MIN_TURNS         fallback turn count if no transcript size is available (default 2)

Writes only to /dev/tty (never stdout, which would be injected into the
conversation) and always exits 0 so it can never block a turn.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

THRESHOLD_CHARS = int(os.environ.get("CLAUDE_TITLE_THRESHOLD_CHARS", "8000"))
MIN_TURNS = int(os.environ.get("CLAUDE_TITLE_MIN_TURNS", "2"))

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

# Gate: has the session used ~1% of context yet? Prefer transcript size; fall
# back to a per-session turn counter when no transcript path is provided.
used_chars = os.path.getsize(tpath) if tpath and os.path.exists(tpath) else 0
ready = used_chars >= THRESHOLD_CHARS
if not ready and used_chars == 0:
    n = int(state.get("turns", 0)) + 1
    state["turns"] = n
    ready = n >= MIN_TURNS
    try:
        json.dump(state, open(state_path, "w"))
    except Exception:
        pass
if not ready:
    sys.exit(0)


def _sh(args, timeout=8):
    try:
        return subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return ""


# Topic = branch slug (the human-authored "what this is about"); fall back to the
# worktree/repo dir name on master/detached.
repo = os.path.basename((_sh(["git", "-C", cwd, "rev-parse", "--show-toplevel"]) or cwd).rstrip("/")) or "session"
branch = _sh(["git", "-C", cwd, "branch", "--show-current"])
on_branch = bool(branch) and branch not in ("master", "main")
topic = branch.split("/", 1)[-1] if on_branch else repo

# PR number — network, so cache it once found and only probe while on a real branch.
pr = str(state.get("pr") or "")
if not pr and on_branch:
    got = _sh(["gh", "pr", "view", "--json", "number", "-q", ".number"])
    pr = got if got.isdigit() else ""

# ADR(s) introduced on this branch: added-vs-origin/master + still-untracked.
adr_lines = _sh(["git", "-C", cwd, "diff", "--name-only", "--diff-filter=A",
                 "origin/master...HEAD", "--", "docs/adr"]).splitlines()
adr_lines += _sh(["git", "-C", cwd, "ls-files", "--others", "--exclude-standard",
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
try:
    json.dump(state, open(state_path, "w"))
except Exception:
    pass

sys.exit(0)
