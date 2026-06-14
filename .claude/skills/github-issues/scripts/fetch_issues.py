#!/usr/bin/env python3
"""Fetch all open GitHub issues and emit a grouped digest for the issue-handoff audit.

Usage:
    python fetch_issues.py [--limit 300] [--json]

Requires the ``gh`` CLI authenticated for the current repo. Default output is a
markdown digest: a summary table (number / prio / topic / type / title) followed
by each issue's full body grouped by topic label. ``--json`` emits the raw issue
list instead (for programmatic post-processing).

This script only *fetches and structures* the issues. The scope / standards /
already-addressed classification is the agent's job (see SKILL.md) because it
requires reading the repo's rules, git history, and ADRs.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict

TOPIC_ORDER = ["topic:core", "topic:data", "topic:performance", "topic:XAI"]


def fetch(limit):
    cmd = ["gh", "issue", "list", "--state", "open", "--limit", str(limit),
           "--json", "number,title,labels,body,createdAt,updatedAt"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr or "gh issue list failed\n")
        sys.exit(out.returncode or 1)
    return json.loads(out.stdout)


def _names(issue):
    return [lbl["name"] for lbl in issue["labels"]]


def topic_of(issue):
    names = _names(issue)
    for t in TOPIC_ORDER:
        if t in names:
            return t
    return "topic:other"


def prio_of(issue):
    names = _names(issue)
    for p in ("prio:1", "prio:2", "prio:3"):
        if p in names:
            return p
    return "prio:?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    issues = fetch(args.limit)
    if args.json:
        print(json.dumps(issues, indent=2))
        return

    print(f"# Open issues digest — {len(issues)} open\n")
    print("| # | prio | topic | type | title |")
    print("|---|---|---|---|---|")
    for it in sorted(issues, key=lambda x: x["number"], reverse=True):
        types = ",".join(n for n in _names(it) if n.startswith("type:")) or "-"
        print(f"| {it['number']} | {prio_of(it)} | {topic_of(it)} | {types} | {it['title']} |")
    print()

    groups = defaultdict(list)
    for it in issues:
        groups[topic_of(it)].append(it)
    for t in TOPIC_ORDER + ["topic:other"]:
        if t not in groups:
            continue
        print(f"\n## {t}\n")
        for it in sorted(groups[t], key=lambda x: (prio_of(x), -x["number"])):
            print(f"### #{it['number']} — {it['title']}")
            print(f"labels: {', '.join(_names(it))}  | updated: {it['updatedAt'][:10]}\n")
            print((it["body"] or "_(no body)_").strip())
            print("\n---\n")


if __name__ == "__main__":
    main()
