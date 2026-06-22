#!/usr/bin/env python3
"""ADR-hygiene checker + overview-table generator for ``agent-readiness-audit``.

The decision log (``docs/adr/``) is part of an agent's mental model: a stale or
contradictory ADR misleads exactly like a stale README. This proves the
*structural* half of ADR hygiene — status lines well-formed, cross-references
resolve, supersession targets exist, **no two ADRs share a number** (the
parallel-session collision hazard), the repo convention "code must never
reference an ADR" holds, and the auto-generated overview table (``INDEX.md``) is
present and current. Whether an ADR's *content* is outdated is judgment a
reviewer makes; this script only flags mechanical defects.

A duplicate number (``ADR-DUP-NUMBER``) is a defect: concurrent branches each
pick ``max + 1`` from a stale checkout and collide. Gaps are NOT flagged — an
ADR legitimately living on an unmerged branch leaves a hole that is normal
mid-flight. Un-numbered ``XXXX-<slug>.md`` drafts are advisory, a reminder to
assign the number last (``docs/adr/README.md`` -> Conventions).

It does **not** delete anything. Per ``docs/adr/README.md`` the log is append-only:
reverse a decision by writing a new ADR that supersedes it and flipping the old
status — delete only when fully obsolete and only with the maintainer's explicit
go-ahead (repo hard rule). Superseded ADRs are surfaced as advisory, not defects.

Output mirrors ``check_agentic_docs.py``: Defects (exit != 0) vs Advisory.

Usage:
    python .../check_adrs.py [--adr-dir docs/adr] [--src aaanalysis]
    python .../check_adrs.py --write-index      # (re)generate docs/adr/INDEX.md
"""
from __future__ import annotations

import argparse
import os
import re

ADR_FILE_RE = re.compile(r"^(\d{4})-.*\.md$")
DRAFT_FILE_RE = re.compile(r"^XXXX-.*\.md$", re.IGNORECASE)
ADR_TOKEN_RE = re.compile(r"ADR-(\d{3,4})")
STATUS_RE = re.compile(r"^Status:\s*(.+)$", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s*ADR-\d{3,4}\s*[—\-–:]\s*(.+?)\s*$", re.MULTILINE)
SUPERSEDE_RE = re.compile(r"[Ss]uperseded by ADR-(\d{3,4})")
STATUS_PARTS_RE = re.compile(
    r"^(?P<word>\w+)\s*[—\-–]?\s*(?P<date>\d{4}-\d{2}-\d{2})?\s*(?:\((?P<note>.*)\))?")

INDEX_FILE = "INDEX.md"
INDEX_START = "<!-- ADR-INDEX:START — auto-generated; regenerate with check_adrs.py --write-index -->"
INDEX_END = "<!-- ADR-INDEX:END -->"
INDEX_HEADER = """# ADR index

Auto-generated overview of every Architecture Decision Record. The conventions
(when to write one, the template, status rules) live in [README.md](README.md).

Regenerate this table with:
`python .claude/skills/agent-readiness-audit/scripts/check_adrs.py --write-index`

"""

ADVISORY_CODES = {"ADR-SUPERSEDED", "ADR-SUPERSEDE-ASYMMETRIC", "ADR-UNNUMBERED-DRAFT"}


class Finding:
    __slots__ = ("where", "code", "detail")

    def __init__(self, where: str, code: str, detail: str) -> None:
        self.where, self.code, self.detail = where, code, detail

    @property
    def is_defect(self) -> bool:
        return self.code not in ADVISORY_CODES


# --------------------------------------------------------------------------- #
# I Helper Functions
# --------------------------------------------------------------------------- #
def _adr_files(adr_dir: str):
    out = {}
    for name in sorted(os.listdir(adr_dir)):
        m = ADR_FILE_RE.match(name)
        if m:
            out[int(m.group(1))] = os.path.join(adr_dir, name)
    return out


def _numbering_findings(adr_dir: str):
    """Defect on duplicate ADR numbers; advisory on un-numbered (``XXXX-``) drafts.

    A duplicate number is the parallel-session hazard: two branches each grab the
    same ``NNNN`` from their own stale checkout. ``_adr_files()`` keys by ``int`` so
    it silently keeps only one of the colliding files — scan the raw listing here
    instead. Gaps are deliberately NOT flagged: ADRs legitimately live on unmerged
    branches, so a missing number is normal mid-flight, not a defect. See
    ``docs/adr/README.md`` -> Conventions ("Number last, against live state").
    """
    findings = []
    by_num: dict[int, list[str]] = {}
    for name in sorted(os.listdir(adr_dir)):
        m = ADR_FILE_RE.match(name)
        if m:
            by_num.setdefault(int(m.group(1)), []).append(name)
        elif DRAFT_FILE_RE.match(name):
            findings.append(Finding(name, "ADR-UNNUMBERED-DRAFT",
                                    "un-numbered draft — assign the real number before merge "
                                    "(one past the max across committed ADRs and open PRs)"))
    for num, names in sorted(by_num.items()):
        if len(names) > 1:
            findings.append(Finding(", ".join(names), "ADR-DUP-NUMBER",
                                    f"number {num:04d} reused by {len(names)} files "
                                    f"({', '.join(names)}); renumber against live state"))
    return findings


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _status(text: str):
    m = STATUS_RE.search(text)
    return m.group(1).strip() if m else None


def _title(text: str):
    m = TITLE_RE.search(text)
    return m.group(1).strip() if m else "(no title)"


def _parse_status(status: str):
    """Return (word, date, note) from a Status: line value."""
    if not status:
        return ("", "", "")
    m = STATUS_PARTS_RE.match(status)
    if not m:
        return (status, "", "")
    return (m.group("word") or "", m.group("date") or "", (m.group("note") or "").strip())


def _build_index_region(adrs) -> str:
    rows = ["| ADR | Title | Status | Date | Notes |", "|----:|-------|--------|------|-------|"]
    for num in sorted(adrs):
        text = _read(adrs[num])
        word, date, note = _parse_status(_status(text) or "")
        link = f"[{num:04d}]({os.path.basename(adrs[num])})"
        rows.append(f"| {link} | {_title(text)} | {word} | {date} | {note} |")
    return f"{INDEX_START}\n" + "\n".join(rows) + f"\n{INDEX_END}\n"


def _extract_region(text: str):
    if INDEX_START in text and INDEX_END in text:
        i = text.index(INDEX_START)
        j = text.index(INDEX_END) + len(INDEX_END)
        return text[i:j].strip() + "\n"
    return None


def _render_full(existing: str, region: str) -> str:
    if existing and INDEX_START in existing and INDEX_END in existing:
        pre = existing[:existing.index(INDEX_START)]
        post = existing[existing.index(INDEX_END) + len(INDEX_END):]
        return pre + region.rstrip("\n") + post
    return INDEX_HEADER + region


def _check_one(num, path, text, existing, files):
    findings = []
    name = os.path.basename(path)
    status = _status(text)
    if status is None:
        findings.append(Finding(name, "ADR-NO-STATUS", "no 'Status:' line"))
    else:
        if not re.match(r"(Accepted|Superseded|Proposed|Deprecated)", status):
            findings.append(Finding(name, "ADR-BAD-STATUS",
                                    f"status not Accepted/Superseded/...: {status!r}"))
        if re.search(r"[Ss]uperseded", status):
            findings.append(Finding(name, "ADR-SUPERSEDED", f"status: {status!r}"))
        for target in (int(t) for t in SUPERSEDE_RE.findall(status)):
            if target in existing:
                back = _read(files[target])
                if f"ADR-{num:04d}" not in back and f"ADR-{num}" not in back:
                    findings.append(Finding(name, "ADR-SUPERSEDE-ASYMMETRIC",
                                            f"ADR-{target:04d} does not reference back to ADR-{num:04d}"))
    for tok in sorted(set(int(t) for t in ADR_TOKEN_RE.findall(text))):
        if tok != num and tok not in existing:
            findings.append(Finding(name, "ADR-XREF-DANGLING",
                                    f"references ADR-{tok:04d} (no such ADR)"))
    return findings


def _check_code_refs(src: str):
    """Convention (docs/adr/README.md): code must never reference an ADR."""
    findings = []
    if not os.path.isdir(src):
        return findings
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            for i, line in enumerate(_read(p).splitlines(), 1):
                m = ADR_TOKEN_RE.search(line)
                if m:
                    findings.append(Finding(f"{p}:{i}", "ADR-IN-CODE",
                                            f"code references ADR-{int(m.group(1)):04d} "
                                            f"(put the rationale inline instead)"))
    return findings


def _check_index(adr_dir, adrs):
    region = _build_index_region(adrs)
    path = os.path.join(adr_dir, INDEX_FILE)
    if not os.path.isfile(path):
        return [Finding(INDEX_FILE, "ADR-INDEX-MISSING",
                        "no overview table (run --write-index)")]
    have = _extract_region(_read(path))
    if have is None:
        return [Finding(INDEX_FILE, "ADR-INDEX-MISSING",
                        "INDEX.md has no ADR-INDEX markers (run --write-index)")]
    if have.strip() != region.strip():
        return [Finding(INDEX_FILE, "ADR-INDEX-STALE",
                        "overview table out of date (run --write-index)")]
    return []


# --------------------------------------------------------------------------- #
# II Main Functions
# --------------------------------------------------------------------------- #
def run(adr_dir: str, src: str, write_index: bool) -> int:
    if not os.path.isdir(adr_dir):
        print(f"error: ADR dir '{adr_dir}' not found")
        return 2
    adrs = _adr_files(adr_dir)
    existing = set(adrs)

    if write_index:
        path = os.path.join(adr_dir, INDEX_FILE)
        prev = _read(path) if os.path.isfile(path) else ""
        full = _render_full(prev, _build_index_region(adrs))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(full)
        print(f"wrote {path} ({len(adrs)} ADRs)")
        return 0

    findings = []
    findings.extend(_numbering_findings(adr_dir))
    for num, path in adrs.items():
        findings.extend(_check_one(num, path, _read(path), existing, adrs))
    findings.extend(_check_code_refs(src))
    findings.extend(_check_index(adr_dir, adrs))

    defects = [f for f in findings if f.is_defect]
    advisory = [f for f in findings if not f.is_defect]

    print(f"agent-readiness-audit — ADR-hygiene check on {adr_dir}")
    print(f"  scanned {len(adrs)} ADR(s); code-ref scan in {src}/\n")
    if defects:
        print("Defects:")
        for f in defects:
            print(f"  [{f.code}] {f.where}: {f.detail}")
    else:
        print("Defects: none — status lines well-formed, cross-refs resolve, "
              "no duplicate ADR numbers, no ADR referenced from code, overview table current.")
    if advisory:
        print("\nAdvisory:")
        for f in advisory:
            print(f"  [{f.code}] {f.where}: {f.detail}")
    print(f"\nSummary: {len(defects)} defect(s), {len(advisory)} advisory.")
    return 1 if defects else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ADR-hygiene checker + overview generator.")
    ap.add_argument("--adr-dir", default="docs/adr", help="ADR directory (default: docs/adr)")
    ap.add_argument("--src", default="aaanalysis", help="source tree to scan for ADR refs")
    ap.add_argument("--write-index", action="store_true",
                    help="(re)generate the overview table at <adr-dir>/INDEX.md")
    args = ap.parse_args()
    return run(args.adr_dir, args.src, args.write_index)


if __name__ == "__main__":
    raise SystemExit(main())
