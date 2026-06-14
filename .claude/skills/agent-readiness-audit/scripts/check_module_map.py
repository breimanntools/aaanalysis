#!/usr/bin/env python3
"""Module-map validator for the ``agent-readiness-audit`` skill.

Checks the curated internal-dataflow map (default ``docs/module_map.md``) stays in
step with the package. It is a **validator, not a generator**: the diagram/prose is
curated by hand because a *semantic* dataflow (which DataFrame flows where) cannot
be reliably derived from imports — in this repo the frontends are decoupled and
route through the ``utils`` barrel, so an import graph would miss the real flow.

What it proves (deterministic, text-based):
  MAP-MISSING        — the map file does not exist.
  MAP-MISSING-SUBPKG — a current public subpackage is absent from the map.
  MAP-STALE-SUBPKG   — the map's roster names a subpackage that no longer exists.
What it cannot prove: that the curated edges/data objects are *correct* — that is a
human judgment (rubric A).

Usage:
    python .../check_module_map.py [--map docs/module_map.md] [--pkg aaanalysis]
    python .../check_module_map.py --write-roster      # regenerate the roster block
"""
from __future__ import annotations

import argparse
import os
import re

SKIP_DIRS = {"_utils", "_backend", "_data", "__pycache__"}
ROSTER_START = "<!-- MAP-SUBPKGS:START — roster checked by check_module_map.py; regenerate with --write-roster -->"
ROSTER_END = "<!-- MAP-SUBPKGS:END -->"
ADVISORY_CODES: set = set()


class Finding:
    __slots__ = ("where", "code", "detail")

    def __init__(self, where, code, detail):
        self.where, self.code, self.detail = where, code, detail

    @property
    def is_defect(self):
        return self.code not in ADVISORY_CODES


# --------------------------------------------------------------------------- #
# I Helper Functions
# --------------------------------------------------------------------------- #
def _public_subpackages(pkg_root):
    out = []
    for name in sorted(os.listdir(pkg_root)):
        if name.startswith("_") or name in SKIP_DIRS:
            continue
        if os.path.isfile(os.path.join(pkg_root, name, "__init__.py")):
            out.append(name)
    return out


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _roster(text):
    if ROSTER_START in text and ROSTER_END in text:
        block = text[text.index(ROSTER_START) + len(ROSTER_START): text.index(ROSTER_END)]
        return [m.group(1) for m in re.finditer(r"^\s*-\s*(\S+)", block, re.MULTILINE)]
    return None


def _render_roster(subpkgs):
    lines = "\n".join(f"- {s}" for s in sorted(subpkgs))
    return f"{ROSTER_START}\n{lines}\n{ROSTER_END}"


# --------------------------------------------------------------------------- #
# II Main Functions
# --------------------------------------------------------------------------- #
def run(map_path, pkg_root, write_roster):
    if not os.path.isfile(os.path.join(pkg_root, "__init__.py")):
        print(f"error: '{pkg_root}' is not a package")
        return 2
    live = _public_subpackages(pkg_root)

    if write_roster:
        if not os.path.isfile(map_path):
            print(f"error: map '{map_path}' not found — create it first")
            return 2
        text = _read(map_path)
        if ROSTER_START not in text:
            print(f"error: no roster markers in {map_path}")
            return 2
        new = text[:text.index(ROSTER_START)] + _render_roster(live) + text[text.index(ROSTER_END) + len(ROSTER_END):]
        with open(map_path, "w", encoding="utf-8") as fh:
            fh.write(new)
        print(f"wrote roster ({len(live)} subpackages) to {map_path}")
        return 0

    findings = []
    if not os.path.isfile(map_path):
        findings.append(Finding(map_path, "MAP-MISSING", "no module map (the inter-module mental model is absent)"))
    else:
        text = _read(map_path)
        for sub in live:                                    # coverage: live ⊆ map text
            if not re.search(rf"\b{re.escape(sub)}\b", text):
                findings.append(Finding(os.path.basename(map_path), "MAP-MISSING-SUBPKG",
                                        f"public subpackage '{sub}' not mentioned in the map"))
        roster = _roster(text)
        if roster is not None:                              # staleness: roster ⊆ live
            for sub in roster:
                if sub not in live:
                    findings.append(Finding(os.path.basename(map_path), "MAP-STALE-SUBPKG",
                                            f"roster names '{sub}' which is no longer a public subpackage"))

    defects = [f for f in findings if f.is_defect]
    print(f"agent-readiness-audit — module-map check on {map_path}")
    print(f"  {len(live)} live public subpackage(s)\n")
    if defects:
        print("Defects:")
        for f in defects:
            print(f"  [{f.code}] {f.where}: {f.detail}")
    else:
        print("Defects: none — the map exists and covers every public subpackage.")
    print(f"\nSummary: {len(defects)} defect(s).")
    print("Note: this validates coverage only — whether the curated edges/data objects "
          "are CORRECT is human judgment (rubric A), not a checker result.")
    return 1 if defects else 0


def main():
    ap = argparse.ArgumentParser(description="Module-map (internal dataflow) validator.")
    ap.add_argument("--map", default="docs/module_map.md", help="map file (default: docs/module_map.md)")
    ap.add_argument("--pkg", default="aaanalysis", help="package root (default: aaanalysis)")
    ap.add_argument("--write-roster", action="store_true", help="regenerate the roster block from live subpackages")
    args = ap.parse_args()
    return run(args.map, args.pkg, args.write_roster)


if __name__ == "__main__":
    raise SystemExit(main())
