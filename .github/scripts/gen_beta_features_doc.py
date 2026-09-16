"""Generate the beta-feature overview page from the ``**Experimental.**`` docstring marker.

The beta set is discovered at run time from ``aaanalysis.__all__`` (see
``aaanalysis._beta``); this script renders it to
``docs/source/index/usage_principles/beta_features.rst`` so the documentation has a single
source of truth. ``tests/unit/api_tests/test_beta_features.py`` re-runs the render and
asserts the committed page matches, so the docs cannot drift from the code.

Run it with the ``pro`` extra installed: a ``[pro]``-gated beta symbol is absent from
``__all__`` in a base install and would be dropped from the page.

Usage:
    python .github/scripts/gen_beta_features_doc.py            # write the page
    python .github/scripts/gen_beta_features_doc.py --check    # exit 1 if out of sync
"""
import argparse
import pathlib
import sys

import aaanalysis.utils as ut

DOC_PATH = (pathlib.Path(__file__).resolve().parents[1].parent
            / "docs/source/index/usage_principles/beta_features.rst")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the committed page is out of sync")
    args = parser.parse_args()
    rendered = ut.render_beta_rst()
    if args.check:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != rendered:
            print("beta_features.rst is out of sync; run gen_beta_features_doc.py",
                  file=sys.stderr)
            return 1
        print("beta_features.rst is in sync")
        return 0
    DOC_PATH.write_text(rendered)
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
