"""Generate the DataFrame-schemas reference page from the machine-readable registry.

The data dictionary lives in ``aaanalysis.utils.DICT_DF_SCHEMAS``; this script renders
it to ``docs/source/index/usage_principles/df_schemas.rst`` so the documentation has a
single source of truth. ``tests/unit/api_tests/test_df_schemas.py`` re-runs the render
and asserts the committed page matches, so the docs cannot drift from the code.

Usage:
    python .github/scripts/gen_df_schemas_doc.py            # write the page
    python .github/scripts/gen_df_schemas_doc.py --check    # exit 1 if out of sync
"""
import argparse
import pathlib
import sys

import aaanalysis.utils as ut

DOC_PATH = (pathlib.Path(__file__).resolve().parents[1].parent
            / "docs/source/index/usage_principles/df_schemas.rst")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the committed page is out of sync")
    args = parser.parse_args()
    rendered = ut.render_schemas_rst()
    if args.check:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != rendered:
            print("df_schemas.rst is out of sync; run gen_df_schemas_doc.py", file=sys.stderr)
            return 1
        print("df_schemas.rst is in sync")
        return 0
    DOC_PATH.write_text(rendered)
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
