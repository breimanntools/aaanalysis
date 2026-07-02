"""Architecture guard: frontends must not import from another class's dedicated backend.

A frontend module (``aaanalysis/<subpkg>/_<file>.py``) may import backend helpers only
from a **shared** backend module (a top-level ``_backend/*.py`` or an explicitly shared
subpackage) or from its **own** dedicated ``_backend/<subpkg>/`` package — never from a
sibling class's dedicated backend subpackage. Shared helpers belong in a common
``_backend/*.py`` module instead.

This codifies the PR #110/#113 fix (``SequenceFeature`` had reached into
``_backend/num_feat/``, NumericalFeature's dedicated backend). To extend the guard to a
new dedicated backend subpackage, add it to ``DEDICATED_OWNERS`` below.
"""
import ast
import pathlib

import aaanalysis

ROOT = pathlib.Path(aaanalysis.__file__).resolve().parent

# Backend subpackages that are intentionally SHARED across frontends of a subpackage
# (any frontend in that subpackage may import them). Top-level ``_backend/*.py`` modules
# are always treated as shared.
SHARED_BACKEND_SUBPKGS = {
    "feature_engineering": {"cpp"},  # the CPP feature backend is shared by CPP/SequenceFeature/...
}

# Dedicated backend subpackages -> the set of frontend module stems allowed to import them.
DEDICATED_OWNERS = {
    "feature_engineering": {
        "num_feat": {"_numerical_feature"},
        "aaclust": {"_aaclust", "_aaclust_plot"},
    },
    "protein_design": {
        "seqopt": {"_seqopt"},
    },
    "feature_engineering_pro": {
        "cpp_struct": {"_cpp_structure_plot"},
    },
    "prediction": {
        "aa_pred": {"_aa_pred", "_aa_pred_plot"},
    },
}


def _frontend_files(pkg_dir):
    """Frontend modules of a subpackage (``_*.py`` directly under it, excluding dunders)."""
    return [p for p in pkg_dir.glob("_*.py") if not p.name.startswith("__")]


def _backend_subpkg_imports(py_file):
    """Yield the ``_backend/<subpkg>`` subpackage names a file imports from (relative)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level >= 1 and node.module:
            parts = node.module.split(".")
            if parts[0] == "_backend" and len(parts) >= 2:
                yield parts[1]


def test_no_cross_class_backend_imports():
    violations = []
    for pkg, owners in DEDICATED_OWNERS.items():
        shared = SHARED_BACKEND_SUBPKGS.get(pkg, set())
        for f in _frontend_files(ROOT / pkg):
            for subpkg in _backend_subpkg_imports(f):
                if subpkg in shared:
                    continue
                if subpkg in owners and f.stem not in owners[subpkg]:
                    violations.append(
                        f"{pkg}/{f.name} imports from dedicated '_backend/{subpkg}/' "
                        f"(allowed only for {sorted(owners[subpkg])}). "
                        f"Move shared helpers to a common '_backend/*.py' module."
                    )
    assert not violations, "Cross-class backend imports:\n" + "\n".join(violations)


def test_owner_config_points_to_real_dirs():
    """Guard against the config rotting: every configured subpackage must exist on disk."""
    for pkg, owners in DEDICATED_OWNERS.items():
        for subpkg in owners:
            assert (ROOT / pkg / "_backend" / subpkg).is_dir(), \
                f"DEDICATED_OWNERS references missing '_backend/{subpkg}/' in {pkg}"
