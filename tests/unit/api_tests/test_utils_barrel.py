"""This is a script to guard the utils.py import-barrel invariants after the
constants were extracted into the sibling module ``aaanalysis._constants``.

``utils.py`` stays the single public access point (``import aaanalysis.utils as ut``);
the constants now live in ``_constants.py`` and are re-exported via
``from ._constants import *``. These tests pin the two properties that keep that safe:
the constants module must not import ``aaanalysis`` (circular-import guard), and every
public constant must still resolve as ``ut.X``.
"""
import ast
import pathlib

import aaanalysis._constants as _constants
import aaanalysis.utils as ut


def test_constants_module_has_no_aaanalysis_import():
    """_constants.py must depend only on the stdlib + numpy, never on aaanalysis,
    so it can be imported first without a circular dependency."""
    src = pathlib.Path(_constants.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("aaanalysis"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # No relative imports and no aaanalysis imports.
            assert node.level == 0, f"relative import in _constants: level={node.level}"
            assert not module.startswith("aaanalysis"), module


def test_all_public_constants_reexported_by_ut():
    """Every public name defined in _constants is reachable (and identical) via ut,
    so no `ut.X` call site can break from the move."""
    public = [n for n in dir(_constants)
              if not n.startswith("_") and n not in {"os", "platform", "np"}]
    assert public, "expected _constants to expose constants"
    missing = [n for n in public if not hasattr(ut, n)]
    assert not missing, f"constants not re-exported by ut: {missing}"
    for n in public:
        assert getattr(ut, n) is getattr(_constants, n), n


def test_critical_symbols_resolve_through_ut():
    """A representative mix of constants AND functions still resolves through the
    barrel (functions stayed in utils.py; constants moved)."""
    # constants (now from _constants)
    for name in ["SEP", "FOLDER_DATA", "LIST_CANONICAL_AA", "LIST_COLS_FEAT",
                 "DICT_DF_FEAT", "COL_FEATURE", "STR_DICT_CAT", "DICT_COLOR"]:
        assert hasattr(ut, name), f"missing constant ut.{name}"
    # functions (still in utils.py)
    for name in ["split_feat_id", "join_feat_id", "sort_cols_feat", "check_df_feat",
                 "check_df_seq", "plot_get_cmap_", "load_default_scales", "print_out"]:
        assert callable(getattr(ut, name)), f"missing/!callable ut.{name}"
