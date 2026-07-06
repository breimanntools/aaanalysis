"""Tests for the sequence<->structure highlight link (CPPStructurePlot).

The core logic is exercised **purely** (no live py3Dmol 3D render): the region->residue
expansion (:func:`_expand_regions`) and the per-residue styling helper
(:func:`_highlight_style_specs`) are inspected directly. The integration tests (plot_linked
HTML / map_structure) build the view/HTML but never rasterize an image, and are gated on the
optional structure dependencies.
"""
import matplotlib
matplotlib.use("Agg")
import re
import numpy as np
import pandas as pd
import pytest

import aaanalysis.utils as ut
# render.py imports lazily (py3Dmol only inside render_py3dmol), so these pure helpers import
# without the structure extras -> no importorskip needed for the pure-logic tests.
from aaanalysis.feature_engineering_pro._backend.cpp_struct.render import (
    _expand_regions, _highlight_style_specs)

CYAN = ut.COLOR_LINK_HIGHLIGHT   # '#00E5FF'


# --- pure: _expand_regions ---------------------------------------------------
class TestExpandRegions:
    """region(s) -> sorted set of 1-based residue numbers, clamped to [1, seq_len]."""

    def test_single_tuple(self):
        assert _expand_regions((5, 7)) == {5, 6, 7}

    def test_inclusive_bounds(self):
        # stop is inclusive: (10, 10) is the single residue 10.
        assert _expand_regions((10, 10)) == {10}

    def test_list_of_tuples_union(self):
        assert _expand_regions([(5, 7), (10, 11)]) == {5, 6, 7, 10, 11}

    def test_list_dedups_overlap(self):
        assert _expand_regions([(5, 7), (6, 8)]) == {5, 6, 7, 8}

    def test_none_is_empty(self):
        assert _expand_regions(None) == set()

    def test_clamp_upper_to_seq_len(self):
        assert _expand_regions((98, 105), 100) == {98, 99, 100}

    def test_clamp_lower_to_one(self):
        # negative / zero starts clamp up to residue 1 (with or without an upper bound).
        assert _expand_regions((-2, 3), 100) == {1, 2, 3}
        assert _expand_regions((0, 2)) == {1, 2}

    def test_seq_len_none_no_upper_clamp(self):
        assert _expand_regions((5, 8), None) == {5, 6, 7, 8}

    def test_numpy_ints_accepted(self):
        assert _expand_regions((np.int64(4), np.int64(6))) == {4, 5, 6}

    def test_returns_set(self):
        assert isinstance(_expand_regions((1, 3)), set)

    # --- negatives -----------------------------------------------------------
    def test_start_gt_stop_raises(self):
        with pytest.raises(ValueError):
            _expand_regions((7, 5))

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            _expand_regions((1, "a"))

    def test_float_raises(self):
        with pytest.raises(ValueError):
            _expand_regions((1.5, 3))

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            _expand_regions((True, 3))

    def test_wrong_region_shape_raises(self):
        with pytest.raises(ValueError):
            _expand_regions([(1, 2, 3)])

    def test_start_gt_stop_in_list_raises(self):
        with pytest.raises(ValueError):
            _expand_regions([(1, 2), (9, 4)])


# --- pure: _highlight_style_specs -------------------------------------------
class TestHighlightStyleSpecs:
    """The render style path assigns COLOR_LINK_HIGHLIGHT to exactly the present residues."""

    def _colored_resis(self, specs):
        """Residues whose produced cartoon style is the highlight cyan (as ints)."""
        out = []
        for sel, style in specs:
            assert style == {"cartoon": {"color": CYAN}}   # exactly the linked cyan, nothing else
            out.append(int(sel["resi"]))
        return out

    def test_exact_residues_get_cyan(self):
        present = {5, 6, 7, 8, 9, 10}
        specs = _highlight_style_specs(_expand_regions((6, 8)), present)
        assert self._colored_resis(specs) == [6, 7, 8]

    def test_drops_residues_absent_from_structure(self):
        # highlight 20..21 is outside the structure -> only the present 6..8 stay.
        present = {5, 6, 7, 8, 9, 10}
        specs = _highlight_style_specs(_expand_regions([(6, 8), (20, 21)]), present)
        assert self._colored_resis(specs) == [6, 7, 8]

    def test_sorted_ascending(self):
        present = set(range(1, 40))
        specs = _highlight_style_specs(_expand_regions([(30, 31), (3, 4)]), present)
        assert self._colored_resis(specs) == [3, 4, 30, 31]

    def test_chain_qualified_selection(self):
        specs = _highlight_style_specs({6, 7}, {6, 7}, chain_id="A")
        for sel, _style in specs:
            assert sel["chain"] == "A"

    def test_no_chain_key_when_none(self):
        specs = _highlight_style_specs({6}, {6, 7}, chain_id=None)
        assert "chain" not in specs[0][0]

    def test_none_highlight_is_empty(self):
        assert _highlight_style_specs(None, {1, 2, 3}) == []

    def test_empty_highlight_is_empty(self):
        assert _highlight_style_specs(set(), {1, 2, 3}) == []

    def test_no_overlap_is_empty(self):
        assert _highlight_style_specs({50, 51}, {1, 2, 3}) == []


# --- integration fixtures (structure extras) ---------------------------------
def _make_pdb(path, n=30, chain="A"):
    lines, serial = [], 1
    for i in range(n):
        x, y, z = i * 1.5, np.sin(i * 0.5) * 6, np.cos(i * 0.5) * 6
        b = 40 + (i % 60)
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA {chain}{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def pdb_path(tmp_path_factory):
    return _make_pdb(tmp_path_factory.mktemp("struct_highlight") / "synthetic.pdb")


@pytest.fixture(scope="module")
def df_feat():
    import aaanalysis as aa
    df_cat = aa.load_scales(name="scales_cat").head(5).reset_index(drop=True)
    splits = ["Segment(1,2)", "Segment(2,2)", "Segment(1,1)", "Pattern(C,1)", "Segment(1,4)"]
    parts = ["TMD", "TMD", "JMD_N", "TMD", "JMD_C"]
    rows = {ut.COL_FEATURE: [], ut.COL_CAT: [], ut.COL_SUBCAT: [], ut.COL_SCALE_NAME: []}
    for i, row in df_cat.iterrows():
        rows[ut.COL_FEATURE].append(f"{parts[i]}-{splits[i]}-{row[ut.COL_SCALE_ID]}")
        rows[ut.COL_CAT].append(row[ut.COL_CAT])
        rows[ut.COL_SUBCAT].append(row[ut.COL_SUBCAT])
        rows[ut.COL_SCALE_NAME].append(row[ut.COL_SCALE_NAME])
    df = pd.DataFrame(rows)
    df["abs_auc"] = 0.2
    df["abs_mean_dif"] = 0.3
    df["mean_dif"] = [0.3, -0.2, 0.5, -0.4, 0.25]
    df["std_test"] = 0.1
    df["std_ref"] = 0.1
    df["feat_impact"] = [0.8, -0.5, 1.2, -0.3, 0.6]
    df["feat_importance"] = [0.8, 0.5, 1.2, 0.3, 0.6]
    return df


def _csp():
    import aaanalysis as aa
    return aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)


def _linked_cyan_resis(html):
    """Residues whose linked-HTML residue_styles row carries the highlight cyan (as ints)."""
    rows = re.findall(r"\[(\d+),\s*\"" + re.escape(CYAN) + r"\",\s*0\.0\]", html)
    return sorted(int(r) for r in rows)


class TestPlotLinkedHighlight:
    """plot_linked embeds cyan residue_styles rows for the highlighted residues (no 3D render)."""

    @pytest.fixture(autouse=True)
    def _deps(self):
        # Structure parsing needs biopython; plot_linked's py3Dmol gate needs it importable.
        # Neither test rasterizes a 3D image -- only the HTML text is built + inspected.
        pytest.importorskip("Bio")
        pytest.importorskip("py3Dmol")

    def test_single_region_marks_cyan(self, pdb_path, df_feat):
        # start=1 -> window residues 1..30; highlight (3,5) -> residues 3,4,5 cyan.
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                               col_imp="feat_impact", highlight=(3, 5))._repr_html_()
        assert _linked_cyan_resis(h) == [3, 4, 5]

    def test_list_of_regions_marks_cyan(self, pdb_path, df_feat):
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                               col_imp="feat_impact", highlight=[(3, 4), (10, 11)])._repr_html_()
        assert _linked_cyan_resis(h) == [3, 4, 10, 11]

    def test_out_of_structure_residues_dropped(self, pdb_path, df_feat):
        # residues 28..35 -> only 28,29,30 exist in the 30-residue structure.
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                               col_imp="feat_impact", highlight=(28, 35))._repr_html_()
        assert _linked_cyan_resis(h) == [28, 29, 30]

    def test_no_highlight_no_cyan_rows(self, pdb_path, df_feat):
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                               col_imp="feat_impact")._repr_html_()
        assert _linked_cyan_resis(h) == []

    def test_invalid_highlight_raises(self, pdb_path, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                               col_imp="feat_impact", highlight=(9, 4))


class TestMapStructureHighlight:
    """map_structure accepts highlight; validation is fail-fast (before any 3D render)."""

    def test_invalid_highlight_raises_before_render(self, df_feat):
        # _expand_regions runs in the Validate block, before _require_py3dmol / structure parse,
        # so a bad region raises without needing py3Dmol or a real structure file.
        with pytest.raises(ValueError):
            _csp().map_structure(df_feat=df_feat, pdb="/nonexistent.pdb", tmd_len=10,
                                 col_imp="feat_impact", highlight=(9, 4))

    def test_highlight_builds_view(self, pdb_path, df_feat):
        # Builds the py3Dmol view object (in-memory; no image is rasterized).
        pytest.importorskip("Bio")
        pytest.importorskip("py3Dmol")
        view = _csp().map_structure(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                                    col_imp="feat_impact", highlight=[(3, 5), (28, 35)])
        assert view is not None
