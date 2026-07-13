"""This is a script to test CPPStructurePlot.plot_linked() (feature-map <-> structure link)."""
import matplotlib
matplotlib.use("Agg")
import re
import numpy as np
import pandas as pd
import pytest

# Pro-gated: structure parsing needs biopython, rendering needs py3Dmol.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering_pro._backend.cpp_struct.view import LinkedView


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=30, chain="A"):
    lines = []
    serial = 1
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
    return _make_pdb(tmp_path_factory.mktemp("struct_linked") / "synthetic.pdb")


@pytest.fixture(scope="module")
def df_feat():
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
    return aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)


# --- normal cases ------------------------------------------------------------
class TestPlotLinked:
    """Normal cases — one parameter per test."""

    def test_returns_linked_view(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        assert isinstance(view, LinkedView)
        assert isinstance(view._repr_html_(), str)

    def test_width_height(self, pdb_path, df_feat):
        # the viewer panel size is configurable; the chosen px size reaches the rendered HTML
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact",
                                  width=600, height=500)
        assert "width:600px" in view._repr_html_() and "height:500px" in view._repr_html_()

    def test_html_has_viewer_and_link(self, pdb_path, df_feat):
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                               col_imp="feat_impact")._repr_html_()
        assert "createViewer" in h and "3Dmol-min.js" in h
        assert "function highlight" in h and "cpp-col" in h

    def test_columns_map_to_window_residues_in_order(self, pdb_path, df_feat):
        # 10 + 10 + 10 = 30 columns; start=1 -> residues 1..30, left-to-right (N-term first).
        h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=1,
                               col_imp="feat_impact")._repr_html_()
        resis = [int(x) for x in re.findall(r'data-resi="(\d+)"', h)]  # source order
        assert resis == list(range(1, 31))

    def test_unique_view_ids_across_calls(self, pdb_path, df_feat):
        # Two views on one page must not share DOM ids (else hover binds to the wrong panel).
        cpps_plot = _csp()
        h1 = cpps_plot.plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")._repr_html_()
        h2 = cpps_plot.plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")._repr_html_()
        ids1 = set(re.findall(r'id="(cppstruct_view_\w+)"', h1))
        ids2 = set(re.findall(r'id="(cppstruct_view_\w+)"', h2))
        assert ids1 and ids2 and ids1.isdisjoint(ids2)

    def test_start_shifts_columns(self, pdb_path, df_feat):
        # start=100 is outside the 30-residue synthetic structure -> warns, mapping still shifts.
        with pytest.warns(UserWarning):
            h = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, start=100,
                                   col_imp="feat_impact")._repr_html_()
        resis = sorted({int(x) for x in re.findall(r'data-resi="(\d+)"', h)})
        assert resis == list(range(100, 130))

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode_positive(self, pdb_path, df_feat, mode):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", mode=mode)
        assert view.mode == mode

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus_positive(self, pdb_path, df_feat, focus):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", focus=focus)
        assert isinstance(view, LinkedView)

    @pytest.mark.parametrize("size_by_impact", [True, False])
    def test_size_by_impact_positive(self, pdb_path, df_feat, size_by_impact):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", size_by_impact=size_by_impact)
        assert isinstance(view, LinkedView)

    @pytest.mark.parametrize("normalize_by_span", [True, False])
    def test_normalize_by_span_positive(self, pdb_path, df_feat, normalize_by_span):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", normalize_by_span=normalize_by_span)
        assert isinstance(view, LinkedView)

    def test_feature_map_kws_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", feature_map_kws={"name_test": "site"})
        assert isinstance(view, LinkedView)

    def test_width_height_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", width=600, height=500)
        assert isinstance(view, LinkedView)

    def test_feature_map_dpi_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                  col_imp="feat_impact", feature_map_dpi=120)
        assert isinstance(view, LinkedView)

    def test_part_sequences_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact",
                                  tmd_seq="A" * 10, jmd_n_seq="A" * 10, jmd_c_seq="A" * 10)
        assert isinstance(view, LinkedView)

    def test_chain_and_sequence_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact",
                                  chain="A", sequence="A" * 30)
        assert isinstance(view, LinkedView)

    def test_focus_region_positive(self, pdb_path, df_feat):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact",
                                  focus="zoom", focus_region=(11, 20))
        assert isinstance(view, LinkedView)

    # --- negatives -----------------------------------------------------------
    @pytest.mark.parametrize("mode", ["Impact", "shap", ""])
    def test_mode_negative(self, pdb_path, df_feat, mode):
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, mode=mode)

    def test_pdb_and_uniprot_both_negative(self, pdb_path, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, uniprot="P05067", tmd_len=10)

    def test_pdb_and_uniprot_neither_negative(self, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df_feat, tmd_len=10)

    def test_df_feat_missing_col_imp_negative(self, pdb_path, df_feat):
        df = df_feat.drop(columns=["feat_impact"])
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")

    @pytest.mark.parametrize("key", ["col_val", "col_imp", "tmd_len", "df_feat"])
    def test_feature_map_kws_collision_negative(self, pdb_path, df_feat, key):
        with pytest.raises(ValueError):
            _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                               feature_map_kws={key: 1})


# --- combinations & edge interactions ----------------------------------------
class TestPlotLinkedComplex:
    """write_html, the AlphaFold-fetch path, and the py3Dmol gate."""

    def test_write_html(self, pdb_path, df_feat, tmp_path):
        view = _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        out = tmp_path / "linked.html"
        view.write_html(str(out))
        text = out.read_text(encoding="utf-8")
        assert out.stat().st_size > 0
        assert "createViewer" in text and "cpp-col" in text and "<!DOCTYPE html>" in text

    def test_uniprot_fetch_path_mocked(self, df_feat, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch(self, df_seq=None, out_folder=None, **kwargs):
            import pathlib
            p = pathlib.Path(out_folder) / "AF-Q9NQ76-F1-model_v4.pdb"
            _make_pdb(p)
            return pd.DataFrame({"entry": ["Q9NQ76"], "model_path": [str(p)]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch)
        view = _csp().plot_linked(df_feat=df_feat, uniprot="Q9NQ76", tmd_len=10,
                                  col_imp="feat_impact")
        assert isinstance(view, LinkedView)

    def test_missing_py3dmol_raises_friendly(self, pdb_path, df_feat, monkeypatch):
        from aaanalysis.feature_engineering_pro import _cpp_structure_plot as cpps_plot_mod
        monkeypatch.setattr(cpps_plot_mod, "py3dmol_available", lambda: False)
        with pytest.raises(RuntimeError, match="py3Dmol"):
            _csp().plot_linked(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")


def test_plot_linked_in_public_api():
    assert hasattr(aa.CPPStructurePlot, "plot_linked")
