"""This is a script to test CPPStructurePlot.map_structure() (py3Dmol-only)."""
import matplotlib
matplotlib.use("Agg")
import pathlib
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

# Pro-gated: structure parsing needs biopython, rendering needs py3Dmol.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=40, start_resi=1, chain="A"):
    """Write a tiny synthetic PDB: one CA per residue, B-factor used as pLDDT."""
    lines = []
    serial = 1
    for i in range(n):
        resi = start_resi + i
        x, y, z = i * 1.5, np.sin(i * 0.5) * 5, np.cos(i * 0.5) * 5
        b = 40 + (i % 60)
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA {chain}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def pdb_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("struct") / "synthetic.pdb"
    return _make_pdb(p)


def _df_feat(tmd_impact=0.8, jmd_impact=-0.4, col_imp="feat_impact"):
    """Two known features: a whole-TMD segment and a whole-JMD-N segment."""
    return pd.DataFrame({
        ut.COL_FEATURE: ["TMD-Segment(1,1)-ABC", "JMD_N-Segment(1,1)-XYZ"],
        ut.COL_CAT: ["Polarity", "Energy"],
        col_imp: [tmd_impact, jmd_impact],
    })


# --- normal cases ------------------------------------------------------------
class TestMapStructure:
    """Normal cases — one parameter per test."""

    def test_returns_structure_view(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        assert view.backend == "py3dmol"
        assert hasattr(view, "show") and hasattr(view, "write_html")

    def test_df_feat_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        assert isinstance(view.dict_impact, dict) and len(view.dict_impact) > 0

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode_positive(self, pdb_path, mode):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, mode=mode)
        assert view.mode == mode

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus_positive(self, pdb_path, focus):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, focus=focus)
        assert view.backend == "py3dmol"

    def test_col_imp_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        df = _df_feat(col_imp="feat_impact_P05067")
        view = cpps_plot.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10,
                                 col_imp="feat_impact_P05067")
        assert view.max_abs > 0

    @settings(max_examples=5, deadline=None)
    @given(tmd_len=some.integers(min_value=1, max_value=20))
    def test_tmd_len_positive(self, pdb_path, tmd_len):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=tmd_len)
        assert view.backend == "py3dmol"

    @settings(max_examples=5, deadline=None)
    @given(start=some.integers(min_value=1, max_value=15))
    def test_start_positive(self, pdb_path, start):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, start=start)
        assert min(view.dict_impact) == start

    @pytest.mark.parametrize("size_by_impact", [True, False])
    def test_size_by_impact_positive(self, pdb_path, size_by_impact):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 size_by_impact=size_by_impact)
        assert view.backend == "py3dmol"

    @pytest.mark.parametrize("normalize_by_span", [True, False])
    def test_normalize_by_span_positive(self, pdb_path, normalize_by_span):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 normalize_by_span=normalize_by_span)
        assert view.backend == "py3dmol"

    def test_chain_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, chain="A")
        assert view.backend == "py3dmol"

    def test_sequence_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 sequence="A" * 40)
        assert view.backend == "py3dmol"

    def test_focus_region_tuple_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="zoom", focus_region=(11, 20))
        assert view.backend == "py3dmol"

    def test_focus_region_list_positive(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="fade", focus_region=[(1, 10), (11, 20)])
        assert view.backend == "py3dmol"

    # --- negatives -----------------------------------------------------------
    @pytest.mark.parametrize("mode", ["Impact", "shap", "", "color"])
    def test_mode_negative(self, pdb_path, mode):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, mode=mode)

    @pytest.mark.parametrize("focus", ["all", "Fade", "near", ""])
    def test_focus_negative(self, pdb_path, focus):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, focus=focus)

    @pytest.mark.parametrize("tmd_len", [0, -1, -5])
    def test_tmd_len_negative(self, pdb_path, tmd_len):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=tmd_len)

    @pytest.mark.parametrize("start", [-1, -10])
    def test_start_negative(self, pdb_path, start):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, start=start)

    def test_df_feat_missing_col_imp_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        df = _df_feat().drop(columns=["feat_impact"])
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10)

    def test_pdb_and_uniprot_both_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, uniprot="P05067", tmd_len=10)

    def test_pdb_and_uniprot_neither_negative(self):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), tmd_len=10)

    @pytest.mark.parametrize("focus_region", [(20, 11), (5, 1)])
    def test_focus_region_negative(self, pdb_path, focus_region):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                              focus="zoom", focus_region=focus_region)

    def test_chain_unknown_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, chain="Z")


# --- combinations & edge interactions ----------------------------------------
class TestMapStructureComplex:
    """Combinations, HTML export, and edge interactions."""

    def test_plddt_zoom_focus_region(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 mode="plddt", focus="zoom", focus_region=(11, 20))
        assert view.mode == "plddt"

    def test_jmd_lengths_shift_window(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=5, jmd_c_len=5, verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, start=1)
        assert view.dict_impact.get(6, 0) != 0  # first TMD residue carries impact

    @settings(max_examples=5, deadline=None)
    @given(start=some.integers(min_value=100, max_value=300))
    def test_start_offsets_into_structure(self, pdb_path, start):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.warns(UserWarning):  # window outside the 1..40 synthetic structure
            view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, start=start)
        assert view.backend == "py3dmol"

    def test_write_html_writes_file(self, pdb_path, tmp_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        out = tmp_path / "v.html"
        view.write_html(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_repr_html_is_str(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        assert isinstance(view._repr_html_(), str)

    def test_single_feature_df(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        df = pd.DataFrame({ut.COL_FEATURE: ["TMD-Segment(1,1)-ABC"],
                           ut.COL_CAT: ["Polarity"], "feat_impact": [1.0]})
        view = cpps_plot.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10)
        assert view.max_abs > 0


# --- golden values (hand-computed per-residue impact) ------------------------
class TestMapStructureGoldenValues:
    """Hand-computed per-residue impact for both aggregation modes."""

    def test_whole_tmd_segment_app_exact_default(self, pdb_path):
        # Default (app-exact, no divide): TMD Segment(1,1) spans residues 11..20; its full
        # impact 0.8 lands on every spanned residue, matching the deployed app.
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=0.0),
                                 pdb=pdb_path, tmd_len=10, start=1)
        for resi in range(11, 21):
            assert view.dict_impact[resi] == pytest.approx(0.8)
        assert view.max_abs == pytest.approx(0.8)

    def test_whole_tmd_segment_normalize_by_span(self, pdb_path):
        # normalize_by_span=True: impact 0.8 spread over 10 positions -> 0.08 each.
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=0.0),
                                 pdb=pdb_path, tmd_len=10, start=1, normalize_by_span=True)
        for resi in range(11, 21):
            assert view.dict_impact[resi] == pytest.approx(0.08)
        assert view.max_abs == pytest.approx(0.08)

    def test_max_abs_is_max_per_residue(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=-0.4),
                                 pdb=pdb_path, tmd_len=10, start=1)
        assert view.max_abs == pytest.approx(0.8)  # default no-divide: max(0.8, 0.4)

    def test_start_shifts_absolute_residues(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        with pytest.warns(UserWarning):  # window outside the 1..40 synthetic structure
            view = cpps_plot.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=0.0),
                                     pdb=pdb_path, tmd_len=10, start=312)
        assert view.dict_impact[322] == pytest.approx(0.8)
        assert min(view.dict_impact) == 312

    def test_nan_impact_does_not_poison_residue(self, pdb_path):
        df = pd.DataFrame({
            ut.COL_FEATURE: ["TMD-Segment(1,1)-ABC", "TMD-Segment(1,1)-XYZ"],
            ut.COL_CAT: ["Polarity", "Energy"],
            "feat_impact": [1.0, float("nan")],
        })
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = cpps_plot.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10, start=1)
        assert view.dict_impact[11] == pytest.approx(1.0)
        assert view.max_abs == pytest.approx(1.0)


# --- constructor -------------------------------------------------------------
class TestCPPStructurePlotInit:
    """Constructor parameter checks."""

    @settings(max_examples=5, deadline=None)
    @given(jmd_n_len=some.integers(min_value=0, max_value=20))
    def test_jmd_n_len_positive(self, jmd_n_len):
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=jmd_n_len, verbose=False)
        assert cpps_plot._jmd_n_len == jmd_n_len

    @settings(max_examples=5, deadline=None)
    @given(jmd_c_len=some.integers(min_value=0, max_value=20))
    def test_jmd_c_len_positive(self, jmd_c_len):
        cpps_plot = aa.CPPStructurePlot(jmd_c_len=jmd_c_len, verbose=False)
        assert cpps_plot._jmd_c_len == jmd_c_len

    def test_df_scales_positive(self):
        cpps_plot = aa.CPPStructurePlot(df_scales=aa.load_scales(), verbose=False)
        assert cpps_plot._df_scales is not None

    def test_df_cat_positive(self):
        cpps_plot = aa.CPPStructurePlot(df_cat=aa.load_scales(name="scales_cat"), verbose=False)
        assert cpps_plot._df_cat is not None

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_positive(self, verbose):
        cpps_plot = aa.CPPStructurePlot(verbose=verbose)
        assert cpps_plot._verbose == verbose

    @pytest.mark.parametrize("jmd_n_len", [-1, -5])
    def test_jmd_n_len_negative(self, jmd_n_len):
        with pytest.raises(ValueError):
            aa.CPPStructurePlot(jmd_n_len=jmd_n_len)

    @pytest.mark.parametrize("jmd_c_len", [-1, -5])
    def test_jmd_c_len_negative(self, jmd_c_len):
        with pytest.raises(ValueError):
            aa.CPPStructurePlot(jmd_c_len=jmd_c_len)


def _make_multichain_pdb(path, chains=("A", "B"), n=12):
    """Two chains sharing residue numbers 1..n (different coords) to test chain leak."""
    lines = []
    serial = 1
    for ci, ch in enumerate(chains):
        for i in range(n):
            resi = i + 1
            x, y, z = i * 1.5 + ci * 100, ci * 50.0, 0.0
            b = 50 + i
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {ch}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestMapStructureChainHandling:
    """Multi-chain selection, identity warning, and zoom robustness."""

    def test_extract_returns_selected_chain_id(self, tmp_path):
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.structure import (
            load_structure, extract_chain_residues)
        pdb = _make_multichain_pdb(tmp_path / "multi.pdb")
        structure = load_structure(pdb)
        records, identity, chain_id = extract_chain_residues(structure, chain="B")
        assert chain_id == "B"
        assert all(r["coord"][0] >= 100 for r in records)

    def test_multichain_render_uses_chain_qualifier(self, tmp_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        pdb = _make_multichain_pdb(tmp_path / "multi.pdb")
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb, tmd_len=2, start=1, chain="B")
        assert view.backend == "py3dmol"

    def test_explicit_chain_sequence_mismatch_warns(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.warns(UserWarning):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                              chain="A", sequence="W" * 40)

    def test_zoom_region_outside_structure_no_error(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="zoom", focus_region=(900, 950))
        assert view.backend == "py3dmol"


def test_impact_color_matches_package_shap_ramp():
    from aaanalysis.feature_engineering_pro._backend.cpp_struct.colors import impact_to_hex
    import matplotlib.colors as mcolors
    pos = mcolors.to_hex(impact_to_hex(1.0, 1.0))
    neg = mcolors.to_hex(impact_to_hex(-1.0, 1.0))
    assert pos.lower() == ut.COLOR_SHAP_POS.lower()
    assert neg.lower() == ut.COLOR_SHAP_NEG.lower()
    assert impact_to_hex(0.0, 1.0) == "#FFFFFF"


class TestMapStructureCoverage:
    """Verbose, AlphaFold-fetch, py3Dmol view, and the missing-py3Dmol gate."""

    def test_verbose_prints_summary(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=True)
        cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        assert cpps_plot._verbose is True

    def test_uniprot_fetch_path_mocked(self, tmp_path, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch(self, df_seq=None, out_folder=None, **kwargs):
            p = pathlib.Path(out_folder) / "AF-Q9NQ76-F1-model_v4.pdb"
            _make_pdb(p)
            return pd.DataFrame({"entry": ["Q9NQ76"], "model_path": [str(p)]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch)
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), uniprot="Q9NQ76", tmd_len=10)
        assert view.backend == "py3dmol"

    def test_uniprot_fetch_missing_model_raises(self, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch_nan(self, df_seq=None, out_folder=None, **kwargs):
            return pd.DataFrame({"entry": ["X"], "model_path": [float("nan")]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch_nan)
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(RuntimeError):
            cpps_plot.map_structure(df_feat=_df_feat(), uniprot="X", tmd_len=10)

    def test_py3dmol_view_methods(self, pdb_path, tmp_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        view = cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)
        assert isinstance(view._repr_html_(), str)
        out = tmp_path / "v.html"
        view.write_html(str(out))
        assert out.exists() and out.stat().st_size > 0
        view.show()

    def test_missing_py3dmol_raises_friendly(self, pdb_path, monkeypatch):
        # Force py3Dmol "absent" -> friendly RuntimeError, not a bare ImportError.
        from aaanalysis.feature_engineering_pro import _cpp_structure_plot as cpps_plot_mod
        monkeypatch.setattr(cpps_plot_mod, "py3dmol_available", lambda: False)
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(RuntimeError, match="py3Dmol"):
            cpps_plot.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10)


def test_colors_helpers_cover_edges():
    from aaanalysis.feature_engineering_pro._backend.cpp_struct import colors as C
    assert C.perceptual_transform(0.25) == pytest.approx(0.5)
    assert C.perceptual_transform(-0.25) == pytest.approx(-0.5)
    custom = C.impact_to_hex(1.0, 1.0, color_pos="#00FF00", color_neg="#FF00FF")
    assert custom.lower() == "#00ff00"
    assert C.impact_to_hex(-1.0, 1.0, color_neg="#FF00FF").lower() == "#ff00ff"
    assert C.plddt_to_hex(float("nan")) == ut.COLOR_STRUCT_MISSING
    assert C.plddt_to_hex(95).startswith("#")
    assert C.color_for_residue(5, {5: 0.5}, 1.0, 90.0, "plddt").startswith("#")
    assert C.color_for_residue(5, {5: 0.5}, 1.0, 90.0, "impact").startswith("#")


def test_cpp_structure_plot_in_public_api():
    assert "CPPStructurePlot" in aa.__all__
