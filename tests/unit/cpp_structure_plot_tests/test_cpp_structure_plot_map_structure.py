"""This is a script to test CPPStructurePlot.map_structure()."""
import matplotlib
matplotlib.use("Agg")
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

# Pro-gated: structure parsing needs biopython. Skip the whole module cleanly
# when the pro extra is not installed.
pytest.importorskip("Bio")

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

HAS_PY3DMOL = True
try:
    import py3Dmol  # noqa: F401
except ImportError:
    HAS_PY3DMOL = False


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=40, start_resi=1, chain="A"):
    """Write a tiny synthetic PDB: one CA per residue, B-factor used as pLDDT."""
    lines = []
    serial = 1
    for i in range(n):
        resi = start_resi + i
        x, y, z = i * 1.5, np.sin(i * 0.5) * 5, np.cos(i * 0.5) * 5
        b = 40 + (i % 60)  # pLDDT-ish 40..99
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


# --- normal cases, one parameter per test ------------------------------------
class TestMapStructure:
    """Normal cases — one parameter per test."""

    def test_returns_structure_view(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="mpl")
        assert view.backend == "mpl"
        assert hasattr(view, "show") and hasattr(view, "write_html")

    def test_df_feat_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="mpl")
        assert isinstance(view.dict_impact, dict) and len(view.dict_impact) > 0

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode_positive(self, pdb_path, mode):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 mode=mode, backend="mpl")
        assert view.mode == mode

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus_positive(self, pdb_path, focus):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus=focus, backend="mpl")
        assert view.backend == "mpl"

    def test_backend_mpl_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="mpl")
        assert view.backend == "mpl"

    @pytest.mark.skipif(not HAS_PY3DMOL, reason="py3Dmol not installed")
    def test_backend_py3dmol_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="py3dmol")
        assert view.backend == "py3dmol"

    def test_backend_none_autoselects(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend=None)
        assert view.backend in ("py3dmol", "mpl")

    def test_col_imp_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        df = _df_feat(col_imp="feat_impact_P05067")
        view = csp.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10,
                                 col_imp="feat_impact_P05067", backend="mpl")
        assert view.max_abs > 0

    @settings(max_examples=5, deadline=None)
    @given(tmd_len=some.integers(min_value=1, max_value=20))
    def test_tmd_len_positive(self, pdb_path, tmd_len):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=tmd_len, backend="mpl")
        assert view.backend == "mpl"

    @settings(max_examples=5, deadline=None)
    @given(start=some.integers(min_value=1, max_value=15))
    def test_start_positive(self, pdb_path, start):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 start=start, backend="mpl")
        # The smallest mapped residue equals start (first JMD-N residue).
        assert min(view.dict_impact) == start

    @pytest.mark.parametrize("size_by_impact", [True, False])
    def test_size_by_impact_positive(self, pdb_path, size_by_impact):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 size_by_impact=size_by_impact, backend="mpl")
        assert view.backend == "mpl"

    def test_chain_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 chain="A", backend="mpl")
        assert view.backend == "mpl"

    def test_sequence_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 sequence="A" * 40, backend="mpl")
        assert view.backend == "mpl"

    def test_focus_region_tuple_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="zoom", focus_region=(11, 20), backend="mpl")
        assert view.backend == "mpl"

    def test_focus_region_list_positive(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="fade", focus_region=[(1, 10), (11, 20)], backend="mpl")
        assert view.backend == "mpl"

    # --- negatives -----------------------------------------------------------
    @pytest.mark.parametrize("mode", ["Impact", "shap", "", "color"])
    def test_mode_negative(self, pdb_path, mode):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, mode=mode)

    @pytest.mark.parametrize("focus", ["all", "Fade", "near", ""])
    def test_focus_negative(self, pdb_path, focus):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, focus=focus)

    @pytest.mark.parametrize("backend", ["py3DMOL", "matplotlib", "ngl", ""])
    def test_backend_negative(self, pdb_path, backend):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend=backend)

    @pytest.mark.parametrize("tmd_len", [0, -1, -5])
    def test_tmd_len_negative(self, pdb_path, tmd_len):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=tmd_len)

    @pytest.mark.parametrize("start", [-1, -10])
    def test_start_negative(self, pdb_path, start):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, start=start)

    def test_df_feat_missing_col_imp_negative(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        df = _df_feat().drop(columns=["feat_impact"])
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10)

    def test_pdb_and_uniprot_both_negative(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, uniprot="P05067", tmd_len=10)

    def test_pdb_and_uniprot_neither_negative(self):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), tmd_len=10)

    @pytest.mark.parametrize("focus_region", [(20, 11), (5, 1)])
    def test_focus_region_negative(self, pdb_path, focus_region):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                              focus="zoom", focus_region=focus_region)

    def test_chain_unknown_negative(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, chain="Z")


# --- combinations & edge interactions ----------------------------------------
class TestMapStructureComplex:
    """Combinations and edge interactions."""

    def test_plddt_zoom_focus_region(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 mode="plddt", focus="zoom", focus_region=(11, 20),
                                 backend="mpl")
        assert view.mode == "plddt"

    def test_impact_fade_size_by_impact(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 mode="impact", focus="fade", size_by_impact=True,
                                 backend="mpl")
        assert view.backend == "mpl"

    def test_jmd_lengths_shift_window(self, pdb_path):
        csp = aa.CPPStructurePlot(jmd_n_len=5, jmd_c_len=5, verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 start=1, backend="mpl")
        # JMD_N now spans residues 1..5; TMD spans 6..15.
        assert view.dict_impact.get(6, 0) != 0  # first TMD residue carries impact

    @settings(max_examples=5, deadline=None)
    @given(start=some.integers(min_value=100, max_value=300))
    def test_start_offsets_into_structure(self, pdb_path, start):
        csp = aa.CPPStructurePlot(verbose=False)
        # start far outside the 1..40 structure -> warns, still returns a view.
        with pytest.warns(UserWarning):
            view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                     start=start, backend="mpl")
        assert view.backend == "mpl"

    def test_savefig_writes_png(self, pdb_path, tmp_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="mpl")
        out = tmp_path / "v.png"
        view.savefig(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_write_html_mpl_embeds_png(self, pdb_path, tmp_path):
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="mpl")
        out = tmp_path / "v.html"
        view.write_html(str(out))
        assert "base64" in out.read_text(encoding="utf-8")

    def test_savefig_on_py3dmol_raises(self, pdb_path, tmp_path):
        if not HAS_PY3DMOL:
            pytest.skip("py3Dmol not installed")
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10, backend="py3dmol")
        with pytest.raises(RuntimeError):
            view.savefig(str(tmp_path / "x.png"))

    def test_single_feature_df(self, pdb_path):
        csp = aa.CPPStructurePlot(verbose=False)
        df = pd.DataFrame({ut.COL_FEATURE: ["TMD-Segment(1,1)-ABC"],
                           ut.COL_CAT: ["Polarity"], "feat_impact": [1.0]})
        view = csp.map_structure(df_feat=df, pdb=pdb_path, tmd_len=10, backend="mpl")
        assert view.max_abs > 0


# --- golden values (hand-computed normalized perpos) -------------------------
class TestMapStructureGoldenValues:
    """Hand-computed per-residue impact == normalized-sum mapping."""

    def test_whole_tmd_segment_normalized(self, pdb_path):
        # TMD Segment(1,1) spans the whole 10-residue TMD (residues 11..20 with
        # start=1, jmd_n_len=10). Impact 0.8 spread over 10 positions -> 0.08 each.
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = csp.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=0.0),
                                 pdb=pdb_path, tmd_len=10, start=1, backend="mpl")
        for resi in range(11, 21):
            assert view.dict_impact[resi] == pytest.approx(0.08)

    def test_whole_jmd_segment_normalized(self, pdb_path):
        # JMD_N Segment(1,1) spans residues 1..10; impact -0.4 / 10 -> -0.04 each.
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = csp.map_structure(df_feat=_df_feat(tmd_impact=0.0, jmd_impact=-0.4),
                                 pdb=pdb_path, tmd_len=10, start=1, backend="mpl")
        for resi in range(1, 11):
            assert view.dict_impact[resi] == pytest.approx(-0.04)

    def test_max_abs_is_max_per_residue(self, pdb_path):
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        view = csp.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=-0.4),
                                 pdb=pdb_path, tmd_len=10, start=1, backend="mpl")
        assert view.max_abs == pytest.approx(0.08)

    def test_start_shifts_absolute_residues(self, pdb_path):
        # start=312 -> JMD_N spans 312..321, TMD spans 322..331.
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        with pytest.warns(UserWarning):  # window outside the 1..40 synthetic structure
            view = csp.map_structure(df_feat=_df_feat(tmd_impact=0.8, jmd_impact=0.0),
                                     pdb=pdb_path, tmd_len=10, start=312, backend="mpl")
        assert view.dict_impact[322] == pytest.approx(0.08)
        assert min(view.dict_impact) == 312


# --- constructor -------------------------------------------------------------
class TestCPPStructurePlotInit:
    """Constructor parameter checks."""

    @settings(max_examples=5, deadline=None)
    @given(jmd_n_len=some.integers(min_value=0, max_value=20))
    def test_jmd_n_len_positive(self, jmd_n_len):
        csp = aa.CPPStructurePlot(jmd_n_len=jmd_n_len, verbose=False)
        assert csp._jmd_n_len == jmd_n_len

    @settings(max_examples=5, deadline=None)
    @given(jmd_c_len=some.integers(min_value=0, max_value=20))
    def test_jmd_c_len_positive(self, jmd_c_len):
        csp = aa.CPPStructurePlot(jmd_c_len=jmd_c_len, verbose=False)
        assert csp._jmd_c_len == jmd_c_len

    def test_df_scales_positive(self):
        df_scales = aa.load_scales()
        csp = aa.CPPStructurePlot(df_scales=df_scales, verbose=False)
        assert csp._df_scales is not None

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_positive(self, verbose):
        csp = aa.CPPStructurePlot(verbose=verbose)
        assert csp._verbose == verbose

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
    """Multi-chain selection, identity warning, and zoom robustness (review fixes)."""

    def test_extract_returns_selected_chain_id(self, tmp_path):
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.structure import (
            load_structure, extract_chain_residues)
        pdb = _make_multichain_pdb(tmp_path / "multi.pdb")
        structure = load_structure(pdb)
        records, identity, chain_id = extract_chain_residues(structure, chain="B")
        assert chain_id == "B"
        # All records belong to chain B (its x-coords are offset by +100).
        assert all(r["coord"][0] >= 100 for r in records)

    def test_multichain_render_uses_chain_qualifier(self, tmp_path):
        # Regression: residue selections must be chain-qualified so chain B's
        # residues are not painted onto chain A's same-numbered residues.
        csp = aa.CPPStructurePlot(verbose=False)
        pdb = _make_multichain_pdb(tmp_path / "multi.pdb")
        if not HAS_PY3DMOL:
            pytest.skip("py3Dmol not installed")
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb, tmd_len=2, start=1,
                                 chain="B", backend="py3dmol")
        assert view.backend == "py3dmol"

    def test_explicit_chain_sequence_mismatch_warns(self, pdb_path):
        # Explicit chain + a non-matching sequence should still warn (identity < 0.5).
        csp = aa.CPPStructurePlot(verbose=False)
        with pytest.warns(UserWarning):
            csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                              chain="A", sequence="W" * 40, backend="mpl")

    def test_zoom_region_outside_structure_no_error(self, pdb_path):
        # zoomTo on residues absent from the structure must not raise.
        csp = aa.CPPStructurePlot(verbose=False)
        view = csp.map_structure(df_feat=_df_feat(), pdb=pdb_path, tmd_len=10,
                                 focus="zoom", focus_region=(900, 950), backend="mpl")
        assert view.backend == "mpl"


def test_impact_color_matches_package_shap_ramp():
    # The impact ramp reuses the package SHAP palette (white -> SHAP_POS / NEG).
    from aaanalysis.feature_engineering_pro._backend.cpp_struct.colors import impact_to_hex
    import matplotlib.colors as mcolors
    pos = mcolors.to_hex(impact_to_hex(1.0, 1.0))
    neg = mcolors.to_hex(impact_to_hex(-1.0, 1.0))
    assert pos.lower() == ut.COLOR_SHAP_POS.lower()
    assert neg.lower() == ut.COLOR_SHAP_NEG.lower()
    assert impact_to_hex(0.0, 1.0) == "#FFFFFF"


def test_cpp_structure_plot_in_public_api():
    assert "CPPStructurePlot" in aa.__all__
