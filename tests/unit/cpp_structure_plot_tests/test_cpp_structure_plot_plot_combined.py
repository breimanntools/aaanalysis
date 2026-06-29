"""This is a script to test CPPStructurePlot.plot_combined() (py3Dmol + feature_map)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

# Pro-gated: structure parsing needs biopython, rendering needs py3Dmol.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering_pro._backend.cpp_struct.view import CombinedView


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
    return _make_pdb(tmp_path_factory.mktemp("struct_combined") / "synthetic.pdb")


@pytest.fixture(scope="module")
def df_feat():
    """A complete CPP-style df_feat (real scales/categories) for the feature map."""
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
    df["abs_auc"] = [0.2, 0.15, 0.3, 0.1, 0.25]
    df["abs_mean_dif"] = [0.3, 0.2, 0.5, 0.4, 0.35]
    df["mean_dif"] = [0.3, -0.2, 0.5, -0.4, 0.25]
    df["std_test"] = 0.1
    df["std_ref"] = 0.1
    df["feat_impact"] = [0.8, -0.5, 1.2, -0.3, 0.6]
    df["feat_importance"] = [0.8, 0.5, 1.2, 0.3, 0.6]
    return df


def _csp():
    return aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)


# --- normal cases ------------------------------------------------------------
class TestPlotCombined:
    """Normal cases — one parameter per test."""

    def test_returns_combined_view(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact")
        assert isinstance(view, CombinedView)
        assert view.view is not None
        assert isinstance(view._repr_html_(), str)

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode_positive(self, pdb_path, df_feat, mode):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", mode=mode)
        assert view.mode == mode

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus_positive(self, pdb_path, df_feat, focus):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", focus=focus)
        assert isinstance(view, CombinedView)

    @pytest.mark.parametrize("normalize_by_span", [True, False])
    def test_normalize_by_span_positive(self, pdb_path, df_feat, normalize_by_span):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact",
                                    normalize_by_span=normalize_by_span)
        assert isinstance(view, CombinedView)

    @pytest.mark.parametrize("shap_plot", [True, False])
    def test_shap_plot_positive(self, pdb_path, df_feat, shap_plot):
        col_imp = "feat_impact" if shap_plot else "feat_importance"
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp=col_imp, shap_plot=shap_plot)
        assert isinstance(view, CombinedView)

    def test_size_by_impact_positive(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", size_by_impact=False)
        assert isinstance(view, CombinedView)

    def test_focus_region_positive(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", focus="zoom", focus_region=(11, 20))
        assert isinstance(view, CombinedView)

    def test_focus_region_negative(self, pdb_path, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                 col_imp="feat_impact", focus_region=(20, 11))  # start > stop

    def test_part_sequences_positive(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", tmd_seq="A" * 10,
                                    jmd_n_seq="A" * 10, jmd_c_seq="A" * 10)
        assert isinstance(view, CombinedView)

    def test_feature_map_kws_positive(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact",
                                    feature_map_kws={"name_test": "site"})
        assert isinstance(view, CombinedView)

    def test_feature_map_dpi_positive(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", feature_map_dpi=120)
        assert isinstance(view, CombinedView)

    # --- negatives -----------------------------------------------------------
    @pytest.mark.parametrize("mode", ["Impact", "shap", ""])
    def test_mode_negative(self, pdb_path, df_feat, mode):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, mode=mode)

    @pytest.mark.parametrize("focus", ["all", "Fade", ""])
    def test_focus_negative(self, pdb_path, df_feat, focus):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, focus=focus)

    @pytest.mark.parametrize("tmd_len", [0, -1])
    def test_tmd_len_negative(self, pdb_path, df_feat, tmd_len):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=tmd_len)

    def test_pdb_and_uniprot_both_negative(self, pdb_path, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, uniprot="P05067", tmd_len=10)

    def test_pdb_and_uniprot_neither_negative(self, df_feat):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, tmd_len=10)

    def test_df_feat_missing_col_imp_negative(self, pdb_path, df_feat):
        df = df_feat.drop(columns=["feat_impact"])
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")

    @pytest.mark.parametrize("key", ["col_val", "col_imp", "tmd_len", "df_feat", "shap_plot"])
    def test_feature_map_kws_collision_negative(self, pdb_path, df_feat, key):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                 feature_map_kws={key: 1})

    @pytest.mark.parametrize("feature_map_dpi", [0, 10, -5])
    def test_feature_map_dpi_negative(self, pdb_path, df_feat, feature_map_dpi):
        with pytest.raises(ValueError):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                 feature_map_dpi=feature_map_dpi)


# --- combinations & edge interactions ----------------------------------------
class TestPlotCombinedComplex:
    """write_html, custom scales, the AlphaFold-fetch path, and the py3Dmol gate."""

    def test_write_html(self, pdb_path, df_feat, tmp_path):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact")
        out = tmp_path / "combined.html"
        view.write_html(str(out))
        text = out.read_text(encoding="utf-8")
        assert out.stat().st_size > 0
        assert "base64" in text  # the feature-map image is embedded

    def test_plddt_fade_combo(self, pdb_path, df_feat):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                    col_imp="feat_impact", mode="plddt", focus="fade")
        assert view.mode == "plddt"

    def test_custom_df_scales_and_df_cat(self, pdb_path, df_feat):
        # Forwarding df_cat to the inner CPPPlot avoids the "scale ids missing in df_cat" crash.
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10,
                                  df_scales=aa.load_scales(),
                                  df_cat=aa.load_scales(name="scales_cat"), verbose=False)
        view = csp.plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        assert isinstance(view, CombinedView)

    def test_uniprot_fetch_path_mocked(self, df_feat, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch(self, df_seq=None, out_folder=None, **kwargs):
            import pathlib
            p = pathlib.Path(out_folder) / "AF-Q9NQ76-F1-model_v4.pdb"
            _make_pdb(p)
            return pd.DataFrame({"entry": ["Q9NQ76"], "model_path": [str(p)]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch)
        view = _csp().plot_combined(df_feat=df_feat, uniprot="Q9NQ76", tmd_len=10,
                                    col_imp="feat_impact")
        assert isinstance(view, CombinedView)

    def test_missing_py3dmol_raises_friendly(self, pdb_path, df_feat, monkeypatch):
        from aaanalysis.feature_engineering_pro import _cpp_structure_plot as csp_mod
        monkeypatch.setattr(csp_mod, "py3dmol_available", lambda: False)
        with pytest.raises(RuntimeError, match="py3Dmol"):
            _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")


def test_plot_combined_in_public_api():
    assert hasattr(aa.CPPStructurePlot, "plot_combined")


# --- Stage D: static capture (CombinedView.savefig) --------------------------
class TestSavefig:
    """CombinedView.savefig writes the feature-map panel as a static image."""

    @pytest.mark.parametrize("ext,magic", [("png", b"\x89PNG"), ("pdf", b"%PDF-")])
    def test_savefig_formats(self, pdb_path, df_feat, tmp_path, ext, magic):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        out = tmp_path / f"fig.{ext}"
        ret = view.savefig(str(out))
        assert ret == str(out)
        assert out.is_file() and out.read_bytes()[:5].startswith(magic)

    def test_savefig_jpg_flattens_alpha(self, pdb_path, df_feat, tmp_path):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        out = tmp_path / "fig.jpg"
        view.savefig(str(out))
        assert out.is_file() and out.stat().st_size > 0

    def test_savefig_dpi(self, pdb_path, df_feat, tmp_path):
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        out = tmp_path / "fig_dpi.png"
        view.savefig(str(out), dpi=300)
        from PIL import Image
        # PNG stores DPI as pixels/metre, so it round-trips with sub-unit rounding (~299.999)
        assert round(Image.open(str(out)).info.get("dpi", (0,))[0]) == 300

    def test_savefig_no_feature_map_raises(self, pdb_path, df_feat, tmp_path):
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.view import CombinedView
        view = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10, col_imp="feat_impact")
        empty = CombinedView(view=view.view, feature_map_png_b64=None)
        with pytest.raises(RuntimeError, match="no feature-map"):
            empty.savefig(str(tmp_path / "x.png"))
