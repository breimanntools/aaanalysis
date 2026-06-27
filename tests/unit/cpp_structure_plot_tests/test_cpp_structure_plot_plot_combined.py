"""This is a script to test CPPStructurePlot.plot_combined()."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Pro-gated: structure parsing needs biopython. Skip the whole module cleanly
# when the pro extra is not installed.
pytest.importorskip("Bio")

import aaanalysis as aa
import aaanalysis.utils as ut


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=30, chain="A"):
    """Write a tiny synthetic PDB: one CA per residue, B-factor used as pLDDT."""
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
    p = tmp_path_factory.mktemp("struct_combined") / "synthetic.pdb"
    return _make_pdb(p)


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


# --- normal cases, one parameter per test ------------------------------------
class TestPlotCombined:
    """Normal cases — one parameter per test."""

    def test_returns_fig_and_two_axes(self, pdb_path, df_feat):
        fig, ax = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                       col_imp="feat_impact")
        assert isinstance(fig, plt.Figure)
        assert len(ax) == 2
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode_positive(self, pdb_path, df_feat, mode):
        fig, ax = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                       col_imp="feat_impact", mode=mode)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus_positive(self, pdb_path, df_feat, focus):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", focus=focus)
        plt.close(fig)

    @pytest.mark.parametrize("normalize_by_span", [True, False])
    def test_normalize_by_span_positive(self, pdb_path, df_feat, normalize_by_span):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact",
                                      normalize_by_span=normalize_by_span)
        plt.close(fig)

    @pytest.mark.parametrize("shap_plot", [True, False])
    def test_shap_plot_positive(self, pdb_path, df_feat, shap_plot):
        col_imp = "feat_impact" if shap_plot else "feat_importance"
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp=col_imp, shap_plot=shap_plot)
        plt.close(fig)

    def test_size_by_impact_positive(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", size_by_impact=False)
        plt.close(fig)

    def test_focus_region_positive(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", focus="zoom",
                                      focus_region=(11, 20))
        plt.close(fig)

    def test_part_sequences_positive(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", tmd_seq="A" * 10,
                                      jmd_n_seq="A" * 10, jmd_c_seq="A" * 10)
        plt.close(fig)

    def test_feature_map_kws_positive(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact",
                                      feature_map_kws={"name_test": "site"})
        plt.close(fig)

    def test_width_ratios_positive(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", width_ratios=(1, 2))
        plt.close(fig)

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
    """Combinations, savefig, and the AlphaFold-fetch path."""

    def test_savefig_png(self, pdb_path, df_feat, tmp_path):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact")
        out = tmp_path / "combined.png"
        fig.savefig(str(out), dpi=80)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    def test_savefig_pdf(self, pdb_path, df_feat, tmp_path):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact")
        out = tmp_path / "combined.pdf"
        fig.savefig(str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    def test_plddt_zoom_fade_combo(self, pdb_path, df_feat):
        fig, _ = _csp().plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                      col_imp="feat_impact", mode="plddt", focus="fade")
        plt.close(fig)

    def test_custom_df_scales_and_df_cat(self, pdb_path, df_feat):
        # Forwarding df_cat to the inner CPPPlot avoids the "scale ids missing in df_cat"
        # crash when the plotter is built with custom scales.
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10,
                                  df_scales=df_scales, df_cat=df_cat, verbose=False)
        fig, _ = csp.plot_combined(df_feat=df_feat, pdb=pdb_path, tmd_len=10,
                                   col_imp="feat_impact")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_uniprot_fetch_path_mocked(self, df_feat, tmp_path, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch(self, df_seq=None, out_folder=None, **kwargs):
            import pathlib
            p = pathlib.Path(out_folder) / "AF-Q9NQ76-F1-model_v4.pdb"
            _make_pdb(p)
            return pd.DataFrame({"entry": ["Q9NQ76"], "model_path": [str(p)]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch)
        fig, _ = _csp().plot_combined(df_feat=df_feat, uniprot="Q9NQ76", tmd_len=10,
                                      col_imp="feat_impact")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_combined_in_public_api():
    assert hasattr(aa.CPPStructurePlot, "plot_combined")
