"""This is a script to test CPPStructurePlot.explore() (integrated predictor + output dispatch)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

# Pro-gated: structure parsing needs biopython, rendering needs py3Dmol, impact needs SHAP.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")
pytest.importorskip("shap")

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering_pro._backend.cpp_struct.view import CombinedView, LinkedView

_AAS = "ACDEFGHIKLMNPQRSTVWY"


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=30, chain="A"):
    lines, serial = [], 1
    for i in range(n):
        x, y, z = i * 1.5, np.sin(i * 0.5) * 6, np.cos(i * 0.5) * 6
        b = 40 + (i % 60)
        lines.append(f"ATOM  {serial:5d}  CA  ALA {chain}{i + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def pdb_path(tmp_path_factory):
    return _make_pdb(tmp_path_factory.mktemp("struct_explore") / "synthetic.pdb")


@pytest.fixture(scope="module")
def df_feat():
    df_scales = aa.load_scales(name="scales")
    df_cat = aa.load_scales(name="scales_cat")
    df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(df_scales.columns)].head(4).reset_index(drop=True)
    scale_ids = df_cat[ut.COL_SCALE_ID].tolist()
    splits = ["Segment(1,2)", "Segment(2,2)", "Segment(1,1)", "Pattern(C,1)"]
    parts = ["TMD", "TMD", "JMD_N", "TMD"]
    return pd.DataFrame({
        ut.COL_FEATURE: [f"{parts[i]}-{splits[i]}-{scale_ids[i]}" for i in range(4)],
        ut.COL_CAT: df_cat[ut.COL_CAT], ut.COL_SUBCAT: df_cat[ut.COL_SUBCAT],
        ut.COL_SCALE_NAME: df_cat[ut.COL_SCALE_NAME],
        "abs_auc": [0.2, 0.15, 0.3, 0.1], "abs_mean_dif": [0.3, 0.2, 0.5, 0.4],
        "mean_dif": [0.3, -0.2, 0.5, -0.4], "std_test": 0.1, "std_ref": 0.1})


@pytest.fixture(scope="module")
def df_seq():
    rng = np.random.default_rng(0)
    rows = [{ut.COL_ENTRY: f"P{i:03d}",
             ut.COL_SEQ: "".join(rng.choice(list(_AAS), size=40)),
             ut.COL_TMD_START: 11, ut.COL_TMD_STOP: 20} for i in range(16)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def labels():
    return [1, 0] * 8


@pytest.fixture(scope="module")
def query_seq():
    return "".join(np.random.default_rng(1).choice(list(_AAS), size=40))


def _csp():
    # tmd_len 10 / jmd 10 -> window [1..30] fits the 30-residue synthetic pdb at init_site=11
    return aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)


def _explore(csp, df_feat, query_seq, df_seq, labels, pdb_path, **kw):
    kw.setdefault("model", ut.MODEL_RF)
    kw.setdefault("tmd_len", 10)
    kw.setdefault("init_site", 11)
    kw.setdefault("random_state", 42)
    return csp.explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                       df_seq=df_seq, labels=labels, **kw)


# --- normal cases (output dispatch) ------------------------------------------
class TestExplore:
    """Normal behaviour: the built-in predictor + output dispatch."""

    def test_static_returns_combined_view(self, df_feat, query_seq, df_seq, labels, pdb_path):
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static")
        assert isinstance(view, CombinedView)

    def test_html_returns_linked_view(self, df_feat, query_seq, df_seq, labels, pdb_path):
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="html")
        assert isinstance(view, LinkedView)

    def test_html_writes_path(self, df_feat, query_seq, df_seq, labels, pdb_path, tmp_path):
        out = tmp_path / "explore.html"
        _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="html",
                 path=str(out))
        assert out.is_file() and out.stat().st_size > 0

    def test_widget_returns_panel(self, df_feat, query_seq, df_seq, labels, pdb_path):
        pytest.importorskip("ipywidgets")
        panel = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="widget")
        assert panel is not None

    def test_static_writes_path(self, df_feat, query_seq, df_seq, labels, pdb_path, tmp_path):
        # output='static' + path saves the feature-map panel image (Stage D capture)
        out = tmp_path / "static.png"
        _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                 path=str(out))
        assert out.is_file() and out.read_bytes()[:4] == b"\x89PNG"

    @pytest.mark.parametrize("model", [ut.MODEL_RF, ut.MODEL_SVM, ut.MODEL_LOG_REG])
    def test_model_names(self, df_feat, query_seq, df_seq, labels, pdb_path, model):
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                        model=model)
        assert isinstance(view, CombinedView)

    @pytest.mark.parametrize("mode", ["impact", "plddt"])
    def test_mode(self, df_feat, query_seq, df_seq, labels, pdb_path, mode):
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                        mode=mode)
        assert isinstance(view, CombinedView)

    @pytest.mark.parametrize("focus", ["whole", "fade", "zoom"])
    def test_focus(self, df_feat, query_seq, df_seq, labels, pdb_path, focus):
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                        focus=focus)
        assert isinstance(view, CombinedView)

    def test_custom_col_imp(self, df_feat, query_seq, df_seq, labels, pdb_path):
        # col_imp must stay in the feat_impact family for the SHAP feature map (shap_plot=True)
        view = _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                        col_imp="feat_impact_site")
        assert isinstance(view, CombinedView)

    def test_default_init_site(self, df_feat, query_seq, df_seq, labels, pdb_path):
        # init_site omitted -> defaults near the middle; window must still fit the structure
        view = _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                              labels=labels, model=ut.MODEL_RF, tmd_len=10, output="static",
                              random_state=42)
        assert isinstance(view, CombinedView)


# --- escape hatch ------------------------------------------------------------
class TestExploreCustomPredictor:
    """A custom predictor overrides the built-in; df_seq/labels/model are ignored."""

    def test_custom_predictor_static(self, df_feat, query_seq, pdb_path):
        calls = {"n": 0}

        def predictor(sequence, p1):
            calls["n"] += 1
            out = df_feat.copy()
            out[ut.COL_FEAT_IMPACT] = [0.8, -0.5, 1.2, -0.3]
            return out

        view = _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                              predictor=predictor, output="static", tmd_len=10, init_site=11)
        assert isinstance(view, CombinedView) and calls["n"] == 1

    def test_custom_predictor_ignores_training(self, df_feat, query_seq, pdb_path):
        def predictor(sequence, p1):
            out = df_feat.copy()
            out[ut.COL_FEAT_IMPACT] = 1.0
            return out

        # No df_seq / labels given, but a custom predictor makes them unnecessary
        view = _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                              predictor=predictor, output="static", tmd_len=10, init_site=11)
        assert isinstance(view, CombinedView)


# --- negative cases ----------------------------------------------------------
class TestExploreErrors:
    """Validation errors."""

    def test_unknown_output(self, df_feat, query_seq, df_seq, labels, pdb_path):
        with pytest.raises(ValueError):
            _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="movie")

    def test_both_pdb_and_uniprot(self, df_feat, query_seq, df_seq, labels, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, uniprot="P12345",
                          df_seq=df_seq, labels=labels, output="static", tmd_len=10, init_site=11)

    def test_neither_pdb_nor_uniprot(self, df_feat, query_seq, df_seq, labels):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, df_seq=df_seq, labels=labels,
                          output="static", tmd_len=10, init_site=11)

    def test_builtin_without_df_seq(self, df_feat, query_seq, labels, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, labels=labels,
                          output="static", tmd_len=10, init_site=11)

    def test_builtin_without_labels(self, df_feat, query_seq, df_seq, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                          output="static", tmd_len=10, init_site=11)

    def test_non_callable_predictor(self, df_feat, query_seq, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, predictor=123,
                          output="static", tmd_len=10, init_site=11)

    def test_df_feat_missing_feature_col(self, query_seq, df_seq, labels, pdb_path):
        bad = pd.DataFrame({"mean_dif": [0.1, 0.2]})
        with pytest.raises(ValueError):
            _csp().explore(df_feat=bad, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                          labels=labels, output="static", tmd_len=10, init_site=11)

    def test_unknown_model_name(self, df_feat, query_seq, df_seq, labels, pdb_path):
        with pytest.raises(ValueError):
            _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                     model="not_a_model")

    def test_col_imp_out_of_family_with_shap(self, df_feat, query_seq, df_seq, labels, pdb_path):
        # col_imp must be feat_impact-family when shap_plot=True; caught early, not deep in render
        with pytest.raises(ValueError):
            _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                     col_imp="impact")

    def test_init_site_too_small(self, df_feat, query_seq, df_seq, labels, pdb_path):
        # init_site < jmd_n_len + 1 would push the window start below residue 1
        with pytest.raises(ValueError, match="init_site"):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                          labels=labels, output="static", tmd_len=10, init_site=3)

    @pytest.mark.parametrize("bad", ["x", "movie"])
    def test_unknown_mode(self, df_feat, query_seq, df_seq, labels, pdb_path, bad):
        with pytest.raises(ValueError):
            _explore(_csp(), df_feat, query_seq, df_seq, labels, pdb_path, output="static",
                     mode=bad)


# --- Stage C: multi-site live HTML (sites=) ----------------------------------
def _stub_predictor(df_feat):
    def predictor(sequence, p1):
        out = df_feat.copy()
        out[ut.COL_FEAT_IMPACT] = [0.8, -0.5, 1.2, -0.3]
        return out
    return predictor


class TestExploreMultiSite:
    """output='html' with sites=[...] bakes one app-like, client-side-switchable HTML."""

    def test_returns_linked_view(self, df_feat, query_seq, df_seq, labels, pdb_path):
        view = _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                              labels=labels, model="rf", output="html", sites=[12, 16, 20],
                              tmd_len=10, random_state=42)
        assert isinstance(view, LinkedView)

    def test_bakes_all_sites_with_slider(self, df_feat, query_seq, df_seq, labels, pdb_path,
                                         tmp_path):
        out = tmp_path / "live.html"
        _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                       labels=labels, model="rf", output="html", sites=[12, 16, 20, 24],
                       tmd_len=10, path=str(out), random_state=42)
        txt = out.read_text()
        assert "var SITES =" in txt and 'type="range"' in txt and "function showSite" in txt
        assert txt.count('"p1":') == 4

    def test_custom_predictor_multisite(self, df_feat, query_seq, pdb_path):
        view = _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                              predictor=_stub_predictor(df_feat), output="html",
                              sites=[12, 18], tmd_len=10)
        assert isinstance(view, LinkedView)

    # negative
    def test_sites_with_non_html_raises(self, df_feat, query_seq, df_seq, labels, pdb_path):
        with pytest.raises(ValueError, match="only valid with output='html'"):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                          labels=labels, output="static", sites=[12, 16], tmd_len=10)

    def test_empty_sites_raises(self, df_feat, query_seq, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                          predictor=_stub_predictor(df_feat), output="html", sites=[], tmd_len=10)

    def test_site_out_of_range_raises(self, df_feat, query_seq, pdb_path):
        with pytest.raises(ValueError):
            _csp().explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path,
                          predictor=_stub_predictor(df_feat), output="html",
                          sites=[3], tmd_len=10)   # 3 < jmd_n_len+1

    def test_hard_cap_raises(self, df_feat, pdb_path):
        long_seq = "".join(np.random.default_rng(2).choice(list(_AAS), size=230))
        with pytest.raises(ValueError, match="caps at"):
            _csp().explore(df_feat=df_feat, sequence=long_seq, pdb=pdb_path,
                          predictor=_stub_predictor(df_feat), output="html",
                          sites=list(range(11, 212)), tmd_len=10)   # 201 sites > cap, raises pre-bake

    def test_verbose_and_zoom(self, df_feat, query_seq, df_seq, labels, pdb_path):
        # verbose=True + focus='zoom' exercise the progress prints and the zoom-baking branch
        csp = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=True)
        view = csp.explore(df_feat=df_feat, sequence=query_seq, pdb=pdb_path, df_seq=df_seq,
                           labels=labels, model="rf", output="html", sites=[12, 16],
                           tmd_len=10, focus="zoom", random_state=42)
        assert isinstance(view, LinkedView)

    def test_soft_warn_past_threshold(self, df_feat, query_seq, pdb_path):
        # > _WARN_SITES (40) emits a UserWarning (stub predictor keeps the bake cheap)
        long_seq = "".join(np.random.default_rng(3).choice(list(_AAS), size=70))
        with pytest.warns(UserWarning, match="large file"):
            _csp().explore(df_feat=df_feat, sequence=long_seq, pdb=pdb_path,
                          predictor=_stub_predictor(df_feat), output="html",
                          sites=list(range(11, 53)), tmd_len=10)   # 42 sites > warn threshold
