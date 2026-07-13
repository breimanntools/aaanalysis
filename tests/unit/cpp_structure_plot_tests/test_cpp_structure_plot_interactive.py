"""This is a script to test CPPStructurePlot.interactive()."""
import matplotlib
matplotlib.use("Agg")
import asyncio
import sys
import numpy as np
import pandas as pd
import pytest

# Pro-gated: biopython parses structures, py3Dmol renders them, ipywidgets drives the panel.
pytest.importorskip("Bio")
pytest.importorskip("py3Dmol")
pytest.importorskip("ipywidgets")

import aaanalysis as aa
import aaanalysis.utils as ut


# --- fixtures / helpers ------------------------------------------------------
def _make_pdb(path, n=60, chain="A"):
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
    return _make_pdb(tmp_path_factory.mktemp("struct_inter") / "synthetic.pdb")


def _df_feat():
    df_cat = aa.load_scales(name="scales_cat").head(4).reset_index(drop=True)
    splits = ["Segment(1,2)", "Segment(2,2)", "Segment(1,1)", "Pattern(C,1)"]
    parts = ["TMD", "TMD", "JMD_N", "TMD"]
    rows = {ut.COL_FEATURE: [], ut.COL_CAT: [], ut.COL_SUBCAT: [], ut.COL_SCALE_NAME: []}
    for i, r in df_cat.iterrows():
        rows[ut.COL_FEATURE].append(f"{parts[i]}-{splits[i]}-{r[ut.COL_SCALE_ID]}")
        rows[ut.COL_CAT].append(r[ut.COL_CAT])
        rows[ut.COL_SUBCAT].append(r[ut.COL_SUBCAT])
        rows[ut.COL_SCALE_NAME].append(r[ut.COL_SCALE_NAME])
    df = pd.DataFrame(rows)
    df["abs_auc"] = 0.2
    df["abs_mean_dif"] = 0.3
    df["mean_dif"] = [0.3, -0.2, 0.5, -0.4]
    df["std_test"] = 0.1
    df["std_ref"] = 0.1
    df["feat_impact"] = [0.8, -0.5, 1.2, -0.3]
    return df


class _RecordingPredictor:
    """Stub predictor that records its (sequence, p1) calls and returns a fixed df_feat."""

    def __init__(self):
        self.calls = []

    def __call__(self, sequence, p1):
        self.calls.append((sequence, p1))
        return _df_feat()


SEQ = "A" * 60


def _panel(pdb_path, predictor=None, **kwargs):
    cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
    kwargs.setdefault("tmd_len", 10)
    kwargs.setdefault("col_imp", "feat_impact")
    kwargs.setdefault("debounce_ms", 0)
    container = cpps_plot.interactive(predictor=predictor or _RecordingPredictor(),
                                sequence=SEQ, pdb=pdb_path, **kwargs)
    return container, container._cpp_panel


# --- normal cases ------------------------------------------------------------
class TestInteractive:
    """Normal cases — one parameter per test."""

    def test_returns_widget_container(self, pdb_path):
        import ipywidgets as ipw
        container, _ = _panel(pdb_path)
        assert isinstance(container, ipw.Widget)
        assert len(container.children) == 2  # controls row + panel row

    def test_initial_render_calls_predictor_once(self, pdb_path):
        pred = _RecordingPredictor()
        _, panel = _panel(pdb_path, predictor=pred, init_site=25)
        assert len(pred.calls) == 1
        assert pred.calls[0] == (SEQ, 25)

    def test_default_site_to_start(self, pdb_path):
        # default site_to_start: start = p1 - jmd_n_len
        _, panel = _panel(pdb_path, init_site=25)
        assert panel.last["start"] == 25 - 10

    def test_site_change_triggers_predictor(self, pdb_path):
        pred = _RecordingPredictor()
        _, panel = _panel(pdb_path, predictor=pred, init_site=20)
        panel.w_site.value = 40
        assert pred.calls[-1] == (SEQ, 40)
        assert panel.last["p1"] == 40 and panel.last["start"] == 30

    def test_custom_site_to_start(self, pdb_path):
        _, panel = _panel(pdb_path, init_site=30, site_to_start=lambda p1: p1 - 4 - 10)
        assert panel.last["start"] == 30 - 4 - 10

    def test_feature_map_false_no_map_panel(self, pdb_path):
        _, panel = _panel(pdb_path, feature_map=False)
        assert panel.out_map is None

    def test_feature_map_true_has_map_panel(self, pdb_path):
        _, panel = _panel(pdb_path, feature_map=True)
        assert panel.out_map is not None

    def test_mode_and_focus_controls_present(self, pdb_path):
        _, panel = _panel(pdb_path)
        assert list(panel.w_mode.options) == ["impact", "plddt"]
        assert list(panel.w_focus.options) == ["whole", "fade", "zoom"]

    def test_focus_change_repaints(self, pdb_path):
        pred = _RecordingPredictor()
        _, panel = _panel(pdb_path, predictor=pred)
        n0 = len(pred.calls)
        panel.w_focus.value = "zoom"
        assert len(pred.calls) == n0 + 1
        assert panel.last["focus"] == "zoom"

    def test_init_site_default_is_middle(self, pdb_path):
        _, panel = _panel(pdb_path)
        assert panel.w_site.value == len(SEQ) // 2

    def test_linkage_same_impact(self, pdb_path):
        # The structure dict_impact is derived from the same col_imp the feature map uses.
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.mapping import (
            compute_residue_impact)
        _, panel = _panel(pdb_path, init_site=25)
        expected, _, _ = compute_residue_impact(
            df_feat=panel.last["df_feat"], col_imp="feat_impact", start=panel.last["start"],
            tmd_len=10, jmd_n_len=10, jmd_c_len=10)
        assert panel.last["dict_impact"] == expected

    # --- negatives -----------------------------------------------------------
    def test_predictor_not_callable_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor="nope", sequence=SEQ, pdb=pdb_path, tmd_len=10)

    def test_pdb_and_uniprot_both_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            uniprot="P05067", tmd_len=10)

    def test_pdb_and_uniprot_neither_negative(self):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, tmd_len=10)

    @pytest.mark.parametrize("mode", ["Impact", "shap", ""])
    def test_mode_negative(self, pdb_path, mode):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            tmd_len=10, mode=mode)

    @pytest.mark.parametrize("init_site", [0, -1, 999])
    def test_init_site_out_of_range_negative(self, pdb_path, init_site):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            tmd_len=10, init_site=init_site)

    def test_site_to_start_not_callable_negative(self, pdb_path):
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(ValueError):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            tmd_len=10, site_to_start=5)


# --- combinations & edge interactions ----------------------------------------
class TestInteractiveComplex:
    """Debounce, gating, and the AlphaFold-fetch path."""

    def test_debounce_coalesces_on_loop(self, pdb_path):
        # On a running event loop, two rapid changes coalesce into one predictor call.
        pred = _RecordingPredictor()

        async def scenario():
            _, panel = _panel(pdb_path, predictor=pred, debounce_ms=50, init_site=20)
            n0 = len(pred.calls)        # initial render is synchronous (one call)
            panel.w_site.value = 30
            panel.w_site.value = 40
            assert len(pred.calls) == n0   # both scheduled, neither has fired yet
            await asyncio.sleep(0.15)
            assert len(pred.calls) == n0 + 1
            assert pred.calls[-1] == (SEQ, 40)   # the latest value wins

        asyncio.run(scenario())

    def test_site_near_nterminus_no_crash(self, pdb_path):
        # start = p1 - jmd_n_len < 0 must not crash: show a message, skip the predictor.
        pred = _RecordingPredictor()
        _, panel = _panel(pdb_path, predictor=pred, init_site=20)
        n0 = len(pred.calls)
        panel.w_site.value = 3   # start = 3 - 10 = -7 < 0
        assert len(pred.calls) == n0          # predictor not called
        assert panel.last.get("message") is not None

    def test_uniprot_fetch_path_mocked(self, monkeypatch):
        from aaanalysis.data_handling_pro import StructurePreprocessor

        def _fake_fetch(self, df_seq=None, out_folder=None, **kwargs):
            import pathlib
            p = pathlib.Path(out_folder) / "AF-Q9NQ76-F1-model_v4.pdb"
            _make_pdb(p)
            return pd.DataFrame({"entry": ["Q9NQ76"], "model_path": [str(p)]})

        monkeypatch.setattr(StructurePreprocessor, "fetch_alphafold", _fake_fetch)
        cpps_plot = aa.CPPStructurePlot(jmd_n_len=10, jmd_c_len=10, verbose=False)
        container = cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ,
                                    uniprot="Q9NQ76", tmd_len=10, debounce_ms=0)
        assert container._cpp_panel.n_predict == 1

    def test_missing_ipywidgets_raises_friendly(self, pdb_path, monkeypatch):
        # Simulate ipywidgets absent: import fails -> friendly RuntimeError, not ImportError.
        monkeypatch.setitem(sys.modules, "ipywidgets", None)
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(RuntimeError, match="ipywidgets"):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            tmd_len=10)

    def test_missing_py3dmol_raises_friendly(self, pdb_path, monkeypatch):
        from aaanalysis.feature_engineering_pro import _cpp_structure_plot as cpps_plot_mod
        monkeypatch.setattr(cpps_plot_mod, "py3dmol_available", lambda: False)
        cpps_plot = aa.CPPStructurePlot(verbose=False)
        with pytest.raises(RuntimeError, match="py3Dmol"):
            cpps_plot.interactive(predictor=_RecordingPredictor(), sequence=SEQ, pdb=pdb_path,
                            tmd_len=10)


# --- debouncer unit ----------------------------------------------------------
def _debouncer_cls():
    from aaanalysis.feature_engineering_pro._backend.cpp_struct.interactive_widgets import (
        _Debouncer)
    return _Debouncer


class TestDebouncer:
    """The _Debouncer helper in isolation."""

    def test_zero_delay_runs_immediately(self):
        seen = []
        _debouncer_cls()(0, lambda x: seen.append(x))(1)
        assert seen == [1]

    def test_no_loop_falls_back_to_sync(self):
        # Outside an event loop, a positive delay runs synchronously (plain scripts / tests).
        seen = []
        _debouncer_cls()(100, lambda x: seen.append(x))(1)
        assert seen == [1]

    def test_coalesces_on_loop(self):
        seen = []

        async def scenario():
            d = _debouncer_cls()(0.05, lambda x: seen.append(x))
            d(1)
            d(2)
            assert seen == []          # scheduled on the loop, not yet run
            await asyncio.sleep(0.12)
            assert seen == [2]         # only the latest, once

        asyncio.run(scenario())

    def test_cancel_on_loop(self):
        seen = []

        async def scenario():
            d = _debouncer_cls()(0.05, lambda x: seen.append(x))
            d(1)
            d.cancel()
            await asyncio.sleep(0.12)
            assert seen == []

        asyncio.run(scenario())

    def test_flush_runs_pending_on_loop(self):
        seen = []

        async def scenario():
            d = _debouncer_cls()(10, lambda x: seen.append(x))
            d(7)
            d.flush()
            assert seen == [7]

        asyncio.run(scenario())


def test_interactive_in_public_api():
    assert hasattr(aa.CPPStructurePlot, "interactive")


# --- Stage B: feature-map <-> structure linking (highlight slider + click) ----
class TestLinkedHighlight:
    """The highlight (position) slider links the feature map to the structure."""

    def test_w_pos_exists_and_window_range(self, pdb_path):
        # init_site=20, jmd_n_len=10 -> start=10, n_pos=10+10+10=30 -> window [10, 39]
        _, p = _panel(pdb_path, init_site=20)
        assert hasattr(p, "w_pos")
        assert (p.w_pos.min, p.w_pos.max) == (10, 39)
        assert p.w_pos.value == 20

    def test_initial_highlight_recorded(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)
        assert p.last["highlight_resi"] == 20

    def test_highlight_change_does_not_repredict(self, pdb_path):
        pred = _RecordingPredictor()
        _, p = _panel(pdb_path, predictor=pred, init_site=20)
        n0 = p.n_predict
        p.w_pos.value = 15
        assert p.n_predict == n0                 # highlight repaint only, no predictor call
        assert p.last["highlight_resi"] == 15

    def test_site_change_resets_highlight_range(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)
        p.w_site.value = 30                       # start=20 -> window [20, 49]
        assert (p.w_pos.min, p.w_pos.max) == (20, 49)

    def test_highlight_clamped_into_new_window(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)
        p.w_pos.value = 12                        # in [10,39]
        p.w_site.value = 40                       # start=30 -> window [30, 59]; 12 -> clamped to 30
        assert p.w_pos.min <= p.w_pos.value <= p.w_pos.max

    def test_click_maps_xdata_to_residue(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)     # start=10

        class _Evt:
            pass
        e = _Evt()
        e.inaxes = p._click_ctx["heat_ax"]
        e.xdata = 4.7                             # column 4 -> residue start+4 = 14
        p._on_map_click(e)
        assert p.w_pos.value == 14

    def test_click_outside_heatmap_ignored(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)
        before = p.w_pos.value

        class _Evt:
            pass
        e = _Evt()
        e.inaxes = None
        e.xdata = 4.7
        p._on_map_click(e)
        assert p.w_pos.value == before

    def test_site_change_does_not_trigger_highlight_repaint(self, pdb_path):
        # A site change paints the frame itself; the clamp-driven w_pos change must be suppressed
        # (no second, momentarily-stale highlight repaint).
        _, p = _panel(pdb_path, init_site=20)
        h0 = p.n_highlight
        p.w_site.value = 40                       # window moves; w_pos clamps
        assert p.n_highlight == h0

    def test_manual_highlight_counts(self, pdb_path):
        _, p = _panel(pdb_path, init_site=20)
        h0 = p.n_highlight
        p.w_pos.value = 15                         # a manual drag does repaint the highlight
        assert p.n_highlight == h0 + 1

    def test_no_map_panel_still_has_slider(self, pdb_path):
        # feature_map=False -> no map panel, but the structure highlight slider still works
        _, p = _panel(pdb_path, init_site=20, feature_map=False)
        assert hasattr(p, "w_pos") and p.out_map is None
        n0 = p.n_predict
        p.w_pos.value = 15
        assert p.n_predict == n0 and p.last["highlight_resi"] == 15


class TestRenderHighlight:
    """render_py3dmol accepts a highlight_resi marker overlay."""

    def test_highlight_resi_renders(self, pdb_path):
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.render import render_py3dmol
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.structure import (
            load_structure, extract_chain_residues)
        records, _identity, chain_id = extract_chain_residues(load_structure(pdb_path))
        view = render_py3dmol(pdb_path=pdb_path, records=records, dict_impact={12: 0.5},
                              max_abs=0.5, mode="impact", focus="whole", window_resis=None,
                              size_by_impact=True, chain_id=chain_id, highlight_resi=12)
        assert view is not None     # builds without error with a highlight marker

    def test_highlight_resi_absent_ok(self, pdb_path):
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.render import render_py3dmol
        from aaanalysis.feature_engineering_pro._backend.cpp_struct.structure import (
            load_structure, extract_chain_residues)
        records, _identity, chain_id = extract_chain_residues(load_structure(pdb_path))
        # A highlight residue not present in the structure is silently skipped (no crash)
        view = render_py3dmol(pdb_path=pdb_path, records=records, dict_impact={}, max_abs=1.0,
                              mode="impact", focus="whole", window_resis=None,
                              size_by_impact=True, chain_id=chain_id, highlight_resi=9999)
        assert view is not None


class TestInteractiveParams:
    """Positive coverage for interactive parameters beyond the defaults."""

    def test_size_by_impact_true(self, pdb_path):
        container, p = _panel(pdb_path, init_site=20, size_by_impact=True)
        assert p.last["highlight_resi"] == 20

    def test_custom_site_to_start(self, pdb_path):
        # A custom geometry: p1 -> start mapping other than the default (p1 - jmd_n_len)
        _, p = _panel(pdb_path, init_site=20, site_to_start=lambda p1: p1 - 5)
        assert p.last["start"] == 15   # 20 - 5, not 20 - jmd_n_len(10)

    def test_normalize_by_span(self, pdb_path):
        container, p = _panel(pdb_path, init_site=20, normalize_by_span=True)
        assert p.last is not None
