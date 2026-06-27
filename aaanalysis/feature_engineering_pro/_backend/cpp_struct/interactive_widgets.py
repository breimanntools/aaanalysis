"""
This is a script for the backend of CPPStructurePlot.interactive: the ipywidgets
panel that links a site selection to a predictor re-run and a synchronized
structure + feature-map repaint. ipywidgets / IPython are imported lazily by the
frontend, since interactive is a notebook-only pro feature.
"""
import asyncio

from .mapping import compute_residue_impact
from .render import render_py3dmol, render_mpl, py3dmol_available


# I Helper Functions
def _window_resis(focus, focus_region, positions_union):
    """Residues defining the focus window (``None`` for ``'whole'``)."""
    if focus == "whole":
        return None
    if not focus_region:
        return set(positions_union)
    ranges = [focus_region] if isinstance(focus_region, tuple) else focus_region
    resis = set()
    for start, stop in ranges:
        resis.update(range(int(start), int(stop) + 1))
    return resis


class _Debouncer:
    """Coalesce rapid calls on the event loop's thread: run only the last within ``delay_s``.

    Schedules ``func`` via the running asyncio loop (the Jupyter kernel's main thread), so
    the repaint never touches matplotlib / ipywidgets off-thread. A new call cancels the
    pending one, so two changes inside ``delay_s`` trigger a single ``func`` call with the
    latest arguments. With ``delay_s <= 0`` — or when no event loop is running (plain scripts,
    tests) — it runs synchronously. ``flush`` runs the pending call now; ``cancel`` drops it.
    """

    def __init__(self, delay_s, func):
        self._delay = delay_s
        self._func = func
        self._handle = None
        self._pending = None

    def __call__(self, *args):
        self.cancel()
        self._pending = args
        if self._delay <= 0:
            self.flush()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            self.flush()   # no event loop (e.g. headless / tests) -> run now
        else:
            self._handle = loop.call_later(self._delay, self.flush)

    def cancel(self):
        """Discard any pending call: cancel the scheduled callback and drop the arguments."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._pending = None

    def flush(self):
        """Run the pending call now (if any) and clear it."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._pending is not None:
            args, self._pending = self._pending, None
            self._func(*args)


# II Main Functions
class InteractivePanel:
    """State + widgets for the linked site explorer (one per ``interactive()`` call).

    Builds the ipywidgets controls (site slider, colour/focus dropdowns) and two output
    panels (structure, feature map), and on each debounced selection calls the user predictor
    once and repaints both panels from the returned ``df_feat`` — the structure colour and the
    feature map read the same per-residue impact.
    """

    def __init__(self, ipw, *, predictor, sequence, records, pdb_path, chain_id,
                 col_imp, tmd_len, jmd_n_len, jmd_c_len, mode, focus, focus_region,
                 size_by_impact, normalize_by_span, site_to_start, feature_map_renderer,
                 init_site, debounce_ms, tmp_holder=None, verbose=False):
        from IPython.display import display
        self._ipw = ipw
        self._display = display
        self._predictor = predictor
        self._sequence = sequence
        self._records = records
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._col_imp = col_imp
        self._tmd_len = tmd_len
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len
        self._focus_region = focus_region
        self._size_by_impact = size_by_impact
        self._normalize_by_span = normalize_by_span
        self._site_to_start = site_to_start
        self._feature_map_renderer = feature_map_renderer
        self._tmp_holder = tmp_holder   # keep an AlphaFold temp dir alive for the panel's life
        self._verbose = verbose
        self.n_predict = 0              # test hook: count of predictor calls
        self.last = None               # test hook: last rendered {p1, start, ...}

        # Controls (continuous_update=False -> the slider fires only on release)
        self.w_site = ipw.IntSlider(value=init_site, min=1, max=len(sequence),
                                    description="site (P1)", continuous_update=False)
        self.w_mode = ipw.Dropdown(options=["impact", "plddt"], value=mode, description="colour")
        self.w_focus = ipw.Dropdown(options=["whole", "fade", "zoom"], value=focus,
                                    description="focus")
        self.out_struct = ipw.Output()
        self.out_map = ipw.Output() if feature_map_renderer is not None else None

        self._debounced = _Debouncer(debounce_ms / 1000.0, self.render_site)
        for w in (self.w_site, self.w_mode, self.w_focus):
            w.observe(self._on_change, names="value")

        controls = ipw.HBox([self.w_site, self.w_mode, self.w_focus])
        panel_row = [self.out_struct] + ([self.out_map] if self.out_map is not None else [])
        self.container = ipw.VBox([controls, ipw.HBox(panel_row)])
        self.container._cpp_panel = self   # introspection / test handle
        # Initial paint (one predictor call, synchronous)
        self.render_site(init_site, mode, focus)

    def _on_change(self, _change):
        """Widget observer: re-render (debounced) from the current control values."""
        self._debounced(self.w_site.value, self.w_mode.value, self.w_focus.value)

    def render_site(self, p1, mode, focus):
        """Call the predictor for ``p1`` and repaint the structure (+ feature map)."""
        start = int(self._site_to_start(p1))
        if start < 0:
            self._show_message(
                f"Site P1={p1} is too close to the N-terminus for the current window "
                f"(start={start} < 0). Pick a larger site or adjust 'site_to_start'.")
            return
        self.n_predict += 1
        df_feat = self._predictor(self._sequence, p1)
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=self._col_imp, start=start, tmd_len=self._tmd_len,
            jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
            normalize_by_span=self._normalize_by_span)
        window_resis = _window_resis(focus, self._focus_region, positions_union)
        self._paint_structure(dict_impact, max_abs, mode, focus, window_resis)
        if self.out_map is not None:
            self._paint_map(df_feat, start)
        self.last = dict(p1=p1, start=start, mode=mode, focus=focus,
                         df_feat=df_feat, dict_impact=dict_impact, max_abs=max_abs)

    def _paint_structure(self, dict_impact, max_abs, mode, focus, window_resis):
        """Render the structure into the structure output (in place)."""
        if py3dmol_available():
            view = render_py3dmol(pdb_path=self._pdb_path, records=self._records,
                                  dict_impact=dict_impact, max_abs=max_abs, mode=mode,
                                  focus=focus, window_resis=window_resis,
                                  size_by_impact=self._size_by_impact, chain_id=self._chain_id)
        else:
            view = render_mpl(records=self._records, dict_impact=dict_impact, max_abs=max_abs,
                              mode=mode, focus=focus, window_resis=window_resis,
                              size_by_impact=self._size_by_impact)
        self.out_struct.clear_output(wait=True)
        with self.out_struct:
            self._display(view)
        if view.backend == "mpl":
            import matplotlib.pyplot as plt
            plt.close(view.fig)

    def _paint_map(self, df_feat, start):
        """Render the feature map (frontend-supplied) into the map output (in place)."""
        import matplotlib.pyplot as plt
        fig = self._feature_map_renderer(df_feat, start)
        self.out_map.clear_output(wait=True)
        with self.out_map:
            self._display(fig)
        plt.close(fig)

    def _show_message(self, msg):
        """Show an informational message in the structure output (no crash, no print)."""
        self.out_struct.clear_output(wait=True)
        with self.out_struct:
            self._display(self._ipw.HTML(f"<i>{msg}</i>"))
        self.last = dict(p1=None, start=None, message=msg)
