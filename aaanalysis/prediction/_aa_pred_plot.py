"""
This is a script for the frontend of the AAPredPlot class for visualizing AAPred results.
"""
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import aaanalysis.utils as ut
from ._backend.aa_pred.aa_pred_plot_comparison import plot_comparison_
from ._backend.aa_pred.aa_pred_plot_ranking import plot_ranking_, ranking_figheight
from ._backend.aa_pred.aa_pred_plot_rank_scatter import plot_rank_scatter_
from ._backend.aa_pred.aa_pred_plot_clustermap import plot_clustermap_


# I Helper Functions
# Single-protein positional plot kinds dispatched by :meth:`AAPredPlot.predict_sample`.
LIST_SAMPLE_KINDS = ["window", "domain", "sequence"]
# Across-samples plot kinds (of per-sample scores) dispatched by :meth:`AAPredPlot.predict_group`.
LIST_GROUP_KINDS = ["hist", "ranking", "rank_scatter", "scatter", "cutoff"]
# Sample-relation plot kinds (of the sample x feature matrix) dispatched by
# :meth:`AAPredPlot.group_cluster`.
LIST_CLUSTER_KINDS = ["clustermap"]
# Evaluation plot kinds dispatched by :meth:`AAPredPlot.eval`.
LIST_EVAL_KINDS = ["eval", "comparison", "heatmap"]
# Per-kind figure-size defaults used when ``figsize=None`` (split by method).
_DICT_SAMPLE_FIGSIZE = {"window": (10, 4), "domain": (6, 4.5), "sequence": (12, 6)}
# Cap on the number of subcategory rows drawn in the ``kind='sequence'`` heatmap when
# ``subcats=None`` (so a full 74-subcategory classification does not produce a wall of rows).
_SUBCAT_ROW_CAP = 25
_DICT_GROUP_FIGSIZE = {"hist": (6, 4.5), "ranking": None, "rank_scatter": None,
                        "scatter": (5.5, 5.5), "cutoff": (6, 4.5)}
_DICT_CLUSTER_FIGSIZE = {"clustermap": (9, 9)}


def _new_ax(ax=None, figsize=(6, 5)):
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def check_match_scores_labels(scores=None, labels=None):
    """Check that per-sample labels match the length of the scores array."""
    if labels is not None and len(labels) != len(scores):
        raise ValueError(f"'labels' (n={len(labels)}) should match length of 'scores' (n={len(scores)}).")


def _default(val, fallback):
    """Return ``val`` unless it is ``None``, then ``fallback`` (per-kind default resolution)."""
    return fallback if val is None else val


def _resolve_band_colors(colors, cmap, n_bands):
    """Resolve one color per confidence band (low score -> high score).

    ``colors`` (a list of length ``n_bands``) is used verbatim if given; otherwise
    ``n_bands`` colors are sampled across the inner range of ``cmap`` so the low-score
    band is the low end of the colormap and the high-score band the high end.
    """
    if colors is not None:
        if not isinstance(colors, (list, tuple)) or len(colors) != n_bands:
            raise ValueError(f"'colors' ({colors}) should be a list of {n_bands} colors (one per "
                             f"band delimited by 'thresholds') when 'band=True'.")
        return list(colors)
    cmap_obj = plt.get_cmap(cmap)
    return [cmap_obj(x) for x in np.linspace(0.15, 0.85, n_bands)]


def _band_index(left_edge, sorted_thresholds):
    """Band index (0-based, low -> high) of a bar whose left edge is ``left_edge``."""
    return int(np.searchsorted(sorted_thresholds, left_edge, side="right"))


def _resolve_cluster_tracks(tracks, n):
    """Resolve ``group_cluster`` annotation tracks into (column, row) sidebar arguments.

    Returns ``(col_values, col_colors, col_title, row_values, row_colors, row_title)``. A single
    track annotates both sidebars (mirrored, the single-annotation figure) unless its ``where`` is
    ``"left"`` (row only); with two tracks the first is the column (top) sidebar and the second the
    row (left) sidebar, overridable per track via ``where``.
    """
    if tracks is None:
        return None, None, None, None, None, None
    if not isinstance(tracks, (list, tuple)) or not 1 <= len(tracks) <= 2:
        raise ValueError(f"'tracks' ({tracks}) should be a list of 1 or 2 annotation dicts.")
    parsed = []
    for i, t in enumerate(tracks):
        if not isinstance(t, dict) or "values" not in t:
            raise ValueError(f"'tracks[{i}]' ({t}) should be a dict with a 'values' key.")
        vals = ut.check_list_like(name=f"tracks[{i}]['values']", val=t["values"], accept_none=False)
        if len(vals) != n:
            raise ValueError(f"'tracks[{i}][\"values\"]' (n={len(vals)}) should match n_samples ({n}).")
        where = t.get("where")
        if where is not None and where not in ("top", "left", "both"):
            raise ValueError(f"'tracks[{i}][\"where\"]' ('{where}') should be 'top', 'left' or 'both'.")
        ut.check_str(name=f"tracks[{i}]['title']", val=t.get("title"), accept_none=True)
        parsed.append(dict(values=vals, colors=t.get("colors"), title=t.get("title", "Class"), where=where))
    if len(parsed) == 1:
        t = parsed[0]
        if t["where"] == "left":
            return None, None, None, t["values"], t["colors"], t["title"]
        return t["values"], t["colors"], t["title"], None, None, None
    col = next((t for t in parsed if t["where"] in ("top", "both")), parsed[0])
    row = next((t for t in parsed if t["where"] == "left"), None) or next((t for t in parsed if t is not col), parsed[1])
    return (col["values"], col["colors"], col["title"],
            row["values"], row["colors"], row["title"])


def _check_highlight_cells(highlight, data):
    """Resolve/validate the eval-heatmap ``highlight`` to a list of ``(row, col)`` cells to box.

    - ``None`` -> ``[]`` (box nothing).
    - a positive ``int`` N -> the ``N`` highest-value cells, best first (NaNs ignored).
    - ``"max"`` / ``"min"`` -> the single best / worst cell (aliases for ``1`` / worst).
    - a ``(row, col)`` tuple, or a list of them -> those explicit cells (bounds-checked).
    """
    if highlight is None:
        return []
    n_rows, n_cols = data.shape

    def _rank_cells(n, worst=False):
        flat = data.ravel()
        valid = np.where(~np.isnan(flat))[0]
        if valid.size == 0:
            raise ValueError("'highlight' (top-N / 'max' / 'min') needs at least one non-NaN cell "
                             "in 'df_eval'.")
        order = valid[np.argsort(flat[valid], kind="stable")]  # ascending (worst first)
        if not worst:
            order = order[::-1]                                # best first
        return [tuple(int(v) for v in np.unravel_index(k, data.shape)) for k in order[:min(n, order.size)]]

    if isinstance(highlight, str):
        if highlight == "max":
            return _rank_cells(1, worst=False)
        if highlight == "min":
            return _rank_cells(1, worst=True)
        raise ValueError(f"'highlight' ('{highlight}') should be a positive int (top-N), 'max', "
                         f"'min', a (row, col) tuple/list, or None.")
    if _is_int(highlight):
        if highlight < 1:
            raise ValueError(f"'highlight' ({highlight}) as an int should be >= 1 (number of top "
                             f"cells to box).")
        return _rank_cells(int(highlight), worst=False)
    # explicit (row, col) or a list of them
    cells = highlight if (isinstance(highlight, list) and highlight
                          and isinstance(highlight[0], (tuple, list))) else [highlight]
    out = []
    for cell in cells:
        if not (isinstance(cell, (tuple, list)) and len(cell) == 2 and _is_int(cell[0]) and _is_int(cell[1])):
            raise ValueError(f"'highlight' cell ({cell}) should be a (row, col) integer tuple.")
        i, j = int(cell[0]), int(cell[1])
        if not (0 <= i < n_rows and 0 <= j < n_cols):
            raise ValueError(f"'highlight' cell ({i}, {j}) is outside the grid (shape {(n_rows, n_cols)}).")
        out.append((i, j))
    return out


# Multi-track sequence-viewer helpers (shared by the ``window``/``domain`` renderers).
# The base prediction axes are stacked with optional extra tracks that all share the
# residue-position x-axis: a CPP-importance profile, per-subcategory scale profiles,
# user annotation tracks, and a bottom sequence row.
_SEQ_ROW_MAX = 80  # longest visible region for which per-residue letters are still drawn
_HIGHLIGHT_ALPHA = 0.25  # opacity of a highlight span (kept low so it sits behind the data)
_ZOOM_PAD = 5  # residues of padding added on each side of a zoomed highlight region


def _is_int(val):
    """True if ``val`` is a plain / numpy integer (``bool`` is rejected)."""
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def _normalize_highlight(highlight):
    """Normalize ``highlight`` to a list of validated ``(start, stop)`` int tuples.

    Accepts ``None`` (-> ``[]``), a single ``(start, stop)`` tuple, or a list of such
    tuples (1-based, inclusive residue-position / x-axis bounds). Each ``start`` / ``stop``
    must be an integer with ``start <= stop``; anything else raises ``ValueError``.
    """
    if highlight is None:
        return []
    # A length-2 collection of scalars is one region; a list of (start, stop) pairs is many.
    is_single = (isinstance(highlight, (tuple, list)) and len(highlight) == 2
                 and not any(isinstance(v, (tuple, list)) for v in highlight))
    regions = [highlight] if is_single else highlight
    if not isinstance(regions, (list, tuple)):
        raise ValueError(f"'highlight' ({highlight!r}) should be a (start, stop) tuple or a "
                         f"list of (start, stop) tuples.")
    out = []
    for region in regions:
        if not (isinstance(region, (tuple, list)) and len(region) == 2):
            raise ValueError(f"'highlight' region ({region!r}) should be a (start, stop) tuple.")
        start, stop = region
        if not (_is_int(start) and _is_int(stop)):
            raise ValueError(f"'highlight' region ({region!r}) should have integer 'start' and "
                             f"'stop' positions.")
        start, stop = int(start), int(stop)
        if start > stop:
            raise ValueError(f"'highlight' region ({start}, {stop}) should have 'start' <= 'stop'.")
        out.append((start, stop))
    return out


def _count_visible(x, visible_range):
    """Number of plotted x positions inside ``visible_range`` (all of them if ``None``)."""
    if visible_range is None:
        return len(x)
    lo, hi = visible_range
    x = np.asarray(x, dtype=float)
    return int(np.sum((x >= lo) & (x <= hi)))


def _visible_range(regions, x, zoom):
    """``(lo, hi)`` x-window to show when zooming into ``regions``, clamped to the data; else ``None``.

    Spans the minimum ``start`` to the maximum ``stop`` over ``regions`` with a small pad,
    clamped to the plotted x-range. Returns ``None`` when zoom is off or no region is given.
    """
    if not zoom or not regions:
        return None
    lo = min(start for start, _ in regions) - _ZOOM_PAD
    hi = max(stop for _, stop in regions) + _ZOOM_PAD
    x = np.asarray(x, dtype=float)
    if len(x):
        lo = max(lo, float(x.min()))
        hi = min(hi, float(x.max()))
    return float(lo), float(hi)


def _draw_highlight_spans(axes, regions):
    """Shade each ``(start, stop)`` region as a vertical span on every axes (behind the data)."""
    for tax in axes:
        for start, stop in regions:
            tax.axvspan(start, stop, color=ut.COLOR_LINK_HIGHLIGHT, alpha=_HIGHLIGHT_ALPHA,
                        linewidth=0, zorder=0)


def _entry_sequence(df_seq, entry):
    """Return ``(sequence, tmd_start)`` for ``entry`` in ``df_seq``; either may be ``None``.

    ``sequence`` is the full protein string (needed for subcategory / sequence rows) and
    ``tmd_start`` its 1-based TMD start (needed to map ``domain`` offsets onto residues).
    """
    if df_seq is None or entry is None:
        return None, None
    ut.check_df(name="df_seq", df=df_seq, cols_required=[ut.COL_ENTRY])
    sub = df_seq[df_seq[ut.COL_ENTRY] == entry]
    if len(sub) == 0:
        return None, None
    seq = str(sub[ut.COL_SEQ].iloc[0]) if ut.COL_SEQ in sub.columns else None
    tmd_start = None
    if ut.COL_TMD_START in sub.columns and pd.notna(sub[ut.COL_TMD_START].iloc[0]):
        tmd_start = int(sub[ut.COL_TMD_START].iloc[0])
    return seq, tmd_start


def _residue_indices(kind, x, tmd_start):
    """1-based protein residue index for each plotted x position, or ``None`` if unmapped.

    * ``window`` — ``x`` already holds 1-based residue (anchor) positions.
    * ``domain`` — ``x`` holds boundary offsets; residue ``= tmd_start + offset`` (needs
      ``tmd_start``, else the residue-anchored tracks are omitted).
    """
    if kind == "window":
        return np.asarray(x, dtype=int)
    if tmd_start is None:
        return None
    return np.asarray(tmd_start + np.asarray(x, dtype=int), dtype=int)


def _subcat_profile(seq, scale_ids, df_scales):
    """Per-residue (0-based over ``seq``) mean scale value across ``scale_ids``.

    ``profile[r] = mean_s df_scales.loc[seq[r], s]`` for each requested scale ``s``; ``None``
    when none of the ``scale_ids`` are present in ``df_scales``.
    """
    cols = [s for s in scale_ids if s in df_scales.columns]
    if len(cols) == 0:
        return None
    per_aa = df_scales[cols].mean(axis=1)  # index = amino acid letter -> mean scale value
    return np.array([per_aa.get(aa, np.nan) for aa in seq], dtype=float)


def _importance_profile(df_feat):
    """Window-frame position -> summed feature importance, spread over each feature's residues.

    Each feature contributes ``importance / n_positions`` to every 1-based position in its
    ``positions`` column (matching the CPP per-position normalization). Uses
    ``feat_importance`` when present, else ``abs_auc``. Returns ``(positions, values)`` arrays
    or ``(None, None)`` when no usable column is available.
    """
    col = (ut.COL_FEAT_IMPORT if ut.COL_FEAT_IMPORT in df_feat.columns
           else ut.COL_ABS_AUC if ut.COL_ABS_AUC in df_feat.columns else None)
    if col is None or ut.COL_POSITION not in df_feat.columns:
        return None, None
    acc = {}
    for pos_str, weight in zip(df_feat[ut.COL_POSITION].astype(str), df_feat[col].astype(float)):
        parts = [int(p) for p in pos_str.split(",") if p.strip() != ""]
        if len(parts) == 0:
            continue
        share = float(weight) / len(parts)
        for p in parts:
            acc[p] = acc.get(p, 0.0) + share
    if len(acc) == 0:
        return None, None
    positions = np.array(sorted(acc))
    values = np.array([acc[p] for p in positions], dtype=float)
    return positions, values


def _spread_on_x(x, values):
    """Map a per-index ``values`` vector onto the plotted x-range by linear interpolation.

    Used to align a window-frame profile (whose length need not equal ``len(x)``) onto the
    residue x-axis; returned unchanged when the lengths already match.
    """
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(values) == len(x):
        return values
    if len(x) < 2 or len(values) == 0:
        return np.full(len(x), np.nan)
    grid = np.linspace(float(x.min()), float(x.max()), len(values))
    return np.interp(x, grid, values)


def _is_numeric_values(values):
    """True if ``values`` can be cast to a float array (numeric -> line track)."""
    try:
        np.asarray(values, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def _build_extra_tracks(kind, x, seq, tmd_start, df_scales, df_cat, subcats, df_feat,
                        list_annotations, visible_range=None):
    """Assemble the ordered extra-track list: importance, subcats, user tracks, sequence.

    A track whose inputs are missing is simply skipped. Numeric user tracks become line
    profiles; non-numeric ones fall back to an ``imshow`` strip. The sequence row draws
    per-residue letters only when the number of *visible* positions (after any zoom, given
    by ``visible_range``) is short enough, so zooming reveals the letters automatically.
    """
    tracks = []
    residues = _residue_indices(kind=kind, x=x, tmd_start=tmd_start)
    # 1) CPP importance profile (one track)
    if df_feat is not None:
        positions, values = _importance_profile(df_feat)
        if positions is not None:
            vals = _spread_on_x(x, values)
            peak = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0.0
            if peak > 0:
                vals = vals / peak
            tracks.append(dict(type="line", values=vals, label="CPP importance",
                               color=ut.COLOR_FEAT_POS))
    # 2) Per-subcategory scale profiles (one track per subcat)
    if subcats and seq is not None and df_scales is not None and df_cat is not None \
            and residues is not None:
        clist = ut.plot_get_clist_(n_colors=max(len(subcats), 3))
        for i, name in enumerate(subcats):
            scale_ids = df_cat[df_cat[ut.COL_SUBCAT] == name][ut.COL_SCALE_ID].tolist()
            per_res = _subcat_profile(seq, scale_ids, df_scales)
            if per_res is None:
                continue
            vals = np.array([per_res[r - 1] if 1 <= r <= len(per_res) else np.nan
                             for r in residues], dtype=float)
            tracks.append(dict(type="line", values=vals, label=name,
                               color=clist[i % len(clist)]))
    # 3) User annotation tracks (numeric -> line, else categorical imshow strip)
    if list_annotations:
        for track in list_annotations:
            values = track["values"]
            label = track.get("label", "")
            if _is_numeric_values(values):
                tracks.append(dict(type="line", values=_spread_on_x(x, values),
                                   label=label, color=track.get("color")))
            else:
                cats = list(dict.fromkeys(values))
                code = np.array([cats.index(v) for v in values], dtype=float)
                tracks.append(dict(type="imshow", values=code, label=label,
                                   cmap=track.get("cmap", "viridis")))
    # 4) Sequence row at the bottom (letters only when the *visible* region is short enough)
    if seq is not None and residues is not None:
        letters = [seq[r - 1] if 1 <= r <= len(seq) else "" for r in residues]
        tracks.append(dict(type="seq", letters=letters, label="Sequence",
                           show=_count_visible(x, visible_range) <= _SEQ_ROW_MAX))
    return tracks


def _positional_layout(ax, figsize, tracks):
    """Return ``(fig, base_ax, [track_axes])`` for the stacked viewer.

    When an explicit ``ax`` is passed only the base profile is drawn on it (no sub-tracks).
    Otherwise a shared-x stack is created, the figure growing in height per extra track.
    """
    if ax is not None:
        return ax.figure, ax, []
    if not tracks:
        fig, base_ax = plt.subplots(figsize=figsize)
        return fig, base_ax, []
    heights = [6.0] + [0.5 if t["type"] == "seq" else 0.9 for t in tracks]
    figsize = (figsize[0], figsize[1] + 0.55 * len(tracks))
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": heights})
    return fig, axes[0], list(axes[1:])


def _set_track_label(tax, label):
    """Left-side, horizontally written track label aligned with the row."""
    tax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    tax.yaxis.set_label_coords(-0.012, 0.5)


def _draw_track(tax, track, x, is_bottom, visible_range=None):
    """Render one extra track onto its sub-axes (line / imshow / sequence row)."""
    ttype = track["type"]
    if ttype == "line":
        tax.plot(x, track["values"], color=track["color"] or ut.COLOR_FEAT_NEG, linewidth=1.1)
        tax.set_yticks([])
    elif ttype == "imshow":
        tax.imshow(track["values"].reshape(1, -1), aspect="auto",
                   cmap=track.get("cmap", "viridis"),
                   extent=[float(np.min(x)), float(np.max(x)), 0, 1])
        tax.set_yticks([])
    elif ttype == "seq":
        if track.get("show", True):
            lo, hi = visible_range if visible_range is not None else (-np.inf, np.inf)
            for xi, letter in zip(x, track["letters"]):
                if lo <= xi <= hi:
                    tax.text(xi, 0.5, letter, ha="center", va="center",
                             family=ut.FONT_AA, fontsize=7)
        tax.set_ylim(0, 1)
        tax.set_yticks([])
    _set_track_label(tax, track["label"])
    if ttype != "imshow":
        sns.despine(ax=tax, left=True, bottom=not is_bottom)
    if not is_bottom:
        tax.tick_params(axis="x", bottom=False)


# II Main Functions
class AAPredPlot:
    """
    Plotting class for :class:`AAPred` evaluation and prediction results [Breimann25]_.

    The single home for prediction figures, dispatched by ``kind`` from three methods:

    - :meth:`predict_sample` visualizes **single-protein positional predictions**: the
      per-residue profile (``kind='window'``) and the domain boundary-sensitivity curve
      (``kind='domain'``).
    - :meth:`predict_group` visualizes **across-samples predictions**: score histograms
      (``kind='hist'``), ranked candidates (``kind='ranking'``), two-predictor scatters
      (``kind='scatter'``), survival curves (``kind='cutoff'``), and explanation-similarity
      clustermaps (``kind='clustermap'``).
    - :meth:`eval` visualizes **model/feature-set evaluation**: metric bars per model
      (``kind='eval'``) and grouped benchmark comparisons (``kind='comparison'``).

    .. versionadded:: 1.1.0

    See Also
    --------
    * :class:`AAPred`: the logic class whose results this visualizes.
    """

    def __init__(self):
        """
        See Also
        --------
        * :class:`AAPred`: the logic class whose results this visualizes.

        Examples
        --------
        .. include:: examples/aapred_plot.rst
        """

    @staticmethod
    def predict_sample(data: Optional[pd.DataFrame] = None,
                       kind: str = "window",
                       ax: Optional[Axes] = None,
                       figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                       entry: Optional[str] = None,
                       list_annotations: Optional[List[Dict]] = None,
                       threshold: Optional[Union[int, float]] = None,
                       color: Optional[str] = None,
                       xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None,
                       df_seq: Optional[pd.DataFrame] = None,
                       df_scales: Optional[pd.DataFrame] = None,
                       df_cat: Optional[pd.DataFrame] = None,
                       subcats: Optional[List[str]] = None,
                       df_feat: Optional[pd.DataFrame] = None,
                       highlight: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                       zoom: bool = False,
                       ) -> Tuple[Figure, Axes]:
        """
        Visualize single-protein positional predictions as a multi-track sequence viewer.

        One entry point for the three positional figures; ``kind`` selects the base renderer.
        For ``'window'`` / ``'domain'`` ``data`` is the prediction frame from :meth:`AAPred.predict`;
        the base profile is stacked, top to bottom, with optional extra tracks that all share the
        residue-position x-axis: a **CPP-importance** profile (``df_feat``), one **subcategory** scale
        profile per entry in ``subcats``, the **user annotation** tracks (``list_annotations``), and a
        **sequence** row at the bottom. Any track whose inputs are not provided is simply omitted.

        * ``'window'`` — per-residue profile from :meth:`AAPred.predict` (``level='window'``);
          ``data`` is the ``df_window`` frame (columns ``entry``, ``position``, ``score``). The
          x-axis is the residue position, so every extra track aligns residue-by-residue.
        * ``'domain'`` — boundary-sensitivity curve from :meth:`AAPred.predict` (``level='domain'``);
          ``data`` is the ``df_domain`` frame (columns ``entry``, ``offset``, ``score``, ``is_best``).
          The x-axis is the boundary offset; the residue-anchored tracks (subcategory, sequence)
          map each offset to residue ``tmd_start + offset`` and therefore need ``df_seq`` with a
          ``tmd_start`` column.
        * ``'sequence'`` — a CPP-feature-map-style **heatmap over the complete protein sequence**;
          rows are **subcategories** (``subcats``, or every subcategory in ``df_cat`` capped at 25
          when ``subcats=None``), columns are the **residue positions** ``1..len(sequence)``, and each
          cell is that subcategory's mean scale value at that residue (the same per-residue
          subcategory profile the line tracks use, stacked into a matrix). It needs only ``df_seq``
          (plus ``df_scales`` / ``df_cat``, default-loaded when omitted); ``data`` is **optional** and,
          when given as a ``df_window`` frame, is drawn as a thin prediction track above the heatmap.
          The **sequence** row is drawn below the heatmap and ``highlight`` / ``zoom`` behave exactly
          as for the other kinds (cyan spans over the residue columns; zoom reveals the letters).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : pd.DataFrame, optional
            Prediction frame for the selected ``kind``: the ``df_window`` (``'window'``) or
            ``df_domain`` (``'domain'``) output of :meth:`AAPred.predict`. Required for those two
            kinds; **optional** for ``kind='sequence'`` (which only needs ``df_seq``), where a
            ``df_window`` frame is drawn as a thin prediction track above the heatmap.
        kind : str, default="window"
            Which positional figure to draw; one of ``window``, ``domain``, ``sequence``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created. When given, only the
            base profile is drawn (the stacked extra tracks need their own figure).
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, a per-kind default is used; the
            figure grows in height for each extra track.
        entry : str, optional
            Protein to plot; required only when ``data`` holds more than one ``entry``.
        list_annotations : list of dict, optional
            Per-position annotation tracks; each dict has ``values`` (aligned to the plotted
            positions), ``label`` (str), an optional ``color`` (numeric line tracks) and optional
            ``cmap`` (non-numeric/categorical tracks drawn as an ``imshow`` strip).
        threshold : int or float, optional
            (``kind='window'``) Score drawn as a horizontal dashed decision line.
        color : str, optional
            Base line color; defaults to a house color.
        xlabel : str, optional
            x-axis label; defaults to a per-kind label.
        ylabel : str, optional
            y-axis label; defaults to a per-kind label.
        df_seq : pd.DataFrame, optional
            DataFrame containing an ``entry`` column of unique protein identifiers and a
            ``sequence`` column (and, for ``kind='domain'``, ``tmd_start``), used to draw the
            subcategory profiles and the sequence row. Required for those tracks and **required**
            for ``kind='sequence'`` (it supplies the residue sequence spanned by the heatmap).
        df_scales : pd.DataFrame, optional
            Amino-acid scale matrix (letters x scales) for the subcategory profiles / heatmap cells.
            Defaults to the bundled ``load_scales()`` when this is ``None`` and it is needed (i.e.
            ``subcats`` is given, or always for ``kind='sequence'``).
        df_cat : pd.DataFrame, optional
            Scale classification (``scale_id`` / ``subcategory``) mapping each subcategory to its
            scales. Defaults to the bundled ``load_scales(name='scales_cat')`` when this is ``None``
            and it is needed (``subcats`` given, or always for ``kind='sequence'``).
        subcats : list of str, optional
            Subcategory names; one scale-profile line track is added per name (needs ``df_seq``). For
            ``kind='sequence'`` these become the heatmap **rows**; ``None`` uses every subcategory in
            ``df_cat`` (capped at 25, with a verbose note when capped).
        df_feat : pd.DataFrame, optional
            CPP feature frame (with a ``positions`` column and a ``feat_importance`` or ``abs_auc``
            column) mapped onto positions to draw the CPP-importance track.
        highlight : tuple or list of tuple, optional
            One or more ``(start, stop)`` regions (1-based, inclusive) shaded as a bright-cyan
            vertical span across **every** track (base, importance, subcategory, annotation and
            sequence rows), so single or multiple parts are marked consistently down the whole
            stack. The bounds are given in the plot's x-axis units: residue positions for
            ``kind='window'`` and boundary offsets for ``kind='domain'``. Each ``start`` / ``stop``
            must be an integer with ``start <= stop``.
        zoom : bool, default=False
            If ``True`` and ``highlight`` is given, restrict the shared x-axis to the highlighted
            region(s) (from the minimum ``start`` to the maximum ``stop``, padded by a few residues
            and clamped to the data range). Because the visible window then shortens, the sequence
            row renders per-residue letters for the zoomed span. Applies to both kinds; a no-op when
            ``highlight`` is ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The base-profile axes (the top track).

        See Also
        --------
        * :meth:`AAPred.predict` for the predictions this visualizes.
        * :meth:`AAPredPlot.predict_group` for across-samples figures.
        * :meth:`AAPredPlot.eval` for evaluation figures.
        * :meth:`CPPStructurePlot.plot_linked` / :meth:`CPPStructurePlot.map_structure`: pass the
          same ``highlight`` ``(start, stop)`` regions to mirror the cyan selection shaded here on
          the 3D protein structure (shared ``COLOR_LINK_HIGHLIGHT``).

        Examples
        --------
        .. include:: examples/aapred_plot_sample.rst
        """
        if kind not in LIST_SAMPLE_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_SAMPLE_KINDS}.")
        ut.check_bool(name="zoom", val=zoom)
        highlight = _normalize_highlight(highlight)
        figsize = figsize if figsize is not None else _DICT_SAMPLE_FIGSIZE[kind]
        # 'window' and 'domain' need their prediction frame; 'sequence' can stand on df_seq alone.
        if data is None and kind != "sequence":
            raise ValueError(f"'data' (the prediction frame) is required for kind='{kind}'.")
        if kind == "window":
            return AAPredPlot._plot_window(
                df_window=data, entry=entry, list_annotations=list_annotations, threshold=threshold,
                ax=ax, figsize=figsize, color=color,
                xlabel=_default(xlabel, "Residue position"),
                ylabel=_default(ylabel, "Prediction score"),
                df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat,
                highlight=highlight, zoom=zoom)
        if kind == "domain":
            return AAPredPlot._plot_domain(
                df_domain=data, entry=entry, ax=ax, figsize=figsize, color=color,
                xlabel=_default(xlabel, "Boundary offset [residues]"),
                ylabel=_default(ylabel, "Prediction score"),
                df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat,
                list_annotations=list_annotations, highlight=highlight, zoom=zoom)
        # kind == "sequence"
        return AAPredPlot._plot_sequence(
            data=data, entry=entry, ax=ax, figsize=figsize, color=color,
            xlabel=_default(xlabel, "Residue position"),
            ylabel=_default(ylabel, "Subcategory"),
            df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat,
            list_annotations=list_annotations, threshold=threshold, highlight=highlight, zoom=zoom)

    @staticmethod
    def predict_group(data: Union[pd.DataFrame, ut.ArrayLike1D, ut.ArrayLike2D],
                       kind: str = "hist",
                       ax: Optional[Axes] = None,
                       figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                       labels: Optional[ut.ArrayLike1D] = None,
                       bins: int = 20,
                       thresholds: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       band: bool = False,
                       dict_color: Optional[Dict[Union[int, str], str]] = None,
                       col_name: str = "name",
                       col_score: str = "score",
                       col_group: Optional[str] = None,
                       col_std: Optional[str] = None,
                       group_order: Optional[List[str]] = None,
                       colors: Optional[Union[Dict[str, str], List[str]]] = None,
                       cutoffs: Optional[Tuple[Union[int, float], ...]] = (50, 80),
                       top_n: Optional[int] = None,
                       ascending: bool = False,
                       panel_col: Optional[str] = None,
                       title: Optional[str] = None,
                       scores_y: Optional[ut.ArrayLike1D] = None,
                       marker_size: Union[int, float] = 30,
                       diagonal: bool = True,
                       n_steps: int = 101,
                       cmap: str = "GnBu",
                       xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None,
                       ) -> Tuple[Figure, Axes]:
        """
        Visualize across-samples prediction scores, dispatched by ``kind``.

        One entry point for every group score figure; ``kind`` selects the renderer and ``data``
        is its primary input. For sample-relation views (clustering samples by their feature or
        SHAP vectors) see :meth:`AAPredPlot.group_cluster`.

        * ``'hist'`` — histogram of per-sample scores; ``data`` is the ``scores`` array. By default
          class-separated by ``labels``; with ``band=True`` the bars are instead colored by the
          confidence band they fall into (delimited by ``thresholds``), for scoring unlabeled
          samples. Uses ``labels``, ``bins``, ``thresholds``, ``band``, ``dict_color``, ``colors``,
          ``cmap``, ``xlabel``, ``ylabel``.
        * ``'ranking'`` — ranked-candidate horizontal bars; ``data`` is a per-sample ``df_pred``.
          Uses ``col_name``, ``col_score``, ``col_group``, ``col_std``, ``colors``, ``cutoffs``,
          ``top_n``, ``ascending``, ``xlabel``, ``title``. With ``panel_col`` given, one panel is
          drawn per distinct value of that column, side by side.
        * ``'rank_scatter'`` — per-protein rank scatter: proteins ranked by their maximum score
          (x-axis = rank, y-axis = score) and colored by group, the standard sanity check for a
          deployed per-protein predictor; ``data`` is a per-protein ``df_rank``. Uses
          ``col_score``, ``col_group`` (required here), ``group_order``, ``dict_color``,
          ``thresholds`` (drawn as horizontal score lines), ``marker_size``, ``xlabel``, ``ylabel``.
        * ``'scatter'`` — two-predictor agreement scatter; ``data`` is ``scores_x`` and the required
          ``scores_y`` the y-axis. Uses ``labels``, ``dict_color``, ``marker_size``, ``diagonal``,
          ``xlabel``, ``ylabel``.
        * ``'cutoff'`` — survival curve of the scores; ``data`` is the ``scores`` array. Uses
          ``n_steps``, ``thresholds``, ``xlabel``, ``ylabel``. With ``labels`` given, one curve is
          drawn per group (colored by ``dict_color``) over a common cutoff grid.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : pd.DataFrame or array-like
            Primary input for the selected ``kind`` (see above): a per-sample score array
            (``'hist'``/``'scatter'``/``'cutoff'``), a per-sample ranking frame (``'ranking'``), or a
            per-protein ranking frame (``'rank_scatter'``).
        kind : str, default="hist"
            Which group figure to draw; one of ``hist``, ``ranking``, ``rank_scatter``, ``scatter``,
            ``cutoff``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, a per-kind default is used.
        labels : array-like, optional
            (``kind='hist'``/``'scatter'``/``'cutoff'``) Per-sample class labels used to
            color/separate the data (adds a class legend, or one curve per group).
        bins : int, default=20
            (``kind='hist'``) Number of histogram bins.
        thresholds : int, float, or list, optional
            (``kind='hist'``/``'cutoff'``) One or more score values drawn as vertical dashed lines;
            with ``kind='hist'`` and ``band=True`` they also delimit the confidence bands.
            (``kind='rank_scatter'``) Drawn as horizontal dashed lines on the score axis.
        band : bool, default=False
            (``kind='hist'``) If ``True``, color each histogram bar by the confidence band it falls
            into (bands delimited by ``thresholds``) instead of splitting by ``labels``. Requires
            ``thresholds`` and is mutually exclusive with ``labels``.

            .. versionadded:: 1.1.0
        dict_color : dict, optional
            (``kind='hist'``/``'scatter'``/``'cutoff'``) Mapping ``label -> color``; defaults to the
            locked positive/negative sample palette. (``kind='rank_scatter'``) Mapping
            ``group -> color``; canonical group names (``substrate``, ``non-substrate``,
            ``hold-out``) default to the locked sample palette, with a fallback palette for other
            groups.
        col_name : str, default="name"
            (``kind='ranking'``) Column with the per-item labels shown as y-tick labels.
        col_score : str, default="score"
            (``kind='ranking'``) Column with the numeric prediction score used to rank the bars.
            (``kind='rank_scatter'``) Column with the per-protein max score ranked on the y-axis.
        col_group : str, optional
            (``kind='ranking'``) Column whose distinct values color the bars (adds a class legend).
            (``kind='rank_scatter'``) Column with the per-protein group label used for coloring
            (required for this kind).
        col_std : str, optional
            (``kind='ranking'``) Column with per-item standard deviations, drawn as error bars.
        group_order : list of str, optional
            (``kind='rank_scatter'``) Order in which groups are colored, drawn, and legended;
            defaults to first-appearance order in ``data``.
        colors : dict or list, optional
            (``kind='ranking'``) A ``group -> color`` mapping; defaults to the house categorical
            palette. (``kind='hist'`` with ``band=True``) A list of one color per band (low to high
            score); defaults to a sampling of ``cmap``.
        cutoffs : tuple, optional
            (``kind='ranking'``) x-positions of dashed confidence cut-off lines.
        top_n : int, optional
            (``kind='ranking'``) If given, keep only the top ``top_n`` ranked items.
        ascending : bool, default=False
            (``kind='ranking'``) Sort order; ``False`` ranks the highest score first (on top).
        panel_col : str, optional
            (``kind='ranking'``) Column whose distinct values each get their own ranked-bar panel,
            drawn side by side (requires ``ax=None``); ``None`` draws a single panel. Panels share
            the x-axis and return an array of axes.

            .. versionadded:: 1.1.0
        title : str, optional
            (``kind='ranking'``) Axes title.
        scores_y : array-like, optional
            (``kind='scatter'``, required there) Per-sample scores of the second predictor (y-axis).
        marker_size : int or float, default=30
            (``kind='scatter'``/``'rank_scatter'``) Scatter marker size.
        diagonal : bool, default=True
            (``kind='scatter'``) If ``True``, draw the ``y = x`` agreement line.
        n_steps : int, default=101
            (``kind='cutoff'``) Number of evenly spaced cutoffs between the min and max score.
        cmap : str, default="GnBu"
            (``kind='hist'`` with ``band=True`` and no ``colors``) Colormap sampled for the per-band
            colors.
        xlabel : str, optional
            x-axis label; defaults to a per-kind label.
        ylabel : str, optional
            y-axis label; defaults to a per-kind label.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the requested plot.

        See Also
        --------
        * :meth:`AAPred.predict` for the predictions this visualizes.
        * :meth:`AAPredPlot.predict_sample` for single-protein positional figures.
        * :meth:`AAPredPlot.eval` for evaluation figures.

        Examples
        --------
        .. include:: examples/aapred_plot_group.rst
        """
        if kind not in LIST_GROUP_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_GROUP_KINDS}.")
        figsize = figsize if figsize is not None else _DICT_GROUP_FIGSIZE[kind]
        if kind == "hist":
            return AAPredPlot._plot_hist(
                scores=data, labels=labels, ax=ax, figsize=figsize, bins=bins,
                thresholds=thresholds, band=band, dict_color=dict_color, colors=colors, cmap=cmap,
                xlabel=_default(xlabel, "Prediction score"),
                ylabel=_default(ylabel, "Number of samples"))
        if kind == "ranking":
            return AAPredPlot._plot_ranking(
                df_pred=data, col_name=col_name, col_score=col_score, col_group=col_group,
                col_std=col_std, colors=colors, cutoffs=cutoffs, top_n=top_n, ascending=ascending,
                ax=ax, figsize=figsize, xlabel=_default(xlabel, "Prediction score"), title=title,
                panel_col=panel_col)
        if kind == "rank_scatter":
            return AAPredPlot._plot_rank_scatter(
                df_rank=data, col_score=col_score, col_group=col_group, group_order=group_order,
                dict_color=dict_color, thresholds=thresholds, marker_size=marker_size, ax=ax,
                figsize=figsize, xlabel=_default(xlabel, "Protein rank"),
                ylabel=_default(ylabel, "Max score per protein"))
        if kind == "scatter":
            if scores_y is None:
                raise ValueError("'kind'='scatter' requires 'scores_y' (the second predictor's scores).")
            return AAPredPlot._plot_scatter(
                scores_x=data, scores_y=scores_y, labels=labels, ax=ax, figsize=figsize,
                dict_color=dict_color, marker_size=marker_size, diagonal=diagonal,
                xlabel=_default(xlabel, "Predictor 1 score"),
                ylabel=_default(ylabel, "Predictor 2 score"))
        # kind == "cutoff"
        return AAPredPlot._plot_cutoff(
            scores=data, labels=labels, ax=ax, figsize=figsize, n_steps=n_steps, color=None,
            dict_color=dict_color, thresholds=thresholds, xlabel=_default(xlabel, "Score cutoff"),
            ylabel=_default(ylabel, "Samples above cutoff [%]"))

    @staticmethod
    def group_cluster(X: Union[pd.DataFrame, ut.ArrayLike2D],
                      kind: str = "clustermap",
                      tracks: Optional[List[Dict]] = None,
                      names: Optional[List[str]] = None,
                      cmap: str = "GnBu",
                      figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                      cbar_label: str = "Pearson correlation (r)",
                      title: Optional[str] = None,
                      ) -> Tuple[Figure, Axes]:
        """
        Cluster the sample group by how its members relate, dispatched by ``kind``.

        The sample-relation counterpart of :meth:`AAPredPlot.predict_group` (which plots the
        prediction *scores*): ``group_cluster`` takes the sample x feature matrix ``X`` (feature
        values from :meth:`SequenceFeature.feature_matrix`, or SHAP values from a fitted
        :class:`ShapModel`) and lays the samples out by their similarity, overlaying prediction /
        metadata as **annotation tracks**.

        * ``'clustermap'`` — hierarchically clustered sample x sample correlation heatmap of the
          feature/SHAP vectors, with one or two annotation sidebars (``tracks``).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Per-sample feature or importance matrix, shape ``(n_samples, n_features)``.
        kind : str, default="clustermap"
            Which relation figure to draw; currently ``clustermap``.
        tracks : list of dict, optional
            Ordered annotation tracks drawn as colored sidebars. Each track is a dict with keys
            ``values`` (per-sample labels, required), ``colors`` (a ``label -> color`` mapping or
            list; house palette when omitted), ``title`` (legend title, default ``"Class"``) and
            ``where`` (``"top"``, ``"left"`` or ``"both"``). A single track defaults to ``"both"``
            (mirrored onto both sidebars); with two tracks the first defaults to ``"top"`` and the
            second to ``"left"``. At most two tracks (two sidebars).
        names : list of str, optional
            Per-sample tick labels; defaults to positional indices.
        cmap : str, default="GnBu"
            Colormap for the correlation heatmap.
        figsize : tuple, optional
            Figure size; defaults to a per-kind default (the clustermap owns its figure, so no
            ``ax`` is accepted).
        cbar_label : str, default="Pearson correlation (r)"
            Label of the colorbar.
        title : str, optional
            Figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The clustermap heatmap axes.

        See Also
        --------
        * :meth:`AAPredPlot.predict_group` for across-samples views of the prediction scores.
        * :meth:`ShapModel` and :meth:`SequenceFeature.feature_matrix` for the input matrix.

        Examples
        --------
        .. include:: examples/aapred_plot_group_cluster.rst
        """
        if kind not in LIST_CLUSTER_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_CLUSTER_KINDS}.")
        figsize = figsize if figsize is not None else _DICT_CLUSTER_FIGSIZE[kind]
        # kind == "clustermap"
        return AAPredPlot._plot_clustermap(
            data=X, tracks=tracks, names=names, cmap=cmap, figsize=figsize,
            cbar_label=cbar_label, title=title)

    @staticmethod
    def eval(df_eval: pd.DataFrame,
             kind: str = "eval",
             ax: Optional[Axes] = None,
             figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
             dict_color: Optional[Dict[str, str]] = None,
             baseline: Optional[Union[int, float]] = None,
             group: str = "group",
             condition: str = "condition",
             value: str = "value",
             baseline_label: Optional[str] = None,
             annotate: bool = True,
             annotation_fmt: Optional[str] = None,
             group_order: Optional[List[str]] = None,
             condition_order: Optional[List[str]] = None,
             colors: Optional[Union[List[str], Dict[str, str]]] = None,
             bar_width: Union[int, float] = 0.8,
             xlabel: Optional[str] = None,
             ylabel: str = "Score",
             title: Optional[str] = None,
             ylim: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
             fontsize_annotations: Union[int, float] = 10,
             xtick_rotation: Union[int, float] = 0,
             highlight: Optional[Union[int, str, Tuple[int, int], List[Tuple[int, int]]]] = 1,
             vmin: Optional[Union[int, float]] = None,
             vmax: Optional[Union[int, float]] = None,
             cmap: str = "viridis",
             cbar_label: Optional[str] = None,
             ) -> Tuple[Figure, Axes]:
        """
        Visualize model / feature-set evaluation, dispatched by ``kind``.

        Three evaluation figures share one entry point:

        * ``'eval'`` — grouped bar plot comparing **models** across metrics (hue = model), from the
          long-format ``df_eval`` of :meth:`AAPred.eval` (columns ``model``, ``metric``,
          ``principle``, ``score``, ``score_std``). Cross-validation bars carry ``score_std`` error
          bars and held-out bars are hatched. Uses ``dict_color``, ``baseline``, ``ylabel``.
        * ``'comparison'`` — grouped ``condition`` x ``group`` benchmark barplot with per-bar value
          labels and an optional baseline, from a tidy ``df_eval`` with ``group`` / ``condition`` /
          ``value`` columns. Uses ``group``, ``condition``, ``value``, ``baseline``,
          ``baseline_label``, ``annotate``, ``annotation_fmt``, ``group_order``, ``condition_order``,
          ``colors``, ``bar_width``, ``xlabel``, ``ylabel``, ``title``, ``ylim``,
          ``fontsize_annotations``, ``xtick_rotation``.
        * ``'heatmap'`` — square annotated heatmap of a 2D score grid (``df_eval`` is a wide
          DataFrame whose rows x columns are the two sweep axes and whose cells are the scores),
          with the best cell(s) boxed (``highlight`` selects how many). Consolidates the recurring
          "grid of scores -> seaborn heatmap -> box the best configuration" block. Uses ``annotate``,
          ``annotation_fmt``, ``highlight``, ``vmin``, ``vmax``, ``cmap``, ``cbar_label``, ``title``.

        To compare **CPP parameter combinations** instead, use the feature-optimization protocol
        :func:`aaanalysis.pipe.find_features` and its evaluation-grid :func:`aaanalysis.pipe.plot_eval`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame
            Evaluation table. For ``kind='eval'`` the :meth:`AAPred.eval` output (columns ``model``,
            ``metric``, ``principle``, ``score``, ``score_std``); for ``kind='comparison'`` a tidy
            frame with the ``group`` / ``condition`` / ``value`` columns; for ``kind='heatmap'`` a
            wide numeric grid whose row index and columns are the two sweep axes and whose cells are
            the scores.
        kind : str, default="eval"
            Which evaluation figure to draw; one of ``eval``, ``comparison``, ``heatmap``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, a per-kind default is used.
        dict_color : dict, optional
            (``kind='eval'``) Mapping ``model -> color`` (the bar hue).
        baseline : int or float, optional
            y-value of a dashed chance / baseline line (e.g. ``0.5`` for ``kind='eval'``, ``50`` for
            ``kind='comparison'``). If ``None``, no line is drawn.
        group : str, default="group"
            (``kind='comparison'``) Column whose distinct values become the colored bars (legend).
        condition : str, default="condition"
            (``kind='comparison'``) Column whose distinct values become the x-axis clusters.
        value : str, default="value"
            (``kind='comparison'``) Column with the numeric bar heights.
        baseline_label : str, optional
            (``kind='comparison'``) Legend label for the baseline; ``None`` generates
            ``"chance (<baseline>)"``; ``""`` draws the line without a legend entry.
        annotate : bool, default=True
            (``kind='comparison'``) If ``True``, write each bar's value above it.
            (``kind='heatmap'``) If ``True``, write each cell's value inside it.
        annotation_fmt : str, optional
            (``kind='comparison'``) Format string for the value labels; auto-chosen when ``None``.
            (``kind='heatmap'``) Cell-value format; when ``None``, ``".2f"`` for ``[0, 1]``-scaled
            scores and ``".0f"`` otherwise.
        group_order : list of str, optional
            (``kind='comparison'``) Order of the groups (bars within a cluster).
        condition_order : list of str, optional
            (``kind='comparison'``) Order of the conditions (x-axis clusters).
        colors : list of str or dict, optional
            (``kind='comparison'``) Bar colors aligned to ``group_order`` or a ``group -> color`` dict.
        bar_width : int or float, default=0.8
            (``kind='comparison'``) Total width of each cluster (split across the groups); in (0, 1].
        xlabel : str, optional
            (``kind='comparison'``) x-axis label.
        ylabel : str, default="Score"
            y-axis label.
        title : str, optional
            (``kind='comparison'``/``'heatmap'``) Axes title.
        ylim : tuple, optional
            (``kind='comparison'``) y-axis limits ``(bottom, top)``.
        fontsize_annotations : int or float, default=10
            (``kind='comparison'``) Font size of the per-bar value labels.
        xtick_rotation : int or float, default=0
            (``kind='comparison'``) Rotation (degrees) of the cluster tick labels.
        highlight : int, str, tuple, or list, default=1
            (``kind='heatmap'``) Which cell(s) to box with a bold frame: a positive int ``N`` boxes
            the ``N`` best (highest-value) cells (``1`` = the single best), ``"max"`` / ``"min"`` box
            the single best / worst cell, an explicit ``(row, col)`` (or list of them) boxes those
            cells, and ``None`` boxes nothing.

            .. versionadded:: 1.1.0
        vmin : int or float, optional
            (``kind='heatmap'``) Lower bound of the color scale; auto-scaled when ``None``.

            .. versionadded:: 1.1.0
        vmax : int or float, optional
            (``kind='heatmap'``) Upper bound of the color scale; auto-scaled when ``None``.

            .. versionadded:: 1.1.0
        cmap : str, default="viridis"
            (``kind='heatmap'``) Colormap for the heatmap cells.

            .. versionadded:: 1.1.0
        cbar_label : str, optional
            (``kind='heatmap'``) Label of the colorbar; defaults to ``"Score"``.

            .. versionadded:: 1.1.0

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the requested evaluation plot.

        See Also
        --------
        * :meth:`AAPred.eval` for the evaluation table this visualizes.
        * :func:`aaanalysis.pipe.find_features` and :func:`aaanalysis.pipe.plot_eval` for
          comparing CPP parameter combinations (a heatmap over the parameter grid).

        Examples
        --------
        .. include:: examples/aapred_plot_eval.rst
        """
        if kind not in LIST_EVAL_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_EVAL_KINDS}.")
        if kind == "eval":
            return AAPredPlot._eval_bars(
                df_eval=df_eval, ax=ax, figsize=_default(figsize, (7, 5)),
                dict_color=dict_color, baseline=baseline, ylabel=ylabel)
        if kind == "heatmap":
            return AAPredPlot._plot_heatmap(
                df_eval=df_eval, ax=ax, figsize=_default(figsize, (6, 5)),
                annotate=annotate, annotation_fmt=annotation_fmt,
                highlight=highlight, vmin=vmin, vmax=vmax, cmap=cmap,
                cbar_label=_default(cbar_label, "Score"), title=title)
        # kind == "comparison"
        return AAPredPlot._plot_comparison(
            df_eval=df_eval, group=group, condition=condition, value=value, baseline=baseline,
            baseline_label=baseline_label, annotate=annotate, annotation_fmt=annotation_fmt,
            group_order=group_order, condition_order=condition_order, colors=colors,
            bar_width=bar_width, ax=ax, figsize=_default(figsize, (7, 4.2)), xlabel=xlabel,
            ylabel=ylabel, title=title, ylim=ylim, fontsize_annotations=fontsize_annotations,
            xtick_rotation=xtick_rotation)

    # III Private renderers (one per kind; kept as the original drawing logic)
    @staticmethod
    def _eval_bars(df_eval, ax=None, figsize=(7, 5), dict_color=None, baseline=None,
                   ylabel="Score"):
        """Grouped bar plot comparing methods across metrics (hue = model)."""
        # Check input
        cols = ut.COLS_EVAL_PRED
        ut.check_df(name="df_eval", df=df_eval, cols_required=cols)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        if baseline is not None:
            ut.check_number_val(name="baseline", val=baseline, just_int=False)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Grouped bar plot: metrics on the x-axis, one bar per (feature set x) model x principle.
        # A ``features`` column (from AAPred.eval(baseline=...)) splits each group into the compared
        # feature sets ('cpp' vs the baselines) so their rows are NEVER averaged together; it then
        # becomes the hue. Without it the plot is unchanged (one hued bar per model).
        has_features = ut.COL_FEATURES in df_eval.columns
        metrics = list(dict.fromkeys(df_eval[ut.COL_METRIC].tolist()))
        models = list(dict.fromkeys(df_eval[ut.COL_MODEL].tolist()))
        principles = list(dict.fromkeys(df_eval[ut.COL_PRINCIPLE].tolist()))
        feature_sets = list(dict.fromkeys(df_eval[ut.COL_FEATURES].tolist())) if has_features else [None]
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        dict_color = dict(dict_color) if dict_color is not None else {}
        hue_keys = feature_sets if has_features else models
        clist = ut.plot_get_clist_(n_colors=max(len(hue_keys), 2))
        dict_hue_color = {k: dict_color.get(k, clist[i % len(clist)]) for i, k in enumerate(hue_keys)}
        n_groups = len(models) * len(principles) * len(feature_sets)
        width = 0.8 / max(n_groups, 1)
        x = np.arange(len(metrics))
        idx = 0
        for feat in feature_sets:
            for model in models:
                for principle in principles:
                    mask = (df_eval[ut.COL_MODEL] == model) & (df_eval[ut.COL_PRINCIPLE] == principle)
                    if has_features:
                        mask = mask & (df_eval[ut.COL_FEATURES] == feat)
                    sub = df_eval[mask]
                    heights = [float(sub[sub[ut.COL_METRIC] == m][ut.COL_SCORE].mean()) for m in metrics]
                    errs = [float(sub[sub[ut.COL_METRIC] == m][ut.COL_SCORE_STD].mean()) for m in metrics]
                    errs = [0 if np.isnan(e) else e for e in errs]
                    hatch = "//" if principle == ut.STR_PRINCIPLE_HOLDOUT else None
                    color = dict_hue_color[feat] if has_features else dict_hue_color[model]
                    if has_features:
                        parts = [str(feat)]
                        if len(models) > 1:
                            parts.append(model)
                        if len(principles) > 1:
                            parts.append(principle)
                        label = " · ".join(parts)
                    else:
                        label = model if len(principles) == 1 else f"{model} ({principle})"
                    ax.bar(x + (idx - (n_groups - 1) / 2) * width, heights, width=width,
                           color=color, edgecolor="black", linewidth=0.6,
                           hatch=hatch, yerr=errs, capsize=2.5, label=label)
                    idx += 1
        if baseline is not None:
            ax.axhline(baseline, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_heatmap(df_eval, ax=None, figsize=(6, 5), annotate=True, annotation_fmt=None,
                      highlight="max", vmin=None, vmax=None, cmap="viridis", cbar_label="Score",
                      title=None):
        """Square annotated heatmap of a 2D score grid, boxing the optimal cell."""
        # Check input
        ut.check_df(name="df_eval", df=df_eval, accept_none=False, accept_nan=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_bool(name="annotate", val=annotate)
        ut.check_str(name="annotation_fmt", val=annotation_fmt, accept_none=True)
        ut.check_number_val(name="vmin", val=vmin, accept_none=True)
        ut.check_number_val(name="vmax", val=vmax, accept_none=True)
        ut.check_str(name="cmap", val=cmap, accept_none=False)
        ut.check_str(name="cbar_label", val=cbar_label, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        try:
            data = np.asarray(df_eval.to_numpy(), dtype=float)
        except (ValueError, TypeError):
            raise ValueError("'df_eval' (for kind='heatmap') should contain only numeric scores.")
        if data.size == 0:
            raise ValueError("'df_eval' (empty) should have at least one cell for kind='heatmap'.")
        highlight_cells = _check_highlight_cells(highlight=highlight, data=data)
        if annotation_fmt is None:
            # [0, 1]-scaled scores need decimals; percent-scaled ones read best as integers.
            finite = data[np.isfinite(data)]
            annotation_fmt = ".2f" if finite.size and np.max(np.abs(finite)) <= 1.5 else ".0f"
        # Draw the square annotated heatmap; let matplotlib place a layout-robust colorbar (a
        # free-floating cax positioned from draw-time coords would detach under tight_layout).
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        sns.heatmap(df_eval, vmin=vmin, vmax=vmax, cmap=cmap, annot=annotate, fmt=annotation_fmt,
                    square=True, linewidth=0.1, cbar=False, ax=ax)
        ax.tick_params(left=False, bottom=False)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=0)
        # Box the selected cell(s) with a full-cell frame flush to the grid lines
        for i, j in highlight_cells:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=3,
                                       clip_on=False))
        # Colorbar tracking the axes height, with an edge drawn on the tick (right) side
        cb = fig.colorbar(ax.collections[-1], ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        cb.outline.set_visible(False)
        cb.ax.add_line(plt.Line2D([1, 1], [0, 1], transform=cb.ax.transAxes, color="black",
                                  linewidth=plt.rcParams["axes.linewidth"], clip_on=False))
        if title is not None:
            ax.set_title(title)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_comparison(df_eval, group="group", condition="condition", value="value",
                         baseline=None, baseline_label=None, annotate=True, annotation_fmt=None,
                         group_order=None, condition_order=None, colors=None, bar_width=0.8,
                         ax=None, figsize=(7, 4.2), xlabel=None, ylabel="Score", title=None,
                         ylim=None, fontsize_annotations=10, xtick_rotation=0):
        """Grouped method x condition comparison barplot with value labels and a baseline."""
        # Check input
        ut.check_str(name="group", val=group)
        ut.check_str(name="condition", val=condition)
        ut.check_str(name="value", val=value)
        if len({group, condition, value}) < 3:
            raise ValueError(f"'group', 'condition', and 'value' should be three distinct columns, "
                             f"got group={group!r}, condition={condition!r}, value={value!r}.")
        ut.check_df(name="df_eval", df=df_eval, cols_required=[group, condition, value])
        if len(df_eval) == 0:
            raise ValueError("'df_eval' (0 rows) should contain at least one row.")
        if not pd.api.types.is_numeric_dtype(df_eval[value]):
            raise ValueError(f"'{value}' column of 'df_eval' should be numeric, "
                             f"got dtype '{df_eval[value].dtype}'.")
        ut.check_number_val(name="baseline", val=baseline, accept_none=True, just_int=False)
        ut.check_str(name="baseline_label", val=baseline_label, accept_none=True)
        ut.check_bool(name="annotate", val=annotate)
        ut.check_str(name="annotation_fmt", val=annotation_fmt, accept_none=True)
        ut.check_list_like(name="group_order", val=group_order, accept_none=True)
        ut.check_list_like(name="condition_order", val=condition_order, accept_none=True)
        ut.check_number_range(name="bar_width", val=bar_width, min_val=0, max_val=1, just_int=False)
        if bar_width == 0:
            raise ValueError("'bar_width' should be greater than 0.")
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        ut.check_number_range(name="fontsize_annotations", val=fontsize_annotations, min_val=0,
                              just_int=False)
        ut.check_number_val(name="xtick_rotation", val=xtick_rotation, just_int=False)
        if ylim is not None:
            ut.check_lim(name="ylim", val=ylim)
        # Plot
        fig, ax = plot_comparison_(df_eval=df_eval, group=group, condition=condition, value=value,
                                   baseline=baseline, baseline_label=baseline_label, annotate=annotate,
                                   annotation_fmt=annotation_fmt, group_order=group_order,
                                   condition_order=condition_order, colors=colors, bar_width=bar_width,
                                   ax=ax, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title,
                                   ylim=ylim, fontsize_annotations=fontsize_annotations,
                                   xtick_rotation=xtick_rotation)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_ranking(df_pred, col_name="name", col_score="score", col_group=None, col_std=None,
                      colors=None, cutoffs=(50, 80), top_n=None, ascending=False, ax=None,
                      figsize=None, xlabel="Prediction score", title=None, panel_col=None):
        """Ranked candidates as horizontal bars colored by class, with cut-off lines.

        With ``panel_col`` given, one ranked-bar panel is drawn per distinct value, side by side.
        """
        # Check input
        ut.check_str(name="col_name", val=col_name)
        ut.check_str(name="col_score", val=col_score)
        ut.check_str(name="col_group", val=col_group, accept_none=True)
        ut.check_str(name="col_std", val=col_std, accept_none=True)
        ut.check_str(name="panel_col", val=panel_col, accept_none=True)
        cols_required = [c for c in [col_name, col_score, col_group, col_std, panel_col] if c is not None]
        ut.check_df(name="df_pred", df=df_pred, cols_required=cols_required)
        if len(df_pred) == 0:
            raise ValueError("'df_pred' (0 rows) should contain at least one row.")
        if not pd.api.types.is_numeric_dtype(df_pred[col_score]):
            raise ValueError(f"'{col_score}' column of 'df_pred' should be numeric.")
        ut.check_bool(name="ascending", val=ascending)
        if top_n is not None:
            ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        # Single-panel (default) vs one panel per distinct ``panel_col`` value
        if panel_col is None:
            fig, ax = plot_ranking_(df_pred=df_pred, col_name=col_name, col_score=col_score,
                                    col_group=col_group, col_std=col_std, colors=colors,
                                    cutoffs=cutoffs, top_n=top_n, ascending=ascending, ax=ax,
                                    figsize=figsize, xlabel=xlabel, title=title)
            return ut.FigAxResult(fig, ax)
        if ax is not None:
            raise ValueError("'panel_col' draws multiple panels and requires 'ax' (None).")
        panels = list(dict.fromkeys(df_pred[panel_col].tolist()))
        n_rows_max = max(len(df_pred[df_pred[panel_col] == p]) for p in panels)
        if figsize is None:
            figsize = (5.0 * len(panels), ranking_figheight(n_rows_max))
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=True, squeeze=False)
        axes = axes[0]
        for panel, ax_i in zip(panels, axes):
            d = df_pred[df_pred[panel_col] == panel]
            plot_ranking_(df_pred=d, col_name=col_name, col_score=col_score, col_group=col_group,
                          col_std=col_std, colors=colors, cutoffs=cutoffs, top_n=top_n,
                          ascending=ascending, ax=ax_i, figsize=figsize, xlabel=xlabel,
                          title=str(panel) if title is None else f"{title}: {panel}")
        return ut.FigAxResult(fig, axes)

    @staticmethod
    def _plot_rank_scatter(df_rank, col_score="score", col_group=None, group_order=None,
                           dict_color=None, thresholds=None, marker_size=25, ax=None,
                           figsize=None, xlabel="Protein rank", ylabel="Max score per protein"):
        """Per-protein rank scatter: max-score-per-protein sorted by score, colored by group."""
        # Check input
        if col_group is None:
            raise ValueError("'kind'='rank_scatter' requires 'col_group' (the per-protein group "
                             "column used for coloring).")
        ut.check_str(name="col_score", val=col_score)
        ut.check_str(name="col_group", val=col_group)
        ut.check_df(name="df_rank", df=df_rank, cols_required=[col_score, col_group])
        if len(df_rank) == 0:
            raise ValueError("'df_rank' (0 rows) should contain at least one protein.")
        if not pd.api.types.is_numeric_dtype(df_rank[col_score]):
            raise ValueError(f"'{col_score}' column of 'df_rank' should be numeric.")
        ut.check_list_like(name="group_order", val=group_order, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_number_range(name="marker_size", val=marker_size, min_val=0, just_int=False)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        list_thresholds = []
        if thresholds is not None:
            list_thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
            for i, t in enumerate(list_thresholds):
                ut.check_number_val(name=f"thresholds[{i}]", val=t, just_int=False)
        if group_order is not None:
            missing = set(df_rank[col_group]) - set(group_order)
            if missing:
                raise ValueError(f"'group_order' is missing groups present in 'df_rank': {missing}")
        # Plot
        fig, ax = plot_rank_scatter_(df_rank=df_rank, col_score=col_score, col_group=col_group,
                                     group_order=group_order, dict_color=dict_color,
                                     thresholds=list_thresholds, ax=ax, figsize=figsize,
                                     marker_size=marker_size, xlabel=xlabel, ylabel=ylabel)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_clustermap(data, tracks=None, names=None, cmap="GnBu", figsize=(9, 9),
                         cbar_label="Pearson correlation (r)", title=None):
        """Cluster samples by explanation similarity (correlation of importance vectors)."""
        # Check input
        data = ut.check_X(X=data, min_n_samples=2, min_n_features=1)
        if names is not None:
            ut.check_list_like(name="names", val=names)
            if len(names) != data.shape[0]:
                raise ValueError(f"'names' (n={len(names)}) should match n_samples ({data.shape[0]}).")
        # Resolve the annotation-track list into the (column, row) sidebar arguments. Track
        # labels are purely cosmetic (sidebar coloring), so any hashable class values are allowed.
        (labels, colors, legend_title,
         labels_row, colors_row, legend_title_row) = _resolve_cluster_tracks(tracks, data.shape[0])
        ut.check_str(name="cmap", val=cmap)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="cbar_label", val=cbar_label, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        # Plot
        fig, ax = plot_clustermap_(data=data, names=names, labels=labels, labels_row=labels_row,
                                   colors=colors, colors_row=colors_row, cmap=cmap,
                                   figsize=figsize, cbar_label=cbar_label, title=title,
                                   legend_title=legend_title, legend_title_row=legend_title_row)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_hist(scores, labels=None, ax=None, figsize=(6, 4.5), bins=20, thresholds=None,
                   band=False, dict_color=None, colors=None, cmap="viridis",
                   xlabel="Prediction score", ylabel="Number of samples"):
        """Histogram of per-sample prediction scores, class-separated or confidence-banded."""
        # Check input
        scores = ut.check_array_like(name="scores", val=scores, expected_dim=1)
        labels = ut.check_labels(labels=labels) if labels is not None else None
        check_match_scores_labels(scores=scores, labels=labels)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="bins", val=bins, min_val=1, just_int=True)
        ut.check_bool(name="band", val=band)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_str(name="cmap", val=cmap, accept_none=False)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        list_thresholds = []
        if thresholds is not None:
            list_thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
            for i, t in enumerate(list_thresholds):
                ut.check_number_val(name=f"thresholds[{i}]", val=t, just_int=False)
        if band:
            if not list_thresholds:
                raise ValueError("'band' (True) should be used with 'thresholds' delimiting the "
                                 "confidence bands.")
            if labels is not None:
                raise ValueError("'band' (True) colors bars by score band and should be used with "
                                 "'labels' (None); pass one or the other, not both.")
        # Resolve colors
        dict_color = dict(dict_color) if dict_color is not None else {}
        default_cycle = [ut.COLOR_POS, ut.COLOR_NEG, ut.COLOR_REL_NEG, ut.COLOR_UNL]
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        bin_edges = np.linspace(float(np.min(scores)), float(np.max(scores)), bins + 1)
        if band:
            sorted_ths = sorted(list_thresholds)
            band_colors = _resolve_band_colors(colors=colors, cmap=cmap, n_bands=len(sorted_ths) + 1)
            _, edges, patches = ax.hist(scores, bins=bin_edges, edgecolor="black", linewidth=0.6)
            for patch, left in zip(patches, edges[:-1]):
                patch.set_facecolor(band_colors[_band_index(left, sorted_ths)])
        elif labels is None:
            ax.hist(scores, bins=bin_edges, color=ut.COLOR_POS, edgecolor="black", linewidth=0.6)
        else:
            for i, lab in enumerate(sorted(set(labels))):
                color = dict_color.get(lab, default_cycle[i % len(default_cycle)])
                ax.hist(np.asarray(scores)[np.asarray(labels) == lab], bins=bin_edges, alpha=0.7,
                        color=color, edgecolor="black", linewidth=0.6, label=str(lab))
            ax.legend(frameon=False)
        for t in list_thresholds:
            ax.axvline(t, color="0.3", linestyle="--", linewidth=1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_scatter(scores_x, scores_y, labels=None, ax=None, figsize=(5.5, 5.5), dict_color=None,
                      marker_size=30, diagonal=True, xlabel="Predictor 1 score",
                      ylabel="Predictor 2 score"):
        """2D scatter comparing per-sample scores of two predictors."""
        # Check input
        scores_x = ut.check_array_like(name="scores_x", val=scores_x, expected_dim=1)
        scores_y = ut.check_array_like(name="scores_y", val=scores_y, expected_dim=1)
        if len(scores_x) != len(scores_y):
            raise ValueError(f"'scores_x' (n={len(scores_x)}) and 'scores_y' (n={len(scores_y)}) should match in length.")
        labels = ut.check_labels(labels=labels) if labels is not None else None
        check_match_scores_labels(scores=scores_x, labels=labels)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_number_range(name="marker_size", val=marker_size, min_val=0, just_int=False)
        ut.check_bool(name="diagonal", val=diagonal)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Resolve colors
        dict_color = dict(dict_color) if dict_color is not None else {}
        default_cycle = [ut.COLOR_POS, ut.COLOR_NEG, ut.COLOR_REL_NEG, ut.COLOR_UNL]
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        if diagonal:
            lo = float(min(np.min(scores_x), np.min(scores_y)))
            hi = float(max(np.max(scores_x), np.max(scores_y)))
            ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1, zorder=0)
        if labels is None:
            ax.scatter(scores_x, scores_y, s=marker_size, color=ut.COLOR_POS,
                       edgecolors="white", linewidths=0.3)
        else:
            sx, sy, la = np.asarray(scores_x), np.asarray(scores_y), np.asarray(labels)
            for i, lab in enumerate(sorted(set(labels))):
                color = dict_color.get(lab, default_cycle[i % len(default_cycle)])
                mask = la == lab
                ax.scatter(sx[mask], sy[mask], s=marker_size, color=color,
                           edgecolors="white", linewidths=0.3, label=str(lab))
            ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_cutoff(scores, labels=None, ax=None, figsize=(6, 4.5), n_steps=101, color=None,
                     dict_color=None, thresholds=None, xlabel="Score cutoff",
                     ylabel="Samples above cutoff [%]"):
        """Line plot of the percentage of samples scoring at or above each cutoff.

        With ``labels`` given, one survival curve is drawn per group over a common cutoff grid.
        """
        # Check input
        scores = ut.check_array_like(name="scores", val=scores, expected_dim=1)
        if len(scores) == 0:
            raise ValueError("'scores' (0 values) should contain at least one score.")
        labels = ut.check_list_like(name="labels", val=labels) if labels is not None else None
        if labels is not None and len(labels) != len(scores):
            raise ValueError(f"'labels' (n={len(labels)}) should match 'scores' (n={len(scores)}).")
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="n_steps", val=n_steps, min_val=2, just_int=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        list_thresholds = []
        if thresholds is not None:
            list_thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
            for i, t in enumerate(list_thresholds):
                ut.check_number_val(name=f"thresholds[{i}]", val=t, just_int=False)
        # Compute over a common cutoff grid so grouped curves are comparable
        scores = np.asarray(scores, dtype=float)
        cutoffs = np.linspace(float(np.min(scores)), float(np.max(scores)), n_steps)
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        if labels is None:
            pct = np.array([100.0 * np.mean(scores >= c) for c in cutoffs])
            ax.plot(cutoffs, pct, color=color or ut.COLOR_FEAT_POS, linewidth=2)
        else:
            labels = np.asarray(labels)
            dict_color = dict(dict_color) if dict_color is not None else {}
            default_cycle = [ut.COLOR_POS, ut.COLOR_NEG, ut.COLOR_REL_NEG, ut.COLOR_UNL]
            for i, lab in enumerate(sorted(set(labels.tolist()))):
                g_scores = scores[labels == lab]
                pct = np.array([100.0 * np.mean(g_scores >= c) for c in cutoffs])
                ax.plot(cutoffs, pct, color=dict_color.get(lab, default_cycle[i % len(default_cycle)]),
                        linewidth=2, label=str(lab))
            ax.legend(frameon=False)
        for t in list_thresholds:
            ax.axvline(t, color="0.3", linestyle="--", linewidth=1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _check_track_inputs(df_seq=None, df_scales=None, df_cat=None, subcats=None, df_feat=None):
        """Validate and default-resolve the multi-track viewer inputs.

        Returns the (possibly default-loaded) ``(df_scales, df_cat, subcats)`` triple; the
        bundled scales / classification are loaded only when ``subcats`` is requested and the
        matching frame is ``None``.
        """
        if df_seq is not None:
            ut.check_df(name="df_seq", df=df_seq, cols_required=[ut.COL_ENTRY])
        if df_scales is not None:
            ut.check_df(name="df_scales", df=df_scales)
        if df_cat is not None:
            ut.check_df(name="df_cat", df=df_cat, cols_required=[ut.COL_SCALE_ID, ut.COL_SUBCAT])
        if df_feat is not None:
            ut.check_df(name="df_feat", df=df_feat, cols_required=[ut.COL_FEATURE])
        if subcats is not None:
            subcats = ut.check_list_like(name="subcats", val=subcats, accept_str=True, accept_none=True)
            if isinstance(subcats, str):
                subcats = [subcats]
        if subcats and (df_scales is None or df_cat is None):
            from aaanalysis.data_handling import load_scales
            if df_scales is None:
                df_scales = load_scales(name="scales")
            if df_cat is None:
                df_cat = load_scales(name="scales_cat")
        return df_scales, df_cat, subcats

    @staticmethod
    def _draw_extra_tracks(track_axes, tracks, x, visible_range=None):
        """Render every extra track and return the bottom-most axes (for the x-label)."""
        for i, (tax, track) in enumerate(zip(track_axes, tracks)):
            _draw_track(tax, track, x, is_bottom=(i == len(tracks) - 1),
                        visible_range=visible_range)
        return track_axes[-1] if track_axes else None

    @staticmethod
    def _plot_window(df_window, entry=None, list_annotations=None, threshold=None, ax=None,
                     figsize=(10, 4), color=None, xlabel="Residue position",
                     ylabel="Prediction score", df_seq=None, df_scales=None, df_cat=None,
                     subcats=None, df_feat=None, highlight=None, zoom=False):
        """Per-residue prediction profile from AAPred.predict(level='window'), with extra tracks."""
        # Check input
        ut.check_df(name="df_window", df=df_window,
                    cols_required=[ut.COL_ENTRY, ut.COL_RESIDUE_POS, ut.COL_SCORE])
        ut.check_str(name="entry", val=entry, accept_none=True)
        ut.check_list_like(name="list_annotations", val=list_annotations, accept_none=True)
        if threshold is not None:
            ut.check_number_val(name="threshold", val=threshold, just_int=False)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        df_scales, df_cat, subcats = AAPredPlot._check_track_inputs(
            df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat)
        # Resolve the entry
        entries = list(dict.fromkeys(df_window[ut.COL_ENTRY].tolist()))
        if entry is None:
            if len(entries) > 1:
                raise ValueError(f"'df_window' contains multiple entries {entries}; pass 'entry='.")
            entry = entries[0]
        elif entry not in entries:
            raise ValueError(f"'entry' ({entry}) not in 'df_window' entries {entries}.")
        sub = df_window[df_window[ut.COL_ENTRY] == entry].sort_values(ut.COL_RESIDUE_POS)
        pos = sub[ut.COL_RESIDUE_POS].to_numpy()
        score = sub[ut.COL_SCORE].to_numpy()
        # Resolve the zoom window (min start to max stop, padded), so the sequence row keys
        # per-residue letters off the *visible* span rather than the full data length.
        visible_range = _visible_range(highlight, pos, zoom)
        # Build the extra tracks (importance, subcats, user annotations, sequence) + layout
        seq, tmd_start = _entry_sequence(df_seq, entry)
        tracks = _build_extra_tracks(kind="window", x=pos, seq=seq, tmd_start=tmd_start,
                                     df_scales=df_scales, df_cat=df_cat, subcats=subcats,
                                     df_feat=df_feat, list_annotations=list_annotations,
                                     visible_range=visible_range)
        fig, ax, track_axes = _positional_layout(ax=ax, figsize=figsize, tracks=tracks)
        # Draw the base profile
        ax.plot(pos, score, color=color or ut.COLOR_FEAT_NEG, linewidth=1.2)
        if threshold is not None:
            ax.axhline(threshold, color="0.4", linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        # Shade the highlighted region(s) behind the data on every track axes; zoom last.
        _draw_highlight_spans([ax] + track_axes, highlight)
        if visible_range is not None:
            ax.set_xlim(*visible_range)
        elif len(pos):
            ax.set_xlim(pos.min(), pos.max())
        sns.despine(ax=ax, bottom=bool(track_axes))
        if track_axes:
            ax.tick_params(axis="x", bottom=False)
        # Draw the extra tracks; the x-label goes on the bottom-most axes
        bottom = AAPredPlot._draw_extra_tracks(track_axes, tracks, pos, visible_range=visible_range)
        (bottom if bottom is not None else ax).set_xlabel(xlabel)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _plot_domain(df_domain, entry=None, ax=None, figsize=(6, 4.5), color=None,
                     xlabel="Boundary offset [residues]", ylabel="Prediction score",
                     df_seq=None, df_scales=None, df_cat=None, subcats=None, df_feat=None,
                     list_annotations=None, highlight=None, zoom=False):
        """Domain boundary-sensitivity plot from AAPred.predict(level='domain'), with extra tracks."""
        # Check input
        ut.check_df(name="df_domain", df=df_domain,
                    cols_required=[ut.COL_ENTRY, ut.COL_OFFSET, ut.COL_SCORE])
        ut.check_str(name="entry", val=entry, accept_none=True)
        ut.check_list_like(name="list_annotations", val=list_annotations, accept_none=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        df_scales, df_cat, subcats = AAPredPlot._check_track_inputs(
            df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat)
        # Resolve the entry
        entries = list(dict.fromkeys(df_domain[ut.COL_ENTRY].tolist()))
        if entry is None:
            if len(entries) > 1:
                raise ValueError(f"'df_domain' contains multiple entries {entries}; pass 'entry='.")
            entry = entries[0]
        elif entry not in entries:
            raise ValueError(f"'entry' ({entry}) not in 'df_domain' entries {entries}.")
        sub = df_domain[df_domain[ut.COL_ENTRY] == entry].sort_values(ut.COL_OFFSET)
        offsets = sub[ut.COL_OFFSET].to_numpy()
        score = sub[ut.COL_SCORE].to_numpy()
        # Resolve the zoom window (highlight bounds are boundary offsets here, the x-axis).
        visible_range = _visible_range(highlight, offsets, zoom)
        # Build the extra tracks (importance, subcats, user annotations, sequence) + layout
        seq, tmd_start = _entry_sequence(df_seq, entry)
        tracks = _build_extra_tracks(kind="domain", x=offsets, seq=seq, tmd_start=tmd_start,
                                     df_scales=df_scales, df_cat=df_cat, subcats=subcats,
                                     df_feat=df_feat, list_annotations=list_annotations,
                                     visible_range=visible_range)
        fig, ax, track_axes = _positional_layout(ax=ax, figsize=figsize, tracks=tracks)
        # Draw the base curve
        ax.plot(offsets, score, color=color or ut.COLOR_FEAT_POS, marker="o", linewidth=1.6)
        i_best = int(np.argmax(score))
        ax.scatter([offsets[i_best]], [score[i_best]], s=110, marker="*",
                   color=ut.COLOR_FEAT_POS, edgecolors="black", zorder=5, label="best")
        ax.axvline(0, color="0.6", linestyle="--", linewidth=1)  # annotated boundary
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        # Shade the highlighted region(s) behind the data on every track axes; zoom last.
        _draw_highlight_spans([ax] + track_axes, highlight)
        if visible_range is not None:
            ax.set_xlim(*visible_range)
        sns.despine(ax=ax, bottom=bool(track_axes))
        if track_axes:
            ax.tick_params(axis="x", bottom=False)
        # Draw the extra tracks; the x-label goes on the bottom-most axes
        bottom = AAPredPlot._draw_extra_tracks(track_axes, tracks, offsets,
                                               visible_range=visible_range)
        (bottom if bottom is not None else ax).set_xlabel(xlabel)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def _sequence_matrix(seq, subcats, df_scales, df_cat):
        """Stack the per-residue subcategory scale profiles into a ``(n_rows, len(seq))`` matrix.

        Reuses the per-residue subcategory profile (``_subcat_profile``) that feeds the line
        tracks: one row vector per subcategory (``profile[r] = mean scale value at residue r``),
        vertically stacked. Subcategories with no scale in ``df_scales`` are dropped. Returns
        ``(matrix, row_labels)`` where ``row_labels`` are the kept subcategory names.
        """
        rows, row_labels = [], []
        for name in subcats:
            scale_ids = df_cat[df_cat[ut.COL_SUBCAT] == name][ut.COL_SCALE_ID].tolist()
            per_res = _subcat_profile(seq, scale_ids, df_scales)
            if per_res is None:
                continue
            rows.append(per_res)
            row_labels.append(name)
        if not rows:
            return None, []
        return np.vstack(rows), row_labels

    @staticmethod
    def _plot_sequence(data=None, entry=None, ax=None, figsize=(12, 6), color=None,
                       xlabel="Residue position", ylabel="Subcategory", df_seq=None,
                       df_scales=None, df_cat=None, subcats=None, df_feat=None,
                       list_annotations=None, threshold=None, highlight=None, zoom=False):
        """Full-sequence subcategory x residue scale-value heatmap (CPP feature-map style).

        The heatmap is the base track; a ``df_window`` ``data`` frame (if given) is drawn as a
        thin prediction profile above it, and the sequence row below. ``highlight`` / ``zoom``
        act on the residue-position columns exactly as for the window/domain viewers.
        """
        # Check input
        if data is not None:
            ut.check_df(name="data", df=data, cols_required=[ut.COL_ENTRY])
        ut.check_str(name="entry", val=entry, accept_none=True)
        ut.check_list_like(name="list_annotations", val=list_annotations, accept_none=True)
        if threshold is not None:
            ut.check_number_val(name="threshold", val=threshold, just_int=False)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # df_seq supplies the residue sequence the heatmap spans, so it is required here.
        if df_seq is None:
            raise ValueError("'df_seq' is required for kind='sequence' (it supplies the sequence "
                             "spanned by the heatmap columns).")
        df_scales, df_cat, subcats = AAPredPlot._check_track_inputs(
            df_seq=df_seq, df_scales=df_scales, df_cat=df_cat, subcats=subcats, df_feat=df_feat)
        # The heatmap always needs both scale frames (even when subcats=None), so default-load.
        if df_scales is None or df_cat is None:
            from aaanalysis.data_handling import load_scales
            if df_scales is None:
                df_scales = load_scales(name="scales")
            if df_cat is None:
                df_cat = load_scales(name="scales_cat")
        # Resolve the entry (from the prediction frame when given, else from df_seq)
        seq_entries = list(dict.fromkeys(df_seq[ut.COL_ENTRY].tolist()))
        if entry is None:
            src_entries = (list(dict.fromkeys(data[ut.COL_ENTRY].tolist()))
                           if data is not None else seq_entries)
            if len(src_entries) > 1:
                raise ValueError(f"Multiple entries {src_entries}; pass 'entry='.")
            if len(src_entries) == 0:
                raise ValueError("'df_seq' contains no entries.")
            entry = src_entries[0]
        seq, _ = _entry_sequence(df_seq, entry)
        if seq is None:
            raise ValueError(f"'entry' ({entry}) not found in 'df_seq' (needs an 'entry' and a "
                             f"'sequence' column).")
        # Resolve the subcategory rows (all subcategories capped when subcats is None)
        if subcats:
            subcat_names = list(subcats)
        else:
            subcat_names = list(dict.fromkeys(df_cat[ut.COL_SUBCAT].tolist()))
            if len(subcat_names) > _SUBCAT_ROW_CAP:
                if ut.check_verbose(False):
                    ut.print_out(f"Note: {len(subcat_names)} subcategories found in 'df_cat'; "
                                 f"showing the first {_SUBCAT_ROW_CAP}. Pass 'subcats=' to choose.")
                subcat_names = subcat_names[:_SUBCAT_ROW_CAP]
        # Build the subcategory x residue matrix (rows stacked from the per-residue profiles)
        matrix, row_labels = AAPredPlot._sequence_matrix(seq, subcat_names, df_scales, df_cat)
        if matrix is None:
            raise ValueError("No subcategory produced a scale profile; check 'subcats', 'df_cat' "
                             "and 'df_scales'.")
        n_rows, seq_len = matrix.shape
        residues = np.arange(1, seq_len + 1)
        # Optional prediction profile (df_window-style: per-residue position + score) above the map
        pred_pos = pred_score = None
        if data is not None and ut.COL_RESIDUE_POS in data.columns and ut.COL_SCORE in data.columns:
            sub = data[data[ut.COL_ENTRY] == entry].sort_values(ut.COL_RESIDUE_POS)
            if len(sub):
                pred_pos = sub[ut.COL_RESIDUE_POS].to_numpy()
                pred_score = sub[ut.COL_SCORE].to_numpy()
        # Zoom window (highlight bounds are residue positions here, the heatmap x-axis)
        visible_range = _visible_range(highlight, residues, zoom)
        # Layout: [optional prediction track] / heatmap / sequence row, sharing the residue x-axis,
        # plus a dedicated colorbar column. A user-supplied ax only carries the heatmap.
        include_pred = pred_pos is not None
        if ax is not None:
            fig = ax.figure
            heatmap_ax, pred_ax, seq_ax = ax, None, None
            all_axes = [heatmap_ax]
        else:
            heights = ([0.9] if include_pred else []) + [max(0.4 * n_rows, 3.0), 0.5]
            fig_h = figsize[1] + 0.22 * max(n_rows - 8, 0)
            n_grid = len(heights)
            gs = plt.figure(figsize=(figsize[0], fig_h)).add_gridspec(
                n_grid, 2, width_ratios=[40, 1], height_ratios=heights, hspace=0.12, wspace=0.03)
            fig = gs.figure
            main_axes, prev = [], None
            for r in range(n_grid):
                a = fig.add_subplot(gs[r, 0], sharex=prev) if prev is not None else fig.add_subplot(gs[r, 0])
                main_axes.append(a)
                prev = a
            cax = fig.add_subplot(gs[:, 1])
            i = 0
            pred_ax = main_axes[i] if include_pred else None
            i += int(include_pred)
            heatmap_ax = main_axes[i]
            seq_ax = main_axes[i + 1]
            all_axes = [a for a in [pred_ax, heatmap_ax, seq_ax] if a is not None]
        # Heatmap: residues on the x-axis (extent in residue units so tracks/highlight align),
        # subcategories top-to-bottom on the y-axis.
        im = heatmap_ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest",
                               extent=[0.5, seq_len + 0.5, n_rows, 0])
        heatmap_ax.set_yticks(np.arange(n_rows) + 0.5)
        heatmap_ax.set_yticklabels(row_labels, fontsize=8)
        heatmap_ax.set_ylabel(ylabel)
        if ax is not None:
            fig.colorbar(im, ax=heatmap_ax, fraction=0.025, pad=0.02).set_label("scale value",
                                                                                fontsize=8)
        else:
            fig.colorbar(im, cax=cax).set_label("scale value", fontsize=8)
        # Prediction track above the heatmap (optional)
        if pred_ax is not None:
            pred_ax.plot(pred_pos, pred_score, color=color or ut.COLOR_FEAT_NEG, linewidth=1.2)
            if threshold is not None:
                pred_ax.axhline(threshold, color="0.4", linestyle="--", linewidth=1.2)
            pred_ax.set_ylim(0, 1)
            pred_ax.set_yticks([])
            _set_track_label(pred_ax, "Prediction")
            sns.despine(ax=pred_ax, left=True, bottom=True)
        # Sequence row below the heatmap (letters only when the visible span is short enough)
        if seq_ax is not None:
            seq_track = dict(type="seq", letters=list(seq), label="Sequence",
                             show=_count_visible(residues, visible_range) <= _SEQ_ROW_MAX)
            _draw_track(seq_ax, seq_track, residues, is_bottom=True, visible_range=visible_range)
        # Highlight spans behind the data on every axes; then zoom the shared x-axis.
        _draw_highlight_spans(all_axes, highlight)
        if visible_range is not None:
            heatmap_ax.set_xlim(*visible_range)
        else:
            heatmap_ax.set_xlim(0.5, seq_len + 0.5)
        # Only the bottom-most axes shows the x tick labels + x-label
        for a in all_axes[:-1]:
            a.tick_params(axis="x", labelbottom=False)
        all_axes[-1].set_xlabel(xlabel)
        return ut.FigAxResult(fig, heatmap_ax)
