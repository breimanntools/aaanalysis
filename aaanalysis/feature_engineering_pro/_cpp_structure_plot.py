"""
This is a script for the frontend of the CPPStructurePlot class, painting
per-residue CPP / CPP-SHAP feature impact onto a 3D protein structure.
"""
import os
import itertools
import tempfile
import warnings
from typing import Optional, List, Tuple, Union, Literal, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import aaanalysis.utils as ut

from ._backend.cpp_struct.predict import build_builtin_predictor
from ._backend.cpp_struct.mapping import compute_residue_impact
from ._backend.cpp_struct.structure import (load_structure, extract_chain_residues,
                                            residue_numbers)
from ._backend.cpp_struct.render import (render_py3dmol, py3dmol_available,
                                         _stick_radius, _read_structure_text)
from ._backend.cpp_struct.colors import impact_to_hex, plddt_to_hex
from ._backend.cpp_struct.view import StructureView, CombinedView, LinkedView
from ._backend.cpp_struct.linked_html import (build_linked_html, build_linked_html_multi,
                                              _columns)

LIST_MODES = ["impact", "plddt"]
LIST_FOCUS = ["whole", "fade", "zoom"]
LIST_OUTPUTS = ["widget", "html", "static"]
# Multi-site live HTML (explore(output="html", sites=...)): each baked site embeds a feature-map
# image and costs one predictor refit, so warn past _WARN and hard-cap at _MAX (runaway guard).
_WARN_SITES = 40
_MAX_SITES = 200
# Per-call counter for unique linked-view DOM ids (deterministic within a kernel session,
# so committed notebook outputs are stable, but unique across views on one page).
_LINKED_UID = itertools.count(1)


# I Helper Functions
def check_focus_region(focus_region=None):
    """Validate ``focus_region``: ``None``, a ``(start, stop)`` tuple, or a list of them."""
    if focus_region is None:
        return None
    ranges = [focus_region] if isinstance(focus_region, tuple) else focus_region
    if not isinstance(ranges, list):
        raise ValueError(f"'focus_region' ({focus_region}) should be a (start, stop) "
                         f"tuple or a list of such tuples")
    for rng in ranges:
        ut.check_tuple(name="focus_region range", val=rng, n=2, check_number=True)
        if int(rng[0]) > int(rng[1]):
            raise ValueError(f"'focus_region' range ({rng}) should have start <= stop")
    return ranges


def _resolve_window_resis(focus, focus_region, positions_union):
    """Resolve the set of residues that define the focus window (None for 'whole')."""
    if focus == "whole":
        return None
    if focus_region is None:
        return set(positions_union)
    ranges = [focus_region] if isinstance(focus_region, tuple) else focus_region
    resis = set()
    for start, stop in ranges:
        resis.update(range(int(start), int(stop) + 1))
    return resis


def _check_feature_map_kws(feature_map_kws=None):
    """Validate ``feature_map_kws``: a dict whose keys this method does not already own."""
    if feature_map_kws is None:
        return {}
    ut.check_dict(name="feature_map_kws", val=feature_map_kws)
    owned = {"df_feat", "col_val", "col_imp", "shap_plot", "tmd_len", "start",
             "tmd_seq", "jmd_n_seq", "jmd_c_seq", "jmd_n_len", "jmd_c_len"}
    clash = owned.intersection(feature_map_kws)
    if clash:
        raise ValueError(f"'feature_map_kws' ({sorted(clash)}) should not set keys already "
                         f"controlled by plot_combined: {sorted(owned)}")
    return dict(feature_map_kws)


def _require_ipywidgets():
    """Import ipywidgets or raise the friendly pro-install hint (interactive needs it)."""
    try:
        import ipywidgets
        return ipywidgets
    except ImportError as e:
        raise RuntimeError("CPPStructurePlot.interactive requires the optional 'ipywidgets' "
                           "package; install it via \"pip install 'aaanalysis[pro]'\" (or "
                           "\"pip install ipywidgets\"). The static map_structure / "
                           "plot_combined work without it.") from e


def _require_py3dmol():
    """Raise the friendly pro-install hint if py3Dmol (the structure renderer) is absent."""
    if not py3dmol_available():
        raise RuntimeError("CPPStructurePlot structure rendering requires the optional "
                           "'py3Dmol' package; install it via \"pip install 'aaanalysis[pro]'\" "
                           "(or \"pip install py3Dmol\").")


# II Main Functions
class CPPStructurePlot:
    """
    Plotting class for painting :class:`CPP` feature impact onto a 3D protein structure
    (**[pro]**, requires ``aaanalysis[pro]``) [Breimann25]_.

    .. warning::

        **Experimental.** This class and its methods are under active development; their API
        (signatures, defaults, return objects) may change between minor releases without the
        usual deprecation cycle. Pin a version if you depend on the current behaviour.

    Each feature's signed impact is mapped to the residue positions it spans and painted
    residue-by-residue onto the protein cartoon, rendered with the interactive
    `py3Dmol <https://pypi.org/project/py3Dmol/>`_ viewer. A red-white-blue ramp shows where
    features raise (red) or lower (blue) the prediction; an AlphaFold pLDDT mode shows
    per-residue model confidence instead.

    Five methods drive it: :meth:`map_structure` returns a ``StructureView`` (the interactive
    3D cartoon); :meth:`plot_combined` returns a ``CombinedView`` (the cartoon next to the
    :meth:`CPPPlot.feature_map` image, the deployed app's layout); :meth:`plot_linked` returns a
    ``LinkedView`` (a self-contained HTML where hovering a feature-map column highlights the
    matching residue); :meth:`interactive` returns a live ipywidgets explorer (a site slider that
    re-predicts and a feature-map-to-structure highlight link); and :meth:`explore` is the
    integrated one call (a built-in per-site predictor plus a selectable ``output`` of widget /
    HTML / static). All render real 3D structures via py3Dmol — there is no matplotlib structure
    fallback.

    .. versionadded:: 1.1.0

    Notes
    -----
    * The ``jmd_n_len`` and ``jmd_c_len`` values supplied at construction are stored as
      ``_jmd_n_len`` and ``_jmd_c_len`` and reused by the plot methods, mirroring
      :class:`CPPPlot` so juxta-membrane domain (JMD) lengths stay consistent.
    * This is a ``pro`` feature: ``biopython`` parses structures, ``py3Dmol`` renders them, and
      ``ipywidgets`` powers :meth:`interactive` — all in the ``pro`` extra.

    """
    def __init__(self,
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        jmd_n_len : int, default=10
            Length of JMD-N (>=0). Must match the value used when the features were generated.
        jmd_c_len : int, default=10
            Length of JMD-C (>=0). Must match the value used when the features were generated.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Not required for
            structure mapping; forwarded to :class:`CPPPlot` by :meth:`plot_combined`.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for the scales. Not required for structure mapping; forwarded
            to :class:`CPPPlot` by :meth:`plot_combined` (must cover the scales in ``df_feat``).
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * :class:`CPPPlot` : the group- and sample-level CPP result plots.
        * :class:`ShapModel` : produces the sample-level ``feat_impact`` column painted here.
        * :class:`StructurePreprocessor` : parses PDB / CIF / AlphaFold files and fetches AlphaFold models.

        Examples
        --------
        .. include:: examples/csp_map_structure.rst
        """
        # Check input
        verbose = ut.check_verbose(verbose)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        if df_scales is not None:
            ut.check_df(name="df_scales", df=df_scales)
        if df_cat is not None:
            ut.check_df(name="df_cat", df=df_cat)
        # General settings
        self._verbose = verbose
        self._df_scales = df_scales
        self._df_cat = df_cat
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    def map_structure(self,
                      df_feat: pd.DataFrame,
                      pdb: Optional[str] = None,
                      uniprot: Optional[str] = None,
                      col_imp: str = ut.COL_FEAT_IMPACT,
                      tmd_len: int = 20,
                      start: int = 1,
                      chain: Optional[str] = None,
                      sequence: Optional[str] = None,
                      mode: Literal["impact", "plddt"] = "impact",
                      focus: Literal["whole", "fade", "zoom"] = "whole",
                      focus_region: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                      size_by_impact: bool = True,
                      normalize_by_span: bool = False,
                      ) -> StructureView:
        """
        Paint per-residue CPP feature impact onto an interactive 3D protein structure.

        Each feature in ``df_feat`` is mapped to the residue positions it spans (shifted to
        absolute residue numbers by ``start``) and its ``col_imp`` value is aggregated per
        position. By default (``normalize_by_span=False``) each feature's full signed impact is
        added to every residue it spans, reproducing the deployed app's per-residue colouring;
        set ``normalize_by_span=True`` for the span-normalized sum used by :meth:`CPPPlot.profile`.
        The per-residue signed impact is then painted onto the structure cartoon.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a ``feature`` column and the signed per-feature impact column
            ``col_imp`` (e.g. from :meth:`ShapModel.add_feat_impact` or :meth:`CPP.run`).
        pdb : str, optional
            Path to a ``.pdb`` / ``.cif`` structure file. Exactly one of ``pdb`` or ``uniprot``
            must be given.
        uniprot : str, optional
            UniProt accession; the AlphaFold model is fetched from AlphaFold DB into a temporary
            folder via :meth:`StructurePreprocessor.fetch_alphafold`. Exactly one of ``pdb`` or
            ``uniprot`` must be given.
        col_imp : str, default='feat_impact'
            Column of ``df_feat`` holding the signed per-feature impact to paint.
        tmd_len : int, default=20
            Length of the TMD (>=1). Must match the value used when the features were generated.
        start : int, default=1
            Absolute residue number of the first JMD-N residue in the structure (>=0); shifts
            window-relative feature positions onto the structure's residue numbering.
        chain : str, optional
            Chain id to render. Default selects the best-matching chain when ``sequence`` is
            given, otherwise the first amino-acid chain.
        sequence : str, optional
            Full protein sequence; enables best-matching-chain selection (reusing the structure
            backend's alignment) and a sanity check that ``start`` lines up with the structure.
        mode : {'impact', 'plddt'}, default='impact'
            ``'impact'`` paints the red-white-blue feature-impact ramp; ``'plddt'`` paints the
            AlphaFold pLDDT confidence palette.
        focus : {'whole', 'fade', 'zoom'}, default='whole'
            ``'whole'`` styles every residue equally; ``'fade'`` ghosts residues outside the
            window; ``'zoom'`` points the camera at the window.
        focus_region : tuple or list of tuples, optional
            ``(start, stop)`` residue range (or list of ranges) defining the focus window.
            Default derives the window from the union of ``df_feat`` positions.
        size_by_impact : bool, default=True
            If ``True``, draw a stick whose radius is proportional to ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            If ``False`` (default), add each feature's full impact to every residue it spans
            (app-fidelity colouring). If ``True``, divide each feature's impact by its span
            count first (the span-normalized sum of :meth:`CPPPlot.profile` and the
            :meth:`CPPPlot.feature_map` top per-position bar).

        Returns
        -------
        view : StructureView
            A thin wrapper over the interactive py3Dmol view exposing ``show()``,
            ``write_html(path)``, and ``_repr_html_`` for inline display, plus the mapped
            ``dict_impact`` / ``max_abs`` for inspection.

        Notes
        -----
        ``tmd_len``, ``start``, ``jmd_n_len`` and ``jmd_c_len`` must match the geometry used when
        the features were generated, otherwise the impact lands on the wrong residues.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. an unknown ``mode`` / ``focus``, neither or both of
            ``pdb`` / ``uniprot``, a ``df_feat`` missing ``col_imp``, or an unknown ``chain``).
        RuntimeError
            If py3Dmol is not installed, or an AlphaFold model for ``uniprot`` cannot be fetched.

        Examples
        --------
        .. include:: examples/csp_map_structure.rst
        """
        # Validate
        ut.check_df(name="df_feat", df=df_feat, cols_required=[ut.COL_FEATURE, col_imp])
        ut.check_str(name="col_imp", val=col_imp)
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
        ut.check_str_options(name="focus", val=focus, list_str_options=LIST_FOCUS)
        ut.check_bool(name="size_by_impact", val=size_by_impact)
        ut.check_bool(name="normalize_by_span", val=normalize_by_span)
        if chain is not None:
            ut.check_str(name="chain", val=chain)
        if sequence is not None:
            ut.check_str(name="sequence", val=sequence)
        focus_region = check_focus_region(focus_region=focus_region)
        if (pdb is None) == (uniprot is None):
            raise ValueError("Exactly one of 'pdb' or 'uniprot' should be given "
                             f"(got pdb={pdb}, uniprot={uniprot})")
        _require_py3dmol()

        # Compute per-residue impact (shared CPP feature-position backend)
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=col_imp, start=start, tmd_len=tmd_len,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
            normalize_by_span=normalize_by_span)
        window_resis = _resolve_window_resis(focus, focus_region, positions_union)

        # Resolve the structure file, parse it, and render inside one temp context
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
            self._check_start_alignment(records, positions_union, identity, sequence)
            view = render_py3dmol(pdb_path=pdb_path, records=records, dict_impact=dict_impact,
                                  max_abs=max_abs, mode=mode, focus=focus,
                                  window_resis=window_resis, size_by_impact=size_by_impact,
                                  chain_id=chain_id)

        if self._verbose:
            struct_resis = residue_numbers(records)
            n_painted = sum(1 for r in struct_resis if abs(dict_impact.get(r, 0.0)) > 0)
            ut.print_out(f"CPPStructurePlot: mapped {len(df_feat)} features onto "
                         f"{len(records)} residues ({n_painted} carry non-zero impact), "
                         f"mode='{mode}'.")
        return view

    def plot_combined(self,
                      df_feat: pd.DataFrame,
                      pdb: Optional[str] = None,
                      uniprot: Optional[str] = None,
                      col_imp: str = ut.COL_FEAT_IMPACT,
                      col_val: str = "mean_dif",
                      shap_plot: bool = True,
                      tmd_len: int = 20,
                      start: int = 1,
                      chain: Optional[str] = None,
                      sequence: Optional[str] = None,
                      mode: Literal["impact", "plddt"] = "impact",
                      focus: Literal["whole", "fade", "zoom"] = "zoom",
                      focus_region: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                      size_by_impact: bool = True,
                      normalize_by_span: bool = False,
                      tmd_seq: Optional[str] = None,
                      jmd_n_seq: Optional[str] = None,
                      jmd_c_seq: Optional[str] = None,
                      feature_map_dpi: int = 200,
                      feature_map_kws: Optional[dict] = None,
                      ) -> CombinedView:
        """
        Show the 3D structure and the CPP feature map side by side.

        Reproduces the deployed app's signature layout: the **left** panel is the interactive
        py3Dmol cartoon painted with per-residue CPP feature impact (zoomed to the feature
        window), the **right** panel is the :meth:`CPPPlot.feature_map` of the same ``df_feat``
        (a high-resolution image). Both read the same per-residue impact, so the structure
        colours and the feature map tell one consistent story. Returns a ``CombinedView`` that
        renders inline and exports the pair with ``write_html(path)``.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a ``feature`` column, the signed per-feature impact column
            ``col_imp``, and the scale-information columns the feature map needs.
        pdb : str, optional
            Path to a ``.pdb`` / ``.cif`` structure file. Exactly one of ``pdb`` or ``uniprot``
            must be given.
        uniprot : str, optional
            UniProt accession; the AlphaFold model is fetched into a temporary folder via
            :meth:`StructurePreprocessor.fetch_alphafold`. Exactly one of ``pdb`` or ``uniprot``
            must be given.
        col_imp : str, default='feat_impact'
            Column of ``df_feat`` holding the signed per-feature impact (painted on the structure
            and shown in the feature map).
        col_val : str, default='mean_dif'
            Column shown in the feature-map heatmap cells (passed to :meth:`CPPPlot.feature_map`).
        shap_plot : bool, default=True
            Passed to :meth:`CPPPlot.feature_map` (sample-level CPP-SHAP layout if ``True``).
        tmd_len : int, default=20
            Length of the TMD (>=1). Must match the value used when the features were generated.
        start : int, default=1
            Absolute residue number of the first JMD-N residue; shifts window-relative positions
            onto the structure's numbering.
        chain : str, optional
            Chain id to render. Default selects the best-matching chain when ``sequence`` is
            given, otherwise the first amino-acid chain.
        sequence : str, optional
            Full protein sequence; enables best-matching-chain selection and a ``start`` sanity
            check against the structure.
        mode : {'impact', 'plddt'}, default='impact'
            Structure colouring: the feature-impact ramp or the AlphaFold pLDDT palette.
        focus : {'whole', 'fade', 'zoom'}, default='zoom'
            Structure framing: ``'zoom'`` points the camera at the feature window, ``'fade'``
            ghosts residues outside it, ``'whole'`` styles every residue equally.
        focus_region : tuple or list of tuples, optional
            ``(start, stop)`` residue range (or list of ranges) defining the focus window.
            Default derives the window from the union of ``df_feat`` positions.
        size_by_impact : bool, default=True
            If ``True``, draw each impact residue's stick scaled by ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            Per-residue aggregation for the structure colouring; see :meth:`map_structure`.
        tmd_seq, jmd_n_seq, jmd_c_seq : str, optional
            TMD / JMD-N / JMD-C sequences shown along the feature-map x-axis.
        feature_map_dpi : int, default=200
            Resolution of the feature-map image shown beside the structure (>=50).
        feature_map_kws : dict, optional
            Extra keyword arguments forwarded to :meth:`CPPPlot.feature_map`. Keys already
            controlled by this method (e.g. ``df_feat``, ``col_val``, ``col_imp``, ``tmd_len``,
            ``start``, ``shap_plot``, the part sequences) are rejected.

        Returns
        -------
        view : CombinedView
            A wrapper showing the py3Dmol cartoon next to the feature-map image, exposing
            ``show()``, ``write_html(path)``, and ``_repr_html_``, plus the mapped
            ``dict_impact`` / ``max_abs``.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. an unknown ``mode`` / ``focus``, neither or both of
            ``pdb`` / ``uniprot``, a ``df_feat`` missing ``col_imp``, or a colliding
            ``feature_map_kws`` key).
        RuntimeError
            If py3Dmol is not installed, or an AlphaFold model for ``uniprot`` cannot be fetched.

        See Also
        --------
        * :meth:`plot_linked`: the same two panels with live hover linking (column to residue).
        * :meth:`explore`: build the per-site prediction and pick the output in one call.

        Examples
        --------
        .. include:: examples/csp_plot_combined.rst
        """
        # Validate
        ut.check_df(name="df_feat", df=df_feat, cols_required=[ut.COL_FEATURE, col_imp])
        ut.check_str(name="col_imp", val=col_imp)
        ut.check_str(name="col_val", val=col_val)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
        ut.check_str_options(name="focus", val=focus, list_str_options=LIST_FOCUS)
        ut.check_bool(name="size_by_impact", val=size_by_impact)
        ut.check_bool(name="normalize_by_span", val=normalize_by_span)
        ut.check_number_range(name="feature_map_dpi", val=feature_map_dpi, min_val=50,
                              just_int=True)
        if chain is not None:
            ut.check_str(name="chain", val=chain)
        if sequence is not None:
            ut.check_str(name="sequence", val=sequence)
        focus_region = check_focus_region(focus_region=focus_region)
        if (pdb is None) == (uniprot is None):
            raise ValueError("Exactly one of 'pdb' or 'uniprot' should be given "
                             f"(got pdb={pdb}, uniprot={uniprot})")
        feature_map_kws = _check_feature_map_kws(feature_map_kws)
        _require_py3dmol()

        # Compute per-residue impact (shared CPP backend)
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=col_imp, start=start, tmd_len=tmd_len,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
            normalize_by_span=normalize_by_span)
        window_resis = _resolve_window_resis(focus, focus_region, positions_union)

        # Feature-map image (built first so an error leaves no structure view dangling)
        png_b64 = self._feature_map_png_b64(df_feat=df_feat, col_val=col_val, col_imp=col_imp,
                                            shap_plot=shap_plot, tmd_len=tmd_len, start=start,
                                            tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
                                            jmd_c_seq=jmd_c_seq, dpi=feature_map_dpi,
                                            feature_map_kws=feature_map_kws)
        # py3Dmol cartoon (left panel), rendered inside the structure temp context
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
            self._check_start_alignment(records, positions_union, identity, sequence)
            view = render_py3dmol(pdb_path=pdb_path, records=records, dict_impact=dict_impact,
                                  max_abs=max_abs, mode=mode, focus=focus,
                                  window_resis=window_resis, size_by_impact=size_by_impact,
                                  chain_id=chain_id)

        if self._verbose:
            ut.print_out(f"CPPStructurePlot: combined view of {len(df_feat)} features on "
                         f"{len(records)} residues, mode='{mode}'.")
        return CombinedView(view=view.view, feature_map_png_b64=png_b64,
                            dict_impact=dict_impact, max_abs=max_abs, mode=mode)

    # Internal orchestration helpers
    def _feature_map_png_b64(self, df_feat, col_val, col_imp, shap_plot, tmd_len, start,
                             tmd_seq, jmd_n_seq, jmd_c_seq, dpi, feature_map_kws):
        """Render CPPPlot.feature_map to its own figure and return it as a base64 PNG string."""
        import io
        import base64
        import matplotlib.pyplot as plt
        from aaanalysis import CPPPlot
        cpp_plot = CPPPlot(df_scales=self._df_scales, df_cat=self._df_cat,
                           jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                           accept_gaps=True, verbose=False)
        fig_fm, _ax_fm = cpp_plot.feature_map(
            df_feat=df_feat, shap_plot=shap_plot, col_val=col_val, col_imp=col_imp,
            tmd_len=tmd_len, start=start, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
            jmd_c_seq=jmd_c_seq, **feature_map_kws)
        try:
            buffer = io.BytesIO()
            fig_fm.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig_fm)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def plot_linked(self,
                    df_feat: pd.DataFrame,
                    pdb: Optional[str] = None,
                    uniprot: Optional[str] = None,
                    col_imp: str = ut.COL_FEAT_IMPACT,
                    col_val: str = "mean_dif",
                    shap_plot: bool = True,
                    tmd_len: int = 20,
                    start: int = 1,
                    chain: Optional[str] = None,
                    sequence: Optional[str] = None,
                    mode: Literal["impact", "plddt"] = "impact",
                    focus: Literal["whole", "fade", "zoom"] = "zoom",
                    focus_region: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                    size_by_impact: bool = True,
                    normalize_by_span: bool = False,
                    tmd_seq: Optional[str] = None,
                    jmd_n_seq: Optional[str] = None,
                    jmd_c_seq: Optional[str] = None,
                    feature_map_dpi: int = 150,
                    feature_map_kws: Optional[dict] = None,
                    width: int = 520,
                    height: int = 460,
                    ) -> LinkedView:
        """
        Build a self-contained HTML view with the feature map and structure **linked**.

        Reproduces the deployed app's signature interaction: the :meth:`CPPPlot.feature_map`
        is shown beside an interactive 3Dmol cartoon, and **hovering a feature-map column
        highlights the corresponding residue in the structure** (the column's position maps to
        the absolute residue via ``start``). Returns a ``LinkedView`` that renders inline where
        embedded scripts run (classic Notebook, nbviewer, Read the Docs) and exports a
        standalone, shareable ``.html`` via ``write_html(path)`` — ideal for exploring a site and
        as a publication-figure source. In JupyterLab (which sandboxes output scripts), use
        ``write_html`` and open the page in a browser.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with ``feature``, the signed impact column ``col_imp``, and the
            scale-information columns the feature map needs.
        pdb : str, optional
            Path to a ``.pdb`` / ``.cif`` file. Exactly one of ``pdb`` or ``uniprot`` is required.
        uniprot : str, optional
            UniProt accession; the AlphaFold model is fetched automatically. Exactly one of
            ``pdb`` or ``uniprot`` is required.
        col_imp : str, default='feat_impact'
            Column holding the signed per-feature impact (painted on the structure + feature map).
        col_val : str, default='mean_dif'
            Column shown in the feature-map heatmap cells.
        shap_plot : bool, default=True
            Passed to :meth:`CPPPlot.feature_map`.
        tmd_len : int, default=20
            Length of the TMD (>=1). Must match the geometry the features were generated with.
        start : int, default=1
            Absolute residue number of the first JMD-N residue; maps feature-map columns to residues.
        chain : str, optional
            Chain id to render (default best-matching / first amino-acid chain).
        sequence : str, optional
            Full protein sequence; enables best-matching-chain selection + a ``start`` check.
        mode : {'impact', 'plddt'}, default='impact'
            Structure colouring: the feature-impact ramp or the AlphaFold pLDDT palette.
        focus : {'whole', 'fade', 'zoom'}, default='zoom'
            Structure framing: ``'zoom'`` frames the feature window, ``'fade'`` ghosts the rest.
        focus_region : tuple or list of tuples, optional
            ``(start, stop)`` focus window; default from the union of ``df_feat`` positions.
        size_by_impact : bool, default=True
            Scale each impact residue's stick by ``|impact|`` (impact mode).
        normalize_by_span : bool, default=False
            Per-residue aggregation for the structure colouring; see :meth:`map_structure`.
        tmd_seq, jmd_n_seq, jmd_c_seq : str, optional
            Part sequences shown along the feature-map x-axis.
        feature_map_dpi : int, default=150
            Resolution of the embedded feature-map image (>=50).
        feature_map_kws : dict, optional
            Extra keyword arguments forwarded to :meth:`CPPPlot.feature_map` (keys this method
            already controls are rejected).
        width, height : int, default 520, 460
            Pixel size of the 3D viewer panel.

        Returns
        -------
        view : LinkedView
            A wrapper exposing ``show()``, ``write_html(path)``, and ``_repr_html_`` over the
            linked feature-map + structure HTML, plus ``dict_impact`` / ``max_abs``.

        Raises
        ------
        ValueError
            On invalid arguments (unknown ``mode`` / ``focus``, neither/both of ``pdb`` /
            ``uniprot``, ``df_feat`` missing ``col_imp``, or a colliding ``feature_map_kws`` key).
        RuntimeError
            If py3Dmol is not installed, or an AlphaFold model for ``uniprot`` cannot be fetched.

        See Also
        --------
        * :meth:`plot_combined`: the same two panels as a static side-by-side (no live linking).
        * :meth:`explore`: pass ``sites=[...]`` to bake a multi-site *live* linked HTML.

        Examples
        --------
        .. include:: examples/csp_plot_linked.rst
        """
        # Validate (same contract as plot_combined)
        ut.check_df(name="df_feat", df=df_feat, cols_required=[ut.COL_FEATURE, col_imp])
        ut.check_str(name="col_imp", val=col_imp)
        ut.check_str(name="col_val", val=col_val)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
        ut.check_str_options(name="focus", val=focus, list_str_options=LIST_FOCUS)
        ut.check_bool(name="size_by_impact", val=size_by_impact)
        ut.check_bool(name="normalize_by_span", val=normalize_by_span)
        ut.check_number_range(name="feature_map_dpi", val=feature_map_dpi, min_val=50, just_int=True)
        ut.check_number_range(name="width", val=width, min_val=100, just_int=True)
        ut.check_number_range(name="height", val=height, min_val=100, just_int=True)
        if chain is not None:
            ut.check_str(name="chain", val=chain)
        if sequence is not None:
            ut.check_str(name="sequence", val=sequence)
        focus_region = check_focus_region(focus_region=focus_region)
        if (pdb is None) == (uniprot is None):
            raise ValueError("Exactly one of 'pdb' or 'uniprot' should be given "
                             f"(got pdb={pdb}, uniprot={uniprot})")
        feature_map_kws = _check_feature_map_kws(feature_map_kws)
        _require_py3dmol()

        # Per-residue impact + the feature window
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=col_imp, start=start, tmd_len=tmd_len,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, normalize_by_span=normalize_by_span)
        window_resis = _resolve_window_resis(focus, focus_region, positions_union)
        n_pos = jmd_n_len + tmd_len + jmd_c_len

        # Feature map image + per-column geometry (columns -> residues)
        png_b64, columns, band_top, band_height = self._feature_map_png_and_columns(
            df_feat=df_feat, col_val=col_val, col_imp=col_imp, shap_plot=shap_plot,
            tmd_len=tmd_len, start=start, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
            jmd_c_seq=jmd_c_seq, dpi=feature_map_dpi, feature_map_kws=feature_map_kws, n_pos=n_pos)

        # Structure: parse, read text (embedded in the HTML), compute per-residue JS styles
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
            self._check_start_alignment(records, positions_union, identity, sequence)
            pdb_text, fmt = _read_structure_text(pdb_path)

        present = {r["resi"] for r in records}
        in_focus = present if window_resis is None else (set(window_resis) & present)
        faded = focus != "whole" and bool(in_focus)
        residue_styles = self._linked_residue_styles(records, dict_impact, max_abs, mode,
                                                     size_by_impact, faded, in_focus)
        zoom_resis = sorted(in_focus) if (focus == "zoom" and in_focus) else None

        uid = "%04d" % next(_LINKED_UID)
        html_body = build_linked_html(
            uid=uid, pdb_text=pdb_text, fmt=fmt, chain_id=chain_id,
            residue_styles=residue_styles, base_color=ut.COLOR_STRUCT_MISSING,
            faded=faded, fade_opacity=0.45, zoom_resis=zoom_resis, fmap_png_b64=png_b64,
            columns=columns, band_top=band_top, band_height=band_height,
            width=width, height=height)
        if self._verbose:
            ut.print_out(f"CPPStructurePlot: linked view of {len(df_feat)} features on "
                         f"{len(records)} residues, mode='{mode}'.")
        return LinkedView(html_body=html_body, dict_impact=dict_impact, max_abs=max_abs, mode=mode)

    def _feature_map_png_and_columns(self, df_feat, col_val, col_imp, shap_plot, tmd_len,
                                     start, tmd_seq, jmd_n_seq, jmd_c_seq, dpi,
                                     feature_map_kws, n_pos):
        """Render CPPPlot.feature_map (no tight bbox) -> base64 PNG + per-column geometry.

        Without ``bbox_inches='tight'`` the figure fractions map directly onto the saved PNG,
        so the heatmap column x-positions (computed from the heatmap axes) line up with the
        image — the same mapping the deployed app uses to box a residue.
        """
        import io
        import base64
        import matplotlib.pyplot as plt
        from aaanalysis import CPPPlot
        cpp_plot = CPPPlot(df_scales=self._df_scales, df_cat=self._df_cat,
                           jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                           accept_gaps=True, verbose=False)
        fig_fm, ax_fm = cpp_plot.feature_map(
            df_feat=df_feat, shap_plot=shap_plot, col_val=col_val, col_imp=col_imp,
            tmd_len=tmd_len, start=start, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
            jmd_c_seq=jmd_c_seq, **feature_map_kws)
        try:
            fig_fm.canvas.draw()
            # The heatmap axes span all n_pos columns; pick the tallest such axes (not the
            # short top impact-bar that shares the x-range).
            cands = [a for a in fig_fm.get_axes()
                     if abs((a.get_xlim()[1] - a.get_xlim()[0]) - n_pos) < 1e-6]
            heat_ax = max(cands, key=lambda a: a.get_position().height) if cands else ax_fm
            columns, band_top, band_height = _columns(heat_ax, n_pos, start)
            buffer = io.BytesIO()
            fig_fm.savefig(buffer, format="png", dpi=dpi)  # no tight: fractions map to pixels
        finally:
            plt.close(fig_fm)
        png_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return png_b64, columns, band_top, band_height

    @staticmethod
    def _linked_residue_styles(records, dict_impact, max_abs, mode, size_by_impact,
                               faded, in_focus):
        """Per-residue ``[resi, color_hex, stick_radius]`` for the linked HTML's base styling."""
        styles = []
        if mode == "plddt":
            for res in records:
                if faded and res["resi"] not in in_focus:
                    continue
                styles.append([res["resi"], plddt_to_hex(res["plddt"]), 0.0])
            return styles
        for res in records:
            resi = res["resi"]
            impact = dict_impact.get(resi, 0.0)
            if not np.isfinite(impact) or impact == 0:
                continue
            color = impact_to_hex(impact, max_abs)
            radius = _stick_radius(impact, max_abs, size_by_impact)
            styles.append([resi, color, round(float(radius), 3)])
        return styles

    def _linked_multi_view(self, *, predictor, sequence, sites, pdb, uniprot, col_imp, col_val,
                           shap_plot, tmd_len, jmd_n_len, jmd_c_len, mode, focus, focus_region,
                           size_by_impact, normalize_by_span, feature_map_dpi, width, height,
                           chain):
        """Build the app-like multi-site live HTML: predict each P1 once and bake them all in.

        The structure is parsed once and reused; for each P1 the predictor is run, the per-residue
        impact / feature-map image / column geometry / styles are computed (the same way as
        :meth:`plot_linked`), and the set is handed to :func:`build_linked_html_multi`, whose JS
        slider switches sites client-side (no kernel). Returns a ``LinkedView``.
        """
        n_pos = jmd_n_len + tmd_len + jmd_c_len
        # Resolve the structure once (the PDB is the same across sites).
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
            pdb_text, fmt = _read_structure_text(pdb_path)
        present = {r["resi"] for r in records}
        # Chain-identity is site-independent, so warn once here (the per-site positions checks
        # would be noisy across many sites).
        if sequence is not None and identity < 0.5:
            warnings.warn(
                f"The selected chain matches 'sequence' with low identity ({identity:.2f}); "
                f"the painted residues may be misaligned.", UserWarning, stacklevel=2)

        baked = []
        last_dict_impact, last_max_abs = {}, 0.0
        for p1 in sites:
            start = int(p1) - jmd_n_len
            df_feat_p1 = predictor(sequence, int(p1))
            dict_impact, max_abs, positions_union = compute_residue_impact(
                df_feat=df_feat_p1, col_imp=col_imp, start=start, tmd_len=tmd_len,
                jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, normalize_by_span=normalize_by_span)
            window_resis = _resolve_window_resis(focus, focus_region, positions_union)
            in_focus = present if window_resis is None else (set(window_resis) & present)
            faded = focus != "whole" and bool(in_focus)
            png_b64, columns, band_top, band_height = self._feature_map_png_and_columns(
                df_feat=df_feat_p1, col_val=col_val, col_imp=col_imp, shap_plot=shap_plot,
                tmd_len=tmd_len, start=start, tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                dpi=feature_map_dpi, feature_map_kws={}, n_pos=n_pos)
            styles = self._linked_residue_styles(records, dict_impact, max_abs, mode,
                                                 size_by_impact, faded, in_focus)
            baked.append(dict(p1=int(p1), styles=styles, fmap=png_b64, columns=columns,
                              band_top=band_top, band_height=band_height,
                              zoom=(sorted(in_focus) if (focus == "zoom" and in_focus) else None),
                              faded=faded))
            last_dict_impact, last_max_abs = dict_impact, max_abs
            if self._verbose:
                ut.print_out(f"CPPStructurePlot.explore: baked site P1={int(p1)} "
                             f"({len(baked)}/{len(sites)})")
        uid = "%04d" % next(_LINKED_UID)
        html_body = build_linked_html_multi(
            uid=uid, pdb_text=pdb_text, fmt=fmt, chain_id=chain_id, sites=baked,
            base_color=ut.COLOR_STRUCT_MISSING, fade_opacity=0.45, width=width, height=height)
        if self._verbose:
            ut.print_out(f"CPPStructurePlot: live linked view over {len(baked)} sites on "
                         f"{len(records)} residues, mode='{mode}'.")
        return LinkedView(html_body=html_body, dict_impact=last_dict_impact,
                          max_abs=last_max_abs, mode=mode)

    def interactive(self,
                    predictor: Callable,
                    sequence: str,
                    pdb: Optional[str] = None,
                    uniprot: Optional[str] = None,
                    col_imp: str = ut.COL_FEAT_IMPACT,
                    col_val: str = "mean_dif",
                    shap_plot: bool = True,
                    tmd_len: int = 20,
                    mode: Literal["impact", "plddt"] = "impact",
                    focus: Literal["whole", "fade", "zoom"] = "fade",
                    focus_region: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                    size_by_impact: bool = True,
                    normalize_by_span: bool = False,
                    feature_map: bool = True,
                    site_to_start: Optional[Callable] = None,
                    chain: Optional[str] = None,
                    init_site: Optional[int] = None,
                    debounce_ms: int = 250,
                    ) -> object:  # an ipywidgets.Widget (kept loose to avoid importing ipywidgets)
        """
        Build a live, selection-linked explorer that re-predicts and repaints on each site.

        Returns an `ipywidgets <https://ipywidgets.readthedocs.io>`_ panel (**[pro]**, needs
        ``ipywidgets``) reproducing the deployed app's per-site explore loop in a notebook:
        a site slider drives a user ``predictor`` that returns a ``df_feat`` for that site, and
        both the 3D structure (the :meth:`map_structure` render path) and the
        :meth:`CPPPlot.feature_map` repaint in place from that one selection — reading the same
        per-residue impact. Rapid changes are debounced so the predictor is not re-run on every
        intermediate slider value.

        The exact prediction itself runs on the live Python kernel via ``predictor``; this class
        does not hard-code :class:`CPP` / :class:`TreeModel` / :class:`ShapModel`.

        A **highlight (position) slider** links the feature map to the structure: pick a residue in
        the current window and it lights up in the 3D cartoon (a bold marker) while a vertical line
        marks its feature-map column — without re-running the predictor. When the ``ipympl``
        (``%matplotlib widget``) backend is active the feature map is also **clickable** (clicking a
        column drives the same highlight); ``ipympl`` is optional — the slider is the always-present
        link, so no extra dependency is required.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        predictor : callable
            User callable ``(sequence, p1) -> df_feat`` returning a feature DataFrame (with the
            ``col_imp`` and feature-map columns) for the site ``p1``. Example wiring
            :class:`CPP` + :class:`ShapModel` into such a callable is shown in the notebook.
        sequence : str
            Full protein sequence; the site slider ranges over ``1..len(sequence)``.
        pdb : str, optional
            Path to a ``.pdb`` / ``.cif`` structure file. Exactly one of ``pdb`` or ``uniprot``
            must be given. The structure is parsed once and reused across selections.
        uniprot : str, optional
            UniProt accession; the AlphaFold model is fetched once into a temporary folder.
            Exactly one of ``pdb`` or ``uniprot`` must be given.
        col_imp : str, default='feat_impact'
            Column of the predictor's ``df_feat`` holding the signed per-feature impact.
        col_val : str, default='mean_dif'
            Column shown in the feature-map heatmap cells.
        shap_plot : bool, default=True
            Passed to :meth:`CPPPlot.feature_map`.
        tmd_len : int, default=20
            Length of the TMD (>=1). Must match the geometry the predictor's features use.
        mode : {'impact', 'plddt'}, default='impact'
            Initial structure colouring (a live dropdown also toggles it).
        focus : {'whole', 'fade', 'zoom'}, default='fade'
            Initial structure framing (a live dropdown also toggles it).
        focus_region : tuple or list of tuples, optional
            Fixed ``(start, stop)`` focus window; default derives it from each selection's
            ``df_feat`` positions.
        size_by_impact : bool, default=True
            Scale each impact residue's stick / marker by ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            Per-residue aggregation for the structure colouring; see :meth:`map_structure`.
        feature_map : bool, default=True
            If ``True``, show the linked :meth:`CPPPlot.feature_map` panel; if ``False``, the
            3D structure panel only.
        site_to_start : callable, optional
            Maps the selected site to the structure anchor ``start`` (first JMD-N residue):
            ``p1 -> start``. Default ``lambda p1: p1 - jmd_n_len`` (the site is the first TMD
            residue). Supply your own to match a different window geometry.
        chain : str, optional
            Chain id to render; default selects the best-matching / first amino-acid chain.
        init_site : int, optional
            Initial selected site (default the middle of ``sequence``).
        debounce_ms : int, default=250
            Coalesce slider/dropdown changes within this many milliseconds into one predictor
            call and repaint (>=0; 0 renders synchronously).

        Returns
        -------
        panel : ipywidgets.Widget
            A widget container (controls + linked structure / feature-map outputs) that displays
            inline in Jupyter.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. ``predictor`` not callable, neither or both of
            ``pdb`` / ``uniprot``, an unknown ``mode`` / ``focus``, an out-of-range ``init_site``).
        RuntimeError
            If ``ipywidgets`` is not installed, or an AlphaFold model cannot be fetched.

        Examples
        --------
        .. include:: examples/csp_interactive.rst
        """
        # Validate
        if not callable(predictor):
            raise ValueError(f"'predictor' ({predictor}) should be a callable (sequence, p1) "
                             f"-> df_feat")
        ut.check_str(name="sequence", val=sequence, accept_none=False)
        ut.check_str(name="col_imp", val=col_imp)
        ut.check_str(name="col_val", val=col_val)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_bool(name="feature_map", val=feature_map)
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
        ut.check_str_options(name="focus", val=focus, list_str_options=LIST_FOCUS)
        ut.check_bool(name="size_by_impact", val=size_by_impact)
        ut.check_bool(name="normalize_by_span", val=normalize_by_span)
        ut.check_number_range(name="debounce_ms", val=debounce_ms, min_val=0, just_int=True)
        if site_to_start is not None and not callable(site_to_start):
            raise ValueError(f"'site_to_start' ({site_to_start}) should be a callable p1 -> start")
        if chain is not None:
            ut.check_str(name="chain", val=chain)
        focus_region = check_focus_region(focus_region=focus_region)
        if (pdb is None) == (uniprot is None):
            raise ValueError("Exactly one of 'pdb' or 'uniprot' should be given "
                             f"(got pdb={pdb}, uniprot={uniprot})")
        n_res = len(sequence)
        if init_site is None:
            init_site = max(1, n_res // 2)
        ut.check_number_range(name="init_site", val=init_site, min_val=1, max_val=n_res,
                              just_int=True)

        _require_py3dmol()
        ipw = _require_ipywidgets()
        if site_to_start is None:
            site_to_start = lambda p1: p1 - jmd_n_len   # noqa: E731 (site = first TMD residue)

        # Resolve the structure once (reused across selections)
        records, chain_id, pdb_path, tmp_holder = self._resolve_structure_persistent(
            pdb, uniprot, chain, sequence)

        # Feature-map renderer (frontend owns CPPPlot); None disables the map panel. Returns
        # (fig, heat_ax): heat_ax is the column-spanning heatmap axes (for the highlight line +
        # the ipympl click->residue mapping), found like plot_linked's _feature_map_png_and_columns.
        feature_map_renderer = None
        if feature_map:
            n_pos = self._jmd_n_len + tmd_len + self._jmd_c_len

            def _render_feature_map(df_feat, start, highlight_resi=None):
                from aaanalysis import CPPPlot
                cpp_plot = CPPPlot(df_scales=self._df_scales, df_cat=self._df_cat,
                                   jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                   accept_gaps=True, verbose=False)
                fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=shap_plot,
                                               col_val=col_val, col_imp=col_imp,
                                               tmd_len=tmd_len, start=start)
                fig.canvas.draw()
                cands = [a for a in fig.get_axes()
                         if abs((a.get_xlim()[1] - a.get_xlim()[0]) - n_pos) < 1e-6]
                heat_ax = max(cands, key=lambda a: a.get_position().height) if cands else ax
                if highlight_resi is not None and heat_ax is not None:
                    # Column i (0-based) is residue start+i; draw the line at the column centre.
                    heat_ax.axvline(x=(int(highlight_resi) - int(start)) + 0.5,
                                    color=ut.COLOR_LINK_HIGHLIGHT, linewidth=2.0, alpha=0.95,
                                    zorder=10)
                return fig, heat_ax
            feature_map_renderer = _render_feature_map

        from ._backend.cpp_struct.interactive_widgets import InteractivePanel
        panel = InteractivePanel(
            ipw, predictor=predictor, sequence=sequence, records=records, pdb_path=pdb_path,
            chain_id=chain_id, col_imp=col_imp, tmd_len=tmd_len, jmd_n_len=jmd_n_len,
            jmd_c_len=jmd_c_len, mode=mode, focus=focus, focus_region=focus_region,
            size_by_impact=size_by_impact, normalize_by_span=normalize_by_span,
            site_to_start=site_to_start, feature_map_renderer=feature_map_renderer,
            init_site=init_site, debounce_ms=debounce_ms, tmp_holder=tmp_holder,
            verbose=self._verbose)
        if self._verbose:
            ut.print_out(f"CPPStructurePlot: interactive explorer over {n_res} residues "
                         f"(feature_map={feature_map}, debounce_ms={debounce_ms}).")
        return panel.container

    def explore(self,
                df_feat: pd.DataFrame,
                sequence: str,
                pdb: Optional[str] = None,
                uniprot: Optional[str] = None,
                df_seq: Optional[pd.DataFrame] = None,
                labels: Optional[ut.ArrayLike1D] = None,
                model: Union[str, BaseEstimator, List] = ut.MODEL_RF,
                predictor: Optional[Callable] = None,
                output: Literal["widget", "html", "static"] = "widget",
                path: Optional[str] = None,
                col_imp: str = ut.COL_FEAT_IMPACT,
                col_val: str = "mean_dif",
                shap_plot: bool = True,
                tmd_len: int = 20,
                init_site: Optional[int] = None,
                sites: Optional[List[int]] = None,
                label_target_class: int = 1,
                mode: Literal["impact", "plddt"] = "impact",
                focus: Literal["whole", "fade", "zoom"] = "fade",
                focus_region: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                size_by_impact: bool = True,
                normalize_by_span: bool = False,
                chain: Optional[str] = None,
                random_state: Optional[int] = None,
                n_jobs: Optional[int] = None,
                debounce_ms: int = 250,
                ) -> object:
        """
        Predict per site and paint the structure, with a selectable output type.

        The integrated explorer (**[pro]**): given a feature set ``df_feat`` plus a labeled training
        population (``df_seq`` + ``labels``) and a ``model``, it builds a per-site predictor that, for
        a P1 site, computes the query window's feature values for the **fixed** feature set (never a
        :meth:`CPP.run` rediscovery), predicts its probability, and attaches the per-site SHAP impact
        (a default :class:`ShapModel` refit, fuzzy interpolate). It then dispatches to one of the
        render paths by ``output``:

        - ``'widget'`` -> :meth:`interactive` (a live ipywidgets explorer; the P1 slider re-predicts
          and repaints per site; needs a kernel + ``ipywidgets``).
        - ``'html'`` -> a self-contained linked HTML, written to ``path`` if given. Baked for the
          single ``init_site`` by default; pass ``sites=[...]`` to bake a **multi-site live** page
          whose JS slider switches the pre-computed per-site prediction client-side (no kernel).
        - ``'static'`` -> :meth:`plot_combined` (the structure beside the feature map, baked for
          ``init_site``). With ``path`` given, the feature-map panel is saved (``CombinedView.savefig``,
          format from the extension, e.g. PNG/PDF); the 3D structure stays interactive.

        The site geometry follows the package convention: ``p1`` is the first TMD residue, so the
        TMD spans ``[p1, p1 + tmd_len - 1]`` and ``start = p1 - jmd_n_len`` (the construction
        ``jmd_n_len`` / ``jmd_c_len``). Pass ``predictor`` to override the built-in with a custom
        ``(sequence, p1) -> df_feat`` callable, in which case ``df_seq`` / ``labels`` / ``model`` are
        ignored.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a ``feature`` column (the fixed feature set) and the
            scale-information columns the feature map needs. The per-site impact is computed and
            written to ``col_imp``; an existing ``col_imp`` is overwritten per site.
        sequence : str
            Full protein sequence of the query protein the P1 site ranges over.
        pdb : str, optional
            Path to a ``.pdb`` / ``.cif`` structure file. Exactly one of ``pdb`` or ``uniprot``
            must be given.
        uniprot : str, optional
            UniProt accession; the AlphaFold model is fetched. Exactly one of ``pdb`` or ``uniprot``
            must be given.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
            DataFrame containing an ``entry`` column with a unique protein identifier per row, the
            labeled training population row-aligned to ``labels``. Required for the built-in
            predictor (``predictor=None``); the training feature matrix is built from it via
            :meth:`SequenceFeature.get_df_parts`.
        labels : array-like, shape (n_samples,), optional
            Class labels for ``df_seq`` (typically 1=positive, 0=negative). Required for the built-in
            predictor.
        model : str, estimator, or list, default='rf'
            Prediction model for the per-site probability: a name (``'rf'``, ``'svm'``,
            ``'log_reg'``, ``'extra_trees'``), a scikit-learn estimator, or a list of those
            (probabilities averaged). The :class:`ShapModel` that computes the impact always uses
            its own defaults (``TreeExplainer`` + RandomForest / ExtraTrees), so the prediction
            model and the explanation model may differ.
        predictor : callable, optional
            Escape hatch: a custom ``(sequence, p1) -> df_feat`` callable. When given, the built-in
            predictor is not built and ``df_seq`` / ``labels`` / ``model`` are ignored.
        output : {'widget', 'html', 'static'}, default='widget'
            Which output to produce (see the method summary).
        path : str, optional
            Output file path. For ``output='html'`` writes the self-contained page; for
            ``output='static'`` saves the feature-map panel as an image (PNG/PDF/..., format from
            the extension). Ignored for ``output='widget'``.
        col_imp : str, default='feat_impact'
            Column the per-site signed impact is written to (and painted from). With
            ``shap_plot=True`` it must be ``'feat_impact'`` or follow ``'feat_impact_<name>'`` (the
            SHAP feature map's requirement).
        col_val : str, default='mean_dif'
            Column shown in the feature-map heatmap cells.
        shap_plot : bool, default=True
            Passed to :meth:`CPPPlot.feature_map`.
        tmd_len : int, default=20
            TMD length of the predicted window (>=1).
        init_site : int, optional
            The P1 site to bake for ``output`` in {``'html'``, ``'static'``} and the initial slider
            site for ``'widget'`` (default near the middle of ``sequence``).
        sites : list of int, optional
            Only for ``output='html'``: a list of P1 positions to bake into one **multi-site live**
            HTML. The page gets a client-side JS site slider that switches the pre-computed
            prediction per site (feature map + structure restyle) with no kernel, keeping the
            column-residue linking. ``None`` (default) bakes only the single ``init_site``. Each
            site embeds a feature-map image and runs one predictor refit, so a warning is emitted
            past 40 sites and a hard cap applies at 200.
        label_target_class : int, default=1
            Class whose probability is predicted and explained.
        mode : {'impact', 'plddt'}, default='impact'
            Initial structure colouring.
        focus : {'whole', 'fade', 'zoom'}, default='fade'
            Structure framing.
        focus_region : tuple or list of tuples, optional
            Fixed ``(start, stop)`` focus window; default from the window's positions.
        size_by_impact : bool, default=True
            Scale each impact residue's stick by ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            Per-residue aggregation for the structure colouring; see :meth:`map_structure`.
        chain : str, optional
            Chain id to render; default selects the best-matching / first amino-acid chain.
        random_state : int, optional
            Seed for the prediction estimator and the SHAP estimation (reproducibility).
        n_jobs : int, optional
            Cores for building the training feature matrix.
        debounce_ms : int, default=250
            For ``output='widget'``, coalesce slider changes within this many milliseconds.

        Returns
        -------
        view : object
            For ``output='widget'`` an ``ipywidgets`` panel; for ``'html'`` a ``LinkedView``; for
            ``'static'`` a ``CombinedView``. All render inline in Jupyter.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. an unknown ``output``, neither or both of ``pdb`` /
            ``uniprot``, ``predictor=None`` without ``df_seq`` / ``labels``, or an unknown ``model``
            name).
        RuntimeError
            If a required optional package (py3Dmol, or ``ipywidgets`` for the widget) is missing, or
            an AlphaFold model cannot be fetched.

        Notes
        -----
        * **Per-site cost.** The built-in predictor refits a :class:`ShapModel` per site
          (the predicted probability is fit once; only the SHAP impact refits). For
          ``output='html'`` with ``sites=[...]`` every site additionally embeds a feature-map
          image, so file size and build time scale linearly with ``len(sites)`` — a warning is
          emitted past 40 sites and a hard cap applies at 200.
        * **Static capture.** ``output='static'`` with ``path`` saves the feature-map panel only
          (``CombinedView.savefig``); the 3D structure is interactive and has no headless image.

        See Also
        --------
        * :meth:`plot_combined` / :meth:`plot_linked` / :meth:`interactive`: the lower-level
          render surfaces this dispatches to.

        Examples
        --------
        .. include:: examples/csp_explore.rst
        """
        # Validate (cheap checks first, so an invalid argument fails before the predictor is built)
        ut.check_df(name="df_feat", df=df_feat, cols_required=[ut.COL_FEATURE])
        ut.check_str(name="sequence", val=sequence, accept_none=False)
        ut.check_str_options(name="output", val=output, list_str_options=LIST_OUTPUTS)
        ut.check_str(name="col_imp", val=col_imp)
        ut.check_str(name="col_val", val=col_val)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_bool(name="size_by_impact", val=size_by_impact)
        ut.check_bool(name="normalize_by_span", val=normalize_by_span)
        ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
        ut.check_str_options(name="focus", val=focus, list_str_options=LIST_FOCUS)
        focus_region = check_focus_region(focus_region=focus_region)
        ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
        ut.check_number_range(name="label_target_class", val=label_target_class, min_val=0,
                              just_int=True)
        # The SHAP feature map (shap_plot=True) requires col_imp in the feat_impact family; fail
        # here with a clear message instead of deep inside CPPPlot.feature_map after a full refit.
        if shap_plot and ut.COL_FEAT_IMPACT not in col_imp:
            raise ValueError(f"With 'shap_plot=True', 'col_imp' ('{col_imp}') must be "
                             f"'{ut.COL_FEAT_IMPACT}' or follow '{ut.COL_FEAT_IMPACT}_<name>'")
        # The built-in predictor produces SHAP feature impact, so it needs the SHAP feature map.
        # shap_plot=False (the feature-importance layout) is only meaningful with a custom predictor.
        if predictor is None and not shap_plot:
            raise ValueError("The built-in predictor produces SHAP feature impact, so "
                             "'shap_plot' must be True; pass a custom 'predictor' that yields a "
                             "feature-importance 'col_imp' to use 'shap_plot=False'.")
        if (pdb is None) == (uniprot is None):
            raise ValueError("Exactly one of 'pdb' or 'uniprot' should be given "
                             f"(got pdb={pdb}, uniprot={uniprot})")
        if predictor is not None and not callable(predictor):
            raise ValueError(f"'predictor' ({predictor}) should be a callable (sequence, p1) "
                             f"-> df_feat")
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        n_res = len(sequence)
        # The site geometry needs jmd_n_len residues before p1, so the first JMD-N residue (start =
        # p1 - jmd_n_len) is valid; validate init_site here (naming init_site, not 'start').
        if init_site is not None:
            ut.check_number_range(name="init_site", val=init_site, min_val=jmd_n_len + 1,
                                  max_val=n_res, just_int=True)
        # Validate `sites` (the multi-site live HTML) BEFORE the predictor is built, so a bad
        # argument fails fast (cheap checks first), consistent with the rest of this method.
        if sites is not None:
            if output != "html":
                raise ValueError(f"'sites' is only valid with output='html' (got output={output!r})")
            sites = [int(s) for s in sites]
            if len(sites) == 0:
                raise ValueError("'sites' should be a non-empty list of P1 positions (or None)")
            for s in sites:
                ut.check_number_range(name="sites", val=s, min_val=jmd_n_len + 1, max_val=n_res,
                                      just_int=True)
            if len(sites) > _MAX_SITES:
                raise ValueError(
                    f"'sites' has {len(sites)} positions; the live HTML caps at {_MAX_SITES} "
                    f"(each site bakes a feature-map image and runs one predictor refit). Pass "
                    f"fewer / coarser sites.")
            if len(sites) > _WARN_SITES:
                warnings.warn(
                    f"Baking {len(sites)} sites into the live HTML embeds {len(sites)} feature-map "
                    f"images and runs {len(sites)} predictor refits; expect a large file and a "
                    f"longer build.", UserWarning, stacklevel=2)

        # Resolve the per-site predictor (built-in unless the escape hatch is given)
        if predictor is None:
            if df_seq is None or labels is None:
                raise ValueError("The built-in predictor needs 'df_seq' and 'labels' (or pass a "
                                 "custom 'predictor'); got "
                                 f"df_seq={'set' if df_seq is not None else None}, "
                                 f"labels={'set' if labels is not None else None}")
            ut.check_df_seq(df_seq=df_seq)
            predictor = build_builtin_predictor(
                df_feat=df_feat, df_seq=df_seq, labels=labels, tmd_len=tmd_len,
                jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len, model=model,
                col_imp=col_imp, df_scales=self._df_scales,
                label_target_class=label_target_class, random_state=random_state,
                n_jobs=n_jobs, verbose=self._verbose)

        # Multi-site live HTML: bake a JS site slider over `sites` (re-prediction is precomputed,
        # so the standalone page switches sites client-side with no kernel). Already validated above.
        if sites is not None:
            view = self._linked_multi_view(
                predictor=predictor, sequence=sequence, sites=sites, pdb=pdb, uniprot=uniprot,
                col_imp=col_imp, col_val=col_val, shap_plot=shap_plot, tmd_len=tmd_len,
                jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, mode=mode, focus=focus,
                focus_region=focus_region, size_by_impact=size_by_impact,
                normalize_by_span=normalize_by_span, feature_map_dpi=150, width=520, height=460,
                chain=chain)
            if path is not None:
                view.write_html(path)
            return view

        # Dispatch on the selected output
        if output == "widget":
            return self.interactive(
                predictor=predictor, sequence=sequence, pdb=pdb, uniprot=uniprot, col_imp=col_imp,
                col_val=col_val, shap_plot=shap_plot, tmd_len=tmd_len, mode=mode, focus=focus,
                focus_region=focus_region, size_by_impact=size_by_impact,
                normalize_by_span=normalize_by_span, chain=chain, init_site=init_site,
                debounce_ms=debounce_ms)

        # One-shot bake (no live kernel): predict the chosen site once and render it
        if init_site is None:
            init_site = max(jmd_n_len + 1, n_res // 2)
            ut.check_number_range(name="init_site", val=init_site, min_val=jmd_n_len + 1,
                                  max_val=n_res, just_int=True)  # guards a very short 'sequence'
        df_feat_site = predictor(sequence, init_site)
        start = init_site - jmd_n_len
        if output == "static":
            view = self.plot_combined(
                df_feat=df_feat_site, pdb=pdb, uniprot=uniprot, col_imp=col_imp, col_val=col_val,
                shap_plot=shap_plot, tmd_len=tmd_len, start=start, chain=chain, sequence=sequence,
                mode=mode, focus=focus, focus_region=focus_region, size_by_impact=size_by_impact,
                normalize_by_span=normalize_by_span)
            if path is not None:
                view.savefig(path)   # PNG/PDF/... of the feature-map panel (structure is interactive)
            return view
        # output == "html"
        view = self.plot_linked(
            df_feat=df_feat_site, pdb=pdb, uniprot=uniprot, col_imp=col_imp, col_val=col_val,
            shap_plot=shap_plot, tmd_len=tmd_len, start=start, chain=chain, sequence=sequence,
            mode=mode, focus=focus, focus_region=focus_region, size_by_impact=size_by_impact,
            normalize_by_span=normalize_by_span)
        if path is not None:
            view.write_html(path)
        return view

    # Internal orchestration helpers (map_structure)
    def _resolve_structure_persistent(self, pdb, uniprot, chain, sequence):
        """Resolve the structure once for interactive use; keep any temp dir alive.

        Returns ``(records, chain_id, pdb_path, tmp_holder)``. ``pdb_path`` stays valid for the
        panel's life (py3Dmol re-reads it each repaint); ``tmp_holder`` is the AlphaFold
        ``TemporaryDirectory`` (or ``None``) the caller must keep referenced.
        """
        tmp_holder = None
        if uniprot is not None:
            tmp_holder = tempfile.TemporaryDirectory()
            pdb_path = self._fetch_alphafold(uniprot, sequence, tmp_holder.name)
        else:
            pdb_path = pdb
        structure = load_structure(pdb_path)
        records, identity, chain_id = extract_chain_residues(
            structure, chain=chain, sequence=sequence)
        # Per-selection positions vary, so only the chain-identity check applies here.
        if sequence is not None and identity < 0.5:
            warnings.warn(
                f"The selected chain matches 'sequence' with low identity ({identity:.2f}); "
                f"the painted residues may be misaligned.", UserWarning, stacklevel=2)
        return records, chain_id, pdb_path, tmp_holder

    def _fetch_alphafold(self, uniprot, sequence, out_folder):
        """Fetch the AlphaFold model for ``uniprot`` into ``out_folder``; return its path."""
        from aaanalysis.data_handling_pro import StructurePreprocessor
        # The download is by accession; the sequence column only satisfies df_seq validation.
        seq = sequence if sequence is not None else "M"
        df_seq = pd.DataFrame({ut.COL_ENTRY: [uniprot], ut.COL_SEQ: [seq]})
        stp = StructurePreprocessor(verbose=self._verbose)
        df_status = stp.fetch_alphafold(df_seq=df_seq, out_folder=out_folder,
                                        file_format="pdb", on_failure="raise")
        model_path = df_status["model_path"].iloc[0]
        if not (isinstance(model_path, str) and os.path.isfile(model_path)):
            raise RuntimeError(f"AlphaFold model for '{uniprot}' could not be fetched "
                               f"(model_path={model_path})")
        return model_path

    @staticmethod
    def _check_start_alignment(records, positions_union, identity, sequence):
        """Warn if the mapped positions miss the structure (likely a wrong ``start``)."""
        if not positions_union:
            return
        struct_resis = residue_numbers(records)
        if struct_resis and not (set(positions_union) & struct_resis):
            warnings.warn(
                f"None of the mapped residue positions "
                f"[{positions_union[0]}..{positions_union[-1]}] are present in the "
                f"structure's residues; check that 'start' matches the structure numbering.",
                UserWarning, stacklevel=2)
        if sequence is not None and identity < 0.5:
            warnings.warn(
                f"The selected chain matches 'sequence' with low identity ({identity:.2f}); "
                f"the painted residues may be misaligned.",
                UserWarning, stacklevel=2)
