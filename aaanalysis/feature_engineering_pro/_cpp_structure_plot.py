"""
This is a script for the frontend of the CPPStructurePlot class, painting
per-residue CPP / CPP-SHAP feature impact onto a 3D protein structure.
"""
import os
import tempfile
import warnings
from typing import Optional, List, Tuple, Union, Literal

import numpy as np
import pandas as pd

import aaanalysis.utils as ut

from ._backend.cpp_struct.mapping import compute_residue_impact
from ._backend.cpp_struct.structure import (load_structure, extract_chain_residues,
                                            residue_numbers)
from ._backend.cpp_struct.render import (render_py3dmol, render_mpl,
                                         draw_structure_mpl, py3dmol_available)
from ._backend.cpp_struct.view import StructureView

LIST_MODES = ["impact", "plddt"]
LIST_FOCUS = ["whole", "fade", "zoom"]
LIST_BACKENDS = ["py3dmol", "mpl"]


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


def _resolve_backend(backend):
    """Resolve the render backend; raise a friendly hint if py3Dmol is forced but absent."""
    if backend == "py3dmol" and not py3dmol_available():
        raise RuntimeError("backend='py3dmol' requires the optional 'py3Dmol' package; "
                           "install it via \"pip install 'aaanalysis[pro]'\" (or "
                           "\"pip install py3Dmol\"), or use backend='mpl'")
    if backend is not None:
        return backend
    return "py3dmol" if py3dmol_available() else "mpl"


# II Main Functions
class CPPStructurePlot:
    """
    Plotting class for painting :class:`CPP` feature impact onto a 3D protein structure
    (**[pro]**, requires ``aaanalysis[pro]``) [Breimann25]_.

    The per-residue impact is the same normalized-sum that the :meth:`CPPPlot.profile` and
    :meth:`CPPPlot.feature_map` show: each feature's signed impact is spread over the residue
    positions it spans and summed per position, then painted residue-by-residue onto the protein
    cartoon. A red-white-blue ramp shows where features raise (red) or lower (blue) the prediction;
    an AlphaFold pLDDT mode shows per-residue model confidence instead.

    The single method :meth:`CPPStructurePlot.map_structure` returns a ``StructureView`` — a
    thin wrapper exposing a uniform ``show`` / ``write_html`` / ``savefig`` surface over the
    interactive `py3Dmol <https://pypi.org/project/py3Dmol/>`_ backend and the static matplotlib
    fallback (which is used automatically when py3Dmol is absent or ``backend="mpl"``).

    .. versionadded:: 1.1.0

    Notes
    -----
    * The ``jmd_n_len`` and ``jmd_c_len`` values supplied at construction are stored as
      ``_jmd_n_len`` and ``_jmd_c_len`` and reused by :meth:`map_structure`, mirroring
      :class:`CPPPlot` so juxta-membrane domain (JMD) lengths stay consistent.
    * This is a ``pro`` feature (needs ``biopython``; ``py3Dmol`` for the interactive backend).

    """
    def __init__(self,
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 df_scales: Optional[pd.DataFrame] = None,
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
            DataFrame of scales with letters typically representing amino acids. Stored for
            consistency with :class:`CPPPlot`; not required for structure mapping.
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
        # General settings
        self._verbose = verbose
        self._df_scales = df_scales
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
                      backend: Optional[Literal["py3dmol", "mpl"]] = None,
                      ) -> StructureView:
        """
        Paint per-residue CPP feature impact onto a protein structure.

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
            If ``True``, draw a stick whose radius (or marker size) is proportional to
            ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            If ``False`` (default), add each feature's full impact to every residue it spans
            (app-fidelity colouring). If ``True``, divide each feature's impact by its span
            count first (the span-normalized sum of :meth:`CPPPlot.profile` and the
            :meth:`CPPPlot.feature_map` top per-position bar).
        backend : {'py3dmol', 'mpl'}, optional
            ``'py3dmol'`` (interactive) or ``'mpl'`` (static). Default uses py3Dmol when
            available, otherwise matplotlib.

        Returns
        -------
        view : StructureView
            A thin wrapper over the rendered view exposing ``show()``, ``write_html(path)``,
            ``savefig(path)`` (matplotlib backend), and ``_repr_html_`` for inline display, plus
            the mapped ``dict_impact`` / ``max_abs`` for inspection.

        Notes
        -----
        ``tmd_len``, ``start``, ``jmd_n_len`` and ``jmd_c_len`` must match the geometry used when
        the features were generated, otherwise the impact lands on the wrong residues.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. an unknown ``mode`` / ``focus`` / ``backend``, neither or
            both of ``pdb`` / ``uniprot``, a ``df_feat`` missing ``col_imp``, or an unknown ``chain``).
        RuntimeError
            If ``backend='py3dmol'`` is forced without py3Dmol installed, or an AlphaFold model
            for ``uniprot`` cannot be fetched.

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
        ut.check_str_options(name="backend", val=backend, list_str_options=LIST_BACKENDS,
                             accept_none=True)
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

        # Compute per-residue impact (shared CPP normalized-sum backend)
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=col_imp, start=start, tmd_len=tmd_len,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
            normalize_by_span=normalize_by_span)
        resolved_backend = _resolve_backend(backend)
        window_resis = _resolve_window_resis(focus, focus_region, positions_union)

        # Resolve the structure file, parse it, and render inside one temp context
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
            self._check_start_alignment(records, positions_union, identity, sequence)
            view = self._render(resolved_backend, pdb_path, records, dict_impact, max_abs,
                                mode, focus, window_resis, size_by_impact, chain_id)

        if self._verbose:
            struct_resis = residue_numbers(records)
            n_painted = sum(1 for r in struct_resis if abs(dict_impact.get(r, 0.0)) > 0)
            ut.print_out(f"CPPStructurePlot: mapped {len(df_feat)} features onto "
                         f"{len(records)} residues ({n_painted} carry non-zero impact), "
                         f"mode='{mode}', backend='{resolved_backend}'.")
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
                      figsize: Tuple[Union[int, float], Union[int, float]] = (15, 8),
                      width_ratios: Tuple[Union[int, float], Union[int, float]] = (1.0, 1.6),
                      feature_map_dpi: int = 200,
                      feature_map_kws: Optional[dict] = None,
                      ) -> ut.FigAxResult:
        """
        Plot the structure and the CPP feature map side by side in one static figure.

        Reproduces the deployed app's signature layout: the **left** panel paints per-residue
        CPP feature impact onto the protein (matplotlib ``mplot3d``, zoomed to the feature
        window), the **right** panel is the :meth:`CPPPlot.feature_map` of the same ``df_feat``.
        Both panels read the same per-residue impact, so the structure colours and the feature
        map tell one consistent story. The figure is a single matplotlib :class:`~matplotlib.figure.Figure`
        that ``savefig`` exports to PNG / PDF (the feature-map panel is embedded as a
        high-resolution raster at ``feature_map_dpi``).

        For an interactive, rotatable 3D view use :meth:`map_structure` with the py3Dmol backend.

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
            If ``True``, scale each impact residue's marker by ``|impact|`` (impact mode only).
        normalize_by_span : bool, default=False
            Per-residue aggregation for the structure colouring; see :meth:`map_structure`.
        tmd_seq, jmd_n_seq, jmd_c_seq : str, optional
            TMD / JMD-N / JMD-C sequences shown along the feature-map x-axis.
        figsize : tuple, default=(15, 8)
            Size of the combined figure in inches.
        width_ratios : tuple, default=(1.0, 1.6)
            Width ratio of the (structure, feature-map) panels.
        feature_map_dpi : int, default=200
            Resolution of the embedded feature-map raster (>=50).
        feature_map_kws : dict, optional
            Extra keyword arguments forwarded to :meth:`CPPPlot.feature_map`. Keys already
            controlled by this method (e.g. ``df_feat``, ``col_val``, ``col_imp``, ``tmd_len``,
            ``start``, ``shap_plot``, the part sequences) are rejected.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The combined figure.
        ax : numpy.ndarray of matplotlib.axes.Axes
            The ``(ax_structure, ax_feature_map)`` pair.

        Raises
        ------
        ValueError
            On invalid arguments (e.g. an unknown ``mode`` / ``focus``, neither or both of
            ``pdb`` / ``uniprot``, a ``df_feat`` missing ``col_imp``, or a colliding
            ``feature_map_kws`` key).

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
        ut.check_tuple(name="figsize", val=figsize, n=2, check_number=True)
        ut.check_tuple(name="width_ratios", val=width_ratios, n=2, check_number=True)
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

        # Compute per-residue impact (shared CPP backend) and resolve the structure
        dict_impact, max_abs, positions_union = compute_residue_impact(
            df_feat=df_feat, col_imp=col_imp, start=start, tmd_len=tmd_len,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
            normalize_by_span=normalize_by_span)
        window_resis = _resolve_window_resis(focus, focus_region, positions_union)
        records, _chain_id = self._prepare_records(pdb, uniprot, chain, sequence,
                                                   positions_union)

        # Build the combined figure: structure (left) + feature_map raster (right)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=0.02)
        ax_struct = fig.add_subplot(gs[0, 0], projection="3d")
        ax_map = fig.add_subplot(gs[0, 1])
        draw_structure_mpl(ax=ax_struct, records=records, dict_impact=dict_impact,
                           max_abs=max_abs, mode=mode, focus=focus,
                           window_resis=window_resis, size_by_impact=size_by_impact)
        image = self._feature_map_image(df_feat=df_feat, col_val=col_val, col_imp=col_imp,
                                        shap_plot=shap_plot, tmd_len=tmd_len, start=start,
                                        tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
                                        jmd_c_seq=jmd_c_seq, dpi=feature_map_dpi,
                                        feature_map_kws=feature_map_kws)
        ax_map.imshow(image)
        ax_map.axis("off")

        if self._verbose:
            ut.print_out(f"CPPStructurePlot: combined view of {len(df_feat)} features on "
                         f"{len(records)} residues, mode='{mode}'.")
        return ut.FigAxResult(fig, np.array([ax_struct, ax_map], dtype=object))

    # Internal orchestration helpers
    def _prepare_records(self, pdb, uniprot, chain, sequence, positions_union):
        """Resolve / parse the structure and return its (records, chain_id) for mpl drawing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdb_path = pdb if uniprot is None else self._fetch_alphafold(uniprot, sequence, tmp_dir)
            structure = load_structure(pdb_path)
            records, identity, chain_id = extract_chain_residues(
                structure, chain=chain, sequence=sequence)
        # records hold plain coordinates/pLDDT, safe to use after the temp dir is gone.
        self._check_start_alignment(records, positions_union, identity, sequence)
        return records, chain_id

    def _feature_map_image(self, df_feat, col_val, col_imp, shap_plot, tmd_len, start,
                           tmd_seq, jmd_n_seq, jmd_c_seq, dpi, feature_map_kws):
        """Render CPPPlot.feature_map to its own figure and return it as an RGBA image array."""
        import io
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from aaanalysis.feature_engineering._cpp_plot import CPPPlot
        cpp_plot = CPPPlot(df_scales=self._df_scales, jmd_n_len=self._jmd_n_len,
                           jmd_c_len=self._jmd_c_len, accept_gaps=True, verbose=False)
        fig_fm, _ax_fm = cpp_plot.feature_map(
            df_feat=df_feat, shap_plot=shap_plot, col_val=col_val, col_imp=col_imp,
            tmd_len=tmd_len, start=start, tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq,
            jmd_c_seq=jmd_c_seq, **feature_map_kws)
        buffer = io.BytesIO()
        fig_fm.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_fm)
        buffer.seek(0)
        return mpimg.imread(buffer)

    # Internal orchestration helpers (map_structure)
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

    @staticmethod
    def _render(resolved_backend, pdb_path, records, dict_impact, max_abs, mode,
                focus, window_resis, size_by_impact, chain_id):
        """Dispatch to the resolved render backend and return the StructureView."""
        if resolved_backend == "py3dmol":
            return render_py3dmol(pdb_path=pdb_path, records=records,
                                  dict_impact=dict_impact, max_abs=max_abs, mode=mode,
                                  focus=focus, window_resis=window_resis,
                                  size_by_impact=size_by_impact, chain_id=chain_id)
        # The mpl backend only scatters the selected chain's records, so it never
        # leaks onto other chains and needs no chain qualifier.
        return render_mpl(records=records, dict_impact=dict_impact, max_abs=max_abs,
                          mode=mode, focus=focus, window_resis=window_resis,
                          size_by_impact=size_by_impact)
