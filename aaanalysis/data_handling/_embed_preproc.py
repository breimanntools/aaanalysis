"""
This is a script for the frontend of the EmbeddingPreprocessor class for
preparing protein-language-model (PLM) embeddings as inputs to ``CPP.run_num``.
"""
from typing import Dict, Tuple, Union, Literal, Optional
import warnings

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.embed_preproc.encode import encode_
from ._backend.embed_preproc.build_pseudo_scales import build_pseudo_scales_
from ._backend.embed_preproc.cluster_pseudo_scales import cluster_pseudo_scales_
from ._backend.embed_preproc import fetch as _fetch


# I Helper Functions
def _check_handle_failure(on_failure):
    """Validate the shared ``on_failure`` policy (matches the fetch_* siblings)."""
    ut.check_str_options(name="on_failure", val=on_failure,
                         list_str_options=["nan", "drop", "raise"])


def _check_per_residue_dict(name, df_seq, arrays):
    """Validate that ``arrays`` is a dict keyed by entry, each value a
    2D ndarray whose first axis matches the corresponding sequence length,
    and that all entries share the same D."""
    if not isinstance(arrays, dict):
        raise ValueError(f"'{name}' ({type(arrays).__name__}) should be a dict mapping entry to np.ndarray of shape (L, D).")
    entries = df_seq[ut.COL_ENTRY].tolist()
    missing = [e for e in entries if e not in arrays]
    if missing:
        preview = missing[:5] + (["..."] if len(missing) > 5 else [])
        raise ValueError(f"'{name}' ({len(missing)} missing entries) should contain every entry in 'df_seq'. Missing: {preview}")
    seqs = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
    ds = []
    for entry in entries:
        emb = arrays[entry]
        if not isinstance(emb, np.ndarray):
            raise ValueError(f"'{name}[{entry!r}]' ({type(emb).__name__}) should be np.ndarray of shape (L, D).")
        if emb.ndim != 2:
            raise ValueError(f"'{name}[{entry!r}]' (ndim={emb.ndim}) should be 2D of shape (L, D).")
        seq_len = len(seqs[entry])
        if emb.shape[0] != seq_len:
            raise ValueError(f"'{name}[{entry!r}].shape[0]' ({emb.shape[0]}) should equal sequence length ({seq_len}).")
        ds.append(emb.shape[1])
    if len(set(ds)) > 1:
        raise ValueError(f"'{name}' (D values: {sorted(set(ds))}) should have a consistent embedding dimensionality D across all entries.")


def _check_match_cat_subcat_thresholds(cat_min_th, subcat_min_th):
    """Subcat threshold (tighter) must exceed cat threshold (looser)."""
    if cat_min_th >= subcat_min_th:
        raise ValueError(
            f"'cat_min_th' ({cat_min_th}) should be < 'subcat_min_th' ({subcat_min_th}). "
            f"Coarser pseudo-categories require a lower correlation threshold than finer ones."
        )


def _check_df_scales_min_dims(df_scales, min_dims=3):
    """At least ``min_dims`` pseudo-scales (columns) are required: AAclust needs
    ≥3 samples to estimate the lower bound of k."""
    D = df_scales.shape[1]
    if D < min_dims:
        raise ValueError(
            f"'df_scales' (D={D}, shape={df_scales.shape}) should have at least "
            f"{min_dims} columns — AAclust requires ≥{min_dims} pseudo-scales to cluster."
        )


def _check_match_df_scales_df_stds(df_scales, df_stds):
    """If ``df_stds`` is supplied, its shape, index, and columns must match
    ``df_scales`` exactly — they describe the same dimensions across the
    same AAs."""
    if df_stds is None:
        return
    if df_stds.shape != df_scales.shape:
        raise ValueError(
            f"'df_stds' (shape={df_stds.shape}) should have the same shape as "
            f"'df_scales' (shape={df_scales.shape})."
        )
    if not df_stds.index.equals(df_scales.index):
        raise ValueError(
            "'df_stds' should have the same index as 'df_scales' "
            "(AAs in the same order)."
        )
    if not df_stds.columns.equals(df_scales.columns):
        raise ValueError(
            "'df_stds' should have the same columns as 'df_scales' "
            "(dimension labels in the same order)."
        )


# II Main Functions
class EmbeddingPreprocessor:
    """
    Preprocessing class for protein language model (**PLM**) embeddings.

    Turns raw per-residue embeddings into the ``[0, 1]``-normalized ``dict_num``
    consumed by :meth:`CPP.run_num` (the primary, position-preserving path via
    :meth:`encode`), with a secondary scale-based path (``df_scales`` / ``df_cat``
    via :meth:`build_scales` / :meth:`build_cat`) for :meth:`CPP.run`.

    .. versionadded:: 1.1.0

    Attributes
    ----------
    norm_params_ : dict
        Per-dimension normalization parameters fitted by :meth:`encode`; set after
        the first ``encode`` call so the identical transform can be reproduced.
    """

    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * :class:`StructurePreprocessor`: the structure-side analog.
        * :class:`AnnotationPreprocessor`: the annotation-side analog.
        * :func:`aaanalysis.combine_dict_nums`: stitch multiple dict_nums.
        * :meth:`CPP.run_num`: the downstream consumer.

        Examples
        --------
        .. include:: examples/embp_encode.rst
        """
        self._verbose = ut.check_verbose(verbose)

    def encode(
        self,
        df_seq: pd.DataFrame,
        embeddings: Dict[str, np.ndarray],
        method: Literal["minmax", "quantile", "sigmoid"] = "minmax",
        clip: Tuple[float, float] = (1.0, 99.0),
        return_df: bool = False,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
        """
        Encode raw per-residue protein language model (PLM) embeddings into a
        ``[0, 1]``-normalized ``dict_num``.

        Raw PLM embeddings (ESM, ProtT5, …) are unbounded floats, whereas
        :meth:`CPP.run_num` expects per-residue values in ``[0, 1]`` (the same
        normalization convention as :class:`StructurePreprocessor` and
        :class:`AnnotationPreprocessor`). ``encode`` fits one normalizer **per
        embedding dimension** over the whole corpus (all residues of all
        proteins in ``df_seq``) and applies it to every entry, returning a
        ``dict_num`` that feeds straight into
        :meth:`NumericalFeature.get_parts` → :meth:`CPP.run_num`. The fitted
        parameters are stored on the instance (``self.norm_params_``) so the
        identical transform can be reproduced.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            Defines which entries are encoded and validates that each embedding
            array's length matches its sequence.
        embeddings : dict[str, np.ndarray]
            Mapping from entry to a raw per-residue embedding array of shape
            ``(L, D)`` where ``L`` is the protein length and ``D`` is the
            embedding dimensionality. Every entry in ``df_seq`` must be a key;
            all arrays must share the same ``D``. You compute these externally
            with your PLM of choice — AAanalysis does not run the model.
        method : {'minmax', 'quantile', 'sigmoid'}, default='minmax'
            Per-dimension normalization to ``[0, 1]``. ``'minmax'`` linearly
            rescales each dim between its corpus min and max; ``'quantile'``
            does the same between robust percentiles (see ``clip``) so outlier
            residues do not crush the range; ``'sigmoid'`` z-scores each dim
            and applies a logistic squash.
        clip : tuple of float, default=(1.0, 99.0)
            Lower / upper percentiles used only when ``method='quantile'``.
        return_df : bool, default=False
            If ``True``, also return an echo of ``df_seq`` as a second element.

        Returns
        -------
        dict_num : dict[str, np.ndarray]
            ``{entry: (L, D) ndarray}`` with all values in ``[0, 1]``, same
            shape as ``embeddings``. Stack with other per-residue tensors via
            :func:`aaanalysis.combine_dict_nums`, slice with
            :meth:`NumericalFeature.get_parts`, then run :meth:`CPP.run_num`.
        df_seq_out : pd.DataFrame
            Returned only when ``return_df=True``. Echo of ``df_seq``.

        Notes
        -----
        * The normalizer is fit over the supplied corpus, so it is
          **dataset-dependent**: the same embeddings normalized against a
          different ``df_seq`` yield different values. For reproducible
          cross-dataset comparison, fit once on a fixed reference corpus.

        See Also
        --------
        * :meth:`build_scales`: the secondary context-free amino acid (AA)-scale path (for CPP.run).
        * :meth:`CPP.run_num`: the per-residue consumer of the returned dict_num.
        * :func:`aaanalysis.combine_dict_nums`: stitch several dict_nums together.

        Examples
        --------
        .. include:: examples/embp_encode.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        _check_per_residue_dict(name="embeddings", df_seq=df_seq, arrays=embeddings)
        ut.check_str_options(name="method", val=method,
                             list_str_options=["minmax", "quantile", "sigmoid"])
        ut.check_bool(name="return_df", val=return_df)
        if method == "quantile":
            q_lo, q_hi = clip
            ut.check_number_range(name="clip[0]", val=q_lo, min_val=0.0, max_val=100.0, just_int=False)
            ut.check_number_range(name="clip[1]", val=q_hi, min_val=0.0, max_val=100.0, just_int=False)
            if q_lo >= q_hi:
                raise ValueError(f"'clip' ({clip}) should be (lower, upper) with lower < upper.")
        # Encode
        entries = df_seq[ut.COL_ENTRY].tolist()
        dict_num, params = encode_(dict_num=embeddings, entries=entries, method=method, clip=clip)
        self.norm_params_ = params
        if return_df:
            return dict_num, df_seq.copy()
        return dict_num

    def build_scales(
        self,
        df_seq: pd.DataFrame,
        dict_num: Dict[str, np.ndarray],
        return_std: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build pseudo-scales by context-free averaging of per-residue embeddings.

        For each canonical amino acid (AA) ``a`` and each embedding dimension ``d``,
        the pseudo-scale entry is the mean of ``embeddings[entry][i, d]`` over
        all (entry, i) pairs where ``seq[i] == a``, taken over the input
        ``df_seq``. Non-canonical residues are skipped; AAs absent from the
        corpus get NaN rows.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            Used here as the source of empirical amino-acid contexts over
            which embedding dimensions are averaged.
        dict_num : dict[str, np.ndarray]
            Mapping from entry to a per-residue embedding array of shape
            ``(L, D)`` where ``L`` is the protein length and ``D`` is the
            embedding dimensionality. Every entry in ``df_seq`` must be a key;
            all arrays must share the same ``D``. Same shape contract as the
            ``dict_num`` consumed by :meth:`CPP.run_num`.
        return_std : bool, default=False
            If ``True``, also return per-AA population standard deviations in a
            second DataFrame of the same shape. AAs occurring exactly once
            receive std=0; AAs absent from the corpus receive NaN.

        Returns
        -------
        df_scales : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame. Rows are the 20 canonical amino acids in
            alphabetical order (``ACDEFGHIKLMNPQRSTVWY``); columns are
            dimension labels (``dim_0``, ``dim_1``, …, ``dim_{D-1}``). Cells
            are context-free per-AA means of embedding values. Drop-in for the
            ``df_scales`` argument of :meth:`CPP.__init__`.
        df_stds : pd.DataFrame, shape (20, D)
            Per-AA standard deviations, returned only when ``return_std=True``.
            Same index and columns as ``df_scales``.

        Warnings
        --------
        UserWarning
            Pseudo-scales depend on the content of ``df_seq``. The same
            embedding model applied to a different protein corpus produces a
            different pseudo-scale DataFrame.

        See Also
        --------
        * :meth:`build_cat`: derive a two-level pseudo-category table from this output.
        * :meth:`encode`: the primary per-residue path (raw embeddings to a [0, 1] dict_num).

        Examples
        --------
        .. include:: examples/embp_build_scales.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        _check_per_residue_dict(name="dict_num", df_seq=df_seq, arrays=dict_num)
        ut.check_bool(name="return_std", val=return_std)
        warnings.warn(
            "Pseudo-scales are dataset-dependent (averaged over df_seq). "
            "For reproducible cross-dataset comparison, compute them once on a "
            "fixed reference corpus and reuse the resulting df_scales.",
            UserWarning,
            stacklevel=2,
        )
        # Build
        list_aa = list(ut.LIST_CANONICAL_AA)
        result = build_pseudo_scales_(
            df_seq=df_seq,
            dict_num=dict_num,
            list_aa=list_aa,
            col_entry=ut.COL_ENTRY,
            col_seq=ut.COL_SEQ,
            return_std=return_std,
        )
        if return_std:
            means, stds = result
        else:
            means = result
        D = means.shape[1]
        cols = [f"dim_{i}" for i in range(D)]
        df_scales = pd.DataFrame(means, index=list_aa, columns=cols)
        if not return_std:
            return df_scales
        df_stds = pd.DataFrame(stds, index=list_aa, columns=cols)
        return df_scales, df_stds

    def build_cat(
        self,
        df_scales: pd.DataFrame,
        df_stds: Optional[pd.DataFrame] = None,
        cat_min_th: float = 0.5,
        subcat_min_th: float = 0.7,
        metric: Literal["correlation", "cosine"] = "correlation",
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Build a two-level pseudo-category table by clustering pseudo-scales via AAclust.

        Two independent :class:`AAclust` runs at different correlation
        thresholds produce coarser ``cat`` labels and finer ``subcat`` labels
        for each embedding dimension. Mirrors the AAontology ``df_cat`` schema
        so the result is a drop-in for the ``df_cat`` argument of
        :meth:`CPP.__init__`.

        When ``df_stds`` is supplied, clustering becomes **std-aware**:
        each dimension is represented by the per-column z-scored concatenation
        of its per-amino acid (AA) ``(mean, std)`` (shape ``(D, 40)`` instead of
        ``(D, 20)``). Two dimensions with similar per-AA means but very
        different per-AA stds will then *not* collapse into the same cluster.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_scales : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame produced by :meth:`build_scales`
            (or a user-supplied analog with the same shape). Must have at
            least 3 columns.
        df_stds : pd.DataFrame, shape (20, D), optional
            Per-AA standard deviations matching ``df_scales`` exactly in
            shape, index, and columns. Produce via
            ``build_scales(..., return_std=True)``. When supplied,
            enables std-aware clustering (see Notes); when ``None`` (default),
            mean-only clustering is used. Must contain no NaN — drop the same
            rows you dropped from ``df_scales``.
        cat_min_th : float, default=0.5
            AAclust correlation threshold for the coarser (``cat``) level.
            Lower values produce fewer, larger clusters.
        subcat_min_th : float, default=0.7
            AAclust correlation threshold for the finer (``subcat``) level.
            Must be greater than ``cat_min_th``.
        metric : {'correlation', 'cosine'}, default='correlation'
            Distance metric forwarded to :meth:`AAclust.fit`. Controls the
            optional cluster-merging step and medoid selection only; the
            k-optimization phase is always Pearson-correlation-based.
        random_state : int, default=0
            Random seed threaded through AAclust for reproducible cluster IDs.

        Returns
        -------
        df_cat : pd.DataFrame, shape (D, 5)
            Pseudo-category DataFrame with columns ``scale_id``, ``category``
            (``"PLM_cat_<k>"``), ``subcategory`` (``"PLM_subcat_<k>"``),
            ``scale_name``, ``scale_description``. The ``scale_id`` column
            matches the column labels of ``df_scales``. Drop-in for the
            ``df_cat`` argument of :meth:`CPP.__init__`.

        Notes
        -----
        * The two AAclust runs are independent. ``subcat`` labels do **not**
          necessarily nest within ``cat`` labels — they are two views over the
          same pseudo-scales at different correlation thresholds.
        * The ``metric`` parameter only affects post-hoc merging. To
          experiment with non-Pearson similarity during k-optimization, a
          deeper AAclust change is required.
        * **Std-aware recipe (when ``df_stds_emb`` is supplied).** A
          composition of three textbook ingredients, not a single named
          method: (i) each dimension is represented by its per-AA
          ``(mean, std)`` — the sufficient statistics of a 1-D Gaussian over
          that AA's residue embeddings; (ii) per-column z-scoring across the
          D dimensions puts the mean half and std half on a common footing
          so neither dominates row-Pearson [MilliganCooper88]_; (iii) AAclust
          then clusters the (D, 40) descriptor matrix by Pearson
          row-correlation, in the same tradition as gene-expression feature
          clustering [Eisen98]_. This recipe is **not** a closed-form
          approximation of Bhattacharyya / symmetric-KL between per-AA
          Gaussians (under equal variance Bhattacharyya reduces to a function
          of ``(μ₁ − μ₂)²`` alone, which would motivate dropping the std
          half).

        See Also
        --------
        * :class:`AAclust`: the underlying clustering algorithm.
        * :meth:`build_scales`: produces the expected input(s); pass
          ``return_std=True`` to get ``df_stds`` for std-aware mode.

        Examples
        --------
        .. include:: examples/embp_build_cat.rst
        """
        # Validate
        ut.check_df(name="df_scales", df=df_scales, accept_none=False)
        _check_df_scales_min_dims(df_scales=df_scales)
        ut.check_df(name="df_stds", df=df_stds, accept_none=True, accept_nan=False)
        _check_match_df_scales_df_stds(df_scales=df_scales, df_stds=df_stds)
        ut.check_number_range(name="cat_min_th", val=cat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="subcat_min_th", val=subcat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_str_options(name="metric", val=metric, list_str_options=["correlation", "cosine"])
        ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True)
        _check_match_cat_subcat_thresholds(cat_min_th=cat_min_th, subcat_min_th=subcat_min_th)
        # Cluster
        df_cat = cluster_pseudo_scales_(
            df_scales_emb=df_scales,
            df_stds_emb=df_stds,
            cat_min_th=cat_min_th,
            subcat_min_th=subcat_min_th,
            random_state=random_state,
            metric=metric,
        )
        return df_cat

    def fetch_embeddings(
        self,
        df_seq: pd.DataFrame,
        mode: Literal["protein", "residue"] = "protein",
        model: Literal["esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M",
                       "esm2_t33_650M", "esm2_t36_3B", "esm1b",
                       "prott5_xl_u50", "prostt5"] = "esm2_t12_35M",
        pooling: Literal["mean", "max", "cls"] = "mean",
        source: Literal["auto", "compute"] = "auto",
        batch_size: int = 8,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        max_length: Optional[int] = None,
        layer: int = -1,
        allow_oversized: bool = False,
        on_failure: Literal["nan", "drop", "raise"] = "nan",
        return_df: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray],
               Tuple[np.ndarray, pd.DataFrame], Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
        """
        Fetch and compute protein language model (PLM) embeddings for every entry.

        Downloads a curated model (ESM-2, ESM-1b, ProtT5, ProstT5) from the Hugging
        Face Hub and computes its embeddings, returning either one vector per protein
        (``mode='protein'``) or a per-residue array per protein (``mode='residue'``).
        The per-residue output is the raw, unbounded ``{entry: (L, D)}`` mapping that
        :meth:`encode` normalizes into the ``dict_num`` consumed by :meth:`CPP.run_num`;
        the per-protein output is a redundancy-free feature matrix ready for
        :meth:`AAclust.select_proteins` or :class:`TreeModel`. Embeddings are returned
        **raw** — normalization is :meth:`encode`'s job. Requires the ``embed`` extra
        (``pip install 'aaanalysis[embed]'``); the heavy dependencies are imported
        lazily, so the rest of the class works without them.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and
            a ``sequence`` column with full protein sequences. Output rows are aligned to
            ``df_seq``.
        mode : {'protein', 'residue'}, default='protein'
            ``'protein'`` returns one pooled vector per protein; ``'residue'`` returns
            the per-residue ``(L, D)`` array per protein (feeds :meth:`encode`).
        model : str, default='esm2_t12_35M'
            Registry key of the PLM to use — one of ``'esm2_t6_8M'``,
            ``'esm2_t12_35M'``, ``'esm2_t30_150M'``, ``'esm2_t33_650M'``,
            ``'esm2_t36_3B'``, ``'esm1b'``, ``'prott5_xl_u50'``, ``'prostt5'``. See the
            *Notes* table for each model's size, embedding dimension, and whether it runs
            on a typical laptop CPU. An unknown key raises a ``ValueError`` listing the
            valid options.
        pooling : {'mean', 'max', 'cls'}, default='mean'
            Residue→protein reduction for ``mode='protein'``. ``'cls'`` uses the model's
            leading token and is only valid for models that have one (ESM, not ProtT5).
        source : {'auto', 'compute'}, default='auto'
            Acquisition path. Both currently compute locally; ``'uniprot'`` (direct
            fetch of precomputed embeddings) is reserved for a future release.
        batch_size : int, default=8
            Number of sequences embedded per forward pass.
        device : {'auto', 'cpu', 'cuda', 'mps'}, default='auto'
            Compute device; ``'auto'`` picks CUDA, then Apple MPS, else CPU.
        max_length : int, optional
            Truncate sequences to this many residues. Defaults to the model's own cap
            (e.g. 1022 for ESM-1b); longer sequences are truncated with a warning.
        layer : int, default=-1
            Hidden layer to read out; ``-1`` is the last layer.
        allow_oversized : bool, default=False
            If ``False``, a model whose estimated memory footprint exceeds the detected
            device memory raises a ``RuntimeWarning`` (with a smaller-model suggestion)
            but still runs. ``True`` suppresses the guard.
        on_failure : {'nan', 'drop', 'raise'}, default='nan'
            Policy for entries that fail to embed: ``'nan'`` keeps a NaN row/array and
            marks it not-ok; ``'drop'`` removes it; ``'raise'`` raises ``RuntimeError``.
        return_df : bool, default=False
            If ``True``, also return an echo of ``df_seq`` with a boolean
            ``embeddings_ok`` column.

        Returns
        -------
        embeddings : np.ndarray or dict
            ``np.ndarray`` of shape ``(n_samples, D)`` row-aligned to ``df_seq``
            (``mode='protein'``), or ``{entry: (L, D)}`` of raw per-residue arrays
            (``mode='residue'``).
        df_seq_out : pd.DataFrame
            Returned only when ``return_df=True``: an echo of ``df_seq`` plus a boolean
            ``embeddings_ok`` column.

        Notes
        -----
        **Available models.** Footprints are inference floors (real peak grows with
        ``batch_size`` × sequence length); *Local* marks models that run comfortably on a
        typical 16 GB laptop CPU. ESM-2 spans accuracy/speed trade-offs; ProtT5/ProstT5
        are stronger but heavier; ProstT5 is structure-aware (trained on 3Di tokens).

        .. list-table::
           :header-rows: 1
           :widths: 20 8 6 10 8 30

           * - model
             - params
             - dim
             - ~RAM (CPU)
             - Local?
             - best for
           * - ``esm2_t6_8M``
             - 8 M
             - 320
             - ~0.3 GB
             - yes
             - laptops, large corpora, quick tests
           * - ``esm2_t12_35M``
             - 35 M
             - 480
             - ~0.5 GB
             - yes
             - default; best size/quality on CPU
           * - ``esm2_t30_150M``
             - 150 M
             - 640
             - ~1.5 GB
             - yes
             - richer residue features, still CPU-fine
           * - ``esm2_t33_650M``
             - 650 M
             - 1280
             - ~3 GB
             - slow
             - strong; comfortable with a GPU
           * - ``esm2_t36_3B``
             - 3 B
             - 2560
             - ~12 GB
             - GPU
             - highest quality; needs a ≥12 GB GPU
           * - ``esm1b``
             - 650 M
             - 1280
             - ~3 GB
             - slow
             - ESM-1b parity; 1022-residue cap
           * - ``prott5_xl_u50``
             - 1.2 B
             - 1024
             - ~5 GB
             - GPU
             - ProtT5; matches UniProt's embeddings
           * - ``prostt5``
             - 1.2 B
             - 1024
             - ~5 GB
             - GPU
             - structure-aware (3Di) embeddings

        The larger ESM-2 models and both T5 models are slow on CPU and may exhaust memory;
        ``fetch_embeddings`` emits a ``RuntimeWarning`` suggesting a smaller model when the
        estimated footprint exceeds the detected device memory (override with
        ``allow_oversized=True``, lower ``batch_size``, or select a GPU via ``device``).

        **Compute locally vs. fetch precomputed (UniProt).** ``fetch_embeddings`` computes
        embeddings on your machine, which works for *any* sequence — mutants, designs, or
        non-model organisms not in any database. UniProt separately publishes
        **precomputed ProtT5 per-protein embeddings** for UniProtKB/Swiss-Prot and selected
        reference proteomes; when your proteins are covered and ProtT5 is acceptable,
        downloading those (currently a bulk per-proteome file indexed by accession) avoids
        local compute entirely. Prefer the precomputed route for large, fully-covered
        Swiss-Prot sets on a CPU-only machine; compute here when proteins are novel/mutated,
        when you need a non-ProtT5 model (e.g. ESM-2/ProstT5), or when you want per-residue
        output for :meth:`encode` → :meth:`CPP.run_num`. A ``source='uniprot'`` path for the
        precomputed route is reserved for a future release.

        * Embedding extraction is deterministic (eval mode), so no ``random_state`` /
          ``seed`` is needed.
        * Returned embeddings are raw (unbounded) floats; pass ``mode='residue'`` output
          to :meth:`encode` before :meth:`CPP.run_num`.

        See Also
        --------
        * :meth:`EmbeddingPreprocessor.pool_embeddings`: pool per-residue arrays into
          per-protein vectors explicitly.
        * :meth:`EmbeddingPreprocessor.encode`: normalize per-residue embeddings to ``[0, 1]``.
        * :meth:`AAclust.select_proteins`: cluster per-protein embeddings into representatives.

        Raises
        ------
        ValueError
            On invalid arguments (unknown ``model``, ``'cls'`` pooling on a model without
            a CLS token, a pre-existing ``embeddings_ok`` column, ...).
        ImportError
            If the ``embed`` extra (``torch`` / ``transformers``) is not installed.
        RuntimeError
            On an embedding failure under ``on_failure='raise'``.

        Examples
        --------
        .. include:: examples/embp_fetch_embeddings.rst
        """
        # Validate
        verbose = self._verbose
        ut.check_df_seq(df_seq=df_seq)
        ut.check_str_options(name="mode", val=mode, list_str_options=ut.LIST_EMBED_MODES)
        ut.check_str_options(name="model", val=model, list_str_options=_fetch.LIST_MODELS)
        ut.check_str_options(name="pooling", val=pooling, list_str_options=ut.LIST_POOLING)
        ut.check_str_options(name="source", val=source, list_str_options=ut.LIST_EMBED_SOURCES)
        ut.check_str_options(name="device", val=device, list_str_options=ut.LIST_EMBED_DEVICES)
        ut.check_number_range(name="batch_size", val=batch_size, min_val=1, just_int=True)
        ut.check_number_range(name="max_length", val=max_length, min_val=1, just_int=True, accept_none=True)
        ut.check_bool(name="allow_oversized", val=allow_oversized)
        ut.check_bool(name="return_df", val=return_df)
        _check_handle_failure(on_failure)
        if mode == "protein" and pooling == "cls" and not _fetch.REGISTRY[model]["has_cls"]:
            raise ValueError(f"'pooling' ('cls') is not available for model '{model}' "
                             f"(no CLS token); use 'mean' or 'max'.")
        if ut.COL_EMBEDDINGS_OK in df_seq.columns:
            raise ValueError(f"'df_seq' should not already contain a "
                             f"'{ut.COL_EMBEDDINGS_OK}' column. Drop it before calling "
                             f"fetch_embeddings.")
        # Resolve device + hardware guard
        hw = _fetch.detect_hardware()
        eff_device = hw["device"] if device == "auto" else device
        mem = hw["free_vram_gb"] if eff_device in ("cuda", "mps") else hw["total_ram_gb"]
        need = _fetch.estimate_footprint_gb(model, device=eff_device, batch_size=batch_size)
        rec = _fetch.recommend_model(mem_gb=mem, device=eff_device)
        if mem is not None and need > mem and not allow_oversized:
            warnings.warn(
                f"'model' ('{model}', ~{need:.1f} GB) likely exceeds available "
                f"{eff_device} memory ({mem:.1f} GB) at batch_size={batch_size}; this "
                f"may crash. Pass allow_oversized=True, lower batch_size, or use "
                f"'{rec}'.", RuntimeWarning)
        elif verbose:
            mem_str = f"{mem:.1f} GB" if mem is not None else "unknown memory"
            ut.print_out(f"hardware: {eff_device}, {mem_str} -> recommended model '{rec}'")
        # Compute
        embeddings, ok = _fetch.compute_embeddings_(
            df_seq=df_seq, model=model, mode=mode, pooling=pooling,
            batch_size=batch_size, device=eff_device, max_length=max_length,
            layer=layer, on_failure=on_failure)
        if not return_df:
            return embeddings
        df_seq_out = df_seq.copy()
        df_seq_out[ut.COL_EMBEDDINGS_OK] = ok
        return embeddings, df_seq_out

    def pool_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        pooling: Literal["mean", "max"] = "mean",
        df_seq: Optional[pd.DataFrame] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Pool per-residue embeddings into one vector per protein.

        Reduces a ``{entry: (L, D)}`` mapping (e.g. from
        :meth:`fetch_embeddings(mode='residue') <fetch_embeddings>` or your own PLM
        run) to one ``(D,)`` vector per protein. This is the simple statistical
        counterpart to the richer "pooling" that :meth:`CPP.run` / :meth:`CPP.run_num`
        perform when turning per-residue values into features.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        embeddings : dict
            Mapping ``{entry: (L, D)}`` of per-residue embedding arrays.
        pooling : {'mean', 'max'}, default='mean'
            Reduction over residues. (``'cls'`` is unavailable here — residue arrays
            carry no leading token; use ``fetch_embeddings(mode='protein', pooling='cls')``.)
        df_seq : pd.DataFrame, optional
            DataFrame containing an ``entry`` column with unique protein identifiers. If
            given, return a ``(n_samples, D)`` matrix row-aligned to ``df_seq`` instead of
            a dict.

        Returns
        -------
        pooled : dict or np.ndarray
            ``{entry: (D,)}`` of pooled vectors, or a ``(n_samples, D)`` matrix
            row-aligned to ``df_seq`` when ``df_seq`` is given.

        See Also
        --------
        * :meth:`EmbeddingPreprocessor.fetch_embeddings`: obtain the per-residue arrays.

        Raises
        ------
        ValueError
            On invalid ``pooling``, an empty ``embeddings`` dict, or an entry in
            ``df_seq`` missing from ``embeddings``.

        Examples
        --------
        .. include:: examples/embp_pool_embeddings.rst
        """
        # Validate
        ut.check_str_options(name="pooling", val=pooling, list_str_options=["mean", "max"])
        if not isinstance(embeddings, dict) or len(embeddings) == 0:
            raise ValueError("'embeddings' should be a non-empty dict mapping entry to "
                             "an (L, D) array.")
        pooled = {entry: _fetch.pool_residue_(arr, pooling=pooling)
                  for entry, arr in embeddings.items()}
        if df_seq is None:
            return pooled
        ut.check_df_seq(df_seq=df_seq)
        entries = df_seq[ut.COL_ENTRY].tolist()
        missing = [e for e in entries if e not in pooled]
        if missing:
            raise ValueError(f"'embeddings' ({len(missing)} missing) should contain every "
                             f"entry in 'df_seq'. Missing: {missing[:5]}")
        return np.vstack([pooled[e] for e in entries])
