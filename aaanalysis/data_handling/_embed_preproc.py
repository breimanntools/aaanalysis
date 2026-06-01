"""
This is a script for the frontend of the EmbeddingPreprocessor class for
preparing protein-language-model (PLM) embeddings as inputs to ``CPP.run_num``.
"""
from typing import Dict, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.embed_preproc.build_pseudo_scales import build_pseudo_scales_
from ._backend.embed_preproc.cluster_pseudo_scales import cluster_pseudo_scales_


# I Helper Functions
def _check_dict_num(df_seq, dict_num):
    """Validate that ``dict_num`` is a dict keyed by entry, each value a
    2D ndarray whose first axis matches the corresponding sequence length,
    and that all entries share the same D."""
    if not isinstance(dict_num, dict):
        raise ValueError(f"'dict_num' ({type(dict_num).__name__}) should be a dict mapping entry to np.ndarray of shape (L, D).")
    entries = df_seq[ut.COL_ENTRY].tolist()
    missing = [e for e in entries if e not in dict_num]
    if missing:
        preview = missing[:5] + (["..."] if len(missing) > 5 else [])
        raise ValueError(f"'dict_num' ({len(missing)} missing entries) should contain every entry in 'df_seq'. Missing: {preview}")
    seqs = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
    ds = []
    for entry in entries:
        emb = dict_num[entry]
        if not isinstance(emb, np.ndarray):
            raise ValueError(f"'dict_num[{entry!r}]' ({type(emb).__name__}) should be np.ndarray of shape (L, D).")
        if emb.ndim != 2:
            raise ValueError(f"'dict_num[{entry!r}]' (ndim={emb.ndim}) should be 2D of shape (L, D).")
        seq_len = len(seqs[entry])
        if emb.shape[0] != seq_len:
            raise ValueError(f"'dict_num[{entry!r}].shape[0]' ({emb.shape[0]}) should equal sequence length ({seq_len}).")
        ds.append(emb.shape[1])
    if len(set(ds)) > 1:
        raise ValueError(f"'dict_num' (D values: {sorted(set(ds))}) should have a consistent embedding dimensionality D across all entries.")


def _check_match_cat_subcat_thresholds(cat_min_th, subcat_min_th):
    """Subcat threshold (tighter) must exceed cat threshold (looser)."""
    if cat_min_th >= subcat_min_th:
        raise ValueError(
            f"'cat_min_th' ({cat_min_th}) should be < 'subcat_min_th' ({subcat_min_th}). "
            f"Coarser pseudo-categories require a lower correlation threshold than finer ones."
        )


def _check_df_scales_emb_min_dims(df_scales_emb, min_dims=3):
    """At least ``min_dims`` pseudo-scales (columns) are required: AAclust needs
    ≥3 samples to estimate the lower bound of k."""
    D = df_scales_emb.shape[1]
    if D < min_dims:
        raise ValueError(
            f"'df_scales_emb' (D={D}, shape={df_scales_emb.shape}) should have at least "
            f"{min_dims} columns — AAclust requires ≥{min_dims} pseudo-scales to cluster."
        )


def _check_match_df_scales_emb_df_stds_emb(df_scales_emb, df_stds_emb):
    """If ``df_stds_emb`` is supplied, its shape, index, and columns must match
    ``df_scales_emb`` exactly — they describe the same dimensions across the
    same AAs."""
    if df_stds_emb is None:
        return
    if df_stds_emb.shape != df_scales_emb.shape:
        raise ValueError(
            f"'df_stds_emb' (shape={df_stds_emb.shape}) should have the same shape as "
            f"'df_scales_emb' (shape={df_scales_emb.shape})."
        )
    if not df_stds_emb.index.equals(df_scales_emb.index):
        raise ValueError(
            "'df_stds_emb' should have the same index as 'df_scales_emb' "
            "(AAs in the same order)."
        )
    if not df_stds_emb.columns.equals(df_scales_emb.columns):
        raise ValueError(
            "'df_stds_emb' should have the same columns as 'df_scales_emb' "
            "(dimension labels in the same order)."
        )


# II Main Functions
class EmbeddingPreprocessor:
    """
    Utility data preprocessing class for protein-language-model (PLM) embeddings.

    Instance-based, mirroring :class:`StructurePreprocessor` and
    :class:`AnnotationPreprocessor` (``ep = EmbeddingPreprocessor()``). The
    two-step workflow turns per-residue PLM embeddings into the
    (df_scales, df_cat) pair that :meth:`CPP.run` consumes:

    1. :meth:`build_pseudo_scales` averages per-residue embedding values per
       canonical AA, producing a (20, D) ``df_scales_emb`` analog of
       AAontology ``df_scales``.
    2. :meth:`cluster_pseudo_scales` runs AAclust at two correlation
       thresholds to produce a (D, 5) ``df_cat_emb`` analog of AAontology
       ``df_cat`` with ``cat`` / ``subcat`` columns.

    The resulting pair is drop-in compatible with the existing
    :meth:`CPP.run` constructor (``df_scales=df_scales_emb``,
    ``df_cat=df_cat_emb``). See the *Notes* on the semantic compromise of
    context-free averaging.

    .. versionadded:: 1.1.0

    See Also
    --------
    * :class:`SequencePreprocessor` for sequence-side preprocessing utilities.
    * :class:`StructurePreprocessor` for the PDB / DSSP / AlphaFold analog.
    * :class:`AnnotationPreprocessor` for the PTM / functional-site analog.
    * :class:`AAclust` for the correlation-based clustering used internally.
    * :class:`CPP` for the downstream feature-engineering consumer.
    * :func:`aaanalysis.combine_dict_nums` for stitching multiple dict_nums.

    Notes
    -----
    * **Pseudo-scales are dataset-dependent.** The same PLM applied to
      different protein corpora yields different pseudo-scales (and therefore
      different pseudo-categories). For reproducible cross-dataset comparison,
      compute pseudo-scales once on a fixed reference corpus and reuse them.
    * **Per-AA averaging discards positional context.** Pseudo-scales collapse
      a PLM's contextual per-residue embedding into a single per-AA value, so
      when used through :meth:`CPP.run` the same AA in different sequence
      positions receives the same scale value. To preserve positional context,
      pass the per-residue ``dict_num`` to :meth:`CPP.run_num` directly instead
      of routing through pseudo-scales.
    * **Feature categorization.** As of v1.1 every PLM-derived dim emits
      ``category='Embeddings'`` (paired with
      ``ut.DICT_COLOR_CAT['Embeddings'] == '#6B4FB5'``). The AAclust-derived
      coarser/finer cluster IDs move into a structured
      ``Embeddings_cat<i>_subcat<j>`` string in the ``subcategory`` column,
      so the redundancy filter still sees the cluster split via
      ``subcategory`` while ``CPPPlot.heatmap()`` resolves a single color.
    """

    def __init__(self, verbose: bool = True):
        self._verbose = ut.check_verbose(verbose)

    def build_pseudo_scales(
        self,
        df_seq: pd.DataFrame = None,
        dict_num: Dict[str, np.ndarray] = None,
        return_std: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build pseudo-scales by context-free averaging of per-residue embeddings.

        For each canonical amino acid ``a`` and each embedding dimension ``d``,
        the pseudo-scale entry is the mean of ``embeddings[entry][i, d]`` over
        all (entry, i) pairs where ``seq[i] == a``, taken over the input
        ``df_seq``. Non-canonical residues are skipped; AAs absent from the
        corpus get NaN rows.

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
        df_scales_emb : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame. Rows are the 20 canonical amino acids in
            alphabetical order (``ACDEFGHIKLMNPQRSTVWY``); columns are
            dimension labels (``dim_0``, ``dim_1``, …, ``dim_{D-1}``). Cells
            are context-free per-AA means of embedding values.
        df_stds_emb : pd.DataFrame, shape (20, D)
            Per-AA standard deviations, returned only when ``return_std=True``.
            Same index and columns as ``df_scales_emb``.

        Warns
        -----
        UserWarning
            Pseudo-scales depend on the content of ``df_seq``. The same
            embedding model applied to a different protein corpus produces a
            different pseudo-scale DataFrame.

        See Also
        --------
        cluster_pseudo_scales : derive a two-level pseudo-category table from this output.
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        _check_dict_num(df_seq=df_seq, dict_num=dict_num)
        ut.check_bool(name="return_std", val=return_std)
        warnings.warn(
            "Pseudo-scales are dataset-dependent (averaged over df_seq). "
            "For reproducible cross-dataset comparison, compute them once on a "
            "fixed reference corpus and reuse the resulting df_scales_emb.",
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
        df_scales_emb = pd.DataFrame(means, index=list_aa, columns=cols)
        if not return_std:
            return df_scales_emb
        df_stds_emb = pd.DataFrame(stds, index=list_aa, columns=cols)
        return df_scales_emb, df_stds_emb

    def cluster_pseudo_scales(
        self,
        df_scales_emb: pd.DataFrame = None,
        df_stds_emb: pd.DataFrame = None,
        cat_min_th: float = 0.5,
        subcat_min_th: float = 0.7,
        metric: str = "correlation",
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Cluster pseudo-scales into a two-level pseudo-category table via AAclust.

        Two independent :class:`AAclust` runs at different correlation
        thresholds produce coarser ``cat`` labels and finer ``subcat`` labels
        for each embedding dimension. Mirrors the AAontology ``df_cat`` schema
        so the result is a drop-in for the ``df_cat`` argument of
        :meth:`CPP.__init__`.

        When ``df_stds_emb`` is supplied, clustering becomes **std-aware**:
        each dimension is represented by the per-column z-scored concatenation
        of its per-AA ``(mean, std)`` (shape ``(D, 40)`` instead of
        ``(D, 20)``). Two dimensions with similar per-AA means but very
        different per-AA stds will then *not* collapse into the same cluster.

        Parameters
        ----------
        df_scales_emb : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame produced by :meth:`build_pseudo_scales`
            (or a user-supplied analog with the same shape). Must have at
            least 3 columns.
        df_stds_emb : pd.DataFrame, shape (20, D), optional
            Per-AA standard deviations matching ``df_scales_emb`` exactly in
            shape, index, and columns. Produce via
            ``build_pseudo_scales(..., return_std=True)``. When supplied,
            enables std-aware clustering (see Notes); when ``None`` (default),
            mean-only clustering is used. Must contain no NaN — drop the same
            rows you dropped from ``df_scales_emb``.
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
        df_cat_emb : pd.DataFrame, shape (D, 5)
            Pseudo-category DataFrame with columns ``scale_id``, ``category``
            (``"PLM_cat_<k>"``), ``subcategory`` (``"PLM_subcat_<k>"``),
            ``scale_name``, ``scale_description``. The ``scale_id`` column
            matches the column labels of ``df_scales_emb``.

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
        :class:`AAclust` : the underlying clustering algorithm.
        build_pseudo_scales : produces the expected input(s); pass
            ``return_std=True`` to get ``df_stds_emb`` for std-aware mode.

        References
        ----------
        .. [MilliganCooper88] Milligan & Cooper 1988, *A study of standardization of
           variables in cluster analysis*, J. Classification 5(2):181-204.
        .. [Eisen98] Eisen et al. 1998, *Cluster analysis and display of
           genome-wide expression patterns*, PNAS 95(25):14863-14868.
        """
        # Validate
        ut.check_df(name="df_scales_emb", df=df_scales_emb, accept_none=False)
        _check_df_scales_emb_min_dims(df_scales_emb=df_scales_emb)
        ut.check_df(name="df_stds_emb", df=df_stds_emb, accept_none=True, accept_nan=False)
        _check_match_df_scales_emb_df_stds_emb(df_scales_emb=df_scales_emb, df_stds_emb=df_stds_emb)
        ut.check_number_range(name="cat_min_th", val=cat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="subcat_min_th", val=subcat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_str_options(name="metric", val=metric, list_str_options=["correlation", "cosine"])
        ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True)
        _check_match_cat_subcat_thresholds(cat_min_th=cat_min_th, subcat_min_th=subcat_min_th)
        # Cluster
        df_cat_emb = cluster_pseudo_scales_(
            df_scales_emb=df_scales_emb,
            df_stds_emb=df_stds_emb,
            cat_min_th=cat_min_th,
            subcat_min_th=subcat_min_th,
            random_state=random_state,
            metric=metric,
        )
        return df_cat_emb
