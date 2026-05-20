"""
This is a script for the frontend of the EmbeddingPreprocessor class for
preparing protein-language-model (PLM) embeddings as inputs to ``CPP.run_embed``.

Mirrors the :class:`SequencePreprocessor` pattern: stateless namespace of
``@staticmethod`` utilities. Two-step workflow: ``build_pseudo_scales`` derives
a (20, D) pseudo-scale table by context-free averaging of per-residue
embeddings; ``cluster_pseudo_scales`` runs AAclust at two correlation
thresholds to produce a ``df_cat_emb`` that mirrors the AAontology two-level
category hierarchy. Together they provide the data structures ``CPP.run_embed``
needs to consume PLM-sourced features symmetrically with ``CPP.run``.
"""
# TODO this info is too long. keep some of the cross-references but move the rest to the docstring of the class and/or methods.
from typing import Dict
import warnings

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.embed_preproc.build_pseudo_scales import build_pseudo_scales_
from ._backend.embed_preproc.cluster_pseudo_scales import cluster_pseudo_scales_


# I Helper Functions
def _check_embeddings(df_seq, embeddings):
    """Validate that ``embeddings`` is a dict keyed by entry, each value a
    2D ndarray whose first axis matches the corresponding sequence length,
    and that all entries share the same D."""
    if not isinstance(embeddings, dict):
        raise ValueError(f"'embeddings' ({type(embeddings).__name__}) should be a dict mapping entry to np.ndarray of shape (L, D).")
    entries = df_seq[ut.COL_ENTRY].tolist()
    missing = [e for e in entries if e not in embeddings]
    if missing:
        preview = missing[:5] + (["..."] if len(missing) > 5 else [])
        raise ValueError(f"'embeddings' ({len(missing)} missing entries) should contain every entry in 'df_seq'. Missing: {preview}")
    seqs = dict(zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ]))
    ds = []
    for entry in entries:
        emb = embeddings[entry]
        if not isinstance(emb, np.ndarray):
            raise ValueError(f"'embeddings[{entry!r}]' ({type(emb).__name__}) should be np.ndarray of shape (L, D).")
        if emb.ndim != 2:
            raise ValueError(f"'embeddings[{entry!r}]' (ndim={emb.ndim}) should be 2D of shape (L, D).")
        seq_len = len(seqs[entry])
        if emb.shape[0] != seq_len:
            raise ValueError(f"'embeddings[{entry!r}].shape[0]' ({emb.shape[0]}) should equal sequence length ({seq_len}).")
        ds.append(emb.shape[1])
    if len(set(ds)) > 1:
        raise ValueError(f"'embeddings' (D values: {sorted(set(ds))}) should have a consistent embedding dimensionality D across all entries.")


def _check_match_cat_subcat_thresholds(cat_min_th, subcat_min_th):
    """Subcat threshold (tighter) must exceed cat threshold (looser)."""
    if cat_min_th >= subcat_min_th:
        raise ValueError(
            f"'cat_min_th' ({cat_min_th}) should be < 'subcat_min_th' ({subcat_min_th}). "
            f"Coarser pseudo-categories require a lower correlation threshold than finer ones."
        )


# II Main Functions
class EmbeddingPreprocessor:
    """
    Utility data preprocessing class for protein-language-model (PLM) embeddings.

    Produces the data structures ``CPP.run_embed`` consumes: pseudo-scales
    (a 20 × D analog of AAontology ``df_scales``) and pseudo-categories (a
    D-row analog of AAontology ``df_cat`` with ``cat`` / ``subcat`` columns
    derived by AAclust correlation clustering at two thresholds).

    See Also
    --------
    * :class:`SequencePreprocessor` for sequence-side preprocessing utilities.
    * :class:`AAclust` for the correlation-based clustering used internally.

    Notes
    -----
    Pseudo-scales are **dataset-dependent**: the same PLM applied to different
    protein corpora yields different pseudo-scales (and therefore different
    pseudo-categories). For reproducible cross-dataset comparison, compute
    pseudo-scales once on a fixed reference corpus and reuse them.
    """

    @staticmethod
    def build_pseudo_scales(
        df_seq: pd.DataFrame = None,
        embeddings: Dict[str, np.ndarray] = None,
    ) -> pd.DataFrame:
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
        embeddings : dict[str, np.ndarray]
            Mapping from entry to a per-residue embedding array of shape
            ``(L, D)`` where ``L`` is the protein length and ``D`` is the
            embedding dimensionality. Every entry in ``df_seq`` must be a key;
            all arrays must share the same ``D``.

        Returns
        -------
        df_scales_emb : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame. Rows are the 20 canonical amino acids in
            alphabetical order (``ACDEFGHIKLMNPQRSTVWY``); columns are
            dimension labels (``dim_0``, ``dim_1``, …, ``dim_{D-1}``). Cells
            are context-free per-AA means of embedding values.

        Warns
        -----
        UserWarning
            Pseudo-scales depend on the content of ``df_seq``. The same
            embedding model applied to a different protein corpus produces a
            different pseudo-scale DataFrame.

        See Also
        --------
        cluster_pseudo_scales : derive a two-level pseudo-category table from this output.

        Examples
        --------
        .. include:: examples/ep_build_pseudo_scales.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        _check_embeddings(df_seq=df_seq, embeddings=embeddings)
        warnings.warn(
            "Pseudo-scales are dataset-dependent (averaged over df_seq). "
            "For reproducible cross-dataset comparison, compute them once on a "
            "fixed reference corpus and reuse the resulting df_scales_emb.",
            UserWarning,
            stacklevel=2,
        )
        # Build
        list_aa = list(ut.LIST_CANONICAL_AA)
        means = build_pseudo_scales_(
            df_seq=df_seq,
            embeddings=embeddings,
            list_aa=list_aa,
            col_entry=ut.COL_ENTRY,
            col_seq=ut.COL_SEQ,
        )
        D = means.shape[1]
        df_scales_emb = pd.DataFrame(
            means,
            index=list_aa,
            columns=[f"dim_{i}" for i in range(D)],
        )
        return df_scales_emb

    @staticmethod
    def cluster_pseudo_scales(
        df_scales_emb: pd.DataFrame = None,
        cat_min_th: float = 0.5,
        subcat_min_th: float = 0.7,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Cluster pseudo-scales into a two-level pseudo-category table via AAclust.

        Two independent :class:`AAclust` runs at different correlation
        thresholds produce coarser ``cat`` labels and finer ``subcat`` labels
        for each embedding dimension. Mirrors the AAontology ``df_cat`` schema
        so the result is a drop-in for ``CPP.run_embed``'s ``df_cat`` argument.

        Parameters
        ----------
        df_scales_emb : pd.DataFrame, shape (20, D)
            Pseudo-scale DataFrame produced by :meth:`build_pseudo_scales`
            (or a user-supplied analog with the same shape).
        cat_min_th : float, default=0.5
            AAclust correlation threshold for the coarser (``cat``) level.
            Lower values produce fewer, larger clusters.
        subcat_min_th : float, default=0.7
            AAclust correlation threshold for the finer (``subcat``) level.
            Must be greater than ``cat_min_th``.
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
        The two AAclust runs are independent. ``subcat`` labels do **not**
        necessarily nest within ``cat`` labels — they are two views over the
        same pseudo-scales at different correlation thresholds.

        See Also
        --------
        :class:`AAclust` : the underlying clustering algorithm.
        build_pseudo_scales : produces the expected input.

        Examples
        --------
        .. include:: examples/ep_cluster_pseudo_scales.rst
        """
        # Validate
        ut.check_df(name="df_scales_emb", df=df_scales_emb, accept_none=False)
        ut.check_number_range(name="cat_min_th", val=cat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="subcat_min_th", val=subcat_min_th, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="random_state", val=random_state, min_val=0, just_int=True)
        _check_match_cat_subcat_thresholds(cat_min_th=cat_min_th, subcat_min_th=subcat_min_th)
        # Cluster
        df_cat_emb = cluster_pseudo_scales_(
            df_scales_emb=df_scales_emb,
            cat_min_th=cat_min_th,
            subcat_min_th=subcat_min_th,
            random_state=random_state,
        )
        return df_cat_emb
