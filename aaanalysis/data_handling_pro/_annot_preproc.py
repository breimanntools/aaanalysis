"""
This is a script for the frontend of the AnnotationPreprocessor: a pro-extra
class that fetches per-residue PTM / functional-site annotations from UniProt
(or ingests user/predictor labels), maps them into one canonical ``df_annot``
schema, and encodes them into ``[0, 1]``-normalized per-residue ``dict_num``
tensors for :meth:`CPP.run_num`.

The class is pro-extra gated: ``requests`` is required for the UniProt fetch.
It mirrors :class:`StructurePreprocessor`'s instance-based pattern — one source
per encoder, a registry of feature keys, ``build_scales`` (corpus-derived
``df_scales``) + ``build_cat`` (corpus-free ``df_cat``) — so its output stacks
with DSSP / PAE / embedding tensors via :func:`aaanalysis.combine_dict_nums`.

Two top-level categories are registered into ``ut.DICT_COLOR_CAT``: ``'PTMs'``
(closed UniProt vocabulary) and ``'Functional sites'`` (open vocabulary; user /
predictor keys auto-register at ingest time).
"""

import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.annot_preproc.feature_registry import (
    make_instance_registry,
    register_functional_key,
    validate_feature_keys,
    normalize,
    get_total_dims,
    get_dim_names,
    get_categories,
    get_subcategories,
    NORMALIZATION_RECIPES,
)
from ._backend.annot_preproc._uniprot import fetch_and_map

LIST_EVIDENCE_MODES = ["experimental", "manual", "all"]
LIST_ON_MISMATCH = ["raise", "drop", "warn"]

# Per-residue tags emitted by ``to_df_seq`` into the aa_context column. A
# reference anchor is eligible iff its tag == CONTEXT_TAG_ELIGIBLE; pass
# ``context_in="1"`` to AAWindowSampler to keep only those.
CONTEXT_TAG_ELIGIBLE = "1"
CONTEXT_TAG_EXCLUDED = "0"


# I Helper Functions
def _evidence_codes_for_mode(evidence: str) -> Optional[List[str]]:
    """Map the ``evidence=`` toggle to an ECO allow-set (``None`` = no filter)."""
    if evidence == "experimental":
        return list(ut.LIST_ECO_EXPERIMENTAL)
    if evidence == "manual":
        return list(ut.LIST_ECO_MANUAL)
    return None  # "all"


def _check_df_annot(df_annot, name="df_annot"):
    """Validate that ``df_annot`` carries the required canonical columns."""
    if not isinstance(df_annot, pd.DataFrame):
        raise ValueError(
            f"'{name}' ({type(df_annot).__name__}) should be a " f"pandas DataFrame"
        )
    required = [
        ut.COL_PROTEIN_ID,
        ut.COL_START,
        ut.COL_STOP,
        ut.COL_FEATURE_TYPE,
        ut.COL_SCORE,
    ]
    missing = [c for c in required if c not in df_annot.columns]
    if missing:
        raise ValueError(
            f"'{name}' is missing required columns {missing}; expected at "
            f"least {required}"
        )


def _check_scores_unit_range(scores, name="score"):
    """Raise if any non-NaN score lies outside [0, 1]."""
    arr = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy()
    finite = arr[~np.isnan(arr)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(
            f"'{name}' values should lie in [0, 1] (got min={finite.min()}, "
            f"max={finite.max()}). Pre-normalize predictor scores or register "
            f"the feature with a custom normalization recipe."
        )


# II Main Functions
class AnnotationPreprocessor:
    """
    Preprocessing class (**[pro]**, requires ``aaanalysis[pro]``) for per-residue
    post-translational modification (PTM) / functional-site annotations.

    Collects per-residue annotations — fetched from UniProt
    (:meth:`fetch_uniprot`) or ingested from a user / predictor table
    (:meth:`ingest`) — into one canonical ``df_annot`` schema, then encodes
    them into the ``[0, 1]``-normalized per-residue ``dict_num`` consumed by
    :meth:`CPP.run_num` (via :meth:`encode`). Annotations fall into two
    top-level categories: a closed UniProt ``'PTMs'`` vocabulary and an open
    ``'Functional sites'`` vocabulary that user keys extend
    (:meth:`register_feature`). A secondary scale-based path
    (:meth:`build_scales` / :meth:`build_cat`) feeds the amino acid (AA)-scale
    :meth:`CPP.run`, and :meth:`to_df_seq` exports annotations as
    :class:`AAWindowSampler` anchors.

    .. versionadded:: 1.1.0
    """

    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.

        Notes
        -----
        * This is the annotation-side member of the per-residue ``dict_num``
          family, alongside :class:`EmbeddingPreprocessor` (protein language
          model (PLM) embeddings) and :class:`StructurePreprocessor`
          (PDB / Define Secondary Structure of Proteins (DSSP) / AlphaFold).
          All three emit ``[0, 1]``-normalized tensors that
          :meth:`NumericalFeature.get_parts` slices into the per-part inputs of
          :meth:`CPP.run_num`, and that stack along the D axis via
          :func:`aaanalysis.combine_dict_nums`. The accompanying
          ``(df_scales, df_cat)`` pair names the D dimensions.
        * ``df_annot`` is the canonical per-residue schema with columns
          ``protein_id, start, end, aa, feature_type, category, source, evidence,
          score, bond_id`` (positions are 1-based, UniProt-canonical frame).
        * Encoder values are normalized to ``[0, 1]``; non-annotated in-coverage
          residues are ``0.0``; ``NaN`` marks genuinely unresolved positions.
        * Bond features (disulfide / cross-link) expand to two single-residue
          endpoints sharing a ``bond_id``; cleavage P1 anchors come from
          SIGNAL / PROPEP / TRANSIT span ends, not from the ``SITE`` grab-bag.
        * Two methods have no :class:`StructurePreprocessor` analog by design, not
          oversight: :meth:`register_feature` is the surface of the *open*
          ``'Functional sites'`` vocabulary (structure's registry is closed), and
          :meth:`to_df_seq` exports a seq-mode window-split because here an
          annotation *is* the window label (a structure feature never is).

        See Also
        --------
        * :class:`StructurePreprocessor`: the structure-side analog (PDB / DSSP / AlphaFold).
        * :class:`EmbeddingPreprocessor`: the PLM-embedding analog.
        * :func:`aaanalysis.combine_dict_nums`: stitch multiple dict_nums.
        * :meth:`CPP.run_num`: the downstream consumer.

        Examples
        --------
        .. include:: examples/ap_encode.rst
        """
        self._verbose = ut.check_verbose(verbose)
        # Per-instance registry copy so auto-registered Functional keys never
        # leak into the module-global built-ins or other instances.
        self._registry: Dict[str, Dict] = make_instance_registry()
        self._recipes: Dict[str, Callable] = dict(NORMALIZATION_RECIPES)

    # ------------------------------------------------------------------
    # fetch_uniprot
    # ------------------------------------------------------------------
    def fetch_uniprot(
        self,
        df_seq: pd.DataFrame = None,
        features: Optional[List[str]] = None,
        evidence: Literal["experimental", "manual", "all"] = "manual",
        timeout: float = 30.0,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch UniProt features for every entry and map to ``df_annot``.

        Queries the UniProt REST API for each protein accession in ``df_seq``
        and maps the returned post-translational modification (PTM) and site
        annotations into the canonical ``df_annot`` schema, ready to be passed
        to :meth:`encode`. Evidence can be filtered to retain only
        experimentally confirmed or manually curated entries.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers (UniProt accessions). The ``entry`` values are used as
            the UniProtKB accessions to fetch.
        features : list of str, optional
            Registry keys to keep (e.g. ``['phospho', 'disulfide']``).
            ``None`` keeps every built-in key.
        evidence : {'experimental', 'manual', 'all'}, default='manual'
            Evidence allow-set. ``'experimental'`` keeps only ECO:0000269;
            ``'manual'`` also keeps ECO:0007744 (combinatorial, manual);
            ``'all'`` disables evidence filtering. Raw ECO codes are retained
            in the ``evidence`` column regardless.
        timeout : float, default=30.0
            Per-request timeout in seconds.
        max_workers : int, optional
            Number of threads for concurrent fetches. ``None`` or ``1``
            (default) fetches entries sequentially. Greater than ``1`` fetches
            on a thread pool; rows are concatenated in input order and the
            ``df_annot`` is identical to the sequential result. Concurrency is
            opt-in because parallel requests to UniProt risk HTTP-429 throttling
            that can turn successful fetches into failures.

        Returns
        -------
        df_annot : pd.DataFrame
            Canonical per-residue annotation schema with columns
            ``protein_id, start, end, aa, feature_type, category, source,
            evidence, score, bond_id`` (positions are 1-based, UniProt-canonical
            frame).

        Raises
        ------
        ValueError
            On invalid arguments.
        RuntimeError
            On UniProt network / response failure.

        Examples
        --------
        .. include:: examples/ap_fetch_uniprot.rst
        """
        # Validate
        verbose = self._verbose
        ut.check_df_seq(df_seq=df_seq)
        ut.check_str_options(
            name="evidence", val=evidence, list_str_options=LIST_EVIDENCE_MODES
        )
        if features is not None:
            validate_feature_keys(features, registry=self._registry)
        ut.check_number_range(
            name="timeout", val=timeout, min_val=1, accept_none=False, just_int=False
        )
        if max_workers is not None:
            ut.check_number_range(name="max_workers", val=max_workers,
                                  min_val=1, just_int=True)
        # Fetch + map
        entries = df_seq[ut.COL_ENTRY].tolist()
        evidence_codes = _evidence_codes_for_mode(evidence)
        return fetch_and_map(
            entries=entries,
            allowed_features=features,
            evidence_codes=evidence_codes,
            timeout=timeout,
            verbose=verbose,
            max_workers=max_workers,
        )

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    def ingest(self, df_user: pd.DataFrame = None) -> pd.DataFrame:
        """Ingest a user / predictor annotation table into ``df_annot``.

        Every ingested ``feature_type`` is treated as a ``'Functional sites'``
        key; unknown keys auto-register (``num_dims=1``, identity normalization)
        unless previously registered via :meth:`register_feature`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_user : pd.DataFrame
            Must contain ``protein_id``, ``start`` (1-based position), and
            ``feature_type`` columns. Optional: ``end`` (defaults to ``start``),
            ``source`` (defaults to ``'user'``), ``score`` (defaults to ``1.0``,
            must lie in ``[0, 1]``), ``aa`` (expected residue; ``''`` disables
            the encode-time guard for that row).

        Returns
        -------
        df_annot : pd.DataFrame
            Canonical per-residue annotation schema, ``category='Functional
            sites'`` for every row.

        Raises
        ------
        ValueError
            On missing required columns or out-of-range scores.

        Examples
        --------
        .. include:: examples/ap_ingest.rst
        """
        # Validate
        if not isinstance(df_user, pd.DataFrame):
            raise ValueError(
                f"'df_user' ({type(df_user).__name__}) should be " f"a pandas DataFrame"
            )
        required = [ut.COL_PROTEIN_ID, ut.COL_START, ut.COL_FEATURE_TYPE]
        missing = [c for c in required if c not in df_user.columns]
        if missing:
            raise ValueError(
                f"'df_user' is missing required columns {missing}; expected at "
                f"least {required}"
            )
        if ut.COL_SCORE in df_user.columns:
            _check_scores_unit_range(df_user[ut.COL_SCORE])
        # Auto-register unknown feature_types
        for key in df_user[ut.COL_FEATURE_TYPE].astype(str).unique():
            if key not in self._registry:
                register_functional_key(self._registry, self._recipes, key=key)
        # Build canonical rows
        n = len(df_user)
        starts = df_user[ut.COL_START].astype(int)
        ends = (
            df_user[ut.COL_STOP].astype(int)
            if ut.COL_STOP in df_user.columns
            else starts
        )
        out = pd.DataFrame(
            {
                ut.COL_PROTEIN_ID: df_user[ut.COL_PROTEIN_ID].astype(str).tolist(),
                ut.COL_START: starts.tolist(),
                ut.COL_STOP: ends.tolist(),
                ut.COL_AA: (
                    df_user[ut.COL_AA].tolist()
                    if ut.COL_AA in df_user.columns
                    else [""] * n
                ),
                ut.COL_FEATURE_TYPE: df_user[ut.COL_FEATURE_TYPE].astype(str).tolist(),
                ut.COL_CAT: [ut.LIST_CAT[-1]] * n,  # 'Functional sites'
                ut.COL_SOURCE: (
                    df_user[ut.COL_SOURCE].tolist()
                    if ut.COL_SOURCE in df_user.columns
                    else ["user"] * n
                ),
                ut.COL_EVIDENCE: (
                    df_user[ut.COL_EVIDENCE].tolist()
                    if ut.COL_EVIDENCE in df_user.columns
                    else [""] * n
                ),
                ut.COL_SCORE: (
                    df_user[ut.COL_SCORE].tolist()
                    if ut.COL_SCORE in df_user.columns
                    else [1.0] * n
                ),
                ut.COL_BOND_ID: [None] * n,
            }
        )
        return out[ut.COLS_ANNOT]

    # ------------------------------------------------------------------
    # register_feature
    # ------------------------------------------------------------------
    def register_feature(
        self,
        key: str = None,
        subcategory: Optional[str] = None,
        normalization: Optional[Callable] = None,
    ) -> None:
        """Register (or override) an open-vocabulary Functional-sites key.

        Adds a new ``feature_type`` label to the per-instance registry so that
        :meth:`encode` knows how many output dimensions to allocate and which
        normalization recipe to apply. Unknown keys that arrive during
        :meth:`ingest` are auto-registered with default settings; call this
        method first to supply a custom ``subcategory`` or normalization function.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        key : str
            The feature key (the user/predictor ``feature_type``).
        subcategory : str, optional
            Fine-grained label; defaults to ``'FUNC_<key>'``.
        normalization : callable, optional
            Recipe applied to the raw per-residue values; defaults to a clip to
            ``[0, 1]`` (values must already lie in that range).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``key`` is not a non-empty string.

        Examples
        --------
        .. include:: examples/ap_register_feature.rst
        """
        ut.check_str(name="key", val=key, accept_none=False)
        register_functional_key(
            self._registry,
            self._recipes,
            key=key,
            subcategory=subcategory,
            normalization=normalization,
        )

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    def encode(
        self,
        df_seq: pd.DataFrame = None,
        df_annot: pd.DataFrame = None,
        features: List[str] = None,
        on_mismatch: Literal["raise", "drop", "warn"] = "raise",
        return_df: bool = False,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
        """Encode ``df_annot`` into a ``[0, 1]``-normalized per-residue ``dict_num``.

        Converts the canonical annotation table (from :meth:`fetch_uniprot` or
        :meth:`ingest`) into a ``{entry: (L, D) ndarray}`` tensor where each
        dimension corresponds to a registered ``feature_type`` and values are
        normalized to ``[0, 1]``. The result can be stacked with other
        per-residue tensors via :func:`aaanalysis.combine_dict_nums` and
        consumed directly by :meth:`CPP.run_num`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            The target coordinate frame; the residue-identity guard checks each
            annotated position against ``sequence``.
        df_annot : pd.DataFrame
            Canonical per-residue annotation schema (from :meth:`fetch_uniprot`
            or :meth:`ingest`).
        features : list of str
            Registry keys to encode, in the order they should occupy the D axis.
        on_mismatch : {'raise', 'drop', 'warn'}, default='raise'
            Behavior when ``df_seq[sequence][pos-1] != df_annot.aa`` for a row
            carrying a non-empty ``aa`` (an off-by-isoform / coordinate-frame
            error). ``'raise'`` aborts; ``'drop'`` silently skips the row;
            ``'warn'`` warns and skips.
        return_df : bool, default=False
            If ``True``, also return the per-row status DataFrame as a second
            element ``(dict_num, df_seq_out)``. If ``False`` (default), return
            only ``dict_num``.

        Returns
        -------
        dict_num : dict[str, np.ndarray]
            ``{entry: (L_entry, D) ndarray}`` where ``D == len(features)`` and
            ``L_entry == len(sequence)``. Stack with other per-residue tensors
            via :func:`aaanalysis.combine_dict_nums`.
        df_seq_out : pd.DataFrame
            Returned only when ``return_df=True``. Echo of ``df_seq`` plus a
            boolean ``encode_ok`` column — ``False`` for entries that had at
            least one position skipped due to a residue-identity mismatch
            under ``on_mismatch='drop'`` / ``'warn'`` (always ``True`` under
            ``'raise'``, which aborts instead).

        Raises
        ------
        ValueError
            On invalid arguments, missing schema columns, or (default) a
            residue-identity mismatch.

        Examples
        --------
        .. include:: examples/ap_encode.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for encode"
            )
        _check_df_annot(df_annot)
        validate_feature_keys(features, registry=self._registry)
        ut.check_str_options(
            name="on_mismatch", val=on_mismatch, list_str_options=LIST_ON_MISMATCH
        )
        ut.check_bool(name="return_df", val=return_df)
        # Layout: feature_key -> (col_start, num_dims)
        D = get_total_dims(features, registry=self._registry)
        layout: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for f in features:
            nd = self._registry[f]["num_dims"]
            layout[f] = (cursor, nd)
            cursor += nd
        feature_set = set(features)
        # Build per-entry tensors
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = dict(zip(entries, df_seq[ut.COL_SEQ].tolist()))
        grouped = {pid: sub for pid, sub in df_annot.groupby(ut.COL_PROTEIN_ID)}
        dict_num: Dict[str, np.ndarray] = {}
        mismatch_entries: set = set()
        for entry in entries:
            seq = sequences[entry]
            L = len(seq)
            arr = np.zeros((L, D), dtype=np.float64)
            sub = grouped.get(entry)
            if sub is not None:
                for row in sub.itertuples(index=False):
                    key = getattr(row, ut.COL_FEATURE_TYPE)
                    if key not in feature_set:
                        continue
                    col0, _nd = layout[key]
                    start = int(getattr(row, ut.COL_START))
                    end = int(getattr(row, ut.COL_STOP))
                    score = getattr(row, ut.COL_SCORE)
                    aa = getattr(row, ut.COL_AA, "")
                    single = start == end
                    for pos in range(start, end + 1):
                        if pos < 1 or pos > L:
                            continue  # out of coverage → leave default
                        if (
                            single
                            and isinstance(aa, str)
                            and aa != ""
                            and seq[pos - 1] != aa
                        ):
                            msg = (
                                f"'df_seq[sequence][{pos}]' "
                                f"('{seq[pos - 1]}') should be '{aa}' for "
                                f"entry '{entry}' feature '{key}' — "
                                f"UniProt-canonical position does not match "
                                f"the target sequence (likely isoform / "
                                f"coordinate-frame mismatch)"
                            )
                            if on_mismatch == "raise":
                                raise ValueError(msg)
                            if on_mismatch == "warn":
                                warnings.warn(msg, UserWarning, stacklevel=2)
                            mismatch_entries.add(entry)
                            continue  # drop / warn → skip this position
                        arr[pos - 1, col0] = score
            # Normalize each feature block in-place
            for f in features:
                col0, nd = layout[f]
                block = arr[:, col0 : col0 + nd]
                arr[:, col0 : col0 + nd] = normalize(f, block, recipes=self._recipes)
            dict_num[entry] = arr
        if return_df:
            df_seq_out = df_seq.copy()
            df_seq_out["encode_ok"] = [e not in mismatch_entries for e in entries]
            return dict_num, df_seq_out
        return dict_num

    # ------------------------------------------------------------------
    # build_scales
    # ------------------------------------------------------------------
    def build_scales(
        self,
        df_seq: pd.DataFrame = None,
        dict_num: Dict[str, np.ndarray] = None,
        features: List[str] = None,
        return_std: bool = False,
        dim_names_override: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Build ``df_scales`` by context-free per-amino acid (AA) averaging of the corpus.

        Mirrors :meth:`StructurePreprocessor.build_scales`: for each
        canonical amino acid and each D dimension, the pseudo-scale entry is the
        mean of the normalized per-residue values over occurrences of that AA.
        Required so :meth:`CPP.run_num`'s ``cor > max_cor`` redundancy gate is
        discriminative (an all-equal ``df_scales`` disables it).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            Used here as the source of empirical amino-acid contexts.
        dict_num : dict[str, np.ndarray]
            Per-residue tensors ``{entry: (L_entry, D) ndarray}`` from
            :meth:`encode` (or combined via :func:`aaanalysis.combine_dict_nums`).
        features : list of str
            Registry keys in the same order as the ``dict_num`` D-axis layout.
        return_std : bool, default=False
            If ``True``, also return per-AA standard deviations.
        dim_names_override : list of str, optional
            Replacement names for the D columns; length must equal ``D``.

        Returns
        -------
        df_scales : pd.DataFrame, shape (20, D)
            Rows are the 20 canonical AAs; columns are dim names; cells are
            per-AA means of normalized values (NaN where the AA is absent).
        df_stds : pd.DataFrame, shape (20, D)
            Per-AA standard deviations, only when ``return_std=True``.

        Raises
        ------
        ValueError
            On missing corpus, mismatched D, missing entries, or invalid keys.

        Warnings
        --------
        UserWarning
            Pseudo-scales depend on the content of ``df_seq`` + ``dict_num``.

        Examples
        --------
        .. include:: examples/ap_build_scales.rst
        """
        # Validate
        if df_seq is None or dict_num is None:
            raise ValueError(
                "'df_seq' / 'dict_num' (None) should both be provided. "
                "Pseudo-scales need a real corpus — fall back to build_cat() "
                "if you only want the (D, 5) metadata."
            )
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"build_scales"
            )
        validate_feature_keys(features, registry=self._registry)
        ut.check_bool(name="return_std", val=return_std)
        D = get_total_dims(features, registry=self._registry)
        dim_names = self._resolve_dim_names(
            features=features, D=D, override=dim_names_override
        )
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = dict(zip(entries, df_seq[ut.COL_SEQ].tolist()))
        missing = [e for e in entries if e not in dict_num]
        if missing:
            preview = missing[:5] + (["..."] if len(missing) > 5 else [])
            raise ValueError(
                f"'dict_num' ({len(missing)} missing entries) should contain "
                f"every entry in 'df_seq'. Missing: {preview}"
            )
        for entry in entries:
            arr = dict_num[entry]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(
                    f"'dict_num[{entry!r}]' should be a 2-D np.ndarray of "
                    f"shape (L, D)"
                )
            if arr.shape[0] != len(sequences[entry]):
                raise ValueError(
                    f"'dict_num[{entry!r}].shape[0]' ({arr.shape[0]}) should "
                    f"equal len(sequence) ({len(sequences[entry])})"
                )
            if arr.shape[1] != D:
                raise ValueError(
                    f"'dict_num[{entry!r}].shape[1]' ({arr.shape[1]}) should "
                    f"equal sum of num_dims across features ({D})"
                )
        warnings.warn(
            "Pseudo-scales are dataset-dependent (averaged over df_seq + "
            "dict_num). For reproducible cross-dataset comparison, compute them "
            "once on a fixed reference corpus and reuse the resulting "
            "df_scales.",
            UserWarning,
            stacklevel=2,
        )
        # Accumulate per-AA sums / squares / counts
        list_aa = list(ut.LIST_CANONICAL_AA)
        aa_to_idx = {a: i for i, a in enumerate(list_aa)}
        sums = np.zeros((len(list_aa), D), dtype=np.float64)
        sqs = np.zeros((len(list_aa), D), dtype=np.float64)
        counts = np.zeros((len(list_aa), D), dtype=np.float64)
        for entry in entries:
            arr = dict_num[entry]
            seq = sequences[entry]
            for i, a in enumerate(seq):
                if a not in aa_to_idx:
                    continue
                row = arr[i]
                mask = ~np.isnan(row)
                if not mask.any():
                    continue
                ai = aa_to_idx[a]
                sums[ai, mask] += row[mask]
                sqs[ai, mask] += row[mask] ** 2
                counts[ai, mask] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        df_scales = pd.DataFrame(means, index=list_aa, columns=dim_names)
        if not return_std:
            return df_scales
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sq = np.where(counts > 0, sqs / counts, np.nan)
            var = np.maximum(mean_sq - means**2, 0.0)
            stds = np.where(counts > 0, np.sqrt(var), np.nan)
        df_stds = pd.DataFrame(stds, index=list_aa, columns=dim_names)
        return df_scales, df_stds

    # ------------------------------------------------------------------
    # build_cat
    # ------------------------------------------------------------------
    def build_cat(
        self,
        features: List[str] = None,
        dim_names_override: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build the ``df_cat`` metadata frame for ``features`` (corpus-free).

        ``df_cat[category]`` is ``'PTMs'`` or ``'Functional sites'``;
        ``df_cat[subcategory]`` carries the per-key semantic split.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        features : list of str
            Registry keys, in the order they appear along the D axis.
        dim_names_override : list of str, optional
            Replacement names for the D columns.

        Returns
        -------
        df_cat : pd.DataFrame, shape (D, 5)
            One row per dimension: ``scale_id``, ``category``, ``subcategory``,
            ``scale_name``, ``scale_description``.

        Raises
        ------
        ValueError
            On invalid or unregistered feature keys in ``features``.

        Examples
        --------
        .. include:: examples/ap_build_cat.rst
        """
        validate_feature_keys(features, registry=self._registry)
        D = get_total_dims(features, registry=self._registry)
        dim_names = self._resolve_dim_names(
            features=features, D=D, override=dim_names_override
        )
        categories = get_categories(features, registry=self._registry)
        subcategories = get_subcategories(features, registry=self._registry)
        return pd.DataFrame(
            {
                ut.COL_SCALE_ID: dim_names,
                ut.COL_CAT: categories,
                ut.COL_SUBCAT: subcategories,
                ut.COL_SCALE_NAME: dim_names,
                ut.COL_SCALE_DES: [
                    f"{c}/{s}" for c, s in zip(categories, subcategories)
                ],
            }
        )

    # ------------------------------------------------------------------
    # to_df_seq
    # ------------------------------------------------------------------
    def to_df_seq(
        self,
        df_seq: pd.DataFrame = None,
        df_annot: pd.DataFrame = None,
        feature_type: str = None,
        match_residue_type: bool = True,
        exclude_other_annotations: bool = True,
        pos_col: str = None,
        aa_context_col: str = "aa_context",
    ) -> pd.DataFrame:
        """Project annotations onto ``df_seq`` for AAWindowSampler negative sampling.

        Builds a ``df_seq`` copy with (1) a positives column listing the 1-based
        positions annotated with ``feature_type`` (the test anchors) and (2) an
        ``aa_context`` per-residue eligibility mask where ``'1'`` marks an
        eligible reference anchor and ``'0'`` everything excluded, so
        ``AAWindowSampler`` can draw residue-type-matched references.

        Eligibility rules:

        * The ``feature_type`` positives are always excluded from the reference
          pool (they are the test anchors).
        * ``exclude_other_annotations=True`` (default) additionally excludes any
          residue carrying a *different* ``feature_type`` — keeps the reference
          set from being contaminated by, e.g., glyco-Ser when profiling
          phospho-Ser (which would inflate the score).
        * ``match_residue_type=True`` (default) restricts eligible anchors to the
          amino-acid types observed among the ``feature_type`` positives across
          ``df_annot`` (e.g. {S, T, Y} for phospho) — the residue-type-matched
          negative. Set ``False`` for residue-type-agnostic classes (e.g.
          predictor hotspots): the reference is then any non-annotated residue.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            The coordinate frame the annotation positions are projected onto.
        df_annot : pd.DataFrame
            Canonical per-residue annotation schema (from :meth:`fetch_uniprot`
            or :meth:`ingest`).
        feature_type : str
            The registry key whose annotated residues become positives.
        match_residue_type : bool, default=True
            Restrict eligible reference anchors to the residue types of the
            positives (residue-type-matched negative).
        exclude_other_annotations : bool, default=True
            Exclude residues carrying any other ``feature_type`` from the
            reference pool.
        pos_col : str, optional
            Name of the emitted positives column; defaults to ``ut.COL_POS``.
        aa_context_col : str, default='aa_context'
            Name of the emitted per-residue eligibility-mask column.

        Returns
        -------
        df_seq_out : pd.DataFrame
            A copy of ``df_seq`` with the ``pos_col`` (list[int]) and
            ``aa_context_col`` (str of '1'/'0', one char per residue) columns
            appended.

        Raises
        ------
        ValueError
            On invalid arguments, missing schema columns, or a target-column
            name collision with an existing ``df_seq`` column.

        Warnings
        --------
        UserWarning
            If no residue is annotated with ``feature_type`` in ``df_annot``.

        Notes
        -----
        Feed the result to ``AAWindowSampler`` with ``context_in='1'`` to keep
        only eligible reference anchors:

        .. code-block:: python

            df_ws = ap.to_df_seq(df_seq, df_annot, feature_type="phospho")
            df_ref = aa.AAWindowSampler().sample_same_protein(
                df_seq=df_ws, pos_col="pos", window_size=9,
                aa_context_col="aa_context", context_in="1",
                min_distance_to_pos=9)

        Examples
        --------
        .. include:: examples/ap_to_df_seq.rst
        """
        # Validate
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for to_df_seq"
            )
        _check_df_annot(df_annot)
        ut.check_str(name="feature_type", val=feature_type, accept_none=False)
        ut.check_bool(name="match_residue_type", val=match_residue_type)
        ut.check_bool(name="exclude_other_annotations", val=exclude_other_annotations)
        pos_col = ut.COL_POS if pos_col is None else pos_col
        ut.check_str(name="pos_col", val=pos_col, accept_none=False)
        ut.check_str(name="aa_context_col", val=aa_context_col, accept_none=False)
        for col in (pos_col, aa_context_col):
            if col in df_seq.columns:
                raise ValueError(
                    f"'df_seq' already has a '{col}' column; choose a different "
                    f"pos_col / aa_context_col or drop the existing column"
                )
        # Index annotations by entry; split chosen feature_type vs the rest
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = dict(zip(entries, df_seq[ut.COL_SEQ].tolist()))
        is_chosen = df_annot[ut.COL_FEATURE_TYPE].astype(str) == feature_type
        df_pos = df_annot[is_chosen]
        df_other = df_annot[~is_chosen]
        if len(df_pos) == 0:
            warnings.warn(
                f"'feature_type' ('{feature_type}') matches no rows in "
                f"'df_annot'; every '{pos_col}' will be empty.",
                UserWarning,
                stacklevel=2,
            )
        pos_by_entry = {
            e: sorted({int(p) for p in sub[ut.COL_START]})
            for e, sub in df_pos.groupby(ut.COL_PROTEIN_ID)
        }
        other_by_entry = {
            e: {int(p) for p in sub[ut.COL_START]}
            for e, sub in df_other.groupby(ut.COL_PROTEIN_ID)
        }
        # Global residue-type set of the positives (from the target sequences)
        residue_types = set()
        if match_residue_type:
            for e, plist in pos_by_entry.items():
                seq = sequences.get(e, "")
                for p in plist:
                    if 1 <= p <= len(seq):
                        residue_types.add(seq[p - 1])
        # Build per-entry positives list + eligibility mask
        pos_cells, ctx_cells = [], []
        for entry in entries:
            seq = sequences[entry]
            pos_set = set(pos_by_entry.get(entry, []))
            other_set = other_by_entry.get(entry, set())
            tags = []
            for i in range(1, len(seq) + 1):
                if i in pos_set:
                    eligible = False
                elif exclude_other_annotations and i in other_set:
                    eligible = False
                elif match_residue_type and seq[i - 1] not in residue_types:
                    eligible = False
                else:
                    eligible = True
                tags.append(CONTEXT_TAG_ELIGIBLE if eligible else CONTEXT_TAG_EXCLUDED)
            pos_cells.append(sorted(pos_set))
            ctx_cells.append("".join(tags))
        df_out = df_seq.copy()
        df_out[pos_col] = pos_cells
        df_out[aa_context_col] = ctx_cells
        return df_out

    def _resolve_dim_names(self, features, D, override):
        """Validate ``dim_names_override`` against D; fall back to registry."""
        if override is None:
            return get_dim_names(features, registry=self._registry)
        if not isinstance(override, list):
            raise ValueError(
                f"'dim_names_override' ({type(override).__name__}) should be "
                f"a list of {D} str"
            )
        if len(override) != D:
            raise ValueError(
                f"'dim_names_override' (len={len(override)}) should be a list "
                f"of {D} str (one per output dim)"
            )
        for n in override:
            if not isinstance(n, str):
                raise ValueError(f"'dim_names_override' items ({n!r}) should be str")
        return list(override)
