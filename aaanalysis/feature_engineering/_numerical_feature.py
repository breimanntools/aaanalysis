"""
This is a script for the frontend of the NumericalFeature class, a supportive class for the CPP feature engineering,
including scale and feature filtering methods.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict, Union, List, Tuple, Type

import aaanalysis.utils as ut

from ._backend.check_feature import (check_df_scales, expand_pos_anchors_,
                                     check_match_df_scales_features)
from ._backend.feature_filter import filter_correlation_
from ._backend.num_feat.extend_alphabet import extend_alphabet_
from ._backend.num_feat.feature_matrix import get_feature_matrix_num_
from ._backend.cpp._filters._assign import assign_dict_num_to_parts
from ._cpp import _derive_dict_part_lens


# I Helper Functions
def check_match_df_scales_letter_new(df_scales=None, letter_new=None) -> None:
    """Check if new letter not already in df_scales"""
    alphabet = df_scales.index.to_list()
    if letter_new in alphabet:
        raise ValueError(f"Letter '{letter_new}' already exists in alphabet of 'df_scales': {alphabet}")


def check_match_df_seq_dict_num(df_seq=None, dict_num=None) -> None:
    """Validate that ``dict_num`` matches ``df_seq`` (Layer-1 contract for ``get_parts``).

    Checks (per entry):
    * Every ``entry`` in ``df_seq`` is a key in ``dict_num``.
    * ``dict_num[entry].shape[0] == len(df_seq.loc[entry, 'sequence'])``.
    * ``D = dict_num[entry].shape[1]`` is consistent across all entries.

    Raises ``ValueError`` at the first failure with the offending entry name
    and the expected vs. got shape.
    """
    entries = df_seq[ut.COL_ENTRY].to_list()
    seqs = df_seq[ut.COL_SEQ].to_list()
    missing = [e for e in entries if e not in dict_num]
    if missing:
        raise ValueError(
            f"'dict_num' is missing {len(missing)} entries present in 'df_seq': {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    D_seen = set()
    for i, entry in enumerate(entries):
        arr = dict_num[entry]
        if not hasattr(arr, "shape") or arr.ndim != 2:
            raise ValueError(
                f"'dict_num[{entry!r}]' should be a 2-D ndarray (L, D); "
                f"got ndim={getattr(arr, 'ndim', None)}"
            )
        L_expected = len(seqs[i])
        if arr.shape[0] != L_expected:
            raise ValueError(
                f"'dict_num[{entry!r}]' shape ({arr.shape[0]}, {arr.shape[1]}) "
                f"should be ({L_expected}, D) — len(sequence)={L_expected}"
            )
        D_seen.add(arr.shape[1])
    if len(D_seen) > 1:
        raise ValueError(
            f"'dict_num' has inconsistent D across entries: {sorted(D_seen)} — "
            f"all entries must share the same dimensionality."
        )
    if D_seen and 0 in D_seen:
        raise ValueError("'dict_num[*]' has D=0; should be >= 1.")


def check_dict_num_parts(dict_num_parts=None) -> Tuple[int, int]:
    """Validate ``dict_num_parts`` (the value source for ``feature_matrix``) and return ``(n_samples, D)``.

    Each value must be a 3-D ``(n_samples, L_part_max, D)`` ndarray, with a
    consistent ``n_samples`` (axis 0) and ``D`` (axis 2) across all parts — the
    contract emitted by :meth:`NumericalFeature.get_parts`.
    """
    ut.check_dict(name="dict_num_parts", val=dict_num_parts, accept_none=False)
    if len(dict_num_parts) == 0:
        raise ValueError("'dict_num_parts' should not be empty; it must map each part name "
                         "to a (n_samples, L_part_max, D) tensor (see NumericalFeature.get_parts).")
    n_seen, d_seen = set(), set()
    for part, arr in dict_num_parts.items():
        if not hasattr(arr, "shape") or getattr(arr, "ndim", None) != 3:
            raise ValueError(
                f"'dict_num_parts[{part!r}]' should be a 3-D ndarray (n_samples, L_part_max, D); "
                f"got ndim={getattr(arr, 'ndim', None)}."
            )
        n_seen.add(arr.shape[0])
        d_seen.add(arr.shape[2])
    if len(n_seen) > 1:
        raise ValueError(f"'dict_num_parts' has inconsistent n_samples (axis 0) across parts: "
                         f"{sorted(n_seen)} — all parts must share the same number of samples.")
    if len(d_seen) > 1:
        raise ValueError(f"'dict_num_parts' has inconsistent D (axis 2) across parts: "
                         f"{sorted(d_seen)} — all parts must share the same dimensionality.")
    if d_seen and 0 in d_seen:
        raise ValueError("'dict_num_parts[*]' has D=0; should be >= 1.")
    return n_seen.pop(), d_seen.pop()


# II Main Functions
class NumericalFeature:
    """
    Utility feature engineering class to process and filter numerical data structures,
    such as amino acid scales or a feature matrix.

    It provides numeric helpers for the :class:`CPP` feature engineering pipeline:
    extending the amino acid alphabet of a scale DataFrame, slicing per-residue tensors
    into sequence parts (numerical analog of :meth:`SequenceFeature.get_df_parts`),
    reconstructing the model matrix ``X`` from :meth:`CPP.run_num`-selected features
    (:meth:`feature_matrix`), and removing redundant features by Pearson correlation.

    .. versionadded:: 0.1.3
    """

    @staticmethod
    def filter_correlation(X: ut.ArrayLike2D,
                           max_cor: float = 0.7
                           ) -> np.ndarray:
        """
        Filter features based on Pearson correlation.

        Removes redundant columns from a feature matrix by iterating through features
        in order and discarding any feature that exceeds ``max_cor`` Pearson correlation
        with an already-kept feature. Use this after :meth:`CPP.run` to reduce the
        selected feature set to a non-redundant subset.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        max_cor : float, default=0.7
            Maximum Pearson correlation [0-1] of feature scales used as threshold for filtering.

        Returns
        -------
        is_selected
            1D boolean array with shape (n_features) indicating which features are selected (True) or not (False).

        Notes
        -----
        * Features in ``X`` should be provided in decreasing order of importance. The first occurring features
          will be kept, while subsequent features that correlate with them will be removed.
        * The number of selected features (``True`` entries) can be **fewer** than ``n_features`` when redundant
          columns are removed; raise ``max_cor`` to retain more (admitting more redundancy).
        Examples
        --------
        .. include:: examples/nf_filter_correlation.rst
        """
        # Check input
        X = ut.check_X(X=X, min_n_unique_features=2)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0, max_val=1, just_int=False, accept_none=False)
        # Filter features
        is_selected = filter_correlation_(X, max_cor=max_cor)
        return is_selected

    @staticmethod
    def get_parts(df_seq: pd.DataFrame,
                  dict_num: Dict[str, np.ndarray],
                  list_parts: Optional[List[str]] = None,
                  all_parts: bool = False,
                  jmd_n_len: int = 10,
                  jmd_c_len: int = 10,
                  tmd_len: Optional[int] = None,
                  ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Prepare Comparative Physicochemical Profiling (CPP) numerical-mode inputs by slicing
        sequences AND per-residue tensors with shared boundaries.

        Numerical analog of :meth:`SequenceFeature.get_df_parts` for the
        ``CPP.run_num`` workflow: the same `(start, end)` boundaries used to
        slice the sequence STRINGS into parts are reused to slice each entry's
        ``dict_num[entry]`` per-residue tensor along the L axis. Returns both
        results from one call so the user never has to pass
        ``df_seq + tmd_len + jmd_n_len + jmd_c_len`` to two separate helpers.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers
            and a ``sequence`` column with full protein sequences. Must also carry
            ``tmd_start`` / ``tmd_stop`` columns (the position-based schema) so the
            slicing boundaries can be computed.
        dict_num : dict[str, np.ndarray]
            Mapping ``entry -> (L, D)`` per-residue numerical tensor, where ``L``
            matches ``len(df_seq.loc[entry, 'sequence'])`` and ``D`` is consistent
            across all entries. Source: protein language model (PLM) embeddings, DSSP one-hots,
            post-translational modification (PTM) dummies, or any other per-residue numerical
            representation.
        list_parts : list of str, optional
            Subset of part names to materialize (e.g. ``["tmd", "jmd_n_tmd_n",
            "tmd_c_jmd_c"]``). Defaults to the same default as
            :meth:`SequenceFeature.get_df_parts`.
        all_parts : bool, default=False
            If ``True``, return all available parts; ignored when ``list_parts``
            is supplied.
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        tmd_len : int, optional
            Target middle domain (TMD) length for the **anchor-based format** only (a ``sequence`` +
            ``pos`` ``df_seq``): each 1-based anchor in ``pos`` is exploded into one row with the
            TMD centered (right-heavy for even ``tmd_len``) on the anchor, and the matching
            ``dict_num`` tensor is sliced with the same boundaries. Ignored for the position-based
            schema.

        Returns
        -------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            Per-part sequence STRINGS, same shape as
            :meth:`SequenceFeature.get_df_parts`'s output. Pass directly to
            ``CPP(df_parts=df_parts, ...)``.
        dict_num_parts : dict[str, np.ndarray]
            Per-part NaN-padded numerical tensors. Each value has shape
            ``(n_samples, L_part_max, D)`` aligned row-for-row with the
            ``df_parts`` index. Pass directly to
            ``cpp.run_num(dict_num_parts=dict_num_parts, ...)``.

        Raises
        ------
        ValueError
            If ``dict_num`` is missing an entry from ``df_seq``, any
            ``dict_num[entry]`` has the wrong row count vs the sequence length,
            or D varies across entries.

        Notes
        -----
        * **Call order — ``get_parts`` then ``run_num``.** This is step 1 of the
          two-step numerical-mode workflow: it only *slices* ``df_seq`` + ``dict_num``
          into ``(df_parts, dict_num_parts)``. Pass ``df_parts`` to the :class:`CPP`
          constructor and ``dict_num_parts`` to :meth:`CPP.run_num` (step 2);
          ``run_num`` has no raw-``df_seq`` / ``dict_num`` entry point, so this order
          is the only supported one.
        * **``dict_num`` must already be ``[0, 1]``-normalized.** ``get_parts`` slices
          values verbatim — it does NOT rescale. The ``max_std_test`` pre-filter in
          :meth:`CPP.run_num` is calibrated for the ``[0, 1]`` range; passing unbounded
          values (e.g. raw protein language model embeddings) leaves that pre-filter
          miscalibrated, so the feature funnel silently keeps/drops the wrong features
          (no error is raised). Normalize first — e.g. via
          :meth:`EmbeddingPreprocessor.encode`, :class:`StructurePreprocessor`, or
          :class:`AnnotationPreprocessor`, all of which emit ``[0, 1]`` ``dict_num``.
        * ``dict_num_parts`` carries NaN padding at the trailing rows for entries
          whose JMD doesn't fit the requested length. The corresponding per-part
          string in ``df_parts`` also pads with ``'-'`` (gap), so the two outputs
          stay aligned. The real per-(entry, part) length is recoverable as the
          non-gap character count in ``df_parts``.
        * For seq-only mode (no per-residue tensor), use
          :meth:`SequenceFeature.get_df_parts` directly; ``NumericalFeature.get_parts``
          is only useful when you have a ``dict_num`` to slice.

        See Also
        --------
        * :meth:`SequenceFeature.get_df_parts` — string-only analog.
        * :meth:`CPP.run_num` — consumes ``df_parts`` (via constructor) and
          ``dict_num_parts`` (per call).

        Examples
        --------
        .. include:: examples/nf_get_parts.rst
        """
        # Check input
        ut.check_df_seq(df_seq=df_seq, accept_none=False)
        if dict_num is None:
            raise ValueError(
                "'dict_num' (None) should be a Dict[str, np.ndarray] mapping each "
                "entry in df_seq to a (L, D) per-residue tensor."
            )
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        ut.check_bool(name="all_parts", val=all_parts)
        # Layer-1: df_seq ↔ dict_num pairing (on the original, one-row-per-entry df_seq).
        check_match_df_seq_dict_num(df_seq=df_seq, dict_num=dict_num)

        # Anchor-based format ('sequence' + 'pos' + tmd_len): explode to position-based
        # so the SAME exploded boundaries drive both string parts AND tensor slices (D5a).
        anchor_mode = (isinstance(df_seq, pd.DataFrame)
                       and ut.COL_SEQ in df_seq.columns and ut.COL_POS in df_seq.columns
                       and not set(ut.COLS_SEQ_POS).issubset(df_seq.columns)
                       and not set(ut.COLS_SEQ_PARTS).issubset(df_seq.columns)
                       and not set(ut.COLS_SEQ_TMD).issubset(df_seq.columns))
        list_entry_win = None
        if anchor_mode:
            ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
            df_seq, list_entry_win = expand_pos_anchors_(df_seq=df_seq, tmd_len=tmd_len,
                                                         jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)

        # Build df_parts via SequenceFeature (shared with seq-mode users). df_seq is now
        # position-based, so get_df_parts does not re-explode.
        from ._sequence_feature import SequenceFeature
        sf = SequenceFeature(verbose=False)
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts,
                                   all_parts=all_parts,
                                   jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        list_parts_resolved = list(df_parts.columns)

        # Slice the per-residue tensors with the SAME boundaries (backend reused). With the
        # exploded df_seq, repeated entries reuse one tensor sliced at per-row boundaries.
        dict_part_vals, _dict_part_lens = assign_dict_num_to_parts(
            df_seq=df_seq, dict_num=dict_num, list_parts=list_parts_resolved,
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
        )
        if anchor_mode:
            df_parts.index = list_entry_win
        # lens are recoverable from df_parts (non-gap char count) in run_num —
        # no need to expose them as a separate output (keeps the API one-shape).
        return df_parts, dict_part_vals

    @staticmethod
    def feature_matrix(features: Union[ut.ArrayLike1D, pd.DataFrame],
                       dict_num_parts: Dict[str, np.ndarray],
                       df_parts: pd.DataFrame,
                       df_scales: Optional[pd.DataFrame] = None,
                       n_jobs: Union[int, None] = 1,
                       ) -> np.ndarray:
        """
        Create the numerical-mode feature matrix ``X`` for given feature ids and per-part tensors.

        Numerical analog of :meth:`SequenceFeature.feature_matrix`: for each sample and each
        feature id, reconstructs the feature value from the pre-sliced per-residue tensors in
        ``dict_num_parts`` (PLM embeddings, structure, annotations, ...) instead of an
        amino-acid-to-scale lookup, so per-residue context is preserved. The result is the
        numerical input ``X`` for a downstream model or for
        :meth:`NumericalFeature.filter_correlation`. This is the missing step that turns
        :meth:`CPP.run_num`-selected features back into a model matrix.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            Ids of features (``'PART-SPLIT-SCALE'``) for which the matrix of feature values should
            be created. Alternatively, a ``df_feat`` DataFrame (e.g. from :meth:`CPP.run_num`), in
            which case its ``'feature'`` column is used.
        dict_num_parts : dict[str, np.ndarray]
            Per-part NaN-padded numerical tensors, as produced by :meth:`NumericalFeature.get_parts`
            and consumed by :meth:`CPP.run_num`. Each value has shape ``(n_samples, L_part_max, D)``,
            row-aligned across parts; must cover every part referenced in ``features``.
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            The string ``df_parts`` returned **alongside** ``dict_num_parts`` by
            :meth:`NumericalFeature.get_parts` (row-aligned with it). Supplies each part's real
            residue length via the exact same helper :meth:`CPP.run_num` uses internally (non-gap
            character count), so every split lands on the residues ``run_num`` selected. Its columns
            must cover every part in ``dict_num_parts``.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame whose columns **name the D dimensions** of ``dict_num_parts`` (the same
            ``df_scales`` used to construct the :class:`CPP` for :meth:`CPP.run_num`); its column
            order defines the ``SCALE`` -> D-index mapping. Its row (amino acid) values are unused in
            numerical mode. Default from :meth:`load_scales` unless specified in
            ``options['df_scales']`` — pass your own when the D axis is a custom (e.g. embedding)
            space, since the default AA-scale set will not match a custom ``D``.
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized
            automatically. If ``-1``, the number is set to all available cores. Overridden by
            ``options['n_jobs']`` when set.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Feature matrix containing feature values for samples. Column ``i`` corresponds to
            feature ``i`` in ``features``; values are byte-identical to those :meth:`CPP.run_num`
            computed for the same feature ids — both take per-part lengths from the same ``df_parts``,
            so they select identical residues for every split.

        Raises
        ------
        ValueError
            If ``dict_num_parts`` is empty or has inconsistent / non-3D tensors, if ``df_parts`` is
            missing a part column or its row count / real lengths do not match ``dict_num_parts``, if
            the ``D`` axis does not match ``len(df_scales.columns)``, if a feature id is malformed or
            references a part/scale absent from ``dict_num_parts`` / ``df_scales``, or if a feature's
            split selects only padded (all-NaN) residues (producing a ``NaN`` value).

        Notes
        -----
        * **Mapping ``positions`` back to ``dict_num_parts`` — do not index with it.** The
          ``'positions'`` column of a :meth:`CPP.run_num` ``df_feat`` is a **display numbering** in
          TMD-JMD coordinate space: positions are offset by the JMD length and use the ``start`` /
          ``tmd_len`` / ``jmd_n_len`` / ``jmd_c_len`` shown at run time (e.g. a ``jmd_n_len=10`` TMD
          is numbered ``21..30``, decoupled from the actual per-sample TMD length). These numbers do
          **not** directly index the ``(L, D)`` per-part array. ``feature_matrix`` therefore does not
          read ``positions``; it reconstructs each value the same way :meth:`CPP.run_num` does — by
          re-applying the ``SPLIT`` encoded in the feature id to the part's **0-based residue axis**
          (``arange(L_part)``, where ``L_part`` is the sample's real residue count taken from
          ``df_parts``), selecting the ``SCALE`` column along ``D``, and averaging the selected
          residues (``nanmean``, rounded to 5 decimals). Because the split is applied to the residue
          axis and not to the display numbering, ``X`` lines up with ``run_num`` by construction.
        * **Real per-part lengths come from ``df_parts``**, via the very same length rule
          (non-gap character count of ``df_parts``) that :meth:`CPP.run_num` applies
          internally — not inferred from the tensor's NaN padding. Passing the ``df_parts`` that
          :meth:`NumericalFeature.get_parts` returns alongside ``dict_num_parts`` therefore makes
          ``X`` identical to ``run_num`` in every case, including when a *real* residue is all-NaN
          across ``D`` (an unresolved structure position or a masked embedding): its length is still
          counted from the string, exactly as ``run_num`` counts it.
        * A feature whose split selects only padded (all-NaN) residues yields a ``NaN`` value; this
          raises a ``ValueError`` rather than returning a silently-``NaN`` column.
        * Unlike :meth:`SequenceFeature.feature_matrix` there is no ``accept_gaps`` option: a numeric
          tensor carries no gap characters, and NaN padding is ignored via ``nanmean`` automatically.

        See Also
        --------
        * :meth:`SequenceFeature.feature_matrix`: sequence-mode (per-AA scale) equivalent.
        * :meth:`NumericalFeature.get_parts`: produces the ``dict_num_parts`` consumed here.
        * :meth:`CPP.run_num`: selects the feature ids whose values this method reconstructs.

        Examples
        --------
        .. include:: examples/nf_feature_matrix.rst
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        # Check input
        _n_samples, D = check_dict_num_parts(dict_num_parts=dict_num_parts)
        ut.check_df(name="df_parts", df=df_parts, accept_none=False,
                    cols_required=list(dict_num_parts.keys()))
        if len(df_parts) != _n_samples:
            raise ValueError(
                f"'df_parts' has {len(df_parts)} rows but 'dict_num_parts' has {_n_samples} samples. "
                f"Pass the 'df_parts' returned alongside 'dict_num_parts' by NumericalFeature.get_parts."
            )
        check_df_scales(df_scales=df_scales)
        n_scales = len(df_scales.columns)
        if D != n_scales:
            raise ValueError(
                f"'dict_num_parts' D={D} should equal len(df_scales.columns)={n_scales}. "
                f"'df_scales' names the D dimensions in numerical mode — pass the same 'df_scales' "
                f"used to construct the CPP for 'run_num'."
            )
        features = ut.check_features(features=features, list_parts=list(dict_num_parts.keys()),
                                     list_scales=list(df_scales.columns))
        check_match_df_scales_features(df_scales=df_scales, features=features)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # Real per-part lengths from the SAME source CPP.run_num uses (the string 'df_parts',
        # non-gap character count) rather than the tensor's NaN padding, so each split lands on
        # exactly the residues run_num selected — identical X even when a real residue is all-NaN.
        part_lens = _derive_dict_part_lens(df_parts=df_parts)
        for part, arr in dict_num_parts.items():
            if part_lens[part].size and int(part_lens[part].max()) > arr.shape[1]:
                raise ValueError(
                    f"'df_parts[{part!r}]' has a real residue length exceeding the padded tensor "
                    f"length L_part_max={arr.shape[1]} in 'dict_num_parts'. Pass 'df_parts' and "
                    f"'dict_num_parts' from the same NumericalFeature.get_parts call."
                )
        # Build the feature matrix (byte-identical to CPP.run_num's value reconstruction)
        try:
            X = get_feature_matrix_num_(features=features, dict_num_parts=dict_num_parts,
                                        part_lens=part_lens, df_scales=df_scales, n_jobs=n_jobs)
        except IndexError as error:
            raise ValueError(
                "A feature's 'SPLIT' references a residue position beyond a part's length in "
                "'dict_num_parts'. Ensure 'features' were generated for these parts (e.g. via "
                "CPP.run_num on the same 'dict_num_parts')."
            ) from error
        if not np.isfinite(X).all():
            raise ValueError(
                "'feature_matrix' produced NaN feature values: at least one feature's split selects "
                "only padded (all-NaN) residues in 'dict_num_parts'. Drop that feature or widen the "
                "part/split so it covers real residues."
            )
        return X

    @staticmethod
    def extend_alphabet(df_scales: pd.DataFrame,
                        new_letter: str,
                        value_type: Literal["min", "mean", "median", "max"] = "mean",
                        ) -> pd.DataFrame:
        """
        Extend amino acid alphabet of ``df_scales`` by new letter.

        This function adds a new row to the DataFrame, representing the new amino acid letter.
        For each scale (column), it computes a specific statistic (min, mean, median, max) based on the
        values of existing amino acids (rows) and assigns this computed value to the new amino acid.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        df_scales : pd.DataFrame, shape (n_letters, n_scales)
            DataFrame of scales with letters typically representing amino acids.
        new_letter : str
            The new letter to be added to the alphabet.
        value_type : {'min', 'mean', 'median', 'max'}, default='mean'
            The type of statistic to compute for the new letter.

        Returns
        -------
        df_scales : pd.DataFrame, shape (n_letters + 1, n_scales)
            DataFrame with the extended alphabet including the new amino acid letter.

        Notes
        -----
        * If ``new_letter`` is already present in the index of ``df_scales``, a ``ValueError`` is raised.
          Use this method only to add letters that do not yet appear in the alphabet.

        Examples
        --------
        .. include:: examples/nf_extend_alphabet.rst
        """
        # Check input
        df_scales = df_scales.copy()
        check_df_scales(df_scales=df_scales)
        ut.check_str(name="letter_new", val=new_letter)
        ut.check_str_options(name="value_type", val=value_type, accept_none=False,
                             list_str_options=["min", "mean", "median", "max"])
        check_match_df_scales_letter_new(df_scales=df_scales, letter_new=new_letter)
        # Compute the statistic for each scale
        df_scales = extend_alphabet_(df_scales=df_scales, new_letter=new_letter, value_type=value_type)
        return df_scales
