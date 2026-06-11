"""
This is a script for the frontend of the NumericalFeature class, a supportive class for the CPP feature engineering,
including scale and feature filtering methods.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict, Union, List, Tuple, Type

import aaanalysis.utils as ut

from ._backend.check_feature import check_df_scales, expand_pos_anchors_
from ._backend.feature_filter import filter_correlation_
from ._backend.num_feat.extend_alphabet import extend_alphabet_
from ._backend.cpp._filters._assign import assign_dict_num_to_parts


# I Helper Functions
def check_match_df_scales_letter_new(df_scales=None, letter_new=None):
    """Check if new letter not already in df_scales"""
    alphabet = df_scales.index.to_list()
    if letter_new in alphabet:
        raise ValueError(f"Letter '{letter_new}' already exists in alphabet of 'df_scales': {alphabet}")


def check_match_df_seq_dict_num(df_seq=None, dict_num=None):
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


# II Main Functions
class NumericalFeature:
    """
    Utility feature engineering class to process and filter numerical data structures,
    such as amino acid scales or a feature matrix.

    It provides numeric helpers for the :class:`CPP` feature engineering pipeline:
    extending the amino acid alphabet of a scale DataFrame, slicing per-residue tensors
    into sequence parts (numerical analog of :meth:`SequenceFeature.get_df_parts`),
    and removing redundant features by Pearson correlation.

    .. versionadded:: 0.1.3
    """

    @staticmethod
    def filter_correlation(X: ut.ArrayLike2D,
                           max_cor: float = 0.7
                           ) -> np.array:
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
    def get_parts(df_seq: pd.DataFrame = None,
                  dict_num: Dict[str, np.ndarray] = None,
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
    def extend_alphabet(df_scales: pd.DataFrame = None,
                        new_letter: str = None,
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
