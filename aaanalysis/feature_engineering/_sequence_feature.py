"""
This is a script for the frontend of the SequenceFeature class, a supportive class for the CPP feature engineering.
"""
from typing import Literal, Optional, Union, List, Dict, Tuple, Sequence
import inspect
import warnings
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.check_feature import (check_split_kws,
                                     check_parts_len,
                                     check_match_features_seq_parts,
                                     check_match_df_seq_jmd_len,
                                     expand_pos_anchors_,
                                     check_match_df_parts_features,
                                     check_match_df_parts_df_scales,
                                     check_df_scales,
                                     check_match_df_scales_features,
                                     check_match_df_scales_df_cat,
                                     check_df_cat,
                                     check_match_df_cat_features)
from ._backend.cpp.utils_feature import (get_df_parts_,
                                         remove_entries_with_gaps_,
                                         replace_non_canonical_aa_,
                                         get_positions_, get_amino_acids_,
                                         get_df_pos_, get_df_pos_parts_)
from ._backend.cpp.sequence_feature import (get_split_kws_, get_features_, get_feature_names_,
                                            get_feature_descriptions_, get_df_feat_,
                                            get_labels_ovr_, get_labels_ovo_, get_labels_quantile_,
                                            get_labels_tiered_)
from ._backend.cpp_run import _pick_feature_matrix_builder
from ._backend.feature_filter import filter_correlation_, filter_variance_


# I Helper Functions
def _valid_df_parts_kws_keys():
    """Return the ``SequenceFeature.get_df_parts`` parameter names valid in a ``df_parts_kws`` dict."""
    params = inspect.signature(SequenceFeature.get_df_parts).parameters
    return {p for p in params if p not in ("self", "df_seq")}


def check_df_parts_kws(df_parts_kws=None, df_seq=None) -> None:
    """Check ``df_parts_kws`` is a valid-key dict of ``get_df_parts`` kwargs, only used with ``df_seq``."""
    if df_parts_kws is None:
        return
    ut.check_dict(name="df_parts_kws", val=df_parts_kws, accept_none=True)
    if df_seq is None:
        raise ValueError("'df_parts_kws' is only used together with 'df_seq'; with 'df_parts' the parts "
                         "are taken directly from its columns.")
    valid_keys = _valid_df_parts_kws_keys()
    invalid_keys = set(df_parts_kws) - valid_keys
    if invalid_keys:
        raise ValueError(f"'df_parts_kws' should only contain 'get_df_parts' parameter names. Invalid keys: "
                         f"{sorted(invalid_keys)}. Valid keys: {sorted(valid_keys)}.")
def check_split_types(split_types=None):
    """Check if split types valid (Segment, Pattern, or PeriodicPattern)"""
    split_types = ut.check_list_like(name="split_types", val=split_types, accept_str=True, accept_none=True)
    if split_types is None:
        split_types = ut.LIST_SPLIT_TYPES
    wrong_split_type = [x for x in split_types if x not in ut.LIST_SPLIT_TYPES]
    if len(wrong_split_type) > 0:
        raise ValueError(f"Wrong 'split_types' ({wrong_split_type}). Chose from {ut.LIST_SPLIT_TYPES}")
    return split_types


def check_steps(steps=None, steps_name="steps_pattern", len_min=2, fixed_len=False):
    """Sort steps and warn if empty list"""
    if steps is None:
        return steps # Skip tests
    steps = list(sorted(steps))
    if len(steps) < len_min:
        if fixed_len:
            raise ValueError(f"'{steps_name}' ({steps}) should contain exactly {len_min} non-negative integers.")
        else:
            raise ValueError(f"'{steps_name}' ({steps}) should contain >= {len_min} non-negative integers.")
    return steps


def warn_creation_of_feature_matrix(features=None, df_parts=None, name="Feature matrix") -> None:
    """Warn if feature matrix gets too large"""
    n_feat = len(features)
    n_samples = len(df_parts)
    n_vals = n_feat * n_samples
    ut.print_out(f"'{name}' for {n_feat} features and {n_samples} samples will be created.")
    if n_vals > 1000 * 1000:
        warning = f"Feature matrix with n={n_vals}>=10^6 values will be created, which will take some time.\n" \
                  "It is recommended to create a feature matrix for a pre-selected number features " \
                  "so that 10^6 values are not exceeded."
        warnings.warn(warning)


def check_match_labels_label_test_label_ref(labels=None, label_test=1, label_ref=0) -> None:
    """Check if labels only contains label_test and label_ref"""
    wrong_labels = [x for x in labels if x not in [label_ref, label_test]]
    unique_wrong_labels = list(set(wrong_labels))
    n_wrong_labels = len(unique_wrong_labels)
    if n_wrong_labels > 0:
        raise ValueError(f"'labels' contains {n_wrong_labels} wrong labels: {unique_wrong_labels}")


def check_match_df_parts_label_test_label_ref(df_parts=None, labels=None, label_test=1, label_ref=0) -> None:
    """Check if 'jmd_n', 'tmd', and 'jmd_c' in df_parts if amino acid for label_test or label_ref should be retrieved"""
    list_parts = list(df_parts)
    required_parts = ["jmd_n", "tmd", "jmd_c"]
    mask_test = [x == label_test for x in labels]
    mask_ref = [x == label_ref for x in labels]
    if sum(mask_test) == 1:
        missing_parts = [x for x in required_parts if x not in list_parts]
        if len(missing_parts) > 0:
            raise ValueError(f"'df_parts' misses '{missing_parts}' parts necessary to retrieve amino acid positions"
                             f" for 'label_test' ({label_test}) if only one sample of it occurs in 'labels'."
                             f"\n Add them to the current parts of 'df_parts': {list_parts}")
    if sum(mask_ref) == 1:
        missing_parts = [x for x in required_parts if x not in list_parts]
        if len(missing_parts) > 0:
            raise ValueError(f"'df_parts' misses '{missing_parts}' parts necessary to retrieve amino acid positions"
                             f" for 'label_ref' ({label_ref}) if only one sample of it occurs in 'labels'."
                             f"\n Add them to the current parts of 'df_parts': {list_parts}")


def check_col_cat(col_cat=None) -> None:
    """Check if col_cat valid column from df_feat"""
    if col_cat not in ut.COLS_FEAT_SCALES:
        raise ValueError(f"'col_cat' {col_cat} should be one of: {ut.COLS_FEAT_SCALES}")


def check_col_val(col_val=None) -> None:
    """Check if col_val valid column from df_feat"""
    cols_feat = ut.COLS_FEAT_STAT + ut.COLS_FEAT_WEIGHT
    if col_val not in cols_feat:
        raise ValueError(f"'col_val' {col_val} should be one of: {cols_feat}")


def check_match_labels_value_sources(labels=None, df_parts=None, dict_num_parts=None, name="labels") -> None:
    """Check at least one value source is given and each aligns row-wise with labels/targets."""
    if df_parts is None and dict_num_parts is None:
        raise ValueError("Provide at least one of 'df_parts' or 'dict_num_parts' to subset per group.")
    n = len(labels)
    if df_parts is not None:
        ut.check_df_parts(df_parts=df_parts)
        if len(df_parts) != n:
            raise ValueError(f"'df_parts' (n={len(df_parts)}) should have the same number of rows "
                             f"as '{name}' (n={n}).")
    if dict_num_parts is not None:
        ut.check_dict(name="dict_num_parts", val=dict_num_parts, accept_none=False)
        for part, arr in dict_num_parts.items():
            n_part = np.asarray(arr).shape[0]
            if n_part != n:
                raise ValueError(f"'dict_num_parts['{part}']' (axis-0 n={n_part}) should match "
                                 f"'{name}' (n={n}).")


def subset_value_sources(row_mask=None, df_parts=None, dict_num_parts=None):
    """Return row-matched copies of the provided value sources (boolean mask applied positionally)."""
    df_parts_sub = df_parts[row_mask].copy() if df_parts is not None else None
    if dict_num_parts is not None:
        dict_num_parts_sub = {part: np.asarray(arr)[row_mask].copy() for part, arr in dict_num_parts.items()}
    else:
        dict_num_parts_sub = None
    return df_parts_sub, dict_num_parts_sub


def check_match_df_feat_X(df_feat=None, X=None):
    """Check that a pre-computed feature matrix X aligns column-wise with df_feat."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"'X' (ndim={X.ndim}) should be a 2D feature matrix of shape (n_samples, n_features).")
    n_features = X.shape[1]
    if n_features != len(df_feat):
        raise ValueError(f"'X' (n_features={n_features}) should have one column per feature "
                         f"in 'df_feat' (n_features={len(df_feat)}).")
    if not np.isfinite(X).all():
        raise ValueError("'X' should not contain NaN or infinite values.")
    return X


def recover_seq_parts_from_df_parts_row(row=None):
    """Recover the basic ``(jmd_n_seq, tmd_seq, jmd_c_seq)`` from a single ``df_parts`` row.

    The JMD lengths are read off the parts (never given): the basic part set
    (``jmd_n``/``tmd``/``jmd_c``) is used directly, otherwise the default CPP extended set
    (``tmd``/``jmd_n_tmd_n``/``tmd_c_jmd_c``) is split around the TMD. A JMD that cannot be
    derived from the available parts is returned as an empty string (the features then carry
    no JMD-N/JMD-C either).
    """
    cols = set(row.index)
    _, col_jmd_n_tmd_n, col_tmd_c_jmd_c = ut.LIST_PARTS  # ['tmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c']
    tmd_seq = row[ut.COL_TMD] if ut.COL_TMD in cols else ""
    # JMD-N: basic column if present, else the JMD-N portion of jmd_n_tmd_n (prefix before the TMD)
    if ut.COL_JMD_N in cols:
        jmd_n_seq = row[ut.COL_JMD_N]
    elif col_jmd_n_tmd_n in cols and tmd_seq:
        jmd_n_tmd_n = row[col_jmd_n_tmd_n]
        n_overlap = max((i for i in range(len(tmd_seq) + 1) if jmd_n_tmd_n.endswith(tmd_seq[:i])), default=0)
        jmd_n_seq = jmd_n_tmd_n[:len(jmd_n_tmd_n) - n_overlap]
    else:
        jmd_n_seq = ""
    # JMD-C: basic column if present, else the JMD-C portion of tmd_c_jmd_c (suffix after the TMD)
    if ut.COL_JMD_C in cols:
        jmd_c_seq = row[ut.COL_JMD_C]
    elif col_tmd_c_jmd_c in cols and tmd_seq:
        tmd_c_jmd_c = row[col_tmd_c_jmd_c]
        c_overlap = max((i for i in range(1, len(tmd_seq) + 1) if tmd_c_jmd_c.startswith(tmd_seq[-i:])), default=0)
        jmd_c_seq = tmd_c_jmd_c[c_overlap:]
    else:
        jmd_c_seq = ""
    return jmd_n_seq, tmd_seq, jmd_c_seq


def check_match_df_seq_df_parts(df_seq=None, entry=None, jmd_n_seq="", tmd_seq="", jmd_c_seq="") -> None:
    """Check that the parts recovered from ``df_parts`` are consistent with ``df_seq`` for one protein."""
    mask = df_seq[ut.COL_ENTRY] == entry
    if not mask.any():
        raise ValueError(f"'sample' entry ('{entry}') from 'df_parts' is not in the '{ut.COL_ENTRY}' "
                         f"column of 'df_seq'. Available entries: "
                         f"{ut.preview_options(df_seq[ut.COL_ENTRY])}.")
    row = df_seq[mask].iloc[0]
    span = jmd_n_seq + tmd_seq + jmd_c_seq
    if ut.COL_SEQ in df_seq.columns:
        if span and span not in row[ut.COL_SEQ]:
            raise ValueError(f"'df_seq' and 'df_parts' do not match for entry ('{entry}'): the TMD-JMD span "
                             f"derived from 'df_parts' is not contained in the 'df_seq' sequence.")
    elif set(ut.COLS_SEQ_PARTS).issubset(df_seq.columns):
        seq_parts = (row[ut.COL_JMD_N], row[ut.COL_TMD], row[ut.COL_JMD_C])
        if seq_parts != (jmd_n_seq, tmd_seq, jmd_c_seq):
            raise ValueError(f"'df_seq' and 'df_parts' do not match for entry ('{entry}'): the TMD-JMD parts differ.")


# II Main Functions
class SequenceFeature:
    """
    Utility feature engineering class using sequences to create :class:`CPP` feature components (**Parts**, **Splits**,
    and  **Scales**) and data structures [Breimann25]_.

    The three feature components are the primary input for the :class:`aaanalysis.CPP` class and define
    Comparative Physicochemical Profiling (CPP) features.

    .. versionadded:: 0.1.0

    Notes
    -----
    Feature Components:
        - **Part**: A continuous subset of a sequence, such as a protein domain.
        - **Split**: Continuous or discontinuous subset of a **Part**, either segment or pattern.
        - **Scale**: A physicochemical scale, i.e., a set of numerical values (typically [0-1]) assigned to amino acids.

    Main Parts:
        We define three main parts from which each other part can be derived from:

        - **TMD (target middle domain)**: Protein domain of interest with varying length.
        - **JMD-N (juxta middle domain N-terminal)**: Protein domain or sequence region directly
          N-terminally next to the TMD, typically set to a fixed length (10 by default).
        - **JMD-C (juxta middle domain C-terminal)**: Protein domain or sequence region directly
          C-terminally next to the TMD, typically set to a fixed length (10 by default).

    Feature: Part + Split + Scale
        Physicochemical property (expressed as numerical scale) present at distinct amino acid
        positions within a protein sequence. The positions are obtained by splitting sequence parts
        into segments or patterns.

    Feature value: Realization of a Feature
        For a given sequence, a feature value is the average of a physicochemical scale over
        all amino acids obtained by splitting a sequence part.

    Valid sequence parts:
        - ``tmd``: Target Middle Domain (TMD).
        - ``tmd_e``: TMD extended N- and C-terminally by a number of residues, defined by the ``ext_len`` configuration option.
        - ``tmd_n``: N-terminal half of the TMD.
        - ``tmd_c``: C-terminal half of the TMD.
        - ``jmd_n``: N-terminal Juxt Middle Domain (JMD).
        - ``jmd_c``: C-terminal JMD.
        - ``ext_c``: Extended C-terminal region.
        - ``ext_n``: Extended N-terminal region.
        - ``tmd_jmd``: Combination of JMD-N, TMD, and JMD-C.
        - ``jmd_n_tmd_n``: Combination of JMD-N and N-terminal half of TMD.
        - ``tmd_c_jmd_c``: Combination of C-terminal half of TMD and JMD-C.
        - ``ext_n_tmd_n``: Extended N-terminal region and N-terminal half of TMD.
        - ``tmd_c_ext_c``: C-terminal half of TMD and extended C-terminal region.

    Default parts:
        The following three parts are provided by default: ``tmd``, ``jmd_n_tmd_n``, ``tmd_c_jmd_c``.

    """

    def __init__(self,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        """
        self.verbose = ut.check_verbose(verbose)

    # Part and Split methods
    def get_df_parts(self,
                     df_seq: pd.DataFrame,
                     list_parts: Optional[Union[str, List[str]]] = None,
                     all_parts: bool = False,
                     jmd_n_len: Union[int, None] = 10,
                     jmd_c_len: Union[int, None] = 10,
                     tmd_len: Optional[int] = None,
                     remove_entries_with_gaps: bool = False,
                     replace_non_canonical_aa: bool = False,
                     ) -> pd.DataFrame:
        """
        Create DataFrame with selected sequence parts.

        Slices each protein sequence in ``df_seq`` into the requested Parts
        (target middle domain (TMD), JMD-N, JMD-C, and combinations thereof) using the
        boundary information supplied with the sequences. The resulting ``df_parts``
        DataFrame is the primary sequence input for :class:`CPP` and for
        :meth:`SequenceFeature.feature_matrix`.

        .. versionadded:: 0.1.0

        .. versionchanged:: 1.1.0
            Added the ``pos``-anchor input mode (``tmd_len``).

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct format: **Position-based**, **Part-based**, **Sequence-based**, or **Sequence-TMD-based**.
        list_parts: list of str, default={``tmd``, ``jmd_n_tmd_n``, ``tmd_c_jmd_c``}
            Names of sequence parts that should be obtained for sequences from ``df_seq``.
        jmd_n_len: int, default=10
            Length of JMD-N in number of amino acids. If ``None``, ``jmd_n`` and ``jmd_c`` should be given.
        jmd_c_len: int, default=10
            Length of JMD-C in number of amino acids. If ``None``, ``jmd_n`` and ``jmd_c`` should be given.
        tmd_len: int, optional
            TMD length in amino acids for the **Anchor-based format** only (a ``sequence`` + ``pos`` ``df_seq``).
            Each 1-based anchor in ``pos`` is placed at the P1 position of a length-``tmd_len`` TMD
            (right-heavy for even ``tmd_len``); ignored for the other formats.
        all_parts: bool, default=False
            Whether to create DataFrame with all possible sequence parts (if ``True``) or parts given by ``list_parts``.
        remove_entries_with_gaps: bool, default=False
            Whether to exclude entries containing missing residues in their sequence parts (if ``True``),
            usually resulting from sequences being too short.
        replace_non_canonical_aa: bool, default=False
            Whether to replace non-canonical amino acids (e.g., 'X') by gap ('-') symbol.

        Returns
        -------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            Sequence parts DataFrame.

        See Also
        --------
        * :class:`aaanalysis.SequenceFeature` for definition of parts, and lists of all existing and default parts.

        Notes
        -----
        * If ``ext_len`` in aaanalysis.options is not set to > 0, following parts containing extended tmd are not
          considered for ``all_parts=True``: ['tmd_e', 'ext_c', 'ext_n', 'ext_n_tmd_n', 'tmd_c_ext_c'].
        * ``jmd_n_len`` and ``jmd_c_len`` must be both given, except for the part-based format.
        * ``tmd_start`` and ``tmd_stop`` use **1-based indexing** to follow standard biological annotation conventions 
          (e.g., UniProt), where residue positions start at 1. This allows direct use of annotated positions without conversion.

        Formats for ``df_seq`` are differentiated by their respective columns:

        **Position-based format**
            - 'sequence': The complete amino acid sequence.
            - 'tmd_start': Starting position of the TMD in the sequence (1-based, inclusive).
            - 'tmd_stop': Ending position of the TMD in the sequence (1-based, inclusive).

        **Part-based format**
            - 'jmd_n': Amino acid sequence for JMD-N.
            - 'tmd': Amino acid sequence for TMD.
            - 'jmd_c': Amino acid sequence for JMD-C.

        **Sequence-TMD-based format**
            - 'sequence' and 'tmd' columns.

        **Sequence-based format**
            - Only the 'sequence' column.

        **Anchor-based format**
            - 'sequence' and 'pos' columns, together with the ``tmd_len`` argument.
            - 'pos': per-row 1-based P1 anchor position(s) — a single ``int`` or a ``list[int]``. Each anchor
              is exploded into one row whose TMD is centered (right-heavy for even ``tmd_len``) on the anchor;
              multi-anchor rows yield multiple rows, ided in the index by ``<entry>_<win_start>-<win_stop>``.

        Examples
        --------
        .. include:: examples/sf_get_df_parts.rst
        """
        # Check input
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        check_parts_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_none_tmd_len=True)
        # Anchor-based format ('sequence' + 'pos' + tmd_len): explode to position-based, then reuse the normal path
        anchor_mode = (isinstance(df_seq, pd.DataFrame)
                       and ut.COL_SEQ in df_seq.columns and ut.COL_POS in df_seq.columns
                       and not set(ut.COLS_SEQ_POS).issubset(df_seq.columns)
                       and not set(ut.COLS_SEQ_PARTS).issubset(df_seq.columns)
                       and not set(ut.COLS_SEQ_TMD).issubset(df_seq.columns))
        list_entry_win = None
        if anchor_mode:
            ut.check_number_range(name="tmd_len", val=tmd_len, min_val=1, just_int=True)
            if jmd_n_len is None or jmd_c_len is None:
                raise ValueError("'jmd_n_len' and 'jmd_c_len' should both be given (not None) "
                                 "for the anchor-based ('sequence' + 'pos') format.")
            df_seq, list_entry_win = expand_pos_anchors_(df_seq=df_seq, tmd_len=tmd_len,
                                                         jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_df_seq(df_seq=df_seq)
        ut.check_bool(name="all_parts", val=all_parts)
        ut.check_bool(name="replace_non_canonical_aa", val=replace_non_canonical_aa)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts, accept_none=True)
        df_seq = check_match_df_seq_jmd_len(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        # Create df parts
        df_parts = get_df_parts_(df_seq=df_seq, list_parts=list_parts, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if anchor_mode:
            df_parts.index = list_entry_win
        if remove_entries_with_gaps:
            n_before = len(df_parts)
            df_parts = remove_entries_with_gaps_(df_parts=df_parts)
            n_removed = n_before - len(df_parts)
            if n_removed > 0 and self.verbose:
                warnings.warn(f"{n_removed} entries have been removed from 'df_seq' due to introduced gaps.")
        if replace_non_canonical_aa:
            df_parts = replace_non_canonical_aa_(df_parts=df_parts)
        if len(df_parts) == 0:
            raise ValueError(f"All entries have been removed from 'df_seq'. "
                             f"Reduce 'jmd_n_len' ({jmd_n_len}) and 'jmd_c_len' ({jmd_c_len}) settings.")
        return df_parts

    def get_seq_kws(self,
                    df_seq: pd.DataFrame,
                    df_parts: pd.DataFrame,
                    sample: Union[int, str],
                    ) -> dict:
        """
        Get the per-part sequence keyword arguments (``jmd_n_seq``, ``tmd_seq``, ``jmd_c_seq``) for one protein.

        Returns the TMD-JMD sequence parts of a single protein as a ready-to-use ``seq_kws`` dictionary, so
        the per-protein sequence can be passed directly to the :class:`CPPPlot` methods (e.g. for sample-level
        plots) via ``**seq_kws``, without manually slicing the DataFrame. The parts are taken from ``df_parts``
        (the same sequence parts that produced ``df_feat`` via :meth:`CPP.run`), so the displayed residues are
        always bound to the feature geometry; ``df_seq`` is cross-checked for consistency. The JMD lengths are
        read off ``df_parts`` (no length argument); a JMD that the parts do not contain is returned empty.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct format: **Position-based**, **Part-based**, **Sequence-based**, or **Sequence-TMD-based**.
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            Sequence parts DataFrame (indexed by ``entry``) as produced by :meth:`SequenceFeature.get_df_parts`
            and passed to :meth:`CPP.run`. Defines the TMD-JMD geometry; must be consistent with ``df_seq``.
        sample : int or str
            The protein to extract, given either as a row position in ``df_parts`` or as an entry name (str)
            from its index.

        Returns
        -------
        seq_kws : dict
            Dictionary with the keys ``jmd_n_seq``, ``tmd_seq``, and ``jmd_c_seq`` mapping to the
            corresponding amino acid sequence parts of the selected protein (empty string where a JMD part
            is not encoded in ``df_parts``).

        See Also
        --------
        * :meth:`SequenceFeature.get_df_parts` for creating the underlying sequence parts DataFrame.
        * :meth:`CPPPlot.profile` and :meth:`CPPPlot.feature_map`, which accept the returned parts via ``**seq_kws``.

        Examples
        --------
        .. include:: examples/sf_get_seq_kws.rst
        """
        # Check input
        ut.check_df(name="df_seq", df=df_seq, cols_required=[ut.COL_ENTRY])
        ut.check_df(name="df_parts", df=df_parts)
        entries = list(df_parts.index)
        if isinstance(sample, str):
            if sample not in entries:
                raise ValueError(f"'sample' ({sample}) should be an entry in the index of 'df_parts'. "
                                 f"First entries are: {entries[:5]}")
            row, entry = df_parts.loc[sample], sample
        elif isinstance(sample, (int, np.integer)) and not isinstance(sample, bool):
            ut.check_number_range(name="sample", val=int(sample), min_val=0, max_val=len(df_parts) - 1, just_int=True)
            row, entry = df_parts.iloc[int(sample)], entries[int(sample)]
        else:
            raise ValueError(f"'sample' ({sample}) should be an entry name (str) or a row position (int).")
        # Recover the basic parts from df_parts (JMD lengths encoded in the parts) and verify against df_seq
        jmd_n_seq, tmd_seq, jmd_c_seq = recover_seq_parts_from_df_parts_row(row=row)
        check_match_df_seq_df_parts(df_seq=df_seq, entry=entry,
                                    jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        seq_kws = {f"{ut.COL_JMD_N}_seq": jmd_n_seq, f"{ut.COL_TMD}_seq": tmd_seq, f"{ut.COL_JMD_C}_seq": jmd_c_seq}
        return seq_kws

    @staticmethod
    def get_split_kws(split_types: Optional[Union[Literal["Segment", "Pattern", "PeriodicPattern"],
                                                   List[Literal["Segment", "Pattern", "PeriodicPattern"]]]] = None,
                      n_split_min: int = 1,
                      n_split_max: int = 15,
                      steps_pattern: Optional[List[int]] = None,
                      n_min: int = 2,
                      n_max: int = 4,
                      len_max: int = 15,
                      steps_periodicpattern: Optional[List[int]] = None,
                      ) -> dict:
        """
        Create dictionary with kwargs for three split types:

            - **Segment**: continuous sub-sequence.
            - **Pattern**: non-periodic discontinuous sub-sequence
            - **PeriodicPattern**: periodic discontinuous sub-sequence.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        split_types: list of str, default=[``Segment``, ``Pattern``, ``PeriodicPattern``]
            Split types for which parameter dictionary should be generated.
        n_split_min: int, default=1
            Number to specify the greatest ``Segment``. Should be > 0.
        n_split_max: int, default=15,
            Number to specify the smallest ``Segment``. Should be >= ``n_split_min``.
        steps_pattern: list of int, default=[3, 4], optional
            Possible steps sizes for ``Pattern``. Should contain at least 1 non-negative integers
            if ``Pattern`` split_type is used. If ``None``, default is used.
        n_min: int, default=2
            Minimum number of steps for ``Pattern``. Should be <= ``n_max``.
        n_max: int, default=4
            Maximum number of steps for ``Pattern``. Should be >= ``n_min``.
        len_max: int, default=15
            Maximum length in amino acid position for ``Pattern`` by varying start position.
            Should be > min(``steps_pattern``).
        steps_periodicpattern: list of int, default=[3, 4], optional
            Size of odd and even steps for ``PeriodicPattern``. Should contain two non-negative integers if
            ``PeriodicPattern`` split_type is used. If ``None``, default is used.

        Returns
        -------
        split_kws : dict
            Nested dictionary with parameters for chosen split_types:

            - Segment: {n_split_min:1, n_split_max=15}
            - Pattern: {steps=[3, 4], n_min=2, n_max=4, len_max=15}
            - PeriodicPattern: {steps=[3, 4]}

        Notes
        -----
        The split bounds returned here are validated for internal consistency
        (e.g., ``n_split_min <= n_split_max``, ``n_min <= n_max``,
        ``len_max > min(steps_pattern)``); inconsistent values raise a ``ValueError``.

        Beyond that, the *feasible* maxima are effectively capped by the CPP part
        lengths used to build ``df_parts`` (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``;
        defaults 20/10/10): a ``Segment`` cannot be split into more pieces than its
        part has residues, and a ``Pattern``/``PeriodicPattern`` cannot span beyond
        the part length. Choosing ``n_split_max``, ``n_max``, or ``len_max`` larger
        than a part can accommodate does not raise here — those splits simply yield
        empty feature buckets downstream. The one config that is always degenerate
        regardless of part length, an empty ``Pattern`` bucket where even the shortest
        repeat exceeds ``len_max`` (``n_min * min(steps_pattern) > len_max``), emits a
        ``UserWarning`` naming the offending parameters so it can be fixed by raising
        ``len_max`` or lowering ``steps_pattern``/``n_min``.

        Examples
        --------
        .. include:: examples/sf_get_split_kws.rst
        """
        # Check input
        split_types = check_split_types(split_types=split_types)
        args_int = dict(n_split_min=n_split_min, n_split_max=n_split_max, n_min=n_min, n_max=n_max, len_max=len_max)
        for name in args_int:
            ut.check_number_range(name=name, val=args_int[name], just_int=False, min_val=1)
        steps_pattern = ut.check_list_like(name="steps_pattern", val=steps_pattern,
                                           accept_none=True, check_all_non_neg_int=True)
        steps_periodicpattern = ut.check_list_like(name="steps_periodicpattern", val=steps_periodicpattern,
                                                   accept_none=True, check_all_non_neg_int=True)
        steps_pattern = check_steps(steps=steps_pattern, steps_name="steps_pattern", len_min=1, fixed_len=False)
        steps_periodicpattern = check_steps(steps=steps_periodicpattern, steps_name="steps_periodicpattern",
                                            len_min=2, fixed_len=True)
        # Create kws for splits
        split_kws = get_split_kws_(n_split_min=n_split_min,
                                   n_split_max=n_split_max,
                                   steps_pattern=steps_pattern,
                                   n_min=n_min,
                                   n_max=n_max,
                                   len_max=len_max,
                                   steps_periodicpattern=steps_periodicpattern,
                                   split_types=split_types)
        # Post check
        check_split_kws(split_kws=split_kws)
        return split_kws

    # Feature methods
    def get_df_feat(self,
                    features: Union[ut.ArrayLike1D, pd.DataFrame],
                    df_parts: pd.DataFrame,
                    labels: ut.ArrayLike1D,
                    label_test: int = 1,
                    label_ref: int = 0,
                    df_scales: Optional[pd.DataFrame] = None,
                    df_cat: Optional[pd.DataFrame] = None,
                    start: int = 1,
                    tmd_len: int = 20,
                    jmd_c_len: int = 10,
                    jmd_n_len: int = 10,
                    accept_gaps: bool = False,
                    parametric: bool = False,
                    n_jobs: Union[int, None] = 1,
                    ) -> pd.DataFrame:
        """
        Create feature DataFrame for given features.

        Depending on the provided labels, the DataFrame is created for one of the three following cases:

            1. Group vs group comparison
            2. Sample vs group comparison
            3. Sample vs sample comparison

        * For the group vs group comparison, the general feature position will be provided.
        * For sample vs group or sample vs sample comparison, the amino acid segments
          and patterns for the respective sample from the test dataset (label = 1) will be given.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            Ids of features (``'PART-SPLIT-SCALE'``) for which ``df_feat`` should be created. Alternatively,
            a ``df_feat`` DataFrame, in which case its ``'feature'`` column is used.
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts. Must cover all parts in ``features``.
        labels: array-like, shape (n_samples,)
            Class labels for samples in ``df_parts``. Should contain only two different integer label values,
            representing test and reference group (typically, 1 and 0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney U test) test for p-value computation.
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        * Use parallel processing only for high number of features (>~1000 features per core)
        * For sample vs group or sample vs sample comparison, ``df_parts`` must comprise ``jmd_n``, ``tmd``, and
          ``jmd_c`` sequence parts as well as all parts in features.

        See Also
        --------
        * The :meth:`CPP.run` method for creating and filtering Comparative Physicochemical
          Profiling (CPP) features for discriminating between two groups of sequences.

        Examples
        --------
        .. include:: examples/sf_get_df_feat.rst
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        ut.check_df_parts(df_parts=df_parts)
        check_df_scales(df_scales=df_scales)
        features = ut.check_features(features=features, list_parts=list(df_parts), list_scales=list(df_scales))
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                                 len_required=len(df_parts), allow_other_vals=False)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        check_match_df_parts_features(df_parts=df_parts, features=features)
        check_match_df_scales_features(df_scales=df_scales, features=features)
        check_match_features_seq_parts(features=features, **args_len)
        check_match_df_scales_df_cat(df_scales=df_scales, df_cat=df_cat, verbose=self.verbose)
        df_parts = check_match_df_parts_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        check_match_labels_label_test_label_ref(labels=labels, label_test=label_test, label_ref=label_ref)
        check_match_df_parts_label_test_label_ref(df_parts=df_parts, labels=labels,
                                                  label_test=label_test, label_ref=label_ref)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # User warning
        if self.verbose:
            warn_creation_of_feature_matrix(features=features, df_parts=df_parts, name="df_feat")
        # Get sample difference to reference group
        df_feat = get_df_feat_(features=features, df_parts=df_parts, labels=labels,
                               label_test=label_test, label_ref=label_ref,
                               df_scales=df_scales, df_cat=df_cat,
                               accept_gaps=accept_gaps, parametric=parametric,
                               start=start, jmd_n_len=jmd_n_len, tmd_len=tmd_len, jmd_c_len=jmd_c_len,
                               n_jobs=n_jobs)
        return df_feat

    def feature_matrix(self,
                       features: Union[ut.ArrayLike1D, pd.DataFrame],
                       df_parts: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
                       df_scales: Optional[pd.DataFrame] = None,
                       accept_gaps: bool = False,
                       n_jobs: Union[int, None] = 1,
                       batch: bool = False,
                       df_seq: Optional[pd.DataFrame] = None,
                       df_parts_kws: Optional[dict] = None,
                       ) -> Union[ut.ArrayLike2D, List[ut.ArrayLike2D]]:
        """
        Create feature matrix for given feature ids and sequence parts.

        For each sample (row of ``df_parts``) and each feature id, looks up the
        physicochemical scale values at the residue positions defined by the feature's
        Part and Split components and averages them into a single feature value.
        The result is the numerical input ``X`` consumed by :meth:`CPP.run` and
        by :meth:`NumericalFeature.filter_correlation`.

        .. versionadded:: 0.1.0

        .. versionchanged:: 1.1.0
            Added the ``batch`` parameter for building a list of ``df_parts`` in a single pass.

        .. versionchanged:: 1.1.0
            Added the ``df_seq`` and ``df_parts_kws`` parameters to build ``df_parts`` internally, so the
            sequence-to-matrix step no longer requires a separate :meth:`get_df_parts` call.

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            Ids of features (``'PART-SPLIT-SCALE'``) for which matrix of feature values should be created.
            Alternatively, a ``df_feat`` DataFrame, in which case its ``'feature'`` column is used.
        df_parts : pd.DataFrame, shape (n_samples, n_parts), optional
            DataFrame with sequence parts. If ``batch=True``, instead a **list of such
            DataFrames** (one per batch; all must share the same part columns). Provide exactly one of
            ``df_parts`` or ``df_seq``.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.
        batch : bool, default=False
            If ``True``, ``df_parts`` is a list of part DataFrames processed in one amortized call
            (concatenated → Cython builder runs **once** → split back), returning one matrix per batch.
            Use for per-protein sliding scoring where the same ``features`` are applied to many small
            ``df_parts`` in a tight loop; the result is **byte-identical** to calling this per batch.
            Not supported together with ``df_seq``.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info), optional
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct format: **Position-based**, **Part-based**, **Sequence-based**, or **Sequence-TMD-based**.
            If given, ``df_parts`` is built internally via :meth:`get_df_parts`, as an alternative to passing
            ``df_parts`` directly. Provide exactly one of ``df_parts`` or ``df_seq``.
        df_parts_kws : dict, optional
            Keyword arguments forwarded to :meth:`get_df_parts` when building ``df_parts`` from ``df_seq``
            (e.g. ``{"list_parts": ["tmd"], "jmd_n_len": 10, "tmd_len": 20}``). Keys must be
            :meth:`get_df_parts` parameter names (``df_seq`` excluded); unset options use their defaults.
            Only valid together with ``df_seq``.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Feature matrix containing feature values for samples. If ``batch=True``, a **list** of such
            matrices aligned to the input list of ``df_parts``.

        Notes
        -----
        * Use parallel processing only for high number of features (>~1000 features per core)
        * ``batch=True`` amortizes the per-call scale-lookup build and kernel warm-up that dominate when
          this method is called thousands of times on small ``df_parts``.

        Examples
        --------
        .. include:: examples/sf_feature_matrix.rst
        """
        # Resolve sequence input: exactly one of 'df_parts' / 'df_seq' (mutually exclusive)
        ut.check_bool(name="batch", val=batch)
        if (df_parts is None) == (df_seq is None):
            raise ValueError("Exactly one of 'df_parts' or 'df_seq' should be given (not both, not neither).")
        check_df_parts_kws(df_parts_kws=df_parts_kws, df_seq=df_seq)
        if df_seq is not None:
            if batch:
                raise ValueError("'batch=True' is not supported together with 'df_seq'; "
                                 "build the part DataFrames yourself and pass them as a list via 'df_parts'.")
            # Build df_parts internally via the existing get_df_parts logic (same result as the two-step call)
            df_parts = self.get_df_parts(df_seq=df_seq, **(df_parts_kws or {}))
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        # Check input
        check_df_scales(df_scales=df_scales)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # Normalize to a list of df_parts; remember whether to unwrap the single result.
        if batch:
            if isinstance(df_parts, pd.DataFrame):
                raise ValueError("With 'batch=True', 'df_parts' should be a list of part "
                                 "DataFrames, not a single DataFrame.")
            list_df_parts = ut.check_list_like(name="df_parts", val=df_parts, accept_none=False)
            if len(list_df_parts) == 0:
                raise ValueError("'df_parts' should contain at least one DataFrame when 'batch=True'.")
        else:
            ut.check_df_parts(df_parts=df_parts)
            list_df_parts = [df_parts]
        for i, dfp in enumerate(list_df_parts):
            ut.check_df_parts(df_parts=dfp)
        cols0 = list(list_df_parts[0].columns)
        features = ut.check_features(features=features, list_parts=cols0, list_scales=list(df_scales))
        check_match_df_scales_features(df_scales=df_scales, features=features)
        for i, dfp in enumerate(list_df_parts):
            if list(dfp.columns) != cols0:
                raise ValueError(f"'df_parts' entry {i} parts {list(dfp.columns)} should match "
                                 f"the first entry's parts {cols0}.")
            check_match_df_parts_features(df_parts=dfp, features=features)
        # User warning (single-mode parity with the historical behavior)
        if self.verbose and not batch:
            warn_creation_of_feature_matrix(features=features, df_parts=list_df_parts[0])
        # Concatenate -> build ONCE (same Cython/fast builder as CPP.run; byte-identical to
        # the legacy ``get_feature_matrix_``, parity-tested) -> split back per batch.
        lengths = [len(dfp) for dfp in list_df_parts]
        df_all = pd.concat(list_df_parts, axis=0, ignore_index=True)
        df_all = check_match_df_parts_df_scales(df_scales=df_scales, df_parts=df_all, accept_gaps=accept_gaps)
        builder = _pick_feature_matrix_builder()
        X_all = builder(features=features, df_parts=df_all, df_scales=df_scales,
                        accept_gaps=accept_gaps, n_jobs=n_jobs)
        list_X, start = [], 0
        for n in lengths:
            list_X.append(X_all[start:start + n])
            start += n
        return list_X if batch else list_X[0]

    def prune_by_variance(self,
                          df_feat: pd.DataFrame,
                          df_parts: Optional[pd.DataFrame] = None,
                          df_scales: Optional[pd.DataFrame] = None,
                          threshold: float = 0.0,
                          X: Optional[ut.ArrayLike2D] = None,
                          accept_gaps: bool = False,
                          n_jobs: Union[int, None] = 1,
                          ) -> pd.DataFrame:
        """
        Prune near-constant features from a feature DataFrame by variance.

        Model-free **feature pruning** step: drops every feature whose column variance in the
        realized feature matrix (built from ``df_parts``, or supplied directly via ``X``) is at or
        below ``threshold``, and returns the row-filtered ``df_feat``. Use it on a fitted
        ``df_feat`` (e.g. from :meth:`CPP.run`) as the first reduction stage, before
        :meth:`SequenceFeature.prune_by_correlation` and :meth:`TreeModel.select_features`.

        This is distinct from CPP's in-run pre-filter (which screens candidate features by the
        test-group standard deviation): pruning measures variance over **all** samples of the
        already-selected features.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for
            each feature.
        df_parts : pd.DataFrame, shape (n_samples, n_parts), optional
            DataFrame with sequence parts. Used to build the feature matrix; not required if ``X`` is given.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        threshold : float, default=0.0
            Minimum **population variance** (``numpy`` ``var``, ``ddof=0``) a feature's column must exceed
            to be kept; the threshold is in **variance units**, not standard-deviation units. Feature
            values are means of (typically ``[0, 1]``-normalized) scale values, so variances are small —
            commonly below ``0.1`` — and a useful range is: ``0.0`` removes only strictly constant
            features, while ``~0.01`` to ``~0.05`` also prunes low-variance features. The variance is
            computed **over all provided samples** (every row of ``df_parts`` / ``X``, both classes
            together) per feature column — not the test group only, and not per split.
        X : array-like, shape (n_samples, n_features), optional
            Pre-computed feature matrix. Column ``i`` must correspond to the feature in row ``i`` of
            ``df_feat`` (same order). If given, it is used directly and ``df_parts`` / ``df_scales`` are
            ignored (e.g. to reuse a matrix across pruning calls or to prune a :meth:`CPP.run_num`
            ``df_feat``).
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_selected_features, n_feature_info)
            Feature DataFrame filtered to the features with variance above ``threshold``, with a reset index.

        Notes
        -----
        * **Variance metric**: population variance (``ddof=0``) of each feature column over all samples.
          A feature that is constant across the samples (zero peak-to-peak range) is treated as exactly
          zero variance, so ``threshold=0.0`` removes precisely the constant features even when floating
          point would otherwise leave a tiny non-zero variance.
        * **Scope**: variance reflects how much a feature varies across *your* samples; it is unrelated to
          CPP's in-run pre-filter, which screens *candidate* features by the **test-group** standard
          deviation (``max_std_test``) rather than the spread over all samples.
        * Recommended pruning order: variance (this method) -> correlation
          (:meth:`SequenceFeature.prune_by_correlation`) -> :meth:`TreeModel.select_features`.
        * A pruning that retains no feature (e.g. ``threshold`` above every feature's variance) raises a
          ``ValueError`` rather than returning an empty DataFrame.

        See Also
        --------
        * :meth:`SequenceFeature.prune_by_correlation` for the complementary redundancy-pruning step.
        * :meth:`SequenceFeature.feature_matrix` for the feature matrix that variance is computed over.
        * :meth:`TreeModel.select_features` for the model-based selection that follows pruning.

        Examples
        --------
        .. include:: examples/sf_prune_by_variance.rst
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_number_range(name="threshold", val=threshold, min_val=0, just_int=False, accept_none=False)
        if X is None:
            ut.check_df_parts(df_parts=df_parts)
            X = self.feature_matrix(features=df_feat, df_parts=df_parts, df_scales=df_scales,
                                    accept_gaps=accept_gaps, n_jobs=n_jobs)
        else:
            X = check_match_df_feat_X(df_feat=df_feat, X=X)
        # Prune features whose variance is at or below the threshold
        is_selected = filter_variance_(X, threshold=threshold)
        if not is_selected.any():
            raise ValueError(f"'threshold' ({threshold}) removed all features. Lower it to retain features.")
        df_feat = df_feat[is_selected].reset_index(drop=True)
        return df_feat

    def prune_by_correlation(self,
                             df_feat: pd.DataFrame,
                             df_parts: Optional[pd.DataFrame] = None,
                             df_scales: Optional[pd.DataFrame] = None,
                             max_cor: float = 0.7,
                             X: Optional[ut.ArrayLike2D] = None,
                             accept_gaps: bool = False,
                             n_jobs: Union[int, None] = 1,
                             ) -> pd.DataFrame:
        """
        Prune mutually correlated features from a feature DataFrame.

        Model-free **feature pruning** step: among features whose realized feature values
        (built from ``df_parts``, or supplied directly via ``X``) are pairwise correlated beyond
        ``max_cor``, keeps the one with the higher ``abs_auc`` and drops the others, returning the
        row-filtered ``df_feat``. Use it after :meth:`SequenceFeature.prune_by_variance` and before
        :meth:`TreeModel.select_features`.

        The correlation is **empirical** — measured over the actual samples in ``df_parts``. This is
        deliberately different from CPP's in-run redundancy reduction, which compares the underlying
        scale vectors (``df_scales.corr()``) together with positional overlap. Pruning here catches
        features that happen to be redundant on a specific dataset even when their scales are not.

        Compared with the lower-level :meth:`NumericalFeature.filter_correlation`, which takes a raw
        matrix ``X`` and returns a boolean mask keeping the **first** column of each correlated pair (in
        the order given), this method is df_feat-in / df_feat-out: it builds ``X`` for you, **ranks
        features by ``abs_auc`` first** so the dropped feature of a pair is always the weaker one, and
        returns the row-filtered ``df_feat``.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for
            each feature. Must contain the ``abs_auc`` statistic used as the deterministic tie-break.
        df_parts : pd.DataFrame, shape (n_samples, n_parts), optional
            DataFrame with sequence parts. Used to build the feature matrix; not required if ``X`` is given.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        max_cor : float, default=0.7
            Maximum **absolute Pearson correlation** [0-1] allowed between any two retained features.
            For each pair whose ``|corr| > max_cor``, the feature with the **lower** ``abs_auc`` is
            dropped (and the higher-``abs_auc`` one kept) — regardless of the input row order, because
            the method ranks by ``abs_auc`` internally. Lower ``max_cor`` to prune more aggressively.
        X : array-like, shape (n_samples, n_features), optional
            Pre-computed feature matrix. Column ``i`` must correspond to the feature in row ``i`` of the
            ``df_feat`` you pass (same order); the method then re-ranks ``df_feat`` and ``X`` together by
            ``abs_auc`` internally, so you do **not** pre-sort. If given, it is used directly and
            ``df_parts`` / ``df_scales`` are ignored (e.g. to reuse a matrix or to prune a
            :meth:`CPP.run_num` ``df_feat``).
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_selected_features, n_feature_info)
            Feature DataFrame filtered to a non-redundant subset (sorted by descending ``abs_auc``),
            with a reset index.

        Notes
        -----
        * **Tie-break / determinism**: features are sorted by descending ``abs_auc`` (ties broken by
          ``abs_mean_dif``) before pruning, so for every correlated pair the lower-``abs_auc`` feature is
          the one removed. This makes the output independent of the input row order and byte-identical
          across runs; the returned ``df_feat`` is in descending-``abs_auc`` order.
        * **X alignment**: if you pass a pre-computed ``X``, its columns must be aligned to the ``df_feat``
          rows you pass (column ``i`` = feature in row ``i``); the method reorders both together, so a
          mis-aligned ``X`` would correlate the wrong features.
        * The retained set is guaranteed to contain no feature pair with ``|corr| > max_cor``.
        * Constant (zero-variance) features have undefined correlation and are always retained here; run
          :meth:`SequenceFeature.prune_by_variance` first to remove them.
        * A ``df_feat`` with fewer than two features is returned unchanged (nothing to compare).

        See Also
        --------
        * :meth:`SequenceFeature.prune_by_variance` for the variance-pruning step that should precede this.
        * :meth:`NumericalFeature.filter_correlation` for the underlying correlation primitive on a matrix.
        * :meth:`TreeModel.select_features` for the model-based selection that follows pruning.

        Examples
        --------
        .. include:: examples/sf_prune_by_correlation.rst
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0, max_val=1, just_int=False, accept_none=False)
        if X is not None:
            X = check_match_df_feat_X(df_feat=df_feat, X=X)
        elif df_parts is not None:
            ut.check_df_parts(df_parts=df_parts)
        else:
            raise ValueError("Either 'df_parts' or a pre-computed 'X' should be provided.")
        # Nothing to compare with fewer than two features
        if len(df_feat) < 2:
            return df_feat.reset_index(drop=True)
        # Order by descending abs_auc (tie-break abs_mean_dif) so the stronger feature of a pair is kept
        order = df_feat.sort_values(by=[ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF],
                                    ascending=False).index.to_numpy()
        df_feat = df_feat.loc[order].reset_index(drop=True)
        if X is None:
            X = self.feature_matrix(features=df_feat, df_parts=df_parts, df_scales=df_scales,
                                    accept_gaps=accept_gaps, n_jobs=n_jobs)
        else:
            X = X[:, order]
        # Prune redundant features (constant columns have undefined correlation -> always kept).
        # Detect constant columns by zero peak-to-peak range (robust to float epsilon).
        is_selected = np.ones(X.shape[1], dtype=bool)
        non_constant = np.ptp(np.asarray(X, dtype=float), axis=0) > 0
        idx_nc = np.where(non_constant)[0]
        if idx_nc.size >= 2:
            is_selected[idx_nc] = filter_correlation_(X[:, idx_nc], max_cor=max_cor)
        df_feat = df_feat[is_selected].reset_index(drop=True)
        return df_feat

    def get_features(self,
                     list_parts: Optional[List[str]] = None,
                     all_parts: bool = False,
                     split_kws: Optional[dict] = None,
                     list_scales: Optional[List[str]] = None,
                     ) -> List[str]:
        """
        Create list of all feature ids for given Parts, Splits, and Scales.

        Enumerates every combination of the requested sequence parts, split types
        (Segment, Pattern, PeriodicPattern from :meth:`SequenceFeature.get_split_kws`),
        and scale names, returning structured ``PART-SPLIT-SCALE`` feature ids.
        These ids can be passed directly to :meth:`SequenceFeature.feature_matrix`
        or used to pre-select a feature space before calling :meth:`CPP.run`.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        list_parts: list of str, default=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"]
            Names of sequence parts which should be created (e.g., 'tmd'). Length should be >= 1.
        all_parts: bool, default=False
            Whether to create DataFrame with all possible sequence parts (if ``True``) or parts given by ``list_parts``.
        split_kws : dict, optional
            Dictionary with parameter dictionary for each chosen split_type. Default from :meth:`SequenceFeature.get_split_kws`.
        list_scales : list of str, optional
            Names of scales. Default scales from :meth:`load_scales` with ``name='scales'``.

        Returns
        -------
        features : list of str
            Ids of all possible features for combination of Parts, Splits, and Scales with form: PART-SPLIT-SCALE

        Notes
        -----
        * If ``ext_len`` in aaanalysis.options is not set to > 0, following parts containing extended tmd are not
          considered for ``all_parts=True``: ['tmd_e', 'ext_c', 'ext_n', 'ext_n_tmd_n', 'tmd_c_ext_c'].

        Examples
        --------
        .. include:: examples/sf_get_features.rst
        """
        # Load defaults
        if list_scales is None:
            list_scales = list(ut.load_default_scales())
        if split_kws is None:
            split_kws = self.get_split_kws()
        # Check input
        ut.check_bool(name="all_parts", val=all_parts)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        check_split_kws(split_kws=split_kws)
        list_scales = ut.check_list_like(name="list_scales", val=list_scales, accept_none=False)
        # Get features
        features = get_features_(list_parts=list_parts, split_kws=split_kws, list_scales=list_scales)
        return features

    @staticmethod
    def get_feature_names(features: Union[ut.ArrayLike1D, pd.DataFrame],
                          df_cat: Optional[pd.DataFrame] = None,
                          start: int = 1,
                          tmd_len: int = 20,
                          jmd_n_len: int = 10,
                          jmd_c_len: int = 10,
                          ) -> List[str]:
        """
        Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions]).

        Replaces the compact ``PART-SPLIT-SCALE`` id format produced by
        :meth:`SequenceFeature.get_features` with a human-readable string that
        shows the full scale name from ``df_cat`` together with the residue
        positions covered by the feature's Split, making feature results easier
        to interpret in :class:`CPP` output DataFrames.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            List of feature ids (``'PART-SPLIT-SCALE'``). Alternatively, a ``df_feat`` DataFrame,
            in which case its ``'feature'`` column is used.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).

        Returns
        -------
        feat_names : list of str
            Names of features.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with ids in ``features``.
        * Positions are given depending on the three split types:

            - Segment: [first...last]
            - Pattern: [all positions]
            - PeriodicPattern: [first..step1/step2..last]

        Examples
        --------
        .. include:: examples/sf_get_feature_names.rst
        """
        # Load defaults
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        features = ut.check_features(features=features)
        check_df_cat(df_cat=df_cat)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        check_match_df_cat_features(df_cat=df_cat, features=features)
        check_match_features_seq_parts(features=features, **args_len)
        # Get feature names
        feat_names = get_feature_names_(features=features,
                                        df_cat=df_cat,
                                        start=start,
                                        tmd_len=tmd_len,
                                        jmd_c_len=jmd_c_len,
                                        jmd_n_len=jmd_n_len)
        return feat_names

    @staticmethod
    def get_feature_descriptions(features: Union[ut.ArrayLike1D, pd.DataFrame],
                                 df_cat: Optional[pd.DataFrame] = None,
                                 start: int = 1,
                                 tmd_len: int = 20,
                                 jmd_n_len: int = 10,
                                 jmd_c_len: int = 10,
                                 ) -> List[str]:
        """
        Build a standardized, human-readable description for each feature id (PART-SPLIT-SCALE).

        Complements the compact :meth:`SequenceFeature.get_feature_names` label
        (``'scale name [positions]'``) with one self-contained sentence per
        feature that spells out all three id fields: the sequence part as a
        readable label, the Split as a phrase (e.g. ``'segment 2 of 4'``),
        and the scale as its AAontology name together with the category and
        subcategory from ``df_cat``. Terminology is drawn from fixed vocabularies
        (the part labels and the AAontology category/subcategory wording), so the
        output is deterministic and consistent across runs. The result can be
        assigned to a ``df_feat`` column (``'feature_description'``) for readable
        :class:`CPP` output without changing the ``'feature'`` id string.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            List of feature ids (``'PART-SPLIT-SCALE'``). Alternatively, a ``df_feat`` DataFrame,
            in which case its ``'feature'`` column is used.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).

        Returns
        -------
        feat_descriptions : list of str
            Human-readable description for each feature, one per feature id.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with ids in ``features``.
        * Part labels come from a fixed vocabulary; category and subcategory wording is taken
          verbatim from ``df_cat`` (the AAontology scale categories table).

        See Also
        --------
        * :meth:`SequenceFeature.get_feature_names` for the compact label form.

        Examples
        --------
        .. include:: examples/sf_get_feature_descriptions.rst
        """
        # Load defaults
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        features = ut.check_features(features=features)
        check_df_cat(df_cat=df_cat)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        check_match_df_cat_features(df_cat=df_cat, features=features)
        check_match_features_seq_parts(features=features, **args_len)
        # Get feature descriptions
        feat_descriptions = get_feature_descriptions_(features=features,
                                                      df_cat=df_cat,
                                                      start=start,
                                                      tmd_len=tmd_len,
                                                      jmd_c_len=jmd_c_len,
                                                      jmd_n_len=jmd_n_len)
        return feat_descriptions

    @staticmethod
    def get_feature_positions(features: Union[ut.ArrayLike1D, pd.DataFrame],
                              start: int = 1,
                              tmd_len: int = 20,
                              jmd_n_len: int = 10,
                              jmd_c_len: int = 10,
                              tmd_seq: Optional[str] = None,
                              jmd_n_seq: Optional[str] = None,
                              jmd_c_seq: Optional[str] = None,
                              ) -> ut.ArrayLike1D:
        """
        Create for features a list of corresponding positions or amino acids.

        Resolves each ``PART-SPLIT-SCALE`` feature id produced by
        :meth:`SequenceFeature.get_features` to the concrete residue positions it
        covers, using the supplied domain lengths. When sequence strings
        (``tmd_seq``, ``jmd_n_seq``, ``jmd_c_seq``) are also provided the method
        returns the actual amino acid segments or patterns instead of position
        numbers, which is useful for inspecting :class:`CPP` feature results on a
        specific protein.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        features : array-like, shape (n_features,) or pd.DataFrame
            List of feature ids (``'PART-SPLIT-SCALE'``). Alternatively, a ``df_feat`` DataFrame,
            in which case its ``'feature'`` column is used.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        tmd_seq : str, optional
            Sequence of TMD. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_n_seq : str, optional
            Sequence of JMD-N. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_c_seq : str, optional
            Sequence of JMD-C. If given, respective amino acid segments/patterns will be returned instead of positions.

        Returns
        -------
        list_pos : list
            List of residue positions for each feature. Returned when no sequence arguments are provided.
        list_aa : list
            List of amino acid segments or patterns for each feature. Returned when ``tmd_seq``,
            ``jmd_n_seq``, and ``jmd_c_seq`` are all provided.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with ids in ``features``.
        * Length of sequence (``tmd_seq``, ``jmd_n_seq``, ``jmd_c_seq``) must match with ids in ``features``.

        Examples
        --------
        .. include:: examples/sf_get_feature_positions.rst
        """
        # Check input
        features = ut.check_features(features=features)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                             tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        check_match_features_seq_parts(features=features, **args_seq, **args_len)
        # Get feature position
        if args_seq["tmd_seq"] is not None:
            list_aa = get_amino_acids_(features=features, **args_seq)
            return list_aa
        else:
            list_pos = get_positions_(features=features, start=start, **args_len)
            return list_pos

    @staticmethod
    def get_df_pos(df_feat: pd.DataFrame,
                   col_val: str = "mean_dif",
                   col_cat: str = "category",
                   start: int = 1,
                   tmd_len: int = 20,
                   jmd_n_len: int = 10,
                   jmd_c_len: int = 10,
                   list_parts: Optional[Union[str, List[str]]] = None,
                   normalize : bool = False,
                   ) -> pd.DataFrame:
        """
        Create DataFrame of aggregated (mean or sum) feature values per residue position and scale.

        Projects the per-feature statistics from a ``df_feat`` DataFrame (typically
        the output of :meth:`CPP.run`) onto individual residue positions by
        spreading each feature's value across every position its Split covers and
        then aggregating by scale category. The resulting position-by-category
        matrix is the direct input for :class:`CPPPlot` position plots.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        col_val : {'abs_auc', 'abs_mean_dif', 'mean_dif', 'std_test', 'std_ref'}, default='mean_dif'
            Column name in ``df_feat`` containing numerical values to ``average``. If feature importance and impact
            are provided as {'feat_importance', 'feat_impact'} columns, their ``sum`` of values is computed.
        col_cat : {'category', 'subcategory', 'scale_name'}, default='category'
            Column name in ``df_feat`` for categorizing the numerical values during aggregation.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        list_parts: str or list of str, optional
            Specific sequence parts to consider for numerical value aggregation.
        normalize : bool, default=False
            If ``True``, normalizes aggregated numerical values to a total of 100%.

        Returns
        -------
        df_pos : pd.DataFrame, shape (n_categories, n_positions)
            DataFrame with aggregated numerical values per position.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with feature ids in ``df_feat``.

        Examples
        --------
        .. include:: examples/sf_get_df_pos.rst
        """
        # Check input
        list_parts = ut.check_list_parts(list_parts=list_parts, return_default=False, accept_none=True)
        # Do not check for list_parts since df_pos can be obtained for any part
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_col_cat(col_cat=col_cat)
        check_col_val(col_val=col_val)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_bool(name="normalize", val=normalize)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE], **args_len)
        # Get df pos
        stop = start + jmd_n_len + tmd_len + jmd_c_len - 1
        value_type = ut.DICT_VALUE_TYPE[col_val]
        df_pos = get_df_pos_(df_feat=df_feat, col_cat=col_cat, col_val=col_val, value_type=value_type,
                             start=start, stop=stop)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        if list_parts is not None:
            df_pos = get_df_pos_parts_(df_pos=df_pos, value_type=value_type,
                                       start=start, **args_len, list_parts=list_parts)
        return df_pos

    @staticmethod
    def get_labels_ovr(labels: ut.ArrayLike1D,
                       label_test: int = 1,
                       label_ref: int = 0,
                       ) -> Dict[int, np.ndarray]:
        """
        Convert multi-class labels into one-vs-rest (OvR) binary label arrays.

        One-vs-rest (OvR) maps each of the K classes to a full-length binary label
        array in which that class is the test group and all remaining classes are
        the reference group. Since no samples are dropped, the K arrays can be
        looped through a single :class:`CPP` instance via :meth:`CPP.run` (the
        ``df_parts`` is unchanged), yielding one binary feature set per class.
        Discarding the other classes instead is :meth:`get_labels_ovo`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Multi-class labels for samples. Must be integers (more than one distinct
            value); for continuous targets discretize first with :meth:`get_labels_quantile`.
        label_test : int, default=1
            Value assigned to the target class of each one-vs-rest array.
        label_ref : int, default=0
            Value assigned to all remaining classes.

        Returns
        -------
        dict_labels : dict
            Dictionary mapping each class label to its one-vs-rest binary label array
            (numpy array of shape (n_samples,)), keyed in sorted class order.

        Notes
        -----
        * Each returned binary label array is directly usable as the ``labels``
          argument of :meth:`CPP.run` / :meth:`CPP.run_num`.
        * To aggregate the per-class results, run CPP per array and concatenate the
          returned ``df_feat`` frames, tagging each with its class key.
        * **Complexity:** O(n_samples x n_classes); scales linearly in both, so OvR
          stays cheap for large K.

        See Also
        --------
        * :class:`aaanalysis.CPP`: consumes each returned binary label array via :meth:`CPP.run`.
        * :meth:`get_labels_ovo`: pairwise (one-vs-one) alternative that subsets samples.
        * :meth:`get_labels_quantile`: discretize a continuous target into binary labels.

        Examples
        --------
        .. include:: examples/sf_get_labels_ovr.rst
        """
        # Check input
        labels = ut.check_labels(labels=labels)
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        if label_test == label_ref:
            raise ValueError(f"'label_test' ({label_test}) should differ from 'label_ref' ({label_ref}).")
        # Convert labels
        return get_labels_ovr_(labels=labels, label_test=label_test, label_ref=label_ref)

    @staticmethod
    def get_labels_ovo(labels: ut.ArrayLike1D,
                       df_parts: Optional[pd.DataFrame] = None,
                       dict_num_parts: Optional[Dict[str, np.ndarray]] = None,
                       label_test: int = 1,
                       label_ref: int = 0,
                       ) -> Dict[Tuple[int, int], Tuple[Optional[pd.DataFrame], Optional[Dict[str, np.ndarray]], np.ndarray]]:
        """
        Convert multi-class labels into one-vs-one (OvO) binary labels with row-matched parts.

        One-vs-one (OvO) maps each unordered pair of classes ``(a, b)`` to the subset of
        samples belonging to either class together with a binary label array over that
        subset (class ``a`` as test, class ``b`` as reference). Because the other classes
        are discarded, each pair needs its own row subset of the value source. This method
        applies the selection for you: pass ``df_parts`` (for :meth:`CPP.run`) and/or
        ``dict_num_parts`` (for :meth:`CPP.run_num`) and it returns, per pair, the
        row-matched **copies** alongside the binary labels — ready to drop straight into a
        :class:`CPP` instance built on that pair.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Multi-class labels for samples. Must be integers (more than one distinct value).
        df_parts : pd.DataFrame, optional
            Parts table (``CPP.run`` value source) aligned row-wise with ``labels``. When
            given, the returned per-pair copy is subset to that pair's samples.
        dict_num_parts : dict, optional
            Per-part numerical tensors (``CPP.run_num`` value source, ``{part: array}`` with
            sample axis first) aligned row-wise with ``labels``. Subset per pair like
            ``df_parts``. At least one of ``df_parts`` / ``dict_num_parts`` is required.
        label_test : int, default=1
            Value assigned to the first class of each pair.
        label_ref : int, default=0
            Value assigned to the second class of each pair.

        Returns
        -------
        dict_labels : dict
            Dictionary mapping each class pair ``(a, b)`` to a tuple
            ``(df_parts_pair, dict_num_parts_pair, labels_pair)``: the row-matched
            ``df_parts`` copy (or ``None`` if not supplied), the row-matched
            ``dict_num_parts`` copy (or ``None`` if not supplied), and the binary label
            array over that pair's samples. The original inputs are never modified.

        Notes
        -----
        * The selection is applied positionally; ``df_parts_pair.index`` records which
          original rows the pair retained.
        * **Complexity:** O(n_samples x n_classes^2): K classes produce K(K-1)/2 pairs
          (K=10 -> 45, K=20 -> 190), each needing its own CPP instance. Prefer OvO for
          small K (~<10) and :meth:`get_labels_ovr` for larger problems.

        See Also
        --------
        * :class:`aaanalysis.CPP`: consumes each pair's ``df_parts_pair`` / ``dict_num_parts_pair`` and ``labels_pair``.
        * :meth:`get_labels_ovr`: one-vs-rest alternative that keeps all samples.

        Examples
        --------
        .. include:: examples/sf_get_labels_ovo.rst
        """
        # Check input
        labels = ut.check_labels(labels=labels)
        check_match_labels_value_sources(labels=labels, df_parts=df_parts,
                                         dict_num_parts=dict_num_parts, name="labels")
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        if label_test == label_ref:
            raise ValueError(f"'label_test' ({label_test}) should differ from 'label_ref' ({label_ref}).")
        # Convert labels and subset value sources per pair
        dict_mask_labels = get_labels_ovo_(labels=labels, label_test=label_test, label_ref=label_ref)
        dict_labels = {}
        for pair, (row_mask, labels_pair) in dict_mask_labels.items():
            df_parts_pair, dict_num_parts_pair = subset_value_sources(
                row_mask=row_mask, df_parts=df_parts, dict_num_parts=dict_num_parts)
            dict_labels[pair] = (df_parts_pair, dict_num_parts_pair, labels_pair)
        return dict_labels

    @staticmethod
    def get_labels_quantile(targets: ut.ArrayLike1D,
                            q: float = 0.5,
                            label_test: int = 1,
                            label_ref: int = 0,
                            ) -> np.ndarray:
        """
        Convert a continuous target into a binary label array by a single quantile threshold.

        Splits samples at the ``q``-quantile of ``targets``: samples at or above the
        cut become the test group and the remainder the reference group. No samples
        are dropped, so the result is directly usable as the ``labels`` argument of
        :meth:`CPP.run` / :meth:`CPP.run_num`, enabling regression-style tasks (e.g.
        thermostability, binding affinity) within the binary CPP framework. For a
        fixed positive set with stepwise-lowered negative cuts, use
        :meth:`get_labels_tiered`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        targets : array-like, shape (n_samples,)
            Continuous target values for samples.
        q : float, default=0.5
            Quantile in the open interval (0, 1) defining the split threshold
            (``0.5`` splits at the median).
        label_test : int, default=1
            Value assigned to samples at or above the threshold.
        label_ref : int, default=0
            Value assigned to samples below the threshold.

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
            Binary label array.

        Notes
        -----
        * Targets are converted to ``float64``. Raises ``ValueError`` up front if the
          split would yield only one class (constant targets, or a cut leaving one
          side empty), instead of failing later inside :meth:`CPP.run`.
        * **Complexity:** O(n_samples log n_samples) from the quantile, negligible
          beside CPP runtime.

        See Also
        --------
        * :class:`aaanalysis.CPP`: consumes the returned binary label array via :meth:`CPP.run`.
        * :meth:`get_labels_tiered`: fixed positive set vs stepwise-lowered negative cuts.

        Examples
        --------
        .. include:: examples/sf_get_labels_quantile.rst
        """
        # Check input
        targets = ut.check_list_like(name="targets", val=targets, accept_none=False)
        targets = np.asarray(targets, dtype=float)
        ut.check_number_range(name="q", val=q, min_val=0, max_val=1, exclusive_limits=True, just_int=False)
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        if label_test == label_ref:
            raise ValueError(f"'label_test' ({label_test}) should differ from 'label_ref' ({label_ref}).")
        # Convert targets
        labels = get_labels_quantile_(targets=targets, q=q, label_test=label_test, label_ref=label_ref)
        if len(np.unique(labels)) < 2:
            raise ValueError(f"'targets' produce a single class at q={q} (all values equal, or the "
                             f"cut leaves one side empty); adjust 'q' or 'targets'.")
        return labels

    @staticmethod
    def get_labels_tiered(targets: ut.ArrayLike1D,
                          q_pos: float = 0.8,
                          list_q_neg: Sequence[float] = (0.8, 0.5, 0.3),
                          df_parts: Optional[pd.DataFrame] = None,
                          dict_num_parts: Optional[Dict[str, np.ndarray]] = None,
                          label_test: int = 1,
                          label_ref: int = 0,
                          ) -> Dict[float, Tuple[Optional[pd.DataFrame], Optional[Dict[str, np.ndarray]], np.ndarray]]:
        """
        Build tiered binary labels sharing a fixed positive set, with row-matched parts.

        Holds the positive set fixed at ``targets >= Q(q_pos)`` and sweeps a series of
        negative cuts ``targets <= Q(q_neg)`` for each ``q_neg`` in ``list_q_neg``,
        dropping the middle band each time. This compares CPP settings against the same
        positives while the negatives move toward more extreme low values. Like
        :meth:`get_labels_ovo`, each tier drops samples, so this method applies the
        selection for you: pass ``df_parts`` (for :meth:`CPP.run`) and/or
        ``dict_num_parts`` (for :meth:`CPP.run_num`) and it returns, per tier, the
        row-matched **copies** alongside the binary labels.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        targets : array-like, shape (n_samples,)
            Continuous target values for samples.
        q_pos : float, default=0.8
            Quantile in (0, 1) defining the fixed positive cut: positives are
            ``targets >= Q(q_pos)``.
        list_q_neg : sequence of float, default=(0.8, 0.5, 0.3)
            Quantiles in (0, 1); for each, negatives are ``targets <= Q(q_neg)``
            (positives take precedence on ties) and the middle band is dropped.
        df_parts : pd.DataFrame, optional
            Parts table (``CPP.run`` value source) aligned row-wise with ``targets``. When
            given, the returned per-tier copy is subset to that tier's samples.
        dict_num_parts : dict, optional
            Per-part numerical tensors (``CPP.run_num`` value source, ``{part: array}`` with
            sample axis first) aligned row-wise with ``targets``. Subset per tier like
            ``df_parts``. At least one of ``df_parts`` / ``dict_num_parts`` is required.
        label_test : int, default=1
            Value assigned to positive samples.
        label_ref : int, default=0
            Value assigned to negative samples.

        Returns
        -------
        dict_labels : dict
            Dictionary mapping each ``q_neg`` to a tuple
            ``(df_parts_tier, dict_num_parts_tier, labels_tier)``: the row-matched
            ``df_parts`` copy (or ``None`` if not supplied), the row-matched
            ``dict_num_parts`` copy (or ``None`` if not supplied), and the binary label
            array over that tier's samples. The original inputs are never modified.

        Notes
        -----
        * Raises ``ValueError`` if any tier yields only one class (e.g. ``q_neg`` above
          ``q_pos`` leaving no negatives).
        * The selection is applied positionally; ``df_parts_tier.index`` records which
          original rows the tier retained.
        * **Complexity:** O(n_samples log n_samples x n_tiers).

        See Also
        --------
        * :class:`aaanalysis.CPP`: consumes each tier's ``df_parts_tier`` / ``dict_num_parts_tier`` and ``labels_tier``.
        * :meth:`get_labels_quantile`: single-cut variant that keeps all samples.

        Examples
        --------
        .. include:: examples/sf_get_labels_tiered.rst
        """
        # Check input
        targets = ut.check_list_like(name="targets", val=targets, accept_none=False)
        targets = np.asarray(targets, dtype=float)
        ut.check_number_range(name="q_pos", val=q_pos, min_val=0, max_val=1,
                              exclusive_limits=True, just_int=False)
        list_q_neg = ut.check_list_like(name="list_q_neg", val=list_q_neg, accept_none=False)
        if len(list_q_neg) == 0:
            raise ValueError("'list_q_neg' should contain at least one quantile.")
        for q_neg in list_q_neg:
            ut.check_number_range(name="list_q_neg", val=q_neg, min_val=0, max_val=1,
                                  exclusive_limits=True, just_int=False)
        check_match_labels_value_sources(labels=targets, df_parts=df_parts,
                                         dict_num_parts=dict_num_parts, name="targets")
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        if label_test == label_ref:
            raise ValueError(f"'label_test' ({label_test}) should differ from 'label_ref' ({label_ref}).")
        # Convert targets and subset value sources per tier
        dict_mask_labels = get_labels_tiered_(targets=targets, q_pos=q_pos, list_q_neg=list_q_neg,
                                              label_test=label_test, label_ref=label_ref)
        dict_labels = {}
        for q_neg, (row_mask, labels_tier) in dict_mask_labels.items():
            df_parts_tier, dict_num_parts_tier = subset_value_sources(
                row_mask=row_mask, df_parts=df_parts, dict_num_parts=dict_num_parts)
            dict_labels[q_neg] = (df_parts_tier, dict_num_parts_tier, labels_tier)
        return dict_labels

    @staticmethod
    def get_df_parts_from_windows(dict_parts: Dict[str, Union[pd.DataFrame, Sequence[str]]],
                                  ) -> pd.DataFrame:
        """
        Assemble a ``df_parts`` from per-part window sets (e.g. :class:`AAWindowSampler` outputs).

        Builds a reference ``df_parts`` by stitching one window set per sequence part,
        so each part can be generated with its **own** recipe. This unlocks
        biologically-motivated reference backgrounds where the parts differ in
        physicochemical prior, e.g. a coil-propensity JMD-N, an alpha-helix TMD, and a
        coil-propensity JMD-C, each produced by a separate call to
        :meth:`AAWindowSampler.sample_synthetic` with a different ``generator`` and
        ``window_size``. The assembled frame is used as the reference class for
        :class:`CPP` exactly like a real ``df_parts``. This method does not sample
        sequences itself; it only consumes window sets produced by
        :class:`AAWindowSampler`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        dict_parts : dict
            Dictionary mapping each part name (one of :attr:`aaanalysis.utils.LIST_ALL_PARTS`,
            e.g. ``'jmd_n'``, ``'tmd'``, ``'jmd_c'``) to its window set. Each value is
            either a DataFrame with a ``'window'`` column (the output of
            :meth:`AAWindowSampler.sample_synthetic`) or a sequence of window strings.
            All window lists must be in the **same order** across parts (the i-th window
            of each part forms the i-th reference row); differing orders silently break
            the biological meaning of the assembled rows.

        Returns
        -------
        df_parts : pd.DataFrame, shape (n_windows, n_parts)
            Reference parts with one column per key in ``dict_parts`` and an index of
            ``'REF<i>'`` identifiers.

        Notes
        -----
        * If the parts supply different numbers of windows, a ``RuntimeWarning`` is
          issued and all parts are truncated to the smallest count.
        * Concatenate the result with a real ``df_parts`` (matching columns) and label
          the two groups before calling :meth:`CPP.run`.

        See Also
        --------
        * :class:`aaanalysis.AAWindowSampler`: produces the per-part window sets (``sample_synthetic``).
        * :class:`aaanalysis.CPP`: consumes the assembled ``df_parts`` via :meth:`CPP.run`.

        Examples
        --------
        .. include:: examples/sf_get_df_parts_from_windows.rst
        """
        # Check input
        ut.check_dict(name="dict_parts", val=dict_parts, accept_none=False)
        if len(dict_parts) == 0:
            raise ValueError("'dict_parts' should not be empty.")
        data = {}
        counts = {}
        for part, source in dict_parts.items():
            if part not in ut.LIST_ALL_PARTS:
                raise ValueError(f"'dict_parts' key '{part}' should be one of: {ut.LIST_ALL_PARTS}")
            if isinstance(source, pd.DataFrame):
                if ut.COL_WINDOW not in source.columns:
                    raise ValueError(f"'dict_parts['{part}']' DataFrame should contain a "
                                     f"'{ut.COL_WINDOW}' column (the output of "
                                     f"AAWindowSampler.sample_synthetic).")
                windows = source[ut.COL_WINDOW].tolist()
            else:
                windows = list(ut.check_list_like(name=f"dict_parts['{part}']", val=source))
            if len(windows) == 0:
                raise ValueError(f"'dict_parts['{part}']' should contain at least one window.")
            if not all(isinstance(w, str) for w in windows):
                raise ValueError(f"'dict_parts['{part}']' windows should all be strings.")
            data[part] = windows
            counts[part] = len(windows)
        # Align by position; on mismatch warn and truncate to the smallest count
        n_windows = min(counts.values())
        if len(set(counts.values())) > 1:
            warnings.warn(f"window counts differ across parts {counts}; truncating to {n_windows}.",
                          RuntimeWarning)
            data = {part: windows[:n_windows] for part, windows in data.items()}
        # Assemble parts
        df_parts = pd.DataFrame(data, index=[f"REF{i}" for i in range(n_windows)])
        ut.check_df_parts(df_parts=df_parts)
        return df_parts
