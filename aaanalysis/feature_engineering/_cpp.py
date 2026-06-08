"""
This is a script for the frontend of the CPP class, a sequence-based feature engineering object.
"""
import warnings
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

# Import supportive class (exception for importing from same sub-package)
from ._backend.cpp.sequence_feature import get_split_kws_
from ._backend.cpp.utils_feature import get_df_parts_
from ._backend.check_feature import (check_split_kws,
                                     check_parts_len, check_match_df_parts_split_kws,
                                     check_df_scales, check_df_cat, check_match_df_parts_df_scales,
                                     check_match_df_scales_df_cat, check_match_df_parts_features)
from ._backend.cpp_run import (
    cpp_run_single, cpp_run_batch, cpp_run_batch_num, cpp_run_sample_batched,
    _pick_feature_matrix_builder,
)
from ._backend.cpp._filters._get_feature_matrix_fast import AALookupCache
from ._backend.cpp.cpp_eval import evaluate_features
from ._backend.cpp._simplify import simplify_cpp_


# I Helper Functions
def _warn_gaps_encountered(df_parts=None, accept_gaps=False):
    """Warn when ``accept_gaps=True`` and a gap symbol is actually present.

    With ``accept_gaps=True`` the gap-symbol guard is bypassed and all-gap
    splits produce ``NaN`` feature values, which downstream scorers may skip
    or fail on. Surface a ``UserWarning`` only when a gap is truly encountered
    (not merely allowed)."""
    if not accept_gaps:
        return
    has_gap = any(df_parts[p].astype(str).str.contains(ut.STR_AA_GAP, regex=False).any()
                  for p in list(df_parts))
    if has_gap:
        warnings.warn(
            f"'accept_gaps' (True) encountered the gap symbol "
            f"('{ut.STR_AA_GAP}') in 'df_parts'; affected feature values may be "
            f"NaN. Ensure downstream scorers handle NaN.",
            UserWarning,
        )


def _finalize_run_output(df_feat=None, return_stats=False):
    """Shared tail for ``run`` / ``run_num``: stringify object columns, capture
    ``last_filter_stats_``, and optionally return the stats dict alongside."""
    for col in df_feat.select_dtypes(include=["object", "string"]).columns:
        df_feat[col] = df_feat[col].apply(
            lambda x: str(x) if isinstance(x, (np.str_, np.generic)) else x)
    stats = df_feat.attrs.get("last_filter_stats")
    if return_stats:
        return df_feat, stats
    return df_feat


def check_sample_in_df_seq(sample_name=None, df_seq=None):
    """Check if sample name in df_seq"""
    list_names = list(df_seq[ut.COL_NAME])
    if sample_name not in list_names:
        error = f"'sample_name' ('{sample_name}') not in '{ut.COL_NAME}' of 'df_seq'." \
                f"\nValid names are: {list_names}"
        raise ValueError(error)


def check_match_list_df_feat_list_df_parts(list_df_feat=None, list_df_parts=None):
    """Check if all elements in list are valid feature DataFrames"""
    for df_feat, df_parts in zip(list_df_feat, list_df_parts):
        ut.check_df_feat(df_feat=df_feat, list_parts=list(df_parts))


def check_match_max_interpretability_top_n(max_interpretability=None, top_n=None):
    """``simplify`` target selectors are mutually exclusive (at most one set)."""
    if max_interpretability is not None and top_n is not None:
        raise ValueError(
            f"'max_interpretability' ({max_interpretability}) and 'top_n' ({top_n}) are "
            f"mutually exclusive target selectors; set at most one (or neither to attempt "
            f"every improvable feature).")


def check_n_cv_labels(n_cv=None, labels=None):
    """Validate ``n_cv`` (>=2, <= smallest class count) for the simplify RF+CV gate."""
    ut.check_number_range(name="n_cv", val=n_cv, min_val=2, just_int=True)
    min_class_count = min(pd.Series(labels).value_counts())
    if n_cv > min_class_count:
        raise ValueError(f"'n_cv' ({n_cv}) should not be greater than the smallest class "
                         f"count ({min_class_count}).")


def check_match_list_df_feat_names_feature_sets(list_df_feat=None, names_feature_sets=None):
    """Check if length of list_df_feat and names match"""
    if names_feature_sets is None:
        return None # Skip check
    if len(list_df_feat) != len(names_feature_sets):
        raise ValueError(f"Length of 'list_df_feat' ({len(list_df_feat)}) and 'names_feature_sets'"
                         f" ({len(names_feature_sets)} does not match) ")


def _check_dict_num(df_seq=None, dict_num=None):
    """Validate ``dict_num`` shape/dtype contract."""
    if not isinstance(dict_num, dict):
        raise ValueError(
            f"'dict_num' ({type(dict_num).__name__}) should be a dict mapping entry to "
            f"np.ndarray of shape (L, D)."
        )
    entries = df_seq[ut.COL_ENTRY].to_list()
    missing = [e for e in entries if e not in dict_num]
    if missing:
        preview = missing[:5] + (["..."] if len(missing) > 5 else [])
        raise ValueError(
            f"'dict_num' ({len(missing)} missing entries) should contain every entry "
            f"in 'df_seq'. Missing: {preview}"
        )
    ds = []
    for entry in entries:
        emb = dict_num[entry]
        if not isinstance(emb, np.ndarray):
            raise ValueError(
                f"'dict_num[{entry!r}]' ({type(emb).__name__}) should be np.ndarray of shape (L, D)."
            )
        if emb.ndim != 2:
            raise ValueError(
                f"'dict_num[{entry!r}]' (ndim={emb.ndim}) should be 2D of shape (L, D)."
            )
        ds.append(emb.shape[1])
    if len(set(ds)) > 1:
        raise ValueError(
            f"'dict_num' (D values: {sorted(set(ds))}) should have a consistent embedding "
            f"dimensionality D across all entries."
        )


def _check_dict_num_df_scales_match(dict_num=None, df_scales=None):
    """When ``dict_num`` is supplied, ``df_scales`` columns name the D dimensions."""
    D = next(iter(dict_num.values())).shape[1]
    if len(df_scales.columns) != D:
        raise ValueError(
            f"'df_scales' (n_columns={len(df_scales.columns)}) should have D={D} columns "
            f"matching the embedding dimensionality of 'dict_num'."
        )


def check_match_df_seq_df_parts(df_seq=None, df_parts=None, jmd_n_len=None, jmd_c_len=None):
    """Verify ``df_seq`` derives parts equal to ``df_parts`` (CPP.run_num parity guard)."""
    list_parts = list(df_parts)
    try:
        df_parts_derived = get_df_parts_(
            df_seq=df_seq, list_parts=list_parts, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len
        )
    except Exception as e:
        raise ValueError(
            f"'df_seq' could not be turned into parts matching the CPP instance's "
            f"'df_parts' (jmd_n_len={jmd_n_len}, jmd_c_len={jmd_c_len}): {e}"
        ) from e
    if not df_parts_derived.equals(df_parts):
        raise ValueError(
            f"'df_seq' derives parts that disagree with the CPP instance's 'df_parts'. "
            f"For bit-identical parity with CPP.run, pass the same df_seq that produced "
            f"the constructor's df_parts (and the same jmd_n_len / jmd_c_len)."
        )


def _check_dict_num_parts(dict_num_parts=None, df_parts=None):
    """Layer-2 validation for CPP.run_num: ``dict_num_parts`` must align with ``self.df_parts``.

    Catches the cases where a user constructed ``dict_num_parts`` by hand (without
    going through ``NumericalFeature.get_parts``) or mixed parts from a different
    preprocessing run.
    """
    if not isinstance(dict_num_parts, dict):
        raise ValueError(
            f"'dict_num_parts' ({type(dict_num_parts).__name__}) should be a "
            f"Dict[part_name, np.ndarray] produced by NumericalFeature.get_parts(...)."
        )
    expected_parts = set(df_parts.columns)
    got_parts = set(dict_num_parts.keys())
    if got_parts != expected_parts:
        missing = sorted(expected_parts - got_parts)
        extra = sorted(got_parts - expected_parts)
        raise ValueError(
            f"'dict_num_parts' part names {sorted(got_parts)} should match CPP's "
            f"df_parts.columns {sorted(expected_parts)} "
            f"(missing: {missing}, extra: {extra}). Re-run NumericalFeature.get_parts(...) "
            f"with the same df_seq + jmd_n_len + jmd_c_len that produced the CPP's df_parts."
        )
    n_samples = len(df_parts)
    D_seen = set()
    for part in sorted(expected_parts):
        arr = dict_num_parts[part]
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"'dict_num_parts[{part!r}]' ({type(arr).__name__}) should be np.ndarray "
                f"of shape (n_samples, L_part_max, D)."
            )
        if arr.ndim != 3:
            raise ValueError(
                f"'dict_num_parts[{part!r}]' (ndim={arr.ndim}) should be 3D "
                f"(n_samples, L_part_max, D)."
            )
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"'dict_num_parts[{part!r}]' shape {arr.shape} should have "
                f"n_samples={n_samples} matching CPP's df_parts row count."
            )
        D_seen.add(arr.shape[2])
    if len(D_seen) > 1:
        raise ValueError(
            f"'dict_num_parts' has inconsistent D across parts: {sorted(D_seen)} — "
            f"all parts must share the same dimensionality."
        )
    if D_seen and 0 in D_seen:
        raise ValueError("'dict_num_parts' has D=0; should be >= 1.")


def _derive_dict_part_lens(df_parts=None):
    """Compute per-(entry, part) real residue count from ``df_parts`` strings.

    Tensor backends need per-sample real lengths to know where the NaN padding
    starts in each row. We re-derive them from the string ``df_parts`` (non-gap
    character count) rather than threading them through the user-facing API —
    keeps ``dict_num_parts`` a single-shape dict.
    """
    dict_part_lens = {}
    for part in df_parts.columns:
        col = df_parts[part].to_list()
        lens = np.fromiter(
            (sum(c != ut.STR_AA_GAP for c in s) for s in col),
            dtype=np.int64, count=len(col),
        )
        dict_part_lens[part] = lens
    return dict_part_lens


# II Main Functions
class CPP(Tool):
    """
    Comparative Physicochemical Profiling (**CPP**) class to create and filter features that are most discriminant
    between two sets of sequences [Breimann25a]_.

    CPP aims at identifying a set of non-redundant features that are most discriminant between the
    test and reference group of sequences.

    .. versionadded:: 0.1.0

    Attributes
    ----------
    df_parts
        DataFrame with sequence **Parts**.
    split_kws
        Nested dictionary defining **Splits** with parameter dictionary for each chosen split_type.
    df_scales
        DataFrame with amino acid **Scales**.
    df_cat
        DataFrame with categories for physicochemical amino acid **Scales**.
    """
    def __init__(self,
                 df_parts: pd.DataFrame = None,
                 split_kws: Optional[dict] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 accept_gaps: bool = False,
                 verbose: bool = True,
                 random_state: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts.
        split_kws : dict, optional
            Dictionary with parameter dictionary for each chosen split_type. Default from :meth:`SequenceFeature.get_split_kws`.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All scales from ``df_scales`` must be contained in ``df_cat``

        See Also
        --------
        * :class:`CPPPlot`: the respective plotting class.
        * :class:`SequenceFeature` for definition of sequence **Parts**.
        * :meth:`SequenceFeature.split_kws` for definition of **Splits** key word arguments.
        * :func:`load_scales` for definition of amino acid **Scales** and their categories.

        Examples
        --------
        .. include:: examples/cpp.rst
        """
        # Load defaults
        if split_kws is None:
            split_kws = get_split_kws_()
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        ut.check_df_parts(df_parts=df_parts)
        check_split_kws(split_kws=split_kws)
        check_df_scales(df_scales=df_scales)
        check_df_cat(df_cat=df_cat)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        df_parts = check_match_df_parts_df_scales(df_parts=df_parts, df_scales=df_scales, accept_gaps=accept_gaps)
        check_match_df_parts_split_kws(df_parts=df_parts, split_kws=split_kws)
        df_scales, df_cat = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        # Internal attributes
        self._accept_gaps = accept_gaps
        self._verbose = verbose
        self._random_state = random_state
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat.copy()
        self.df_scales = df_scales.copy()
        self.df_parts = df_parts.copy()
        self.split_kws = split_kws
        # Phase A.4: lazy cache for the AA-index + float64 scale_matrix used by
        # ``get_feature_matrix_fast_`` in ``CPP.run_num`` pass 2. Built on the
        # first call and reused across repeat calls on this instance.
        self._aa_lookup_cache = None
        # Filter-funnel counts from the most recent ``run`` / ``run_num``
        # (sklearn trailing-underscore convention: set during the call).
        self.last_filter_stats_ = None

    # Main method
    def run(self,
            labels: ut.ArrayLike1D = None,
            label_test: int = 1,
            label_ref: int = 0,
            n_filter: int = 100,
            n_pre_filter: Optional[int] = None,
            pct_pre_filter: int = 5,
            max_std_test: float = 0.2,
            max_overlap: float = 0.5,
            max_cor: float = 0.5,
            check_cat: bool = True,
            parametric: bool = False,
            start: int = 1,
            tmd_len: int = 20,
            jmd_n_len: int = 10,
            jmd_c_len: int = 10,
            n_jobs: Optional[int] = None,
            vectorized: bool = True,
            n_batches: Optional[int] = None,
            n_sample_batches: Optional[int] = None,
            return_stats: bool = False,
            ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
        """
        Perform Comparative Physicochemical Profiling (CPP) algorithm: creation and two-step filtering of
        interpretable sequence-based features.

        The aim of the CPP algorithm is to identify a set of unique, non-redundant features that are most
        discriminant between the test and reference group of sequences. See [Breimann25a]_ for details on the algorithm.

        .. versionchanged:: 1.1.0
            Added the ``return_stats`` parameter, returning the filter-funnel statistics alongside ``df_feat``.

        .. versionchanged:: 1.1.0
            Added the ``n_sample_batches`` parameter for sample-axis batching (memory bounded by batch size, not n).

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        n_filter : int, default=100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter : int, optional
            Number of feature to be pre-filtered by CPP algorithm. If ``None``, a percentage of all features is used.
        pct_pre_filter : int, default=5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test : float, default=0.2
            Maximum standard deviation [>0-<1] within the test group used as threshold for pre-filtering.
        max_overlap : float, default=0.5
            Maximum positional overlap [0-1] of features used as threshold for filtering.
        max_cor : float, default=0.5
            Maximum Pearson correlation [0-1] of feature scales used as threshold for filtering.
        check_cat : bool, default=True
            Whether to check for redundancy within scale categories during filtering.
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney U test) test for p-value computation.
            This also sets the p-value column name in ``df_feat`` ('p_val_ttest_indep' vs 'p_val_mann_whitney').
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        n_jobs : int, None, or -1, default=None
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

            .. warning::
               On Python 3.14 + macOS, calling this with ``n_jobs > 1`` (or ``-1`` /
               ``None``) from a script that lacks an ``if __name__ == "__main__":``
               guard (or from a bare REPL / heredoc) can trigger a recursive process
               spawn (``FileNotFoundError`` / ``EOFError`` / ``cannot pickle '_thread.RLock'``).
               Guard your entry point, or run serially with ``n_jobs=1``. See also
               :class:`CPPGrid` (default ``backend="threads"``), which sidesteps this.
        vectorized : bool, default=True
            Whether to apply sequence splitting and the Mann-Whitney U test in 'vectorized' mode (``True``),
            improving speed but increasing memory consumption.
        n_batches : int, None, default=None
            Number of batches (>=2) used for batch processing. If ``None``, single-processing is used, which is faster
            but more memory-intensive. Increasing ``n_batches`` (up to the maximum number of scales in ``df_scales``)
            reduces memory consumption but slows down processing.
        n_sample_batches : int, None, default=None
            Number of sample-axis batches (>=2, up to the number of samples) for sample-batched processing. If ``None``,
            sample-batching is disabled. Bounds peak memory by the batch size rather than the full sample count ``n``,
            so it is the option for very large ``n``. Mutually exclusive with ``n_batches`` (which batches over scales).
        return_stats : bool, default=False
            If ``True``, also return the filter-funnel statistics (``last_filter_stats_``)
            as a second element ``(df_feat, stats)``; if ``False``, return only ``df_feat``.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        * Pre-filtering can be adjusted by the following parameters: {'n_pre_filter', 'pct_pre_filter', 'max_std_test'}.
        * Filtering can be adjusted by the following parameters: {'n_filter', 'max_overlap', 'max_cor', 'check_cat'}.
        * **Cost** scales as ``O(n_scales x n_parts x n_splits)`` (the candidate feature count), so larger
          scale sets / wider ``split_kws`` are proportionally slower — budget a sweep accordingly, or use
          :class:`CPPGrid` (which runs CPP once per ``n_filter`` group and slices the rest).
        * **Classifier head tracks the metric** when training a downstream model on ``df_feat``: in practice
          SVM tends to be best for AP (ranking), logistic regression for balanced accuracy, and random forest
          for MCC at a fixed threshold (detection). Pick the head to match the objective you report.
        * For large datasets (due to long sequences or a high number of samples) or memory-limited systems,
          memory consumption can be reduced by:

          - Disabling vectorized mode (``vectorized=False``)
          - Reducing ``n_jobs`` (down to ``n_jobs=1``)
          - Using batch processing (``n_batches>=2``, with higher values reducing memory usage)

          While this helps to prevent crashes, it may slow down processing.

        * ``df_feat`` follows a **standardized, deterministic column order** (the
          canonical schema), with the unique feature id (1), scale information (2-5),
          statistical results for filtering and ranking (6-12), and feature positions (13):

            1. 'feature': Feature ID (PART-SPLIT-SCALE)
            2. 'category': Scale category
            3. 'subcategory': Sub category of scales
            4. 'scale_name': Name of scales
            5. 'scale_description': Description of the scale
            6. 'abs_auc': Absolute adjusted AUC (area under the curve) [-0.5 to 0.5]
            7. 'abs_mean_dif': Absolute mean differences between test and reference group [0 to 1]
            8. 'mean_dif': Mean differences between test and reference group [-1 to 1]
            9. 'std_test': Standard deviation in test group
            10. 'std_ref': Standard deviation in reference group
            11. 'p_val_mann_whitney' or 'p_val_ttest_indep': p-value of the non-parametric
                Mann-Whitney test (default) or, when ``parametric=True``, the independent
                t-test. The column **name** reflects which test was run.
            12. 'p_val_fdr_bh': Benjamini-Hochberg False Discovery Rate (FDR) corrected p-values
            13. 'positions': Feature positions for default settings

          The feature id (column 1) is an opaque ``PART-SPLIT-SCALE`` string; split it with
          :func:`aaanalysis.utils.split_feat_id` rather than parsing it by hand. Columns added
          downstream — the explainable-AI columns ('feat_importance', 'feat_impact') and the
          per-substrate SHAP columns ('feat_impact_<name>', 'mean_dif_<name>', ...) added by
          :class:`TreeModel` / :class:`ShapModel` — are appended after 'positions' in a stable
          order, so the canonical order is a lower bound, never a restriction.

        * **Compositional vs positional features** are not a separate setting — the distinction
          emerges from ``split_kws``. A single whole-part average (``n_split_max=1`` with no
          ``Pattern`` / ``PeriodicPattern``) yields **compositional** features (an
          amino-acid-composition-like mean over the entire part, position-agnostic); using
          ``n_split_max>1`` and/or patterns yields **positional** features resolved to specific
          sub-regions.

        See Also
        --------
        * :func:`comp_auc_adjusted` for details on ``abs_auc``.

        Examples
        --------
        .. include:: examples/cpp_run.rst
        """
        # Check input
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                                 len_required=len(self.df_parts), allow_other_vals=False)
        ut.check_number_range(name="n_filter", val=n_filter, min_val=1, just_int=True)
        ut.check_number_range(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True, just_int=True)
        ut.check_number_range(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100, just_int=True)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0,
                              just_int=False, exclusive_limits=True)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="check_cat", val=check_cat)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        ut.check_bool(name="vectorized", val=vectorized)
        ut.check_bool(name="return_stats", val=return_stats)
        n_scales = len(list(self.df_scales))
        ut.check_number_range(name="n_batches", val=n_batches, just_int=True,
                              accept_none=True, min_val=2, max_val=n_scales)
        ut.check_number_range(name="n_sample_batches", val=n_sample_batches, just_int=True,
                              accept_none=True, min_val=2, max_val=len(self.df_parts))
        if n_batches is not None and n_sample_batches is not None:
            raise ValueError(f"'n_batches' ({n_batches}) and 'n_sample_batches' ({n_sample_batches}) "
                             f"should not be set together; choose scale-batching or sample-batching, not both.")
        _warn_gaps_encountered(df_parts=self.df_parts, accept_gaps=self._accept_gaps)
        # Route through the unified CPP pipeline + Cython kernel
        # (with Python fallback) — bit-exact, guaranteed by
        # ``test_run_num_parity``.
        args = dict(df_parts=self.df_parts, split_kws=self.split_kws,
                    df_scales=self.df_scales, df_cat=self.df_cat,
                    verbose=self._verbose, accept_gaps=self._accept_gaps,
                    labels=labels, label_test=label_test, label_ref=label_ref,
                    n_filter=n_filter, n_pre_filter=n_pre_filter, pct_pre_filter=pct_pre_filter,
                    max_std_test=max_std_test, max_overlap=max_overlap, max_cor=max_cor,
                    check_cat=check_cat, parametric=parametric,
                    start=start, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                    n_jobs=n_jobs, vectorized=vectorized)
        builder = _pick_feature_matrix_builder()
        if self._aa_lookup_cache is None:
            self._aa_lookup_cache = AALookupCache.from_df(
                df_parts=self.df_parts, df_scales=self.df_scales,
            )
        if n_sample_batches is not None:
            df_feat = cpp_run_sample_batched(
                **args, n_sample_batches=n_sample_batches,
            )
        elif n_batches is None:
            df_feat = cpp_run_single(
                df_seq=None, dict_num=None,
                aa_lookup_cache=self._aa_lookup_cache,
                feature_matrix_builder=builder,
                **args,
            )
        else:
            df_feat = cpp_run_batch(
                **args, n_batches=n_batches,
                feature_matrix_builder=builder,
            )
        self.last_filter_stats_ = df_feat.attrs.get("last_filter_stats")
        return _finalize_run_output(df_feat=df_feat, return_stats=return_stats)

    def run_num(self,
                dict_num_parts: Dict[str, np.ndarray] = None,
                labels: ut.ArrayLike1D = None,
                label_test: int = 1,
                label_ref: int = 0,
                n_filter: int = 100,
                n_pre_filter: Optional[int] = None,
                pct_pre_filter: int = 5,
                max_std_test: float = 0.2,
                max_overlap: float = 0.5,
                max_cor: float = 0.5,
                check_cat: bool = True,
                parametric: bool = False,
                start: int = 1,
                tmd_len: int = 20,
                jmd_n_len: int = 10,
                jmd_c_len: int = 10,
                n_jobs: Optional[int] = None,
                vectorized: bool = True,
                n_batches: Optional[int] = None,
                return_stats: bool = False,
                ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
        """
        Numerical-mode Comparative Physicochemical Profiling (CPP): same algorithm as
        :meth:`run`, but per-residue values come from a pre-sliced numerical tensor
        (`dict_num_parts`) instead of an AA→scale lookup. Use for PLM embeddings, DSSP
        one-hots, PTM dummies, or any per-residue numerical representation.

        Same pipeline (pre-filter stats, pre-filter, recompute, add_stat, redundancy
        filter) and same output schema as :meth:`run`. The constructor-bound
        ``df_scales`` / ``df_cat`` provide DIMENSION NAMES + categories for the D axis
        of ``dict_num_parts`` (the per-AA values they would normally provide are
        unused — ``dict_num_parts`` is the value source).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        dict_num_parts : dict[str, np.ndarray], required
            Per-part NaN-padded numerical tensors, produced by
            :meth:`NumericalFeature.get_parts`. Each value has shape
            ``(n_samples, L_part_max, D)`` aligned row-for-row with
            ``self.df_parts``. Keys must match ``self.df_parts.columns``. ``D`` must
            equal ``len(self.df_scales.columns)`` (each D dimension names a "scale").
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1
            Class label of test group in ``labels``.
        label_ref : int, default=0
            Class label of reference group in ``labels``.
        n_filter : int, default=100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter : int, optional
            Number of features to be pre-filtered. If ``None``, a percentage of all features is used.
        pct_pre_filter : int, default=5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test : float, default=0.2
            Maximum standard deviation [>0-<1] within the test group used as threshold for pre-filtering.
        max_overlap : float, default=0.5
            Maximum positional overlap [0-1] of features used as threshold for filtering.
        max_cor : float, default=0.5
            Maximum Pearson correlation [0-1] of feature scales used as threshold for filtering.
        check_cat : bool, default=True
            Whether to check for redundancy within scale categories during filtering.
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney U test) for p-value computation.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of target middle domain (TMD) (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        n_jobs : int, None, or -1, default=None
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized
            automatically; if ``-1``, all available cores are used. Overridden by ``options['n_jobs']``
            when set. The Python 3.14 + macOS spawn caveat documented in :meth:`run` applies here too.
        vectorized : bool, default=True
            Whether to apply sequence splitting and the Mann-Whitney U test in 'vectorized' mode (``True``),
            improving speed but increasing memory consumption.
        n_batches : int, None, default=None
            Number of batches (2 to ``len(df_scales.columns)``) over the D axis of
            ``dict_num_parts``. If ``None``, single-pass (faster, higher peak memory);
            a value bounds the pass-1 stat working set to one D-chunk. Output is
            bit-exact with the single-pass result.
        return_stats : bool, default=False
            If ``True``, also return the filter-funnel statistics (``last_filter_stats_``) as a second
            element ``(df_feat, stats)``; if ``False``, return only ``df_feat``.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Same schema as :meth:`run`.

        Raises
        ------
        ValueError
            If ``dict_num_parts`` is ``None`` (use :meth:`run` for seq-mode), or if
            its shape / part names / D don't align with the constructor's
            ``self.df_parts`` and ``self.df_scales``.

        Notes
        -----
        * **Raw PLM embeddings are not directly usable — normalize them first.**
          Per-residue values are expected in ``[0, 1]`` (the ``StructurePreprocessor`` /
          ``AnnotationPreprocessor`` normalization convention), since the default
          ``max_std_test=0.2`` pre-filter is calibrated for that range. Raw embeddings
          (unbounded floats) must be passed through
          :meth:`EmbeddingPreprocessor.encode` to obtain a ``[0, 1]``-normalized
          ``{entry: (L, D)}`` ``dict_num`` before :meth:`NumericalFeature.get_parts`.
          (``EmbeddingPreprocessor.build_scales`` / ``build_cat`` serve the *other*,
          AA-scale path via :meth:`run`; they are not a per-residue value source here.)
        * **Three arms, one entry point.** *structure-only* (``dict_num`` from
          :class:`StructurePreprocessor`), *embedding* (``EmbeddingPreprocessor.encode``),
          and *fused* (concatenate sources with :func:`aaanalysis.combine_dict_nums` first)
          all flow through ``get_parts`` → ``run_num`` — only the ``dict_num`` differs.
        * **Compositional vs positional** features emerge from ``split_kws`` exactly as in
          :meth:`run` (``n_split_max=1`` with no patterns ⇒ compositional whole-part mean;
          otherwise positional).

        See Also
        --------
        * :meth:`run`: sequence-mode equivalent (no ``dict_num_parts``).
        * :meth:`NumericalFeature.get_parts`: produces ``(df_parts, dict_num_parts)``
          from raw ``df_seq + dict_num``.
        * :class:`EmbeddingPreprocessor`, :class:`StructurePreprocessor`,
          :class:`AnnotationPreprocessor`: the per-residue ``dict_num`` sources
          (PLM embeddings / structure / annotations), combinable via
          :func:`aaanalysis.combine_dict_nums`.

        Examples
        --------
        .. include:: examples/cpp_run.rst
        """
        # Validate
        if dict_num_parts is None:
            raise ValueError(
                "'dict_num_parts' (None) is required. Use CPP.run() for seq-mode, or call "
                "NumericalFeature.get_parts(df_seq, dict_num, ...) to produce "
                "dict_num_parts from a per-residue tensor input."
            )
        _check_dict_num_parts(dict_num_parts=dict_num_parts, df_parts=self.df_parts)
        # D must match the constructor's df_scales (it names the D dimensions).
        D = next(iter(dict_num_parts.values())).shape[2]
        n_df_scales_cols = len(self.df_scales.columns)
        if D != n_df_scales_cols:
            raise ValueError(
                f"'dict_num_parts' D={D} should equal len(self.df_scales.columns)="
                f"{n_df_scales_cols}. df_scales names the D dimensions in numerical mode."
            )
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                                 len_required=len(self.df_parts), allow_other_vals=False)
        ut.check_number_range(name="n_filter", val=n_filter, min_val=1, just_int=True)
        ut.check_number_range(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True, just_int=True)
        ut.check_number_range(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100, just_int=True)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0,
                              just_int=False, exclusive_limits=True)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="check_cat", val=check_cat)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        ut.check_bool(name="vectorized", val=vectorized)
        ut.check_bool(name="return_stats", val=return_stats)
        ut.check_number_range(name="n_batches", val=n_batches, just_int=True,
                              accept_none=True, min_val=2, max_val=n_df_scales_cols)

        # Re-derive per-(entry, part) real lengths from df_parts non-gap chars.
        # The user-facing dict_num_parts carries only the NaN-padded tensor; lens
        # are deterministic from the string df_parts (keeps the API one-shape).
        dict_part_lens = _derive_dict_part_lens(df_parts=self.df_parts)

        args = dict(
            df_parts=self.df_parts, split_kws=self.split_kws,
            df_scales=self.df_scales, df_cat=self.df_cat,
            verbose=self._verbose, accept_gaps=self._accept_gaps,
            labels=labels, label_test=label_test, label_ref=label_ref,
            n_filter=n_filter, n_pre_filter=n_pre_filter, pct_pre_filter=pct_pre_filter,
            max_std_test=max_std_test, max_overlap=max_overlap, max_cor=max_cor,
            check_cat=check_cat, parametric=parametric,
            start=start, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
            n_jobs=n_jobs, vectorized=vectorized,
        )

        # Cython by default; falls back to pure-Python kernel if the compiled `.so`
        # isn't present (e.g. user installed without the wheel). Same auto-dispatch
        # as cpp.run — see _pick_feature_matrix_builder docstring.
        builder = _pick_feature_matrix_builder()
        if n_batches is None:
            df_feat = cpp_run_single(
                df_seq=None, dict_num=None,
                dict_part_vals=dict_num_parts, dict_part_lens=dict_part_lens,
                aa_lookup_cache=None,            # numerical path doesn't use AA lookup
                feature_matrix_builder=builder,  # consumed by seq-mode only; harmless here
                **args,
            )
        else:
            # Batch the D axis (the dict_num_parts dimensions). Bit-exact with the
            # single-pass path — see cpp_run_batch_num.
            df_feat = cpp_run_batch_num(
                dict_part_vals=dict_num_parts, dict_part_lens=dict_part_lens,
                n_batches=n_batches, **args,
            )
        self.last_filter_stats_ = df_feat.attrs.get("last_filter_stats")
        return _finalize_run_output(df_feat=df_feat, return_stats=return_stats)

    def eval(self,
             list_df_feat: List[pd.DataFrame] = None,
             labels: ut.ArrayLike1D = None,
             label_test: int = 1,
             label_ref: int = 0,
             min_th: float = 0.0,
             names_feature_sets: Optional[List[str]] = None,
             list_cat: Optional[List[str]] = None,
             list_df_parts: Optional[List[pd.DataFrame]] = None,
             n_jobs: Optional[int] = 1,
             ) -> pd.DataFrame:
        """
        Evaluate the quality of different sets of identified Comparative Physicochemical
        Profiling (CPP) features.

        Feature sets are evaluated regarding two quality groups:

        - **Discriminative Power**: The capability of features to distinguish between test and reference datasets.
        - **Redundancy**: Assessed by the optimized number of clusters, based on Pearson correlation among features.

        Parameters
        ----------
        list_df_feat : list of pd.DataFrames
            List of feature DataFrames each of shape (n_features, n_feature_info)
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        min_th : float, default=0.0
            Pearson correlation threshold for clustering optimization (between -1 and 1).
        names_feature_sets : list of str, optional
            List of names for feature sets corresponding to ``list_df_feat``.
        list_cat : list of str, optional
            List of scale categories to retrieve number of features from. Default:
            ['ASA/Volume', 'Composition', 'Conformation', 'Energy', 'Others', 'Polarity', 'Shape', 'Structure-Activity']
        list_df_parts : list of pd.DataFrames, optional
            List of part DataFrames each of shape (n_samples, n_parts). Must match with ``list_df_feat``.
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores. Overridden by ``options['n_jobs']`` when set.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for each set of identified features. For each set, statistical
            measures were averaged across all features.

        Notes
        -----
        * ``df_eval`` includes the following columns (upper-case indicates direct reference to ``df_feat`` columns):

            - 'name': Name of the feature set, typically based on CPP run settings, if ``names`` is provided.
            - 'n_features': Tuple with total number of features and list of number of features per scale category from ``list_cat``.
            - 'avg_ABS_AUC': Absolute Area Under the Curve (AUC) averaged across all features.
            - 'range_ABS_AUC': Quintile range of absolute AUC among all features (min, 25%, median, 75%, max).
            - 'avg_MEAN_DIF': Tuple of mean differences averaged across all features separately
              for features with positive and negative 'mean_dif'.
            - 'n_clusters': Optimal number of clusters [2,100].
            - 'avg_n_feat_per_clust': Average number of features per cluster.
            - 'std_n_feat_per_clust': Standard deviation of feature number per cluster.

        * 'n_clusters' is optimized for a KMeans clustering model based on the minimum Pearson correlation between
          the cluster center and all cluster members across all clusters (``min_cor_center`` in :class:`AAclust`),
          which has to exceed the minimum correlation threshold ``min_th``.

        See Also
        --------
        * :func:`CPPPlot.eval`: the respective plotting method.
        * :ref:`usage_principles_aaontology` for details on scale categories.
        * :meth:`CPP.run` for details on CPP statistical measures.
        * :func:`comp_auc_adjusted` for details on 'abs_auc'.
        * :class:`sklearn.cluster.KMeans` for employed clustering model.
        * :class:`AAclust` ([Breimann24a]_) for details on cluster optimization using Pearson correlation.

        Examples
        --------
        .. include:: examples/cpp_eval.rst
        """
        # Check input
        list_df_feat = ut.check_list_like(name="list_df_feat", val=list_df_feat, min_len=2)
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                                 len_required=len(self.df_parts), allow_other_vals=False)
        ut.check_number_range(name="min_th", val=min_th, min_val=-1, max_val=1, just_int=False)
        names_feature_sets = ut.check_list_like(name="names_feature_sets", val=names_feature_sets, accept_none=True,
                                                accept_str=True, check_all_str_or_convertible=True)
        list_cat = ut.check_list_like(name="list_cat", val=list_cat, accept_none=True, accept_str=True,
                                      check_all_str_or_convertible=True)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        check_match_list_df_feat_names_feature_sets(list_df_feat=list_df_feat,
                                                    names_feature_sets=names_feature_sets)
        list_df_parts = ut.check_list_like(name="list_df_parts", val=list_df_parts, accept_none=True)
        mask_test = [x == label_test for x in labels]
        if list_df_parts is None:
            list_df_parts = [self.df_parts[mask_test]] * len(list_df_feat)
        if list_cat is None:
            list_cat = ut.LIST_CAT
        check_match_list_df_feat_list_df_parts(list_df_feat=list_df_feat, list_df_parts=list_df_parts)
        df_eval = evaluate_features(list_df_feat=list_df_feat,
                                    names_feature_sets=names_feature_sets,
                                    list_cat=list_cat,
                                    list_df_parts=list_df_parts,
                                    df_scales=self.df_scales,
                                    accept_gaps=self._accept_gaps,
                                    min_th=min_th,
                                    n_jobs=n_jobs,
                                    random_state=self._random_state)
        return df_eval

    def simplify(self,
                 df_feat: pd.DataFrame = None,
                 labels: ut.ArrayLike1D = None,
                 X: Optional[ut.ArrayLike2D] = None,
                 strategy: str = "greedy",
                 max_interpretability: Optional[int] = None,
                 top_n: Optional[int] = None,
                 min_cor: float = 0.7,
                 metric: str = "balanced_accuracy",
                 tol: float = 0.0,
                 n_cv: int = 5,
                 on_unimprovable: str = "keep",
                 redundancy_tie_break: str = "interpretability",
                 label_test: int = 1,
                 label_ref: int = 0,
                 max_std_test: float = 0.2,
                 max_cor: float = 0.5,
                 max_overlap: float = 0.5,
                 check_cat: bool = True,
                 return_details: bool = False,
                 random_state: Optional[int] = None,
                 ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Simplify a feature set by swapping scales for more interpretable correlated ones.

        For each feature (``PART-SPLIT-SCALE``), an alternative scale from a **more
        interpretable AAontology subcategory** (interpretability rating 1-10, 1 = best;
        see :func:`load_scales` ``top_explain_n`` and ADR-0025) that **correlates** with the
        original scale is substituted, keeping ``PART-SPLIT``. The swapped feature's statistics
        are recomputed; a swap is accepted only if it passes CPP's per-feature filtering
        (``max_std_test``) and a random-forest cross-validation gate (performance not worse than
        the current set, within ``tol``). The swapped set is then redundancy-reduced, yielding a
        more interpretable and ideally smaller ``df_feat``. The candidate pool (the full rated
        AAontology scale set) is loaded internally.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame from :meth:`run` (the standardized CPP output schema).
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        X : array-like, shape (n_samples, n_features), optional
            Feature matrix matching ``df_feat`` (baseline only; row-aligned to ``self.df_parts``).
            If ``None``, it is recomputed internally. Swapped columns are always recomputed.
        strategy : str, default='greedy'
            Simplification strategy. ``'greedy'`` swaps feature-by-feature behind the RF+CV gate.
            (``'consolidate'`` / ``'swap_all'`` are reserved for a future release.)
        max_interpretability : int, optional
            Interpretability ceiling (1-10): every feature whose scale subcategory is rated
            worse (higher) than this is targeted for replacement. Mutually exclusive with ``top_n``.
        top_n : int, optional
            Target the ``top_n`` worst-interpretability features. Mutually exclusive with
            ``max_interpretability``. If both are ``None``, every improvable feature is attempted.
        min_cor : float, default=0.7
            Minimum absolute Pearson correlation between a candidate scale and the original scale
            (between 0 and 1); anti-correlation is allowed via the absolute value.
        metric : str, default='balanced_accuracy'
            Scoring metric for the RF+CV gate (any scikit-learn classification scorer name).
        tol : float, default=0.0
            A swap is accepted if its CV score is at least ``baseline - tol`` (>=0).
        n_cv : int, default=5
            Number of cross-validation folds (>=2, <= smallest class count).
        on_unimprovable : str, default='keep'
            What to do with a targeted feature that cannot be improved: ``'keep'`` (retain the
            original), ``'drop'`` (remove it), or ``'drop_if_perf_allows'`` (remove only if the
            CV score does not drop). The last feature is never dropped.
        redundancy_tie_break : str, default='interpretability'
            When two swapped features are redundant, keep the ``'interpretability'``-best (then
            ``abs_auc``) or the ``'performance'``-best (``abs_auc``).
        label_test : int, default=1
            Class label of the test group in ``labels``.
        label_ref : int, default=0
            Class label of the reference group in ``labels``.
        max_std_test : float, default=0.2
            Per-feature pre-filter threshold a swapped feature must satisfy (between 0 and 1).
        max_cor : float, default=0.5
            Redundancy correlation threshold for the post-swap reduction (between 0 and 1).
        max_overlap : float, default=0.5
            Redundancy position-overlap threshold for the post-swap reduction (between 0 and 1).
        check_cat : bool, default=True
            Whether the redundancy reduction only compares features within the same scale category.
        return_details : bool, default=False
            If ``True``, also return a long-form ``df_candidates`` reporting every candidate
            considered (scale, interpretability, correlation, recomputed std, accepted-flag).
        random_state : int, optional
            Random state for the random forest. Overrides the constructor's ``random_state``.

        Returns
        -------
        df_feat : pd.DataFrame
            The simplified feature DataFrame (CPP output schema), with swapped scales, recomputed
            statistics, and redundant features removed.
        df_candidates : pd.DataFrame
            Returned only if ``return_details=True``: one row per candidate considered.

        Notes
        -----
        * Features whose scale is **not a rated AAontology scale** (e.g. ``run_num`` pseudo-scales
          or unclassified scales) carry no interpretability rating and are skipped. If no feature
          is rated, ``df_feat`` is returned unchanged with a ``RuntimeWarning``.
        * An anti-correlated swap flips the sign of ``mean_dif`` (the feature still discriminates);
          the correlation sign is reported in ``df_candidates``.

        See Also
        --------
        * :meth:`run` for the feature DataFrame produced and its schema.
        * :func:`load_scales` for the interpretability-tiered explainable scale sets (``top_explain_n``).

        Examples
        --------
        .. include:: examples/cpp_simplify.rst
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat, list_parts=list(self.df_parts))
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_required=[label_test, label_ref],
                                 len_required=len(self.df_parts), allow_other_vals=False)
        check_match_df_parts_features(df_parts=self.df_parts, features=list(df_feat[ut.COL_FEATURE]))
        ut.check_str_options(name="strategy", val=strategy, list_str_options=ut.LIST_SIMPLIFY_STRATEGIES)
        ut.check_str_options(name="on_unimprovable", val=on_unimprovable,
                             list_str_options=ut.LIST_ON_UNIMPROVABLE)
        ut.check_str_options(name="redundancy_tie_break", val=redundancy_tie_break,
                             list_str_options=ut.LIST_REDUNDANCY_TIE_BREAK)
        ut.check_str(name="metric", val=metric)
        ut.check_number_range(name="min_cor", val=min_cor, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0, max_val=1, just_int=False)
        ut.check_number_val(name="tol", val=tol, just_int=False)
        ut.check_number_range(name="max_interpretability", val=max_interpretability, min_val=1,
                              max_val=10, just_int=True, accept_none=True)
        ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True, accept_none=True)
        ut.check_bool(name="check_cat", val=check_cat)
        ut.check_bool(name="return_details", val=return_details)
        check_match_max_interpretability_top_n(max_interpretability=max_interpretability, top_n=top_n)
        check_n_cv_labels(n_cv=n_cv, labels=labels)
        if X is not None:
            X = ut.check_X(X=X, min_n_features=1)
        random_state = self._random_state if random_state is None else random_state
        random_state = ut.check_random_state(random_state=random_state)
        # The recomputed p-value column must match the input df_feat's test choice.
        parametric = ut.COL_PVAL_TTEST in list(df_feat)
        _warn_gaps_encountered(df_parts=self.df_parts, accept_gaps=self._accept_gaps)
        return simplify_cpp_(df_feat=df_feat, df_parts=self.df_parts, df_scales_self=self.df_scales,
                             labels=labels, X=X, strategy=strategy,
                             max_interpretability=max_interpretability, top_n=top_n, min_cor=min_cor,
                             metric=metric, tol=tol, n_cv=n_cv, on_unimprovable=on_unimprovable,
                             redundancy_tie_break=redundancy_tie_break, label_test=label_test,
                             label_ref=label_ref, max_std_test=max_std_test, max_cor=max_cor,
                             max_overlap=max_overlap, check_cat=check_cat, parametric=parametric,
                             accept_gaps=self._accept_gaps, return_details=return_details,
                             random_state=random_state)
