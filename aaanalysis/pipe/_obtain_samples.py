"""
This is a script for the frontend of the ``aaanalysis.pipe`` (aap) ``obtain_samples`` golden
pipeline: a thin, stateless one-call wrapper that turns a described sampling situation into a
balanced, labeled training set (plus a quick validation report) over the existing AAanalysis
primitives (:class:`AAWindowSampler`, optionally :class:`dPULearn`).
"""
from typing import Optional, List, Tuple
import warnings
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import aaanalysis.utils as ut
from aaanalysis.seq_analysis import AAWindowSampler, AAlogo, AAlogoPlot
from aaanalysis.feature_engineering import SequenceFeature
from aaanalysis.pu_learning import dPULearn


# I Helper Functions
# Situations the user can describe; each maps to one AAWindowSampler.sample_* call.
LIST_SITUATIONS = [ut.STRATEGY_SAME, ut.STRATEGY_DIFF, ut.STRATEGY_SYNTH_PREFIX]
# Default JMD flank length of SequenceFeature.get_df_parts, used to build X on the PU path.
JMD_FLANK_LEN = 10
# On the PU path, sample this many times n_neg unlabeled windows so dPULearn has a pool to
# select the n_neg most-reliable negatives from (rather than relabeling the whole pool).
UNLABELED_POOL_FACTOR = 4


def check_match_reliable_negatives_df_feat(reliable_negatives, df_feat, strategy):
    """The reliable-negatives (PU) path needs a ``df_feat`` and the cross-protein unlabeled pool."""
    if not reliable_negatives:
        return
    if df_feat is None:
        raise ValueError("'df_feat' (None) should be a feature DataFrame when "
                         "'reliable_negatives' is True: dPULearn builds the feature matrix 'X' "
                         "from it to identify reliable negatives.")
    if strategy != ut.STRATEGY_DIFF:
        raise ValueError(f"'strategy' ('{strategy}') should be '{ut.STRATEGY_DIFF}' when "
                         f"'reliable_negatives' is True: the reliable-negatives path treats the "
                         f"cross-protein pool as the unlabeled set 'U'.")


def _coerce_positions(val) -> List[int]:
    """Coerce one ``pos_col`` cell into a list of 1-based int positions (empty if missing)."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return [int(p) for p in val if not (isinstance(p, float) and np.isnan(p))]
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return []
        return [int(tok) for tok in text.replace(";", ",").split(",") if tok.strip()]
    if isinstance(val, float) and np.isnan(val):
        return []
    return [int(val)]


def _positive_segments(df_seq, pos_col, window_size) -> pd.DataFrame:
    """Build the positive test-window segments at each P1 anchor in ``pos_col``.

    Uses the same P1 anchoring as :class:`AAWindowSampler` (``half_left = (window_size - 1)
    // 2`` residues upstream of the anchor); windows that do not fully fit the sequence are
    dropped, matching the sampler's test-window extraction.
    """
    half_left = (window_size - 1) // 2
    rows = []
    for entry, seq, val in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ], df_seq[pos_col]):
        for p in _coerce_positions(val):
            start = (p - 1) - half_left
            stop = start + window_size
            if 0 <= start and stop <= len(seq):
                window = seq[start:stop]
                entry_win = f"{entry}_{start + 1}-{stop}"
                rows.append([entry_win, entry, seq, window, p, 1,
                             ut.ROLE_TEST, ut.STRATEGY_TEST])
    return pd.DataFrame(rows, columns=ut.COLS_SEGMENTS)


def _sample_negatives(aaws, df_seq, strategy, n_neg, window_size, pos_col, seed) -> pd.DataFrame:
    """Draw ``n_neg`` reference windows via the AAWindowSampler call the situation maps to."""
    if strategy == ut.STRATEGY_SAME:
        return aaws.sample_same_protein(df_seq=df_seq, n=n_neg, window_size=window_size,
                                        pos_col=pos_col, seed=seed)
    if strategy == ut.STRATEGY_DIFF:
        return aaws.sample_different_protein(df_seq=df_seq, n=n_neg, window_size=window_size,
                                             pos_col=pos_col, seed=seed)
    return aaws.sample_synthetic(df_seq=df_seq, n=n_neg, window_size=window_size, seed=seed)


def _window_identity(a: str, b: str) -> float:
    """Per-position residue identity of two equal-length windows (0.0 if lengths differ)."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / n


def _max_identity_to_test(neg_windows, test_windows) -> float:
    """Maximum identity of any sampled window to any test window (the leakage indicator)."""
    if len(neg_windows) == 0 or len(test_windows) == 0:
        return float("nan")
    return max(_window_identity(w, t) for w in neg_windows for t in test_windows)


def _fits_part_geometry(df, window_size) -> np.ndarray:
    """Mask of rows whose P1 anchor leaves room for the TMD + default JMD flanks.

    The PU feature matrix anchors each window as a length-``window_size`` TMD with the default
    ``jmd_n``/``jmd_c`` flanks (:data:`JMD_FLANK_LEN`); windows too close to a terminus to admit
    those flanks cannot be featurized and are excluded from the dPULearn step.
    """
    half_left = (window_size - 1) // 2
    fits = []
    for seq, p in zip(df[ut.COL_SEQ], df[ut.COL_SOURCE_POS]):
        start = (p - 1) - half_left
        fits.append(start - JMD_FLANK_LEN >= 0
                    and start + window_size + JMD_FLANK_LEN <= len(seq))
    return np.asarray(fits, dtype=bool)


def _reliable_negatives(df_pos, df_unl, df_feat, window_size, n_neg,
                        seed, n_jobs, verbose) -> pd.DataFrame:
    """Refine an unlabeled window pool into reliable negatives with :class:`dPULearn`.

    Builds the feature matrix ``X`` for the positives + unlabeled windows from the features in
    ``df_feat`` (each window anchored as a length-``window_size`` TMD with the default JMD flanks
    via :meth:`SequenceFeature.get_df_parts`), runs dPULearn, and keeps the windows it relabels 0.
    Windows too close to a terminus to admit the JMD flanks are skipped with a warning.
    """
    pos_fits, unl_fits = _fits_part_geometry(df_pos, window_size), _fits_part_geometry(df_unl, window_size)
    n_dropped = int((~pos_fits).sum() + (~unl_fits).sum())
    if n_dropped:
        warnings.warn(f"{n_dropped} window(s) lack the flanking residues needed to build the "
                      f"feature matrix and are skipped on the reliable-negatives path; widen the "
                      f"sequences or lower 'window_size' to keep them.",
                      UserWarning)
    df_pos_fit, df_unl_fit = df_pos[pos_fits], df_unl[unl_fits]
    if len(df_pos_fit) == 0 or len(df_unl_fit) == 0:
        raise ValueError("'reliable_negatives' (True with 0 featurizable positive or unlabeled "
                         "windows — all anchors lack flanking residues) should have enough "
                         "flanking room; widen the sequences, lower 'window_size', or set "
                         "'reliable_negatives=False'.")
    df_anchor = pd.DataFrame({
        ut.COL_ENTRY: list(df_pos_fit[ut.COL_ENTRY_WIN]) + list(df_unl_fit[ut.COL_ENTRY_WIN]),
        ut.COL_SEQ: list(df_pos_fit[ut.COL_SEQ]) + list(df_unl_fit[ut.COL_SEQ]),
        ut.COL_POS: list(df_pos_fit[ut.COL_SOURCE_POS]) + list(df_unl_fit[ut.COL_SOURCE_POS]),
    })
    sf = SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_anchor, tmd_len=window_size)
    X = sf.feature_matrix(features=df_feat[ut.COL_FEATURE], df_parts=df_parts, n_jobs=n_jobs)
    labels = [1] * len(df_pos_fit) + [2] * len(df_unl_fit)
    # Cannot identify more reliable negatives than the featurizable unlabeled pool holds.
    n_to_identify = min(n_neg, len(df_unl_fit))
    if n_to_identify < n_neg:
        warnings.warn(f"only {n_to_identify} featurizable unlabeled window(s) available; "
                      f"identifying {n_to_identify} reliable negatives instead of {n_neg}.",
                      RuntimeWarning)
    dpul = dPULearn(verbose=verbose, random_state=seed)
    dpul.fit(X, labels=labels, label_pos=1, label_unl=2, n_unl_to_neg=n_to_identify)
    is_neg = np.asarray(dpul.labels_[len(df_pos_fit):]) == 0
    df_neg = df_unl_fit[is_neg].copy()
    df_neg[ut.COL_LABEL] = 0
    df_neg[ut.COL_ROLE] = ut.ROLE_NEG
    return df_neg[ut.COLS_SEGMENTS].copy()


def _build_eval(df_pos, df_neg, df_seq) -> pd.DataFrame:
    """One-row balance / leakage / coverage report on the obtained sample set."""
    n_pos, n_neg = len(df_pos), len(df_neg)
    n_proteins = int(df_seq[ut.COL_ENTRY].nunique())
    # Count only real source proteins; synthetic windows carry an empty entry, not a protein.
    source_entries = df_neg[ut.COL_ENTRY][df_neg[ut.COL_ENTRY].astype(str) != ""] if n_neg else []
    n_source = int(pd.Series(source_entries).nunique())
    max_sim = _max_identity_to_test(list(df_neg[ut.COL_WINDOW]), list(df_pos[ut.COL_WINDOW]))
    return pd.DataFrame([{
        "n_positive": n_pos,
        "n_negative": n_neg,
        "balance_ratio": (n_neg / n_pos) if n_pos else float("nan"),
        "n_source_proteins": n_source,
        "protein_coverage": (n_source / n_proteins) if n_proteins else float("nan"),
        "max_similarity_to_test": max_sim,
    }])


def _plot_logo_comparison(df_samples) -> List[Axes]:
    """Stacked sequence-logo comparison of the sampled groups (one logo per role).

    Each role's windows (the positive ``Test`` windows and the sampled references) are turned
    into a one-column ``tmd`` parts frame and rendered as an information-content sequence logo
    via :meth:`AAlogoPlot.multi_logo`, so composition / conservation differences between the
    groups are directly comparable on a shared scale.
    """
    aal = AAlogo(logo_type="information")
    list_df_logo, list_names = [], []
    for role, df_group in df_samples.groupby(ut.COL_ROLE, sort=False):
        windows = list(df_group[ut.COL_WINDOW])
        if not windows:
            continue
        df_parts = pd.DataFrame({ut.COL_TMD: windows})
        list_df_logo.append(aal.get_df_logo(df_parts=df_parts))
        list_names.append(f"{role} (n={len(windows)})")
    # The whole window is the TMD region (no JMD flanks), so jmd_n_len = jmd_c_len = 0.
    aalp = AAlogoPlot(logo_type="information", jmd_n_len=0, jmd_c_len=0, verbose=False)
    _, axes = aalp.multi_logo(list_df_logo=list_df_logo, list_name_data=list_names)
    return axes


# II Main Functions
def obtain_samples(df_seq: pd.DataFrame,
                   pos_col: str = ut.COL_POS,
                   strategy: str = ut.STRATEGY_SAME,
                   n: Optional[int] = None,
                   window_size: int = 9,
                   reliable_negatives: bool = False,
                   df_feat: Optional[pd.DataFrame] = None,
                   max_similarity_to_test: Optional[float] = None,
                   plot: bool = False,
                   seed: Optional[int] = None,
                   n_jobs: Optional[int] = None,
                   verbose: bool = False,
                   ) -> Tuple[pd.DataFrame, Optional[List[Axes]], pd.DataFrame]:
    """
    Obtain a balanced training set from a described sampling situation in one call.

    The **first** golden pipeline of a typical workflow: instead of hand-driving
    :class:`AAWindowSampler` (roles, strategies, per-call seeds) and, for positive-unlabeled
    problems, :class:`dPULearn`, the parameters here simply **describe the situation**. The
    known positives are the rows of ``df_seq`` carrying 1-based anchors in ``pos_col``; ``n``
    reference windows are drawn to balance them according to ``strategy`` and concatenated into
    one labeled ``'segments'``-mode set, alongside a small balance / leakage / coverage report.
    A thin, stateless facade: it adds no sampling algorithm and its defaults are byte-identical
    to the equivalent explicit primitive calls.

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
        DataFrame with an ``entry`` column of unique protein identifiers, a ``sequence`` column
        of full protein sequences, and a ``pos_col`` column of 1-based positive positions.
    pos_col : str, default='pos'
        Column with per-row 1-based positive (test) positions; list-like, scalar, or a
        comma/semicolon-separated string per row. Rows with empty / missing cells carry no
        positive.
    strategy : str, default='same_protein'
        The sampling situation, mapped to one :class:`AAWindowSampler` call:

        - ``'same_protein'`` — within-protein hard negatives (:meth:`sample_same_protein`).
        - ``'different_protein'`` — cross-protein unlabeled windows (:meth:`sample_different_protein`);
          the natural unlabeled pool ``U`` and the required situation for ``reliable_negatives``.
        - ``'synthetic'`` — synthetic control windows (:meth:`sample_synthetic`).
    n : int, optional
        Number of reference/negative windows to draw. If ``None`` (default), it matches the
        number of positive windows, yielding a balanced set.
    window_size : int, default=9
        Length of each sampled window in residues.
    reliable_negatives : bool, default=False
        If ``True``, engage the positive-unlabeled (PU) path: sample an unlabeled cross-protein
        pool and refine it into reliable negatives with :class:`dPULearn`. Requires ``df_feat``
        and ``strategy='different_protein'``.
    df_feat : pd.DataFrame, optional
        Feature DataFrame with a ``feature`` column. Required when ``reliable_negatives`` is
        ``True`` (used to build the feature matrix ``X`` for dPULearn); ignored otherwise. The
        features must be compatible with windows anchored as a length-``window_size`` TMD with
        the default JMD flanks.
    max_similarity_to_test : float in [0, 1], optional
        If set, drop sampled windows whose per-position identity to any test window exceeds this
        threshold (anti-leakage); passed through to :class:`AAWindowSampler`.
    plot : bool, default=False
        If ``True``, draw a stacked sequence-logo comparison of the sampled groups (one
        information-content logo per ``role``, e.g. ``Test`` vs the references) so their
        composition / conservation can be compared, and return its list of ``Axes``.
    seed : int, optional
        The seed used by the random number generator. If a positive integer, results of
        stochastic processes are reproducible.
    n_jobs : int, optional
        Number of CPU cores (>=1) for building the feature matrix on the PU path. If ``None``,
        the optimized number is used.
    verbose : bool, default=False
        If ``True``, verbose progress information is printed.

    Returns
    -------
    df_samples : pd.DataFrame
        The balanced training set in ``'segments'`` mode: positive test windows (``label=1``,
        ``role='Test'``) plus the sampled references (``label=0``), with full row provenance in
        the ``role`` / ``strategy`` columns.
    ax : list of matplotlib.axes.Axes or None
        The per-group sequence-logo comparison axes, or ``None`` when ``plot=False``.
    df_eval : pd.DataFrame
        One-row balance / leakage / coverage report (positive / negative counts, balance ratio,
        number of source proteins, protein coverage, and maximum identity of any sampled window
        to a test window).

    See Also
    --------
    * :class:`AAWindowSampler` for the window-sampling primitives this pipeline wraps.
    * :class:`dPULearn` for the reliable-negative identification used by the PU path.
    * :class:`AAlogoPlot` for the sequence-logo comparison drawn when ``plot=True``.

    Examples
    --------
    .. include:: examples/aap_obtain_samples.rst
    """
    # Validate (thin facade: the wrapped primitives validate the rest)
    ut.check_df_seq(df_seq=df_seq)
    ut.check_str(name="pos_col", val=pos_col, accept_none=False)
    if ut.COL_SEQ not in df_seq.columns:
        raise ValueError(f"'df_seq' (columns {list(df_seq.columns)}) should contain a "
                         f"'{ut.COL_SEQ}' column of full sequences for window sampling.")
    if pos_col not in df_seq.columns:
        raise ValueError(f"'pos_col' ('{pos_col}') should be a column of 'df_seq' "
                         f"(columns {list(df_seq.columns)}).")
    ut.check_str_options(name="strategy", val=strategy, list_str_options=LIST_SITUATIONS)
    ut.check_number_range(name="n", val=n, min_val=1, just_int=True, accept_none=True)
    ut.check_number_range(name="window_size", val=window_size, min_val=1, just_int=True)
    ut.check_bool(name="reliable_negatives", val=reliable_negatives)
    ut.check_number_range(name="max_similarity_to_test", val=max_similarity_to_test,
                          min_val=0, max_val=1, just_int=False, accept_none=True)
    ut.check_bool(name="plot", val=plot)
    ut.check_number_range(name="seed", val=seed, min_val=0, just_int=True, accept_none=True)
    ut.check_bool(name="verbose", val=verbose)
    check_match_reliable_negatives_df_feat(reliable_negatives=reliable_negatives,
                                           df_feat=df_feat, strategy=strategy)
    if reliable_negatives:
        df_feat = ut.check_df_feat(df_feat=df_feat)

    # Positives (known test windows) and the balance target
    df_pos = _positive_segments(df_seq, pos_col, window_size)
    if len(df_pos) == 0:
        raise ValueError(f"'df_seq' (0 positive windows of size {window_size} at the anchors in "
                         f"'{pos_col}') should yield at least one positive window to build a "
                         f"training set.")
    n_neg = n if n is not None else len(df_pos)

    # Negatives / references from the situation's AAWindowSampler call
    aaws = AAWindowSampler(verbose=verbose, random_state=seed,
                           max_similarity_to_test=max_similarity_to_test)
    if reliable_negatives:
        # Sample a larger unlabeled pool so dPULearn genuinely selects the n_neg most-reliable.
        df_unl = aaws.sample_different_protein(df_seq=df_seq, n=n_neg * UNLABELED_POOL_FACTOR,
                                               window_size=window_size, pos_col=pos_col, seed=seed)
        df_neg = _reliable_negatives(df_pos, df_unl, df_feat, window_size, n_neg,
                                     seed, n_jobs, verbose)
    else:
        df_neg = _sample_negatives(aaws, df_seq, strategy, n_neg, window_size, pos_col, seed)

    # Build the balanced set and the validation report
    df_samples = pd.concat([df_pos, df_neg], ignore_index=True)
    df_eval = _build_eval(df_pos, df_neg, df_seq)
    ax = _plot_logo_comparison(df_samples) if plot else None
    return df_samples, ax, df_eval
