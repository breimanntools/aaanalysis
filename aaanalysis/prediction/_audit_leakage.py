"""
This is a script for the frontend of the audit_leakage function for diagnosing leakage
risks in an evaluation setup.
"""
from typing import Optional, Union, List, Tuple, Iterable
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.audit_leakage import audit_leakage_, comp_status_, get_col_seq_


# I Helper Functions
def check_groups(groups=None):
    """Check the group labels and return them as a 1D array, or None."""
    if groups is None:
        return None
    groups = ut.check_array_like(name="groups", val=groups, accept_none=False)
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError(f"'groups' (n dimensions={groups.ndim}) should be one-dimensional")
    if len(groups) == 0:
        raise ValueError("'groups' (length=0) should contain at least one group label")
    return groups


def check_labels_audit(labels=None):
    """Check the labels permissively, since a single-class setup is itself a finding."""
    if labels is None:
        return None
    labels = ut.check_array_like(name="labels", val=labels, accept_none=False)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"'labels' (n dimensions={labels.ndim}) should be one-dimensional")
    if len(labels) == 0:
        raise ValueError("'labels' (length=0) should contain at least one label")
    return labels


def check_splits(splits=None):
    """Check the splits and return them as a list of (train_idx, test_idx) index arrays."""
    if splits is None:
        return None
    try:
        splits = list(splits)
    except TypeError:
        raise ValueError(f"'splits' ({type(splits).__name__}) should be an iterable of "
                         f"(train_idx, test_idx) pairs, e.g. 'list(cv.split(X, labels))'")
    if len(splits) == 0:
        raise ValueError("'splits' (n folds=0) should contain at least one fold")
    list_splits = []
    for fold, split in enumerate(splits):
        if not isinstance(split, (tuple, list)) or len(split) != 2:
            raise ValueError(f"'splits' should contain (train_idx, test_idx) pairs, but fold "
                             f"{fold} ({split}) is not a pair of two index arrays")
        idx = []
        for name, val in zip(["train_idx", "test_idx"], split):
            arr = np.asarray(val)
            if arr.ndim != 1:
                raise ValueError(f"'{name}' of fold {fold} (n dimensions={arr.ndim}) should be "
                                 f"one-dimensional")
            if len(arr) > 0 and not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"'{name}' of fold {fold} (dtype={arr.dtype}) should contain "
                                 f"integer row positions, not labels or a boolean mask")
            idx.append(arr.astype(int))
        list_splits.append((idx[0], idx[1]))
    return list_splits


def check_names(names=None, n_features=None):
    """Check the feature names and return them as a list of strings, or None."""
    if names is None:
        return None
    names = ut.check_list_like(name="names", val=names, accept_none=False, accept_str=False)
    if n_features is not None and len(names) != n_features:
        raise ValueError(f"'names' (length={len(names)}) should have one name per feature in 'X' "
                         f"(n features={n_features})")
    return [str(n) for n in names]


def check_anything_to_audit(df_seq=None, groups=None, splits=None, X=None):
    """Check that at least one input was given, since otherwise there is nothing to inspect."""
    if df_seq is None and groups is None and splits is None and X is None:
        raise ValueError("'df_seq', 'groups', 'splits' and 'X' are all None; at least one should "
                         "be given, since there is otherwise nothing to audit")


def check_df_seq_audit(df_seq=None):
    """Check the sequence frame permissively, since the audit inspects whatever columns exist."""
    if df_seq is None:
        return None
    ut.check_df(name="df_seq", df=df_seq, accept_none=False)
    if len(df_seq) == 0:
        raise ValueError("'df_seq' (n rows=0) should contain at least one sequence")
    if get_col_seq_(df_seq=df_seq) is None and ut.COL_ENTRY not in df_seq:
        raise ValueError(f"'df_seq' (columns={list(df_seq)}) should contain at least one of "
                         f"'{ut.COL_SEQ}', '{ut.COL_WINDOW}' (the sample sequence) or "
                         f"'{ut.COL_ENTRY}' (the protein it comes from)")
    return df_seq


def check_match_splits_n_samples(splits=None, n_samples=None, name=None):
    """Check that the row positions of every fold are within the given number of samples."""
    if splits is None or n_samples is None:
        return None
    for fold, (train_idx, test_idx) in enumerate(splits):
        for str_part, idx in zip(["train_idx", "test_idx"], [train_idx, test_idx]):
            if len(idx) == 0:
                continue
            if idx.min() < 0 or idx.max() >= n_samples:
                raise ValueError(f"'{str_part}' of fold {fold} (min={idx.min()}, max={idx.max()}) "
                                 f"should contain row positions within '{name}' "
                                 f"(n samples={n_samples})")


def check_match_n_samples(df_seq=None, labels=None, groups=None, X=None):
    """Check that every given input describes the same number of samples."""
    dict_n = {}
    if df_seq is not None:
        dict_n["df_seq"] = len(df_seq)
    if labels is not None:
        dict_n["labels"] = len(labels)
    if groups is not None:
        dict_n["groups"] = len(groups)
    if X is not None:
        dict_n["X"] = len(X)
    if len(set(dict_n.values())) > 1:
        str_n = ", ".join(f"'{k}' (n samples={v})" for k, v in dict_n.items())
        raise ValueError(f"{str_n} should all describe the same samples, but their lengths differ")


def raise_on_findings(df_audit: pd.DataFrame, raise_on=None):
    """Raise when a finding reaches the severity the caller chose to hard-fail on."""
    if raise_on is None or len(df_audit) == 0:
        return None
    th = ut.LIST_SEVERITIES.index(raise_on)
    mask = [ut.LIST_SEVERITIES.index(s) >= th for s in df_audit[ut.COL_SEVERITY]]
    df_hit = df_audit[mask]
    if len(df_hit) == 0:
        return None
    str_checks = "; ".join(f"{r[ut.COL_CHECK]} ({r[ut.COL_SEVERITY]}): {r[ut.COL_DETAIL]}"
                           for _, r in df_hit.iterrows())
    raise ValueError(f"'raise_on' ('{raise_on}') matched {len(df_hit)} leakage finding(s) at or "
                     f"above that severity: {str_checks}")


# II Main Functions
def audit_leakage(df_seq: Optional[pd.DataFrame] = None,
                  *,
                  labels: Optional[ut.ArrayLike1D] = None,
                  groups: Optional[Union[ut.ArrayLike1D, pd.Series]] = None,
                  splits: Optional[Iterable[Tuple[ut.ArrayLike1D, ut.ArrayLike1D]]] = None,
                  X: Optional[ut.ArrayLike2D] = None,
                  names: Optional[List[str]] = None,
                  raise_on: Optional[str] = None,
                  ) -> pd.DataFrame:
    """
    Audit an evaluation setup for leakage risks and report them as a table of findings.

    Leakage inflates a score without ever raising an error: a duplicated sequence on both sides
    of a split, two windows of one protein pulled apart, a feature quietly derived from the
    label. On the small, windowed datasets typical of sequence-based prediction these are easy
    to introduce and hard to see, and the only symptom is an implausibly good number. This
    function runs a set of cheap heuristics over whatever parts of the setup are supplied and
    lists what looks wrong, before a score is trusted [Kaufman12]_.

    It works in two stages, each optional. Without ``splits`` only the dataset itself is
    inspected, which is the check to run *before* choosing a split. With ``splits`` the folds
    are inspected too, comparing the training against the test part **within each fold**.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    df_seq : pd.DataFrame, shape (n_samples, n_features), optional
        DataFrame containing an ``entry`` column with protein identifiers, one row per sample.
        An identifier repeats when several windows come from one protein, which is what the
        per-protein check looks for. ``'sequence'``, or ``'window'`` when present, is compared
        for duplicates; for windowed output the window is the sample, so it takes precedence
        over the parent ``'sequence'``.
    labels : array-like, shape (n_samples,), optional
        Class labels per sample. Used for the class-balance and target-derived checks. Taken
        from ``df_seq['label']`` when that column exists and ``labels`` is not given. Unlike
        elsewhere in this package, a single class is accepted, since it is itself reported.
    groups : array-like, shape (n_samples,), optional
        Group label per sample, as passed to :func:`bind_groups`: a protein accession, a family
        name, or an externally computed homology cluster. A group in both parts of a fold is a
        leak.
    splits : iterable of (train_idx, test_idx), optional
        The folds to inspect, as integer **row positions**, e.g. ``list(cv.split(X, labels))``
        or the output of a splitter returned by :func:`bind_groups`. Without it only the
        dataset-level checks run.
    X : array-like, shape (n_samples, n_features), optional
        Feature matrix to inspect for columns that track the label too closely. Requires
        ``labels``.
    names : list of str, optional
        Feature name per column of ``X``, e.g. ``df_feat['feature'].to_list()``, so a finding
        names the feature instead of its column position.
    raise_on : str, optional
        Severity at or above which a finding raises a ``ValueError`` instead of being reported:
        one of ``'low'``, ``'medium'`` or ``'high'``. The default reports only and never raises,
        so an audit can be added to a workflow without changing its control flow.

    Returns
    -------
    df_audit : pd.DataFrame
        Findings, one row per issue, worst first, with the columns ``'check'`` (which heuristic
        fired), ``'severity'`` (``'low'``, ``'medium'`` or ``'high'``), ``'detail'`` (a sentence
        describing the finding and why it matters) and ``'ids'`` (the affected identifiers, at
        most 10 of them; the true count is in ``'detail'``). A clean setup gives an empty frame
        with those columns. ``df_audit.attrs['status']`` carries the worst severity present, or
        ``'ok'`` when there is no finding.

    Raises
    ------
    ValueError
        If the given inputs do not describe the same samples, if ``splits`` holds anything other
        than pairs of integer row positions within range, or if ``raise_on`` is set and a finding
        reaches that severity.

    Notes
    -----
    * **The audit is heuristic, and a clean report is not a proof that no leakage exists.** It
      inspects only the data and the folds it is given. It cannot see how a feature was built,
      whether a scaler was fitted before the split, or that two sequences are near-identical
      rather than byte-identical. Treat ``status='ok'`` as "nothing obvious", not as a guarantee.
    * The severities are labels for a human reader, not a machine taxonomy: whether a finding
      must block a workflow is a policy decision this package deliberately leaves to the caller.
    * Thresholds are fixed so a report means the same thing everywhere: a feature correlates
      with the label at :math:`|r| \\ge 0.95`, the largest test fold holds twice the rows of the
      smallest, or a fold's class share departs from the whole dataset by more than 0.2.
    * Row positions, not index labels, are expected in ``splits``, matching scikit-learn. A
      ``df_seq`` with a non-default index is therefore audited by position.

    See Also
    --------
    * :func:`bind_groups` to *prevent* the group leak this reports, by keeping every group
      within one fold.
    * :class:`AAPred` and :class:`ModelEvaluator`, whose ``cv`` argument takes the splitter
      whose folds are audited here.

    Examples
    --------
    .. include:: examples/audit_leakage.rst
    """
    # Validate
    df_seq = check_df_seq_audit(df_seq=df_seq)
    if labels is None and df_seq is not None and ut.COL_LABEL in df_seq:
        labels = df_seq[ut.COL_LABEL].to_numpy()
    labels = check_labels_audit(labels=labels)
    groups = check_groups(groups=groups)
    splits = check_splits(splits=splits)
    names_x = list(X.columns) if isinstance(X, pd.DataFrame) else None
    X = ut.check_X(X, accept_none=True, min_n_samples=1, min_n_features=1)
    names = check_names(names=names, n_features=None if X is None else X.shape[1])
    ut.check_str_options(name="raise_on", val=raise_on, accept_none=True,
                         list_str_options=ut.LIST_SEVERITIES)
    check_anything_to_audit(df_seq=df_seq, groups=groups, splits=splits, X=X)
    check_match_n_samples(df_seq=df_seq, labels=labels, groups=groups, X=X)
    for name, val in zip(["df_seq", "labels", "groups", "X"], [df_seq, labels, groups, X]):
        if val is not None:
            check_match_splits_n_samples(splits=splits, n_samples=len(val), name=name)
    if X is not None and labels is None:
        raise ValueError("'labels' should be given together with 'X', since a feature can only "
                         "be checked against the target it might be derived from")
    # Run the heuristics
    df_audit = audit_leakage_(df_seq=df_seq, labels=labels, groups=groups, splits=splits, X=X,
                              names=names if names is not None else names_x)
    df_audit.attrs["status"] = comp_status_(df_audit=df_audit)
    raise_on_findings(df_audit=df_audit, raise_on=raise_on)
    return df_audit
