"""
This is a script for the backend of the audit_leakage function, i.e. the individual
leakage heuristics and the assembly of their findings into one table.
"""
from typing import List, Optional
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# Heuristic thresholds. Deliberately fixed rather than exposed: they mark the point where a
# pattern is worth a human's attention, not a decision boundary the caller should tune.
TH_TARGET_CORR = 0.95   # |correlation| of a feature with the label above which it looks derived
TH_SIZE_RATIO = 2.0     # largest/smallest test fold above which the folds are called uneven
TH_BALANCE_DEV = 0.2    # absolute deviation of a fold's class share from the whole dataset
N_IDS_MAX = 10          # affected ids kept per finding (the true count stays in the detail text)


# I Helper Functions
def _format_ids(ids) -> List[str]:
    """Render affected identifiers as a short, capped list of strings."""
    return [str(i) for i in list(ids)[:N_IDS_MAX]]


def _format_folds(folds) -> str:
    """Render the affected fold indices as a readable enumeration."""
    return ", ".join(str(f) for f in folds)


def _finding(check=None, severity=None, detail=None, ids=None) -> list:
    """Build one positional row of the findings frame, ordered as ut.COLS_AUDIT_LEAKAGE."""
    return [check, severity, detail, _format_ids(ids if ids is not None else [])]


def get_col_seq_(df_seq=None) -> Optional[str]:
    """Obtain the column holding the sequence of the *sample*, or None when there is none.

    Windowed output carries both the parent protein sequence and the window cut from it, so the
    window is the sample; comparing the parent instead would call every window of one protein a
    duplicate of its siblings.
    """
    if df_seq is None:
        return None
    for col in [ut.COL_WINDOW, ut.COL_SEQ]:
        if col in df_seq:
            return col
    return None


def _get_ids(df_seq=None, n_rows=0) -> np.ndarray:
    """Obtain the sample identifiers of a sequence frame, falling back to the positional index."""
    if df_seq is not None:
        for col in [ut.COL_ENTRY_WIN, ut.COL_ENTRY]:
            if col in df_seq:
                return np.asarray(df_seq[col])
    return np.arange(n_rows)


def _comp_overlap(values=None, train_idx=None, test_idx=None) -> np.ndarray:
    """Obtain the values shared by the training and the test part of one fold."""
    return np.intersect1d(np.unique(values[train_idx]), np.unique(values[test_idx]))


def _scan_folds(splits=None, values=None):
    """Collect, over all folds, the values appearing in both parts and the folds they appear in."""
    shared_all, folds = [], []
    for fold, (train_idx, test_idx) in enumerate(splits):
        shared = _comp_overlap(values=values, train_idx=train_idx, test_idx=test_idx)
        if len(shared) > 0:
            shared_all.extend(shared.tolist())
            folds.append(fold)
    return sorted(set(shared_all), key=str), folds


def _comp_corr_with_labels(X=None, labels=None) -> np.ndarray:
    """Compute the absolute Pearson correlation of every feature with the label vector."""
    y = np.asarray(labels, dtype=float)
    x_cent = X - X.mean(axis=0)
    y_cent = y - y.mean()
    denom = np.sqrt((x_cent ** 2).sum(axis=0) * (y_cent ** 2).sum())
    corr = np.zeros(X.shape[1], dtype=float)
    valid = denom > 0
    corr[valid] = (x_cent[:, valid] * y_cent[:, None]).sum(axis=0) / denom[valid]
    return np.abs(corr)


def _comp_class_shares(labels=None, idx=None, classes=None) -> np.ndarray:
    """Compute the share of every class within one part of a fold."""
    part = labels[idx]
    if len(part) == 0:
        return np.zeros(len(classes), dtype=float)
    return np.array([np.mean(part == c) for c in classes], dtype=float)


# II Main Functions
def check_duplicate_seq_(df_seq=None) -> List[list]:
    """Flag identical sequence strings, which become a leak as soon as the data is split."""
    col_seq = get_col_seq_(df_seq=df_seq)
    if col_seq is None:
        return []
    seqs = df_seq[col_seq].astype(str)
    is_dup = seqs.duplicated(keep=False)
    if not is_dup.any():
        return []
    ids = _get_ids(df_seq=df_seq, n_rows=len(df_seq))
    n_rows, n_seqs = int(is_dup.sum()), int(seqs[is_dup].nunique())
    detail = (f"{n_rows} rows share only {n_seqs} distinct '{col_seq}' value(s). Duplicates that "
              f"end up on both sides of a split are scored on what the model already memorized")
    return [_finding(check=ut.STR_CHECK_DUPLICATE_SEQ, severity=ut.STR_SEVERITY_MEDIUM,
                     detail=detail, ids=ids[np.asarray(is_dup)])]


def check_train_test_overlap_(splits=None) -> List[list]:
    """Flag a row index used for both training and testing in the same fold."""
    shared_all, folds = [], []
    for fold, (train_idx, test_idx) in enumerate(splits):
        shared = np.intersect1d(np.asarray(train_idx), np.asarray(test_idx))
        if len(shared) > 0:
            shared_all.extend(shared.tolist())
            folds.append(fold)
    if len(folds) == 0:
        return []
    ids = sorted(set(shared_all))
    detail = (f"{len(ids)} row(s) are in both the training and the test part of fold(s) "
              f"{_format_folds(folds)}, so the model is scored on rows it was fitted on")
    return [_finding(check=ut.STR_CHECK_TRAIN_TEST_OVERLAP, severity=ut.STR_SEVERITY_HIGH,
                     detail=detail, ids=ids)]


def check_duplicate_seq_folds_(df_seq=None, splits=None) -> List[list]:
    """Flag an identical sequence whose copies are split across the two parts of a fold."""
    col_seq = get_col_seq_(df_seq=df_seq)
    if col_seq is None:
        return []
    seqs = np.asarray(df_seq[col_seq].astype(str))
    shared, folds = _scan_folds(splits=splits, values=seqs)
    if len(folds) == 0:
        return []
    detail = (f"{len(shared)} '{col_seq}' value(s) occur in both the training and the test part "
              f"of fold(s) {_format_folds(folds)}, so an identical copy was seen during training")
    # Report the affected samples, not the sequence strings, which are too long to read
    ids = _get_ids(df_seq=df_seq, n_rows=len(df_seq))[np.isin(seqs, list(shared))]
    return [_finding(check=ut.STR_CHECK_DUPLICATE_SEQ_FOLDS, severity=ut.STR_SEVERITY_HIGH,
                     detail=detail, ids=ids)]


def check_entry_folds_(df_seq=None, splits=None) -> List[list]:
    """Flag windows of one protein that are split across the two parts of a fold."""
    if df_seq is None or ut.COL_ENTRY not in df_seq:
        return []
    entries = np.asarray(df_seq[ut.COL_ENTRY])
    # Only proteins contributing several rows can have their windows split apart; with one row
    # per protein this would merely restate a train/test row overlap.
    if not pd.Series(entries).duplicated().any():
        return []
    shared, folds = _scan_folds(splits=splits, values=entries)
    if len(folds) == 0:
        return []
    detail = (f"{len(shared)} protein(s) have windows in both the training and the test part of "
              f"fold(s) {_format_folds(folds)}, so neighbouring windows leak across the split")
    return [_finding(check=ut.STR_CHECK_ENTRY_FOLDS, severity=ut.STR_SEVERITY_HIGH,
                     detail=detail, ids=shared)]


def check_group_folds_(groups=None, splits=None) -> List[list]:
    """Flag a group label appearing in both parts of a fold."""
    if groups is None:
        return []
    shared, folds = _scan_folds(splits=splits, values=groups)
    if len(folds) == 0:
        return []
    detail = (f"{len(shared)} group(s) appear in both the training and the test part of fold(s) "
              f"{_format_folds(folds)}; bind the groups to a group-aware splitter to prevent it")
    return [_finding(check=ut.STR_CHECK_GROUP_FOLDS, severity=ut.STR_SEVERITY_HIGH,
                     detail=detail, ids=shared)]


def check_target_leak_(X=None, labels=None, names=None) -> List[list]:
    """Flag a feature column that tracks the label closely enough to look derived from it."""
    if X is None or labels is None:
        return []
    corr = _comp_corr_with_labels(X=X, labels=labels)
    is_leaky = corr >= TH_TARGET_CORR
    if not is_leaky.any():
        return []
    idx = np.flatnonzero(is_leaky)
    ids = [names[i] for i in idx] if names is not None else [f"column {i}" for i in idx]
    detail = (f"{len(idx)} feature(s) correlate with the label at |r| >= {TH_TARGET_CORR} "
              f"(max {corr[idx].max():.3f}), which is what a feature derived from the target "
              f"looks like. Confirm each is computed without the label")
    return [_finding(check=ut.STR_CHECK_TARGET_LEAK, severity=ut.STR_SEVERITY_HIGH,
                     detail=detail, ids=ids)]


def check_fold_size_(splits=None) -> List[list]:
    """Flag an empty fold part, and test folds whose sizes are strongly uneven."""
    n_train = np.array([len(train_idx) for train_idx, _ in splits])
    n_test = np.array([len(test_idx) for _, test_idx in splits])
    empty = np.flatnonzero((n_train == 0) | (n_test == 0))
    if len(empty) > 0:
        detail = (f"fold(s) {_format_folds(empty.tolist())} have an empty training or test part, "
                  f"so no score from them describes the model")
        return [_finding(check=ut.STR_CHECK_FOLD_SIZE, severity=ut.STR_SEVERITY_HIGH,
                         detail=detail, ids=empty.tolist())]
    if len(n_test) < 2 or n_test.min() == 0:
        return []
    ratio = n_test.max() / n_test.min()
    if ratio < TH_SIZE_RATIO:
        return []
    extremes = [int(np.argmin(n_test)), int(np.argmax(n_test))]
    detail = (f"the largest test fold holds {ratio:.1f}x the rows of the smallest "
              f"({n_test.max()} vs {n_test.min()}), so the folds are not comparable and the mean "
              f"score is dominated by the large one. Group splitters do this legitimately")
    return [_finding(check=ut.STR_CHECK_FOLD_SIZE, severity=ut.STR_SEVERITY_LOW,
                     detail=detail, ids=extremes)]


def check_class_balance_(labels=None, splits=None) -> List[list]:
    """Flag a single-class test fold, and folds whose class balance departs from the dataset."""
    if labels is None:
        return []
    labels = np.asarray(labels)
    classes = np.unique(labels)
    shares_all = _comp_class_shares(labels=labels, idx=np.arange(len(labels)), classes=classes)
    single, skewed, devs = [], [], []
    for fold, (_, test_idx) in enumerate(splits):
        test_idx = np.asarray(test_idx)
        if len(test_idx) == 0:
            continue
        if len(np.unique(labels[test_idx])) < 2 and len(classes) > 1:
            single.append(fold)
            continue
        shares = _comp_class_shares(labels=labels, idx=test_idx, classes=classes)
        dev = float(np.abs(shares - shares_all).max())
        if dev > TH_BALANCE_DEV:
            skewed.append(fold)
            devs.append(dev)
    findings = []
    if len(single) > 0:
        detail = (f"the test part of fold(s) {_format_folds(single)} holds a single class, so "
                  f"every class-aware metric there is undefined or trivially satisfied")
        findings.append(_finding(check=ut.STR_CHECK_CLASS_BALANCE, severity=ut.STR_SEVERITY_HIGH,
                                 detail=detail, ids=single))
    if len(skewed) > 0:
        detail = (f"the class balance of fold(s) {_format_folds(skewed)} departs from the whole "
                  f"dataset by up to {max(devs):.2f}, so their scores are not comparable. "
                  f"A stratified splitter keeps the balance")
        findings.append(_finding(check=ut.STR_CHECK_CLASS_BALANCE,
                                 severity=ut.STR_SEVERITY_MEDIUM, detail=detail, ids=skewed))
    return findings


def comp_status_(df_audit: pd.DataFrame) -> str:
    """Summarise the findings as the worst severity present, or 'ok' when there are none."""
    if len(df_audit) == 0:
        return ut.STR_STATUS_OK
    severities = set(df_audit[ut.COL_SEVERITY])
    for severity in reversed(ut.LIST_SEVERITIES):
        if severity in severities:
            return severity
    return ut.STR_STATUS_OK


def audit_leakage_(df_seq=None, labels=None, groups=None, splits=None, X=None,
                   names: Optional[list] = None) -> pd.DataFrame:
    """Run every applicable heuristic and assemble the findings into one table."""
    rows = []
    rows.extend(check_duplicate_seq_(df_seq=df_seq))
    rows.extend(check_target_leak_(X=X, labels=labels, names=names))
    if splits is not None:
        rows.extend(check_train_test_overlap_(splits=splits))
        rows.extend(check_duplicate_seq_folds_(df_seq=df_seq, splits=splits))
        rows.extend(check_entry_folds_(df_seq=df_seq, splits=splits))
        rows.extend(check_group_folds_(groups=groups, splits=splits))
        rows.extend(check_fold_size_(splits=splits))
        rows.extend(check_class_balance_(labels=labels, splits=splits))
    df_audit = pd.DataFrame(rows, columns=ut.COLS_AUDIT_LEAKAGE)
    # Worst findings first, so the top of the table is what a reader must act on
    order = {s: i for i, s in enumerate(reversed(ut.LIST_SEVERITIES))}
    if len(df_audit) > 0:
        df_audit = (df_audit.sort_values(by=ut.COL_SEVERITY, key=lambda x: x.map(order),
                                         kind="stable")
                    .reset_index(drop=True))
    return df_audit
