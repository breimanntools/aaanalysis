"""
This is a script for the frontend of the bind_groups function for group-aware evaluation.
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

import aaanalysis.utils as ut


# I Helper Functions
def check_cv(cv):
    """Check that ``cv`` is a usable scikit-learn splitter instance."""
    if isinstance(cv, type):
        raise ValueError(f"'cv' ({cv.__name__}) should be a splitter instance, not the class "
                         f"itself (use '{cv.__name__}()')")
    for method in ["split", "get_n_splits"]:
        if not callable(getattr(cv, method, None)):
            raise ValueError(f"'cv' ({type(cv).__name__}) should be a scikit-learn splitter "
                             f"providing '{method}()'")


def check_groups(groups):
    """Check the group labels and return them as a 1D object array."""
    groups = ut.check_array_like(name="groups", val=groups, accept_none=False)
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError(f"'groups' (n dimensions={groups.ndim}) should be one-dimensional")
    if len(groups) == 0:
        raise ValueError("'groups' (length=0) should contain at least one group label")
    if pd.isna(groups).any():
        raise ValueError("'groups' should not contain missing values (one label per sample)")
    return groups


def check_match_groups_n_samples(groups, n_samples):
    """Check that one group label was given per sample."""
    if len(groups) != n_samples:
        raise ValueError(f"'groups' (length={len(groups)}) should have one label per sample "
                         f"in 'X' (n samples={n_samples})")


def check_match_cv_groups(cv, groups):
    """Check that the requested number of folds is feasible for the group structure."""
    n_groups = len(np.unique(groups))
    n_splits = getattr(cv, "n_splits", None)
    if n_splits is not None and n_splits > n_groups:
        raise ValueError(f"'n_splits' ({n_splits}) of 'cv' ({type(cv).__name__}) should be <= the "
                         f"number of distinct groups (n groups={n_groups}); a fold cannot be "
                         f"filled without splitting a group")


def _comp_pos_rate(idx, labels=None, label_pos=1):
    """Compute the share of the positive class in one part of a fold."""
    if labels is None:
        return np.nan
    part = np.asarray(labels)[idx]
    if len(part) == 0:
        return np.nan
    return float(np.mean(part == label_pos))


def _comp_fold_row(fold, train_idx, test_idx, groups, labels=None):
    """Build one positional row of the fold-metadata frame."""
    return [fold,
            len(train_idx), len(test_idx),
            len(np.unique(groups[train_idx])), len(np.unique(groups[test_idx])),
            _comp_pos_rate(idx=train_idx, labels=labels),
            _comp_pos_rate(idx=test_idx, labels=labels)]


def _check_no_group_overlap(fold, train_idx, test_idx, groups):
    """Raise when a group id appears in both parts of a fold (the leak this guards)."""
    shared = np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[test_idx]))
    if len(shared) > 0:
        raise ValueError(f"'cv' put {len(shared)} group(s) in both train and test of fold {fold} "
                         f"(e.g. '{shared[0]}'), which leaks between folds. Use a group-aware "
                         f"splitter (e.g. 'GroupKFold', 'StratifiedGroupKFold', "
                         f"'LeaveOneGroupOut') or set 'allow_overlap=True' to permit it")


# II Main Functions
class _GroupBoundSplitter(BaseCrossValidator):
    """Splitter that supplies its own group labels to a wrapped scikit-learn splitter."""

    def __init__(self, cv, groups, allow_overlap=False):
        self._cv = cv
        self._groups = groups
        self._allow_overlap = allow_overlap
        self.df_folds_ = None

    def __repr__(self):
        n_groups = len(np.unique(self._groups))
        return f"bind_groups({type(self._cv).__name__}(), n_groups={n_groups})"

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Obtain the number of folds the wrapped splitter yields.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Feature matrix, passed through to the wrapped splitter.
        y : array-like, shape (n_samples,), optional
            Labels, passed through to the wrapped splitter.
        groups : None
            Ignored, since the group labels are bound. Passing an array raises a ``ValueError``.

        Returns
        -------
        n_splits : int
            Number of folds.
        """
        if groups is not None:
            raise ValueError("'groups' should be None, since it is bound to this splitter "
                             "(it was given to 'bind_groups')")
        return self._cv.get_n_splits(X, y, self._groups)

    def split(self, X=None, y=None, groups=None):
        """
        Generate the train/test indices of each fold, keeping every group whole.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. Only its length is used here; it is passed to the wrapped splitter.
        y : array-like, shape (n_samples,), optional
            Labels. Used by a stratified splitter and to record the per-fold class balance.
        groups : None
            Ignored, since the group labels are bound. Passing an array raises a ``ValueError``.

        Yields
        ------
        train_idx : array, shape (n_train,)
            Row indices of the training part.
        test_idx : array, shape (n_test,)
            Row indices of the test part.
        """
        if groups is not None:
            raise ValueError("'groups' should be None, since it is bound to this splitter "
                             "(it was given to 'bind_groups')")
        n_samples = len(X) if X is not None else len(self._groups)
        check_match_groups_n_samples(groups=self._groups, n_samples=n_samples)
        rows = []
        for fold, (train_idx, test_idx) in enumerate(self._cv.split(X, y, self._groups)):
            if not self._allow_overlap:
                _check_no_group_overlap(fold=fold, train_idx=train_idx, test_idx=test_idx,
                                        groups=self._groups)
            rows.append(_comp_fold_row(fold=fold, train_idx=train_idx, test_idx=test_idx,
                                       groups=self._groups, labels=y))
            self.df_folds_ = pd.DataFrame(rows, columns=ut.COLS_FOLDS_GROUPS)
            yield train_idx, test_idx

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Satisfy the ``BaseCrossValidator`` interface; splitting is delegated in ``split``."""
        for _, test_idx in self.split(X=X, y=y):
            yield test_idx


def bind_groups(cv: BaseCrossValidator,
                groups: Union[ut.ArrayLike1D, pd.Series],
                *,
                allow_overlap: bool = False,
                ) -> BaseCrossValidator:
    """
    Bind group labels to a cross-validation splitter, so dependent samples stay in one fold.

    Protein datasets are full of dependent samples: several windows cut from one protein,
    near-identical homologues across a family. A plain split scatters them over train and test, so
    the model is scored on what it has already seen and the reported performance is inflated
    [Roberts17]_. Keeping every group whole removes that inflation, and the group vocabulary is
    whatever is passed: an accession keeps each protein together, a family name or a homology
    cluster keeps each family together.

    Group splitters need their labels at split time, which the evaluation entry points of this
    package do not pass. Binding them to the splitter closes that gap: the returned object carries
    the labels and is accepted anywhere a splitter is.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    cv : splitter
        A scikit-learn splitter **instance** whose folds are to respect the groups, e.g.
        ``GroupKFold(n_splits=5)``, ``StratifiedGroupKFold(n_splits=5)`` (keeps the class
        balance too) or ``LeaveOneGroupOut()`` (leave-one-protein-out). Its own parameters,
        including ``random_state``, are untouched, so reproducibility is the splitter's.
    groups : array-like, shape (n_samples,)
        Group label per sample, aligned with the rows of ``X``: a protein accession, a family
        name, or a cluster id. Any hashable labels work, and the number of distinct values is the
        number of groups. The labels are taken as given; no homology search or clustering is run
        here.
    allow_overlap : bool, default=False
        Whether a group may appear in both train and test of a fold. ``False`` verifies every
        fold and raises a ``ValueError`` naming the shared groups, which turns a splitter that
        ignores groups (e.g. ``KFold``) into an error instead of a silent leak.

    Returns
    -------
    cv_bound : splitter
        Splitter with the group labels bound, to pass as ``cv`` to ``AAPred.eval``,
        ``ModelEvaluator.run`` or any scikit-learn routine. Once its folds are consumed, the
        attribute ``df_folds_`` holds one row per fold with the columns ``fold``, ``n_train``,
        ``n_test``, ``n_groups_train``, ``n_groups_test``, ``pos_rate_train`` and
        ``pos_rate_test`` (the two rates are ``NaN`` when labels were not passed).

    Notes
    -----
    * Group splitters trade exactness for independence: ``GroupKFold`` cannot balance the folds
      as evenly as ``KFold``, so fold sizes differ. Inspect ``df_folds_`` rather than assuming
      equal folds.
    * A splitter whose test folds do not partition the samples (e.g. ``GroupShuffleSplit``)
      works for scoring but not where every sample needs one out-of-fold prediction.
    * Comparing a grouped score against an ungrouped one on the same data measures the leak;
      report the grouped score as the honest one.

    See Also
    --------
    * :class:`AAPred` and its ``eval`` method, which takes the returned object as ``cv``.
    * :class:`ModelEvaluator` for repeated cross-validation with confidence intervals.
    * :meth:`SequenceFeature.get_split_kws`, which splits a *sequence* into parts, unrelated
      to the cross-validation splits meant here.

    Examples
    --------
    .. include:: examples/bind_groups.rst
    """
    # Validate
    check_cv(cv=cv)
    groups = check_groups(groups=groups)
    ut.check_bool(name="allow_overlap", val=allow_overlap)
    check_match_cv_groups(cv=cv, groups=groups)
    # Bind the labels to the splitter
    return _GroupBoundSplitter(cv=cv, groups=groups, allow_overlap=allow_overlap)
