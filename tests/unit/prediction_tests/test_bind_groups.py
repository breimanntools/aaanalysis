"""This is a script to test the aa.bind_groups() function."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut,
                                     GroupShuffleSplit, KFold, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _data(n_groups=8, n_per_group=5, n_features=6, seed=0):
    """Seeded windows-per-protein fixture: the label is a property of the protein."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"P{i}" for i in range(n_groups)], n_per_group)
    labels = np.repeat([i % 2 for i in range(n_groups)], n_per_group)
    X = rng.random((n_groups * n_per_group, n_features))
    return X, labels, groups


def _interleaved(n_groups=8, n_per_group=5, n_features=6, seed=0):
    """Same sizes, but group ids alternate, so a plain KFold splits groups apart."""
    rng = np.random.default_rng(seed)
    groups = np.tile([f"P{i}" for i in range(n_groups)], n_per_group)
    labels = np.tile([i % 2 for i in range(n_groups)], n_per_group)
    X = rng.random((n_groups * n_per_group, n_features))
    return X, labels, groups


# I Normal cases, one parameter per test
class TestBindGroups:
    """Test bind_groups() parameter by parameter."""

    # cv
    def test_group_kfold_is_accepted(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert cv.get_n_splits(X, labels) == 4

    def test_stratified_group_kfold_is_accepted(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(StratifiedGroupKFold(n_splits=4), groups=groups)
        assert len(list(cv.split(X, labels))) == 4

    def test_leave_one_group_out_yields_one_fold_per_group(self):
        X, labels, groups = _data(n_groups=6)
        cv = aa.bind_groups(LeaveOneGroupOut(), groups=groups)
        assert cv.get_n_splits(X, labels) == 6

    def test_group_shuffle_split_is_accepted(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=0),
                            groups=groups)
        assert len(list(cv.split(X, labels))) == 3

    @settings(max_examples=5)
    @given(n_splits=some.integers(min_value=2, max_value=6))
    def test_any_feasible_n_splits(self, n_splits):
        X, labels, groups = _data(n_groups=8)
        cv = aa.bind_groups(GroupKFold(n_splits=n_splits), groups=groups)
        assert len(list(cv.split(X, labels))) == n_splits

    def test_cv_class_instead_of_instance_raises(self):
        _, _, groups = _data()
        with pytest.raises(ValueError, match="splitter instance"):
            aa.bind_groups(GroupKFold, groups=groups)

    def test_cv_without_split_raises(self):
        _, _, groups = _data()
        with pytest.raises(ValueError, match="split"):
            aa.bind_groups(object(), groups=groups)

    def test_cv_none_raises(self):
        _, _, groups = _data()
        with pytest.raises(ValueError):
            aa.bind_groups(None, groups=groups)

    # groups
    def test_groups_as_list(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=list(groups))
        assert len(list(cv.split(X, labels))) == 4

    def test_groups_as_series(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=pd.Series(groups))
        assert len(list(cv.split(X, labels))) == 4

    def test_groups_as_integer_ids(self):
        X, labels, _ = _data()
        groups = np.repeat(np.arange(8), 5)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert len(list(cv.split(X, labels))) == 4

    @settings(max_examples=5)
    @given(n_groups=some.integers(min_value=4, max_value=10))
    def test_any_group_count(self, n_groups):
        X, labels, groups = _data(n_groups=n_groups)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert len(list(cv.split(X, labels))) == 4

    def test_groups_none_raises(self):
        with pytest.raises(ValueError):
            aa.bind_groups(GroupKFold(n_splits=2), groups=None)

    def test_groups_empty_raises(self):
        with pytest.raises(ValueError):
            aa.bind_groups(GroupKFold(n_splits=2), groups=[])

    def test_groups_two_dimensional_raises(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            aa.bind_groups(GroupKFold(n_splits=2), groups=np.zeros((10, 2)))

    def test_groups_with_numeric_nan_raises(self):
        """Caught by the shared array validator, before the group-specific check."""
        with pytest.raises(ValueError, match="NaN"):
            aa.bind_groups(GroupKFold(n_splits=2), groups=[1.0, np.nan, 2.0, 2.0])

    def test_groups_with_none_label_raises(self):
        """A None in an object array passes the shared validator, so this guard is needed.

        Without it the missing label would silently become a group of its own.
        """
        with pytest.raises(ValueError, match="missing values"):
            aa.bind_groups(GroupKFold(n_splits=2), groups=["A", None, "B", "B"])

    def test_groups_length_mismatch_raises(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=2), groups=groups[:10])
        with pytest.raises(ValueError, match="one label per sample"):
            list(cv.split(X, labels))

    # allow_overlap
    def test_allow_overlap_false_rejects_a_non_group_splitter(self):
        X, labels, groups = _interleaved()
        cv = aa.bind_groups(KFold(n_splits=4), groups=groups)
        with pytest.raises(ValueError, match="both train and test"):
            list(cv.split(X, labels))

    def test_allow_overlap_true_permits_it(self):
        X, labels, groups = _interleaved()
        cv = aa.bind_groups(KFold(n_splits=4), groups=groups, allow_overlap=True)
        assert len(list(cv.split(X, labels))) == 4

    def test_allow_overlap_must_be_bool(self):
        _, _, groups = _data()
        with pytest.raises(ValueError):
            aa.bind_groups(GroupKFold(n_splits=2), groups=groups, allow_overlap="yes")

    def test_allow_overlap_is_keyword_only(self):
        _, _, groups = _data()
        with pytest.raises(TypeError):
            aa.bind_groups(GroupKFold(n_splits=2), groups, True)

    # infeasible fold counts
    def test_more_folds_than_groups_raises(self):
        _, _, groups = _data(n_groups=3)
        with pytest.raises(ValueError, match="number of distinct groups"):
            aa.bind_groups(GroupKFold(n_splits=5), groups=groups)

    def test_infeasible_message_names_both_counts(self):
        _, _, groups = _data(n_groups=3)
        with pytest.raises(ValueError, match=r"'n_splits' \(5\).*n groups=3"):
            aa.bind_groups(GroupKFold(n_splits=5), groups=groups)

    def test_folds_equal_to_groups_is_feasible(self):
        X, labels, groups = _data(n_groups=4)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert len(list(cv.split(X, labels))) == 4

    # bound groups are not overridable
    def test_passing_groups_to_split_raises(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        with pytest.raises(ValueError, match="bound to this splitter"):
            list(cv.split(X, labels, groups))

    def test_passing_groups_to_get_n_splits_raises(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        with pytest.raises(ValueError, match="bound to this splitter"):
            cv.get_n_splits(X, labels, groups)


# II Combinations and edge interactions
class TestBindGroupsComplex:
    """Test bind_groups() where several concerns cross."""

    def test_no_group_is_ever_in_both_parts(self):
        """KPI: over all folds, no group id is in both train and test."""
        X, labels, groups = _interleaved(n_groups=9, n_per_group=4)
        cv = aa.bind_groups(GroupKFold(n_splits=3), groups=groups)
        for train_idx, test_idx in cv.split(X, labels):
            assert not set(groups[train_idx]) & set(groups[test_idx])

    def test_every_sample_is_tested_exactly_once(self):
        """A partitioning splitter stays partitioning once bound."""
        X, labels, groups = _data(n_groups=8)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        tested = np.concatenate([test_idx for _, test_idx in cv.split(X, labels)])
        assert sorted(tested.tolist()) == list(range(len(labels)))

    def test_df_folds_is_none_before_the_folds_are_consumed(self):
        _, _, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert cv.df_folds_ is None

    def test_df_folds_has_one_row_per_fold_with_the_documented_columns(self):
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        list(cv.split(X, labels))
        assert isinstance(cv.df_folds_, pd.DataFrame)
        assert len(cv.df_folds_) == 4
        assert cv.df_folds_.columns.tolist() == ["fold", "n_train", "n_test", "n_groups_train",
                                                 "n_groups_test", "pos_rate_train", "pos_rate_test"]

    def test_df_folds_group_counts_sum_to_all_groups(self):
        X, labels, groups = _data(n_groups=8)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        list(cv.split(X, labels))
        totals = cv.df_folds_["n_groups_train"] + cv.df_folds_["n_groups_test"]
        assert (totals == 8).all()

    def test_df_folds_sample_counts_sum_to_n_samples(self):
        X, labels, groups = _data(n_groups=8, n_per_group=5)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        list(cv.split(X, labels))
        totals = cv.df_folds_["n_train"] + cv.df_folds_["n_test"]
        assert (totals == 40).all()

    def test_pos_rate_is_nan_without_labels(self):
        X, _, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        list(cv.split(X))
        assert cv.df_folds_["pos_rate_test"].isna().all()

    def test_pos_rate_matches_a_hand_computed_value(self):
        """Golden value: one protein per fold, labels alternate by protein."""
        X, labels, groups = _data(n_groups=4, n_per_group=5)
        cv = aa.bind_groups(LeaveOneGroupOut(), groups=groups)
        list(cv.split(X, labels))
        # Each held-out protein is entirely one class, so its test rate is 0.0 or 1.0
        assert sorted(cv.df_folds_["pos_rate_test"].tolist()) == [0.0, 0.0, 1.0, 1.0]

    def test_works_as_cv_in_sklearn_cross_val_score(self):
        X, labels, groups = _data(n_groups=8)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        scores = cross_val_score(RandomForestClassifier(n_estimators=10, random_state=0),
                                 X, labels, cv=cv)
        assert len(scores) == 4

    def test_aa_pred_eval_runs_unchanged(self):
        """KPI: AAPred.eval(cv=bind_groups(...)) runs with no change to eval."""
        X, labels, groups = _data(n_groups=10, n_per_group=4)
        cv = aa.bind_groups(StratifiedGroupKFold(n_splits=4), groups=groups)
        df_eval = aa.AAPred(verbose=False, random_state=42).eval(X, labels=labels,
                                                                 metrics=["accuracy"], cv=cv)
        assert len(df_eval) >= 1
        assert len(cv.df_folds_) == 4

    def test_grouping_lowers_an_inflated_score(self):
        """The leak the issue is about: near-identical windows of one protein.

        A random split scores partly on rows it effectively trained on, so the
        grouped score must not exceed the random one on this construction.
        """
        rng = np.random.default_rng(0)
        n_prot, n_win = 12, 6
        groups = np.repeat([f"P{i}" for i in range(n_prot)], n_win)
        labels = np.repeat(rng.integers(0, 2, n_prot), n_win)
        base = rng.random((n_prot, 8))
        X = np.repeat(base, n_win, axis=0) + rng.normal(0, 0.01, (n_prot * n_win, 8))
        aap = aa.AAPred(verbose=False, random_state=42)
        random_score = aap.eval(X, labels=labels, metrics=["accuracy"],
                                cv=StratifiedKFold(n_splits=4, shuffle=True,
                                                   random_state=42))["score"].iloc[0]
        grouped_score = aap.eval(X, labels=labels, metrics=["accuracy"],
                                 cv=aa.bind_groups(StratifiedGroupKFold(n_splits=4),
                                                   groups=groups))["score"].iloc[0]
        assert grouped_score <= random_score

    def test_reuse_of_one_bound_splitter_is_stable(self):
        """Two passes over the same object yield the same folds and overwrite df_folds_."""
        X, labels, groups = _data()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        first = [tuple(t.tolist()) for _, t in cv.split(X, labels)]
        n_rows_first = len(cv.df_folds_)
        second = [tuple(t.tolist()) for _, t in cv.split(X, labels)]
        assert first == second
        assert len(cv.df_folds_) == n_rows_first

    def test_seeded_splitter_is_reproducible(self):
        X, labels, groups = _data()
        kws = dict(n_splits=3, test_size=0.25, random_state=42)
        a = [t.tolist() for _, t in aa.bind_groups(GroupShuffleSplit(**kws),
                                                   groups=groups).split(X, labels)]
        b = [t.tolist() for _, t in aa.bind_groups(GroupShuffleSplit(**kws),
                                                   groups=groups).split(X, labels)]
        assert a == b

    def test_single_group_with_leave_one_group_out_raises(self):
        X, labels, groups = _data(n_groups=1, n_per_group=10)
        cv = aa.bind_groups(LeaveOneGroupOut(), groups=groups)
        with pytest.raises(ValueError):
            list(cv.split(X, labels))

    def test_repr_names_the_wrapped_splitter_and_group_count(self):
        _, _, groups = _data(n_groups=8)
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        assert "GroupKFold" in repr(cv)
        assert "n_groups=8" in repr(cv)

    def test_overlap_message_names_a_shared_group(self):
        X, labels, groups = _interleaved()
        cv = aa.bind_groups(KFold(n_splits=4), groups=groups)
        with pytest.raises(ValueError, match="P0"):
            list(cv.split(X, labels))
