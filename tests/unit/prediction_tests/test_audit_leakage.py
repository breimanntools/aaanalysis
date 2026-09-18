"""This is a script to test the aa.audit_leakage() function."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

AA = list("ACDEFGHIKLMNPQRSTVWY")
COLS = ["check", "severity", "detail", "ids"]


def _seqs(n=40, length=20, seed=0):
    """Draw n distinct random sequences."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(AA, length)) for _ in range(n)]


def _df_seq(n=40, seed=0):
    """Clean dataset: one row per protein, all sequences distinct, labels balanced."""
    return pd.DataFrame({"entry": [f"P{i}" for i in range(n)],
                         "sequence": _seqs(n=n, seed=seed),
                         "label": [i % 2 for i in range(n)]})


def _splits(n=40, n_splits=4, labels=None):
    """Stratified folds over n rows, as a materialized list of index pairs."""
    labels = labels if labels is not None else [i % 2 for i in range(n)]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return list(cv.split(np.zeros((n, 1)), labels))


def _df_windows(n_prot=10, n_win=4, seed=0):
    """Windowed dataset: several distinct windows per protein, label per protein."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_prot):
        parent = "".join(rng.choice(AA, 60))
        for w in range(n_win):
            rows.append({"entry": f"P{i}", "entry_win": f"P{i}_w{w}",
                         "sequence": parent, "window": parent[w:w + 10],
                         "label": i % 2})
    return pd.DataFrame(rows)


def _checks(df_audit):
    """Return the set of check names in a findings table."""
    return set(df_audit["check"])


# I Normal cases, one parameter per test
class TestAuditLeakage:
    """Test audit_leakage() parameter by parameter."""

    # df_seq
    def test_clean_dataset_is_ok(self):
        df_audit = aa.audit_leakage(_df_seq())
        assert len(df_audit) == 0
        assert df_audit.attrs["status"] == "ok"

    def test_returns_plain_dataframe_with_schema(self):
        df_audit = aa.audit_leakage(_df_seq())
        assert type(df_audit) is pd.DataFrame
        assert list(df_audit.columns) == COLS

    @settings(max_examples=5)
    @given(n=some.integers(min_value=4, max_value=30))
    def test_clean_dataset_of_any_size_is_ok(self, n):
        assert aa.audit_leakage(_df_seq(n=n)).attrs["status"] == "ok"

    def test_duplicate_sequence_is_flagged(self):
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq)
        assert "duplicate_sequences" in _checks(df_audit)
        assert df_audit.attrs["status"] == "medium"

    def test_duplicate_sequence_reports_affected_entries(self):
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        row = aa.audit_leakage(df_seq).set_index("check").loc["duplicate_sequences"]
        assert set(row["ids"]) == {"P0", "P1"}

    def test_df_seq_without_sequence_column_uses_entry(self):
        df_seq = _df_seq()[["entry", "label"]]
        assert aa.audit_leakage(df_seq).attrs["status"] == "ok"

    def test_df_seq_none_with_other_input_works(self):
        df_audit = aa.audit_leakage(groups=np.repeat(["a", "b"], 5),
                                    splits=[(np.arange(0, 5), np.arange(5, 10))])
        assert df_audit.attrs["status"] == "ok"

    def test_df_seq_is_not_mutated(self):
        df_seq = _df_seq()
        before = df_seq.copy()
        aa.audit_leakage(df_seq, splits=_splits())
        pd.testing.assert_frame_equal(df_seq, before)

    def test_df_seq_empty_raises(self):
        with pytest.raises(ValueError, match="at least one sequence"):
            aa.audit_leakage(pd.DataFrame({"entry": [], "sequence": []}))

    def test_df_seq_wrong_type_raises(self):
        for invalid in ["seq", 42, [1, 2, 3]]:
            with pytest.raises(ValueError):
                aa.audit_leakage(invalid)

    def test_df_seq_without_usable_columns_raises(self):
        with pytest.raises(ValueError, match="should contain at least one of"):
            aa.audit_leakage(pd.DataFrame({"foo": [1, 2, 3]}))

    # labels
    def test_labels_taken_from_df_seq_label_column(self):
        df_seq = _df_seq(n=20)
        df_seq["label"] = [0] * 10 + [1] * 10
        df_audit = aa.audit_leakage(df_seq, splits=[(np.arange(0, 10), np.arange(10, 20))])
        assert "class_balance_anomaly" in _checks(df_audit)

    def test_labels_explicit_override_df_seq(self):
        df_seq = _df_seq(n=20)
        labels = [i % 2 for i in range(20)]
        df_audit = aa.audit_leakage(df_seq, labels=labels, splits=_splits(n=20, labels=labels))
        assert "class_balance_anomaly" not in _checks(df_audit)

    def test_labels_single_class_is_accepted_not_raised(self):
        df_seq = _df_seq(n=20)
        df_audit = aa.audit_leakage(df_seq, labels=[1] * 20)
        assert df_audit.attrs["status"] == "ok"

    def test_labels_wrong_length_raises(self):
        with pytest.raises(ValueError, match="same samples"):
            aa.audit_leakage(_df_seq(n=20), labels=[0, 1, 0])

    def test_labels_two_dimensional_raises(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            aa.audit_leakage(_df_seq(n=4), labels=np.zeros((4, 2)))

    # groups
    def test_groups_overlap_is_flagged(self):
        groups = np.tile([f"P{i}" for i in range(10)], 4)
        splits = list(KFold(n_splits=4).split(np.zeros((40, 1))))
        df_audit = aa.audit_leakage(groups=groups, splits=splits)
        assert "group_overlap_across_folds" in _checks(df_audit)
        assert df_audit.attrs["status"] == "high"

    def test_groups_respected_by_group_splitter_is_ok(self):
        groups = np.repeat([f"P{i}" for i in range(10)], 4)
        splits = list(GroupKFold(n_splits=4).split(np.zeros((40, 1)), groups=groups))
        df_audit = aa.audit_leakage(groups=groups, splits=splits)
        assert "group_overlap_across_folds" not in _checks(df_audit)

    def test_groups_as_list_and_series_agree(self):
        groups = list(np.tile(["a", "b", "c", "d", "e"], 4))
        splits = list(KFold(n_splits=4).split(np.zeros((20, 1))))
        one = aa.audit_leakage(groups=groups, splits=splits)
        two = aa.audit_leakage(groups=pd.Series(groups), splits=splits)
        assert _checks(one) == _checks(two)

    def test_groups_wrong_length_raises(self):
        with pytest.raises(ValueError, match="same samples"):
            aa.audit_leakage(_df_seq(n=20), groups=["a", "b"])

    def test_groups_empty_raises(self):
        with pytest.raises(ValueError):
            aa.audit_leakage(groups=[], splits=[(np.array([0]), np.array([1]))])

    # splits
    def test_splits_none_runs_dataset_checks_only(self):
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq)
        assert _checks(df_audit) == {"duplicate_sequences"}

    def test_splits_accepts_a_generator(self):
        df_seq = _df_seq()
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        df_audit = aa.audit_leakage(df_seq, splits=cv.split(df_seq, df_seq["label"]))
        assert df_audit.attrs["status"] == "ok"

    @settings(max_examples=5)
    @given(n_splits=some.integers(min_value=2, max_value=5))
    def test_splits_any_feasible_fold_count_is_clean(self, n_splits):
        df_audit = aa.audit_leakage(_df_seq(), splits=_splits(n_splits=n_splits))
        assert df_audit.attrs["status"] == "ok"

    def test_splits_train_test_row_overlap_is_flagged(self):
        splits = [(np.array([0, 1, 2, 3]), np.array([3, 4]))]
        df_audit = aa.audit_leakage(_df_seq(n=10), splits=splits)
        assert "train_test_overlap" in _checks(df_audit)
        assert df_audit.attrs["status"] == "high"

    def test_splits_not_a_pair_raises(self):
        with pytest.raises(ValueError, match="pair of two index arrays"):
            aa.audit_leakage(_df_seq(n=10), splits=[(np.array([0]), np.array([1]), np.array([2]))])

    def test_splits_empty_raises(self):
        with pytest.raises(ValueError, match="at least one fold"):
            aa.audit_leakage(_df_seq(n=10), splits=[])

    def test_splits_boolean_mask_raises(self):
        mask = np.array([True] * 5 + [False] * 5)
        with pytest.raises(ValueError, match="integer row positions"):
            aa.audit_leakage(_df_seq(n=10), splits=[(mask, ~mask)])

    def test_splits_out_of_range_raises(self):
        with pytest.raises(ValueError, match="row positions within"):
            aa.audit_leakage(_df_seq(n=10), splits=[(np.array([0, 1]), np.array([99]))])

    # X and names
    def test_x_target_derived_feature_is_flagged(self):
        labels = np.array([i % 2 for i in range(40)])
        X = np.random.default_rng(0).random((40, 4))
        X[:, 2] = labels
        df_audit = aa.audit_leakage(X=X, labels=labels)
        assert "target_derived_feature" in _checks(df_audit)
        assert df_audit.attrs["status"] == "high"

    def test_x_independent_features_are_ok(self):
        labels = np.array([i % 2 for i in range(40)])
        X = np.random.default_rng(1).random((40, 4))
        assert aa.audit_leakage(X=X, labels=labels).attrs["status"] == "ok"

    def test_names_appear_in_the_finding(self):
        labels = np.array([i % 2 for i in range(40)])
        X = np.random.default_rng(0).random((40, 3))
        X[:, 1] = labels
        row = aa.audit_leakage(X=X, labels=labels, names=["a", "leaky", "c"])
        assert row.set_index("check").loc["target_derived_feature", "ids"] == ["leaky"]

    def test_names_default_to_dataframe_columns(self):
        labels = np.array([i % 2 for i in range(40)])
        X = pd.DataFrame({"a": np.random.default_rng(0).random(40), "b": labels * 1.0})
        row = aa.audit_leakage(X=X, labels=labels)
        assert row.set_index("check").loc["target_derived_feature", "ids"] == ["b"]

    def test_names_wrong_length_raises(self):
        labels = np.array([i % 2 for i in range(40)])
        X = np.random.default_rng(0).random((40, 3))
        with pytest.raises(ValueError, match="one name per feature"):
            aa.audit_leakage(X=X, labels=labels, names=["a", "b"])

    def test_x_without_labels_raises(self):
        X = np.random.default_rng(0).random((40, 3))
        with pytest.raises(ValueError, match="'labels' should be given together with 'X'"):
            aa.audit_leakage(X=X)

    # raise_on
    def test_raise_on_none_never_raises(self):
        groups = np.tile(["a", "b"], 10)
        splits = list(KFold(n_splits=2).split(np.zeros((20, 1))))
        df_audit = aa.audit_leakage(groups=groups, splits=splits)
        assert df_audit.attrs["status"] == "high"

    def test_raise_on_high_raises_when_high_finding_exists(self):
        groups = np.tile(["a", "b"], 10)
        splits = list(KFold(n_splits=2).split(np.zeros((20, 1))))
        with pytest.raises(ValueError, match="group_overlap_across_folds"):
            aa.audit_leakage(groups=groups, splits=splits, raise_on="high")

    def test_raise_on_high_does_not_raise_without_high_finding(self):
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq, raise_on="high")
        assert df_audit.attrs["status"] == "medium"

    def test_raise_on_medium_raises_on_medium_finding(self):
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        with pytest.raises(ValueError, match="duplicate_sequences"):
            aa.audit_leakage(df_seq, raise_on="medium")

    def test_raise_on_never_raises_on_clean_setup(self):
        for severity in ["low", "medium", "high"]:
            assert aa.audit_leakage(_df_seq(), raise_on=severity).attrs["status"] == "ok"

    def test_raise_on_invalid_option_raises(self):
        for invalid in ["critical", "HIGH", "severe", 1]:
            with pytest.raises(ValueError):
                aa.audit_leakage(_df_seq(), raise_on=invalid)

    # general
    def test_no_input_at_all_raises(self):
        with pytest.raises(ValueError, match="nothing to audit"):
            aa.audit_leakage()

    def test_every_check_name_is_registered(self):
        import aaanalysis.utils as ut
        df_seq = _df_seq()
        df_seq.loc[1, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq, splits=[(np.arange(0, 39), np.array([39, 1]))])
        assert _checks(df_audit).issubset(set(ut.LIST_CHECKS_LEAKAGE))


# II Complex cases
class TestAuditLeakageComplex:
    """Test audit_leakage() with interacting parameters and edge cases."""

    def test_duplicate_sequence_split_across_folds_is_high(self):
        """KPI: a duplicated sequence placed in two folds is reported."""
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        splits = [(np.arange(0, 19), np.array([19]))]
        df_audit = aa.audit_leakage(df_seq, splits=splits)
        assert "duplicate_sequences_across_folds" in _checks(df_audit)
        row = df_audit.set_index("check").loc["duplicate_sequences_across_folds"]
        assert row["severity"] == "high"
        assert set(row["ids"]) == {"P0", "P19"}

    def test_clean_setup_with_every_input_given_is_ok(self):
        """KPI: a clean setup returns an empty table with status ok."""
        df_seq = _df_seq(n=40)
        labels = df_seq["label"].to_numpy()
        groups = np.array([f"G{i}" for i in range(40)])
        X = np.random.default_rng(3).random((40, 5))
        df_audit = aa.audit_leakage(df_seq, labels=labels, groups=groups,
                                    splits=_splits(n=40, labels=labels), X=X,
                                    names=[f"f{i}" for i in range(5)])
        assert len(df_audit) == 0
        assert df_audit.attrs["status"] == "ok"
        assert list(df_audit.columns) == COLS

    def test_windows_of_one_protein_split_apart_is_flagged(self):
        df_seq = _df_windows(n_prot=10, n_win=4)
        splits = list(KFold(n_splits=4, shuffle=True, random_state=0).split(df_seq))
        df_audit = aa.audit_leakage(df_seq, splits=splits)
        assert "same_protein_across_folds" in _checks(df_audit)

    def test_windows_kept_whole_by_group_splitter_is_clean(self):
        df_seq = _df_windows(n_prot=12, n_win=4)
        groups = df_seq["entry"].to_numpy()
        splits = list(GroupKFold(n_splits=4).split(df_seq, groups=groups))
        df_audit = aa.audit_leakage(df_seq, groups=groups, splits=splits)
        assert "same_protein_across_folds" not in _checks(df_audit)
        assert "group_overlap_across_folds" not in _checks(df_audit)

    def test_windowed_parent_sequence_is_not_called_a_duplicate(self):
        """The parent 'sequence' repeats per window; only 'window' may flag duplicates."""
        df_seq = _df_windows(n_prot=10, n_win=4)
        assert "duplicate_sequences" not in _checks(aa.audit_leakage(df_seq))

    def test_repeated_window_is_flagged_as_duplicate(self):
        df_seq = _df_windows(n_prot=10, n_win=4)
        df_seq.loc[5, "window"] = df_seq.loc[0, "window"]
        row = aa.audit_leakage(df_seq).set_index("check").loc["duplicate_sequences"]
        assert set(row["ids"]) == {"P0_w0", "P1_w1"}

    def test_findings_are_sorted_worst_first(self):
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq, splits=[(np.arange(0, 19), np.array([19]))])
        order = {"high": 0, "medium": 1, "low": 2}
        ranks = [order[s] for s in df_audit["severity"]]
        assert ranks == sorted(ranks)

    def test_status_is_the_worst_severity_present(self):
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq, splits=[(np.arange(0, 19), np.array([19]))])
        assert df_audit.attrs["status"] == "high"
        assert "high" in set(df_audit["severity"])

    def test_single_class_test_fold_is_high(self):
        df_seq = _df_seq(n=20)
        labels = np.array([0] * 10 + [1] * 10)
        splits = [(np.arange(0, 10), np.arange(10, 20))]
        df_audit = aa.audit_leakage(df_seq, labels=labels, splits=splits)
        row = df_audit.set_index("check").loc["class_balance_anomaly"]
        assert row["severity"] == "high"

    def test_uneven_fold_sizes_are_low(self):
        df_seq = _df_seq(n=30)
        labels = df_seq["label"].to_numpy()
        splits = [(np.arange(0, 20), np.arange(20, 30)), (np.arange(10, 30), np.arange(0, 2))]
        df_audit = aa.audit_leakage(df_seq, labels=labels, splits=splits)
        row = df_audit.set_index("check").loc["fold_size_anomaly"]
        assert row["severity"] == "low"

    def test_empty_fold_part_is_high(self):
        df_seq = _df_seq(n=20)
        splits = [(np.arange(0, 20), np.array([], dtype=int))]
        df_audit = aa.audit_leakage(df_seq, splits=splits)
        row = df_audit.set_index("check").loc["fold_size_anomaly"]
        assert row["severity"] == "high"

    def test_bind_groups_output_is_a_valid_splits_input(self):
        """A split from bind_groups feeds straight into splits, and shows no group leak."""
        df_seq = _df_windows(n_prot=12, n_win=4)
        groups = df_seq["entry"].to_numpy()
        cv = aa.bind_groups(GroupKFold(n_splits=4), groups=groups)
        df_audit = aa.audit_leakage(df_seq, groups=groups, splits=cv.split(df_seq))
        assert list(df_audit.columns) == COLS
        assert _checks(df_audit).isdisjoint({"group_overlap_across_folds",
                                             "same_protein_across_folds",
                                             "train_test_overlap"})

    def test_plain_kfold_on_windows_leaks_where_bind_groups_does_not(self):
        """The audit separates a leaky ungrouped split from the group-aware one."""
        df_seq = _df_windows(n_prot=12, n_win=4)
        groups = df_seq["entry"].to_numpy()
        leaky = list(KFold(n_splits=4, shuffle=True, random_state=0).split(df_seq))
        clean = list(aa.bind_groups(GroupKFold(n_splits=4), groups=groups).split(df_seq))
        assert "same_protein_across_folds" in _checks(aa.audit_leakage(df_seq, splits=leaky))
        assert "same_protein_across_folds" not in _checks(aa.audit_leakage(df_seq, splits=clean))

    def test_non_default_index_is_audited_by_position(self):
        df_seq = _df_seq(n=20)
        df_seq.index = [f"row{i}" for i in range(20)]
        df_seq.loc["row19", "sequence"] = df_seq.loc["row0", "sequence"]
        df_audit = aa.audit_leakage(df_seq, splits=[(np.arange(0, 19), np.array([19]))])
        assert "duplicate_sequences_across_folds" in _checks(df_audit)

    def test_ids_are_capped_at_ten(self):
        df_seq = _df_seq(n=40)
        df_seq["sequence"] = df_seq.loc[0, "sequence"]
        row = aa.audit_leakage(df_seq).set_index("check").loc["duplicate_sequences"]
        assert len(row["ids"]) == 10
        assert "40 rows" in row["detail"]

    def test_mismatched_lengths_across_inputs_raise(self):
        with pytest.raises(ValueError, match="same samples"):
            aa.audit_leakage(_df_seq(n=20), groups=["a"] * 20,
                             X=np.random.default_rng(0).random((10, 3)),
                             labels=[0, 1] * 5)

    def test_splits_out_of_range_for_groups_raises(self):
        with pytest.raises(ValueError, match="row positions within"):
            aa.audit_leakage(groups=["a", "b", "c"], splits=[(np.array([0]), np.array([7]))])

    def test_several_independent_findings_are_separate_rows(self):
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        labels = np.array([0] * 10 + [1] * 10)
        df_audit = aa.audit_leakage(df_seq, labels=labels,
                                    splits=[(np.arange(0, 10), np.arange(10, 20))])
        assert len(_checks(df_audit)) >= 2
        assert len(df_audit) == len(df_audit.drop_duplicates(subset=["check", "detail"]))

    def test_repeated_calls_are_deterministic(self):
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        splits = [(np.arange(0, 19), np.array([19]))]
        one = aa.audit_leakage(df_seq, splits=splits)
        two = aa.audit_leakage(df_seq, splits=splits)
        pd.testing.assert_frame_equal(one, two)
        assert one.attrs["status"] == two.attrs["status"]

    @settings(max_examples=5)
    @given(n_prot=some.integers(min_value=4, max_value=10))
    def test_group_splitter_never_flags_group_overlap(self, n_prot):
        df_seq = _df_windows(n_prot=n_prot, n_win=3)
        groups = df_seq["entry"].to_numpy()
        splits = list(GroupKFold(n_splits=2).split(df_seq, groups=groups))
        df_audit = aa.audit_leakage(groups=groups, splits=splits)
        assert "group_overlap_across_folds" not in _checks(df_audit)

    def test_status_survives_on_a_filtered_copy_only_via_attrs(self):
        df_seq = _df_seq(n=20)
        df_seq.loc[19, "sequence"] = df_seq.loc[0, "sequence"]
        df_audit = aa.audit_leakage(df_seq)
        assert df_audit.attrs["status"] == "medium"
        assert "status" not in df_audit.columns
