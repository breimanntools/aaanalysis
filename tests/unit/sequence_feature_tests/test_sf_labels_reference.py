"""Tests for the SequenceFeature label-conversion and reference-assembly methods.

Covers get_labels_ovr, get_labels_ovo, get_labels_quantile, get_labels_tiered, and
get_df_parts_from_windows (issue #61: multi-class & regression CPP). Follows the house
testing template: a normal-case ``Test<Method>`` class (one parameter per test, positive
via hypothesis + negative via pytest.raises), a ``Test<Method>Complex`` class crossing
parameters, and a ``Test<Method>GoldenValues`` class with hand-computed expectations.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

SF = aa.SequenceFeature


# I Helper functions
def _multiclass(n_classes, per_class=3):
    """Balanced integer multi-class label list, e.g. [0,0,0,1,1,1,...]."""
    return [c for c in range(n_classes) for _ in range(per_class)]


def _toy_windows(n=3):
    return {
        "jmd_n": ["AAAA", "CDEC", "DDWD", "KKRR", "STST"][:n],
        "tmd": ["EEFEEY", "FFLFFF", "GMGGGG", "HVHVHV", "ILILIL"][:n],
        "jmd_c": ["HHHK", "IIRI", "KKQK", "PPNP", "QQWQ"][:n],
    }


def _toy_df_parts(n):
    """df_parts of n rows (default RangeIndex) with the three default parts as strings."""
    pool = ["AAAA", "CDEC", "DDWD", "KKRR", "STST", "EYEY",
            "FLFL", "GMGM", "HVHV", "ILIL", "PNPN", "QWQW"]
    seqs = [pool[i % len(pool)] for i in range(n)]
    return pd.DataFrame({"jmd_n": seqs, "tmd": [s + "YY" for s in seqs], "jmd_c": seqs})


def _toy_dict_num_parts(n, part_len=4, d=2):
    """{part: (n, part_len, d)} numerical tensor with sample axis first."""
    return {"tmd": np.arange(n * part_len * d, dtype=float).reshape(n, part_len, d)}


# ======================================================================================
# get_labels_ovr
# ======================================================================================
class TestGetLabelsOvr:
    """Normal cases for get_labels_ovr (one parameter per test)."""

    @settings(max_examples=5, deadline=None)
    @given(n_classes=some.integers(min_value=2, max_value=6))
    def test_labels_n_classes(self, n_classes):
        d = SF.get_labels_ovr(_multiclass(n_classes))
        assert list(d.keys()) == list(range(n_classes))

    @settings(max_examples=5, deadline=None)
    @given(per_class=some.integers(min_value=1, max_value=5))
    def test_labels_full_length(self, per_class):
        labels = _multiclass(3, per_class=per_class)
        d = SF.get_labels_ovr(labels)
        assert all(len(v) == len(labels) for v in d.values())

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=2, max_value=9))
    def test_label_test(self, label_test):
        d = SF.get_labels_ovr([0, 1, 2], label_test=label_test, label_ref=0)
        assert set(np.unique(d[1])).issubset({label_test, 0})
        assert d[1][1] == label_test

    @settings(max_examples=5, deadline=None)
    @given(label_ref=some.integers(min_value=-9, max_value=-1))
    def test_label_ref(self, label_ref):
        d = SF.get_labels_ovr([0, 1, 2], label_test=1, label_ref=label_ref)
        assert d[1][0] == label_ref

    def test_keys_sorted(self):
        d = SF.get_labels_ovr([2, 2, 0, 1])
        assert list(d.keys()) == [0, 1, 2]

    def test_keys_are_python_int(self):
        d = SF.get_labels_ovr(np.array([0, 1, 2]))
        assert all(isinstance(k, int) for k in d)

    # Negative
    def test_none_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr(None)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([1, 1, 1])

    def test_float_labels_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0.0, 1.0, 2.0])

    def test_string_labels_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr(["a", "b", "c"])

    def test_equal_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0, 1], label_test=1, label_ref=1)

    @settings(max_examples=5, deadline=None)
    @given(bad=some.floats(min_value=0.1, max_value=0.9))
    def test_float_label_test_raises(self, bad):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0, 1, 2], label_test=bad)


class TestGetLabelsOvrComplex:
    """Cross-parameter cases for get_labels_ovr."""

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=1, max_value=5),
           label_ref=some.integers(min_value=-5, max_value=0))
    def test_custom_test_ref_combo(self, label_test, label_ref):
        d = SF.get_labels_ovr([0, 1, 2], label_test=label_test, label_ref=label_ref)
        assert set(np.unique(d[0])) == {label_test, label_ref}

    @settings(max_examples=5, deadline=None)
    @given(n_classes=some.integers(min_value=2, max_value=5),
           per_class=some.integers(min_value=1, max_value=4))
    def test_each_vector_binary_and_sums(self, n_classes, per_class):
        labels = _multiclass(n_classes, per_class=per_class)
        d = SF.get_labels_ovr(labels)
        for c, v in d.items():
            assert set(np.unique(v)).issubset({0, 1})
            assert v.sum() == per_class

    def test_negative_classes_allowed(self):
        d = SF.get_labels_ovr([-1, -1, 0, 2])
        assert list(d.keys()) == [-1, 0, 2]

    def test_imbalanced_partition(self):
        labels = [0, 0, 0, 0, 1, 1, 2]
        d = SF.get_labels_ovr(labels)
        assert d[0].sum() == 4 and d[1].sum() == 2 and d[2].sum() == 1

    def test_two_classes_complement(self):
        d = SF.get_labels_ovr([0, 0, 1, 1])
        assert (d[0] == 1 - d[1]).all()

    # Negative combos
    def test_equal_negative_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0, 1, 2], label_test=-3, label_ref=-3)

    def test_mixed_float_int_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0, 1, 2.0])

    def test_scalar_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr(5)

    def test_bare_string_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr("abc")

    def test_single_repeated_value_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([5, 5, 5, 5])


class TestGetLabelsOvrGoldenValues:
    """Hand-computed expectations for get_labels_ovr."""

    def test_membership(self):
        d = SF.get_labels_ovr([0, 0, 1, 1, 2, 2])
        assert d[0].tolist() == [1, 1, 0, 0, 0, 0]
        assert d[1].tolist() == [0, 0, 1, 1, 0, 0]
        assert d[2].tolist() == [0, 0, 0, 0, 1, 1]


# ======================================================================================
# get_labels_ovo
# ======================================================================================
class TestGetLabelsOvo:
    """Normal cases for get_labels_ovo (one parameter per test)."""

    @settings(max_examples=5, deadline=None)
    @given(n_classes=some.integers(min_value=2, max_value=6))
    def test_n_pairs(self, n_classes):
        labels = _multiclass(n_classes)
        o = SF.get_labels_ovo(labels, df_parts=_toy_df_parts(len(labels)))
        assert len(o) == n_classes * (n_classes - 1) // 2

    @settings(max_examples=5, deadline=None)
    @given(per_class=some.integers(min_value=1, max_value=4))
    def test_subset_rows_match_labels(self, per_class):
        labels = _multiclass(3, per_class=per_class)
        o = SF.get_labels_ovo(labels, df_parts=_toy_df_parts(len(labels)))
        assert all(len(dps) == len(y) for dps, _, y in o.values())

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=2, max_value=9))
    def test_label_test(self, label_test):
        o = SF.get_labels_ovo([0, 0, 1, 1], df_parts=_toy_df_parts(4),
                              label_test=label_test, label_ref=0)
        _, _, binary = o[(0, 1)]
        assert set(np.unique(binary)).issubset({label_test, 0})

    @settings(max_examples=5, deadline=None)
    @given(label_ref=some.integers(min_value=-9, max_value=-1))
    def test_label_ref(self, label_ref):
        o = SF.get_labels_ovo([0, 0, 1, 1], df_parts=_toy_df_parts(4),
                              label_test=1, label_ref=label_ref)
        _, _, binary = o[(0, 1)]
        assert label_ref in np.unique(binary)

    def test_pairs_sorted(self):
        o = SF.get_labels_ovo([2, 1, 0], df_parts=_toy_df_parts(3))
        assert list(o.keys()) == [(0, 1), (0, 2), (1, 2)]

    def test_keys_are_int_tuples(self):
        o = SF.get_labels_ovo(np.array([0, 1, 2]), df_parts=_toy_df_parts(3))
        assert all(isinstance(a, int) and isinstance(b, int) for a, b in o)

    def test_df_parts_only_returns_dnp_none(self):
        o = SF.get_labels_ovo([0, 1, 2], df_parts=_toy_df_parts(3))
        assert all(dps is not None and dnp is None for dps, dnp, _ in o.values())

    def test_dict_num_parts_only_returns_df_parts_none(self):
        o = SF.get_labels_ovo([0, 0, 1, 1], dict_num_parts=_toy_dict_num_parts(4))
        assert all(dps is None and dnp is not None for dps, dnp, _ in o.values())

    # Negative
    def test_none_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo(None, df_parts=_toy_df_parts(3))

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([1, 1, 1], df_parts=_toy_df_parts(3))

    def test_float_labels_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([0.0, 1.0], df_parts=_toy_df_parts(2))

    def test_string_labels_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo(["x", "y"], df_parts=_toy_df_parts(2))

    def test_equal_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([0, 1], df_parts=_toy_df_parts(2), label_test=1, label_ref=1)

    def test_scalar_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo(7, df_parts=_toy_df_parts(3))

    def test_no_value_source_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SF.get_labels_ovo([0, 1, 2])


class TestGetLabelsOvoComplex:
    """Cross-parameter cases for get_labels_ovo."""

    @settings(max_examples=5, deadline=None)
    @given(n_classes=some.integers(min_value=2, max_value=5))
    def test_subset_selects_exactly_pair(self, n_classes):
        labels = np.array(_multiclass(n_classes))
        o = SF.get_labels_ovo(labels, df_parts=_toy_df_parts(len(labels)))
        for (a, b), (dps, _, binary) in o.items():
            assert set(np.unique(labels[dps.index.to_numpy()])) == {a, b}
            assert len(binary) == len(dps)

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=1, max_value=5),
           label_ref=some.integers(min_value=-5, max_value=0))
    def test_binary_values_combo(self, label_test, label_ref):
        o = SF.get_labels_ovo([0, 0, 1, 1, 2, 2], df_parts=_toy_df_parts(6),
                              label_test=label_test, label_ref=label_ref)
        for _, _, binary in o.values():
            assert set(np.unique(binary)).issubset({label_test, label_ref})

    def test_binary_assignment_a_is_test(self):
        o = SF.get_labels_ovo([0, 0, 1, 1], df_parts=_toy_df_parts(4))
        _, _, binary = o[(0, 1)]
        assert binary.tolist() == [1, 1, 0, 0]

    def test_imbalanced_pair(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2])
        o = SF.get_labels_ovo(labels, df_parts=_toy_df_parts(len(labels)))
        dps, _, binary = o[(0, 1)]
        assert len(dps) == 6 and binary.sum() == 5

    def test_three_classes_three_pairs(self):
        o = SF.get_labels_ovo([0, 1, 2, 0, 1, 2], df_parts=_toy_df_parts(6))
        assert set(o.keys()) == {(0, 1), (0, 2), (1, 2)}

    def test_both_value_sources_subset_aligned(self):
        labels = [0, 0, 1, 1, 2, 2]
        o = SF.get_labels_ovo(labels, df_parts=_toy_df_parts(6),
                              dict_num_parts=_toy_dict_num_parts(6))
        for dps, dnp, y in o.values():
            assert len(dps) == dnp["tmd"].shape[0] == len(y)

    def test_inputs_not_mutated(self):
        dp = _toy_df_parts(6)
        dnp = _toy_dict_num_parts(6)
        dp_copy, arr_copy = dp.copy(deep=True), dnp["tmd"].copy()
        o = SF.get_labels_ovo([0, 0, 1, 1, 2, 2], df_parts=dp, dict_num_parts=dnp)
        dps, dnps, _ = o[(0, 1)]
        dps.iloc[0, 0] = "ZZZZ"
        dnps["tmd"][0, 0, 0] = -999.0
        assert dp.equals(dp_copy)
        assert np.array_equal(dnp["tmd"], arr_copy)

    # Negative combos
    def test_equal_test_ref_negative_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([0, 1, 2], df_parts=_toy_df_parts(3), label_test=4, label_ref=4)

    def test_float_label_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([0, 1], df_parts=_toy_df_parts(2), label_ref=0.5)

    def test_mixed_types_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovo([0, 1, "2"], df_parts=_toy_df_parts(3))

    def test_df_parts_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            SF.get_labels_ovo([0, 1, 2], df_parts=_toy_df_parts(4))

    def test_dict_num_parts_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="axis-0"):
            SF.get_labels_ovo([0, 1, 2], dict_num_parts=_toy_dict_num_parts(4))


class TestGetLabelsOvoGoldenValues:
    """Hand-computed expectations for get_labels_ovo."""

    def test_pair_subset_and_binary(self):
        o = SF.get_labels_ovo([0, 0, 1, 1, 2, 2], df_parts=_toy_df_parts(6))
        dps, _, binary = o[(0, 1)]
        assert dps.index.tolist() == [0, 1, 2, 3]
        assert binary.tolist() == [1, 1, 0, 0]
        dps02, _, binary02 = o[(0, 2)]
        assert dps02.index.tolist() == [0, 1, 4, 5]
        assert binary02.tolist() == [1, 1, 0, 0]


# ======================================================================================
# get_labels_quantile
# ======================================================================================
class TestGetLabelsQuantile:
    """Normal cases for get_labels_quantile (one parameter per test)."""

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=4, max_value=30))
    def test_targets_length_preserved(self, n):
        labels = SF.get_labels_quantile(list(np.linspace(0, 1, n)), q=0.5)
        assert len(labels) == n

    @settings(max_examples=8, deadline=None)
    @given(q=some.floats(min_value=0.1, max_value=0.9))
    def test_q_two_classes(self, q):
        labels = SF.get_labels_quantile(list(np.linspace(0, 100, 50)), q=q)
        assert set(np.unique(labels)) == {0, 1}

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=2, max_value=9))
    def test_label_test(self, label_test):
        labels = SF.get_labels_quantile([1.0, 2, 3, 4], q=0.5, label_test=label_test, label_ref=0)
        assert label_test in np.unique(labels)

    @settings(max_examples=5, deadline=None)
    @given(label_ref=some.integers(min_value=-9, max_value=-1))
    def test_label_ref(self, label_ref):
        labels = SF.get_labels_quantile([1.0, 2, 3, 4], q=0.5, label_test=1, label_ref=label_ref)
        assert label_ref in np.unique(labels)

    def test_binary_only(self):
        labels = SF.get_labels_quantile([0.0, 0.2, 0.8, 1.0], q=0.5)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_float32_input_ok(self):
        labels = SF.get_labels_quantile(np.array([1, 2, 3, 4], dtype=np.float32), q=0.5)
        assert labels.tolist() == [0, 0, 1, 1]

    # Negative
    def test_none_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile(None)

    def test_constant_targets_raise(self):
        with pytest.raises(ValueError, match="single class"):
            SF.get_labels_quantile([5.0, 5.0, 5.0], q=0.5)

    @settings(max_examples=5, deadline=None)
    @given(q=some.floats(min_value=1.0001, max_value=5.0))
    def test_q_too_high_raises(self, q):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=q)

    @settings(max_examples=5, deadline=None)
    @given(q=some.floats(min_value=-5.0, max_value=0.0))
    def test_q_too_low_raises(self, q):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=q)

    def test_equal_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=0.5, label_test=1, label_ref=1)

    def test_float_label_test_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], label_test=0.5)


class TestGetLabelsQuantileComplex:
    """Cross-parameter cases for get_labels_quantile."""

    @settings(max_examples=5, deadline=None)
    @given(q=some.floats(min_value=0.2, max_value=0.8),
           label_test=some.integers(min_value=1, max_value=4))
    def test_q_and_test_combo(self, q, label_test):
        labels = SF.get_labels_quantile(list(np.linspace(0, 1, 20)), q=q,
                                        label_test=label_test, label_ref=0)
        assert set(np.unique(labels)).issubset({label_test, 0})

    def test_reproducible_deterministic(self):
        t = list(np.linspace(0, 1, 17))
        assert SF.get_labels_quantile(t, q=0.4).tolist() == SF.get_labels_quantile(t, q=0.4).tolist()

    def test_median_balanced(self):
        labels = SF.get_labels_quantile(list(range(10)), q=0.5)
        assert labels.sum() == 5

    def test_high_q_few_positives(self):
        labels = SF.get_labels_quantile(list(range(100)), q=0.9)
        assert labels.sum() == 10

    def test_custom_labels_combo(self):
        labels = SF.get_labels_quantile([1.0, 2, 3, 4], q=0.5, label_test=7, label_ref=3)
        assert set(np.unique(labels)) == {7, 3}

    # Negative combos
    def test_constant_targets_combo_raises(self):
        with pytest.raises(ValueError, match="single class"):
            SF.get_labels_quantile([0.0, 0.0, 0.0, 0.0, 0.0], q=0.5)

    def test_equal_custom_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=0.5, label_test=2, label_ref=2)

    def test_string_targets_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile(["a", "b", "c"])

    def test_q_zero_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=0.0)

    def test_q_one_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=1.0)


class TestGetLabelsQuantileGoldenValues:
    """Hand-computed expectations for get_labels_quantile."""

    def test_median_split(self):
        # cut = quantile([1..6], 0.5) = 3.5 -> >=3.5 are test
        assert SF.get_labels_quantile([1.0, 2, 3, 4, 5, 6], q=0.5).tolist() == [0, 0, 0, 1, 1, 1]

    def test_q75(self):
        # quantile([10,20,30,40], 0.75) = 32.5 -> only 40 >= cut
        assert SF.get_labels_quantile([10.0, 20, 30, 40], q=0.75).tolist() == [0, 0, 0, 1]


# ======================================================================================
# get_labels_tiered
# ======================================================================================
class TestGetLabelsTiered:
    """Normal cases for get_labels_tiered (one parameter per test)."""

    @settings(max_examples=5, deadline=None)
    @given(q_pos=some.floats(min_value=0.55, max_value=0.9))
    def test_q_pos_keys_unchanged(self, q_pos):
        t = list(np.linspace(0, 1, 40))
        tiers = SF.get_labels_tiered(t, q_pos=q_pos, list_q_neg=[0.5, 0.3], df_parts=_toy_df_parts(len(t)))
        assert list(tiers.keys()) == [0.5, 0.3]

    @settings(max_examples=5, deadline=None)
    @given(n_neg=some.integers(min_value=1, max_value=4))
    def test_list_q_neg_length(self, n_neg):
        t = list(np.linspace(0, 1, 50))
        q_negs = list(np.linspace(0.1, 0.6, n_neg))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=q_negs, df_parts=_toy_df_parts(len(t)))
        assert len(tiers) == n_neg

    def test_positives_fixed_across_tiers(self):
        t = list(np.linspace(0, 1, 50))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.7, 0.5, 0.3], df_parts=_toy_df_parts(len(t)))
        n_pos = [int((y == 1).sum()) for _, _, y in tiers.values()]
        assert len(set(n_pos)) == 1  # same positive count every tier

    def test_negatives_shrink_as_q_neg_drops(self):
        t = list(np.linspace(0, 1, 50))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.7, 0.5, 0.3], df_parts=_toy_df_parts(len(t)))
        n_neg = [int((y == 0).sum()) for _, _, y in tiers.values()]
        assert n_neg[0] >= n_neg[1] >= n_neg[2]

    @settings(max_examples=5, deadline=None)
    @given(label_test=some.integers(min_value=2, max_value=9))
    def test_label_test(self, label_test):
        t = list(np.linspace(0, 1, 30))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(len(t)),
                                     label_test=label_test, label_ref=0)
        _, _, y = tiers[0.3]
        assert label_test in np.unique(y)

    def test_subset_drops_middle(self):
        t = list(np.linspace(0, 1, 50))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(len(t)))
        dps, _, y = tiers[0.3]
        assert len(dps) == len(y) < len(t)  # middle band dropped

    def test_dict_num_parts_only(self):
        t = list(np.linspace(0, 1, 20))
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], dict_num_parts=_toy_dict_num_parts(len(t)))
        dps, dnp, y = tiers[0.3]
        assert dps is None and dnp["tmd"].shape[0] == len(y)

    # Negative
    def test_none_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(None, df_parts=_toy_df_parts(3))

    def test_q_pos_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered([1.0, 2, 3], q_pos=1.5, list_q_neg=[0.3], df_parts=_toy_df_parts(3))

    def test_q_neg_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[1.5], df_parts=_toy_df_parts(10))

    def test_constant_targets_single_class_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered([5.0, 5.0, 5.0, 5.0], q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(4))

    def test_equal_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(10),
                                 label_test=1, label_ref=1)

    def test_q_pos_zero_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.0, list_q_neg=[0.3], df_parts=_toy_df_parts(10))

    def test_no_value_source_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[0.3])


class TestGetLabelsTieredComplex:
    """Cross-parameter cases for get_labels_tiered."""

    @settings(max_examples=5, deadline=None)
    @given(q_pos=some.floats(min_value=0.6, max_value=0.85))
    def test_each_tier_binary_subset(self, q_pos):
        t = np.linspace(0, 1, 60)
        tiers = SF.get_labels_tiered(t, q_pos=q_pos, list_q_neg=[0.5, 0.3], df_parts=_toy_df_parts(len(t)))
        for q_neg, (dps, _, y) in tiers.items():
            assert len(y) == len(dps)
            assert set(np.unique(y)).issubset({0, 1})

    def test_positives_match_quantile_cut(self):
        t = np.arange(10).astype(float)
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(len(t)))
        _, _, y = tiers[0.3]
        cut_pos = np.quantile(t, 0.8)
        assert int((y == 1).sum()) == int((t >= cut_pos).sum())

    def test_custom_labels_combo(self):
        t = np.linspace(0, 1, 40)
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(len(t)),
                                     label_test=5, label_ref=2)
        _, _, y = tiers[0.3]
        assert set(np.unique(y)).issubset({5, 2})

    def test_single_tier(self):
        t = np.linspace(0, 1, 20)
        tiers = SF.get_labels_tiered(t, q_pos=0.7, list_q_neg=[0.3], df_parts=_toy_df_parts(len(t)))
        assert list(tiers.keys()) == [0.3]

    def test_deterministic(self):
        t = np.linspace(0, 1, 25)
        dp = _toy_df_parts(len(t))
        a = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.5, 0.3], df_parts=dp)
        b = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.5, 0.3], df_parts=dp)
        assert all((a[k][2] == b[k][2]).all() for k in a)

    def test_inputs_not_mutated(self):
        t = np.linspace(0, 1, 20)
        dp = _toy_df_parts(len(t))
        dp_copy = dp.copy(deep=True)
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=dp)
        dps, _, _ = tiers[0.3]
        dps.iloc[0, 0] = "ZZZZ"
        assert dp.equals(dp_copy)

    # Negative combos
    def test_empty_list_q_neg_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[], df_parts=_toy_df_parts(10))

    def test_float_label_test_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(10),
                                 label_test=0.5)

    def test_q_neg_zero_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[0.0], df_parts=_toy_df_parts(10))

    def test_string_targets_raise(self):
        with pytest.raises(ValueError):
            SF.get_labels_tiered(["a", "b", "c"], q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(3))

    def test_df_parts_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            SF.get_labels_tiered(list(range(10)), q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(8))


class TestGetLabelsTieredGoldenValues:
    """Hand-computed expectations for get_labels_tiered."""

    def test_subset_index_and_binary(self):
        t = np.arange(10).astype(float)  # 0..9
        tiers = SF.get_labels_tiered(t, q_pos=0.8, list_q_neg=[0.3], df_parts=_toy_df_parts(10))
        # cut_pos = quantile(0..9, 0.8) = 7.2 -> pos = {8,9}
        # cut_neg = quantile(0..9, 0.3) = 2.7 -> neg = {0,1,2}
        dps, _, y = tiers[0.3]
        assert dps.index.tolist() == [0, 1, 2, 8, 9]  # neg rows then pos rows kept; middle dropped
        assert y.tolist() == [0, 0, 0, 1, 1]


# ======================================================================================
# get_df_parts_from_windows
# ======================================================================================
class TestGetDfPartsFromWindows:
    """Normal cases for get_df_parts_from_windows (one parameter per test)."""

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=5))
    def test_n_rows(self, n):
        df = SF.get_df_parts_from_windows(_toy_windows(n))
        assert len(df) == n

    def test_columns_match_keys(self):
        df = SF.get_df_parts_from_windows(_toy_windows(3))
        assert list(df) == ["jmd_n", "tmd", "jmd_c"]

    def test_ref_index(self):
        df = SF.get_df_parts_from_windows(_toy_windows(2))
        assert df.index.tolist() == ["REF0", "REF1"]

    def test_from_aawindowsampler_output(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        aws = aa.AAWindowSampler(random_state=0)
        d = {
            "jmd_n": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=10, generator="coil"),
            "tmd": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=20, generator="alpha_helix"),
            "jmd_c": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=10, generator="coil"),
        }
        df = SF.get_df_parts_from_windows(d)
        assert df.shape == (6, 3)
        assert all(len(s) == 20 for s in df["tmd"])

    def test_single_part(self):
        df = SF.get_df_parts_from_windows({"tmd": ["AAAAAA", "CCCCCC"]})
        assert list(df) == ["tmd"] and len(df) == 2

    def test_values_preserved(self):
        df = SF.get_df_parts_from_windows({"jmd_n": ["AAAA", "CCCC"], "tmd": ["EEEEEE", "FFFFFF"],
                                           "jmd_c": ["HHHH", "KKKK"]})
        assert df.loc["REF1", "tmd"] == "FFFFFF"

    # Negative
    def test_none_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows(None)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({})

    def test_invalid_part_name_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"not_a_part": ["AAAA", "CCCC"]})

    def test_missing_window_column_raises(self):
        bad = pd.DataFrame({"sequence": ["AAAA", "CCCC"]})
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": bad})

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": [1, 2], "tmd": [3, 4]})

    def test_empty_windows_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": []})


class TestGetDfPartsFromWindowsComplex:
    """Cross-parameter / edge interactions for get_df_parts_from_windows."""

    def test_mismatch_warns_and_truncates(self):
        with pytest.warns(RuntimeWarning):
            df = SF.get_df_parts_from_windows({"jmd_n": ["AAAA", "CCCC", "DDDD"],
                                               "tmd": ["EEEEEE", "FFFFFF"]})
        assert len(df) == 2

    def test_mismatch_truncates_in_order(self):
        with pytest.warns(RuntimeWarning):
            df = SF.get_df_parts_from_windows({"jmd_n": ["AAAA", "CCCC", "DDDD"],
                                               "tmd": ["EEEEEE", "FFFFFF"]})
        assert df["jmd_n"].tolist() == ["AAAA", "CCCC"]

    def test_combo_part_names(self):
        df = SF.get_df_parts_from_windows({"jmd_n_tmd_n": ["AAAAAA", "CCCCCC"],
                                           "tmd_c_jmd_c": ["EEEEEE", "FFFFFF"]})
        assert list(df) == ["jmd_n_tmd_n", "tmd_c_jmd_c"]

    def test_dataframe_and_list_mixed(self):
        win_df = pd.DataFrame({ut.COL_WINDOW: ["AAAA", "CCCC"]})
        df = SF.get_df_parts_from_windows({"jmd_n": win_df, "tmd": ["EEEEEE", "FFFFFF"]})
        assert len(df) == 2 and df.loc["REF0", "jmd_n"] == "AAAA"

    def test_three_way_mismatch_min(self):
        with pytest.warns(RuntimeWarning):
            df = SF.get_df_parts_from_windows({"jmd_n": ["A", "C", "D"], "tmd": ["EE", "FF"],
                                               "jmd_c": ["H", "I", "K", "P"]})
        assert len(df) == 2

    # Negative combos
    def test_one_invalid_among_valid_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"tmd": ["AAAAAA", "CCCCCC"], "bogus": ["EE", "FF"]})

    def test_df_without_window_among_valid_raises(self):
        bad = pd.DataFrame({"foo": ["AAAA"]})
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": ["AAAA"], "tmd": bad})

    def test_non_string_in_one_part_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": ["AAAA", "CCCC"], "tmd": ["EEEEEE", 5]})

    def test_all_empty_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": [], "tmd": []})

    def test_scalar_value_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"tmd": "AAAA"})


class TestGetDfPartsFromWindowsGoldenValues:
    """Hand-computed expectations for get_df_parts_from_windows."""

    def test_exact_frame(self):
        df = SF.get_df_parts_from_windows({"jmd_n": ["AAAA", "CDEC"], "tmd": ["EEFEEY", "FFLFFF"],
                                           "jmd_c": ["HHHK", "IIRI"]})
        expected = pd.DataFrame({"jmd_n": ["AAAA", "CDEC"], "tmd": ["EEFEEY", "FFLFFF"],
                                 "jmd_c": ["HHHK", "IIRI"]}, index=["REF0", "REF1"])
        pd.testing.assert_frame_equal(df, expected)


# ======================================================================================
# Integration through the real CPP pipeline
# ======================================================================================
class TestIntegrationWithCPP:
    """Label helpers + reference assembly flow through CPP.run."""

    def test_ovr_vector_runs_and_equals_manual(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=9)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        mc = np.array([i % 3 for i in range(len(df_parts))])
        cpp = aa.CPP(df_parts=df_parts)
        for c, vec in SF.get_labels_ovr(mc).items():
            assert vec.tolist() == np.where(mc == c, 1, 0).tolist()
            assert len(cpp.run(labels=vec, n_filter=3)) == 3

    def test_quantile_labels_run(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        labels = SF.get_labels_quantile(np.linspace(0, 1, len(df_parts)), q=0.5)
        assert len(aa.CPP(df_parts=df_parts).run(labels=labels, n_filter=4)) == 4

    def test_per_part_prior_reference_runs(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        n_ref = len(df_parts)
        aws = aa.AAWindowSampler(random_state=0)
        df_ref = SF.get_df_parts_from_windows({
            "jmd_n": aws.sample_synthetic(df_seq=df_seq, n=n_ref, window_size=10, generator="coil"),
            "tmd": aws.sample_synthetic(df_seq=df_seq, n=n_ref, window_size=20, generator="alpha_helix"),
            "jmd_c": aws.sample_synthetic(df_seq=df_seq, n=n_ref, window_size=10, generator="coil"),
        })
        df_all = pd.concat([df_parts, df_ref])
        labels = [1] * len(df_parts) + [0] * len(df_ref)
        split_kws = sf.get_split_kws(n_split_max=5, steps_pattern=[3, 4], n_min=2, n_max=3, len_max=8)
        assert len(aa.CPP(df_parts=df_all, split_kws=split_kws).run(labels=labels, n_filter=5)) == 5


class TestImbalancedMultiClass:
    """Label helpers with unequal class sizes (Haiku #17)."""

    def test_ovr_imbalanced(self):
        d = SF.get_labels_ovr([0, 0, 0, 0, 0, 0, 1, 1, 2])
        assert d[0].sum() == 6 and d[1].sum() == 2 and d[2].sum() == 1

    def test_ovo_imbalanced_pair(self):
        o = SF.get_labels_ovo(np.array([0, 0, 0, 0, 0, 1, 2, 2, 2]), df_parts=_toy_df_parts(9))
        dps, _, binary = o[(0, 1)]
        assert len(dps) == 6 and binary.sum() == 5

    def test_quantile_outlier(self):
        labels = SF.get_labels_quantile(np.array([0, 0, 0, 0, 0, 0, 10.0]), q=0.9)
        assert labels.sum() == 1
