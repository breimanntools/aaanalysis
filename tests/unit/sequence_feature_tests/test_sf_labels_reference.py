"""This is a script to test the SequenceFeature label-conversion and reference-generation methods.

Covers get_labels_ovr, get_labels_ovo, get_labels_quantile, and get_df_parts_reference
(issue #61: multi-class & regression CPP via label conversion + reference generation).
"""
import numpy as np
import pandas as pd
from collections import Counter
import pytest
import aaanalysis as aa

SF = aa.SequenceFeature


def _toy_df_parts():
    return pd.DataFrame(
        {
            "jmd_n": ["AAAA", "CDEC", "DDWD"],
            "tmd": ["EEFEEY", "FFLFFF", "GMGGGG"],
            "jmd_c": ["HHHK", "IIRI", "KKQK"],
        }
    )


# Normal Cases
class TestGetLabelsOvr:
    """Test get_labels_ovr."""

    def test_keys_are_classes(self):
        d = SF.get_labels_ovr([0, 0, 1, 1, 2, 2])
        assert list(d.keys()) == [0, 1, 2]

    def test_one_vs_rest_membership(self):
        d = SF.get_labels_ovr([0, 0, 1, 1, 2, 2])
        assert d[1].tolist() == [0, 0, 1, 1, 0, 0]
        assert d[2].tolist() == [0, 0, 0, 0, 1, 1]

    def test_full_length_no_drop(self):
        labels = [0, 1, 2, 0, 1]
        d = SF.get_labels_ovr(labels)
        assert all(len(v) == len(labels) for v in d.values())

    def test_binary_only(self):
        d = SF.get_labels_ovr([0, 1, 2])
        for v in d.values():
            assert set(np.unique(v)).issubset({0, 1})

    def test_custom_test_ref(self):
        d = SF.get_labels_ovr([0, 1, 2], label_test=5, label_ref=-1)
        assert set(np.unique(d[0])) == {5, -1}


class TestGetLabelsOvo:
    """Test get_labels_ovo."""

    def test_pairs(self):
        o = SF.get_labels_ovo([0, 0, 1, 1, 2, 2])
        assert list(o.keys()) == [(0, 1), (0, 2), (1, 2)]

    def test_mask_selects_pair_only(self):
        o = SF.get_labels_ovo([0, 0, 1, 1, 2, 2])
        mask, binary = o[(0, 1)]
        assert mask.tolist() == [True, True, True, True, False, False]
        assert binary.tolist() == [1, 1, 0, 0]

    def test_binary_len_matches_mask(self):
        o = SF.get_labels_ovo([0, 1, 2, 0, 1, 2])
        for mask, binary in o.values():
            assert len(binary) == int(mask.sum())
            assert set(np.unique(binary)).issubset({0, 1})


class TestGetLabelsQuantile:
    """Test get_labels_quantile."""

    def test_median_split(self):
        labels = SF.get_labels_quantile([1.0, 2, 3, 4, 5, 6], q=0.5)
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]

    def test_length_preserved(self):
        t = [0.1, 0.2, 0.3, 0.9]
        assert len(SF.get_labels_quantile(t, q=0.25)) == len(t)

    def test_quantile_threshold(self):
        labels = SF.get_labels_quantile([10, 20, 30, 40], q=0.75)
        # cut = 75th percentile = 32.5 -> only 40 >= cut
        assert labels.tolist() == [0, 0, 0, 1]

    def test_custom_test_ref(self):
        labels = SF.get_labels_quantile([1.0, 2, 3, 4], q=0.5, label_test=9, label_ref=2)
        assert set(np.unique(labels)).issubset({9, 2})


class TestGetDfPartsReference:
    """Test get_df_parts_reference."""

    def test_same_columns(self):
        dfp = _toy_df_parts()
        ref = SF.get_df_parts_reference(dfp, method="scrambled", random_state=0)
        assert list(ref) == list(dfp)

    def test_per_part_length_matches_a_real_row(self):
        dfp = _toy_df_parts()
        ref = SF.get_df_parts_reference(dfp, method="global_freq", n=5, random_state=0)
        for part in dfp:
            real_lengths = {len(s) for s in dfp[part]}
            assert all(len(s) in real_lengths for s in ref[part])

    def test_scrambled_preserves_composition(self):
        dfp = _toy_df_parts()
        ref = SF.get_df_parts_reference(dfp, method="scrambled", n=10, random_state=3)
        # Each scrambled part is an anagram of SOME real part in that column.
        for part in dfp:
            real_comps = {frozenset(Counter(s).items()) for s in dfp[part]}
            for s in ref[part]:
                assert frozenset(Counter(s).items()) in real_comps

    def test_n_rows(self):
        dfp = _toy_df_parts()
        assert len(SF.get_df_parts_reference(dfp, n=7, random_state=0)) == 7
        assert len(SF.get_df_parts_reference(dfp, random_state=0)) == len(dfp)

    def test_seed_determinism(self):
        dfp = _toy_df_parts()
        a = SF.get_df_parts_reference(dfp, method="global_freq", n=6, random_state=42)
        b = SF.get_df_parts_reference(dfp, method="global_freq", n=6, random_state=42)
        assert a.equals(b)

    def test_global_freq_only_canonical(self):
        dfp = _toy_df_parts()
        ref = SF.get_df_parts_reference(dfp, method="global_freq", n=8, random_state=1)
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        for part in dfp:
            assert set("".join(ref[part])).issubset(canonical)


class TestGetDfPartsFromWindows:
    """Test get_df_parts_from_windows (assemble df_parts from per-part window sets)."""

    def test_from_lists(self):
        d = {
            "jmd_n": ["AAAA", "CCCC"],
            "tmd": ["EEEEEE", "FFFFFF"],
            "jmd_c": ["HHHH", "KKKK"],
        }
        df_parts = SF.get_df_parts_from_windows(d)
        assert list(df_parts) == ["jmd_n", "tmd", "jmd_c"]
        assert df_parts.index.tolist() == ["REF0", "REF1"]
        assert df_parts.loc["REF1", "tmd"] == "FFFFFF"

    def test_from_aawindowsampler_output(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        aws = aa.AAWindowSampler(random_state=0)
        d = {
            "jmd_n": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=10, generator="coil"),
            "tmd": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=20, generator="alpha_helix"),
            "jmd_c": aws.sample_synthetic(df_seq=df_seq, n=6, window_size=10, generator="coil"),
        }
        df_parts = SF.get_df_parts_from_windows(d)
        assert df_parts.shape == (6, 3)
        assert all(len(s) == 10 for s in df_parts["jmd_n"])
        assert all(len(s) == 20 for s in df_parts["tmd"])

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({"jmd_n": ["AAAA"], "tmd": ["EEE", "FFF"]})

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

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_from_windows({})


# Integration
class TestIntegrationWithCPP:
    """Label helpers + reference generator flow through the real CPP pipeline."""

    def test_reference_as_negative_class_runs(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_ref = SF.get_df_parts_reference(df_parts, method="scrambled", random_state=0)
        df_all = pd.concat([df_parts, df_ref])
        labels = [1] * len(df_parts) + [0] * len(df_ref)
        df_feat = aa.CPP(df_parts=df_all).run(labels=labels, n_filter=5)
        assert ut_cols_present(df_feat)
        assert len(df_feat) == 5

    def test_ovr_vector_equals_manual_binary_run(self):
        # Build a 3-class label vector by relabeling DOM_GSEC into thirds.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=9)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        n = len(df_parts)
        mc = np.array([i % 3 for i in range(n)])
        d = SF.get_labels_ovr(mc)
        cpp = aa.CPP(df_parts=df_parts)
        for c, vec in d.items():
            manual = np.where(mc == c, 1, 0)
            assert vec.tolist() == manual.tolist()
            df_feat = cpp.run(labels=vec, n_filter=3)
            assert len(df_feat) == 3

    def test_quantile_labels_run(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        sf = SF()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        targets = np.linspace(0, 1, len(df_parts))
        labels = SF.get_labels_quantile(targets, q=0.5)
        df_feat = aa.CPP(df_parts=df_parts).run(labels=labels, n_filter=4)
        assert len(df_feat) == 4

    def test_per_part_prior_reference_runs(self):
        # jmd_n=coil, tmd=alpha_helix, jmd_c=coil reference, assembled then run via CPP.
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
        df_feat = aa.CPP(df_parts=df_all, split_kws=split_kws).run(labels=labels, n_filter=5)
        assert len(df_feat) == 5


def ut_cols_present(df_feat):
    import aaanalysis.utils as ut
    return set(ut.LIST_COLS_FEAT).issubset(set(df_feat.columns))


# Error Cases
class TestErrors:
    """Invalid inputs raise informative errors."""

    def test_ovr_single_class_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([1, 1, 1])

    def test_ovr_equal_test_ref_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_ovr([0, 1], label_test=1, label_ref=1)

    def test_quantile_bad_q_raises(self):
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=0)
        with pytest.raises(ValueError):
            SF.get_labels_quantile([1.0, 2, 3], q=1)

    def test_reference_bad_method_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_reference(_toy_df_parts(), method="nonsense")

    def test_reference_bad_seed_raises(self):
        with pytest.raises(ValueError):
            SF.get_df_parts_reference(_toy_df_parts(), random_state=-1)
