"""
This is a script for testing the CPP().run() method.
"""
import pytest
import pandas as pd
import aaanalysis as aa
import random

aa.options["verbose"] = False


def get_parts_splits_scales():
    """"""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=2)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
    df_scales = aa.load_scales().T.head(2).T
    split_kws = aa.SequenceFeature().get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
    return df_parts, labels, split_kws, df_scales 


class TestCPPRun:
    """
    Test class for positive simple test cases of the CPP class run() method.
    """

    # Positive tests
    def test_defaults(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_all_parst(self):
        _, _, split_kws, df_scales = get_parts_splits_scales()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=3)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, all_parts=True)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels)
        assert isinstance(df_feat, pd.DataFrame)


    def test_valid_labels(self):
        all_data_set_names = [x for x in aa.load_dataset()["Dataset"].to_list() if "AA" not in x
                              and "AMYLO" not in x and "PU" not in x]
        sampled_names = random.sample(all_data_set_names, 3)
        df_scales = aa.load_scales().T.head(2).T
        for name in sampled_names:
            df_seq = aa.load_dataset(name=name, n=10, min_len=50)
            labels = df_seq["label"].to_list()
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"])
            split_kws = sf.get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
            cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
            df_feat = cpp.run(labels=labels, n_jobs=1)
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_label_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        labels = [10 if l == 1 else l for l in labels]
        df_feat = cpp.run(labels=labels, label_test=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_label_ref(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        split_kws = aa.SequenceFeature().get_split_kws(split_types=["Segment"], n_split_min=1, n_split_max=2)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        labels = [10 if l == 0 else l for l in labels]
        df_feat = cpp.run(labels=labels, label_ref=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
    
    def test_valid_n_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_filter=50)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_pre_filter=200, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_pct_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, pct_pre_filter=10, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_std_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_std_test=0.99, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_std_test=0.1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_overlap(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_overlap=1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_overlap=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_max_cor(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, max_cor=1, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = cpp.run(labels=labels, max_cor=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_check_cat(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, check_cat=False, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_parametric(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, parametric=True, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_start(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, start=-4, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_tmd_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, tmd_len=25, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_jmd_n_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, jmd_n_len=5, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_jmd_c_len(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, jmd_c_len=2, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_jobs(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_feat = cpp.run(labels=labels, n_jobs=4)
        assert isinstance(df_feat, pd.DataFrame)

    def test_valid_vectorized(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        for vectorized in [True, False]:
            df_feat = cpp.run(labels=labels, n_jobs=1, vectorized=vectorized)
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_batches(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        list_n_batches = [random.randint(2, len(list(df_scales))) for _ in range(3)]
        for n_batches in list_n_batches:
            df_feat = cpp.run(labels=labels, n_jobs=1, n_batches=n_batches)
            assert isinstance(df_feat, pd.DataFrame)

    def test_valid_n_sample_batches(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        for n_sample_batches in [2, len(df_parts)]:
            df_feat = cpp.run(labels=labels, n_jobs=1, n_sample_batches=n_sample_batches)
            assert isinstance(df_feat, pd.DataFrame)
            assert len(df_feat) > 0

    # Negative tests
    def test_invalid_n_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_filter=-1)

    def test_invalid_n_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_pre_filter=-10)

    def test_invalid_pct_pre_filter(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, pct_pre_filter=-5)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, pct_pre_filter=None)

    def test_invalid_max_std_test(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_std_test=-0.5)

    def test_invalid_max_overlap(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_overlap=-0.1)

    def test_invalid_max_cor(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, max_cor=-0.1)

    def test_empty_df_parts(self):
        df_scales = aa.load_scales().T.head(2).T
        with pytest.raises(ValueError):
            cpp = aa.CPP(df_parts=pd.DataFrame(), df_scales=df_scales)
            cpp.run(labels=[0, 1])

    def test_mismatched_labels(self):
        df_parts, _, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=[1])  # Assuming df_parts has more than 1 row

    def test_invalid_n_jobs(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_jobs=0)

    def test_invalid_vectorized(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        for vectorized in ["asdf", None, 1, [True, True]]:
            with pytest.raises(ValueError):
                cpp.run(labels=labels, n_jobs=1, vectorized=vectorized)

    def test_invalid_n_batches(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        list_n_batches = [1, "non", True, len(list(df_scales)) + 1, [None, None]]
        for n_batches in list_n_batches:
            with pytest.raises(ValueError):
                cpp.run(labels=labels, n_jobs=1, n_batches=n_batches)

    def test_invalid_n_sample_batches(self):
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        list_invalid = [1, "non", True, len(df_parts) + 1, 2.5]
        for n_sample_batches in list_invalid:
            with pytest.raises(ValueError):
                cpp.run(labels=labels, n_jobs=1, n_sample_batches=n_sample_batches)


class TestCPPRunComplex:
    """Edge-case and multi-parameter interactions for the CPP.run() method."""

    def test_n_sample_batches_matches_single_pass(self):
        """Sample-batched run returns the same feature set as the single-pass run."""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        df_single = cpp.run(labels=labels, n_jobs=1, n_filter=20)
        df_sb = cpp.run(labels=labels, n_jobs=1, n_filter=20, n_sample_batches=2)
        assert list(df_single.columns) == list(df_sb.columns)
        assert set(df_single["feature"]) == set(df_sb["feature"])

    def test_n_batches_and_n_sample_batches_mutually_exclusive(self):
        """Setting both scale- and sample-batching at once raises ValueError."""
        df_parts, labels, split_kws, df_scales = get_parts_splits_scales()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, n_jobs=1, n_batches=2, n_sample_batches=2)

    def test_empty_pattern_bucket_silently_dropped(self):
        # Regression: a Pattern config whose every repeated-step cumsum exceeds
        # len_max (steps=[3] -> cumsum([3, 3])=6 > len_max=4) yields zero Pattern
        # splits for ALL parts. This previously crashed with a ZeroDivisionError
        # in the vectorized pre-filter (per_scale_bytes == 0). It must now run,
        # warn at construction, and drop Pattern features silently while keeping
        # Segment / PeriodicPattern features. Config taken verbatim from a real
        # downstream grid sweep.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=6)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
        df_scales = aa.load_scales().T.head(4).T
        split_kws = {
            "Segment": {"n_split_min": 1, "n_split_max": 4},
            "Pattern": {"steps": [3], "n_min": 2, "n_max": 3, "len_max": 4},
            "PeriodicPattern": {"steps": [3, 4]},
        }
        with pytest.warns(UserWarning, match="Pattern"):
            cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws)
        for vectorized in [True, False]:
            df_feat = cpp.run(labels=labels, n_jobs=1, vectorized=vectorized)
            assert isinstance(df_feat, pd.DataFrame)
            assert len(df_feat) > 0
            # Pattern features were silently dropped; "-Pattern(" excludes the
            # surviving "-PeriodicPattern(" features (no leading dash inside it).
            assert not df_feat["feature"].str.contains("-Pattern(", regex=False).any()


class TestCPPRunProperties:
    """Output invariants of CPP.run (ranking / filter / range), not just 'is a DataFrame'."""

    @staticmethod
    def _setup(n=20, n_scales=8):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(n_scales).T
        return labels, df_parts, df_scales

    def test_filter_count_invariant(self):
        labels, df_parts, df_scales = self._setup()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
        for n_filter in (5, 10, 25):
            df_feat = cpp.run(labels=labels, n_filter=n_filter, n_jobs=1)
            assert len(df_feat) == n_filter

    def test_returned_features_unique(self):
        labels, df_parts, df_scales = self._setup()
        df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                         verbose=False).run(labels=labels, n_filter=20, n_jobs=1)
        assert df_feat["feature"].is_unique

    def test_abs_auc_in_unit_half_range(self):
        # abs_auc = |AUC - 0.5| so it must lie in [0, 0.5].
        labels, df_parts, df_scales = self._setup()
        df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                         verbose=False).run(labels=labels, n_filter=15, n_jobs=1)
        assert df_feat["abs_auc"].min() >= 0.0
        assert df_feat["abs_auc"].max() <= 0.5

    def test_deterministic_same_input(self):
        # Same input -> identical selected features in identical order.
        labels, df_parts, df_scales = self._setup()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
        a = cpp.run(labels=labels, n_filter=15, n_jobs=1)["feature"].tolist()
        b = cpp.run(labels=labels, n_filter=15, n_jobs=1)["feature"].tolist()
        assert a == b

    def test_n_jobs_does_not_change_selection(self):
        labels, df_parts, df_scales = self._setup()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
        s1 = set(cpp.run(labels=labels, n_filter=15, n_jobs=1)["feature"])
        s2 = set(cpp.run(labels=labels, n_filter=15, n_jobs=2)["feature"])
        assert s1 == s2


def _run_num_args(n=6, d=4):
    """Build (df_parts, dict_num_parts, df_scales, df_cat, labels) for run_num."""
    import numpy as np
    import aaanalysis.utils as ut
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3, "MKTAYIAKQRQISFVKSHFSRQ" * 3,
            "GAVLIMFWPSTCYNQDEKRHG" * 3, "LLLLLKKKKKDDDDDEEEEEAA" * 3,
            "WYWYWYWYWYWYWYWYWYWY" * 3, "QQQQQNNNNNSSSSSTTTTTT" * 3][:n]
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs})
    df_seq["tmd_start"] = 11
    df_seq["tmd_stop"] = df_seq["sequence"].str.len() - 10
    rng = np.random.default_rng(0)
    dict_num = {e: rng.random((len(s), d)) for e, s in zip(df_seq["entry"], df_seq["sequence"])}
    nf = aa.NumericalFeature()
    df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num, jmd_n_len=10, jmd_c_len=10)
    df_scales = pd.DataFrame(rng.random((20, d)), index=list(ut.LIST_CANONICAL_AA),
                             columns=[f"d{i}" for i in range(d)])
    df_cat = pd.DataFrame({"scale_id": [f"d{i}" for i in range(d)], "category": ["X"] * d,
                           "subcategory": ["x"] * d, "scale_name": [f"d{i}" for i in range(d)],
                           "scale_description": ["d"] * d})
    labels = [1] * (n // 2) + [0] * (n - n // 2)
    return df_parts, dict_num_parts, df_scales, df_cat, labels


class TestCPPRunBatchBranches:
    """Branch coverage for the scale-batched / sample-batched / numerical-batched
    orchestrators behind CPP.run / CPP.run_num — exercises the per-batch verbose
    progress arms (which only execute under verbose=True) and the batched output
    invariants. These orchestrators are otherwise only reached via the n_batches /
    n_sample_batches parameters."""

    def test_scale_batched_verbose_progress(self):
        # Drives cpp_run_batch's per-batch verbose progress branches.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(6).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=True)
        df_feat = cpp.run(labels=labels, n_filter=10, n_jobs=1, n_batches=3)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) == 10

    def test_scale_batched_matches_single_pass(self):
        # Batched and single-pass select the same feature set (output invariant).
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(6).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
        df_single = cpp.run(labels=labels, n_filter=10, n_jobs=1)
        df_batch = cpp.run(labels=labels, n_filter=10, n_jobs=1, n_batches=3)
        assert set(df_single["feature"]) == set(df_batch["feature"])

    def test_sample_batched_verbose_progress(self):
        # Drives cpp_run_sample_batched's pass-1 / pass-2 verbose progress branches.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(6).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=True)
        df_feat = cpp.run(labels=labels, n_filter=10, n_jobs=1, n_sample_batches=2)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) == 10

    def test_sample_batched_max_batches_equals_n_samples(self):
        # n_sample_batches == n_samples -> batch_size 1, every batch non-empty.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=6)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(5).T
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=8, n_jobs=1, n_sample_batches=len(df_parts))
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) > 0

    def test_run_num_scale_batched_verbose_progress(self):
        # Drives cpp_run_batch_num's per-D-batch verbose progress branches.
        dp, dnp, dsc, dca, labels = _run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca, verbose=True)
        df_feat = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1, n_batches=2)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) == 3

    def test_run_num_scale_batched_matches_single_pass(self):
        # Numerical batched path is bit-exact with the single-pass numerical path.
        dp, dnp, dsc, dca, labels = _run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca, verbose=False)
        df_single = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1)
        df_batch = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1, n_batches=2)
        assert set(df_single["feature"]) == set(df_batch["feature"])

    def test_scale_batched_verbose_multi_split_types(self):
        # Verbose batched run across all three split types exercises the per-batch
        # progress arms with a richer split_kws.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=6)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
        df_scales = aa.load_scales(top60_n=20).T.head(4).T
        split_kws = aa.SequenceFeature().get_split_kws(
            split_types=["Segment", "PeriodicPattern"], n_split_min=1, n_split_max=3)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=True)
        df_feat = cpp.run(labels=labels, n_filter=8, n_jobs=1, n_batches=2)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) > 0
