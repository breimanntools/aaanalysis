"""
This is a script for testing CPP bootstrap / stability feature selection
(the CPP(bootstrap=True) mode, a thin wrapper around run / run_num / run_composit).

Semantics: bootstrap does NOT change which features are selected — the selected set is exactly the
ordinary full-data run (n_filter is the selection criterion). It adds a ``selection_frequency``
column reporting how often each feature was selected across the resampling rounds.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

COL_FREQ = "selection_frequency"


def _fixture(n=5, n_scales=10):
    """Small DOM_GSEC fixture with enough candidate features for a selection."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
    df_scales = aa.load_scales().T.head(n_scales).T
    split_kws = aa.SequenceFeature().get_split_kws(
        split_types=["Segment"], n_split_min=1, n_split_max=3
    )
    return df_parts, labels, split_kws, df_scales


def _make_cpp(n_bootstrap=4, resample="reference", bootstrap_frac=0.8,
              random_state=42, bootstrap=True, **kwargs):
    df_parts, labels, split_kws, df_scales = _fixture()
    cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                 random_state=random_state, bootstrap=bootstrap, n_bootstrap=n_bootstrap,
                 resample=resample, bootstrap_frac=bootstrap_frac, **kwargs)
    return cpp, labels


class TestCPPBootstrap:
    """Positive and negative per-parameter tests for the bootstrap selection mode."""

    # ---- Positive: one parameter per test ------------------------------------------------
    @settings(max_examples=3, deadline=None)
    @given(n_bootstrap=some.integers(min_value=2, max_value=4))
    def test_n_bootstrap_positive(self, n_bootstrap):
        cpp, labels = _make_cpp(n_bootstrap=n_bootstrap)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ in df_feat.columns
        assert len(df_feat) >= 1

    @pytest.mark.parametrize("resample", ["both", "reference", "test"])
    def test_resample_positive(self, resample):
        cpp, labels = _make_cpp(n_bootstrap=3, resample=resample)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ in df_feat.columns

    @settings(max_examples=3, deadline=None)
    @given(bootstrap_frac=some.floats(min_value=0.5, max_value=1.0))
    def test_bootstrap_frac_positive(self, bootstrap_frac):
        cpp, labels = _make_cpp(n_bootstrap=3, bootstrap_frac=bootstrap_frac)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ in df_feat.columns

    def test_bootstrap_flag_toggles_mode(self):
        cpp, labels = _make_cpp(bootstrap=True)
        assert COL_FREQ in cpp.run(labels=labels, n_filter=15, n_jobs=1).columns

    def test_random_state_reproducible(self):
        cpp1, labels = _make_cpp(random_state=0)
        cpp2, _ = _make_cpp(random_state=0)
        df1 = cpp1.run(labels=labels, n_filter=15, n_jobs=1)
        df2 = cpp2.run(labels=labels, n_filter=15, n_jobs=1)
        assert df1.equals(df2)

    def test_selection_frequency_in_unit_interval(self):
        cpp, labels = _make_cpp(n_bootstrap=4)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert df_feat[COL_FREQ].between(0.0, 1.0).all()

    def test_sorted_by_abs_auc(self):
        cpp, labels = _make_cpp(n_bootstrap=4)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        auc = df_feat["abs_auc"].values
        assert (auc[:-1] >= auc[1:]).all()

    def test_default_off_has_no_frequency_column(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ not in df_feat.columns

    # ---- Negative: one parameter per test ------------------------------------------------
    @pytest.mark.parametrize("bootstrap", ["yes", 1, None, 1.5])
    def test_bootstrap_negative(self, bootstrap):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, bootstrap=bootstrap)

    @pytest.mark.parametrize("n_bootstrap", [0, -1, -5, 1.5, "5"])
    def test_n_bootstrap_negative(self, n_bootstrap):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                   bootstrap=True, n_bootstrap=n_bootstrap)

    @pytest.mark.parametrize("resample", ["ref", "neither", "positive", 1, None])
    def test_resample_negative(self, resample):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                   bootstrap=True, resample=resample)

    @pytest.mark.parametrize("bootstrap_frac", [0.0, -0.5, 1.5, 2.0])
    def test_bootstrap_frac_negative(self, bootstrap_frac):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                   bootstrap=True, bootstrap_frac=bootstrap_frac)


class TestCPPBootstrapComplex:
    """Combination and edge-interaction tests."""

    # ---- Positive combinations -----------------------------------------------------------
    def test_byte_identical_when_off(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp_default = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                             verbose=False, random_state=42)
        cpp_off = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                         verbose=False, random_state=42, bootstrap=False)
        df_default = cpp_default.run(labels=labels, n_filter=15, n_jobs=1)
        df_off = cpp_off.run(labels=labels, n_filter=15, n_jobs=1)
        assert df_default.equals(df_off)

    def test_selection_equals_normal_run(self):
        # Annotate semantics: the selected feature set is exactly the ordinary full-data run;
        # only the selection_frequency column is added.
        df_parts, labels, split_kws, df_scales = _fixture()
        normal = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                        random_state=42).run(labels=labels, n_filter=15, n_jobs=1)
        boot = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                      random_state=42, bootstrap=True, n_bootstrap=4).run(labels=labels, n_filter=15, n_jobs=1)
        assert list(boot["feature"]) == list(normal["feature"])
        # Non-frequency columns are unchanged from the normal run.
        assert boot.drop(columns=[COL_FREQ]).equals(normal)

    def test_config_ignored_when_off(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        base = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                      verbose=False, random_state=42).run(labels=labels, n_filter=15, n_jobs=1)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=42, bootstrap=False, n_bootstrap=17, resample="test",
                     bootstrap_frac=0.5)
        assert base.equals(cpp.run(labels=labels, n_filter=15, n_jobs=1))

    def test_run_num_bootstrap(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        labels = df_seq["label"].to_list()
        df_scales = aa.load_scales().T.head(8).T
        nf = aa.NumericalFeature()
        scale_map = {a: df_scales.loc[a].values for a in df_scales.index}
        d = df_scales.shape[1]
        dict_num = {
            entry: np.array([scale_map.get(c, np.full(d, np.nan)) for c in seq], dtype=float)
            for entry, seq in zip(df_seq["entry"], df_seq["sequence"])
        }
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num,
                                                jmd_n_len=10, jmd_c_len=10)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=1,
                     bootstrap=True, n_bootstrap=3, resample="reference")
        df_feat = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_filter=12, n_jobs=1)
        assert COL_FREQ in df_feat.columns
        assert df_feat[COL_FREQ].between(0.0, 1.0).all()

    @pytest.mark.parametrize("composition", ["aac", "dpc"])
    def test_run_composit_bootstrap(self, composition):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, n_bootstrap=3)
        df_feat = cpp.run_composit(labels=labels, composition=composition, n_filter=10, n_jobs=1)
        assert COL_FREQ in df_feat.columns
        assert df_feat[COL_FREQ].between(0.0, 1.0).all()

    def test_bootstrap_with_parametric(self):
        cpp, labels = _make_cpp(n_bootstrap=3)
        df_feat = cpp.run(labels=labels, n_filter=15, parametric=True, n_jobs=1)
        assert "p_val_ttest_indep" in df_feat.columns
        assert COL_FREQ in df_feat.columns

    def test_verbose_bootstrap_runs(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=True,
                     random_state=0, bootstrap=True, n_bootstrap=3, resample="both")
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ in df_feat.columns

    def test_accept_gaps_bootstrap(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, n_bootstrap=3, accept_gaps=True)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert COL_FREQ in df_feat.columns
        assert df_feat[COL_FREQ].between(0.0, 1.0).all()

    # ---- Negative combinations -----------------------------------------------------------
    def test_bootstrap_incompatible_with_n_batches(self):
        cpp, labels = _make_cpp(n_bootstrap=3)
        with pytest.raises(ValueError, match="bootstrap"):
            cpp.run(labels=labels, n_filter=15, n_batches=3, n_jobs=1)

    def test_bootstrap_incompatible_with_n_sample_batches(self):
        cpp, labels = _make_cpp(n_bootstrap=3)
        with pytest.raises(ValueError, match="bootstrap"):
            cpp.run(labels=labels, n_filter=15, n_sample_batches=2, n_jobs=1)

    def test_run_num_bootstrap_incompatible_with_n_batches(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        labels = df_seq["label"].to_list()
        df_scales = aa.load_scales().T.head(8).T
        nf = aa.NumericalFeature()
        scale_map = {a: df_scales.loc[a].values for a in df_scales.index}
        d = df_scales.shape[1]
        dict_num = {
            entry: np.array([scale_map.get(c, np.full(d, np.nan)) for c in seq], dtype=float)
            for entry, seq in zip(df_seq["entry"], df_seq["sequence"])
        }
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num,
                                                jmd_n_len=10, jmd_c_len=10)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=1,
                     bootstrap=True, n_bootstrap=3)
        with pytest.raises(ValueError, match="bootstrap"):
            cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_filter=12, n_batches=2)


class TestCPPBootstrapGoldenValues:
    """Hand-verifiable invariants of the selection-frequency output."""

    def test_frequency_values_are_multiples_of_inverse_n_bootstrap(self):
        n_bootstrap = 4
        cpp, labels = _make_cpp(n_bootstrap=n_bootstrap)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        counts = np.rint(df_feat[COL_FREQ].values * n_bootstrap)
        assert np.allclose(df_feat[COL_FREQ].values, counts / n_bootstrap, atol=1e-9)
        assert (counts >= 0).all() and (counts <= n_bootstrap).all()

    def test_selection_frequency_is_last_column(self):
        cpp, labels = _make_cpp(n_bootstrap=3)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert list(df_feat.columns)[-1] == COL_FREQ
        assert list(df_feat.columns).index("positions") < list(df_feat.columns).index(COL_FREQ)

    def test_output_never_exceeds_n_filter(self):
        cpp, labels = _make_cpp(n_bootstrap=3)
        df_feat = cpp.run(labels=labels, n_filter=10, n_jobs=1)
        assert len(df_feat) <= 10

    def test_reproducible_frequencies_exact(self):
        cpp1, labels = _make_cpp(random_state=123, n_bootstrap=4)
        cpp2, _ = _make_cpp(random_state=123, n_bootstrap=4)
        df1 = cpp1.run(labels=labels, n_filter=15, n_jobs=1)
        df2 = cpp2.run(labels=labels, n_filter=15, n_jobs=1)
        assert df1[COL_FREQ].tolist() == df2[COL_FREQ].tolist()
