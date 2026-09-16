"""
This is a script for testing the opt-in per-feature bootstrap confidence intervals
(the CPP(bootstrap=True, bootstrap_kws={'ci': <level>}) mode).

Semantics: the bootstrap wrapper already re-runs the ordinary selection on each resample; with a
confidence level set, the statistics of those rounds are retained and summarized into a per-feature
percentile interval, adding 'abs_auc_ci_low' / 'abs_auc_ci_high' and 'mean_dif_ci_low' /
'mean_dif_ci_high' after 'selection_frequency'. The interval covers the rounds in which the feature
was selected; a feature observed in fewer than two rounds gets NaN bounds. With 'ci' unset (the
default) the output is unchanged.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from aaanalysis.feature_engineering._cpp import _add_bootstrap_ci_cols, _collect_round_stats

aa.options["verbose"] = False

settings.register_profile("ci_profile", deadline=None)
settings.load_profile("ci_profile")

COL_FREQ = "selection_frequency"
COL_AUC_LOW, COL_AUC_HIGH = "abs_auc_ci_low", "abs_auc_ci_high"
COL_DIF_LOW, COL_DIF_HIGH = "mean_dif_ci_low", "mean_dif_ci_high"
COLS_CI = [COL_AUC_LOW, COL_AUC_HIGH, COL_DIF_LOW, COL_DIF_HIGH]


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


def _make_cpp(ci=0.95, rounds=3, resample="reference", frac=0.8, random_state=42, **kwargs):
    df_parts, labels, split_kws, df_scales = _fixture()
    cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                 random_state=random_state, bootstrap=True,
                 bootstrap_kws=dict(rounds=rounds, resample=resample, frac=frac, ci=ci), **kwargs)
    return cpp, labels


class TestCPPBootstrapCI:
    """Positive and negative tests for the bootstrap_kws['ci'] confidence level."""

    # ---- Positive: one setting per test --------------------------------------------------
    @settings(max_examples=3, deadline=None)
    @given(ci=some.floats(min_value=0.5, max_value=0.99))
    def test_ci_positive(self, ci):
        cpp, labels = _make_cpp(ci=ci)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    def test_ci_none_adds_no_columns(self):
        cpp, labels = _make_cpp(ci=None)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert not any(col in df_feat.columns for col in COLS_CI)
        assert COL_FREQ in df_feat.columns

    def test_ci_omitted_key_defaults_to_off(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, bootstrap_kws=dict(rounds=3))
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert not any(col in df_feat.columns for col in COLS_CI)

    def test_ci_only_key_merges_defaults(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, bootstrap_kws=dict(ci=0.9))
        assert cpp._n_bootstrap == 20 and cpp._bootstrap_frac == 0.8

    @pytest.mark.parametrize("rounds", [2, 3, 4])
    def test_ci_with_rounds_positive(self, rounds):
        cpp, labels = _make_cpp(ci=0.9, rounds=rounds)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    @pytest.mark.parametrize("resample", ["both", "reference", "test"])
    def test_ci_with_resample_positive(self, resample):
        cpp, labels = _make_cpp(ci=0.9, resample=resample)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    @settings(max_examples=3, deadline=None)
    @given(frac=some.floats(min_value=0.5, max_value=1.0))
    def test_ci_with_frac_positive(self, frac):
        cpp, labels = _make_cpp(ci=0.9, frac=frac)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    def test_low_not_above_high(self):
        cpp, labels = _make_cpp(ci=0.95, rounds=4)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        for low, high in [(COL_AUC_LOW, COL_AUC_HIGH), (COL_DIF_LOW, COL_DIF_HIGH)]:
            observed = df_feat.dropna(subset=[low, high])
            assert (observed[low] <= observed[high]).all()

    def test_abs_auc_bounds_in_valid_range(self):
        cpp, labels = _make_cpp(ci=0.95, rounds=4)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        observed = df_feat.dropna(subset=[COL_AUC_LOW, COL_AUC_HIGH])
        assert observed[COL_AUC_LOW].between(-0.5, 0.5).all()
        assert observed[COL_AUC_HIGH].between(-0.5, 0.5).all()

    def test_mean_dif_bounds_in_valid_range(self):
        cpp, labels = _make_cpp(ci=0.95, rounds=4)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        observed = df_feat.dropna(subset=[COL_DIF_LOW, COL_DIF_HIGH])
        assert observed[COL_DIF_LOW].between(-1.0, 1.0).all()
        assert observed[COL_DIF_HIGH].between(-1.0, 1.0).all()

    def test_ci_columns_follow_selection_frequency(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=3)
        cols = list(cpp.run(labels=labels, n_filter=15, n_jobs=1).columns)
        assert cols[-4:] == COLS_CI
        assert cols.index(COL_FREQ) < cols.index(COL_AUC_LOW)

    def test_ci_reproducible_with_random_state(self):
        cpp1, labels = _make_cpp(ci=0.9, random_state=7)
        cpp2, _ = _make_cpp(ci=0.9, random_state=7)
        df1 = cpp1.run(labels=labels, n_filter=15, n_jobs=1)
        df2 = cpp2.run(labels=labels, n_filter=15, n_jobs=1)
        assert df1.equals(df2)

    def test_ci_columns_are_float(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=3)
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(df_feat[col].dtype == float for col in COLS_CI)

    def test_ci_stored_on_instance(self):
        cpp, _ = _make_cpp(ci=0.8)
        assert cpp._bootstrap_ci == 0.8

    # ---- Negative -----------------------------------------------------------------------
    @pytest.mark.parametrize("ci", ["0.95", "", 0.0, 1.0, -0.5, 1.5, 2, [0.9], (0.9,),
                                    {"low": 0.1}, True, np.array([0.9])])
    def test_ci_negative(self, ci):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(rounds=2, ci=ci))

    def test_ci_error_message_names_the_key(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError, match=r"bootstrap_kws\['ci'\]"):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(ci=1.2))

    def test_ci_error_message_names_the_bounds(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError, match=r"0 < n < 1"):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(ci=0.0))

    def test_ci_with_unknown_key_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError, match="unknown key"):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(ci=0.95, confidence=0.95))

    def test_ci_in_non_dict_kws_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws="ci=0.95")


class TestCPPBootstrapCIComplex:
    """Combination and edge-interaction tests."""

    # ---- Positive ------------------------------------------------------------------------
    def test_ci_off_is_byte_identical(self):
        # The default (no 'ci' key) and an explicit ci=None must give the identical frame.
        df_parts, labels, split_kws, df_scales = _fixture()
        kws = dict(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   random_state=42, bootstrap=True)
        df_no_key = aa.CPP(**kws, bootstrap_kws=dict(rounds=3)).run(
            labels=labels, n_filter=15, n_jobs=1)
        df_none = aa.CPP(**kws, bootstrap_kws=dict(rounds=3, ci=None)).run(
            labels=labels, n_filter=15, n_jobs=1)
        assert df_no_key.equals(df_none)

    def test_ci_does_not_change_the_selection(self):
        # The interval columns are additive: dropping them reproduces the plain bootstrap run.
        df_parts, labels, split_kws, df_scales = _fixture()
        kws = dict(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   random_state=42, bootstrap=True)
        plain = aa.CPP(**kws, bootstrap_kws=dict(rounds=3)).run(
            labels=labels, n_filter=15, n_jobs=1)
        with_ci = aa.CPP(**kws, bootstrap_kws=dict(rounds=3, ci=0.95)).run(
            labels=labels, n_filter=15, n_jobs=1)
        assert with_ci.drop(columns=COLS_CI).equals(plain)

    def test_ci_ignored_when_bootstrap_off(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        base = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                      random_state=42).run(labels=labels, n_filter=15, n_jobs=1)
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=42, bootstrap=False, bootstrap_kws=dict(rounds=3, ci=0.95))
        assert base.equals(cpp.run(labels=labels, n_filter=15, n_jobs=1))

    def test_wider_level_gives_wider_interval(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        kws = dict(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   random_state=3, bootstrap=True)
        narrow = aa.CPP(**kws, bootstrap_kws=dict(rounds=5, ci=0.5)).run(
            labels=labels, n_filter=15, n_jobs=1)
        wide = aa.CPP(**kws, bootstrap_kws=dict(rounds=5, ci=0.99)).run(
            labels=labels, n_filter=15, n_jobs=1)
        w_narrow = (narrow[COL_AUC_HIGH] - narrow[COL_AUC_LOW]).dropna()
        w_wide = (wide[COL_AUC_HIGH] - wide[COL_AUC_LOW]).dropna()
        assert (w_wide.values >= w_narrow.values - 1e-9).all()

    def test_run_num_ci(self):
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
                     bootstrap=True, bootstrap_kws=dict(rounds=3, ci=0.9))
        df_feat = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_filter=12, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    @pytest.mark.parametrize("composition", ["aac", "dpc"])
    def test_run_composit_ci(self, composition):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, bootstrap_kws=dict(rounds=3, ci=0.9))
        df_feat = cpp.run_composit(labels=labels, composition=composition, n_filter=10, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    def test_ci_with_parametric(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=3)
        df_feat = cpp.run(labels=labels, n_filter=15, parametric=True, n_jobs=1)
        assert "p_val_ttest_indep" in df_feat.columns
        assert all(col in df_feat.columns for col in COLS_CI)

    def test_ci_with_accept_gaps(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                     random_state=0, bootstrap=True, accept_gaps=True,
                     bootstrap_kws=dict(rounds=3, ci=0.9))
        df_feat = cpp.run(labels=labels, n_filter=15, n_jobs=1)
        assert all(col in df_feat.columns for col in COLS_CI)

    def test_ci_with_return_stats(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=3)
        df_feat, stats = cpp.run(labels=labels, n_filter=15, n_jobs=1, return_stats=True)
        assert all(col in df_feat.columns for col in COLS_CI)
        assert set(stats) >= {"n_candidates", "n_final"}

    # ---- Negative ------------------------------------------------------------------------
    def test_ci_with_n_batches_negative(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=2)
        with pytest.raises(ValueError, match=r"'n_batches' \(2\) should be None"):
            cpp.run(labels=labels, n_filter=15, n_batches=2, n_jobs=1)

    def test_ci_with_n_sample_batches_negative(self):
        cpp, labels = _make_cpp(ci=0.9, rounds=2)
        with pytest.raises(ValueError, match=r"'n_sample_batches' \(2\) should be None"):
            cpp.run(labels=labels, n_filter=15, n_sample_batches=2, n_jobs=1)

    def test_ci_with_invalid_rounds_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(rounds=0, ci=0.95))

    def test_ci_with_invalid_resample_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(resample="ref", ci=0.95))

    def test_ci_with_invalid_frac_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(frac=0.0, ci=0.95))

    def test_ci_on_run_num_with_invalid_level_negative(self):
        df_parts, labels, split_kws, df_scales = _fixture()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, verbose=False,
                   bootstrap=True, bootstrap_kws=dict(rounds=2, ci="high"))


class TestCPPBootstrapCIGoldenValues:
    """Hand-computed percentile intervals of the summarizing step."""

    @staticmethod
    def _df(features=("f1",)):
        return pd.DataFrame({"feature": list(features), "abs_auc": [0.2] * len(features),
                             "mean_dif": [0.1] * len(features)})

    def test_percentile_hand_computed(self):
        # Values [0.1, 0.2, 0.3, 0.4], ci=0.5 -> 25th/75th percentile with linear interpolation:
        # low  = 0.1 + 0.75 * (0.2 - 0.1) = 0.175; high = 0.3 + 0.25 * (0.4 - 0.3) = 0.325.
        stats = {"f1": {"abs_auc": [0.1, 0.2, 0.3, 0.4], "mean_dif": [0.1, 0.2, 0.3, 0.4]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.5)
        assert df[COL_AUC_LOW].tolist() == [0.175]
        assert df[COL_AUC_HIGH].tolist() == [0.325]
        assert df[COL_DIF_LOW].tolist() == [0.175]
        assert df[COL_DIF_HIGH].tolist() == [0.325]

    def test_narrow_level_hand_computed(self):
        # Values [0.0, 1.0, 2.0, 3.0], ci=0.98 -> 1st/99th percentile:
        # low  = 0.0 + 0.03 * 1.0 = 0.03; high = 2.0 + 0.97 * 1.0 = 2.97.
        stats = {"f1": {"abs_auc": [0.0, 1.0, 2.0, 3.0]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.98)
        assert df[COL_AUC_LOW].tolist() == [0.03]
        assert df[COL_AUC_HIGH].tolist() == [2.97]

    def test_rounded_to_three_decimals(self):
        # Two values, ci=0.5 -> 25th/75th percentile of [0.1234567, 0.7654321]:
        # low = 0.1234567 + 0.25 * 0.6419754 = 0.28394... -> 0.284
        # high = 0.1234567 + 0.75 * 0.6419754 = 0.60494... -> 0.605
        stats = {"f1": {"abs_auc": [0.1234567, 0.7654321]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.5)
        assert df[COL_AUC_LOW].tolist() == [0.284]
        assert df[COL_AUC_HIGH].tolist() == [0.605]

    def test_single_observation_is_nan(self):
        stats = {"f1": {"abs_auc": [0.3], "mean_dif": [0.3]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.95)
        assert df[COLS_CI].isna().all().all()

    def test_unobserved_feature_is_nan(self):
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats={}, ci=0.95)
        assert df[COLS_CI].isna().all().all()

    def test_identical_values_collapse_to_a_point(self):
        stats = {"f1": {"abs_auc": [0.25, 0.25, 0.25], "mean_dif": [-0.4, -0.4]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.95)
        assert df[COL_AUC_LOW].tolist() == [0.25] and df[COL_AUC_HIGH].tolist() == [0.25]
        assert df[COL_DIF_LOW].tolist() == [-0.4] and df[COL_DIF_HIGH].tolist() == [-0.4]

    def test_column_order_is_auc_then_mean_dif(self):
        stats = {"f1": {"abs_auc": [0.1, 0.2], "mean_dif": [0.1, 0.2]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(), per_feat_stats=stats, ci=0.9)
        assert list(df.columns)[-4:] == COLS_CI

    def test_per_feature_intervals_are_independent(self):
        stats = {"f1": {"abs_auc": [0.1, 0.2, 0.3, 0.4]}, "f2": {"abs_auc": [0.0, 0.0]}}
        df = _add_bootstrap_ci_cols(df_feat=self._df(("f1", "f2")), per_feat_stats=stats, ci=0.5)
        assert df[COL_AUC_LOW].tolist() == [0.175, 0.0]
        assert df[COL_AUC_HIGH].tolist() == [0.325, 0.0]

    def test_collect_round_stats_appends_one_value_per_round(self):
        stats = {}
        for auc in (0.11, 0.22):
            df_round = pd.DataFrame({"feature": ["f1"], "abs_auc": [auc], "mean_dif": [auc / 2]})
            _collect_round_stats(df_round=df_round, per_feat_stats=stats)
        assert stats == {"f1": {"abs_auc": [0.11, 0.22], "mean_dif": [0.055, 0.11]}}

    def test_collect_round_stats_skips_absent_statistic(self):
        stats = {}
        df_round = pd.DataFrame({"feature": ["f1"], "abs_auc": [0.4]})
        _collect_round_stats(df_round=df_round, per_feat_stats=stats)
        assert stats == {"f1": {"abs_auc": [0.4]}}
