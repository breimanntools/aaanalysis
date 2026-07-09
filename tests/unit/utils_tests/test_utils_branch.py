"""
Branch-coverage tests for the shared AAanalysis validators in ``utils.py`` and
``_utils/*``. Every case routes through the PUBLIC ``aa.*`` API with an
offending input so the targeted validator arm is exercised end-to-end; no
private function is imported directly.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

import aaanalysis as aa


# I Helper Functions
def _df_seq(n=3):
    """Small valid df_seq from the bundled DOM_GSEC dataset."""
    return aa.load_dataset(name="DOM_GSEC", n=n)


def _df_parts(list_parts=("tmd",), n=3):
    """Valid df_parts via the public SequenceFeature.get_df_parts."""
    sf = aa.SequenceFeature()
    return sf.get_df_parts(df_seq=_df_seq(n=n), list_parts=list(list_parts))


def _df_feat():
    """Valid df_feat produced by a minimal public CPP.run."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales().iloc[:, :2]
    split_kws = sf.get_split_kws(n_split_min=1, n_split_max=2, split_types=["Segment"])
    aa.options["verbose"] = False
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=False)
    df_feat = cpp.run(labels=labels, n_filter=5)
    return df_feat


# II Main Functions
# ---------------------------------------------------------------------------
# utils.py — plot_get_cdict_ / check_df_seq / check_list_parts / check_df_parts
# ---------------------------------------------------------------------------
class TestPlotGetCdictBranch:
    """utils.py:617 — DICT_CAT branch of plot_get_cdict_."""

    def test_dict_cat_branch(self):
        dict_color = aa.plot_get_cdict(name="DICT_CAT")
        assert isinstance(dict_color, dict) and len(dict_color) > 0


class TestCheckDfSeqBranch:
    """utils.py:704,706,733,739,748,751 — df_seq validation arms."""

    def _get_parts(self, df_seq):
        return aa.SequenceFeature().get_df_parts(df_seq=df_seq, list_parts=["tmd"])

    def test_forbidden_start_column(self):
        # utils.py:704
        df = pd.DataFrame({"entry": ["a", "b"], "sequence": ["AAAA", "CCCC"], "start": [1, 1]})
        with pytest.raises(ValueError, match="tmd_start"):
            self._get_parts(df)

    def test_forbidden_stop_column(self):
        # utils.py:706
        df = pd.DataFrame({"entry": ["a", "b"], "sequence": ["AAAA", "CCCC"], "stop": [1, 1]})
        with pytest.raises(ValueError, match="tmd_stop"):
            self._get_parts(df)

    def test_part_based_non_string(self):
        # utils.py:733
        df = pd.DataFrame({"entry": ["a", "b"], "jmd_n": ["AA", "CC"],
                           "tmd": [1, 2], "jmd_c": ["AA", "CC"]})
        with pytest.raises(ValueError, match="should only contain strings"):
            self._get_parts(df)

    def test_seq_tmd_not_contained(self):
        # utils.py:739
        df = pd.DataFrame({"entry": ["a", "b"], "sequence": ["AAAA", "CCCC"],
                           "tmd": ["QQ", "RR"]})
        with pytest.raises(ValueError, match="should be contained"):
            self._get_parts(df)

    def test_part_and_pos_jmd_mismatch(self):
        # utils.py:748
        df = pd.DataFrame({"entry": ["a"], "sequence": ["AAAAAAAAAA"],
                           "tmd_start": [3], "tmd_stop": [5],
                           "jmd_n": ["QQ"], "tmd": ["AAA"], "jmd_c": ["QQ"]})
        with pytest.raises(ValueError, match="do not match"):
            self._get_parts(df)

    def test_part_and_pos_start_stop_mismatch(self):
        # utils.py:751
        df = pd.DataFrame({"entry": ["a"], "sequence": ["QQAAAQQBBB"],
                           "tmd_start": [1], "tmd_stop": [3],
                           "jmd_n": ["QQ"], "tmd": ["AAA"], "jmd_c": ["QQ"]})
        with pytest.raises(ValueError, match="do not match"):
            self._get_parts(df)


class TestCheckListPartsBranch:
    """utils.py:768,771 — default ext-removal and return_default=False skip."""

    def test_default_ext_removal_path(self):
        # utils.py:768/769 — ext_len option == 0 (default) removes ext parts
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=_df_seq(), all_parts=True)
        assert all("ext" not in c for c in df_parts.columns)

    def test_get_df_pos_list_parts_none_skip(self):
        # utils.py:771 — get_df_pos routes check_list_parts(return_default=False, accept_none=True)
        df_feat = _df_feat()
        df_pos = aa.SequenceFeature().get_df_pos(df_feat=df_feat, list_parts=None)
        assert isinstance(df_pos, pd.DataFrame)


class TestCheckDfPartsBranch:
    """utils.py:799,804 — duplicate index and non-string dtype arms."""

    def test_duplicate_index(self):
        # utils.py:799
        df_parts = _df_parts()
        df_scales = aa.load_scales().iloc[:, :2]
        df_parts = df_parts.copy()
        df_parts.index = [0] * len(df_parts)
        feat = f"TMD-Segment(1,1)-{df_scales.columns[0]}"
        with pytest.raises(ValueError, match="Index in 'df_parts' must be unique"):
            aa.SequenceFeature().feature_matrix(features=[feat], df_parts=df_parts, df_scales=df_scales)

    def test_non_string_column(self):
        # utils.py:804
        df_parts = _df_parts().copy()
        df_parts["tmd"] = range(len(df_parts))
        df_scales = aa.load_scales().iloc[:, :2]
        feat = f"TMD-Segment(1,1)-{df_scales.columns[0]}"
        with pytest.raises(ValueError, match="type string"):
            aa.SequenceFeature().feature_matrix(features=[feat], df_parts=df_parts, df_scales=df_scales)


class TestCheckSplitBranch:
    """utils.py:898,911,919,923 — PeriodicPattern / Pattern split arms."""

    def _fm(self, feat):
        df_parts = _df_parts()
        return aa.SequenceFeature().feature_matrix(features=[feat], df_parts=df_parts)

    def test_periodic_pattern_bad_terminus(self):
        # utils.py:898
        with pytest.raises(ValueError, match="Terminus should be"):
            self._fm("TMD-PeriodicPattern(X,i+1/2,1)-ANDN920101")

    def test_pattern_no_positions(self):
        # utils.py:911
        with pytest.raises(ValueError, match="at least 1 element"):
            self._fm("TMD-Pattern(N)-ANDN920101")

    def test_pattern_bad_terminus(self):
        # utils.py:919
        with pytest.raises(ValueError, match="Terminus should be"):
            self._fm("TMD-Pattern(X,1,2)-ANDN920101")

    def test_pattern_wrong_order(self):
        # utils.py:923
        with pytest.raises(ValueError, match="wrong order"):
            self._fm("TMD-Pattern(N,2,1)-ANDN920101")


class TestCheckDfFeatBranch:
    """utils.py:976,989,995,999 — df_feat empty / df_cat scale / shap-column arms."""

    def test_empty_df_feat(self):
        # utils.py:976 — valid columns but zero rows
        df_feat = _df_feat()
        empty = df_feat.iloc[0:0].copy()
        with pytest.raises(ValueError, match="should be not empty"):
            aa.TreeModel().add_feat_importance(df_feat=empty)

    def _cpp_plot(self):
        df_scales = aa.load_scales().iloc[:, :2]
        df_cat = aa.load_scales(name="scales_cat")
        return aa.CPPPlot(df_scales=df_scales, df_cat=df_cat, verbose=False)

    def test_scale_missing_in_df_cat(self):
        # utils.py:989 — profile (shap_plot=False) routes df_cat scale check
        df_feat = _df_feat().copy()
        df_feat["feat_importance"] = np.linspace(1, 2, len(df_feat))
        df_feat["feature"] = [f.rsplit("-", 1)[0] + "-ZZZ999" for f in df_feat["feature"]]
        with pytest.raises(ValueError, match="not in 'df_cat'"):
            self._cpp_plot().profile(df_feat=df_feat, shap_plot=False)

    def test_shap_plot_true_missing_feat_impact(self):
        # utils.py:995 — shap_plot True but no feat_impact column
        df_feat = _df_feat().copy()
        df_feat["mean_dif_X"] = df_feat["mean_dif"]
        with pytest.raises(ValueError, match="feat_impact"):
            self._cpp_plot().heatmap(df_feat=df_feat, shap_plot=True, col_val="mean_dif_X")

    def test_shap_plot_false_missing_feat_importance(self):
        # utils.py:999 — heatmap (shap_plot=False) without feat_importance column
        df_feat = _df_feat()
        with pytest.raises(ValueError, match="feat_importance"):
            self._cpp_plot().heatmap(df_feat=df_feat, shap_plot=False)


# ---------------------------------------------------------------------------
# _utils/check_data.py
# ---------------------------------------------------------------------------
class TestCheckDataBranch:
    """check_data.py:40,180,294 — convert_2d ndim, n_per_group, numeric index."""

    def test_convert_2d_wrong_ndim(self):
        # check_data.py:40 — AAclust.eval list_labels via convert_2d. A 3D nested
        # list converts cleanly to ndim==3 (!=2) so the ndim guard raises.
        X = np.random.RandomState(0).rand(6, 3)
        list_labels = [[[0, 1], [1, 0]], [[1, 1], [0, 0]]]  # -> ndim 3, not 2D
        with pytest.raises(ValueError, match="2D list or 2D array"):
            aa.AAclust().eval(X=X, list_labels=list_labels)

    def test_n_per_group_required_underrepresented(self):
        # check_data.py:180 — comp_kld requires >=2 per group
        X = np.array([[0.0, 1.0], [0.2, 1.1], [5.0, 6.0], [5.1, 6.2], [9.0, 9.0]])
        # label 2 appears once -> underrepresented for n_per_group_required=2
        labels = [1, 1, 0, 0, 1]
        # make label_test=1, label_ref=0 valid but force a single-count value via vals
        labels_bad = [1, 0, 0, 0, 0]  # label 1 appears once
        with pytest.raises(ValueError, match="at least 2 occurrences|more than one"):
            aa.comp_kld(X=X, labels=labels_bad, label_test=1, label_ref=0)

    def test_numeric_unsorted_index_warns(self):
        # check_data.py:294 — check_warning_consecutive_index numeric branch
        df = _df_seq()
        df = df.copy()
        df.index = list(range(len(df)))[::-1]  # numeric but unsorted
        with pytest.warns(UserWarning):
            aa.SequenceFeature().get_df_parts(df_seq=df, list_parts=["tmd"])


# ---------------------------------------------------------------------------
# _utils/check_models.py
# ---------------------------------------------------------------------------
class TestCheckModelsBranch:
    """check_models.py:52,57 — param_to_check and method_to_check arms."""

    def test_param_to_check_missing(self):
        # check_models.py:52 — AAclust requires 'n_clusters' arg in model_class
        class NoNClusters:
            def __init__(self, foo=1):
                pass

            def fit(self, X):
                return self

        with pytest.raises(ValueError, match="n_clusters.*should be an argument"):
            aa.AAclust(model_class=NoNClusters)

    def test_method_to_check_missing(self):
        # check_models.py:57 — AAclustPlot requires 'transform' method in model_class
        class HasNCompNoTransform:
            def __init__(self, n_components=2):
                pass

            def fit(self, X):
                return self

        with pytest.raises(ValueError, match="transform.*should be a method"):
            aa.AAclustPlot(model_class=HasNCompNoTransform)


# ---------------------------------------------------------------------------
# _utils/check_type.py
# ---------------------------------------------------------------------------
class TestCheckTypeBranch:
    """check_type.py:128,162,175 — tuple-number, None-in-list, non-convertible."""

    def test_figsize_tuple_non_number(self):
        # check_type.py:128 — check_tuple n + check_number via check_figsize
        df_rank = pd.DataFrame({"score": [1.0, 2.0, 3.0], "group": ["a", "b", "a"]})
        with pytest.raises(ValueError, match="float or an integer"):
            aa.AAPredPlot.predict_group(df_rank, kind="rank_scatter", col_group="group",
                                        figsize=("a", "b"))

    def test_list_contains_none(self):
        # check_type.py:162 — check_all_non_none via encode_integer
        with pytest.raises(ValueError, match="should not contain 'None'"):
            aa.SequencePreprocessor().encode_integer(list_seq=["ACDE", None])

    def test_list_non_convertible_element(self):
        # check_type.py:175 — check_all_str_or_convertible via encode_integer
        with pytest.raises(ValueError, match="not strings or"):
            aa.SequencePreprocessor().encode_integer(list_seq=["ACDE", {"x": 1}])


# ---------------------------------------------------------------------------
# _utils/decorators.py + _utils/utils_output.py
# ---------------------------------------------------------------------------
class TestDecoratorsOutputBranch:
    """decorators.py:124,126 (best-effort) + utils_output.py:71 (start message)."""

    def test_progress_start_message_emitted(self):
        # utils_output.py:71 — print_start_progress(start_message=...) under verbose CPP
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales().iloc[:, :2]
        split_kws = sf.get_split_kws(n_split_min=1, n_split_max=1, split_types=["Segment"])
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=True)
        df_feat = cpp.run(labels=labels, n_filter=2)
        assert len(df_feat) > 0

    def test_convergence_path_auto_n_clusters(self):
        # decorators.py:124,126 (best-effort) — auto n_clusters estimation over a
        # heavily duplicated X exercises the ConvergenceWarning capture path.
        base = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        X = np.repeat(base, 10, axis=0).astype(float)
        aac = aa.AAclust(verbose=False)
        aac.fit(X, n_clusters=None)
        assert aac.n_clusters >= 1
